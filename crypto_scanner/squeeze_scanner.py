#!/usr/bin/env python3
"""
Scanner de compression crypto : détecte, sur tout l'univers USDT d'un exchange, les actifs en
compression de volatilité élevée avec absorption, donc candidats à une expansion imminente.

Pas de stop, pas de TP : l'outil classe, il ne trade pas.

Score de compression (0-100), calculé par bougie et par timeframe :
    30 %  Largeur de Bollinger au plus bas de son propre historique (rang percentile, 250 bougies)
    15 %  Contraction de l'ATR court vs long (ATR5 / ATR50, rang percentile)
    15 %  Étroitesse du box des 20 dernières bougies (HH-LL en ATR, rang percentile)
    20 %  Squeeze TTM à 3 niveaux (BB dans KC 2.0 / 1.5 / 1.0) pondéré par sa durée
    20 %  Absorption : bougies "volume anormal + petite range" dans le squeeze, volume relatif
Les rangs percentiles rendent le score comparable entre BTC et une low-cap : on mesure la
compression d'un actif par rapport à lui-même, pas en valeur absolue.

Biais directionnel (-1 à +1) : momentum du squeeze (régression linéaire type LazyBear),
position du close dans le box, pente de l'OBV. Une compression prédit une expansion de
volatilité, pas sa direction : le biais est un indice, pas une certitude.

Statut :
    IMMINENT      score élevé, squeeze actif, absorption présente, prix à < 0.5 ATR d'un bord du box
    COMPRESSION   score élevé, pas encore de pression sur un bord
    CASSURE ↑/↓   squeeze relâché dans les 2 dernières bougies avec clôture hors du box

Usage :
    python squeeze_scanner.py                                   # Binance spot, 1h + 4h
    python squeeze_scanner.py --exchange bybit --market swap    # perps Bybit
    python squeeze_scanner.py --timeframes 15m 1h 4h --top 30 --min-volume 500000
    python squeeze_scanner.py --validate                        # mesure du pouvoir prédictif du score
    python squeeze_scanner.py --loop 60                         # rescanne toutes les 60 minutes
    python squeeze_scanner.py --synthetic                       # test hors-ligne
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("scanner")

TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# Poids des timeframes dans le score combiné : le TF supérieur porte plus d'énergie stockée.
TF_WEIGHT = {"5m": 0.5, "15m": 0.7, "1h": 1.0, "4h": 1.5, "1d": 2.0}
STABLES = {"USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "EUR", "EURI", "USDE", "PYUSD",
           "USDD", "AEUR", "UST", "USTC", "XUSD", "GBP", "TRY", "BRL", "USD1", "RLUSD"}
LEVERAGED_SUFFIX = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")


@dataclass
class Params:
    bb_len: int = 20
    bb_mult: float = 2.0
    kc_len: int = 20
    kc_mults: tuple = (2.0, 1.5, 1.0)   # squeeze faible / moyen / fort (TTM Squeeze Pro)
    vol_len: int = 20
    vol_mult: float = 1.4
    atr_len: int = 14
    range_mult: float = 0.85
    box_len: int = 20
    rank_window: int = 250
    absorption_window: int = 10
    score_threshold: float = 65.0
    edge_atr: float = 0.5


# --------------------------------------------------------------------------------------
# Indicateurs
# --------------------------------------------------------------------------------------
def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)


def rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Rang percentile (0-1) de la dernière valeur dans sa fenêtre glissante."""
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def consecutive_true(mask: pd.Series) -> pd.Series:
    """Nombre de bougies consécutives où le masque est vrai (0 sinon)."""
    grp = (~mask).cumsum()
    return mask.astype(int).groupby(grp).cumsum()


def linreg_last(s: pd.Series, n: int) -> pd.Series:
    """Valeur de la régression linéaire au dernier point (équivalent ta.linreg(src, n, 0))."""
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def f(y: np.ndarray) -> float:
        slope = ((x - xm) * (y - y.mean())).sum() / denom
        return y.mean() + slope * (n - 1 - xm)
    return s.rolling(n).apply(f, raw=True)


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]
    tr = true_range(out)
    atr = rma(tr, p.atr_len)
    atr_long = rma(tr, 50)

    mid = c.rolling(p.bb_len).mean()
    sd = c.rolling(p.bb_len).std(ddof=0)
    bb_up, bb_lo = mid + p.bb_mult * sd, mid - p.bb_mult * sd
    kc_mid = c.ewm(span=p.kc_len, adjust=False).mean()
    kc_atr = rma(tr, p.kc_len)

    # Squeeze à 3 niveaux : 1 = BB dans KC(2.0), 2 = dans KC(1.5), 3 = dans KC(1.0)
    level = pd.Series(0, index=out.index)
    for i, m in enumerate(p.kc_mults, start=1):
        inside = (bb_up < kc_mid + m * kc_atr) & (bb_lo > kc_mid - m * kc_atr)
        level = level.where(~inside, i)
    out["sq_level"] = level
    out["sq_bars"] = consecutive_true(level >= 2)  # durée du squeeze "classique" (KC 1.5)

    # Composantes de compression (1 = compression maximale)
    bbw = (bb_up - bb_lo) / mid
    out["bbw"] = bbw
    comp_bbw = 1 - rolling_pct_rank(bbw, p.rank_window)
    comp_atr = 1 - rolling_pct_rank(rma(tr, 5) / atr_long, p.rank_window)
    hh, ll = h.rolling(p.box_len).max(), l.rolling(p.box_len).min()
    comp_box = 1 - rolling_pct_rank((hh - ll) / atr_long, p.rank_window)
    sq_score = (level / 3) * np.minimum(out["sq_bars"] / 20, 1).clip(lower=0.25) * (level >= 1)

    # Absorption : volume anormal sur petite range pendant le squeeze (signal pre-expansion)
    high_vol = v > v.rolling(p.vol_len).mean() * p.vol_mult
    small_rng = (h - l) < atr * p.range_mult
    pre_exp = high_vol & small_rng & (level >= 2)
    out["pre_exp_count"] = pre_exp.rolling(p.absorption_window).sum()
    rel_vol = v.rolling(5).mean() / v.rolling(50).mean()
    out["rel_vol"] = rel_vol
    absorption = (0.6 * np.minimum(out["pre_exp_count"] / 3, 1)
                  + 0.4 * (rel_vol - 1).clip(0, 1))

    out["score"] = 100 * (0.30 * comp_bbw + 0.15 * comp_atr + 0.15 * comp_box
                          + 0.20 * sq_score + 0.20 * absorption)

    # Biais directionnel
    donch_mid = (hh + ll) / 2
    mom = linreg_last(c - (donch_mid + c.rolling(p.box_len).mean()) / 2, p.box_len)
    mom_bias = np.tanh(mom / atr)                             # force du momentum
    mom_turn = np.sign(mom.diff())                            # accélération
    box_pos = ((c - ll) / (hh - ll)).clip(0, 1) * 2 - 1       # -1 bas du box, +1 haut
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_slope = np.tanh(obv.diff(p.box_len) / (v.rolling(p.box_len).mean() * p.box_len))
    out["bias"] = (0.35 * mom_bias + 0.15 * mom_turn + 0.25 * box_pos + 0.25 * obv_slope).clip(-1, 1)

    out["box_high"], out["box_low"], out["atr"] = hh, ll, atr
    out["dist_up_atr"] = (hh - c) / atr
    out["dist_dn_atr"] = (c - ll) / atr
    out["bbw_pct"] = rolling_pct_rank(bbw, p.rank_window) * 100
    # box figé à la bougie précédente pour détecter une clôture hors du box
    out["prev_box_high"], out["prev_box_low"] = hh.shift(), ll.shift()
    return out


def classify(row: pd.Series, recent_release: bool, p: Params) -> str:
    if recent_release and row["close"] > row["prev_box_high"]:
        return "CASSURE ↑"
    if recent_release and row["close"] < row["prev_box_low"]:
        return "CASSURE ↓"
    if row["score"] >= p.score_threshold and row["sq_level"] >= 2:
        near_edge = min(row["dist_up_atr"], row["dist_dn_atr"]) <= p.edge_atr
        if near_edge and row["pre_exp_count"] >= 1:
            return "IMMINENT"
        return "COMPRESSION"
    return "-"


def snapshot(df: pd.DataFrame, p: Params) -> dict | None:
    """Métriques de la dernière bougie clôturée."""
    f = compute_features(df, p)
    if len(f) < p.rank_window or not np.isfinite(f["score"].iloc[-1]):
        return None
    last = f.iloc[-1]
    released = bool(((f["sq_level"].shift() >= 2) & (f["sq_level"] < 2)).iloc[-2:].any())
    return {
        "score": last["score"], "bias": last["bias"], "sq_level": int(last["sq_level"]),
        "sq_bars": int(last["sq_bars"]), "bbw_pct": last["bbw_pct"],
        "absorb": int(last["pre_exp_count"]), "rel_vol": last["rel_vol"],
        "close": last["close"], "box_high": last["box_high"], "box_low": last["box_low"],
        "dist_up_atr": last["dist_up_atr"], "dist_dn_atr": last["dist_dn_atr"],
        "status": classify(last, released, p),
    }


# --------------------------------------------------------------------------------------
# Univers & données
# --------------------------------------------------------------------------------------
def is_tradable(base: str) -> bool:
    return base not in STABLES and not any(base.endswith(s) and len(base) > len(s)
                                           for s in LEVERAGED_SUFFIX)


async def load_universe(ex, market: str, min_volume: float, max_symbols: int) -> list[str]:
    await ex.load_markets()
    cands = [s for s, m in ex.markets.items()
             if m.get("active", True) and m.get("quote") == "USDT"
             and (m.get("spot") if market == "spot" else (m.get("swap") and m.get("linear")))
             and is_tradable(m.get("base", ""))]
    tickers = await ex.fetch_tickers(cands)

    def qvol(t: dict) -> float:
        qv = t.get("quoteVolume")
        if qv is None and t.get("baseVolume") and t.get("last"):
            qv = t["baseVolume"] * t["last"]
        return qv or 0.0

    ranked = sorted(((s, qvol(tickers[s])) for s in cands if s in tickers), key=lambda x: -x[1])
    universe = [s for s, qv in ranked if qv >= min_volume][:max_symbols]
    log.info("Univers : %d paires USDT %s avec volume 24h >= %s", len(universe), market,
             f"{min_volume:,.0f}")
    return universe


async def fetch_all(ex, symbols: list[str], timeframes: list[str], bars: int,
                    concurrency: int) -> dict[tuple[str, str], pd.DataFrame]:
    sem = asyncio.Semaphore(concurrency)
    data: dict[tuple[str, str], pd.DataFrame] = {}

    async def one(sym: str, tf: str) -> None:
        async with sem:
            try:
                since = ex.milliseconds() - (bars + 1) * TF_MINUTES[tf] * 60_000
                rows: list = []
                while len(rows) < bars:
                    batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
                    if not batch:
                        break
                    rows += batch
                    since = batch[-1][0] + TF_MINUTES[tf] * 60_000
                    if len(batch) < 2 or since >= ex.milliseconds():
                        break
            except Exception as exc:
                log.debug("%s %s : %s", sym, tf, exc)
                return
        if len(rows) < 60:
            return
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").set_index("ts").sort_index().iloc[:-1]  # bougie en cours exclue
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
        data[(sym, tf)] = df

    await asyncio.gather(*(one(s, tf) for s in symbols for tf in timeframes))
    return data


async def fetch_live(args) -> dict[tuple[str, str], pd.DataFrame]:
    import ccxt.async_support as ccxt_async

    opts = {"enableRateLimit": True}
    if args.market == "swap":
        opts["options"] = {"defaultType": "swap"}
    ex = getattr(ccxt_async, args.exchange)(opts)
    try:
        symbols = args.symbols or await load_universe(ex, args.market, args.min_volume,
                                                      args.max_symbols)
        return await fetch_all(ex, symbols, args.timeframes, args.bars, args.concurrency)
    finally:
        await ex.close()


def synthetic_data(timeframes: list[str], bars: int, n_symbols: int = 40
                   ) -> dict[tuple[str, str], pd.DataFrame]:
    """Univers simulé à régimes de volatilité, uniquement pour tester le pipeline hors-ligne."""
    data = {}
    for k in range(n_symbols):
        for tf in timeframes:
            rng = np.random.default_rng(k * 97 + TF_MINUTES[tf])
            regime = np.empty(bars)
            i = 0
            while i < bars:
                calm, burst = rng.integers(30, 150), rng.integers(10, 60)
                regime[i:i + calm] = rng.uniform(0.15, 0.5)
                regime[i + calm:i + calm + burst] = rng.uniform(1.2, 2.5)
                i += calm + burst
            sigma = 0.01 * np.sqrt(TF_MINUTES[tf] / 60) * regime
            close = 10 * np.exp(np.cumsum(sigma * rng.standard_normal(bars)))
            open_ = np.r_[close[0], close[:-1]]
            wick = np.abs(rng.standard_normal((2, bars))) * sigma * close * 0.5
            nxt = np.r_[regime[1:], regime[-1]]
            volume = rng.lognormal(10, 0.35, bars) * (1 + 1.2 * ((nxt > 1) & (regime < 1)))
            idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("h"), periods=bars,
                                freq=f"{TF_MINUTES[tf]}min")
            data[(f"SIM{k:02d}/USDT", tf)] = pd.DataFrame(
                {"open": open_, "high": np.maximum(open_, close) + wick[0],
                 "low": np.minimum(open_, close) - wick[1], "close": close,
                 "volume": volume}, index=idx)
    return data


# --------------------------------------------------------------------------------------
# Scan & classement
# --------------------------------------------------------------------------------------
def scan(data: dict[tuple[str, str], pd.DataFrame], timeframes: list[str], p: Params
         ) -> pd.DataFrame:
    per_tf: dict[str, dict[str, dict]] = {}
    for (sym, tf), df in data.items():
        snap = snapshot(df, p)
        if snap:
            per_tf.setdefault(sym, {})[tf] = snap

    rows = []
    for sym, snaps in per_tf.items():
        if len(snaps) < len(timeframes):
            continue  # historique insuffisant sur un TF : pas de score combiné fiable
        w = np.array([TF_WEIGHT[tf] for tf in timeframes])
        scores = np.array([snaps[tf]["score"] for tf in timeframes])
        biases = np.array([snaps[tf]["bias"] for tf in timeframes])
        combined = float((scores * w).sum() / w.sum())
        confluence = int((scores >= p.score_threshold).sum())
        # Le TF de référence pour les niveaux = le plus élevé (box le plus significatif)
        ref = snaps[timeframes[-1]]
        # Statut du TF le plus élevé qui signale quelque chose, suffixé de ce TF
        trig_tf = next((tf for tf in reversed(timeframes) if snaps[tf]["status"] != "-"), None)
        status = f"{snaps[trig_tf]['status']} ({trig_tf})" if trig_tf else "-"
        row = {"Symbol": sym,
               "Score": combined + 5 * (confluence - 1) * (confluence > 1),  # bonus confluence MTF
               "Confl.": f"{confluence}/{len(timeframes)}",
               "Statut": status,
               "Biais": float((biases * w).sum() / w.sum())}
        for tf in timeframes:
            row[f"S {tf}"] = snaps[tf]["score"]
        row.update({
            "Sqz": "·" * ref["sq_level"] or "-", "Sqz bars": ref["sq_bars"],
            "BBW pct": ref["bbw_pct"], "Absorb": ref["absorb"], "RelVol": ref["rel_vol"],
            "Prix": ref["close"], "Box haut": ref["box_high"], "Box bas": ref["box_low"],
            "→haut ATR": ref["dist_up_atr"], "→bas ATR": ref["dist_dn_atr"],
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def bias_label(b: float) -> str:
    return "↑ haussier" if b > 0.25 else "↓ baissier" if b < -0.25 else "neutre"


def print_report(res: pd.DataFrame, top: int, timeframes: list[str], synthetic: bool) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    view = res.head(top).copy()
    view["Biais"] = view["Biais"].map(lambda b: f"{b:+.2f} {bias_label(b)}")
    view["Prix"] = view["Prix"].map(lambda x: f"{x:.6g}")
    for col in ("Box haut", "Box bas"):
        view[col] = view[col].map(lambda x: f"{x:.6g}")
    view.index = view.index + 1
    ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    print("\n" + "=" * 120)
    print(f" SCANNER COMPRESSION / PRE-EXPANSION — {ts} — TF {', '.join(timeframes)}"
          + ("  [DONNÉES SYNTHÉTIQUES]" if synthetic else ""))
    print("=" * 120)
    print(view.round(2).to_string())
    counts = res["Statut"].str.replace(r" \(.*\)", "", regex=True).value_counts()
    print("\nStatuts :", ", ".join(f"{k} = {v}" for k, v in counts.items() if k != "-") or "aucun")
    print("Lecture : Score 0-100 (≥65 = compression forte) · Sqz ··· = BB dans KC(1.0), squeeze max"
          " · Absorb = bougies volume anormal/petite range sur 10 bougies\n"
          "          →haut/→bas ATR = distance aux bords du box en ATR · le Biais est indicatif,"
          " la compression ne prédit pas la direction.")


# --------------------------------------------------------------------------------------
# Validation : le score prédit-il réellement une expansion ?
# --------------------------------------------------------------------------------------
def validate(data: dict[tuple[str, str], pd.DataFrame], p: Params, horizon: int) -> None:
    """Le score prédit-il une expansion ? Pour chaque bougie historique :
        vol_ratio = TR moyen sur t+1..t+H  /  TR moyen sur t-H+1..t
    Mesure symétrique (même fenêtre avant/après) : pas de biais de retard d'un ATR lissé.
    vol_ratio > 1 = la volatilité s'étend ; P(x2) = probabilité qu'elle double (explosion).
    Si le score a du pouvoir prédictif, ces valeurs doivent croître avec le bucket de score."""
    frames = []
    for (sym, tf), df in data.items():
        f = compute_features(df, p)
        tr = true_range(f)
        past = tr.rolling(horizon).mean()
        fut = tr[::-1].rolling(horizon).mean()[::-1].shift(-1)
        f["vol_ratio"] = fut / past
        f["fwd_ret"] = f["close"].shift(-horizon) / f["close"] - 1
        frames.append(f[["score", "bias", "vol_ratio", "fwd_ret"]].assign(tf=tf).dropna())
    if not frames:
        print("Pas assez d'historique pour valider.")
        return
    allf = pd.concat(frames)
    allf["bucket"] = pd.cut(allf["score"], [0, 20, 40, 50, 60, 70, 80, 100], include_lowest=True)
    tab = allf.groupby(["tf", "bucket"], observed=True).agg(
        n=("score", "size"),
        vol_ratio_med=("vol_ratio", "median"),
        p_x1_5=("vol_ratio", lambda x: (x > 1.5).mean() * 100),
        p_x2=("vol_ratio", lambda x: (x > 2.0).mean() * 100))
    base = allf.groupby("tf")["vol_ratio"].apply(lambda x: (x > 2.0).mean() * 100)
    tab["lift_x2"] = tab["p_x2"] / tab.index.get_level_values("tf").map(base).to_numpy()
    print("\n--- VALIDATION : expansion de volatilité future par bucket de score ---")
    print(f"Horizon = {horizon} bougies · vol_ratio = TR moyen futur / TR moyen passé · "
          "p_x2 = % de cas où la volatilité double")
    print(tab.round(2).to_string())

    hi = allf[(allf["score"] >= p.score_threshold) & (allf["bias"].abs() > 0.25)]
    if len(hi):
        hit = (np.sign(hi["bias"]) == np.sign(hi["fwd_ret"])).mean() * 100
        print(f"\nBiais directionnel (score >= {p.score_threshold:.0f}, |biais| > 0.25) : "
              f"{hit:.1f} % de bonne direction sur {len(hi)} observations (50 % = hasard)")


# --------------------------------------------------------------------------------------
def run_once(args, p: Params) -> None:
    t0 = time.time()
    if args.synthetic:
        data = synthetic_data(args.timeframes, args.bars)
    else:
        data = asyncio.run(fetch_live(args))
    if not data:
        log.error("Aucune donnée récupérée (réseau, exchange, symboles ?). Essayer --synthetic.")
        return
    log.info("%d séries chargées en %.1fs", len(data), time.time() - t0)

    if args.validate:
        validate(data, p, args.horizon)

    res = scan(data, args.timeframes, p)
    if res.empty:
        log.error("Aucun actif avec assez d'historique (augmenter --bars).")
        return
    if args.only_signals:
        res = res[res["Statut"] != "-"]
    print_report(res, args.top, args.timeframes, args.synthetic)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M")
    path = args.outdir / f"scan_{stamp}.csv"
    res.round(4).to_csv(path, index=False)
    print(f"\nClassement complet : {path.resolve()}")


def main() -> None:
    p_ = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p_.add_argument("--exchange", default="binance", help="id ccxt : binance, bybit, okx, ...")
    p_.add_argument("--market", choices=["spot", "swap"], default="spot")
    p_.add_argument("--timeframes", nargs="+", default=["1h", "4h"], choices=list(TF_MINUTES),
                    help="du plus petit au plus grand ; le dernier sert de référence pour le box")
    p_.add_argument("--symbols", nargs="*", help="liste manuelle (sinon tout l'univers USDT)")
    p_.add_argument("--min-volume", type=float, default=2_000_000, help="volume 24h min en USDT")
    p_.add_argument("--max-symbols", type=int, default=300)
    p_.add_argument("--bars", type=int, default=400, help="bougies par série (>= 300)")
    p_.add_argument("--concurrency", type=int, default=8)
    p_.add_argument("--top", type=int, default=25)
    p_.add_argument("--threshold", type=float, default=65.0)
    p_.add_argument("--only-signals", action="store_true", help="n'afficher que les statuts actifs")
    p_.add_argument("--validate", action="store_true",
                    help="teste le pouvoir prédictif du score (utiliser --bars 1000)")
    p_.add_argument("--horizon", type=int, default=12, help="horizon de validation en bougies")
    p_.add_argument("--loop", type=float, default=0, help="rescan toutes les N minutes")
    p_.add_argument("--outdir", type=Path, default=Path(__file__).parent / "output")
    p_.add_argument("--synthetic", action="store_true")
    p_.add_argument("-v", "--verbose", action="store_true")
    args = p_.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    args.timeframes = sorted(args.timeframes, key=TF_MINUTES.get)
    p = Params(score_threshold=args.threshold)
    if args.bars < p.rank_window + 50:
        p_.error(f"--bars doit être >= {p.rank_window + 50}")

    while True:
        run_once(args, p)
        if not args.loop:
            break
        log.info("Prochain scan dans %.0f min", args.loop)
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
