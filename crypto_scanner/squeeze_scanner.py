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
    IMMINENT            compression (score ou squeeze long) + absorption + prix à < 0.5 ATR d'un bord
    ACCUMULATION        squeeze depuis >= 20 bougies + volume qui monte, range qui se resserre
    COMPRESSION         score élevé, pas encore de pression sur un bord
    COMPRESSION LONGUE  squeeze depuis >= 20 bougies, sans absorption visible
    CASSURE ↑/↓   squeeze relâché dans les 2 dernières bougies avec clôture hors du box

Déclencheur intraday (15m / 1h, sur la watchlist des actifs en setup uniquement) :
    ARMÉ ↑/↓      prix à <= 0.3 ATR du bord du box du setup, volume SMA5/SMA50 >= 1.3,
                  creux montants (sommets descendants), squeeze aussi présent en intraday
    DÉCLENCHÉ ↑/↓ clôture hors du box du setup sur une bougie d'ignition : volume >= 2 x SMA20,
                  range >= 1.5 ATR, clôture dans les 40 % extrêmes de la bougie
Les niveaux du setup sont ceux de la dernière bougie 4h/1d clôturée : aucun look-ahead.

Usage :
    python squeeze_scanner.py                                   # Binance spot, setup 4h+1d, trigger 15m+1h
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
PRE_SIGNALS = ("IMMINENT", "ACCUMULATION", "COMPRESSION", "COMPRESSION LONGUE")


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
    long_squeeze: int = 20          # bougies consécutives en squeeze (BB dans KC 1.5) = compression longue
    score_threshold: float = 65.0
    edge_atr: float = 0.5
    # Déclencheur intraday
    trig_rvol: float = 2.0          # volume de la bougie de cassure >= 2 x SMA20
    trig_tr: float = 1.5            # range de la bougie de cassure >= 1.5 x ATR14 (bougie d'ignition)
    trig_close_pos: float = 0.6     # clôture dans les 40 % hauts (bas) de la bougie
    arm_atr: float = 0.3            # ARMÉ : prix à <= 0.3 ATR (du setup) du bord du box
    arm_rvol: float = 1.3           # ARMÉ : volume SMA5 / SMA50 >= 1.3
    trig_lookback: int = 8          # un déclenchement reste affiché pendant 8 bougies intraday
    setup_memory: int = 3           # le setup reste valide 3 bougies (4h/1d) après sa dernière compression


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


def squeeze_level(df: pd.DataFrame, p: Params, tr: pd.Series | None = None):
    """Squeeze à 3 niveaux : 1 = BB dans KC(2.0), 2 = dans KC(1.5), 3 = dans KC(1.0)."""
    c = df["close"]
    tr = true_range(df) if tr is None else tr
    mid = c.rolling(p.bb_len).mean()
    sd = c.rolling(p.bb_len).std(ddof=0)
    bb_up, bb_lo = mid + p.bb_mult * sd, mid - p.bb_mult * sd
    kc_mid = c.ewm(span=p.kc_len, adjust=False).mean()
    kc_atr = rma(tr, p.kc_len)
    level = pd.Series(0, index=df.index)
    for i, m in enumerate(p.kc_mults, start=1):
        inside = (bb_up < kc_mid + m * kc_atr) & (bb_lo > kc_mid - m * kc_atr)
        level = level.where(~inside, i)
    return level, bb_up, bb_lo, mid


def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]
    tr = true_range(out)
    atr = rma(tr, p.atr_len)
    atr_long = rma(tr, 50)

    level, bb_up, bb_lo, mid = squeeze_level(out, p, tr)
    out["sq_level"] = level
    out["sq_bars"] = consecutive_true(level >= 2)  # durée du squeeze "classique" (KC 1.5)
    out["sq3_bars"] = consecutive_true(level >= 3)  # durée du squeeze maximal (KC 1.0)

    # Composantes de compression (1 = compression maximale)
    bbw = (bb_up - bb_lo) / mid
    out["bbw"] = bbw
    comp_bbw = 1 - rolling_pct_rank(bbw, p.rank_window)
    comp_atr = 1 - rolling_pct_rank(rma(tr, 5) / atr_long, p.rank_window)
    hh, ll = h.rolling(p.box_len).max(), l.rolling(p.box_len).min()
    comp_box = 1 - rolling_pct_rank((hh - ll) / atr_long, p.rank_window)
    sq_score = (level / 3) * np.minimum(out["sq_bars"] / 20, 1).clip(lower=0.25) * (level >= 1)
    # Un squeeze qui dure accumule de l'énergie : plancher à 0.8 (1.0 si le squeeze est au niveau max)
    long_sq = out["sq_bars"] >= p.long_squeeze
    sq_score = sq_score.where(~long_sq, np.maximum(sq_score, 0.8 + 0.2 * (level == 3)))

    # Absorption : volume anormal sur petite range pendant le squeeze (signal pre-expansion)
    high_vol = v > v.rolling(p.vol_len).mean() * p.vol_mult
    small_rng = (h - l) < atr * p.range_mult
    pre_exp = high_vol & small_rng & (level >= 2)
    out["pre_exp_count"] = pre_exp.rolling(p.absorption_window).sum()
    rel_vol = v.rolling(5).mean() / v.rolling(50).mean()
    out["rel_vol"] = rel_vol
    # Accumulation sur plusieurs bougies : volume qui monte pendant que la range se resserre.
    # Capte l'absorption étalée que le test bougie par bougie rate (gros volume sur une mèche).
    rng_ratio = (h - l).rolling(5).mean() / (h - l).rolling(50).mean()
    out["accum"] = ((rel_vol - 1) / 0.5).clip(0, 1) * ((1 - rng_ratio) / 0.3).clip(0, 1)
    absorption = 0.4 * np.minimum(out["pre_exp_count"] / 3, 1) + 0.6 * out["accum"]

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


def status_series(f: pd.DataFrame, p: Params) -> pd.Series:
    """Statut à chaque bougie (vectorisé, causal : n'utilise que le passé)."""
    released = ((f["sq_level"].shift() >= 2) & (f["sq_level"] < 2)).rolling(2).max().astype(bool)
    strong = (f["score"] >= p.score_threshold) & (f["sq_level"] >= 2)
    long_sq = f["sq_bars"] >= p.long_squeeze
    near_edge = np.minimum(f["dist_up_atr"], f["dist_dn_atr"]) <= p.edge_atr
    absorbing = (f["pre_exp_count"] >= 1) | (f["accum"] >= 0.5)
    return pd.Series(np.select(
        [released & (f["close"] > f["prev_box_high"]),
         released & (f["close"] < f["prev_box_low"]),
         (strong | long_sq) & near_edge & absorbing,
         long_sq & absorbing,
         strong,
         long_sq],
        ["CASSURE ↑", "CASSURE ↓", "IMMINENT", "ACCUMULATION", "COMPRESSION", "COMPRESSION LONGUE"],
        default="-"), index=f.index)


def snapshot(df: pd.DataFrame, p: Params) -> dict | None:
    """Métriques de la dernière bougie clôturée."""
    f = setup_frame(df, p)
    if len(f) < p.rank_window // 2 + 50 or not np.isfinite(f["score"].iloc[-1]):
        return None
    last = f.iloc[-1]
    return {
        "score": last["score"], "bias": last["bias"], "sq_level": int(last["sq_level"]),
        "sq_bars": int(last["sq_bars"]), "bbw": last["bbw"], "accum": last["accum"], "bbw_pct": last["bbw_pct"],
        "absorb": int(last["pre_exp_count"]), "rel_vol": last["rel_vol"],
        "close": last["close"], "box_high": last["box_high"], "box_low": last["box_low"],
        "dist_up_atr": last["dist_up_atr"], "dist_dn_atr": last["dist_dn_atr"],
        "status": f["status"].iloc[-1], "active": bool(f["active"].iloc[-1]),
    }


# --------------------------------------------------------------------------------------
# Déclencheur intraday (1h / 15m) sur les actifs en setup de compression (4h / 1d)
# --------------------------------------------------------------------------------------
def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Agrège un OHLCV vers un TF supérieur (bougies alignées UTC, comme les exchanges)."""
    return df.resample(f"{TF_MINUTES[tf]}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def align_setup(setup: pd.DataFrame, setup_tf: str, ltf_index: pd.DatetimeIndex, ltf_tf: str,
                cols: list[str]) -> pd.DataFrame:
    """Projette sur chaque bougie intraday les valeurs de la dernière bougie de setup CLÔTURÉE
    à la clôture de cette bougie intraday (pas de look-ahead : une bougie 1d n'est utilisable
    qu'après minuit UTC)."""
    avail = setup[cols].copy()
    avail.index = avail.index + pd.Timedelta(minutes=TF_MINUTES[setup_tf])
    left = pd.DataFrame({"t": ltf_index + pd.Timedelta(minutes=TF_MINUTES[ltf_tf])})
    right = avail.rename_axis("t").reset_index()
    left["t"] = left["t"].astype(right["t"].dtype)
    out = pd.merge_asof(left, right, on="t", direction="backward")
    out.index = ltf_index
    return out[cols]


def trigger_series(ltf: pd.DataFrame, setup_al: pd.DataFrame, p: Params) -> pd.Series:
    """État du déclencheur à chaque bougie intraday.
        DÉCLENCHÉ ↑/↓ : setup actif + clôture hors de la zone de compression gelée sur une bougie d'ignition
                        (volume >= 2 x SMA20, range >= 1.5 ATR, clôture dans le haut/bas de la bougie)
        ARMÉ ↑/↓      : setup actif + prix collé au bord du box (<= 0.3 ATR du setup) + volume qui
                        monte + creux montants (sommets descendants) + squeeze aussi en intraday"""
    c, h, l, v = ltf["close"], ltf["high"], ltf["low"], ltf["volume"]
    tr = true_range(ltf)
    atr_l = rma(tr, p.atr_len)
    rvol = v / v.rolling(p.vol_len).mean()
    rvol5 = v.rolling(5).mean() / v.rolling(50).mean()
    pos = (c - l) / (h - l).replace(0, np.nan)
    ltf_sq = squeeze_level(ltf, p, tr)[0]

    active = setup_al["active"].fillna(False).astype(bool) & setup_al["zone_high"].notna()
    hi, lo, s_atr = setup_al["zone_high"], setup_al["zone_low"], setup_al["zone_atr"]
    ignition = (rvol >= p.trig_rvol) & (tr >= p.trig_tr * atr_l)
    higher_lows = l.rolling(5).min() > l.shift(5).rolling(5).min()
    lower_highs = h.rolling(5).max() < h.shift(5).rolling(5).max()
    armed_base = active & (rvol5 >= p.arm_rvol) & (ltf_sq >= 1)
    return pd.Series(np.select(
        [active & ignition & (c > hi) & (pos >= p.trig_close_pos),
         active & ignition & (c < lo) & (pos <= 1 - p.trig_close_pos),
         armed_base & (c <= hi) & (hi - c <= p.arm_atr * s_atr) & higher_lows,
         armed_base & (c >= lo) & (c - lo <= p.arm_atr * s_atr) & lower_highs],
        ["DÉCLENCHÉ ↑", "DÉCLENCHÉ ↓", "ARMÉ ↑", "ARMÉ ↓"], default="-"), index=ltf.index)


def setup_frame(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Features du setup + zone de compression GELÉE : le box de la dernière bougie en compression,
    conservé setup_memory bougies. Sans ce gel, le box glissant s'élargit avec la cassure elle-même
    et le setup n'est déjà plus "en compression" quand le prix sort."""
    f = compute_features(df, p)
    f["status"] = status_series(f, p)
    comp = f["status"].isin(PRE_SIGNALS)
    lim = p.setup_memory - 1
    f["active"] = comp.astype(float).rolling(p.setup_memory, min_periods=1).max().astype(bool)
    f["zone_high"] = f["box_high"].where(comp).ffill(limit=lim) if lim else f["box_high"].where(comp)
    f["zone_low"] = f["box_low"].where(comp).ffill(limit=lim) if lim else f["box_low"].where(comp)
    f["zone_atr"] = f["atr"].where(comp).ffill(limit=lim) if lim else f["atr"].where(comp)
    return f


SETUP_COLS = ["status", "active", "zone_high", "zone_low", "zone_atr", "score"]


def trigger_snapshot(setup_f: pd.DataFrame, setup_tf: str, ltf: pd.DataFrame, ltf_tf: str,
                     p: Params) -> dict:
    """Dernier état du déclencheur : un DÉCLENCHÉ récent (<= trig_lookback bougies) prime sur ARMÉ."""
    al = align_setup(setup_f, setup_tf, ltf.index, ltf_tf, SETUP_COLS)
    trig = trigger_series(ltf, al, p)
    recent = trig.iloc[-p.trig_lookback:]
    fired = recent[recent.str.startswith("DÉCLENCHÉ")]
    if len(fired):
        t = fired.index[-1]
        return {"state": fired.iloc[-1], "age": len(recent) - 1 - recent.index.get_loc(t),
                "move": (ltf["close"].iloc[-1] / ltf.at[t, "close"] - 1) * 100,
                "level": al.at[t, "zone_high"] if fired.iloc[-1].endswith("↑") else al.at[t, "zone_low"]}
    if trig.iloc[-1] != "-":
        side = "zone_high" if trig.iloc[-1].endswith("↑") else "zone_low"
        return {"state": trig.iloc[-1], "age": 0, "move": 0.0, "level": al[side].iloc[-1]}
    return {"state": "-", "age": np.nan, "move": np.nan, "level": np.nan}


def add_triggers(res: pd.DataFrame, setup_data: dict, trig_data: dict, setup_tfs: list[str],
                 trigger_tfs: list[str], p: Params) -> pd.DataFrame:
    """Ajoute l'état du déclencheur intraday. Setup de référence = le TF de setup le plus élevé
    dont le statut est actif ; déclencheur retenu = DÉCLENCHÉ sur le TF le plus élevé, sinon ARMÉ."""
    rank = {"DÉCLENCHÉ ↑": 0, "DÉCLENCHÉ ↓": 0, "ARMÉ ↑": 1, "ARMÉ ↓": 1, "-": 9}
    out = {k: [] for k in ("Trigger", "Trig âge", "Trig Δ%", "Niveau")}
    for sym in res["Symbol"]:
        best = {"state": "-", "age": np.nan, "move": np.nan, "level": np.nan, "tf": None}
        frames = {tf: setup_frame(setup_data[(sym, tf)], p) for tf in setup_tfs
                  if (sym, tf) in setup_data}
        s_tf = next((tf for tf in reversed(setup_tfs)
                     if tf in frames and frames[tf]["active"].iloc[-1]), None)
        if s_tf:
            for tf in reversed(trigger_tfs):
                if (sym, tf) not in trig_data:
                    continue
                snap = trigger_snapshot(frames[s_tf], s_tf, trig_data[(sym, tf)], tf, p)
                if rank[snap["state"]] < rank[best["state"]]:
                    best = {**snap, "tf": tf}
        out["Trigger"].append(f"{best['state']} ({best['tf']})" if best["tf"] else "-")
        out["Trig âge"].append(best["age"])
        out["Trig Δ%"].append(best["move"])
        out["Niveau"].append(best["level"])
    res = res.copy()
    for k, v in out.items():
        res.insert(res.columns.get_loc("Biais") + 1 if k == "Trigger" else len(res.columns), k, v)
    return res


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


def watchlist(res: pd.DataFrame, p: Params, max_n: int) -> list[str]:
    """Actifs à surveiller en intraday : statut de setup actif ou score proche du seuil."""
    active = (res["Setup"] != "-") | (res["Score"] >= p.score_threshold - 10)
    return res.loc[active, "Symbol"].head(max_n).tolist()


async def fetch_live(args, p: Params):
    """Étape 1 : setup (4h/1d) sur tout l'univers. Étape 2 : intraday (1h/15m) sur la watchlist
    seulement, pour limiter le nombre de requêtes."""
    import ccxt.async_support as ccxt_async

    opts = {"enableRateLimit": True}
    if args.market == "swap":
        opts["options"] = {"defaultType": "swap"}
    ex = getattr(ccxt_async, args.exchange)(opts)
    try:
        symbols = args.symbols or await load_universe(ex, args.market, args.min_volume,
                                                      args.max_symbols)
        setup = await fetch_all(ex, symbols, args.timeframes, args.bars, args.concurrency)
        trig: dict = {}
        if args.trigger_tfs and setup:
            res = scan(setup, args.timeframes, p)
            watch = watchlist(res, p, args.watch_max) if not res.empty else []
            log.info("Watchlist intraday : %d actifs", len(watch))
            trig = await fetch_all(ex, watch, args.trigger_tfs, args.trigger_bars, args.concurrency)
        return setup, trig
    finally:
        await ex.close()


def synthetic_data(timeframes: list[str], bars: int, n_symbols: int = 40, cut: bool = True
                   ) -> dict[tuple[str, str], pd.DataFrame]:
    """Univers simulé, uniquement pour tester le pipeline hors-ligne. Une série 15m par actif
    (compressions longues avec volume d'accumulation, puis expansions directionnelles) est
    agrégée vers chaque TF : setup et déclencheur voient donc le même marché."""
    base_tf = "15m"
    n = bars * TF_MINUTES[max(timeframes, key=TF_MINUTES.get)] // TF_MINUTES[base_tf]
    idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("D"), periods=n, freq="15min")
    data = {}
    for k in range(n_symbols):
        rng = np.random.default_rng(k)
        sigma = np.empty(n)
        drift = np.zeros(n)
        accum = np.zeros(n)
        i = 0
        while i < n:
            calm, burst = rng.integers(300, 2500), rng.integers(40, 400)
            sigma[i:i + calm] = rng.uniform(0.001, 0.003)
            accum[max(i, i + calm - 150):i + calm] = 1.0          # accumulation avant l'expansion
            sigma[i + calm:i + calm + burst] = rng.uniform(0.006, 0.015)
            drift[i + calm:i + calm + burst] = rng.choice([-1, 1]) * rng.uniform(0.0005, 0.003)
            i += calm + burst
        close = 10 * np.exp(np.cumsum(drift + sigma * rng.standard_normal(n)))
        open_ = np.r_[close[0], close[:-1]]
        wick = np.abs(rng.standard_normal((2, n))) * sigma * close * 0.5
        volume = rng.lognormal(10, 0.35, n) * (1 + 1.0 * accum + 2.0 * (sigma > 0.005))
        base = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0],
                             "low": np.minimum(open_, close) - wick[1], "close": close,
                             "volume": volume}, index=idx)
        for tf in timeframes:
            df = base if tf == base_tf else resample_ohlcv(base, tf)
            data[(f"SIM{k:02d}/USDT", tf)] = df.iloc[-bars:] if cut else df
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

    per_tf = {s: v for s, v in per_tf.items() if len(v) == len(timeframes)}  # historique complet
    # Rang cross-sectionnel de la largeur de Bollinger (TF de référence) : un actif comprimé depuis
    # des mois paraît "normal" face à son propre historique, pas face au reste du marché.
    xs = pd.Series({s: v[timeframes[-1]]["bbw"] for s, v in per_tf.items()}).rank(pct=True)

    rows = []
    for sym, snaps in per_tf.items():
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
               "Score": 0.85 * combined + 15 * (1 - xs[sym])
                        + 5 * (confluence - 1) * (confluence > 1),  # bonus confluence MTF
               "Confl.": f"{confluence}/{len(timeframes)}",
               "Statut": status,
               "Setup": ",".join(tf for tf in timeframes if snaps[tf]["active"]) or "-",
               "Biais": float((biases * w).sum() / w.sum())}
        for tf in timeframes:
            row[f"S {tf}"] = snaps[tf]["score"]
        row.update({
            "Sqz": "·" * ref["sq_level"] or "-", "Sqz bars": ref["sq_bars"],
            "BBW pct": ref["bbw_pct"], "BBW xs": xs[sym] * 100, "Absorb": ref["absorb"],
            "Accum": ref["accum"], "RelVol": ref["rel_vol"],
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
    if "Trigger" in res.columns:
        trig = res[res["Trigger"] != "-"].copy()
        print("\n--- DÉCLENCHEURS INTRADAY (setup actif + signal 1h/15m) ---")
        if trig.empty:
            print("Aucun déclencheur actif.")
        else:
            trig["_r"] = trig["Trigger"].str.startswith("ARMÉ").astype(int)
            trig = trig.sort_values(["_r", "Trig âge", "Score"], ascending=[True, True, False])
            cols = ["Symbol", "Trigger", "Trig âge", "Trig Δ%", "Niveau", "Statut", "Score",
                    "Biais", "Prix"]
            view_t = trig[cols].reset_index(drop=True)
            view_t.index = view_t.index + 1
            print(view_t.round(2).to_string())
        print("DÉCLENCHÉ = clôture intraday hors du box du setup sur bougie d'ignition "
              "(volume >= 2x, range >= 1.5 ATR) · ARMÉ = prix collé au bord du box, volume qui"
              " monte, structure dans le sens de la sortie · âge en bougies du TF indiqué")
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
    trig_data: dict = {}
    if args.synthetic:
        allsyn = synthetic_data(args.timeframes + args.trigger_tfs, args.bars)
        data = {k: v for k, v in allsyn.items() if k[1] in args.timeframes}
        trig_data = {k: v for k, v in allsyn.items() if k[1] in args.trigger_tfs}
    else:
        data, trig_data = asyncio.run(fetch_live(args, p))
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
    if args.trigger_tfs:
        watch = set(watchlist(res, p, args.watch_max))
        trig_data = {k: v for k, v in trig_data.items() if k[0] in watch}
        res = add_triggers(res, data, trig_data, args.timeframes, args.trigger_tfs, p)
    if args.only_signals:
        res = res[(res["Statut"] != "-") | (res.get("Trigger", "-") != "-")]
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
    p_.add_argument("--timeframes", nargs="+", default=["4h", "1d"], choices=list(TF_MINUTES),
                    help="TF de setup (compression) ; le plus élevé sert de référence pour le box")
    p_.add_argument("--trigger-tfs", nargs="*", default=["15m", "1h"], choices=list(TF_MINUTES),
                    help="TF du déclencheur intraday (vide = désactivé)")
    p_.add_argument("--trigger-bars", type=int, default=300)
    p_.add_argument("--watch-max", type=int, default=60,
                    help="nb max d'actifs surveillés en intraday")
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
    args.trigger_tfs = sorted(args.trigger_tfs, key=TF_MINUTES.get)
    if args.trigger_tfs and TF_MINUTES[args.trigger_tfs[-1]] >= TF_MINUTES[args.timeframes[0]]:
        p_.error("les TF de déclenchement doivent être inférieurs aux TF de setup")
    p = Params(score_threshold=args.threshold)
    if args.bars < p.rank_window // 2 + 50:
        p_.error(f"--bars doit être >= {p.rank_window // 2 + 50}")

    while True:
        run_once(args, p)
        if not args.loop:
            break
        log.info("Prochain scan dans %.0f min", args.loop)
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
