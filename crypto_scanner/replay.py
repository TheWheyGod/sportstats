#!/usr/bin/env python3
"""
Replay historique du scanner : qu'aurait-il affiché, bougie après bougie, avant les grosses
explosions d'une paire ? Tous les indicateurs sont causaux (fenêtres glissantes vers le passé),
donc la valeur à la date t est exactement celle qu'un scan lancé à la clôture de t aurait donnée.

Sorties :
    1. Pour chaque explosion (plus haut futur >= --move % en --horizon bougies) : ce que le scanner
       montrait dans les --lead bougies précédentes (score max, statuts, squeeze, absorption).
    2. Précision des statuts : parmi les bougies en IMMINENT / COMPRESSION, quelle part a été suivie
       d'une explosion, comparée au taux de base (toutes bougies confondues).
    3. Les dernières bougies avant la date analysée.

Usage :
    python replay.py --csv NEON_USDT_1d.csv                    # CSV timestamp,open,high,low,close,volume
    python replay.py --symbol NEON/USDT --exchange bybit --timeframe 1h --bars 1500
    python replay.py --csv NEON_USDT_1d.csv --move 50 --horizon 5 --lead 5

Mode déclencheur intraday (--trigger-tf) : la série fournie est en 1h/15m, le setup est
reconstruit par agrégation vers --setup-tf (4h/1d) et projeté sans look-ahead.
    python replay.py --symbol NEON/USDT --exchange bybit --timeframe 1h --trigger-tf 1h --setup-tf 1d --bars 6000
    python replay.py --csv NEON_USDT_1h.csv --trigger-tf 1h --setup-tf 1d --move 30 --horizon 24
    python replay.py --synthetic 40 --trigger-tf 15m --setup-tf 4h      # contrôle du moteur hors-ligne
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from squeeze_scanner import (Params, PRE_SIGNALS, SETUP_COLS, TF_MINUTES, align_setup, compute_features,
                             resample_ohlcv, setup_frame, status_series, synthetic_data,
                             trigger_series)

PRE_SIGNALS = set(PRE_SIGNALS)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True) if not pd.api.types.is_numeric_dtype(df[ts]) \
        else pd.to_datetime(df[ts], unit="ms", utc=True)
    return df.set_index(ts).sort_index()[["open", "high", "low", "close", "volume"]].astype(float)


async def load_exchange(symbol: str, exchange: str, timeframe: str, bars: int) -> pd.DataFrame:
    import ccxt.async_support as ccxt_async
    ex = getattr(ccxt_async, exchange)({"enableRateLimit": True})
    try:
        tf_ms = TF_MINUTES[timeframe] * 60_000
        since = ex.milliseconds() - bars * tf_ms
        rows: list = []
        while True:
            batch = await ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            rows += batch
            since = batch[-1][0] + tf_ms
            if len(batch) < 2 or since >= ex.milliseconds():
                break
    finally:
        await ex.close()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").set_index("ts").sort_index().iloc[:-1]
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df


def fmt_ts(ts: pd.Timestamp, intraday: bool) -> str:
    return ts.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--symbol")
    src.add_argument("--synthetic", type=int, metavar="N", help="N actifs simulés (mode déclencheur)")
    ap.add_argument("--trigger-tf", choices=list(TF_MINUTES), help="active le mode déclencheur intraday")
    ap.add_argument("--setup-tf", default="1d", choices=list(TF_MINUTES))
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="1h", choices=list(TF_MINUTES))
    ap.add_argument("--bars", type=int, default=1500)
    ap.add_argument("--move", type=float, default=40.0, help="seuil d'explosion en %% (plus haut futur)")
    ap.add_argument("--horizon", type=int, default=3, help="bougies pour atteindre le seuil")
    ap.add_argument("--lead", type=int, default=5, help="bougies observées avant l'explosion")
    ap.add_argument("--threshold", type=float, default=65.0)
    ap.add_argument("--last", type=int, default=15, help="dernières bougies à afficher")
    ap.add_argument("--until", help="ignorer les bougies après cette date (ex. bougie en cours)")
    args = ap.parse_args()

    if args.trigger_tf:
        return replay_trigger(args)
    if args.synthetic:
        ap.error("--synthetic n'est disponible qu'en mode --trigger-tf")

    df = load_csv(args.csv) if args.csv else asyncio.run(
        load_exchange(args.symbol, args.exchange, args.timeframe, args.bars))
    if args.until:
        df = df[df.index <= pd.Timestamp(args.until, tz="UTC")]
    name = args.csv.stem if args.csv else f"{args.symbol} {args.timeframe}"
    intraday = len(df) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta(days=1)

    p = Params(score_threshold=args.threshold)
    f = compute_features(df, p)
    f["status"] = status_series(f, p)
    f.loc[f["score"].isna(), "status"] = "n/a"   # warm-up : pas encore assez d'historique

    # Explosion à t+1..t+H mesurée depuis la clôture de t (le moment du scan)
    fut_high = f["high"][::-1].rolling(args.horizon, min_periods=1).max()[::-1].shift(-1)
    f["fwd_max_%"] = (fut_high / f["close"] - 1) * 100
    f["explosion"] = f["fwd_max_%"] >= args.move
    valid = f["score"].notna()

    pd.set_option("display.width", 220)
    print("=" * 110)
    print(f" REPLAY SCANNER — {name} — {len(df)} bougies, score valide depuis "
          f"{fmt_ts(f.index[valid.argmax()], intraday)}")
    print(f" Explosion = plus haut >= +{args.move:.0f} % dans les {args.horizon} bougies suivant le scan")
    print("=" * 110)

    # 1) Explosions : premières bougies d'un épisode (on ne compte pas 3 fois le même pump)
    starts = f.index[f["explosion"] & ~f["explosion"].shift(fill_value=False) & valid]
    rows = []
    for t in starts:
        i = f.index.get_loc(t)
        win = f.iloc[max(0, i - args.lead + 1): i + 1]
        seen = [s for s in ("IMMINENT", "ACCUMULATION", "COMPRESSION", "COMPRESSION LONGUE",
                            "CASSURE ↑", "CASSURE ↓") if (win["status"] == s).any()]
        rows.append({"Scan": fmt_ts(t, intraday), "Close": f.at[t, "close"],
                     "Max futur %": f.at[t, "fwd_max_%"],
                     f"Score (t)": f.at[t, "score"], f"Score max {args.lead}b": win["score"].max(),
                     "Sqz lvl": int(f.at[t, "sq_level"]), "Sqz bars": int(f.at[t, "sq_bars"]),
                     "BBW pct": f.at[t, "bbw_pct"], "Absorb": int(f.at[t, "pre_exp_count"]), "Accum": f.at[t, "accum"],
                     "Biais": f.at[t, "bias"], "Statut (t)": f.at[t, "status"],
                     f"Statuts {args.lead}b": ", ".join(seen) or "-",
                     "Détecté": "OUI" if set(seen) & PRE_SIGNALS else "non"})
    ev = pd.DataFrame(rows)
    print(f"\n--- 1. {len(ev)} explosions et ce que le scanner affichait juste avant ---")
    if len(ev):
        print(ev.round(3).to_string(index=False))
        det = (ev["Détecté"] == "OUI").mean() * 100
        print(f"\nRappel : {det:.0f} % des explosions précédées d'un statut pré-explosion "
              f"dans les {args.lead} bougies")

    # 2) Précision : que se passe-t-il après un statut ?
    print("\n--- 2. Précision des statuts (taux d'explosion dans les bougies suivantes) ---")
    base = f.loc[valid, "explosion"].mean() * 100
    prec = f[valid].groupby("status").agg(n=("explosion", "size"),
                                           explosion_pct=("explosion", lambda x: x.mean() * 100),
                                           fwd_max_med=("fwd_max_%", "median"))
    prec["lift"] = prec["explosion_pct"] / base if base > 0 else np.nan
    print(prec.round(2).to_string())
    print(f"Taux de base : {base:.2f} % des bougies suivies d'une explosion")

    # 3) Dernières bougies
    print(f"\n--- 3. {args.last} dernières bougies ---")
    cols = ["open", "high", "low", "close", "score", "sq_level", "sq_bars", "sq3_bars", "bbw_pct",
            "pre_exp_count", "accum", "rel_vol", "bias", "dist_up_atr", "dist_dn_atr", "status", "fwd_max_%"]
    tail = f[cols].tail(args.last).copy()
    tail.index = [fmt_ts(t, intraday) for t in tail.index]
    print(tail.round(4).to_string())


# --------------------------------------------------------------------------------------
# Mode déclencheur intraday
# --------------------------------------------------------------------------------------
def trigger_frame(ltf: pd.DataFrame, ltf_tf: str, setup_tf: str, p: Params) -> pd.DataFrame:
    """Setup reconstruit depuis la série intraday elle-même, puis projeté bougie par bougie."""
    sf = setup_frame(resample_ohlcv(ltf, setup_tf), p)
    al = align_setup(sf, setup_tf, ltf.index, ltf_tf, SETUP_COLS)
    out = ltf.join(al.add_prefix("setup_"))
    out["trigger"] = trigger_series(ltf, al, p)
    return out


def forward_excursions(f: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fut_hi = f["high"][::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    fut_lo = f["low"][::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    f["fwd_max_%"] = (fut_hi / f["close"] - 1) * 100
    f["fwd_min_%"] = (fut_lo / f["close"] - 1) * 100
    return f


def replay_trigger(args) -> None:
    p = Params(score_threshold=args.threshold)
    tf, stf = args.trigger_tf, args.setup_tf
    if TF_MINUTES[tf] >= TF_MINUTES[stf]:
        raise SystemExit("--trigger-tf doit être inférieur à --setup-tf")

    if args.synthetic:
        n_setup = max(args.bars * TF_MINUTES[tf] // TF_MINUTES[stf], 200)
        syn = synthetic_data([tf, stf], n_setup, n_symbols=args.synthetic, cut=False)
        series = {k[0]: v for k, v in syn.items() if k[1] == tf}
    elif args.csv:
        series = {args.csv.stem: load_csv(args.csv)}
    else:
        series = {args.symbol: asyncio.run(load_exchange(args.symbol, args.exchange, tf, args.bars))}

    frames = []
    for name, ltf in series.items():
        if args.until:
            ltf = ltf[ltf.index <= pd.Timestamp(args.until, tz="UTC")]
        f = forward_excursions(trigger_frame(ltf, tf, stf, p), args.horizon)
        f["symbol"] = name
        frames.append(f)
    allf = pd.concat(frames)
    valid = allf["setup_score"].notna()

    pd.set_option("display.width", 230)
    print("=" * 110)
    print(f" REPLAY DÉCLENCHEUR — {', '.join(list(series)[:3])}{' ...' if len(series) > 3 else ''}"
          f" — trigger {tf} / setup {stf} — {len(allf)} bougies")
    print(f" Explosion = plus haut >= +{args.move:.0f} % dans les {args.horizon} bougies {tf} suivantes")
    print("=" * 110)

    # 1) Chaque explosion haussière : signal avant ou pendant le départ ?
    rows = []
    for name, f in allf.groupby("symbol", sort=False):
        expl = f["fwd_max_%"] >= args.move
        starts = f.index[expl & ~expl.shift(fill_value=False) & f["setup_score"].notna()]
        for t in starts:
            i = f.index.get_loc(t)
            win = f.iloc[max(0, i - args.lead): i + args.horizon]
            sig = win[win["trigger"].isin(["ARMÉ ↑", "DÉCLENCHÉ ↑"])]
            row = {"Actif": name, "Départ": t.strftime("%Y-%m-%d %H:%M"),
                   "Max %": f.at[t, "fwd_max_%"], "Setup (t)": f.at[t, "setup_status"]}
            if len(sig):
                s_t = sig.index[0]
                j = f.index.get_loc(s_t)
                peak = f["high"].iloc[j + 1: i + args.horizon + 1].max()
                row.update({"1er signal": sig.iloc[0]["trigger"], "Signal à": s_t.strftime("%Y-%m-%d %H:%M"),
                            "Avance (b)": i - j, "Capté %": (peak / f.at[s_t, "close"] - 1) * 100})
            else:
                row.update({"1er signal": "-", "Signal à": "-", "Avance (b)": np.nan, "Capté %": np.nan})
            rows.append(row)
    ev = pd.DataFrame(rows)
    print(f"\n--- 1. {len(ev)} explosions haussières : premier signal ↑ de {args.lead} bougies avant "
          f"à {args.horizon} bougies après le départ ---")
    if len(ev):
        show = ev if len(ev) <= 40 else ev.sort_values("Max %", ascending=False).head(40)
        print(show.round(2).to_string(index=False))
        hit = ev["1er signal"] != "-"
        print(f"\nRappel : {hit.mean() * 100:.0f} % des explosions signalées "
              f"(ARMÉ : {(ev['1er signal'] == 'ARMÉ ↑').mean() * 100:.0f} %, "
              f"DÉCLENCHÉ : {(ev['1er signal'] == 'DÉCLENCHÉ ↑').mean() * 100:.0f} %) · "
              f"part du mouvement captée (médiane) : {ev.loc[hit, 'Capté %'].median():.1f} %")

    # 2) Précision : que se passe-t-il après chaque nouveau signal ?
    new_sig = allf["trigger"].ne(allf.groupby("symbol")["trigger"].shift()) & (allf["trigger"] != "-")
    sig = allf[new_sig & valid]
    base = allf.loc[valid]
    print(f"\n--- 2. Après chaque nouveau signal (horizon {args.horizon} bougies {tf}) ---")
    tab = sig.groupby("trigger").agg(
        n=("close", "size"),
        explosion_pct=("fwd_max_%", lambda x: (x >= args.move).mean() * 100),
        max_med=("fwd_max_%", "median"), min_med=("fwd_min_%", "median"))
    tab.loc["(toutes bougies)"] = [len(base), (base["fwd_max_%"] >= args.move).mean() * 100,
                                   base["fwd_max_%"].median(), base["fwd_min_%"].median()]
    tab["lift"] = tab["explosion_pct"] / tab.loc["(toutes bougies)", "explosion_pct"]
    print(tab.round(2).to_string())
    print("max_med / min_med = excursion favorable / adverse médiane depuis la clôture du signal")

    # 3) Derniers signaux
    last = allf[new_sig].tail(args.last)
    if len(last):
        print(f"\n--- 3. {len(last)} derniers signaux ---")
        cols = ["symbol", "close", "trigger", "setup_status", "setup_zone_high", "setup_zone_low",
                "fwd_max_%", "fwd_min_%"]
        view = last[cols].copy()
        view.index = view.index.strftime("%Y-%m-%d %H:%M")
        print(view.round(5).to_string())


if __name__ == "__main__":
    main()
