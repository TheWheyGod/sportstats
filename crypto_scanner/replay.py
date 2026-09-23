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
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from squeeze_scanner import Params, compute_features, status_series, TF_MINUTES

PRE_SIGNALS = {"IMMINENT", "ACCUMULATION", "COMPRESSION", "COMPRESSION LONGUE"}


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


if __name__ == "__main__":
    main()
