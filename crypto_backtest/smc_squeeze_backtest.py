#!/usr/bin/env python3
"""
SMC Pre-Expansion backtest : Volatility Squeeze + absorption sur crypto.

Logique du signal (mesurée à la clôture de la bougie t) :
    isSqueeze      = bb_upper < kc_upper  AND  bb_lower > kc_lower
    highVol        = volume > SMA(volume, 20) * 1.4
    smallRange     = (high - low) < ATR(14) * 0.85
    isPreExpansion = isSqueeze AND highVol AND smallRange

Entrée long (ouverture de t+1) :
    - un isPreExpansion dans les 5 bougies précédant t (t-5 .. t-1)
    - close[t] > bb_upper[t]  ET  NOT isSqueeze[t]  (sortie de squeeze par le haut)
    SL = kc_lower de la bougie isPreExpansion la plus récente.

Sorties (deux variantes testées sur les mêmes signaux) :
    fixed_rr : TP = entry + 2.5 R, SL fixe.
    trailing : Chandelier stop = max(SL initial, plus-haut depuis l'entrée - 2.5 * ATR),
               cliquet uniquement vers le haut, recalculé à la clôture, actif à la bougie suivante.

Choix d'ingénierie (volontairement plus stricts qu'un backtest vectorisé naïf) :
    - Moteur événementiel numpy : SL/TP intrabar, gaps, et ordre SL-avant-TP si les deux
      sont touchés dans la même bougie (hypothèse conservatrice, pas d'info intrabar).
    - Frais taker + slippage appliqués à chaque jambe (paramétrables, plus élevés pour les low-caps).
    - Sizing au risque (1.5 % de l'equity courante) plafonné par un levier max : sans ce cap, un SL
      serré en 15m donne un notionnel irréaliste.
    - Zéro look-ahead : indicateurs à la clôture t, exécution à l'open t+1.
    - Indicateurs alignés sur TradingView : ATR = RMA de Wilder, écart-type population (ddof=0),
      base Keltner = EMA(20) (option --kc-basis sma pour la variante LazyBear).

Usage :
    python smc_squeeze_backtest.py                              # données réelles via ccxt
    python smc_squeeze_backtest.py --symbols BTC/USDT --timeframes 1h 4h
    python smc_squeeze_backtest.py --synthetic                  # smoke test hors-ligne
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("smc")

TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240}
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "VET/USDT", "NEAR/USDT", "NEON/USDT"]
MAJORS = {"BTC/USDT", "ETH/USDT"}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    bb_len: int = 20
    bb_mult: float = 2.0
    kc_len: int = 20
    kc_mult: float = 1.5
    kc_basis: str = "ema"          # "ema" (TradingView ta.kc) ou "sma" (LazyBear)
    vol_len: int = 20
    vol_mult: float = 1.4
    atr_len: int = 14
    range_mult: float = 0.85
    lookback: int = 5              # fenêtre de validité du signal pre-expansion
    rr: float = 2.5                # variante TP fixe
    trail_atr_mult: float = 2.5    # variante trailing
    trail_atr_len: int = 14


@dataclass
class ExecConfig:
    capital: float = 10_000.0
    risk_pct: float = 0.015
    max_leverage: float = 3.0      # cap notionnel / equity (perp). Mettre 1.0 pour du spot pur.
    fee: float = 0.00055           # taker Bybit/Binance perp ~5.5 bps
    slippage_major: float = 0.0002
    slippage_alt: float = 0.0010   # low-caps : carnet plus fin, impact réel bien supérieur

    def slippage(self, symbol: str) -> float:
        return self.slippage_major if symbol in MAJORS else self.slippage_alt


# --------------------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, timeframe: str, days: int, exchanges: list[str],
                cache_dir: Path) -> pd.DataFrame | None:
    """Télécharge l'OHLCV paginé via ccxt, avec cache CSV et fallback d'exchange."""
    import ccxt

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = symbol.replace("/", "")
    for ex_id in exchanges:
        path = cache_dir / f"{ex_id}_{tag}_{timeframe}_{days}d.csv"
        if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ex.load_markets()
            if symbol not in ex.markets:
                log.info("%s absent de %s", symbol, ex_id)
                continue
            tf_ms = TF_MINUTES[timeframe] * 60_000
            since = ex.milliseconds() - days * 86_400_000
            rows: list[list] = []
            while True:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                rows += batch
                since = batch[-1][0] + tf_ms
                if since >= ex.milliseconds() or len(batch) < 2:
                    break
        except Exception as exc:  # réseau, symbole délisté, rate-limit...
            log.warning("%s %s via %s : %s", symbol, timeframe, ex_id, exc)
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").set_index("ts").sort_index()
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
        df = df.iloc[:-1]  # dernière bougie non clôturée
        df.to_csv(path)
        log.info("%s %s : %d bougies depuis %s", symbol, timeframe, len(df), ex_id)
        return df
    return None


def synthetic_ohlcv(symbol: str, timeframe: str, days: int, seed: int) -> pd.DataFrame:
    """Série GBM à régimes de volatilité (compressions / expansions) pour tester le moteur hors-ligne.
    Aucune valeur de performance : sert uniquement à valider la mécanique."""
    rng = np.random.default_rng(seed)
    n = days * 1440 // TF_MINUTES[timeframe]
    scale = np.sqrt(TF_MINUTES[timeframe] / 60)
    regime = np.zeros(n)
    i = 0
    while i < n:  # alterne phases calmes et phases d'expansion
        calm = rng.integers(30, 120)
        regime[i:i + calm] = 0.3
        i += calm
        burst = rng.integers(10, 60)
        regime[i:i + burst] = 1.6
        i += burst
    sigma = 0.008 * scale * regime[:n]
    drift = 0.0002 * scale * np.sign(rng.standard_normal(n)).cumsum().clip(-1, 1)
    ret = drift + sigma * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.standard_normal((2, n))) * sigma * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    # volume élevé en fin de compression (accumulation) et pendant les expansions
    pre_burst = np.r_[regime[1:] > regime[:-1], False]
    volume = rng.lognormal(10, 0.4, n) * (1 + 1.5 * pre_burst + 0.8 * (regime > 1))
    idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("h"), periods=n,
                        freq=f"{TF_MINUTES[timeframe]}min")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


# --------------------------------------------------------------------------------------
# Indicateurs & signaux
# --------------------------------------------------------------------------------------
def rma(s: pd.Series, n: int) -> pd.Series:
    """Moyenne de Wilder (équivalent ta.rma de Pine)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def compute_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    mid = c.rolling(cfg.bb_len).mean()
    sd = c.rolling(cfg.bb_len).std(ddof=0)
    out["bb_upper"] = mid + cfg.bb_mult * sd
    out["bb_lower"] = mid - cfg.bb_mult * sd

    kc_mid = c.ewm(span=cfg.kc_len, adjust=False).mean() if cfg.kc_basis == "ema" \
        else c.rolling(cfg.kc_len).mean()
    kc_rng = atr(out, cfg.kc_len)
    out["kc_upper"] = kc_mid + cfg.kc_mult * kc_rng
    out["kc_lower"] = kc_mid - cfg.kc_mult * kc_rng

    out["atr"] = atr(out, cfg.atr_len)
    out["atr_trail"] = atr(out, cfg.trail_atr_len)

    out["is_squeeze"] = (out["bb_upper"] < out["kc_upper"]) & (out["bb_lower"] > out["kc_lower"])
    out["high_vol"] = out["volume"] > out["volume"].rolling(cfg.vol_len).mean() * cfg.vol_mult
    out["small_range"] = (out["high"] - out["low"]) < out["atr"] * cfg.range_mult
    out["is_pre_expansion"] = out["is_squeeze"] & out["high_vol"] & out["small_range"]

    # Distance (en bougies) au dernier signal pre-expansion et kc_lower figé à ce moment.
    pos = pd.Series(np.arange(len(out)), index=out.index, dtype=float)
    last_pre = pos.where(out["is_pre_expansion"]).ffill()
    bars_since = pos - last_pre
    out["sig_kc_lower"] = out["kc_lower"].where(out["is_pre_expansion"]).ffill()

    recent_pre = bars_since.between(1, cfg.lookback)
    breakout = (c > out["bb_upper"]) & ~out["is_squeeze"]
    out["entry_signal"] = (recent_pre & breakout & (c > out["sig_kc_lower"])).fillna(False)
    return out


# --------------------------------------------------------------------------------------
# Moteur d'exécution événementiel
# --------------------------------------------------------------------------------------
@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    meta: dict = field(default_factory=dict)


def run_backtest(sig: pd.DataFrame, mode: str, scfg: StrategyConfig, ecfg: ExecConfig,
                 slippage: float) -> BacktestResult:
    """mode : 'fixed_rr' ou 'trailing'. Une seule position à la fois, long only."""
    o, h, l, c = (sig[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_t = sig["atr_trail"].to_numpy()
    entry_sig = sig["entry_signal"].to_numpy()
    sl_lvl = sig["sig_kc_lower"].to_numpy()
    n = len(sig)

    cash = ecfg.capital
    qty = 0.0
    equity = np.empty(n)
    trades = []
    in_pos = False
    entry_px = stop = tp = init_sl = hh = 0.0
    entry_i = 0

    def close_position(i: int, raw_px: float, reason: str, apply_slip: bool) -> None:
        nonlocal cash, qty, in_pos
        px = raw_px * (1 - slippage) if apply_slip else raw_px
        fee = qty * px * ecfg.fee
        cash += qty * px - fee
        risk_cash = qty * (entry_px - init_sl)
        pnl = qty * (px - entry_px) - fee - qty * entry_px * ecfg.fee
        trades.append({
            "entry_time": sig.index[entry_i], "exit_time": sig.index[i],
            "entry": entry_px, "exit": px, "sl": init_sl, "tp": tp if mode == "fixed_rr" else np.nan,
            "qty": qty, "notional": qty * entry_px, "pnl": pnl,
            "r_multiple": pnl / risk_cash if risk_cash > 0 else np.nan,
            "bars": i - entry_i, "reason": reason,
        })
        qty = 0.0
        in_pos = False

    for i in range(1, n):
        # 1) Entrée à l'open si signal validé à la clôture précédente
        if not in_pos and entry_sig[i - 1]:
            px = o[i] * (1 + slippage)
            sl = sl_lvl[i - 1]
            eq_now = cash  # flat => equity = cash
            if np.isfinite(sl) and px > sl and eq_now > 0:
                risk_unit = px - sl
                q = eq_now * ecfg.risk_pct / risk_unit
                q = min(q, eq_now * ecfg.max_leverage / px)
                cash -= q * px + q * px * ecfg.fee
                qty, entry_px, init_sl, stop, hh = q, px, sl, sl, h[i]
                tp = px + scfg.rr * risk_unit
                entry_i, in_pos = i, True

        # 2) Gestion de la position sur la bougie i (SL prioritaire = hypothèse conservatrice)
        if in_pos:
            if i > entry_i and o[i] <= stop:
                close_position(i, o[i], "gap_stop", True)
            elif l[i] <= stop:
                close_position(i, stop, "stop" if stop <= init_sl else "trail_stop", True)
            elif mode == "fixed_rr" and h[i] >= tp:
                # ordre limite : fill au TP, ou à l'open si gap au-dessus
                close_position(i, max(tp, o[i]) if i > entry_i else tp, "take_profit", False)
            elif mode == "trailing":
                hh = max(hh, h[i])
                if np.isfinite(atr_t[i]):
                    stop = max(stop, hh - scfg.trail_atr_mult * atr_t[i])

        equity[i] = cash + qty * c[i]
    equity[0] = ecfg.capital

    if in_pos:  # clôture mark-to-market en fin d'historique
        close_position(n - 1, c[-1], "end_of_data", True)
        equity[-1] = cash

    tr = pd.DataFrame(trades)
    return BacktestResult(pd.Series(equity, index=sig.index), tr)


# --------------------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------------------
def compute_metrics(res: BacktestResult, sig: pd.DataFrame, timeframe: str,
                    ecfg: ExecConfig) -> dict:
    eq = res.equity
    tr = res.trades
    ppy = 365 * 1440 / TF_MINUTES[timeframe]  # crypto : marché 24/7
    rets = eq.pct_change().dropna()

    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else np.nan
    downside = np.sqrt((np.minimum(rets, 0) ** 2).mean())
    sortino = rets.mean() / downside * np.sqrt(ppy) if downside > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1
    # durée max sous l'eau : du plus haut précédent jusqu'au retour au plus haut (ou fin)
    underwater = dd < 0
    grp = (~underwater).cumsum()
    dd_dur = pd.Timedelta(0)
    for _, g in eq[underwater].groupby(grp[underwater]):
        dd_dur = max(dd_dur, g.index[-1] - g.index[0] + pd.Timedelta(minutes=TF_MINUTES[timeframe]))

    bh = sig["close"].iloc[-1] / sig["open"].iloc[0] - 1
    m = {
        "Return %": (eq.iloc[-1] / ecfg.capital - 1) * 100,
        "B&H %": bh * 100,
        "Trades": len(tr),
        "Win %": np.nan, "PF": np.nan, "Avg Win %": np.nan, "Avg Loss %": np.nan,
        "Expect. R": np.nan,
        "MaxDD %": dd.min() * 100,
        "DD Days": dd_dur.total_seconds() / 86_400,
        "Sharpe": sharpe, "Sortino": sortino,
        "Exposure %": np.nan,
    }
    if len(tr):
        pnl_pct = (tr["exit"] / tr["entry"] - 1) * 100
        wins, losses = tr["pnl"] > 0, tr["pnl"] <= 0
        gross_loss = -tr.loc[losses, "pnl"].sum()
        m.update({
            "Win %": wins.mean() * 100,
            "PF": tr.loc[wins, "pnl"].sum() / gross_loss if gross_loss > 0 else np.inf,
            "Avg Win %": pnl_pct[wins].mean() if wins.any() else np.nan,
            "Avg Loss %": pnl_pct[losses].mean() if losses.any() else np.nan,
            "Expect. R": tr["r_multiple"].mean(),
            "Exposure %": tr["bars"].sum() / len(eq) * 100,
        })
    return m


# --------------------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------------------
def plot_run(sig: pd.DataFrame, results: dict[str, BacktestResult], title: str,
             path: Path, capital: float) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.04, subplot_titles=(title, "Equity"))
    fig.add_trace(go.Candlestick(x=sig.index, open=sig["open"], high=sig["high"],
                                 low=sig["low"], close=sig["close"], name="Prix",
                                 increasing_line_color="#26a69a",
                                 decreasing_line_color="#ef5350"), row=1, col=1)
    for col, color, dash in [("bb_upper", "#5c6bc0", None), ("bb_lower", "#5c6bc0", None),
                             ("kc_upper", "#ffa726", "dot"), ("kc_lower", "#ffa726", "dot")]:
        fig.add_trace(go.Scatter(x=sig.index, y=sig[col], name=col, opacity=0.6,
                                 line=dict(width=1, color=color, dash=dash)), row=1, col=1)

    pre = sig[sig["is_pre_expansion"]]
    fig.add_trace(go.Scatter(x=pre.index, y=pre["low"] * 0.995, mode="markers", name="PreExpansion",
                             marker=dict(symbol="circle", size=5, color="#ab47bc")), row=1, col=1)

    colors = {"fixed_rr": "#1e88e5", "trailing": "#43a047"}
    for mode, res in results.items():
        tr = res.trades
        if len(tr):
            fig.add_trace(go.Scatter(x=tr["entry_time"], y=tr["entry"], mode="markers",
                                     name=f"Entrée {mode}",
                                     marker=dict(symbol="triangle-up", size=10, color=colors[mode])),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=tr["exit_time"], y=tr["exit"], mode="markers",
                                     name=f"Sortie {mode}", text=tr["reason"],
                                     marker=dict(symbol="x", size=9, color=colors[mode])),
                          row=1, col=1)
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name=f"Equity {mode}",
                                 line=dict(color=colors[mode])), row=2, col=1)
    bh = capital * sig["close"] / sig["open"].iloc[0]
    fig.add_trace(go.Scatter(x=sig.index, y=bh, name="Buy & Hold",
                             line=dict(color="#9e9e9e", dash="dash")), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04))
    fig.write_html(path, include_plotlyjs="cdn")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=list(TF_MINUTES), choices=list(TF_MINUTES))
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--exchanges", nargs="+", default=["binance", "bybit", "okx"])
    p.add_argument("--capital", type=float, default=10_000)
    p.add_argument("--risk", type=float, default=0.015)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--fee", type=float, default=0.00055)
    p.add_argument("--kc-basis", choices=["ema", "sma"], default="ema")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "output")
    p.add_argument("--synthetic", action="store_true", help="données simulées (test hors-ligne)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scfg = StrategyConfig(kc_basis=args.kc_basis)
    ecfg = ExecConfig(capital=args.capital, risk_pct=args.risk,
                      max_leverage=args.max_leverage, fee=args.fee)
    args.outdir.mkdir(parents=True, exist_ok=True)
    warmup = max(scfg.bb_len, scfg.kc_len, scfg.vol_len, scfg.atr_len) * 3

    rows, all_trades = [], []
    for tf in args.timeframes:
        for k, sym in enumerate(args.symbols):
            if args.synthetic:
                raw = synthetic_ohlcv(sym, tf, args.days, seed=k * 10 + TF_MINUTES[tf])
            else:
                raw = fetch_ohlcv(sym, tf, args.days, args.exchanges,
                                  Path(__file__).parent / "data_cache")
            if raw is None or len(raw) < warmup * 2:
                log.warning("Pas de données exploitables pour %s %s — ignoré", sym, tf)
                continue

            # Indicateurs sur tout l'historique, puis on coupe le warm-up pour ne pas
            # trader sur des bandes pas encore stabilisées.
            sig = compute_signals(raw, scfg).iloc[warmup:]
            results = {}
            for mode in ("fixed_rr", "trailing"):
                res = run_backtest(sig, mode, scfg, ecfg, ecfg.slippage(sym))
                results[mode] = res
                rows.append({"Symbol": sym, "TF": tf, "Exit": mode,
                             **compute_metrics(res, sig, tf, ecfg)})
                if len(res.trades):
                    all_trades.append(res.trades.assign(symbol=sym, timeframe=tf, exit_mode=mode))

            if not args.no_plot:
                fname = args.outdir / f"{sym.replace('/', '')}_{tf}.html"
                plot_run(sig, results, f"{sym} {tf} — SMC Pre-Expansion", fname, ecfg.capital)

    if not rows:
        log.error("Aucun backtest exécuté (vérifier réseau / symboles, ou --synthetic).")
        return

    summary = pd.DataFrame(rows).set_index(["Symbol", "TF", "Exit"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 110)
    print(" SMC PRE-EXPANSION — PERFORMANCE" + ("  [DONNÉES SYNTHÉTIQUES]" if args.synthetic else ""))
    print("=" * 110)
    print(summary.round(2).to_string())

    agg = summary.groupby(level=["TF", "Exit"]).agg(
        {"Return %": "mean", "Trades": "sum", "Win %": "mean", "PF": "median",
         "MaxDD %": "mean", "Sharpe": "mean", "Expect. R": "mean"})
    print("\n--- Agrégat par timeframe / sortie (moyenne cross-actifs, PF médian) ---")
    print(agg.round(2).to_string())

    summary.round(4).to_csv(args.outdir / "summary.csv")
    if all_trades:
        pd.concat(all_trades).to_csv(args.outdir / "trades.csv", index=False)
    print(f"\nRésultats : {args.outdir.resolve()} (summary.csv, trades.csv, graphiques HTML)")


if __name__ == "__main__":
    main()
