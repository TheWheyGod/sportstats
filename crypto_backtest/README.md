# SMC Pre-Expansion : backtest Squeeze + Absorption

Script : `smc_squeeze_backtest.py` (moteur événementiel pandas/numpy, données via ccxt).

```bash
pip install -r requirements.txt
python smc_squeeze_backtest.py                                   # 5 paires x 15m/1h/4h, 365 jours
python smc_squeeze_backtest.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h --max-leverage 1
python smc_squeeze_backtest.py --synthetic                       # test hors-ligne (données simulées)
```

Sorties dans `output/` : `summary.csv`, `trades.csv`, un graphique HTML par paire/TF
(bougies + BB/KC + signaux pre-expansion + entrées/sorties + equity vs Buy & Hold).

Hypothèses d'exécution : entrée à l'open t+1, SL prioritaire sur le TP si les deux sont touchés
dans la même bougie, frais taker 5.5 bps, slippage 2 bps (majors) / 10 bps (alts),
risque 1.5 % de l'equity, notionnel plafonné à 3x l'equity.
