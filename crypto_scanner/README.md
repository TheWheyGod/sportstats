# Scanner de compression crypto (pre-expansion)

`squeeze_scanner.py` scanne tout l'univers USDT d'un exchange (via ccxt) et classe les actifs
en compression de volatilité forte avec absorption, donc candidats à une expansion imminente.
Aucun stop ni TP : c'est un outil de détection, pas un système de trading.

```bash
pip install -r requirements.txt
python squeeze_scanner.py                                      # Binance spot, 1h + 4h, volume 24h >= 2M$
python squeeze_scanner.py --exchange bybit --market swap       # perps Bybit
python squeeze_scanner.py --timeframes 15m 1h 4h --min-volume 500000 --only-signals
python squeeze_scanner.py --validate --bars 1000               # le score prédit-il vraiment une expansion ?
python squeeze_scanner.py --loop 60                            # rescan toutes les heures
python squeeze_scanner.py --synthetic                          # test hors-ligne
python replay.py --csv NEON_USDT_1d.csv                        # rejoue le scanner sur un historique
python replay.py --symbol NEON/USDT --exchange bybit --timeframe 1h --bars 2000
```

## Score (0-100, par timeframe, puis combiné avec pondération vers le TF supérieur)

| Poids | Composante |
|---|---|
| 30 % | Largeur de Bollinger : rang percentile sur 250 bougies (compression relative à l'actif) |
| 15 % | ATR(5) / ATR(50) : rang percentile |
| 15 % | Étroitesse du box 20 bougies (HH-LL / ATR50) : rang percentile |
| 20 % | Squeeze TTM à 3 niveaux (BB dans KC 2.0 / 1.5 / 1.0) × durée |
| 20 % | Absorption : bougies volume > 1.4 × SMA20 et range < 0.85 × ATR14 pendant le squeeze |

Bonus de +5 par timeframe supplémentaire au-dessus du seuil (confluence multi-TF).

## Statuts

- **IMMINENT** : score ≥ seuil, squeeze actif, absorption, prix à ≤ 0.5 ATR d'un bord du box
- **COMPRESSION** : score ≥ seuil, pas encore de pression sur un bord
- **CASSURE ↑/↓** : squeeze relâché dans les 2 dernières bougies avec clôture hors du box

Le biais (-1 à +1 : momentum du squeeze, position dans le box, pente OBV) est indicatif :
une compression annonce une expansion de volatilité, pas sa direction.
Chaque scan est enregistré dans `output/scan_YYYYMMDD_HHMM.csv`.
