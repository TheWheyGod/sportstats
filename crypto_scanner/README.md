# Scanner de compression crypto (pre-expansion)

`squeeze_scanner.py` scanne tout l'univers USDT d'un exchange (via ccxt) et classe les actifs
en compression de volatilité forte avec absorption, donc candidats à une expansion imminente.
Aucun stop ni TP : c'est un outil de détection, pas un système de trading.

```bash
pip install -r requirements.txt
python squeeze_scanner.py                                      # Binance spot : setup 4h+1d, déclencheur 15m+1h
python squeeze_scanner.py --exchange bybit --market swap       # perps Bybit
python squeeze_scanner.py --timeframes 4h 1d --trigger-tfs 15m 1h --min-volume 500000 --only-signals
python squeeze_scanner.py --trigger-tfs                         # sans déclencheur intraday
python squeeze_scanner.py --validate --bars 1000               # le score prédit-il vraiment une expansion ?
python squeeze_scanner.py --loop 60                            # rescan toutes les heures
python squeeze_scanner.py --synthetic                          # test hors-ligne
python replay.py --csv NEON_USDT_1d.csv                        # rejoue le scanner sur un historique
python replay.py --symbol NEON/USDT --exchange bybit --timeframe 1h --trigger-tf 1h --setup-tf 1d --bars 6000
python replay.py --synthetic 30 --trigger-tf 15m --setup-tf 4h --bars 20000   # contrôle du moteur
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

## Déclencheur intraday (15m / 1h)

Deux étages : le **setup** (compression en 4h/1d) sélectionne une watchlist, le **déclencheur**
(15m/1h, téléchargé pour la watchlist seulement) donne le timing.

- Zone de compression **gelée** : le box de la dernière bougie de setup en compression, valide
  3 bougies de setup (sinon le box glissant s'élargit avec la cassure elle-même).
- **ARMÉ ↑/↓** : prix à ≤ 0.3 ATR du bord de la zone, volume SMA5/SMA50 ≥ 1.3, creux montants
  (sommets descendants), squeeze présent aussi en intraday.
- **DÉCLENCHÉ ↑/↓** : clôture hors de la zone sur une bougie d'ignition (volume ≥ 2 × SMA20,
  range ≥ 1.5 ATR, clôture dans les 40 % extrêmes). Affiché 8 bougies avec son âge et le Δ% depuis.
- Aucun look-ahead : une bougie 4h/1d n'est utilisée qu'après sa clôture.

Le biais (-1 à +1 : momentum du squeeze, position dans le box, pente OBV) est indicatif :
une compression annonce une expansion de volatilité, pas sa direction.
Chaque scan est enregistré dans `output/scan_YYYYMMDD_HHMM.csv`.

## Indicateur TradingView (Pine v6)

`smc_preexpansion.pine` reprend la même logique sur un seul graphique : setup 4H/1D via
`request.security` (dernière bougie clôturée, sans repaint), déclencheur sur le TF du graphique,
zone de compression, signaux GO/ARMÉ, tableau de bord et alertes.

1. TradingView → Éditeur Pine → coller le fichier → « Ajouter au graphique ».
2. Graphique en 15m ou 1h, paramètre « TF du setup » en 4H ou 1D.
3. Alertes : « Any alert() function call » (message détaillé) ou une des alertcondition,
   fréquence « Once per bar close ».

Différence avec le Python : pas de rang cross-sectionnel (Pine ne voit qu'un actif) et le rang
percentile exige 250 bougies du TF de setup. Workflow conseillé : le scanner Python sélectionne
la watchlist, l'indicateur Pine donne le timing et les alertes.
