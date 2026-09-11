# sportvalue

Probabilités de matchs calculées **à partir des données**, pas des cotes.
Football, rugby, basket, tennis — marchés équipe **et** marchés joueurs.

```bash
py -m sportvalue predict --sport football --home Liverpool --away Arsenal --buteurs
```

---

## Le principe

Aucune cote n'entre dans le calcul. Les probabilités sortent de modèles ajustés sur les
résultats passés, la forme récente, les statistiques, les absences et la météo. Elles sont
donc comparables à n'importe quelle ligne de bookmaker sans circularité.

**Garantie de cohérence** : pour chaque sport, tous les marchés dérivent d'**un seul objet de
distribution**. Le 1X2, le total de buts, le BTTS et les scores exacts du football viennent de
la même matrice Dixon-Coles ; le total de points et le total d'essais du rugby viennent des
mêmes tirages Monte-Carlo ; les buteurs sont normalisés pour que la somme de leurs espérances
égale celle de l'équipe. Il est structurellement impossible d'obtenir un total de buts qui
contredise le 1X2 — ce qui arrive dès qu'on entraîne un modèle séparé par marché.

---

## Ce que sort l'outil, par sport

### Football
| Marché | Détail |
|---|---|
| **1X2** | + double chance, draw no bet |
| **Nombre de buts** | over/under 0.5 → 5.5, avec push sur lignes entières |
| **BTTS** | les deux équipes marquent |
| **Scores exacts** | top 12 |
| **Totaux par équipe** | 0.5 / 1.5 / 2.5 |
| **Buteurs** | λ individuel, P(marque), P(2 buts et +), P(triple), **P(premier buteur)**, P(aucun buteur) |

### Rugby
| Marché | Détail |
|---|---|
| **Vainqueur** | 1 / nul / 2 |
| **Nombre de points** | over/under 39.5 → 59.5 |
| **Écart par tranche** | 1-7, 8-12, 13-21, 22+ de chaque côté |
| **Nombre d'essais** | total 3.5 → 7.5, et par équipe |
| **Marqueurs d'essais** | λ individuel, P(marque), P(2 essais et +) |
| **Météo** | vent, pluie, froid → essais et réussite au pied |

### Tennis
| Marché | Détail |
|---|---|
| **Vainqueur** | via Elo par surface |
| **Score en sets** | 2-0, 2-1, 3-1… selon le format |
| **Nombre de sets** | over/under |
| **Nombre de jeux** | 5 lignes centrées sur l'espérance |
| **Handicap jeux** | −5.5 → +5.5 |
| **Diagnostic** | % de points gagnés au service, taux de conservation |

### Basket
| Marché | Détail |
|---|---|
| **Vainqueur** | sans nul (prolongations résolues) |
| **Total de points** | 5 lignes centrées |
| **Handicap** | 5 lignes centrées sur l'écart attendu |
| **Totaux par équipe** | 3 lignes |

---

## Démarrage

```bash
pip install numpy pandas scipy scikit-learn requests pyyaml
```

### Football — entièrement automatique

Résultats depuis football-data.co.uk, statistiques joueurs depuis l'API Fantasy Premier League.
Aucune clé, aucun compte.

```bash
py -m sportvalue predict --sport football --home Liverpool --away Arsenal --buteurs
```

```bash
py -m sportvalue predict --sport football --league F1 --home Marseille --away Lyon
```

Codes de championnat : `E0` Premier League, `F1` Ligue 1, `D1` Bundesliga, `SP1` Liga,
`I1` Serie A, et les divisions inférieures (`E1`, `F2`…). Les buteurs automatiques ne sont
disponibles qu'en Premier League ; ailleurs, passer `--joueurs` avec un CSV.

### Rugby, basket, tennis — via un CSV de résultats

```bash
py -m sportvalue template-resultats top14.csv --joueurs joueurs_top14.csv
```

Format minimal, cinq colonnes :

```
date,home,away,home_score,away_score
2026-08-30,Toulouse,Bordeaux,27,19
```

```bash
py -m sportvalue predict --sport rugby --historique top14.csv --joueurs joueurs_top14.csv --home Toulouse --away Perpignan --meteo
```

```bash
py -m sportvalue predict --sport tennis --historique atp.csv --home Alcaraz --away Zverev --surface Clay --bo 5
```

```bash
py -m sportvalue predict --sport basket --historique nba.csv --home Boston --away Sacramento
```

Des jeux d'exemple sont fournis : `exemple_rugby_top14.csv`, `exemple_rugby_joueurs.csv`,
`exemple_basket.csv`, `exemple_tennis.csv`.

### Vue journée — tous les matchs, tous les sports, une page

```bash
py -m sportvalue journee --sport football --leagues E0,F1,SP1,I1,D1 --buteurs --html journee.html --ouvrir
```

Récupère les rencontres à venir (football-data), ajuste **un modèle par championnat**
puis prédit tout. La page se filtre par sport et par équipe ; chaque match affiche son
résumé, et le détail complet s'ouvre au clic.

Pour empiler les autres sports sur la **même** page :

```bash
py -m sportvalue journee --sport rugby --historique top14.csv --matchs matchs_rugby.csv --joueurs joueurs.csv --competition "Top 14" --meteo --html journee.html --ajouter
```

L'état est conservé dans un fichier `.slate` à côté du HTML ; `--ajouter` s'y greffe.

### Coupes d'Europe

```bash
py -m sportvalue europe --competition c1 --html c1.html
```

⚠️ **Deux limites structurelles, pas des bugs.**

football-data.co.uk ne couvre que les championnats domestiques : la C1 n'y est pas. Le
calendrier vient donc de The Odds API (**uniquement les dates et les équipes ; aucune cote
n'entre dans le calcul**), ce qui exige `ODDS_API_KEY`.

Surtout : un Dixon-Coles ajusté sur la Serie A note les clubs italiens **par rapport à la
moyenne italienne**. Rien dans les données domestiques ne dit si la Serie A vaut plus ou
moins que la Bundesliga, puisqu'aucune équipe ne joue dans les deux. C'est un défaut
d'**identifiabilité statistique** : agrandir le modèle n'y change rien.

Trois façons d'ancrer l'échelle, par qualité décroissante :

1. **Des résultats européens dans l'historique** — les deux échelles s'y soudent d'elles-mêmes.
   C'est la seule solution vraiment satisfaisante : passez un CSV de matchs C1/C3 à `--historique`.
2. **Promus et relégués**, entre divisions d'un même pays (E0↔E1 partagent des clubs).
   Fonctionne dans un pays, jamais entre pays.
3. **Un prior explicite** (`FORCE_CHAMPIONNAT` dans `data/uefa.py`), posé à la main d'après
   l'ordre des coefficients UEFA. La commande le rappelle à chaque exécution.

Les clubs absents des championnats couverts (Bodø/Glimt, Slavia Praha, Sabah FK…) sont
**listés et écartés** plutôt que prédits au jugé.

### Consulter les résultats

Trois sorties, selon l'usage :

| Sortie | Commande | Pour quoi |
|---|---|---|
| **Terminal** | (par défaut) | vérifier un chiffre vite |
| **Rapport HTML** | `--html rapport.html --ouvrir` | consulter tous les marchés d'un coup, partager |
| **JSON** | `--json sortie.json` | réinjecter dans un tableur ou un autre script |

```bash
py -m sportvalue predict --sport football --home Liverpool --away Arsenal --buteurs --html rapport.html --ouvrir
```

Le rapport HTML est **autonome** : un seul fichier, aucune dépendance, il s'ouvre par
double-clic et se partage tel quel. Chaque probabilité y est doublée de sa **cote équitable**
(1/p), la seule forme sous laquelle une probabilité se compare d'un coup d'œil à une ligne de
bookmaker. Les scores exacts sont rendus en **heatmap 7×7** plutôt qu'en liste : la grille
montre où se concentre la masse et si la diagonale des nuls est marquée, ce qu'un top 12 cache.

---

## Les modèles

| Sport | Modèle | Pourquoi celui-là |
|---|---|---|
| **Football** | Dixon-Coles (Poisson bivarié corrigé) | Poisson simple sous-estime les 0-0 et 1-1 : les buts ne sont pas indépendants. Le paramètre ρ corrige les quatre scores bas, ce qui rend exploitables les marchés « nul » et « under 2.5 ». Décroissance temporelle exponentielle + ridge pour que les promus ne reçoivent pas de note aberrante. |
| **Rugby** | Points composés simulés | Le rugby marque par paliers de 3, 5 et 7. Les essais, transformations et pénalités sont simulés séparément, d'où une distribution discrète réaliste et un **push mesuré à ~2%** sur les handicaps à ligne entière — qu'un modèle gaussien fixe à zéro. |
| **Tennis** | Elo par surface + markovien hiérarchique | La chaîne point → jeu → set → match est **exactement** calculable. On inverse la probabilité Elo pour trouver le % de points au service qui la reproduit, puis tout en découle. C'est ce qui garantit que le total de jeux et le vainqueur ne se contredisent jamais. |
| **Basket** | Moindres carrés pondérés, forme close | Scores élevés ⇒ quasi-gaussiens. La covariance des résidus donne directement l'écart-type de la marge et la corrélation entre les deux scores, dont dépendent tous les handicaps et totaux. |
| **Joueurs** | Répartition du λ d'équipe | λ_joueur = λ_équipe × (taux × minutes) / Σ(taux × minutes). Taux rétréci vers un prior de poste (`(événements + k·prior)/(minutes + k)`), penaltys sortis du pot commun et attribués au tireur désigné. |

### Détails qui comptent

- **xG plutôt que buts** quand il est disponible. Sur 300 minutes de jeu, les buts bruts ne
  distinguent pas un attaquant en forme d'un attaquant chanceux.
- **Rétrécissement bayésien** des taux joueurs : un joueur avec 1 but en 90 minutes n'a pas un
  taux de 1 but par match. En début de saison, le prior de poste domine — comportement voulu.
- **P(premier buteur)** = (λ_joueur / Λ) × (1 − e^−Λ), sous hypothèse de processus de Poisson
  indépendants. Le complément e^−Λ est exactement « aucun buteur ».
- **Conventions de signe opposées** entre `football_dc` (défense élevée = encaisse beaucoup) et
  `linear_score` (défense élevée = bonne défense). Documenté dans chaque `ratings()`.
- **Météo rugby** : les coefficients sont des **priors explicites**, pas des valeurs estimées.
  `calibrate_weather()` les réestime si vous disposez d'un historique avec météo.

---

## Architecture

```
sportvalue/
  predict.py        assemblage des marchés par sport
  core/
    scoredist.py    distribution de scores -> tous les marchés
    oddsmath.py     dé-vig, contrôle de santé des marchés
    calib.py        calibration, fusion
    metrics.py      Brier, log-loss, RPS, ROI, CLV
  models/
    football_dc.py  Dixon-Coles
    tennis.py       Elo surface + markovien exact
    linear_score.py basket
    rugby.py        points composés + météo
    scorers.py      buteurs / marqueurs d'essais
    elo.py          Elo générique
  data/
    football_data.py  résultats (gratuit, sans clé)
    fpl.py            joueurs Premier League (gratuit, sans clé)
    results_csv.py    historique générique, tous sports
    weather.py        Open-Meteo (gratuit)
  backtest/engine.py  walk-forward sans fuite temporelle
```

---

## Validation

```bash
py tests/test_suite.py
```

Priorité aux invariants dont la rupture est **silencieuse**. Ces tests ont attrapé six bugs
réels pendant le développement :

1. Shin dégénérait en normalisation proportionnelle (racine triviale en z=0)
2. `shrink_to_market` violait sa propre borne après renormalisation
3. Signe de la défense inversé dans le classement basket
4. Résolution des noms de books cassée sur le chemin CSV
5. Une référence d'exchange illiquide produisait un « +271% d'edge » fantôme
6. Bug de signe sur le handicap basket donnant une ligne à probabilité 1.0

Le backtest football (walk-forward, 41 000 matchs, 18 championnats) montre que le modèle est
**bien calibré** — écart prédit/observé inférieur à 2 points sur toutes les tranches — mais
qu'il **n'améliore pas** la ligne d'un bookmaker sharp. C'est le résultat attendu : ces lignes
intègrent des informations qu'aucun modèle maison ne reproduit. Le modèle vaut pour ce qu'il
est : une estimation probabiliste indépendante, cohérente entre marchés, et explicable.

---

## Limites

1. **Le backtest ne couvre que le 1X2 football.** Les modèles rugby/basket/tennis sont validés
   sur leur cohérence interne, pas contre un historique de résultats à grande échelle.
2. **Buteurs automatiques = Premier League uniquement.** Ailleurs, CSV obligatoire.
3. **Les absences** ne sont prises en compte que via le statut FPL (Premier League) ou les
   minutes attendues que vous fournissez.
4. **Les coefficients météo rugby sont des priors**, pas des estimations.
5. Le jeu d'argent comporte un risque de dépendance. Ne miser que ce qu'on peut perdre.
