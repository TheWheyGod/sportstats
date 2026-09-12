# Déploiement sur GitHub Pages

Objectif : une page publique, lisible depuis n'importe quel téléphone **sans connexion**,
qui se régénère toute seule **même PC éteint**. Ni Claude ni ta machine dans la boucle.

Tout est prêt localement. Il reste cinq étapes, et seules elles demandent ton compte GitHub.

---

## 1. Créer le dépôt

Fait : `TheWheyGod/sportstats`. Il est **privé**, et c'est un problème : **GitHub Pages
n'est pas disponible sur un dépôt privé avec un compte gratuit.** Il faut le passer en public —
**Settings → General → Danger Zone → Change repository visibility → Make public**.
Le dépôt ne contient aucune clé ni donnée personnelle : c'est sans risque.

## 2. Pousser le code

```bash
cd C:\Users\trand\sportvalue
```

Le remote et la branche `main` sont déjà configurés. Il ne reste que le push :

```bash
git push -u origin main
```

Git te demandera de te connecter à GitHub (navigateur ou jeton). Le dépôt ne contient
**aucune clé API** : elles sont dans `.env`, qui est exclu par `.gitignore`.

## 3. Renseigner les clés en secrets

Dans le dépôt : **Settings → Secrets and variables → Actions → New repository secret**.
Quatre secrets, noms exacts :

| Nom | Valeur |
|---|---|
| `ODDS_API_KEY` | ta clé The Odds API |
| `API_FOOTBALL_KEY` | ta clé API-Football |
| `FOOTBALL_DATA_ORG_KEY` | ton jeton football-data.org (buteurs et passeurs hors Premier League) |
| `HIGHLIGHTLY_KEY` | ta clé Highlightly (heures exactes et direct du Top 14 / Pro D2) |

C'est le seul endroit où elles doivent vivre côté GitHub.

## 4. Activer la page

**Settings → Pages → Build and deployment** : Source « Deploy from a branch »,
Branch `main`, dossier `/docs`, puis **Save**.

L'adresse sera `https://thewheygod.github.io/sportstats/` — c'est celle à mettre en
favori sur ton téléphone.

## 5. Premier lancement

Onglet **Actions → refresh → Run workflow**, mode `complet`. Compte trois à cinq minutes.
Ensuite tout est automatique :

| job | quand (heure de Paris, été) | coût |
|---|---|---|
| `complet` | chaque matin 7h30 | 1 crédit Odds API + 4 requêtes API-Football |
| `live` | toutes les 30 min de 11h à 0h30 | football : 2 requêtes API-Football, 0 crédit ; NRL : 1 crédit Odds API si match en cours |

Le live s'arrête seul sous 25 crédits Odds API pour préserver le cycle du matin.

---

## Une fois que ça tourne

**Désactive les deux tâches planifiées locales** (section « Scheduled » de l'app Claude :
`sportvalue-journee` et `sportvalue-live`). Elles consomment le même quota d'API pour le
même résultat ; les garder doublerait la dépense.

**Le lien claude.ai** (`claude.ai/code/artifact/…`) ne sera plus mis à jour automatiquement —
la page GitHub le remplace. Il reste consultable tel quel.

## Si quelque chose casse

Onglet **Actions**, clique sur le run en échec : la sortie de `refresh.py` y est complète,
y compris `stderr`. Les causes vues en développement :

- **401 sur The Odds API** → le secret `ODDS_API_KEY` est absent ou vaut un placeholder.
- **« Free plans do not have access to this date »** → normal, API-Football ne couvre que
  ±1 jour ; le script ne demande déjà que ces dates.
- **« quota sous la reserve »** → le live est suspendu jusqu'au renouvellement mensuel
  du quota Odds API (500 crédits le 1er de chaque mois). Le cycle du matin continue.

## Changement d'heure

Le cron GitHub est en UTC. Le matin passera de 7h30 à **6h30** fin octobre, et le live de
11h–23h à **10h–22h**. Sans conséquence ; pour recaler, décale d'une heure les deux lignes
`cron:` dans `.github/workflows/refresh.yml`.
