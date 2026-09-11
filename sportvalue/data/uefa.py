"""
Coupes d'Europe : calendrier C1/C3 et force relative des championnats.

LE PROBLEME, ET IL N'EST PAS DE NATURE INFORMATIQUE
----------------------------------------------------
Un modele Dixon-Coles ajuste sur la Serie A donne a chaque club italien une
note d'attaque et de defense relative a la MOYENNE ITALIENNE. Le meme modele
sur la Bundesliga donne des notes relatives a la moyenne allemande. Ces deux
echelles n'ont aucun point commun : aucune equipe ne joue dans les deux
championnats, donc rien dans les donnees domestiques ne dit si la Serie A est
plus forte, plus faible ou equivalente a la Bundesliga.

C'est un defaut d'IDENTIFIABILITE au sens statistique. Ajuster un modele plus
gros n'y change rien : le parametre n'est pas estimable, point.

Trois facons d'ancrer l'echelle, par ordre de qualite :

  1. RESULTATS EUROPEENS DANS L'HISTORIQUE. Si l'historique contient des matchs
     C1/C3, la comparaison devient directe et le parametre s'estime. C'est la
     seule solution reellement satisfaisante. Passer un CSV de resultats
     europeens a `--historique` suffit : les equipes des deux pays s'y
     rencontrent, l'echelle se soude toute seule.

  2. PROMUS ET RELEGUES, a l'interieur d'un meme pays. E0 et E1 partagent des
     clubs d'une saison a l'autre : un ajustement conjoint sur les deux
     divisions relie donc bien leurs echelles. Cela marche entre divisions
     d'un meme pays, jamais entre pays.

  3. UN PRIOR EXPLICITE, ci-dessous. Les valeurs de `FORCE_CHAMPIONNAT` sont
     des DECALAGES POSES A LA MAIN, en log-buts, inspires de l'ordre des
     coefficients UEFA. Ce ne sont PAS des estimations issues de donnees.
     Elles sont documentees pour pouvoir etre discutees et modifiees, et
     l'outil signale toujours quand il s'en sert.

Un decalage de +0.10 en log-buts vaut environ +10% de buts marques attendus a
niveau domestique egal. L'ecart pose entre le sommet et le bas de cette table
est d'environ 0.40, soit un rapport de 1.5 -- ordre de grandeur plausible,
mais assume comme un choix, pas comme une mesure.
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = [
    "FORCE_CHAMPIONNAT",
    "ODDSAPI_TO_FOOTBALLDATA",
    "european_fixtures",
    "resolve_team",
    "COMPETITIONS_EUROPE",
]

# Decalage en log-buts. 0 = moyenne des championnats couverts.
# PRIOR EXPLICITE, non estime. A ajuster librement.
FORCE_CHAMPIONNAT = {
    "E0": 0.10,   # Premier League
    "I1": 0.08,   # Serie A
    "SP1": 0.08,  # Liga
    "D1": 0.06,   # Bundesliga
    "F1": 0.02,   # Ligue 1
    "N1": -0.10,  # Eredivisie
    "P1": -0.10,  # Primeira Liga
    "B1": -0.16,  # Jupiler Pro League
    "T1": -0.16,  # Super Lig
    "G1": -0.22,  # Super League grecque
    "E1": -0.22,  # Championship
    "D2": -0.24,
    "SP2": -0.26,
    "I2": -0.26,
    "F2": -0.28,
    "SC0": -0.30,  # Premiership ecossaise
    "E2": -0.42,
    "E3": -0.55,
}

COMPETITIONS_EUROPE = {
    "c1": ("soccer_uefa_champs_league", "Ligue des champions"),
    "c3": ("soccer_uefa_europa_league", "Ligue Europa"),
    "c4": ("soccer_uefa_europa_conference_league", "Ligue Conference"),
}

# The Odds API et football-data.co.uk n'ecrivent pas les noms pareil.
ODDSAPI_TO_FOOTBALLDATA = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Borussia Monchengladbach": "M'gladbach",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Paris Saint Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "RC Lens": "Lens",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "AS Roma": "Roma",
    "Atalanta BC": "Atalanta",
    "Sporting Lisbon": "Sp Lisbon",
    "FC Porto": "Porto",
    "Benfica": "Benfica",
    "Ajax Amsterdam": "Ajax",
    "PSV Eindhoven": "PSV Eindhoven",
    "Feyenoord": "Feyenoord",
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Club Brugge": "Club Brugge",
}


def resolve_team(nom: str, connues: set) -> str | None:
    """
    Fait correspondre un nom Odds API a un nom football-data.

    Retourne None si l'equipe n'apparait dans aucun championnat couvert : c'est
    le cas normal pour les clubs scandinaves, tcheques, azerbaidjanais... et il
    vaut mieux ecarter la rencontre que produire une prediction inventee.
    """
    if nom in connues:
        return nom
    mappe = ODDSAPI_TO_FOOTBALLDATA.get(nom)
    if mappe and mappe in connues:
        return mappe
    cible = nom.lower().replace("fc ", "").replace(" fc", "").strip()
    for c in connues:
        if c.lower() == cible:
            return c
    # correspondance par prefixe, uniquement si elle est unique
    cands = [c for c in connues if c.lower().startswith(cible[:6])] if len(cible) >= 6 else []
    return cands[0] if len(cands) == 1 else None


def european_fixtures(competition: str = "c1", api_key: str | None = None) -> list[dict]:
    """
    Calendrier des coupes d'Europe via The Odds API.

    Seuls la date et les deux equipes sont utilises : AUCUNE cote n'entre dans
    les predictions. L'API sert ici d'annuaire de rencontres, faute de source
    libre couvrant les competitions europeennes.
    """
    import os

    import requests

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY absente : impossible de recuperer le calendrier europeen.")
    sport_key, _ = COMPETITIONS_EUROPE.get(competition, COMPETITIONS_EUROPE["c1"])

    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
        params={"apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for ev in r.json():
        try:
            dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        out.append(
            {
                "date": dt.astimezone(timezone.utc).replace(tzinfo=None),
                "home": ev.get("home_team", ""),
                "away": ev.get("away_team", ""),
            }
        )
    return sorted(out, key=lambda x: x["date"])
