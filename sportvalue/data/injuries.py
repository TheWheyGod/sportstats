"""
Absences (blessures et suspensions) via API-Football.

C'EST LA DONNEE QUI MANQUAIT depuis le debut : une composition amputee de son
buteur change reellement les probabilites, et aucune des sources gratuites
utilisees jusqu'ici ne la publie -- sauf FPL, limite a la Premier League.

DEUX PIEGES DU PLAN GRATUIT
----------------------------
1. Le parametre `season` est bride aux saisons 2022-2024. Interroger la saison
   en cours renvoie zero resultat avec le message "Free plans do not have
   access to this season". MAIS les requetes PAR DATE ne sont pas bridees :
   `/injuries?date=2026-09-12` renvoie bien les absences du jour. C'est le
   contournement, et il est officiel, pas un detournement.

2. Quota de 100 requetes par jour. Un appel par date couvre TOUTES les ligues
   d'un coup (environ 950 absences), donc une fenetre de dix jours coute dix
   requetes. Interroger ligue par ligue serait vingt fois plus cher pour le
   meme resultat.

UN PIEGE DE DONNEES
-------------------
Les noms de championnats se repetent d'un pays a l'autre : filtrer sur
"Serie A" ramene le Brasileirao en plus de l'Italie. On filtre donc sur les
IDENTIFIANTS de ligue, jamais sur les libelles.
"""
from __future__ import annotations

from collections import defaultdict

from .cache import get_cache

__all__ = ["LEAGUE_IDS", "fetch_injuries", "by_team", "TYPES"]

BASE = "https://v3.football.api-sports.io"
_TTL = 60 * 60 * 3  # 3 h : une compo evolue dans la journee

# Identifiants API-Football des championnats couverts, par code football-data.
LEAGUE_IDS = {
    "E0": 39,    # Premier League
    "E1": 40,    # Championship
    "F1": 61,    # Ligue 1
    "F2": 62,    # Ligue 2
    "SP1": 140,  # La Liga
    "I1": 135,   # Serie A (Italie)
    "D1": 78,    # Bundesliga
    "D2": 79,
    "SP2": 141,  # Segunda
    "I2": 136,   # Serie B
    "N1": 88,    # Eredivisie
    "P1": 94,    # Primeira Liga
    "B1": 144,   # Jupiler Pro League
    "T1": 203,   # Super Lig
    "SC0": 179,  # Premiership ecossaise
    "G1": 197,   # Super League grecque
    "BRA1": 71, "ARG1": 128, "MEX1": 262, "USA1": 253, "JPN1": 98,
    "AUT1": 218, "DNK1": 119, "NOR1": 103, "SWE1": 113, "POL1": 106,
    "SWZ1": 207, "CHN1": 169,
}
ID_TO_CODE = {v: k for k, v in LEAGUE_IDS.items()}

# `type` renvoye par l'API -> poids d'indisponibilite.
# "Questionable" signifie incertain : compter une telle absence comme une
# absence certaine surestime l'impact, l'ignorer le sous-estime. 0.5 est le
# compromis neutre, faute d'information sur la probabilite reelle de forfait.
TYPES = {"Missing Fixture": 1.0, "Questionable": 0.5}


def fetch_injuries(dates, api_key: str | None = None, leagues=None,
                   verbose: bool = True) -> list:
    """
    Absences pour une liste de dates (format AAAA-MM-JJ).

    `leagues` : codes football-data (E0, F1...). None = tout garder.
    Cout : 1 requete par date, toutes ligues confondues.
    """
    import os

    import requests

    key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY absente : absences indisponibles.")
    ids = {LEAGUE_IDS[c] for c in leagues if c in LEAGUE_IDS} if leagues else None

    # Le plan gratuit n'autorise qu'une fenetre de +/- 1 jour autour
    # d'aujourd'hui. Interroger au-dela renvoie une erreur de plan ET consomme
    # une requete du quota (100/jour) : on filtre donc en amont plutot que de
    # bruler huit appels pour huit refus.
    import datetime as _dt

    auj = _dt.date.today()
    fenetre = {(auj + _dt.timedelta(days=k)).isoformat() for k in (-1, 0, 1)}
    demandees = list(dict.fromkeys(dates))
    retenues = [d for d in demandees if d in fenetre]
    hors = [d for d in demandees if d not in fenetre]
    if hors and verbose:
        print(f"   {len(hors)} date(s) hors de la fenetre autorisee par le plan "
              f"gratuit ({min(fenetre)} a {max(fenetre)}), non interrogee(s)")

    out = []
    for d in retenues:
        url = f"{BASE}/injuries?date={d}"

        def loader():
            r = requests.get(f"{BASE}/injuries", headers={"x-apisports-key": key},
                             params={"date": d}, timeout=40)
            r.raise_for_status()
            return r.json()

        try:
            j = get_cache().get_json(url, _TTL, loader)
        except Exception as e:
            if verbose:
                print(f"   [!] absences {d} : {type(e).__name__}")
            continue
        if j.get("errors"):
            if verbose:
                print(f"   [!] absences {d} : {j['errors']}")
            continue

        for x in j.get("response", []):
            lg = (x.get("league") or {}).get("id")
            if ids is not None and lg not in ids:
                continue
            p, t = x.get("player") or {}, x.get("team") or {}
            out.append({
                "date": d,
                "league_code": ID_TO_CODE.get(lg, str(lg)),
                "equipe": t.get("name", ""),
                "joueur": p.get("name", ""),
                "type": p.get("type", ""),
                "poids": TYPES.get(p.get("type", ""), 0.5),
                "raison": p.get("reason", ""),
            })

    # L'API renvoie une ligne par rencontre concernee : un meme joueur
    # apparait plusieurs fois. On deduplique sur (equipe, joueur).
    vus, uniq = set(), []
    for x in out:
        cle = (x["equipe"].lower(), x["joueur"].lower())
        if cle in vus:
            continue
        vus.add(cle)
        uniq.append(x)

    if verbose:
        n_out = sum(1 for x in uniq if x["poids"] >= 1.0)
        print(f"   {len(uniq)} absence(s) : {n_out} forfait(s), "
              f"{len(uniq) - n_out} incertain(s), sur {len(retenues)} date(s)")
    return uniq


def by_team(injuries: list) -> dict:
    """Regroupe par equipe, forfaits certains d'abord."""
    d = defaultdict(list)
    for x in injuries:
        d[x["equipe"]].append(x)
    for k in d:
        d[k].sort(key=lambda x: (-x["poids"], x["joueur"]))
    return dict(d)


# API-Football et football-data n'ecrivent pas les equipes pareil.
ALIAS = {
    "manchester united": "Man United", "manchester city": "Man City",
    "newcastle": "Newcastle", "tottenham": "Tottenham",
    "nottingham forest": "Nott'm Forest", "wolves": "Wolves",
    "brighton": "Brighton", "west ham": "West Ham", "leicester": "Leicester",
    "borussia dortmund": "Dortmund", "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "bayer leverkusen": "Leverkusen",
    "fc koln": "FC Koln", "1899 hoffenheim": "Hoffenheim",
    "paris saint germain": "Paris SG", "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon", "stade rennais": "Rennes",
    "atletico madrid": "Ath Madrid", "athletic club": "Ath Bilbao",
    "rayo vallecano": "Vallecano", "real sociedad": "Sociedad",
    "real betis": "Betis", "celta vigo": "Celta",
    "inter": "Inter", "ac milan": "Milan", "as roma": "Roma",
    "atalanta": "Atalanta", "hellas verona": "Verona",
}


def resolve_team(nom: str, connues) -> str | None:
    """Apparie un nom API-Football a un nom football-data."""
    connues = set(connues)
    if nom in connues:
        return nom
    a = ALIAS.get(str(nom).strip().lower())
    if a and a in connues:
        return a
    cible = str(nom).lower().strip()
    exact = [c for c in connues if c.lower() == cible]
    if exact:
        return exact[0]
    cands = {c for c in connues if cible in c.lower() or c.lower() in cible}
    return cands.pop() if len(cands) == 1 else None
