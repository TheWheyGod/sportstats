"""
Matchs a venir.

football-data.co.uk publie un `fixtures.csv` unique couvrant tous ses
championnats sur les jours qui viennent. C'est la source des rencontres a
predire pour le football, sans cle ni compte.

Attention a deux details qui cassent silencieusement :

  - le fichier commence par un BOM UTF-8, donc la premiere colonne s'appelle
    litteralement "﻿Div" et non "Div". Sans nettoyage, le filtrage par
    championnat ne trouve jamais rien ;
  - les noms d'equipes suivent la convention football-data, la meme que les
    fichiers de resultats. C'est justement ce qui permet d'apparier fixtures et
    historique sans table de correspondance.

Pour les autres sports, aucune source libre : on passe par un CSV de matchs
(`home,away[,date]`), qu'on remplit a la main ou qu'on copie d'un calendrier.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cache import get_cache

__all__ = ["upcoming_football", "upcoming_from_oddsapi", "load_fixtures_csv",
           "FIXTURES_TEMPLATE", "write_fixtures_template", "CODE_TO_ODDSAPI"]

URL = "https://www.football-data.co.uk/fixtures.csv"
_TTL = 60 * 60 * 3  # 3 h

FIXTURES_TEMPLATE = """home,away,date
Toulouse,Perpignan,2026-09-13
La Rochelle,Toulon,2026-09-13
Bordeaux,Clermont,2026-09-14
"""


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes du BOM.

    Piege : le fichier est en UTF-8 avec BOM, mais les autres fichiers
    football-data sont en latin-1. Decoder ce fichier-ci en latin-1 transforme
    les trois octets du BOM (EF BB BF) en la chaine "ï»¿", pas en U+FEFF. Un
    nettoyage qui ne retire que U+FEFF laisse donc passer "ï»¿Div", et le
    filtrage par championnat ne trouve plus jamais rien -- sans lever d'erreur.
    On retire donc les deux formes.
    """
    out = []
    for c in df.columns:
        c = str(c).replace("﻿", "").replace("ï»¿", "").strip()
        out.append(c)
    df.columns = out
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def upcoming_football(leagues=None, verbose: bool = True) -> pd.DataFrame:
    """
    Rencontres a venir. Retourne date, league_code, home, away.

    `leagues` filtre sur les codes football-data (E0, F1, D1...). None = tout.
    """
    import requests

    def loader() -> str:
        r = requests.get(URL, timeout=30, headers={"User-Agent": "sportvalue/1.0"})
        r.raise_for_status()
        return r.content.decode("latin-1")

    text = get_cache().get_text(URL, _TTL, loader)
    df = _clean_columns(pd.read_csv(io.StringIO(text), on_bad_lines="skip"))
    if "Div" not in df.columns or "HomeTeam" not in df.columns:
        return pd.DataFrame(columns=["date", "league_code", "home", "away"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"),
            "heure": df.get("Time", ""),
            "league_code": df["Div"].astype(str).str.strip(),
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
        }
    ).dropna(subset=["home", "away"])

    if leagues:
        codes = [c.strip().upper() for c in leagues]
        out = out[out["league_code"].str.upper().isin(codes)]

    out = out.sort_values(["date", "league_code"]).reset_index(drop=True)
    if verbose:
        if out.empty:
            print("   Aucun match a venir pour ce filtre "
                  "(le fichier ne couvre que les prochains jours).")
        else:
            par = out["league_code"].value_counts().to_dict()
            print(f"   {len(out)} match(s) a venir : {par}")
    return out


# Correspondance cles The Odds API -> codes football-data.
ODDSAPI_TO_CODE = {
    "soccer_epl": "E0",
    "soccer_efl_champ": "E1",
    "soccer_england_league1": "E2",
    "soccer_england_league2": "E3",
    "soccer_france_ligue_one": "F1",
    "soccer_france_ligue_two": "F2",
    "soccer_spain_la_liga": "SP1",
    "soccer_spain_segunda_division": "SP2",
    "soccer_italy_serie_a": "I1",
    "soccer_italy_serie_b": "I2",
    "soccer_germany_bundesliga": "D1",
    "soccer_germany_bundesliga2": "D2",
    "soccer_netherlands_eredivisie": "N1",
    "soccer_portugal_primeira_liga": "P1",
    "soccer_belgium_first_div": "B1",
    "soccer_turkey_super_league": "T1",
    "soccer_greece_super_league": "G1",
    "soccer_spl": "SC0",
    # Championnats du second jeu de fichiers football-data.
    "soccer_argentina_primera_division": "ARG1",
    "soccer_austria_bundesliga": "AUT1",
    "soccer_brazil_campeonato": "BRA1",
    "soccer_china_superleague": "CHN1",
    "soccer_denmark_superliga": "DNK1",
    "soccer_finland_veikkausliiga": "FIN1",
    "soccer_league_of_ireland": "IRL1",
    "soccer_japan_j_league": "JPN1",
    "soccer_mexico_ligamx": "MEX1",
    "soccer_norway_eliteserien": "NOR1",
    "soccer_poland_ekstraklasa": "POL1",
    "soccer_romania_liga_1": "ROU1",
    "soccer_russia_premier_league": "RUS1",
    "soccer_sweden_allsvenskan": "SWE1",
    "soccer_switzerland_superleague": "SWZ1",
    "soccer_usa_mls": "USA1",
}

# Presélections, pour ne pas avoir a taper quinze codes.
PRESETS = {
    "top5": ["E0", "SP1", "I1", "D1", "F1"],
    # Les quinze championnats nationaux les plus suivis pour lesquels on
    # dispose A LA FOIS d'un historique de resultats et d'un calendrier.
    "top15": ["E0", "SP1", "I1", "D1", "F1", "P1", "N1", "T1", "B1",
              "BRA1", "ARG1", "MEX1", "USA1", "JPN1", "SC0"],
    "d2": ["E1", "SP2", "I2", "D2", "F2"],
    "europe": ["E0", "SP1", "I1", "D1", "F1", "P1", "N1", "T1", "B1",
               "SC0", "G1", "AUT1", "DNK1", "NOR1", "SWE1", "POL1", "SWZ1"],
    "monde": ["BRA1", "ARG1", "MEX1", "USA1", "JPN1", "CHN1"],
}


def expand_presets(codes):
    """Remplace un nom de preselection par les codes correspondants."""
    out = []
    for c in codes:
        for x in PRESETS.get(c.strip().lower(), [c.strip().upper()]):
            if x not in out:
                out.append(x)
    return out
CODE_TO_ODDSAPI = {v: k for k, v in ODDSAPI_TO_CODE.items()}


def upcoming_from_oddsapi(codes, api_key: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Calendrier via The Odds API. SEULES la date et les equipes sont lues :
    aucune cote n'entre dans les predictions.

    Pourquoi ne pas se contenter de fixtures.csv : ce fichier est mis a jour
    manuellement par football-data et peut rester fige plusieurs jours. Observe
    en production : un 11/09, l'en-tete Last-Modified indiquait le 08/09 et le
    fichier ne contenait plus que des matchs DEJA JOUES. Une source de
    calendrier qui renvoie le passe sans le signaler est pire qu'une source
    absente, puisque rien dans les donnees ne trahit le probleme.

    Cout : 1 credit par competition (une region, un marche).
    """
    import os

    import requests

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY absente : calendrier live indisponible.")

    lignes = []
    for code in codes:
        sk = CODE_TO_ODDSAPI.get(code.upper())
        if sk is None:
            if verbose:
                print(f"   [!] {code} : pas de correspondance The Odds API, ignore")
            continue
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sk}/odds",
                params={"apiKey": key, "regions": "eu", "markets": "h2h",
                        "oddsFormat": "decimal", "dateFormat": "iso"},
                timeout=30,
            )
            if r.status_code == 422:
                if verbose:
                    print(f"   [!] {code} : competition hors saison")
                continue
            r.raise_for_status()
        except Exception as e:
            if verbose:
                print(f"   [!] {code} : {type(e).__name__}")
            continue
        for ev in r.json():
            try:
                dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            lignes.append({
                "date": dt.astimezone(timezone.utc).replace(tzinfo=None),
                "heure": dt.astimezone(timezone.utc).strftime("%H:%M"),
                "league_code": code.upper(),
                "home": ev.get("home_team", ""),
                "away": ev.get("away_team", ""),
            })

    df = pd.DataFrame(lignes)
    if df.empty:
        return pd.DataFrame(columns=["date", "heure", "league_code", "home", "away"])
    df = df.sort_values(["date", "league_code"]).reset_index(drop=True)
    if verbose:
        print(f"   {len(df)} match(s) a venir : {df['league_code'].value_counts().to_dict()}")
        print(f"   du {df['date'].min():%d/%m} au {df['date'].max():%d/%m}")
    return df


# Noms que la normalisation ne rapproche pas toute seule : football-data
# abrege (Nott'm Forest, M'gladbach) la ou The Odds API ecrit en toutes lettres.
ALIAS = {
    "manchester united": "Man United",
    "manchester city": "Man City",
    "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "west ham united": "West Ham",
    "leicester city": "Leicester",
    "borussia dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "bayer leverkusen": "Leverkusen",
    "paris saint germain": "Paris SG",
    "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "atletico madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "ca osasuna": "Osasuna",
    "rayo vallecano": "Vallecano",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "celta vigo": "Celta",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "atalanta bc": "Atalanta",
    "hellas verona": "Verona",
}


def _normalise(x: str) -> str:
    """Reduit un nom d'equipe a son noyau : sans accents, sans prefixe de club."""
    import re
    import unicodedata

    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    x = re.sub(
        r"\b(fc|sc|ac|as|ss|us|cf|rc|sv|vfb|vfl|fsv|tsg|bsc|ca|cd|ud|rcd|"
        r"1899|1\.|04|05|09|deportivo|club|calcio|real|olympique)\b", " ", x)
    x = re.sub(r"[^a-z ]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def resolve_team(nom: str, connues) -> str | None:
    """
    Apparie un nom de calendrier a un nom d'historique.

    Cinq passes, de la plus sure a la plus permissive. Aucune ne tranche au
    hasard : des qu'une passe trouve plusieurs candidats, elle passe la main.
    Predire le mauvais club est pire que ne rien predire, d'ou un appariement
    volontairement conservateur -- « Paris » reste ambigu entre Paris SG et
    Paris FC, donc None.

    Les deux dernieres passes existent parce que les calendriers mondiaux
    ecrivent les clubs autrement que football-data : "Hiroshima Sanfrecce FC"
    contre "Sanfrecce Hiroshima" (mots inverses), "Atletico Mineiro" contre
    "Atl Mineiro" (abreviation). Un prefixe commun n'y suffit pas.
    """
    import difflib

    connues = set(connues)
    if nom in connues:
        return nom
    a = ALIAS.get(_normalise(nom)) or ALIAS.get(str(nom).strip().lower())
    if a and a in connues:
        return a
    table = {_normalise(k): k for k in connues}
    n = _normalise(nom)
    if n in table:
        return table[n]
    if len(n) >= 5:
        cands = {v for k, v in table.items() if k.startswith(n[:5]) or n.startswith(k[:5])}
        if len(cands) == 1:
            return cands.pop()

    # Passe par MOTS : "hiroshima sanfrecce" et "sanfrecce hiroshima" portent
    # exactement les memes mots significatifs, dans un ordre different.
    mots = {m for m in n.split() if len(m) > 3}
    if mots:
        exacts = [v for k, v in table.items()
                  if {m for m in k.split() if len(m) > 3} == mots]
        if len(exacts) == 1:
            return exacts[0]
        for mot in sorted(mots, key=len, reverse=True):
            porteurs = [v for k, v in table.items() if mot in k.split()]
            if len(porteurs) == 1:
                return porteurs[0]

    # Similarite globale, avec un ECART NET exige entre le meilleur et le
    # suivant. Sans cette marge, "Estudiantes" et "Estudiantes de Rio Cuarto"
    # seraient confondus : precisement l'erreur a eviter.
    scores = sorted(
        ((difflib.SequenceMatcher(None, n, k).ratio(), v) for k, v in table.items()),
        reverse=True,
    )
    if scores and scores[0][0] >= 0.87:
        if len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08:
            return scores[0][1]
    return None


def load_fixtures_csv(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """CSV de matchs a predire : home,away[,date]."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {"domicile": "home", "exterieur": "away", "joueur_a": "home", "joueur_b": "away"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    manque = {"home", "away"} - set(df.columns)
    if manque:
        raise ValueError(f"Colonnes manquantes : {sorted(manque)} (attendu : home, away[, date])")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    df = df.dropna(subset=["home", "away"]).reset_index(drop=True)
    if verbose:
        print(f"   {len(df)} match(s) a predire")
    return df


def write_fixtures_template(path: str | Path) -> Path:
    p = Path(path)
    p.write_text(FIXTURES_TEMPLATE, encoding="utf-8")
    return p


def nrl_fixtures(api_key: str | None = None) -> list[dict]:
    """
    Calendrier NRL via The Odds API (dates et equipes uniquement).

    On interroge les regions au ET eu : la NRL est surtout cotee par des books
    australiens, et se limiter a eu renvoie souvent une liste vide.
    """
    import os

    import requests

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY absente : calendrier NRL indisponible.")
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/rugbyleague_nrl/odds",
        params={"apiKey": key, "regions": "au,eu", "markets": "h2h",
                "oddsFormat": "decimal", "dateFormat": "iso"},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for ev in r.json():
        try:
            dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        out.append({"date": dt.astimezone(timezone.utc).replace(tzinfo=None),
                    "home": ev.get("home_team", ""), "away": ev.get("away_team", "")})
    return sorted(out, key=lambda x: x["date"])


def upcoming_from_apifootball(codes, api_key: str | None = None, verbose: bool = True):
    """
    Calendrier des 48 prochaines heures via API-Football.

    Le plan gratuit bride les dates a +/- 1 jour, mais UNE requete par date
    couvre toutes les competitions. Pour aujourd'hui et demain cela coute donc
    2 requetes sur un quota de 100/jour -- contre 1 credit PAR championnat chez
    The Odds API (20 credits pour 20 championnats sur un quota de 500/mois).

    C'est ce qui rend un rafraichissement quotidien tenable avec le plan
    gratuit : le calendrier proche ne coute plus rien cote Odds API, dont les
    credits sont reserves aux scores en direct.
    """
    import datetime as _dt
    import os

    import requests

    from .injuries import LEAGUE_IDS

    key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY absente.")
    ids = {LEAGUE_IDS[c]: c for c in codes if c in LEAGUE_IDS}
    manquants = [c for c in codes if c not in LEAGUE_IDS]
    if manquants and verbose:
        print(f"   [i] sans identifiant API-Football, hors calendrier proche : {manquants}")

    auj = _dt.date.today()
    lignes = []
    for k in (0, 1):
        d = (auj + _dt.timedelta(days=k)).isoformat()
        url = f"https://v3.football.api-sports.io/fixtures?date={d}"

        def loader():
            r = requests.get("https://v3.football.api-sports.io/fixtures",
                             headers={"x-apisports-key": key}, params={"date": d}, timeout=40)
            r.raise_for_status()
            return r.json()

        try:
            j = get_cache().get_text(url, 60 * 60 * 2, lambda: __import__("json").dumps(loader()))
            j = __import__("json").loads(j)
        except Exception as e:
            if verbose:
                print(f"   [!] calendrier {d} : {type(e).__name__}")
            continue
        if j.get("errors"):
            if verbose:
                print(f"   [!] calendrier {d} : {j['errors']}")
            continue
        for x in j.get("response", []):
            lg = (x.get("league") or {}).get("id")
            if lg not in ids:
                continue
            st = ((x.get("fixture") or {}).get("status") or {}).get("short", "")
            if st in ("FT", "AET", "PEN", "CANC", "PST", "ABD"):
                continue   # deja joue ou annule
            try:
                dt_ = datetime.fromisoformat(x["fixture"]["date"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            lignes.append({
                "date": dt_.astimezone(timezone.utc).replace(tzinfo=None),
                "heure": dt_.astimezone(timezone.utc).strftime("%H:%M"),
                "league_code": ids[lg],
                "home": (x.get("teams") or {}).get("home", {}).get("name", ""),
                "away": (x.get("teams") or {}).get("away", {}).get("name", ""),
            })

    df = pd.DataFrame(lignes)
    if df.empty:
        return pd.DataFrame(columns=["date", "heure", "league_code", "home", "away"])
    df = df.sort_values(["date", "league_code"]).reset_index(drop=True)
    if verbose:
        print(f"   {len(df)} match(s) sur 48 h : {df['league_code'].value_counts().to_dict()}")
    return df
