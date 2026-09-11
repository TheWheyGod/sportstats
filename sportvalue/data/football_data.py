"""
Source football : football-data.co.uk.

C'est la meilleure source gratuite pour construire ET valider un modele foot :
  - resultats depuis 1993 sur ~25 championnats,
  - statistiques de match (tirs, tirs cadres, corners, cartons),
  - et surtout les COTES DE CLOTURE de Pinnacle (PSCH/PSCD/PSCA).

Le dernier point est decisif : Pinnacle est le book de reference (marge ~2%,
accepte les gagnants), donc sa cote de cloture est la meilleure estimation
publique de la vraie probabilite. Sans elle, impossible de mesurer un CLV, donc
impossible de savoir si un edge est reel ou du bruit.

Pas de cle d'API, pas de compte. Un simple CSV par championnat et par saison.
"""
from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd

from .cache import get_cache

__all__ = ["LEAGUES", "load_league", "load_many", "season_codes", "available_odds_columns"]

BASE = "https://www.football-data.co.uk/mmz4281"

# Codes des championnats principaux
LEAGUES = {
    "E0": "Angleterre - Premier League",
    "E1": "Angleterre - Championship",
    "E2": "Angleterre - League One",
    "E3": "Angleterre - League Two",
    "SC0": "Ecosse - Premiership",
    "D1": "Allemagne - Bundesliga",
    "D2": "Allemagne - Bundesliga 2",
    "I1": "Italie - Serie A",
    "I2": "Italie - Serie B",
    "SP1": "Espagne - La Liga",
    "SP2": "Espagne - Segunda",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Pays-Bas - Eredivisie",
    "B1": "Belgique - Pro League",
    "P1": "Portugal - Primeira Liga",
    "T1": "Turquie - Super Lig",
    "G1": "Grece - Super League",
}

# Colonnes de cotes, par ordre de preference.
# Le suffixe C = "Closing" (cote de cloture).
ODDS_SETS = {
    "pinnacle_close": ("PSCH", "PSCD", "PSCA"),
    "pinnacle_open": ("PSH", "PSD", "PSA"),
    "max_close": ("MaxCH", "MaxCD", "MaxCA"),
    "max_open": ("MaxH", "MaxD", "MaxA"),
    "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
    "avg_open": ("AvgH", "AvgD", "AvgA"),
    "b365_close": ("B365CH", "B365CD", "B365CA"),
    "b365_open": ("B365H", "B365D", "B365A"),
    # Books grand public individuels. Ils servent a EMULER un portefeuille de
    # comptes ARJEL : un parieur francais n'a pas acces aux ~30 books derriere
    # la colonne Max, il a 3 a 5 comptes. Mesurer l'edge avec 5 books reels
    # plutot qu'avec le maximum mondial est la seule facon d'obtenir un chiffre
    # transposable a Winamax/Betclic/Unibet/PMU/NetBet.
    "bw_open": ("BWH", "BWD", "BWA"),           # Betway
    "iw_open": ("IWH", "IWD", "IWA"),           # Interwetten
    "vc_open": ("VCH", "VCD", "VCA"),           # VC Bet
    "wh_open": ("WHH", "WHD", "WHA"),           # William Hill
    "bw_close": ("BWCH", "BWCD", "BWCA"),
    "vc_close": ("VCCH", "VCCD", "VCCA"),
    "wh_close": ("WHCH", "WHCD", "WHCA"),
}

# Division inferieure de chaque championnat.
#
# L'inclure dans l'ajustement n'est PAS un detail : un promu n'a que quelques
# matchs dans l'elite, et le modele lui donne alors une note absurde. Hull
# City, promu, ressortait 4e de Premier League apres 3 matchs sans encaisser.
# En ajoutant le Championship il passe a 187 matchs et au 22e rang sur 54.
#
# Ce qui rend les deux echelles comparables, ce sont les PROMUS ET RELEGUES :
# ils jouent dans les deux divisions au fil des saisons, donc l'ecart de niveau
# entre elles est reellement estimable. C'est exactement ce qui manque entre
# deux pays, ou aucune equipe ne fait le pont.
TIER_BELOW = {
    "E0": "E1", "E1": "E2", "E2": "E3",
    "D1": "D2", "SP1": "SP2", "I1": "I2", "F1": "F2",
}


def with_lower_tier(codes):
    """Ajoute la division inferieure de chaque championnat, sans doublon."""
    out = []
    for c in codes:
        c = c.upper()
        if c not in out:
            out.append(c)
        low = TIER_BELOW.get(c)
        if low and low not in out:
            out.append(low)
    return out


# Ensemble emulant un portefeuille de comptes chez des books grand public.
SOFT_BOOK_SET = ("b365_open", "bw_open", "iw_open", "vc_open", "wh_open")

_TTL_HISTORY = 60 * 60 * 24 * 7    # 7 jours
_TTL_CURRENT = 60 * 60 * 6         # 6 h pour la saison en cours


def season_codes(start_year: int, end_year: int) -> list[str]:
    """2019, 2024 -> ['1920', '2021', '2122', '2223', '2324', '2425']"""
    out = []
    for y in range(start_year, end_year + 1):
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out


def _fetch_csv(url: str, ttl: float) -> pd.DataFrame:
    import requests

    def loader() -> str:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "sportvalue/1.0"})
        resp.raise_for_status()
        return resp.content.decode("latin-1")

    text = get_cache().get_text(url, ttl, loader)
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), encoding_errors="replace", on_bad_lines="skip")
    # .copy() : sans lui, pandas signale une fragmentation a chaque
    # ajout de colonne en aval (ces CSV ont ~120 colonnes).
    return df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()


def _parse_dates(s: pd.Series) -> pd.Series:
    """
    football-data melange dd/mm/yy et dd/mm/yyyy, parfois dans le meme fichier.
    On tente les deux formats explicitement plutot que de laisser pandas
    deviner : une inference silencieuse peut inverser jour et mois et decaler
    tout le backtest.
    """
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s[mask], format="%d/%m/%y", errors="coerce")
    return out


def load_league(
    league: str,
    seasons: Iterable[str],
    current_season: str | None = None,
) -> pd.DataFrame:
    """
    Charge un championnat sur plusieurs saisons et normalise les colonnes.

    Retourne un DataFrame trie par date avec au minimum :
        date, competition, saison, home, away, home_score, away_score
    plus les colonnes de cotes disponibles prefixees par leur jeu
    (odds_pinnacle_close_H/D/A, etc.) et les stats de match si presentes.
    """
    frames = []
    for season in seasons:
        url = f"{BASE}/{season}/{league}.csv"
        ttl = _TTL_CURRENT if season == current_season else _TTL_HISTORY
        try:
            raw = _fetch_csv(url, ttl)
        except Exception as exc:  # reseau indisponible, saison absente...
            print(f"  [!] {league} {season} : {type(exc).__name__} - {exc}")
            continue
        if raw.empty or "HomeTeam" not in raw.columns:
            continue
        frames.append(raw.assign(saison=season))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "date": _parse_dates(df["Date"]),
            "competition": LEAGUES.get(league, league),
            "league_code": league,
            "saison": df["saison"],
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
            "home_score": pd.to_numeric(df.get("FTHG"), errors="coerce"),
            "away_score": pd.to_numeric(df.get("FTAG"), errors="coerce"),
            "neutral": False,
        }
    )

    # cotes 1X2
    for name, (h, d_, a) in ODDS_SETS.items():
        if h in df.columns and d_ in df.columns and a in df.columns:
            out[f"odds_{name}_H"] = pd.to_numeric(df[h], errors="coerce")
            out[f"odds_{name}_D"] = pd.to_numeric(df[d_], errors="coerce")
            out[f"odds_{name}_A"] = pd.to_numeric(df[a], errors="coerce")

    # over/under 2.5 (cloture Pinnacle puis max)
    for src, dst in [("PC>2.5", "ou25_pin_over"), ("PC<2.5", "ou25_pin_under"),
                     ("P>2.5", "ou25_pin_over_open"), ("P<2.5", "ou25_pin_under_open"),
                     ("MaxC>2.5", "ou25_max_over"), ("MaxC<2.5", "ou25_max_under")]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    # handicap asiatique de cloture
    for src, dst in [("AHCh", "ah_line"), ("PCAHH", "ah_pin_home"), ("PCAHA", "ah_pin_away"),
                     ("AHh", "ah_line_open"), ("PAHH", "ah_pin_home_open"), ("PAHA", "ah_pin_away_open")]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    # score a la mi-temps et arbitre (l'arbitre n'est fourni que pour
    # l'Angleterre) : nourrissent les marches par periode et l'effet arbitre.
    for src, dst in [("HTHG", "ht_home"), ("HTAG", "ht_away")]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    if "Referee" in df.columns:
        out["referee"] = df["Referee"].astype(str).str.strip().replace({"nan": ""})

    # statistiques de match (utiles pour un modele base sur les tirs cadres)
    stat_map = {
        "HS": "shots_home", "AS": "shots_away",
        "HST": "sot_home", "AST": "sot_away",
        "HC": "corners_home", "AC": "corners_away",
        "HF": "fouls_home", "AF": "fouls_away",
        "HY": "yellow_home", "AY": "yellow_away",
        "HR": "red_home", "AR": "red_away",
    }
    for src, dst in stat_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["date", "home", "away"])
    out = out[out["home"].str.len() > 0]
    return out.sort_values("date").reset_index(drop=True)


def load_many(
    leagues: Iterable[str],
    seasons: Iterable[str],
    current_season: str | None = None,
) -> pd.DataFrame:
    seasons = list(seasons)
    frames = []
    for lg in leagues:
        df = load_league(lg, seasons, current_season)
        if not df.empty:
            print(f"  {lg:<4} {LEAGUES.get(lg, lg):<32} {len(df):>5} matchs "
                  f"({df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m})")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def available_odds_columns(df: pd.DataFrame) -> dict:
    """Diagnostique la couverture des cotes : indispensable avant un backtest."""
    out = {}
    for name in ODDS_SETS:
        cols = [f"odds_{name}_{s}" for s in "HDA"]
        if all(c in df.columns for c in cols):
            cov = float(df[cols].notna().all(axis=1).mean())
            out[name] = round(cov, 4)
    for extra in ["ou25_pin_over", "ou25_max_over", "ah_pin_home"]:
        if extra in df.columns:
            out[extra] = round(float(df[extra].notna().mean()), 4)
    return out


# ==========================================================================
# Championnats hors fichiers mmz4281
# ==========================================================================
# football-data publie un second jeu de fichiers, un par PAYS, couvrant des
# championnats absents des fichiers principaux. Le format differe : un seul
# fichier pour toutes les saisons, colonnes Country/League/Season/Home/Away/HG/AG
# au lieu de Div/HomeTeam/AwayTeam/FTHG/FTAG.
NEW_BASE = "https://www.football-data.co.uk/new"

# code interne -> (fichier pays, libelle de ligue a filtrer, nom affiche)
EXTRA_LEAGUES = {
    "ARG1": ("ARG", None, "Argentine - Liga Profesional"),
    "AUT1": ("AUT", None, "Autriche - Bundesliga"),
    "BRA1": ("BRA", None, "Bresil - Serie A"),
    "CHN1": ("CHN", None, "Chine - Super League"),
    "DNK1": ("DNK", None, "Danemark - Superliga"),
    "FIN1": ("FIN", None, "Finlande - Veikkausliiga"),
    "IRL1": ("IRL", None, "Irlande - Premier Division"),
    "JPN1": ("JPN", None, "Japon - J1 League"),
    "MEX1": ("MEX", None, "Mexique - Liga MX"),
    "NOR1": ("NOR", None, "Norvege - Eliteserien"),
    "POL1": ("POL", None, "Pologne - Ekstraklasa"),
    "ROU1": ("ROU", None, "Roumanie - Superliga"),
    "RUS1": ("RUS", None, "Russie - Premier League"),
    "SWE1": ("SWE", None, "Suede - Allsvenskan"),
    "SWZ1": ("SWZ", "Super League", "Suisse - Super League"),
    "USA1": ("USA", None, "Etats-Unis - MLS"),
}


def load_extra_league(code: str, depuis: int = 2022, verbose: bool = True) -> pd.DataFrame:
    """
    Charge un championnat du second jeu de fichiers.

    Deux differences de format a traiter, toutes deux silencieuses :
      - le fichier porte un BOM UTF-8 decode en latin-1, donc la premiere
        colonne s'appelle "ï»¿Country" et non "Country" ;
      - la saison s'ecrit tantot "2026" (calendrier civil : Bresil, Japon,
        Norvege) tantot "2025/2026" (calendrier europeen). On filtre donc sur
        l'ANNEE DE LA DATE, pas sur le libelle de saison.
    """
    if code not in EXTRA_LEAGUES:
        raise ValueError(f"Code inconnu : {code}. Choix : {sorted(EXTRA_LEAGUES)}")
    pays, filtre_ligue, nom = EXTRA_LEAGUES[code]
    url = f"{NEW_BASE}/{pays}.csv"
    try:
        raw = _fetch_csv(url, _TTL_CURRENT)
    except Exception as exc:
        if verbose:
            print(f"  [!] {code} : {type(exc).__name__} - {exc}")
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    raw.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in raw.columns]
    if "Home" not in raw.columns:
        return pd.DataFrame()
    if filtre_ligue and "League" in raw.columns:
        raw = raw[raw["League"].astype(str).str.strip() == filtre_ligue]

    out = pd.DataFrame({
        "date": _parse_dates(raw["Date"]),
        "competition": nom,
        "league_code": code,
        "saison": raw.get("Season", ""),
        "home": raw["Home"].astype(str).str.strip(),
        "away": raw["Away"].astype(str).str.strip(),
        "home_score": pd.to_numeric(raw.get("HG"), errors="coerce"),
        "away_score": pd.to_numeric(raw.get("AG"), errors="coerce"),
        "neutral": False,
    })
    for name, cols in [("pinnacle_close", ("PSCH", "PSCD", "PSCA")),
                       ("max_close", ("MaxCH", "MaxCD", "MaxCA")),
                       ("avg_close", ("AvgCH", "AvgCD", "AvgCA"))]:
        if all(c in raw.columns for c in cols):
            for suffixe, c in zip("HDA", cols):
                out[f"odds_{name}_{suffixe}"] = pd.to_numeric(raw[c], errors="coerce")

    out = out.dropna(subset=["date", "home", "away"])
    out = out[out["date"].dt.year >= depuis].sort_values("date").reset_index(drop=True)
    if verbose and len(out):
        print(f"  {code:<5} {nom:<32} {len(out):>5} matchs "
              f"({out['date'].min():%Y-%m} -> {out['date'].max():%Y-%m})")
    return out


def nom_competition(code: str) -> str:
    """
    Libelle affichable d'un championnat, quel que soit le jeu de fichiers.

    Sans cette fonction, les championnats du second jeu s'affichaient sous
    leur code brut ("BRA1", "ARG1") : lisible pour moi, opaque pour un
    lecteur de la page.
    """
    code = code.upper()
    if code in LEAGUES:
        return LEAGUES[code]
    if code in EXTRA_LEAGUES:
        return EXTRA_LEAGUES[code][2]
    from .wikipedia import WIKI_LEAGUES
    if code in WIKI_LEAGUES:
        return WIKI_LEAGUES[code][3]
    return code


def has_lower_tier(code: str) -> bool:
    """Les championnats du second jeu de fichiers n'ont pas de D2 exposee."""
    return code.upper() in TIER_BELOW


def load_any(codes, seasons=None, depuis: int = 2022, verbose: bool = True) -> pd.DataFrame:
    """
    Charge indifferemment des championnats des deux jeux de fichiers.

    Permet d'ecrire --leagues E0,F1,BRA1,JPN1 sans se soucier de la source.
    """
    seasons = list(seasons) if seasons else season_codes(depuis, depuis + 4)
    from .wikipedia import WIKI_LEAGUES, load_wiki_league

    principaux = [c for c in codes if c.upper() in LEAGUES]
    extras = [c for c in codes if c.upper() in EXTRA_LEAGUES]
    wikis = [c for c in codes if c.upper() in WIKI_LEAGUES]
    inconnus = [c for c in codes if c.upper() not in LEAGUES
                and c.upper() not in EXTRA_LEAGUES and c.upper() not in WIKI_LEAGUES]
    if inconnus and verbose:
        print(f"  [!] codes inconnus ignores : {inconnus}")

    frames = []
    if principaux:
        d = load_many([c.upper() for c in principaux], seasons)
        if not d.empty:
            frames.append(d)
    for c in extras:
        d = load_extra_league(c.upper(), depuis, verbose)
        if not d.empty:
            frames.append(d)
    for c in wikis:
        d = load_wiki_league(c.upper(), max(depuis, 2023), verbose=verbose)
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
