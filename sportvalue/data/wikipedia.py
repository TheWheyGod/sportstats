"""
Championnats sans source de resultats structuree : tableaux croises Wikipedia.

football-data.co.uk couvre 34 championnats. Au-dela (Arabie saoudite, Coree,
Australie...), il n'existe aucun fichier de resultats libre. Mais les pages de
saison de Wikipedia -- surtout en anglais -- contiennent presque toujours un
tableau croise complet : lignes = equipe a domicile, colonnes = adversaire,
cellule "2-1". Une saison de 18 equipes y tient en 306 cellules.

C'est la meme technique que pour le Top 14 (rugby_fr.py), generalisee :
langue au choix, gabarit de titre par championnat, et un parseur qui tolere
les deux mises en forme rencontrees (ligne d'en-tete presente ou non).

MEMES LIMITES QUE POUR LE RUGBY
-------------------------------
- Pas de dates match par match : on repartit uniformement sur la saison. La
  decroissance temporelle reste juste d'une saison a l'autre.
- Source communautaire : un tableau peut etre incomplet en cours de saison.
  Le taux de remplissage est affiche a chaque chargement.
- Les noms d'equipes sont ceux de Wikipedia, pas ceux du calendrier : la
  resolution de noms (fixtures.resolve_team) fait le pont.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from .cache import get_cache

__all__ = ["WIKI_LEAGUES", "load_wiki_league", "load_wiki_season"]

_TTL = 60 * 60 * 24 * 3
_SCORE = re.compile(r"^\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*$")

# code -> (langue, gabarit de titre, format de saison, nom affiche)
#   format "en" : "2025–26" (tiret demi-cadratin), format "fr" : "2025-2026"
WIKI_LEAGUES = {
    "SAU1": ("en", "{saison} Saudi Pro League", "en", "Arabie saoudite - Pro League"),
    "KOR1": ("en", "{saison} K League 1", "cal", "Coree du Sud - K League 1"),
    "AUS1": ("en", "{saison} A-League Men", "en", "Australie - A-League"),
    "EGY1": ("en", "{saison} Egyptian Premier League", "en", "Egypte - Premier League"),
    "MAR1": ("en", "{saison} Botola", "en", "Maroc - Botola"),
    "CZE1": ("en", "{saison} Czech First League", "en", "Tchequie - First League"),
    "CRO1": ("en", "{saison} Croatian Football League", "en", "Croatie - HNL"),
    "SRB1": ("en", "{saison} Serbian SuperLiga", "en", "Serbie - SuperLiga"),
    "UKR1": ("en", "{saison} Ukrainian Premier League", "en", "Ukraine - Premier League"),
    "COL1": ("en", "{saison} Categoría Primera A", "cal", "Colombie - Primera A"),
}


def _saison_libelle(annee_debut: int, fmt: str) -> str:
    if fmt == "en":
        return f"{annee_debut}–{(annee_debut + 1) % 100:02d}"
    if fmt == "fr":
        return f"{annee_debut}-{annee_debut + 1}"
    return str(annee_debut)          # calendrier civil


def _page_html(lang: str, titre: str) -> str:
    import requests

    url = f"https://{lang}.wikipedia.org/wiki/" + requests.utils.quote(titre.replace(" ", "_"))

    def loader() -> str:
        r = requests.get(url, timeout=45, headers={"User-Agent": "sportvalue/1.0"})
        r.raise_for_status()
        return r.text

    return get_cache().get_text(url, _TTL, loader)


def _find_cross_table(html: str):
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return None
    best, best_n = None, 0
    for t in tables:
        if t.shape[0] < 8 or abs(t.shape[0] - t.shape[1]) > 2:
            continue
        n = sum(1 for v in t.astype(str).values.ravel() if _SCORE.match(str(v)))
        if n > best_n:
            best, best_n = t, n
    return best if best_n >= 20 else None


def _parse(t: pd.DataFrame) -> list:
    """
    Extrait les matchs d'un tableau croise, quelle que soit sa mise en forme.

    Deux cas rencontres :
      - Wikipedia FR : la premiere LIGNE est un en-tete d'abreviations, les
        equipes commencent en ligne 1 ;
      - Wikipedia EN : l'en-tete est deja dans les noms de colonnes, les
        equipes commencent en ligne 0.
    On detecte le cas en regardant si la ligne 0 contient des scores.
    """
    row0 = [str(x) for x in t.iloc[0, 1:]]
    a_entete = not any(_SCORE.match(x) for x in row0)
    debut = 1 if a_entete else 0
    equipes = [str(x).strip() for x in t.iloc[debut:, 0]]
    n = len(equipes)
    # Colonnes : memes equipes, meme ordre, apres la colonne des noms.
    out = []
    for i in range(n):
        for j in range(n):
            if i == j or j + 1 >= t.shape[1]:
                continue
            m = _SCORE.match(str(t.iloc[debut + i, j + 1]).strip())
            if m:
                out.append({"home": equipes[i], "away": equipes[j],
                            "home_score": int(m.group(1)), "away_score": int(m.group(2))})
    return out


def load_wiki_season(code: str, annee_debut: int, verbose: bool = True) -> pd.DataFrame:
    lang, gabarit, fmt, nom = WIKI_LEAGUES[code]
    saison = _saison_libelle(annee_debut, fmt)
    titre = gabarit.format(saison=saison)
    try:
        t = _find_cross_table(_page_html(lang, titre))
    except Exception as e:
        if verbose:
            print(f"   [!] {titre} : {type(e).__name__}")
        return pd.DataFrame()
    if t is None:
        if verbose:
            print(f"   [!] {titre} : tableau croise introuvable")
        return pd.DataFrame()

    lignes = _parse(t)
    if not lignes:
        return pd.DataFrame()
    df = pd.DataFrame(lignes)
    n = df["home"].nunique()
    # Saison sur deux annees civiles (aout -> mai) ou une seule (calendrier).
    if fmt == "cal":
        debut, fin = pd.Timestamp(annee_debut, 2, 1), pd.Timestamp(annee_debut, 12, 15)
    else:
        debut, fin = pd.Timestamp(annee_debut, 8, 15), pd.Timestamp(annee_debut + 1, 5, 31)
    df["date"] = pd.date_range(debut, fin, periods=len(df))
    df["competition"] = nom
    df["league_code"] = code
    df["neutral"] = False
    if verbose:
        attendu = n * (n - 1)
        print(f"  {code:<5} {nom:<32} {len(df):>5} matchs {saison} "
              f"({100*len(df)/max(attendu,1):.0f}% du tableau, {n} equipes)")
    return df[["date", "competition", "league_code", "home", "away",
               "home_score", "away_score", "neutral"]]


def load_wiki_league(code: str, depuis: int = 2023, jusqu: int = 2026,
                     verbose: bool = True) -> pd.DataFrame:
    """Plusieurs saisons d'un championnat Wikipedia. Les pages absentes sont ignorees."""
    frames = [load_wiki_season(code, a, verbose) for a in range(depuis, jusqu + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
