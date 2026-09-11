"""
NRL (rugby a XIII australien) : resultats et statistiques joueurs.

Source : depot public `uselessnrlstats`, qui publie des donnees nettoyees de
1908 a aujourd'hui -- resultats de match, feuilles de match joueur par joueur
avec essais, buts au pied et postes.

C'est, a ce jour, la seule discipline hors football pour laquelle on dispose
d'un historique libre ET de donnees joueurs. Le Top 14 et la Champions Cup
n'ont aucun equivalent accessible : leurs resultats se saisissent a la main
(voir `data/results_csv.py`).

DEUX PIEGES DANS CE JEU DE DONNEES
----------------------------------
1. Il contient des rencontres A VENIR, avec un score a NaN. Les charger sans
   filtrer fait croire au modele que des matchs se sont termines 0-0 -- ou
   plutot les fait disparaitre silencieusement au dropna, ce qui fausse la
   ponderation temporelle. On ne garde que les matchs effectivement joues.

2. Il couvre un siecle et six competitions differentes (NSWRFL, ARL, Super
   League...). Melanger 1912 et 2026 n'a aucun sens : le jeu, les regles de
   points et les equipes n'ont rien de commun. On filtre donc sur l'ere NRL
   moderne, et la decroissance temporelle fait le reste.
"""
from __future__ import annotations

import io

import pandas as pd

from .cache import get_cache

__all__ = ["load_results", "load_player_stats", "NRL_TEAMS", "resolve_team"]

BASE = "https://raw.githubusercontent.com/uselessnrlstats/uselessnrlstats/main/cleaned_data/nrl/"
_TTL = 60 * 60 * 24  # 24 h

# Postes NRL -> categories utilisees par PRIORS_RUGBY.
# Le rugby a XIII n'a pas les memes postes que le XV : on rapproche par role.
POSITION_MAP = {
    "FB": "arriere", "W": "ailier", "C": "centre", "FE": "ouverture",
    "HB": "demi_de_melee", "L": "troisieme_ligne", "SR": "troisieme_ligne",
    "P": "premiere_ligne", "H": "premiere_ligne", "B": "inconnu", "I": "inconnu",
}


def _csv(nom: str) -> pd.DataFrame:
    import requests

    url = BASE + nom

    def loader() -> str:
        r = requests.get(url, timeout=120, headers={"User-Agent": "sportvalue/1.0"})
        r.raise_for_status()
        return r.text

    return pd.read_csv(io.StringIO(get_cache().get_text(url, _TTL, loader)), low_memory=False)


def load_results(depuis: str = "2021-01-01", verbose: bool = True) -> pd.DataFrame:
    """
    Resultats NRL au format attendu par les modeles :
        date, home, away, home_score, away_score, neutral
    """
    df = _csv("match_data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out = pd.DataFrame({
        "date": df["date"],
        "home": df["home_team"].astype(str).str.strip(),
        "away": df["away_team"].astype(str).str.strip(),
        "home_score": pd.to_numeric(df["home_team_score"], errors="coerce"),
        "away_score": pd.to_numeric(df["away_team_score"], errors="coerce"),
        "competition": df["competition"].astype(str),
        "match_id": df["match_id"],
        "neutral": False,
    })
    avant = len(out)
    # Matchs A VENIR : presents dans le fichier, score a NaN. A ecarter.
    out = out.dropna(subset=["date", "home_score", "away_score"])
    joues = len(out)
    out = out[out["date"] >= pd.Timestamp(depuis)].sort_values("date").reset_index(drop=True)
    if verbose:
        print(f"   NRL : {len(out)} matchs joues depuis {depuis[:4]} "
              f"({out['date'].min():%d/%m/%Y} -> {out['date'].max():%d/%m/%Y})")
        print(f"   ({avant - joues} rencontre(s) a venir ecartee(s), score absent)")
        print(f"   {out['home'].nunique()} equipes")
    return out


def load_player_stats(results: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Statistiques joueurs agregees, au format attendu par ScorerModel :
        joueur, equipe, poste, minutes, buts_hors_penalty (= essais)

    Le fichier ne donne pas les minutes jouees. On les approxime par le nombre
    de feuilles de match x 80 : au rugby a XIII, un titulaire joue la quasi
    totalite du match et les remplacements tournent. L'approximation est
    grossiere mais sans biais systematique entre joueurs d'un meme effectif.
    """
    pm = _csv("player_match_data.csv")
    noms = _csv("player_data.csv")[["player_id", "full_name"]]

    if results is not None and "match_id" in results.columns:
        pm = pm[pm["match_id"].isin(set(results["match_id"]))]

    pm = pm.merge(noms, on="player_id", how="left")
    pm["tries"] = pd.to_numeric(pm["tries"], errors="coerce").fillna(0)
    pm["poste"] = pm["position"].map(POSITION_MAP).fillna("inconnu")

    agg = (
        pm.groupby(["full_name", "team"], as_index=False)
        .agg(essais=("tries", "sum"), matchs=("match_id", "nunique"),
             poste=("poste", lambda s: s.mode().iloc[0] if len(s.mode()) else "inconnu"))
    )
    agg = agg.rename(columns={"full_name": "joueur", "team": "equipe"})
    agg["minutes"] = agg["matchs"] * 80.0
    agg["buts_hors_penalty"] = agg["essais"]
    agg["tireur_penalty"] = 0
    agg = agg[agg["matchs"] >= 2].reset_index(drop=True)
    if verbose:
        print(f"   {len(agg)} joueurs, {agg['equipe'].nunique()} equipes, "
              f"{agg['essais'].sum():.0f} essais cumules")
    return agg


NRL_TEAMS = [
    "Brisbane Broncos", "Canberra Raiders", "Canterbury Bankstown Bulldogs",
    "Cronulla Sutherland Sharks", "Dolphins", "Gold Coast Titans",
    "Manly Warringah Sea Eagles", "Melbourne Storm", "Newcastle Knights",
    "North Queensland Cowboys", "Parramatta Eels", "Penrith Panthers",
    "South Sydney Rabbitohs", "St George Illawarra Dragons", "Sydney Roosters",
    "Warriors", "Wests Tigers",
]

# The Odds API abrege les noms d'equipes NRL.
ALIAS = {
    "brisbane broncos": "Brisbane Broncos",
    "canberra raiders": "Canberra Raiders",
    "canterbury bulldogs": "Canterbury Bankstown Bulldogs",
    "canterbury-bankstown bulldogs": "Canterbury Bankstown Bulldogs",
    "bulldogs": "Canterbury Bankstown Bulldogs",
    "cronulla sharks": "Cronulla Sutherland Sharks",
    "cronulla-sutherland sharks": "Cronulla Sutherland Sharks",
    "dolphins": "Dolphins",
    "redcliffe dolphins": "Dolphins",
    "gold coast titans": "Gold Coast Titans",
    "manly sea eagles": "Manly Warringah Sea Eagles",
    "manly-warringah sea eagles": "Manly Warringah Sea Eagles",
    "melbourne storm": "Melbourne Storm",
    "newcastle knights": "Newcastle Knights",
    "north queensland cowboys": "North Queensland Cowboys",
    "parramatta eels": "Parramatta Eels",
    "penrith panthers": "Penrith Panthers",
    "south sydney rabbitohs": "South Sydney Rabbitohs",
    "st george illawarra dragons": "St George Illawarra Dragons",
    "st. george illawarra dragons": "St George Illawarra Dragons",
    "sydney roosters": "Sydney Roosters",
    "new zealand warriors": "Warriors",
    "warriors": "Warriors",
    "wests tigers": "Wests Tigers",
}


def resolve_team(nom: str, connues) -> str | None:
    """Apparie un nom de calendrier a un nom d'historique NRL."""
    connues = set(connues)
    if nom in connues:
        return nom
    a = ALIAS.get(str(nom).strip().lower())
    if a and a in connues:
        return a
    cible = str(nom).lower().strip()
    cands = {c for c in connues if cible in c.lower() or c.lower() in cible}
    if len(cands) == 1:
        return cands.pop()
    # dernier recours : le mot le plus distinctif (le surnom du club)
    mots = [m for m in cible.split() if len(m) > 4]
    for m in reversed(mots):
        cands = {c for c in connues if m in c.lower()}
        if len(cands) == 1:
            return cands.pop()
    return None
