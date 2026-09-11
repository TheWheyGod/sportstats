"""
Historique de resultats au format CSV : le chemin universel.

Le football dispose d'une source gratuite complete (football-data.co.uk) et la
Premier League d'une source joueurs (API FPL). Le rugby, le basket europeen et
le tennis n'ont aucune source libre equivalente et stable : les depots publics
disparaissent ou changent de nom, les APIs passent en payant.

Plutot que de dependre d'une source fragile, l'outil accepte un CSV de
resultats. Les scores sont la donnee la plus facile a recuperer qui soit
(site de la competition, Wikipedia, feuille de match) et ils suffisent a
alimenter tous les modeles d'equipe :

    date,home,away,home_score,away_score[,neutral][,surface]

`surface` ne sert qu'au tennis (Hard / Clay / Grass), `neutral` indique un
terrain neutre (finale, tournoi). Toute autre colonne est ignoree.

Pour les marches joueurs, un second CSV est attendu :

    joueur,equipe,poste,minutes,buts_hors_penalty[,xg_hors_penalty][,tireur_penalty]

au rugby on remplace `buts_hors_penalty` par le nombre d'essais, et les postes
attendus sont ceux de PRIORS_RUGBY (ailier, arriere, centre...).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "load_results",
    "load_players",
    "write_results_template",
    "write_players_template",
    "RESULTS_TEMPLATE",
    "PLAYERS_TEMPLATE_RUGBY",
]

RESULTS_TEMPLATE = """date,home,away,home_score,away_score,neutral
2026-08-30,Toulouse,Bordeaux,27,19,false
2026-08-30,La Rochelle,Toulon,22,25,false
2026-09-06,Bordeaux,La Rochelle,31,17,false
2026-09-06,Toulon,Toulouse,15,28,false
"""

PLAYERS_TEMPLATE_RUGBY = """joueur,equipe,poste,minutes,essais,titulaire
Dupont,Toulouse,demi_de_melee,540,4,1
Ramos,Toulouse,arriere,520,2,1
Mallia,Toulouse,ailier,480,5,1
Danty,La Rochelle,centre,500,2,1
Alldritt,La Rochelle,troisieme_ligne,530,3,1
"""


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {
        "domicile": "home", "exterieur": "away",
        "score_domicile": "home_score", "score_exterieur": "away_score",
        "points_domicile": "home_score", "points_exterieur": "away_score",
        "equipe_domicile": "home", "equipe_exterieur": "away",
        "joueur_a": "home", "joueur_b": "away",
        "sets_a": "home_score", "sets_b": "away_score",
        "terrain_neutre": "neutral",
    }
    return df.rename(columns={k: v for k, v in alias.items() if k in df.columns})


def load_results(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Charge et valide un historique de resultats."""
    df = _norm(pd.read_csv(path))
    besoin = {"date", "home", "away", "home_score", "away_score"}
    manque = besoin - set(df.columns)
    if manque:
        raise ValueError(
            f"Colonnes manquantes : {sorted(manque)}. "
            f"Colonnes trouvees : {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=["date", "home", "away", "home_score", "away_score"])
    if "neutral" not in df.columns:
        df["neutral"] = False
    else:
        df["neutral"] = (
            df["neutral"].astype(str).str.lower().isin(["true", "1", "oui", "yes", "vrai"])
        )
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        rejets = avant - len(df)
        equipes = sorted(set(df["home"]) | set(df["away"]))
        print(f"   {len(df)} matchs, {len(equipes)} equipes, "
              f"{df['date'].min():%d/%m/%Y} -> {df['date'].max():%d/%m/%Y}"
              + (f"  ({rejets} ligne(s) ecartee(s))" if rejets else ""))
        par_equipe = pd.concat([df["home"], df["away"]]).value_counts()
        faibles = par_equipe[par_equipe < 5]
        if len(faibles):
            print(f"   [!] {len(faibles)} equipe(s) avec moins de 5 matchs : "
                  f"leurs notes seront tres incertaines "
                  f"({', '.join(faibles.index[:5])}{'...' if len(faibles) > 5 else ''})")
    return df


def load_players(path: str | Path, sport: str = "football", verbose: bool = True) -> pd.DataFrame:
    """
    Charge un CSV de statistiques joueurs et le normalise au format attendu
    par ScorerModel (colonne d'evenements renommee en `buts_hors_penalty`
    quelle que soit la discipline).
    """
    df = _norm(pd.read_csv(path))
    alias = {
        "essais": "buts_hors_penalty",
        "tries": "buts_hors_penalty",
        "buts": "buts_hors_penalty",
        "points": "buts_hors_penalty",
        "xg": "xg_hors_penalty",
        "penalty": "tireur_penalty",
        "tireur": "tireur_penalty",
        "player": "joueur",
        "team": "equipe",
        "position": "poste",
        "min": "minutes",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})

    besoin = {"joueur", "equipe", "minutes"}
    manque = besoin - set(df.columns)
    if manque:
        raise ValueError(f"Colonnes joueurs manquantes : {sorted(manque)}")
    if "buts_hors_penalty" not in df.columns:
        df["buts_hors_penalty"] = 0.0
        if verbose:
            print("   [!] aucune colonne d'evenements marques : "
                  "tous les joueurs retombent sur le prior de leur poste.")
    if "poste" not in df.columns:
        df["poste"] = "inconnu"
    if "tireur_penalty" not in df.columns:
        df["tireur_penalty"] = 0

    for c in ("minutes", "buts_hors_penalty", "tireur_penalty"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "xg_hors_penalty" in df.columns:
        df["xg_hors_penalty"] = pd.to_numeric(df["xg_hors_penalty"], errors="coerce")
    df["poste"] = df["poste"].astype(str).str.strip().str.lower()

    if verbose:
        print(f"   {len(df)} joueurs, {df['equipe'].nunique()} equipes, "
              f"{df['minutes'].sum():.0f} minutes cumulees")
    return df.reset_index(drop=True)


def write_results_template(path: str | Path) -> Path:
    p = Path(path)
    p.write_text(RESULTS_TEMPLATE, encoding="utf-8")
    return p


def write_players_template(path: str | Path) -> Path:
    p = Path(path)
    p.write_text(PLAYERS_TEMPLATE_RUGBY, encoding="utf-8")
    return p
