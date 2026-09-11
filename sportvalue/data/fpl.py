"""
Statistiques joueurs Premier League via l'API Fantasy Premier League.

Gratuite, sans cle, sans compte, et mise a jour apres chaque journee. Elle
fournit exactement ce dont le modele de buteurs a besoin :

    minutes, buts, passes decisives, xG, xA, poste, ordre des tireurs de
    penalty, titularisations, equipe.

Le champ `penalties_order` vaut 1 pour le tireur designe : c'est ce qui permet
de sortir les penaltys du taux de jeu courant et de les attribuer au bon
joueur. Sans cette information, un tireur attitre voit son taux de marquage
surestime et tous les autres joueurs sous-estimes.

Le xG est prefere aux buts pour estimer le taux : en debut de saison, avec
200 a 300 minutes par joueur, les buts sont trop bruites pour distinguer un
attaquant en forme d'un attaquant chanceux.

Limite : Premier League uniquement. Pour les autres championnats, passer par
un CSV (voir `data/players_csv.py`).
"""
from __future__ import annotations

import pandas as pd

from .cache import get_cache

__all__ = ["load_players", "team_squad", "FPL_POSITIONS", "list_teams"]

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_TTL = 60 * 60 * 6  # 6 h

# element_type -> poste normalise pour PRIORS_FOOT
FPL_POSITIONS = {1: "gardien", 2: "defenseur", 3: "milieu", 4: "attaquant"}


def _fetch() -> dict:
    import requests

    def loader():
        r = requests.get(URL, timeout=30, headers={"User-Agent": "sportvalue/1.0"})
        r.raise_for_status()
        return r.json()

    return get_cache().get_json(URL, _TTL, loader)


def load_players() -> pd.DataFrame:
    """
    Retourne un DataFrame au format attendu par ScorerModel :
        joueur, equipe, poste, minutes, buts_hors_penalty, xg_hors_penalty,
        tireur_penalty, matchs_equipe, titularisations
    """
    data = _fetch()
    teams = {t["id"]: t["name"] for t in data["teams"]}
    # Nombre de journees jouees : sert a estimer les minutes attendues.
    events = data.get("events", [])
    journees_jouees = sum(1 for e in events if e.get("finished"))

    rows = []
    for e in data["elements"]:
        minutes = float(e.get("minutes", 0) or 0)
        buts = float(e.get("goals_scored", 0) or 0)
        pen_marques = float(e.get("penalties_scored", 0) or 0)
        # penalties_scored n'existe pas toujours : on retombe sur 0, ce qui
        # laisse les penaltys dans le taux de jeu courant. Acceptable, mais
        # cela surestime legerement les tireurs attitres.
        xg = e.get("expected_goals")
        try:
            xg = float(xg) if xg is not None else None
        except (TypeError, ValueError):
            xg = None

        pen_order = e.get("penalties_order")
        rows.append(
            {
                "joueur": e.get("web_name", ""),
                "nom_complet": f"{e.get('first_name','')} {e.get('second_name','')}".strip(),
                "equipe": teams.get(e["team"], "?"),
                "poste": FPL_POSITIONS.get(e.get("element_type"), "inconnu"),
                "minutes": minutes,
                "buts": buts,
                "buts_hors_penalty": max(buts - pen_marques, 0.0),
                "xg_hors_penalty": xg,   # xG FPL exclut deja les penaltys non tires
                "passes_d": float(e.get("assists", 0) or 0),
                "tireur_penalty": 1 if pen_order == 1 else 0,
                "titularisations": float(e.get("starts", 0) or 0),
                "matchs_equipe": journees_jouees or None,
                "dispo": e.get("status", "a") == "a",
                "chance_de_jouer": e.get("chance_of_playing_next_round"),
            }
        )
    df = pd.DataFrame(rows)
    return df[df["minutes"] > 0].reset_index(drop=True)


def list_teams() -> list[str]:
    data = _fetch()
    return sorted(t["name"] for t in data["teams"])


def team_squad(
    players: pd.DataFrame,
    equipe: str,
    min_minutes: float = 45.0,
    only_available: bool = True,
) -> pd.DataFrame:
    """
    Effectif d'une equipe, filtre sur un minimum de temps de jeu.

    `only_available` ecarte les joueurs signales blesses ou suspendus par FPL
    (statut different de 'a'). C'est la prise en compte des absences la plus
    simple qui soit fiable : FPL met ce champ a jour avant chaque journee.
    """
    d = players[players["equipe"].str.lower() == equipe.lower()]
    if only_available:
        d = d[d["dispo"]]
    d = d[d["minutes"] >= min_minutes]
    return d.reset_index(drop=True)


def estimate_minutes(squad: pd.DataFrame, full: float = 90.0) -> dict:
    """
    Minutes attendues au prochain match.

    Estimation simple et robuste : minutes par match jouees jusqu'ici, plafonnee
    a 90. Elle capture naturellement le statut titulaire / remplacant sans
    dependre d'une composition annoncee, souvent indisponible avant l'heure du
    coup d'envoi.

    `chance_de_jouer` (0-100, publie par FPL en cas de doute medical) vient
    ponderer le resultat quand il est renseigne.
    """
    out = {}
    for r in squad.itertuples(index=False):
        matchs = getattr(r, "matchs_equipe", None) or 0
        base = (r.minutes / matchs) if matchs else r.minutes
        base = min(float(base), full)
        chance = getattr(r, "chance_de_jouer", None)
        if chance is not None and not pd.isna(chance):
            base *= float(chance) / 100.0
        out[r.joueur] = base
    return out
