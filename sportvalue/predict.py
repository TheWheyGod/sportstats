"""
Prediction pure : toutes les probabilites d'un match, issues des DONNEES.

Aucune cote n'intervient dans ce module. Les probabilites sortent des modeles
d'equipe et de joueur, et sont donc comparables a n'importe quelle ligne de
bookmaker sans circularite.

Garantie de coherence : pour chaque sport, tous les marches derivent d'UN SEUL
objet de distribution. Le 1X2, le total de buts, le BTTS et les scores exacts
du football viennent de la meme matrice Dixon-Coles ; le total de points et le
total d'essais du rugby viennent des memes tirages Monte-Carlo. Il est donc
structurellement impossible d'obtenir un total de buts qui contredise le 1X2,
ce qui arrive des qu'on entraine un modele separe par marche.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core.scoredist import ScoreDistribution

__all__ = [
    "football_markets",
    "rugby_markets",
    "tennis_markets",
    "basket_markets",
    "FPL_TO_FOOTBALLDATA",
]

# FPL et football-data.co.uk ne nomment pas les equipes pareil. Sans cette
# table, un Tottenham cote FPL ne retrouve jamais son modele cote resultats.
FPL_TO_FOOTBALLDATA = {
    "Spurs": "Tottenham",
    "Man Utd": "Man United",
    "Man City": "Man City",
    "Nott'm Forest": "Nott'm Forest",
    "Newcastle": "Newcastle",
    "Wolves": "Wolves",
    "Brighton": "Brighton",
    "West Ham": "West Ham",
    "Sheffield Utd": "Sheffield United",
    "Leeds": "Leeds",
    "Luton": "Luton",
    "Nottingham Forest": "Nott'm Forest",
}


def _ou_block(sd: ScoreDistribution, lignes, label="total") -> list[dict]:
    rows = []
    for l in lignes:
        r = sd.over_under(l)
        rows.append(
            {
                label: f"{l:g}",
                "over": round(r["p_over"], 4),
                "under": round(r["p_under"], 4),
                "push": round(r["p_push"], 4),
            }
        )
    return rows


# --------------------------------------------------------------------------
# FOOTBALL
# --------------------------------------------------------------------------
def football_markets(sd: ScoreDistribution, scorers: pd.DataFrame | None = None,
                     absences: list | None = None, extra: dict | None = None,
                     assists: pd.DataFrame | None = None) -> dict:
    """
    1X2, doubles chances, buts, BTTS, scores exacts, totaux par equipe,
    et buteurs si un effectif est fourni.

    `extra` : marches secondaires deja calcules (mi-temps, premiere equipe a
    marquer, corners, cartons -- voir models.football_extra), simplement
    fusionnes. Ils sont produits en amont parce qu'ils demandent des modeles
    ajustes une fois par championnat, pas par match.
    """
    m = sd.market_1x2()
    out = {
        "buts_attendus": {
            "domicile": round(sd.expected_home, 3),
            "exterieur": round(sd.expected_away, 3),
            "total": round(sd.expected_total, 3),
        },
        "1x2": {k: round(v, 4) for k, v in m.items()},
        "double_chance": {k: round(v, 4) for k, v in sd.double_chance().items()},
        "draw_no_bet": {k: round(v, 4) for k, v in sd.draw_no_bet().items()},
        "total_buts": _ou_block(sd, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], "ligne"),
        "btts": {k: round(v, 4) for k, v in sd.btts().items()},
        "scores_exacts": [
            {"score": c["score"], "p": round(c["p"], 4)} for c in sd.correct_score(12)
        ],
        "total_domicile": [
            {"ligne": f"{l:g}", **{k: round(v, 4) for k, v in sd.team_total("home", l).items()
                                   if k.startswith("p_")}}
            for l in (0.5, 1.5, 2.5)
        ],
        "total_exterieur": [
            {"ligne": f"{l:g}", **{k: round(v, 4) for k, v in sd.team_total("away", l).items()
                                   if k.startswith("p_")}}
            for l in (0.5, 1.5, 2.5)
        ],
    }
    if scorers is not None and not scorers.empty:
        cols = ["joueur", "equipe_cote", "poste", "lambda_joueur",
                "p_marque", "p_2plus", "p_premier_buteur"]
        d = scorers[[c for c in cols if c in scorers.columns]].copy()
        for c in ("lambda_joueur", "p_marque", "p_2plus", "p_premier_buteur"):
            if c in d.columns:
                d[c] = d[c].round(4)
        out["buteurs"] = d.to_dict("records")
        out["aucun_buteur"] = round(float(scorers["p_aucun_buteur"].iloc[0]), 4)
    if assists is not None and not assists.empty:
        # meme modele que les buteurs, applique aux passes decisives :
        # p_marque y signifie "au moins une passe decisive"
        cols = ["joueur", "equipe_cote", "poste", "lambda_joueur", "p_marque"]
        d = assists[[c for c in cols if c in assists.columns]].copy()
        d = d.rename(columns={"p_marque": "p_passe"})
        for c in ("lambda_joueur", "p_passe"):
            d[c] = d[c].round(4)
        out["passeurs"] = d.to_dict("records")
    if absences:
        out["absences"] = absences
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------
# RUGBY
# --------------------------------------------------------------------------
def rugby_markets(sd: ScoreDistribution, try_scorers: pd.DataFrame | None = None) -> dict:
    """
    Vainqueur, points, ecart, essais. La distribution des essais provient des
    memes tirages que celle des points.
    """
    m = sd.market_1x2()
    out = {
        "points_attendus": {
            "domicile": round(sd.expected_home, 2),
            "exterieur": round(sd.expected_away, 2),
            "total": round(sd.expected_total, 2),
            "ecart": round(sd.expected_margin, 2),
        },
        "vainqueur": {k: round(v, 4) for k, v in m.items()},
        "total_points": _ou_block(sd, [39.5, 44.5, 49.5, 54.5, 59.5], "ligne"),
        "ecart_par_tranche": {
            "domicile 1-7": round(sd.margin_band(1, 7), 4),
            "domicile 8-12": round(sd.margin_band(8, 12), 4),
            "domicile 13-21": round(sd.margin_band(13, 21), 4),
            "domicile 22+": round(sd.margin_band(22, 999), 4),
            "nul": round(sd.margin_band(0, 0), 4),
            "exterieur 1-7": round(sd.margin_band(-7, -1), 4),
            "exterieur 8-12": round(sd.margin_band(-12, -8), 4),
            "exterieur 13-21": round(sd.margin_band(-21, -13), 4),
            "exterieur 22+": round(sd.margin_band(-999, -22), 4),
        },
    }
    if hasattr(sd, "weather_notes"):
        out["meteo"] = sd.weather_notes
    if hasattr(sd, "tries"):
        t = sd.tries
        out["essais_attendus"] = {
            "domicile": round(t.expected_home, 2),
            "exterieur": round(t.expected_away, 2),
            "total": round(t.expected_total, 2),
        }
        out["total_essais"] = _ou_block(t, [3.5, 4.5, 5.5, 6.5, 7.5], "ligne")
        out["essais_domicile"] = [
            {"ligne": f"{l:g}", **{k: round(v, 4) for k, v in t.team_total("home", l).items()
                                   if k.startswith("p_")}}
            for l in (1.5, 2.5, 3.5)
        ]
        out["essais_exterieur"] = [
            {"ligne": f"{l:g}", **{k: round(v, 4) for k, v in t.team_total("away", l).items()
                                   if k.startswith("p_")}}
            for l in (1.5, 2.5, 3.5)
        ]
        out["premier_essai_marque"] = round(1.0 - float(np.sum(t.w[(t.h == 0) & (t.a == 0)])), 4)
    if try_scorers is not None and not try_scorers.empty:
        cols = ["joueur", "equipe_cote", "poste", "lambda_joueur", "p_marque", "p_2plus",
                "p_premier_buteur"]
        d = try_scorers[[c for c in cols if c in try_scorers.columns]].copy()
        for c in ("lambda_joueur", "p_marque", "p_2plus", "p_premier_buteur"):
            if c in d.columns:
                d[c] = d[c].round(4)
        out["marqueurs_essais"] = d.to_dict("records")
    return out


# --------------------------------------------------------------------------
# TENNIS
# --------------------------------------------------------------------------
def tennis_markets(pred, joueur_a: str = "A", joueur_b: str = "B") -> dict:
    """Vainqueur, score en sets, nombre de sets, nombre de jeux, handicaps."""
    sets_tri = sorted(pred.set_scores.items(), key=lambda kv: -kv[1])
    out = {
        "vainqueur": {joueur_a: round(pred.p_a, 4), joueur_b: round(1 - pred.p_a, 4)},
        "points_gagnes_au_service": {
            joueur_a: round(pred.serve_a, 4),
            joueur_b: round(pred.serve_b, 4),
        },
        "conservation_du_service": {
            joueur_a: round(pred.hold_a, 4),
            joueur_b: round(pred.hold_b, 4),
        },
        "score_en_sets": [
            {"score": f"{a}-{b}", "p": round(p, 4)} for (a, b), p in sets_tri
        ],
        "jeux_attendus": round(pred.expected_games(), 2),
        "total_jeux": [],
        "handicap_jeux": [],
    }
    centre = round(pred.expected_games() * 2) / 2
    for l in [centre - 3.5, centre - 1.5, centre - 0.5, centre + 1.5, centre + 3.5]:
        r = pred.total_games(l)
        out["total_jeux"].append(
            {"ligne": f"{l:g}", "over": round(r["p_over"], 4), "under": round(r["p_under"], 4)}
        )
    for l in (-5.5, -3.5, -1.5, 1.5, 3.5, 5.5):
        r = pred.games_handicap(l)
        out["handicap_jeux"].append(
            {
                "ligne": f"{l:+g}",
                f"p_{joueur_a}": round(r["home_ev_neutral_prob"], 4),
                f"p_{joueur_b}": round(r["away_ev_neutral_prob"], 4),
            }
        )
    nb = pred.best_of
    seuil = 2.5 if nb == 3 else 3.5
    ts = pred.total_sets(seuil)
    out["nombre_de_sets"] = {
        f"over {seuil:g}": round(ts["p_over"], 4),
        f"under {seuil:g}": round(ts["p_under"], 4),
    }
    return out


# --------------------------------------------------------------------------
# BASKET
# --------------------------------------------------------------------------
def basket_markets(sd: ScoreDistribution, scorers: pd.DataFrame | None = None) -> dict:
    """Vainqueur, ecart, total de points, totaux par equipe."""
    ml = sd.market_moneyline()
    total_attendu = sd.expected_total
    centre = round(total_attendu / 5) * 5
    out = {
        "points_attendus": {
            "domicile": round(sd.expected_home, 1),
            "exterieur": round(sd.expected_away, 1),
            "total": round(total_attendu, 1),
            "ecart": round(sd.expected_margin, 1),
            "ecart_type_ecart": round(sd.margin_std(), 1),
        },
        "vainqueur": {k: round(v, 4) for k, v in ml.items()},
        "total_points": _ou_block(sd, [centre - 10.5, centre - 5.5, centre + 0.5,
                                       centre + 5.5, centre + 10.5], "ligne"),
        "handicap": [],
    }
    # Lignes centrees sur l'ecart attendu. La ligne "equitable" du point de vue
    # du domicile est -ecart_attendu : c'est autour d'elle qu'on echantillonne.
    # Attention au signe : `ligne_domicile` est le handicap PORTE par le
    # domicile (negatif quand il est favori), et asian_handicap() attend ce
    # meme handicap. Melanger les deux conventions produit des lignes du type
    # "+26" avec une probabilite de 1, absurdes mais silencieuses.
    base = round(sd.expected_margin * 2) / 2
    for ecart in (base - 5.5, base - 2.5, base, base + 2.5, base + 5.5):
        ligne_dom = -ecart
        r = sd.asian_handicap(ligne_dom)
        out["handicap"].append(
            {
                "ligne_domicile": f"{ligne_dom:+g}",
                "p_domicile": round(r["home_ev_neutral_prob"], 4),
                "p_exterieur": round(r["away_ev_neutral_prob"], 4),
            }
        )
    tt_h = [{"ligne": f"{l:g}", **{k: round(v, 4) for k, v in sd.team_total("home", l).items()
                                   if k.startswith("p_")}}
            for l in (round(sd.expected_home) - 5.5, round(sd.expected_home) + 0.5,
                      round(sd.expected_home) + 5.5)]
    tt_a = [{"ligne": f"{l:g}", **{k: round(v, 4) for k, v in sd.team_total("away", l).items()
                                   if k.startswith("p_")}}
            for l in (round(sd.expected_away) - 5.5, round(sd.expected_away) + 0.5,
                      round(sd.expected_away) + 5.5)]
    out["total_domicile"] = tt_h
    out["total_exterieur"] = tt_a
    if scorers is not None and not scorers.empty:
        out["scoreurs"] = scorers.to_dict("records")
    return out
