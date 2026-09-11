"""
Marches joueurs : buteurs (foot), marqueurs d'essais (rugby), scoreurs (basket).

PRINCIPE
--------
Le modele d'equipe donne l'esperance de buts de l'equipe (lambda_equipe). Il
reste a la REPARTIR entre les joueurs, proportionnellement a leur taux de
marquage multiplie par leurs minutes attendues :

    lambda_joueur = lambda_equipe x (taux_j x minutes_j) / somme(taux_q x minutes_q)

Cette normalisation garantit que la somme des esperances individuelles egale
l'esperance de l'equipe : impossible d'avoir un 1X2 qui dit 1.6 but et des
probabilites de buteurs qui en impliquent 2.4.

DEUX PIEGES, ET COMMENT ILS SONT TRAITES
-----------------------------------------
1. PETITS ECHANTILLONS. Un joueur avec 1 but en 90 minutes n'a pas un taux de
   1 but par match. On applique un retrecissement bayesien vers un prior de
   poste :

       taux = (buts + k x prior) / (minutes + k)

   avec k ~ 600 minutes. En debut de saison, ou tout le monde a moins de
   300 minutes, le prior domine -- ce qui est le comportement voulu.

2. LES PENALTYS FAUSSENT TOUT. Un tireur de penalty designe accumule des buts
   qui ne disent rien de son jeu. On les traite separement : les buts hors
   penalty alimentent le taux de jeu courant, et l'esperance de penalty est
   attribuee au seul tireur designe.

Quand le xG est disponible, il remplace les buts pour estimer le taux : sur
quelques centaines de minutes, le xG est nettement moins bruite que les buts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ScorerModel", "PRIORS_FOOT", "PRIORS_RUGBY", "PRIORS_BASKET"]

# Priors de taux de marquage par 90 minutes, par poste.
PRIORS_FOOT = {
    "attaquant": 0.36,
    "milieu": 0.13,
    "defenseur": 0.045,
    "gardien": 0.001,
    "inconnu": 0.12,
}

# Essais par 80 minutes, par poste (rugby a XV).
PRIORS_RUGBY = {
    "ailier": 0.42,
    "arriere": 0.28,
    "centre": 0.24,
    "troisieme_ligne": 0.20,
    "demi_de_melee": 0.16,
    "ouverture": 0.12,
    "deuxieme_ligne": 0.10,
    "premiere_ligne": 0.09,
    "inconnu": 0.18,
}

# Points par 36 minutes (basket) -- utilise comme taux, pas comme comptage.
PRIORS_BASKET = {"meneur": 14.0, "arriere": 13.0, "ailier": 12.0,
                 "ailier_fort": 11.0, "pivot": 11.0, "inconnu": 12.0}


class ScorerModel:
    """
    Parameters
    ----------
    priors : dict poste -> taux par match complet.
    k_shrink : force du retrecissement, en minutes. 600 convient au football
        (environ 7 matchs complets). Monter a 900 en debut de saison.
    full_match_minutes : 90 au foot, 80 au rugby, 48 en NBA.
    own_goal_share : part des buts d'equipe non attribuables a un joueur de
        cette equipe (csc adverse). ~2.5% au football, 0 ailleurs.
    penalty_rate : nombre moyen de penaltys obtenus par equipe et par match.
    penalty_conv : taux de reussite des penaltys.
    """

    def __init__(
        self,
        priors: dict | None = None,
        k_shrink: float = 600.0,
        full_match_minutes: float = 90.0,
        own_goal_share: float = 0.025,
        penalty_rate: float = 0.22,
        penalty_conv: float = 0.78,
    ):
        self.priors = priors or PRIORS_FOOT
        self.k_shrink = k_shrink
        self.full = full_match_minutes
        self.own_goal_share = own_goal_share
        self.penalty_rate = penalty_rate
        self.penalty_conv = penalty_conv

    # ------------------------------------------------------------------
    def _rate_per_match(self, row) -> float:
        """
        Taux de marquage par match complet, retreci vers le prior de poste.

        On prefere le xG aux buts quand il est disponible : sur de petits
        volumes de minutes, le xG a une variance bien plus faible et donne
        donc une estimation plus stable du vrai niveau du joueur.
        """
        poste = str(row.get("poste", "inconnu")).lower()
        prior = self.priors.get(poste, self.priors.get("inconnu", 0.12))

        minutes = float(row.get("minutes", 0.0) or 0.0)
        signal = row.get("xg_hors_penalty")
        if signal is None or (isinstance(signal, float) and np.isnan(signal)):
            signal = row.get("buts_hors_penalty")
        if signal is None or (isinstance(signal, float) and np.isnan(signal)):
            signal = 0.0
        signal = float(signal)

        # (evenements + k x prior_par_minute x k) / (minutes + k), exprime par match
        prior_par_minute = prior / self.full
        taux_min = (signal + self.k_shrink * prior_par_minute) / (minutes + self.k_shrink)
        return float(taux_min * self.full)

    # ------------------------------------------------------------------
    def predict_team(
        self,
        squad: pd.DataFrame,
        team_lambda: float,
        expected_minutes: dict | None = None,
    ) -> pd.DataFrame:
        """
        squad : colonnes attendues
            joueur, poste, minutes, buts_hors_penalty [, xg_hors_penalty]
            [, tireur_penalty (bool ou rang)] [, minutes_attendues]
        team_lambda : esperance de buts de l'equipe, issue du modele d'equipe.

        Retourne un DataFrame trie par esperance decroissante, avec
        lambda_joueur, p_marque, p_2plus, p_triple.
        """
        if squad.empty:
            return pd.DataFrame()

        d = squad.copy()
        d["taux"] = d.apply(self._rate_per_match, axis=1)

        # minutes attendues : fournies, sinon deduites du temps de jeu passe
        if expected_minutes:
            d["min_attendues"] = d["joueur"].map(expected_minutes).fillna(0.0)
        elif "minutes_attendues" in d.columns:
            d["min_attendues"] = pd.to_numeric(d["minutes_attendues"], errors="coerce").fillna(0.0)
        elif "matchs_equipe" in d.columns:
            # NB : passer par d.columns et non d.get(). Sur une colonne absente,
            # d.get(col, np.nan) renvoie un SCALAIRE, pas une Series, et tout
            # appel vectorise en aval echoue.
            matchs = pd.to_numeric(d["matchs_equipe"], errors="coerce")
            if matchs.notna().any():
                d["min_attendues"] = (
                    d["minutes"] / matchs.fillna(1).clip(lower=1)
                ).clip(0, self.full)
            else:
                mx = max(float(d["minutes"].max()), 1.0)
                d["min_attendues"] = (d["minutes"] / mx * self.full).clip(0, self.full)
        else:
            # a defaut, on suppose que le temps de jeu passe se reconduit :
            # le joueur le plus utilise est pris comme reference d'un match plein
            mx = max(float(d["minutes"].max()), 1.0)
            d["min_attendues"] = (d["minutes"] / mx * self.full).clip(0, self.full)

        # poids = taux x minutes attendues
        d["poids"] = d["taux"] * (d["min_attendues"] / self.full)
        total = float(d["poids"].sum())
        if total <= 0:
            d["lambda_joueur"] = 0.0
            return d

        # --- penaltys : sortis du pot commun et attribues au tireur designe
        lam_pen_equipe = 0.0
        tireur_idx = None
        if "tireur_penalty" in d.columns:
            marks = pd.to_numeric(d["tireur_penalty"], errors="coerce")
            candidats = d[(marks == 1) & (d["min_attendues"] > 0)]
            if not candidats.empty:
                tireur_idx = candidats.index[0]
                lam_pen_equipe = self.penalty_rate * self.penalty_conv

        lam_jeu = max(team_lambda * (1.0 - self.own_goal_share) - lam_pen_equipe, 0.0)
        d["lambda_joueur"] = lam_jeu * d["poids"] / total
        if tireur_idx is not None:
            d.loc[tireur_idx, "lambda_joueur"] += lam_pen_equipe

        lam = d["lambda_joueur"].to_numpy(dtype=float)
        d["p_marque"] = 1.0 - np.exp(-lam)
        d["p_2plus"] = 1.0 - np.exp(-lam) * (1.0 + lam)
        d["p_triple"] = 1.0 - np.exp(-lam) * (1.0 + lam + lam**2 / 2.0)
        return d.sort_values("lambda_joueur", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def predict_match(
        self,
        squad_home: pd.DataFrame,
        squad_away: pd.DataFrame,
        lambda_home: float,
        lambda_away: float,
        minutes_home: dict | None = None,
        minutes_away: dict | None = None,
    ) -> pd.DataFrame:
        """
        Combine les deux equipes et ajoute la probabilite de MARQUER LE PREMIER.

        Sous hypothese de processus de Poisson independants, sachant qu'un but
        survient, il appartient au joueur p avec probabilite lambda_p / Lambda.
        D'ou :

            P(p marque le premier) = (lambda_p / Lambda) x (1 - exp(-Lambda))

        Le complement, exp(-Lambda), est la probabilite qu'aucun but ne soit
        marque -- c'est exactement la cote "aucun buteur" proposee par les
        books, ce qui donne un point de controle direct du modele.
        """
        h = self.predict_team(squad_home, lambda_home, minutes_home)
        a = self.predict_team(squad_away, lambda_away, minutes_away)
        if h.empty and a.empty:
            return pd.DataFrame()
        h["equipe_cote"] = "domicile"
        a["equipe_cote"] = "exterieur"
        both = pd.concat([h, a], ignore_index=True)

        Lambda = float(lambda_home + lambda_away)
        p_au_moins_un = 1.0 - np.exp(-Lambda)
        both["p_premier_buteur"] = (
            both["lambda_joueur"] / max(Lambda, 1e-9) * p_au_moins_un
        )
        both["p_aucun_buteur"] = float(np.exp(-Lambda))
        return both.sort_values("lambda_joueur", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def coherence_check(self, pred: pd.DataFrame, lambda_home: float, lambda_away: float) -> dict:
        """
        Verifie que la repartition conserve l'esperance d'equipe.

        La somme des lambdas joueurs ne doit PAS egaler lambda_equipe, mais
        lambda_equipe x (1 - part_csc) : les buts contre son camp adverses
        comptent pour l'equipe sans etre attribuables a l'un de ses joueurs.
        La cible est donc explicitement corrigee de cette part.

        Si l'ecart depasse la tolerance, les probabilites de buteurs
        contredisent le marche des buts. Controle a relancer apres tout
        changement de composition.
        """
        if pred.empty:
            return {"ok": False, "motif": "aucune prediction"}
        sh = float(pred.loc[pred["equipe_cote"] == "domicile", "lambda_joueur"].sum())
        sa = float(pred.loc[pred["equipe_cote"] == "exterieur", "lambda_joueur"].sum())
        cible_h = float(lambda_home) * (1.0 - self.own_goal_share)
        cible_a = float(lambda_away) * (1.0 - self.own_goal_share)
        tol = 0.03
        return {
            "somme_lambda_domicile": round(sh, 4),
            "cible_domicile": round(cible_h, 4),
            "somme_lambda_exterieur": round(sa, 4),
            "cible_exterieur": round(cible_a, 4),
            "ecart_domicile": round(sh - cible_h, 4),
            "ecart_exterieur": round(sa - cible_a, 4),
            "somme_p_premier_buteur": round(float(pred["p_premier_buteur"].sum()), 4),
            "p_aucun_buteur": round(float(pred["p_aucun_buteur"].iloc[0]), 4),
            "ok": abs(sh - cible_h) <= tol * max(cible_h, 1e-9)
                  and abs(sa - cible_a) <= tol * max(cible_a, 1e-9),
        }
