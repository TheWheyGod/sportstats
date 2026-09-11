"""
Mise a jour d'une journee sur les matchs en cours.

CE QUI EST POSSIBLE, ET CE QUI NE L'EST PAS
--------------------------------------------
La page publiee ne peut PAS se rafraichir seule : elle tourne sous une
politique de securite qui interdit tout appel reseau sortant, et aucun
connecteur de donnees sportives n'est disponible cote navigateur. Une page qui
s'actualiserait toute seule est donc hors d'atteinte.

Ce qui est possible : regenerer la page. On recupere les scores, on RECALCULE
les probabilites conditionnellement a ce qui s'est deja produit, on reecrit le
fichier. Repete toutes les N minutes, cela suit le match -- la fraicheur etant
celle de la derniere regeneration, affichee explicitement plutot que presentee
comme du temps reel.

LE RECALCUL CONDITIONNEL
------------------------
Afficher une probabilite d'avant-match sur un match en cours est trompeur : a
la 70e minute d'un 0-0, le "plus de 2.5 buts" n'a plus rien a voir avec sa
valeur au coup d'envoi. On conditionne donc :

    points restants ~ meme loi, d'intensite reduite au prorata du temps restant
    total final      = score actuel + points restants

L'hypothese est un rythme de marquage CONSTANT. Elle est fausse dans le detail
-- les fins de match sont plus ouvertes au rugby, les equipes menees poussent
au football -- mais elle est neutre, verifiable et bien meilleure que de ne pas
conditionner du tout.

LIMITE PRINCIPALE, a garder en tete : le temps ecoule est DEDUIT de l'heure de
coup d'envoi et de l'heure de mise a jour du flux, pas lu sur un chrono. Une
erreur de quelques minutes deplace sensiblement les probabilites en fin de
match. La sortie affiche donc toujours la minute supposee.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy.stats import poisson

from .core.scoredist import ScoreDistribution

__all__ = ["minutes_ecoulees", "recalcule_conditionnel", "DUREES"]

# Duree reglementaire par sport, en minutes.
DUREES = {"football": 90.0, "rugby": 80.0, "basket": 48.0, "tennis": None}


def minutes_ecoulees(coup_envoi, maj=None, duree: float = 90.0) -> float:
    """
    Minutes de jeu ecoulees, deduites des horodatages.

    On majore par la duree reglementaire : un flux qui continue de publier
    apres la fin du match donnerait sinon un temps restant negatif.
    L'estimation ignore la mi-temps et les arrets de jeu, ce qui la rend
    optimiste de quelques minutes en seconde periode.
    """
    if coup_envoi is None:
        return 0.0
    fin = maj or datetime.now(timezone.utc)
    if coup_envoi.tzinfo is None:
        coup_envoi = coup_envoi.replace(tzinfo=timezone.utc)
    if fin.tzinfo is None:
        fin = fin.replace(tzinfo=timezone.utc)
    ecoule = (fin - coup_envoi).total_seconds() / 60.0
    return float(np.clip(ecoule, 0.0, duree))


def recalcule_conditionnel(match: dict, score_dom: int, score_ext: int,
                           minutes: float, seed: int = 0) -> ScoreDistribution | None:
    """
    Redistribue les probabilites sachant le score actuel et le temps restant.

    Reconstruit une distribution complete a partir des taux d'avant-match mis
    a l'echelle du temps restant, puis ajoute le score deja acquis. Tous les
    marches derives (total, handicap, vainqueur) restent donc coherents entre
    eux, comme avant le match.
    """
    sport = match.get("sport", "football")
    duree = DUREES.get(sport)
    if duree is None:
        return None
    reste = max(0.0, duree - float(minutes)) / duree
    sd_pre = match.get("sd")
    rng = np.random.default_rng(seed)

    if sport == "rugby" and sd_pre is not None and hasattr(sd_pre, "event_rates"):
        n = 60000
        taux = sd_pre.event_rates

        def simule(r):
            essais = rng.poisson(max(r["lam_try"], 0) * reste, n)
            conv = rng.binomial(essais, r["kick"])
            pen = rng.poisson(max(r["lam_pen"], 0) * reste, n)
            drops = rng.poisson(max(r["lam_drop"], 0) * reste, n)
            return 5 * essais + 2 * conv + 3 * pen + 3 * drops

        ech = np.column_stack([
            score_dom + simule(taux["domicile"]),
            score_ext + simule(taux["exterieur"]),
        ]).astype(float)
        sd = ScoreDistribution.from_samples(ech)
        # On reconduit les taux, mis a l'echelle, pour les marches d'essais.
        t_dom = rng.poisson(max(taux["domicile"]["lam_try"], 0) * reste, n)
        t_ext = rng.poisson(max(taux["exterieur"]["lam_try"], 0) * reste, n)
        sd.tries = ScoreDistribution.from_samples(
            np.column_stack([t_dom, t_ext]).astype(float))
        sd.expected_tries = (float(t_dom.mean()), float(t_ext.mean()))
        return sd

    if sport == "football" and sd_pre is not None:
        lam = sd_pre.expected_home * reste
        mu = sd_pre.expected_away * reste
        maxg = 10
        k = np.arange(maxg + 1)
        mat = np.outer(poisson.pmf(k, max(lam, 1e-6)), poisson.pmf(k, max(mu, 1e-6)))
        mat /= mat.sum()
        # Le score acquis translate simplement la grille.
        hh, aa = np.meshgrid(k + score_dom, k + score_ext, indexing="ij")
        return ScoreDistribution(hh.ravel(), aa.ravel(), mat.ravel())

    if sport == "basket" and sd_pre is not None:
        n = 60000
        sigma = sd_pre.margin_std() / np.sqrt(2.0)
        tirages = rng.normal(
            [sd_pre.expected_home * reste, sd_pre.expected_away * reste],
            max(sigma * np.sqrt(max(reste, 1e-6)), 1e-6), size=(n, 2))
        ech = np.column_stack([score_dom + np.maximum(tirages[:, 0], 0),
                               score_ext + np.maximum(tirages[:, 1], 0)])
        return ScoreDistribution.from_samples(np.rint(ech))

    return None
