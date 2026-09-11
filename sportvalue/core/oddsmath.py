"""
Conversion de cotes et suppression de la marge bookmaker (de-vigging).

Le de-vigging est la brique critique de la detection de mauvais prix : une
normalisation naive (proportionnelle) surestime systematiquement la probabilite
des outsiders, parce que la marge n'est pas repartie uniformement sur les
issues (favourite-longshot bias). Sur un favori a 1.30, la proportionnelle
SOUS-estime sa vraie probabilite -> on rate des value bets a haute probabilite.
C'est exactement le cas d'usage vise ici, donc la methode par defaut est Shin.

Methodes implementees :
  - multiplicative : p_i = q_i / sum(q). Rapide, biaisee.
  - power          : p_i = q_i^k, k tel que sum = 1.
  - shin           : modele d'insiders de Shin (1993). Defaut.
  - odds_ratio     : methode odds-ratio (shift constant en log-odds).
  - additive       : retire une marge egale a chaque issue.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "decimal_to_prob",
    "prob_to_decimal",
    "american_to_decimal",
    "decimal_to_american",
    "fractional_to_decimal",
    "overround",
    "margin_pct",
    "devig",
    "devig_all_methods",
    "is_sane_market",
    "devig_two_way",
    "fair_odds",
    "best_odds_across_books",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------
def decimal_to_prob(odds):
    """Probabilite implicite BRUTE (marge incluse) d'une cote decimale."""
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError("Une cote decimale doit etre > 1.0")
    out = 1.0 / o
    return float(out) if out.ndim == 0 else out


def prob_to_decimal(p):
    arr = np.asarray(p, dtype=float)
    if np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError("Probabilite hors ]0,1[")
    out = 1.0 / arr
    return float(out) if out.ndim == 0 else out


def american_to_decimal(american: float) -> float:
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def decimal_to_american(dec: float) -> float:
    d = float(dec)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def fractional_to_decimal(num: float, den: float) -> float:
    return 1.0 + num / den


# --------------------------------------------------------------------------
# Marge
# --------------------------------------------------------------------------
def overround(odds: Sequence[float]) -> float:
    """Somme des probabilites implicites brutes. 1.05 => 5% d'overround."""
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)))


def margin_pct(odds: Sequence[float]) -> float:
    """Marge reelle du book en % du volume mise : 1 - 1/overround."""
    return float(1.0 - 1.0 / overround(odds))


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------
def _multiplicative(q: np.ndarray) -> np.ndarray:
    return q / q.sum()


def _power(q: np.ndarray) -> np.ndarray:
    """Trouve k tel que sum(q_i^k) = 1 (k > 1 quand il y a de la marge)."""
    if abs(q.sum() - 1.0) < 1e-9:
        return q.copy()

    def f(k):
        return float(np.sum(np.power(q, k)) - 1.0)

    lo, hi = 0.2, 1.0
    while f(hi) > 0 and hi < 50.0:
        hi *= 2.0
    while f(lo) < 0 and lo > 1e-4:
        lo /= 2.0
    try:
        k = brentq(f, lo, hi, xtol=1e-12, maxiter=200)
    except ValueError:
        return _multiplicative(q)
    p = np.power(q, k)
    return p / p.sum()


def _shin(q: np.ndarray) -> np.ndarray:
    """
    Modele de Shin : proportion z de parieurs informes.
        p_i = ( sqrt(z^2 + 4(1-z) q_i^2 / Q) - z ) / (2(1-z)),  Q = sum(q)
    On resout z dans [0,1) tel que sum(p_i) = 1.
    """
    Q = q.sum()
    if abs(Q - 1.0) < 1e-9:
        return q.copy()

    def p_of_z(z):
        # NB : pas de cas special en z=0. La limite de la formule y vaut
        # q_i / sqrt(Q), dont la somme est sqrt(Q) > 1 : c'est ce qui donne
        # a f(0) le signe positif necessaire pour encadrer la racine.
        # Renvoyer q/Q ici ferait de z=0 une racine triviale et Shin
        # degenererait silencieusement en methode proportionnelle.
        disc = z * z + 4.0 * (1.0 - z) * (q * q) / Q
        return (np.sqrt(np.maximum(disc, 0.0)) - z) / (2.0 * (1.0 - z))

    def f(z):
        return float(p_of_z(z).sum() - 1.0)

    lo, hi = 0.0, 1.0 - 1e-9
    if f(lo) * f(hi) > 0:
        return _power(q)
    z = brentq(f, lo, hi, xtol=1e-12, maxiter=200)
    p = np.maximum(p_of_z(z), _EPS)
    return p / p.sum()


def _odds_ratio(q: np.ndarray) -> np.ndarray:
    """Il existe c tel que p_i = q_i / (c + q_i - c*q_i) et sum(p_i) = 1."""
    if abs(q.sum() - 1.0) < 1e-9:
        return q.copy()

    def p_of_c(c):
        return q / (c + q - c * q)

    def f(c):
        return float(p_of_c(c).sum() - 1.0)

    lo, hi = 1e-6, 1.0
    while f(hi) > 0 and hi < 1e6:
        hi *= 2.0
    if f(lo) < 0:
        return _multiplicative(q)
    try:
        c = brentq(f, lo, hi, xtol=1e-12, maxiter=200)
    except ValueError:
        return _multiplicative(q)
    p = p_of_c(c)
    return p / p.sum()


def _additive(q: np.ndarray) -> np.ndarray:
    p = np.maximum(q - (q.sum() - 1.0) / len(q), _EPS)
    return p / p.sum()


_METHODS = {
    "multiplicative": _multiplicative,
    "power": _power,
    "shin": _shin,
    "odds_ratio": _odds_ratio,
    "additive": _additive,
}


def is_sane_market(
    odds: Sequence[float],
    min_overround: float = 0.97,
    max_overround: float = 1.30,
) -> tuple[bool, str]:
    """
    Le marche est-il exploitable ? A APPELER AVANT TOUT DE-VIG.

    Motivation, tiree d'un cas reel rencontre en production : sur Ajax-Willem II,
    l'exchange Betfair renvoyait [1.06, 1.06, 1.01] faute de liquidite, soit un
    overround de 2.88. `devig` normalise docilement ces cotes en
    [0.32, 0.32, 0.35] et en conclut que Willem II a 32% de chances de gagner a
    Amsterdam. L'outil annoncait alors "+271% d'edge" sur un enorme outsider.

    Le de-vig ne PEUT PAS detecter ce cas : il normalise ce qu'on lui donne, et
    toute distribution normalisee a l'air legitime une fois normalisee. La
    verification doit donc porter sur l'overround BRUT, en amont.

    Bornes par defaut :
      < 0.97  : arbitrage apparent, presque toujours une cote perimee ou une
                erreur d'appariement plutot qu'une vraie opportunite ;
      > 1.30  : soit un marche de niche tres surtaxe, soit -- bien plus
                souvent -- un marche suspendu, illiquide ou mal apparie.
    """
    o = np.asarray(odds, dtype=float)
    if o.size < 2:
        return False, "moins de deux issues"
    if not np.all(np.isfinite(o)):
        return False, "cote non numerique"
    if np.any(o <= 1.0):
        return False, "cote <= 1.00"
    orr = float(np.sum(1.0 / o))
    if orr < min_overround:
        return False, f"overround {orr:.3f} anormalement bas (arbitrage suspect)"
    if orr > max_overround:
        return False, f"overround {orr:.3f} anormalement haut (marche illiquide ou suspendu)"
    return True, "ok"


def devig(odds: Sequence[float], method: str = "shin", conservative: bool = False) -> np.ndarray:
    """
    Probabilites "fair" (marge retiree) d'un marche complet.

    odds : cotes decimales de TOUTES les issues mutuellement exclusives
           (ex. [2.10, 3.40, 3.60] pour un 1X2).
    conservative : prend le MAX des probas fair estimees par shin/power/
           odds_ratio puis renormalise. Gonfle la proba du marche, donc reduit
           l'edge apparent. Sert a limiter les faux positifs de value.
    """
    q = np.asarray(odds, dtype=float)
    if np.any(q <= 1.0):
        raise ValueError(f"Cotes decimales invalides : {list(odds)}")
    q = 1.0 / q

    if conservative:
        stack = np.vstack([_METHODS[m](q) for m in ("shin", "power", "odds_ratio")])
        p = stack.max(axis=0)
        return p / p.sum()

    if method not in _METHODS:
        raise ValueError(f"Methode inconnue : {method}. Choix : {list(_METHODS)}")
    return _METHODS[method](q)


def devig_two_way(odds_a: float, odds_b: float, method: str = "shin") -> tuple[float, float]:
    """Raccourci pour les marches a deux issues (O/U, handicap, moneyline)."""
    p = devig([odds_a, odds_b], method=method)
    return float(p[0]), float(p[1])


def devig_all_methods(odds: Sequence[float]) -> dict:
    """Toutes les methodes, pour tester la robustesse d'un edge detecte."""
    q = 1.0 / np.asarray(odds, dtype=float)
    return {name: fn(q) for name, fn in _METHODS.items()}


def fair_odds(odds: Sequence[float], method: str = "shin") -> np.ndarray:
    """Cotes fair (sans marge) correspondant aux cotes proposees."""
    return 1.0 / devig(odds, method=method)


# --------------------------------------------------------------------------
# Agregation multi-books
# --------------------------------------------------------------------------
def best_odds_across_books(quotes: dict) -> tuple:
    """
    quotes : {book: [cote_issue1, cote_issue2, ...]} (meme ordre partout).
    Retourne (meilleures cotes par issue, book correspondant).

    L'edge se calcule TOUJOURS contre la meilleure cote disponible, jamais
    contre une moyenne : c'est la ligne qu'on peut reellement jouer.
    """
    if not quotes:
        raise ValueError("Aucune cote fournie")
    books = list(quotes)
    matrix = np.asarray([quotes[b] for b in books], dtype=float)
    idx = np.nanargmax(matrix, axis=0)
    return matrix[idx, np.arange(matrix.shape[1])], [books[i] for i in idx]
