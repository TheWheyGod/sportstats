"""
Dimensionnement de mise : critere de Kelly et ses variantes prudentes.

Le Kelly plein maximise la croissance logarithmique du capital MAIS suppose que
la probabilite p est connue exactement. Elle ne l'est jamais : elle sort d'un
modele. Une surestimation de 2 points de proba suffit a transformer un Kelly
plein en strategie perdante. On travaille donc en Kelly fractionnaire (0.25 par
defaut) avec plafond par pari, ce qui divise la volatilite par ~4 en ne coutant
qu'environ 25% du taux de croissance theorique.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "kelly_fraction",
    "kelly_stake",
    "kelly_simultaneous_exclusive",
    "growth_rate",
    "risk_of_ruin_mc",
]


def kelly_fraction(p: float, odds: float) -> float:
    """
    Fraction de bankroll a miser (Kelly plein) pour un pari binaire.

        f* = (p * o - 1) / (o - 1)

    Retourne 0 si l'esperance est negative (on ne mise pas).
    """
    o = float(odds)
    if o <= 1.0:
        return 0.0
    f = (float(p) * o - 1.0) / (o - 1.0)
    return float(max(0.0, f))


def kelly_stake(
    p: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.02,
    min_stake: float = 0.0,
    round_to: float | None = None,
) -> dict:
    """
    Mise recommandee en unites monetaires.

    fraction      : coefficient de Kelly fractionnaire (0.25 = quart de Kelly).
    max_stake_pct : plafond dur en % de bankroll, applique APRES la fraction.
                    C'est le garde-fou reel contre l'erreur de modele.
    round_to      : arrondi (ex. 0.5 pour arrondir au demi-euro).
    """
    f_full = kelly_fraction(p, odds)
    f_used = min(f_full * fraction, max_stake_pct)
    stake = bankroll * f_used
    if round_to:
        stake = round(stake / round_to) * round_to
    if stake < min_stake:
        stake = 0.0
    return {
        "kelly_full": f_full,
        "kelly_used": f_used,
        "stake": float(stake),
        "capped": bool(f_full * fraction > max_stake_pct),
        "ev_per_unit": float(p * odds - 1.0),
    }


def kelly_simultaneous_exclusive(
    probs: np.ndarray,
    odds: np.ndarray,
    fraction: float = 0.25,
    max_total_pct: float = 0.10,
) -> np.ndarray:
    """
    Kelly pour plusieurs issues MUTUELLEMENT EXCLUSIVES du meme evenement
    (ex. miser a la fois sur le 1 et sur le X d'un match).

    Maximise E[log(capital)] = sum_i p_i * log(1 + f_i*(o_i - 1) - sum_{j!=i} f_j)
    sous contraintes f_i >= 0 et sum(f_i) <= 1.

    Il n'existe pas de forme close generale : on resout numeriquement (SLSQP).
    """
    p = np.asarray(probs, dtype=float)
    o = np.asarray(odds, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])

    def neg_log_growth(f):
        total = f.sum()
        # capital final si l'issue i se realise
        wealth = 1.0 - total + f * o
        if np.any(wealth <= 1e-9):
            return 1e6
        return -float(np.sum(p * np.log(wealth)))

    cons = [{"type": "ineq", "fun": lambda f: 0.999 - f.sum()}]
    bounds = [(0.0, 0.999)] * n
    x0 = np.full(n, 0.01)
    res = minimize(neg_log_growth, x0, bounds=bounds, constraints=cons, method="SLSQP")
    f_opt = np.maximum(res.x if res.success else x0 * 0.0, 0.0)

    f_opt = f_opt * fraction
    total = f_opt.sum()
    if total > max_total_pct:
        f_opt *= max_total_pct / total
    f_opt[f_opt < 1e-4] = 0.0
    return f_opt


def growth_rate(p: float, odds: float, f: float) -> float:
    """Taux de croissance log esperé par pari pour une fraction f donnee."""
    if f <= 0:
        return 0.0
    win = 1.0 + f * (odds - 1.0)
    lose = 1.0 - f
    if win <= 0 or lose <= 0:
        return -np.inf
    return float(p * np.log(win) + (1 - p) * np.log(lose))


def risk_of_ruin_mc(
    p: float,
    odds: float,
    f: float,
    n_bets: int = 500,
    ruin_level: float = 0.5,
    n_sims: int = 20000,
    seed: int = 0,
) -> dict:
    """
    Simule n_sims sequences de n_bets paris a mise fractionnaire constante.
    Retourne la probabilite de descendre sous `ruin_level` x bankroll initiale,
    plus les quantiles du capital final. Sert a verifier qu'une fraction de
    Kelly choisie est tolerable AVANT de la jouer.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_sims, n_bets)) < p
    mult = np.where(wins, 1.0 + f * (odds - 1.0), 1.0 - f)
    paths = np.cumprod(mult, axis=1)
    ruined = (paths.min(axis=1) < ruin_level).mean()
    final = paths[:, -1]
    return {
        "risk_of_ruin": float(ruined),
        "median_final": float(np.median(final)),
        "p05_final": float(np.quantile(final, 0.05)),
        "p95_final": float(np.quantile(final, 0.95)),
        "mean_log_growth": float(np.mean(np.log(final)) / n_bets),
    }
