"""
Metriques d'evaluation : qualite probabiliste et performance financiere.

Distinction importante :
  - Brier / log-loss mesurent la qualite des PROBABILITES (est-ce que le modele
    dit vrai ?).
  - ROI / yield mesurent le resultat FINANCIER, mais avec une variance enorme :
    sur 200 paris a cote 2.00, un ROI de +8% n'est pas significatif.
  - CLV (Closing Line Value) mesure si on a pris la cote AVANT que le marche ne
    bouge dans notre sens. C'est le signal qui converge le plus vite : quelques
    centaines de paris suffisent, contre plusieurs milliers pour le ROI.
"""
from __future__ import annotations

import numpy as np

from .oddsmath import devig

__all__ = [
    "brier_score",
    "log_loss_multi",
    "rps",
    "accuracy",
    "roi_summary",
    "closing_line_value",
    "clv_baseline_control",
    "max_drawdown",
    "bootstrap_roi_ci",
    "edge_t_stat",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Qualite probabiliste
# --------------------------------------------------------------------------
def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier multi-classes. 0 = parfait. Baseline 1X2 uniforme = 0.667."""
    p = np.atleast_2d(np.asarray(probs, dtype=float))
    y = np.asarray(outcomes, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss_multi(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.atleast_2d(np.asarray(probs, dtype=float))
    y = np.asarray(outcomes, dtype=int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0))))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Ranked Probability Score : penalise davantage une erreur "lointaine".
    Metrique de reference sur le 1X2 football, car l'ordre 1 < X < 2 a un sens
    (predire 2 quand c'est 1 est pire que predire X quand c'est 1).
    """
    p = np.atleast_2d(np.asarray(probs, dtype=float))
    y = np.asarray(outcomes, dtype=int)
    n, k = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    cum_p = np.cumsum(p, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (k - 1)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.atleast_2d(np.asarray(probs, dtype=float))
    return float(np.mean(p.argmax(axis=1) == np.asarray(outcomes, dtype=int)))


# --------------------------------------------------------------------------
# Performance financiere
# --------------------------------------------------------------------------
def roi_summary(stakes: np.ndarray, odds: np.ndarray, won: np.ndarray) -> dict:
    """
    stakes : mises, odds : cotes prises, won : 1 si gagnant, 0 sinon
             (0.5 accepte pour les remboursements partiels type handicap
             asiatique quart de balle ; 'push' complet = won 'nan' exclu).
    """
    s = np.asarray(stakes, dtype=float)
    o = np.asarray(odds, dtype=float)
    w = np.asarray(won, dtype=float)

    keep = ~np.isnan(w)
    s, o, w = s[keep], o[keep], w[keep]
    if len(s) == 0:
        return {"n": 0, "turnover": 0.0, "profit": 0.0, "roi": 0.0, "yield": 0.0}

    # gain = mise * (cote - 1) * part_gagnante  -  mise * part_perdante
    profit = s * (o - 1.0) * w - s * (1.0 - w)
    turnover = float(s.sum())
    total = float(profit.sum())
    return {
        "n": int(len(s)),
        "turnover": turnover,
        "profit": total,
        "roi": total / turnover if turnover else 0.0,
        "yield": total / turnover if turnover else 0.0,
        "win_rate": float(np.mean(w > 0.5)),
        "avg_odds": float(o.mean()),
        "profit_series": profit,
    }


def closing_line_value(
    odds_taken: np.ndarray,
    closing_odds_market: list,
    outcome_index: np.ndarray,
    devig_method: str = "shin",
) -> dict:
    """
    CLV : compare la cote prise a la cote FAIR de cloture (marge retiree).

    odds_taken         : (n,) cote effectivement prise
    closing_odds_market: liste de n listes = toutes les cotes du marche a la
                         cloture (pour pouvoir de-vigger)
    outcome_index      : (n,) index de l'issue sur laquelle on a mise

    beat_close_rate > 55% sur plusieurs centaines de paris = edge reel.
    Un ROI positif SANS CLV positif est presque toujours de la chance.
    """
    taken = np.asarray(odds_taken, dtype=float)
    idx = np.asarray(outcome_index, dtype=int)

    fair_close, clv_pct = [], []
    for i, market in enumerate(closing_odds_market):
        p_fair = devig(market, method=devig_method)[idx[i]]
        fo = 1.0 / p_fair
        fair_close.append(fo)
        clv_pct.append(taken[i] / fo - 1.0)

    fair_close = np.asarray(fair_close)
    clv_pct = np.asarray(clv_pct)
    return {
        "clv_mean_pct": float(clv_pct.mean()),
        "clv_median_pct": float(np.median(clv_pct)),
        "beat_close_rate": float(np.mean(taken > fair_close)),
        "n": int(len(taken)),
        "clv_series": clv_pct,
    }


def clv_baseline_control(
    all_bet_odds: np.ndarray,
    all_closing_odds: np.ndarray,
    devig_method: str = "shin",
    n_draws: int = 20000,
    seed: int = 0,
) -> dict:
    """
    CONTROLE INDISPENSABLE quand on mise a la meilleure cote parmi N books.

    Battre la ligne de cloture devient alors partiellement MECANIQUE : le
    maximum d'un echantillon de 30 books depasse presque toujours la cote fair,
    meme en choisissant l'issue au hasard. Sans ce controle, on attribue a la
    qualite du modele un CLV qui vient simplement du shopping de cotes.

    Cette fonction mesure le CLV d'une selection ALEATOIRE aux memes cotes.
    Le CLV attribuable au modele est la difference :

        CLV_utile = CLV_observe - CLV_aleatoire

    Parameters
    ----------
    all_bet_odds     : (n, k) cotes auxquelles on aurait pu miser
    all_closing_odds : (n, k) cotes de cloture du marche complet
    """
    bet = np.asarray(all_bet_odds, dtype=float)
    close = np.asarray(all_closing_odds, dtype=float)
    ok = np.isfinite(bet).all(axis=1) & np.isfinite(close).all(axis=1) & (close > 1.0).all(axis=1)
    bet, close = bet[ok], close[ok]
    if len(bet) == 0:
        return {"clv_aleatoire_pct": 0.0, "beat_close_aleatoire": 0.5, "n": 0}

    fair = np.vstack([1.0 / devig(row, method=devig_method) for row in close])
    rng = np.random.default_rng(seed)
    n = min(n_draws, len(bet))
    rows = rng.integers(0, len(bet), size=n)
    cols = rng.integers(0, bet.shape[1], size=n)
    taken = bet[rows, cols]
    fair_taken = fair[rows, cols]
    clv = taken / fair_taken - 1.0
    return {
        "clv_aleatoire_pct": float(clv.mean()),
        "beat_close_aleatoire": float(np.mean(taken > fair_taken)),
        "n": int(len(bet)),
    }


def max_drawdown(profit_series: np.ndarray, starting_bankroll: float = 100.0) -> dict:
    """Drawdown maximal en valeur et en % du pic."""
    equity = starting_bankroll + np.cumsum(np.asarray(profit_series, dtype=float))
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    dd_pct = dd / np.where(peak == 0, 1.0, peak)
    i = int(np.argmin(dd_pct)) if len(dd_pct) else 0
    return {
        "max_dd_abs": float(dd.min()) if len(dd) else 0.0,
        "max_dd_pct": float(dd_pct.min()) if len(dd_pct) else 0.0,
        "at_bet": i,
        "final_equity": float(equity[-1]) if len(equity) else starting_bankroll,
    }


def bootstrap_roi_ci(
    profit_series: np.ndarray,
    stakes: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Intervalle de confiance bootstrap sur le ROI.

    A lire systematiquement : si l'IC 95% contient 0, le backtest ne demontre
    rien, quel que soit le ROI ponctuel affiche.
    """
    rng = np.random.default_rng(seed)
    profit = np.asarray(profit_series, dtype=float)
    s = np.asarray(stakes, dtype=float)
    n = len(profit)
    if n == 0:
        return {"roi": 0.0, "lo": 0.0, "hi": 0.0, "p_gt_0": 0.0}
    idx = rng.integers(0, n, size=(n_boot, n))
    rois = profit[idx].sum(axis=1) / np.maximum(s[idx].sum(axis=1), 1e-9)
    return {
        "roi": float(profit.sum() / max(s.sum(), 1e-9)),
        "lo": float(np.quantile(rois, alpha / 2)),
        "hi": float(np.quantile(rois, 1 - alpha / 2)),
        "p_gt_0": float(np.mean(rois > 0)),
    }


def edge_t_stat(profit_series: np.ndarray, stakes: np.ndarray) -> float:
    """
    t-stat du profit moyen par unite misee. |t| > 2 ~ significatif a 5%.
    Sur du pari sportif, atteindre t > 2 demande typiquement 1000+ paris a
    edge realiste (2-4%).
    """
    p = np.asarray(profit_series, dtype=float)
    s = np.asarray(stakes, dtype=float)
    if len(p) < 2:
        return 0.0
    r = p / np.maximum(s, 1e-9)
    sd = r.std(ddof=1)
    return float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else 0.0
