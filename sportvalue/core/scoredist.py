"""
Distribution jointe des scores -> probabilites de TOUS les marches.

Brique partagee par les quatre sports. Deux constructions possibles :

  - `from_matrix`  : matrice (n_home+1, n_away+1) des probabilites exactes.
                     Utilise par le football (Dixon-Coles) ou les scores sont
                     petits et enumerables.
  - `from_samples` : echantillons Monte-Carlo (n, 2) de scores simules.
                     Utilise par le rugby (points composes tries/penalites) et
                     le basket (marge quasi-normale, scores trop grands pour
                     etre enumeres).

L'interet d'unifier : la logique de reglement des paris (push, lignes quart de
balle, handicap asiatique) est ecrite une seule fois et testee une seule fois.
C'est la ou se cachent la plupart des bugs silencieux dans les outils maison :
un handicap -0.25 mal regle fausse tout le backtest sans jamais lever d'erreur.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ScoreDistribution", "settle_asian", "settle_total"]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Reglement des paris (logique pure, testable isolement)
# --------------------------------------------------------------------------
def settle_asian(margin: np.ndarray, line: float) -> np.ndarray:
    """
    Fraction de mise gagnee sur un handicap asiatique applique a l'equipe
    dont on mesure la marge. Retourne un tableau dans {-1, -0.5, 0, 0.5, 1} :

        +1   gagne         0    rembourse (push)
        +0.5 moitie gagnee -0.5 moitie perdue
        -1   perdu

    Une ligne quart (x.25 / x.75) est traitee comme deux demi-mises sur les
    deux lignes adjacentes, ce qui est exactement la regle des bookmakers.
    """
    m = np.asarray(margin, dtype=float)
    frac = abs(line * 4) % 2  # 1 si ligne quart (x.25 ou x.75)
    if abs(frac - 1.0) < 1e-9:
        lo, hi = line - 0.25, line + 0.25
        return 0.5 * (_settle_simple(m, lo) + _settle_simple(m, hi))
    return _settle_simple(m, line)


def _settle_simple(margin: np.ndarray, line: float) -> np.ndarray:
    adj = margin + line
    out = np.zeros_like(margin, dtype=float)
    out[adj > _EPS] = 1.0
    out[adj < -_EPS] = -1.0
    return out  # adj == 0 reste a 0 (push)


def settle_total(total: np.ndarray, line: float, side: str = "over") -> np.ndarray:
    """Idem pour un Over/Under, lignes quart incluses."""
    t = np.asarray(total, dtype=float)
    sign = 1.0 if side == "over" else -1.0
    frac = abs(line * 4) % 2
    if abs(frac - 1.0) < 1e-9:
        lo, hi = line - 0.25, line + 0.25
        return 0.5 * (_settle_simple(sign * (t - lo), 0.0) + _settle_simple(sign * (t - hi), 0.0))
    return _settle_simple(sign * (t - line), 0.0)


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------
class ScoreDistribution:
    """
    Distribution jointe des scores (domicile, exterieur).

    Toutes les methodes renvoient des probabilites NON viggees : ce sont les
    probabilites du modele, a comparer directement aux probabilites fair du
    marche pour detecter un mauvais prix.
    """

    def __init__(self, home_scores: np.ndarray, away_scores: np.ndarray, weights: np.ndarray):
        self.h = np.asarray(home_scores, dtype=float).ravel()
        self.a = np.asarray(away_scores, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        self.w = w / w.sum()
        self.margin = self.h - self.a
        self.total = self.h + self.a

    # ---- constructeurs -------------------------------------------------
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "ScoreDistribution":
        m = np.asarray(matrix, dtype=float)
        hh, aa = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        return cls(hh.ravel(), aa.ravel(), m.ravel())

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "ScoreDistribution":
        s = np.asarray(samples, dtype=float)
        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError("samples doit etre de forme (n, 2)")
        return cls(s[:, 0], s[:, 1], np.ones(len(s)))

    # ---- statistiques descriptives -------------------------------------
    @property
    def expected_home(self) -> float:
        return float(np.sum(self.w * self.h))

    @property
    def expected_away(self) -> float:
        return float(np.sum(self.w * self.a))

    @property
    def expected_total(self) -> float:
        return float(np.sum(self.w * self.total))

    @property
    def expected_margin(self) -> float:
        return float(np.sum(self.w * self.margin))

    def margin_std(self) -> float:
        mu = self.expected_margin
        return float(np.sqrt(np.sum(self.w * (self.margin - mu) ** 2)))

    # ---- marches -------------------------------------------------------
    def market_1x2(self) -> dict:
        """Vainqueur temps reglementaire (1 / nul / 2)."""
        return {
            "1": float(np.sum(self.w[self.margin > 0])),
            "X": float(np.sum(self.w[self.margin == 0])),
            "2": float(np.sum(self.w[self.margin < 0])),
        }

    def market_moneyline(self) -> dict:
        """Deux issues, sans nul (tennis, basket avec prolongations)."""
        p1 = float(np.sum(self.w[self.margin > 0]))
        p2 = float(np.sum(self.w[self.margin < 0]))
        s = p1 + p2
        return {"1": p1 / s, "2": p2 / s} if s > 0 else {"1": 0.5, "2": 0.5}

    def over_under(self, line: float) -> dict:
        """
        Probabilites Over/Under. Sur une ligne entiere (ex. 2.0) le push est
        renvoye separement : l'ignorer surestime l'edge d'environ 1 a 3 points.
        """
        over = settle_total(self.total, line, "over")
        p_win = float(np.sum(self.w[over > 0.9]))
        p_half_win = float(np.sum(self.w[(over > 0.1) & (over < 0.9)]))
        p_push = float(np.sum(self.w[np.abs(over) < 0.1]))
        p_half_loss = float(np.sum(self.w[(over < -0.1) & (over > -0.9)]))
        p_loss = float(np.sum(self.w[over < -0.9]))
        return {
            "line": line,
            "p_over": p_win + 0.5 * p_half_win,
            "p_under": p_loss + 0.5 * p_half_loss,
            "p_push": p_push,
            "over_fair_odds": self._fair_odds_from_settlement(over),
            "under_fair_odds": self._fair_odds_from_settlement(-over),
        }

    def asian_handicap(self, line: float) -> dict:
        """
        Handicap asiatique, `line` du point de vue de l'equipe a domicile.
        Ex. line = -1.5 : domicile doit gagner de 2 buts ou plus.
        """
        home = settle_asian(self.margin, line)
        away = settle_asian(-self.margin, -line)
        return {
            "line": line,
            "home_fair_odds": self._fair_odds_from_settlement(home),
            "away_fair_odds": self._fair_odds_from_settlement(away),
            "home_ev_neutral_prob": self._implied_prob_from_settlement(home),
            "away_ev_neutral_prob": self._implied_prob_from_settlement(away),
        }

    def _fair_odds_from_settlement(self, settlement: np.ndarray) -> float:
        """
        Cote a laquelle l'esperance du pari est nulle.

        EV(o) = sum_i w_i * [ s_i > 0 ? s_i*(o-1) : s_i ]
        On resout EV(o) = 0  =>  o = 1 + perte_esperee / gain_unitaire_espere
        """
        s = np.asarray(settlement, dtype=float)
        win_units = float(np.sum(self.w * np.maximum(s, 0.0)))
        loss_units = float(np.sum(self.w * np.maximum(-s, 0.0)))
        if win_units <= _EPS:
            return float("inf")
        return 1.0 + loss_units / win_units

    def _implied_prob_from_settlement(self, settlement: np.ndarray) -> float:
        fo = self._fair_odds_from_settlement(settlement)
        return 1.0 / fo if np.isfinite(fo) and fo > 0 else 0.0

    def btts(self) -> dict:
        """Les deux equipes marquent."""
        yes = float(np.sum(self.w[(self.h > 0) & (self.a > 0)]))
        return {"oui": yes, "non": 1.0 - yes}

    def double_chance(self) -> dict:
        m = self.market_1x2()
        return {"1X": m["1"] + m["X"], "12": m["1"] + m["2"], "X2": m["X"] + m["2"]}

    def draw_no_bet(self) -> dict:
        m = self.market_1x2()
        s = m["1"] + m["2"]
        return {"1": m["1"] / s, "2": m["2"] / s} if s > 0 else {"1": 0.5, "2": 0.5}

    def team_total(self, team: str, line: float) -> dict:
        scores = self.h if team == "home" else self.a
        over = settle_total(scores, line, "over")
        return {
            "team": team,
            "line": line,
            "p_over": float(np.sum(self.w[over > 0.9]) + 0.5 * np.sum(self.w[(over > 0.1) & (over < 0.9)])),
            "p_under": float(np.sum(self.w[over < -0.9]) + 0.5 * np.sum(self.w[(over < -0.1) & (over > -0.9)])),
        }

    def correct_score(self, top_n: int = 12) -> list[dict]:
        order = np.argsort(-self.w)[:top_n]
        return [
            {"score": f"{int(self.h[i])}-{int(self.a[i])}", "p": float(self.w[i])}
            for i in order
        ]

    def margin_band(self, lo: float, hi: float) -> float:
        """P(marge dans [lo, hi]) - utile en rugby (ecart 1-12, 13+...)."""
        return float(np.sum(self.w[(self.margin >= lo) & (self.margin <= hi)]))

    def summary(self) -> dict:
        m = self.market_1x2()
        return {
            "xg_home": self.expected_home,
            "xg_away": self.expected_away,
            "total_attendu": self.expected_total,
            "marge_attendue": self.expected_margin,
            "ecart_type_marge": self.margin_std(),
            **m,
        }
