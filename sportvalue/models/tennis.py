"""
Tennis : Elo par surface + modele markovien hierarchique (Barnett-Clarke).

Le tennis est le sport ou un modele maison a le plus de chances de battre les
books secondaires, pour trois raisons structurelles :
  - la hierarchie point -> jeu -> set -> match est EXACTEMENT calculable, sans
    approximation, a partir d'un seul parametre par joueur (% de points gagnes
    au service) ;
  - les marches derives (total de jeux, handicap de jeux, score en sets) sont
    souvent cotes par simple regle empirique chez les books non specialises,
    alors qu'ils decoulent mecaniquement du meme modele ;
  - la specificite de surface est forte et lentement integree.

Chaine de calcul :
  1. Elo (global + surface) -> probabilite de victoire du match.
  2. On inverse cette probabilite pour trouver le couple de % de points au
     service coherent avec elle. C'est ce qui garantit que le total de jeux et
     le vainqueur ne se contredisent jamais.
  3. Programmation dynamique exacte -> distribution jointe (sets, jeux).

L'etape 2 est la cle : sans elle, on aurait un modele pour le vainqueur et un
autre pour les jeux, avec des incoherences qui produisent de faux value bets.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from ..core.scoredist import ScoreDistribution
from .elo import EloRating, elo_expected

__all__ = ["game_win_prob", "tiebreak_win_prob", "TennisMatchModel", "TennisPrediction"]

# % moyen de points gagnes au service, par circuit et surface.
# Sert de point d'ancrage quand on ne dispose que d'un Elo.
SERVE_BASELINE = {
    ("atp", "Hard"): 0.645,
    ("atp", "Clay"): 0.625,
    ("atp", "Grass"): 0.665,
    ("atp", "Carpet"): 0.660,
    ("wta", "Hard"): 0.575,
    ("wta", "Clay"): 0.560,
    ("wta", "Grass"): 0.590,
    ("wta", "Carpet"): 0.585,
}


# --------------------------------------------------------------------------
# Niveau jeu
# --------------------------------------------------------------------------
def game_win_prob(p: float) -> float:
    """
    Probabilite que le serveur gagne son jeu, sachant p = P(gagner un point).

    Forme close : 4-0, 4-1, 4-2, puis egalite (deuce) resolue par p^2/(p^2+q^2).
    """
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    q = 1.0 - p
    deuce = p * p / (p * p + q * q)
    return p**4 + 4 * p**4 * q + 10 * p**4 * q * q + 20 * p**3 * q**3 * deuce


@lru_cache(maxsize=None)
def _tb(a: int, b: int, pa: float, pb: float) -> float:
    """P(joueur A gagne le tie-break) depuis le score (a, b). A sert en premier."""
    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0
    if a + b > 60:  # garde-fou : masse residuelle negligeable
        return 0.5
    n = a + b
    server_is_a = ((n + 1) // 2) % 2 == 0
    p_a_wins_point = pa if server_is_a else 1.0 - pb
    return p_a_wins_point * _tb(a + 1, b, pa, pb) + (1 - p_a_wins_point) * _tb(a, b + 1, pa, pb)


def tiebreak_win_prob(pa: float, pb: float) -> float:
    """pa, pb = % de points gagnes au service de chaque joueur. A sert en premier."""
    return _tb(0, 0, round(float(pa), 6), round(float(pb), 6))


# --------------------------------------------------------------------------
# Niveau set
# --------------------------------------------------------------------------
def set_distribution(hold_a: float, hold_b: float, tb_a: float, a_serves_first: bool) -> dict:
    """
    Distribution exacte du score du set : {(jeux_A, jeux_B): probabilite}.

    Le service alterne a chaque jeu, donc le serveur est determine par la
    parite du nombre de jeux deja joues.
    """
    states = {(0, 0): 1.0}
    out: dict[tuple[int, int], float] = {}

    for _ in range(40):  # au plus 13 jeux, marge large
        nxt: dict[tuple[int, int], float] = {}
        for (ga, gb), p in states.items():
            if p <= 0:
                continue
            # terminaisons
            if (ga == 6 and gb <= 4) or (ga == 7 and gb == 5):
                out[(ga, gb)] = out.get((ga, gb), 0.0) + p
                continue
            if (gb == 6 and ga <= 4) or (gb == 7 and ga == 5):
                out[(ga, gb)] = out.get((ga, gb), 0.0) + p
                continue
            if ga == 6 and gb == 6:
                out[(7, 6)] = out.get((7, 6), 0.0) + p * tb_a
                out[(6, 7)] = out.get((6, 7), 0.0) + p * (1 - tb_a)
                continue

            n = ga + gb
            a_serves = (n % 2 == 0) if a_serves_first else (n % 2 == 1)
            p_a_wins_game = hold_a if a_serves else 1.0 - hold_b
            nxt[(ga + 1, gb)] = nxt.get((ga + 1, gb), 0.0) + p * p_a_wins_game
            nxt[(ga, gb + 1)] = nxt.get((ga, gb + 1), 0.0) + p * (1 - p_a_wins_game)
        states = nxt
        if not states:
            break

    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else out


# --------------------------------------------------------------------------
# Niveau match
# --------------------------------------------------------------------------
@dataclass
class TennisPrediction:
    """Toutes les probabilites derivant du meme modele, donc coherentes."""

    p_a: float
    serve_a: float
    serve_b: float
    hold_a: float
    hold_b: float
    set_scores: dict          # (sets_A, sets_B) -> proba
    games_dist: ScoreDistribution   # sur (jeux_A, jeux_B)
    sets_dist: ScoreDistribution    # sur (sets_A, sets_B)
    best_of: int

    def moneyline(self) -> dict:
        return {"1": self.p_a, "2": 1.0 - self.p_a}

    def total_games(self, line: float) -> dict:
        return self.games_dist.over_under(line)

    def games_handicap(self, line: float) -> dict:
        return self.games_dist.asian_handicap(line)

    def set_betting(self) -> dict:
        return {f"{a}-{b}": p for (a, b), p in sorted(self.set_scores.items(), key=lambda kv: -kv[1])}

    def total_sets(self, line: float) -> dict:
        return self.sets_dist.over_under(line)

    def expected_games(self) -> float:
        return self.games_dist.expected_total

    def summary(self) -> dict:
        return {
            "p_joueur_A": round(self.p_a, 4),
            "pts_service_A": round(self.serve_a, 4),
            "pts_service_B": round(self.serve_b, 4),
            "hold_A": round(self.hold_a, 4),
            "hold_B": round(self.hold_b, 4),
            "jeux_attendus": round(self.expected_games(), 2),
            "score_sets_probable": max(self.set_scores.items(), key=lambda kv: kv[1])[0],
        }


def _match_distribution(serve_a: float, serve_b: float, best_of: int) -> tuple:
    """
    DP sur (sets_A, sets_B, jeux_A, jeux_B, premier_serveur).

    Regle d'alternance du service entre sets : si le set a compte un nombre PAIR
    de jeux, le meme joueur sert en premier au set suivant ; sinon c'est l'autre.
    (Un tie-break compte pour un jeu, d'ou 7-6 = 13 jeux = impair.)
    """
    hold_a = game_win_prob(serve_a)
    hold_b = game_win_prob(serve_b)
    tb_a_first = tiebreak_win_prob(serve_a, serve_b)
    tb_b_first = 1.0 - tiebreak_win_prob(serve_b, serve_a)

    sets_needed = best_of // 2 + 1
    dist_a_first = set_distribution(hold_a, hold_b, tb_a_first, True)
    dist_b_first = set_distribution(hold_a, hold_b, tb_b_first, False)

    # etat : (sets_a, sets_b, games_a, games_b, a_serves_first) -> proba
    states = {(0, 0, 0, 0, True): 1.0}
    final: dict[tuple[int, int, int, int], float] = {}

    for _ in range(best_of):
        nxt: dict = {}
        for (sa, sb, ga, gb, a_first), p in states.items():
            if p <= 0:
                continue
            table = dist_a_first if a_first else dist_b_first
            for (dga, dgb), ps in table.items():
                nsa = sa + (1 if dga > dgb else 0)
                nsb = sb + (1 if dgb > dga else 0)
                nga, ngb = ga + dga, gb + dgb
                nfirst = a_first if (dga + dgb) % 2 == 0 else (not a_first)
                w = p * ps
                if nsa == sets_needed or nsb == sets_needed:
                    key = (nsa, nsb, nga, ngb)
                    final[key] = final.get(key, 0.0) + w
                else:
                    k2 = (nsa, nsb, nga, ngb, nfirst)
                    nxt[k2] = nxt.get(k2, 0.0) + w
        states = nxt
        if not states:
            break

    total = sum(final.values())
    if total > 0:
        final = {k: v / total for k, v in final.items()}
    return final, hold_a, hold_b


def _p_win_from_serve(serve_a: float, serve_b: float, best_of: int) -> float:
    final, _, _ = _match_distribution(serve_a, serve_b, best_of)
    sets_needed = best_of // 2 + 1
    return sum(p for (sa, _sb, _ga, _gb), p in final.items() if sa == sets_needed)


class TennisMatchModel:
    """
    Parameters
    ----------
    tour : 'atp' ou 'wta' (fixe la baseline de points au service).
    elo : instance EloRating deja entrainee (optionnel). Si absente, il faut
          fournir p_match ou les stats de service directement.
    """

    def __init__(self, tour: str = "atp", elo: EloRating | None = None):
        self.tour = tour
        self.elo = elo or EloRating(
            k0=250, k_offset=5, k_shape=0.4, home_advantage=0.0, surface_weight=0.5
        )

    # ------------------------------------------------------------------
    def solve_serve_probs(
        self,
        p_match: float,
        surface: str = "Hard",
        best_of: int = 3,
    ) -> tuple[float, float]:
        """
        Trouve (serve_A, serve_B) symetriques autour de la baseline de surface
        tels que le modele markovien reproduise exactement `p_match`.

        On bouge les deux joueurs en sens opposes : cela conserve le nombre
        total de jeux attendu, donc on ne deforme pas le marche des totaux en
        ajustant le marche du vainqueur.
        """
        base = SERVE_BASELINE.get((self.tour, surface), 0.63)
        p_target = float(np.clip(p_match, 0.005, 0.995))

        def f(delta: float) -> float:
            return _p_win_from_serve(base + delta, base - delta, best_of) - p_target

        lo, hi = -0.22, 0.22
        if f(lo) > 0:
            return base + lo, base - lo
        if f(hi) < 0:
            return base + hi, base - hi
        delta = brentq(f, lo, hi, xtol=1e-6, maxiter=60)
        return base + delta, base - delta

    # ------------------------------------------------------------------
    def predict(
        self,
        player_a: str,
        player_b: str,
        surface: str = "Hard",
        best_of: int = 3,
        p_match: float | None = None,
        serve_stats: tuple[float, float] | None = None,
    ) -> TennisPrediction:
        """
        Trois modes, par ordre de precision decroissante :
          - serve_stats fourni : on utilise directement (spw_A, spw_B) ;
          - p_match fourni     : on inverse pour trouver les % de service ;
          - sinon              : p_match vient de l'Elo de surface.
        """
        if serve_stats is not None:
            sa, sb = serve_stats
        else:
            if p_match is None:
                p_match = self.elo.predict(player_a, player_b, surface=surface, neutral=True)
            sa, sb = self.solve_serve_probs(p_match, surface, best_of)

        final, hold_a, hold_b = _match_distribution(sa, sb, best_of)
        sets_needed = best_of // 2 + 1

        set_scores: dict[tuple[int, int], float] = {}
        games_rows, games_w = [], []
        sets_rows, sets_w = [], []
        for (sa_, sb_, ga, gb), p in final.items():
            set_scores[(sa_, sb_)] = set_scores.get((sa_, sb_), 0.0) + p
            games_rows.append((ga, gb))
            games_w.append(p)
            sets_rows.append((sa_, sb_))
            sets_w.append(p)

        games_arr = np.asarray(games_rows, dtype=float)
        sets_arr = np.asarray(sets_rows, dtype=float)
        games_dist = ScoreDistribution(games_arr[:, 0], games_arr[:, 1], np.asarray(games_w))
        sets_dist = ScoreDistribution(sets_arr[:, 0], sets_arr[:, 1], np.asarray(sets_w))

        p_a = sum(p for (s_a, _s_b), p in set_scores.items() if s_a == sets_needed)

        return TennisPrediction(
            p_a=float(p_a),
            serve_a=float(sa),
            serve_b=float(sb),
            hold_a=float(hold_a),
            hold_b=float(hold_b),
            set_scores=set_scores,
            games_dist=games_dist,
            sets_dist=sets_dist,
            best_of=best_of,
        )
