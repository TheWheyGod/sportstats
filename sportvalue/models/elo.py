"""
Elo generique : socle de forme recente pour tous les sports.

Trois raffinements par rapport a l'Elo d'echecs de base, chacun repondant a un
defaut concret quand on l'applique au sport :

  - K DYNAMIQUE : K = k0 / (n + offset)^shape. Un joueur qui debute voit sa note
    bouger vite, un joueur avec 300 matchs bouge lentement. Un K fixe met une
    saison entiere a integrer l'arrivee d'un joueur de haut niveau.

  - MARGE DE VICTOIRE : gagner 40-3 n'est pas gagner 21-20. Le multiplicateur
    log(marge+1) * f(diff_elo) est celui utilise par FiveThirtyEight ; le terme
    en diff_elo empeche l'inflation des notes des equipes deja dominantes qui
    ecrasent des adversaires faibles.

  - SPECIFICITE DE SURFACE (tennis) : on tient une note globale et une note par
    surface, et on predit avec un melange. Un specialiste de terre battue peut
    valoir 200 points d'Elo de plus a Roland-Garros qu'a Wimbledon, ce que les
    books amateurs integrent mal en debut de saison sur terre.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

__all__ = ["EloRating", "elo_expected"]


def elo_expected(diff: float) -> float:
    """Probabilite de victoire pour un ecart de note donne."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


class EloRating:
    """
    Parameters
    ----------
    k0, k_offset, k_shape : parametres du K dynamique.
        K fixe classique : k0=20, k_shape=0 (n'importe quel offset).
        Tennis (Sackmann) : k0=250, k_offset=5, k_shape=0.4.
        Football/basket   : k0=20, k_shape=0 convient.
    home_advantage : en points d'Elo (football ~60, basket NBA ~100,
        rugby ~70, tennis 0).
    use_mov : active le multiplicateur de marge de victoire.
    surface_weight : pour le tennis, poids de la note de surface dans la
        prediction finale (0.5 = moitie surface / moitie global).
    """

    def __init__(
        self,
        k0: float = 20.0,
        k_offset: float = 5.0,
        k_shape: float = 0.0,
        base: float = 1500.0,
        home_advantage: float = 60.0,
        use_mov: bool = False,
        mov_scale: float = 1.0,
        surface_weight: float = 0.5,
        regress_to_mean: float = 0.0,
    ):
        self.k0 = k0
        self.k_offset = k_offset
        self.k_shape = k_shape
        self.base = base
        self.home_advantage = home_advantage
        self.use_mov = use_mov
        self.mov_scale = mov_scale
        self.surface_weight = surface_weight
        self.regress_to_mean = regress_to_mean

        self.ratings: dict[str, float] = defaultdict(lambda: base)
        self.surface_ratings: dict[tuple[str, str], float] = defaultdict(lambda: base)
        self.n_games: dict[str, int] = defaultdict(int)
        self.last_date: dict[str, pd.Timestamp] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _k(self, name: str) -> float:
        if self.k_shape == 0:
            return self.k0
        return self.k0 / (self.n_games[name] + self.k_offset) ** self.k_shape

    def get(self, name: str, surface: str | None = None) -> float:
        """Note utilisee pour la prediction (melangee si surface fournie)."""
        base_r = self.ratings[name]
        if surface is None or self.surface_weight <= 0:
            return base_r
        surf_r = self.surface_ratings[(name, surface)]
        if self.n_games[name] == 0:
            return base_r
        w = self.surface_weight
        return (1 - w) * base_r + w * surf_r

    def predict(
        self,
        home: str,
        away: str,
        surface: str | None = None,
        neutral: bool = False,
    ) -> float:
        """Probabilite de victoire de `home`."""
        adv = 0.0 if neutral else self.home_advantage
        diff = self.get(home, surface) + adv - self.get(away, surface)
        return elo_expected(diff)

    # ------------------------------------------------------------------
    def _mov_multiplier(self, margin: float, diff_winner: float) -> float:
        if not self.use_mov:
            return 1.0
        m = abs(margin) * self.mov_scale
        return float(np.log(max(m, 1.0) + 1.0) * (2.2 / (0.001 * diff_winner + 2.2)))

    def update(
        self,
        home: str,
        away: str,
        home_score: float,
        away_score: float,
        surface: str | None = None,
        neutral: bool = False,
        date: pd.Timestamp | None = None,
    ) -> dict:
        """Met a jour les notes apres un match. Retourne l'etat d'AVANT match."""
        adv = 0.0 if neutral else self.home_advantage
        r_h, r_a = self.ratings[home], self.ratings[away]
        exp_h_global = elo_expected(r_h + adv - r_a)
        pred = self.predict(home, away, surface, neutral)

        if home_score > away_score:
            s_h = 1.0
        elif home_score < away_score:
            s_h = 0.0
        else:
            s_h = 0.5

        margin = home_score - away_score
        diff_winner = (r_h + adv - r_a) if s_h == 1.0 else (r_a - r_h - adv)
        mult = self._mov_multiplier(margin, diff_winner)

        k_h, k_a = self._k(home) * mult, self._k(away) * mult
        delta = s_h - exp_h_global
        self.ratings[home] = r_h + k_h * delta
        self.ratings[away] = r_a - k_a * delta

        if surface is not None:
            sr_h = self.surface_ratings[(home, surface)]
            sr_a = self.surface_ratings[(away, surface)]
            exp_h_surf = elo_expected(sr_h + adv - sr_a)
            d_s = s_h - exp_h_surf
            self.surface_ratings[(home, surface)] = sr_h + k_h * d_s
            self.surface_ratings[(away, surface)] = sr_a - k_a * d_s

        self.n_games[home] += 1
        self.n_games[away] += 1
        if date is not None:
            self.last_date[home] = date
            self.last_date[away] = date

        return {"pred_home": pred, "elo_home": r_h, "elo_away": r_a, "k": k_h}

    # ------------------------------------------------------------------
    def fit_stream(
        self,
        df: pd.DataFrame,
        surface_col: str | None = None,
        record: bool = True,
    ) -> pd.DataFrame:
        """
        Parcours chronologique : pour chaque match on enregistre la prediction
        AVANT mise a jour, puis on met a jour. C'est la seule facon d'obtenir
        des predictions hors echantillon sans fuite temporelle.
        """
        d = df.sort_values("date").reset_index(drop=True)
        rows = []
        for r in d.itertuples(index=False):
            surface = getattr(r, surface_col) if surface_col else None
            neutral = bool(getattr(r, "neutral", False))
            info = self.update(
                r.home, r.away, r.home_score, r.away_score,
                surface=surface, neutral=neutral, date=r.date,
            )
            if record:
                rows.append(
                    {
                        "date": r.date,
                        "home": r.home,
                        "away": r.away,
                        "p_home_elo": info["pred_home"],
                        "elo_home_pre": info["elo_home"],
                        "elo_away_pre": info["elo_away"],
                        "home_score": r.home_score,
                        "away_score": r.away_score,
                    }
                )
        return pd.DataFrame(rows)

    def leaderboard(self, top: int = 25, min_games: int = 5) -> pd.DataFrame:
        rows = [
            {"nom": k, "elo": round(v, 1), "matchs": self.n_games[k]}
            for k, v in self.ratings.items()
            if self.n_games[k] >= min_games
        ]
        return (
            pd.DataFrame(rows).sort_values("elo", ascending=False).head(top).reset_index(drop=True)
            if rows
            else pd.DataFrame(columns=["nom", "elo", "matchs"])
        )
