"""
Modele lineaire de points marques : socle du basket et du rugby.

Contrairement au football, ou les scores sont petits et discrets (Poisson est
naturel), le basket et le rugby produisent des scores eleves dont la
distribution est proche d'une gaussienne. Le modele est donc :

    points_marques = mu + attaque[equipe] - defense[adversaire] + avantage_terrain

ajuste par moindres carres PONDERES (decroissance temporelle) avec penalite
ridge. Deux avantages pratiques sur une descente de gradient :

  - solution EXACTE en forme close (equations normales) : pas de convergence a
    surveiller, pas d'optimum local, quelques millisecondes meme sur 10 saisons ;
  - la matrice de covariance des residus donne directement l'ecart-type de la
    marge ET la correlation entre les scores des deux equipes, qui sont les
    deux quantites dont dependent tous les marches de handicap et de total.

L'ecart-type des residus n'est pas un detail : en NBA, la marge a un ecart-type
d'environ 11-13 points. Se tromper de 2 points dessus deplace la probabilite
d'un handicap -6.5 de plusieurs points de pourcentage, ce qui suffit a inventer
ou effacer un value bet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.scoredist import ScoreDistribution

__all__ = ["LinearScoreModel"]


class LinearScoreModel:
    """
    Parameters
    ----------
    half_life_days : demi-vie de la ponderation temporelle. 240 jours convient
        au basket (une saison et demie), 400 au rugby (moins de matchs).
    ridge : penalite. Plus elle est forte, plus les equipes peu vues sont
        ramenees vers la moyenne de la ligue.
    extra_features : colonnes numeriques additionnelles a inclure du point de
        vue de l'equipe qui marque (ex. 'rest_days', 'back_to_back').
    integer_scores : arrondir les scores simules (True pour le basket).
    """

    def __init__(
        self,
        half_life_days: float = 240.0,
        ridge: float = 3.0,
        extra_features: tuple[str, ...] = (),
        integer_scores: bool = True,
        n_sims: int = 60000,
        seed: int = 0,
    ):
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.extra_features = tuple(extra_features)
        self.integer_scores = integer_scores
        self.n_sims = n_sims
        self.seed = seed

        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.coef_: np.ndarray | None = None
        self.mu_: float = 0.0
        self.hfa_: float = 0.0
        self.sigma_: float = 12.0
        self.corr_: float = 0.0
        self.fitted_ = False

    # ------------------------------------------------------------------
    def _build(self, df: pd.DataFrame, weights: np.ndarray):
        """
        Chaque match donne DEUX lignes : une par equipe qui marque.
        Colonnes : [attaque (n), defense (n), avantage_terrain, extras...]
        """
        n = len(self.teams)
        m = len(df)
        n_extra = len(self.extra_features)
        p = 2 * n + 1 + n_extra

        X = np.zeros((2 * m, p))
        y = np.zeros(2 * m)
        w = np.zeros(2 * m)

        hi = df["home"].map(self._idx).to_numpy()
        ai = df["away"].map(self._idx).to_numpy()
        neutral = (
            df["neutral"].astype(bool).to_numpy() if "neutral" in df.columns else np.zeros(m, bool)
        )
        rows = np.arange(m)

        # lignes "domicile marque"
        X[rows, hi] = 1.0                     # attaque domicile
        X[rows, n + ai] = -1.0                # defense exterieur
        X[rows, 2 * n] = np.where(neutral, 0.0, 1.0)
        y[rows] = df["home_score"].to_numpy(dtype=float)
        w[rows] = weights

        # lignes "exterieur marque"
        X[m + rows, ai] = 1.0
        X[m + rows, n + hi] = -1.0
        X[m + rows, 2 * n] = 0.0
        y[m + rows] = df["away_score"].to_numpy(dtype=float)
        w[m + rows] = weights

        for j, feat in enumerate(self.extra_features):
            hcol = f"home_{feat}"
            acol = f"away_{feat}"
            if hcol in df.columns:
                X[rows, 2 * n + 1 + j] = pd.to_numeric(df[hcol], errors="coerce").fillna(0.0)
            if acol in df.columns:
                X[m + rows, 2 * n + 1 + j] = pd.to_numeric(df[acol], errors="coerce").fillna(0.0)

        return X, y, w

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "LinearScoreModel":
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        if as_of is not None:
            d = d[d["date"] < pd.Timestamp(as_of)]
        d = d.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        if len(d) < 30:
            raise ValueError(f"Historique insuffisant : {len(d)} matchs (minimum 30)")

        self.teams = sorted(set(d["home"]) | set(d["away"]))
        self._idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        ref = pd.Timestamp(as_of) if as_of is not None else d["date"].max()
        age = (ref - d["date"]).dt.total_seconds().to_numpy() / 86400.0
        lam = np.log(2.0) / max(self.half_life_days, 1.0)
        weights = np.exp(-lam * np.maximum(age, 0.0))

        X, y, w = self._build(d, weights)

        # centrage : on retire la moyenne ponderee pour que l'intercept
        # (mu) porte le niveau de scoring de la ligue
        self.mu_ = float(np.average(y, weights=w))
        y_c = y - self.mu_

        Xw = X * w[:, None]
        A = X.T @ Xw
        b = Xw.T @ y_c
        pen = np.eye(A.shape[0]) * self.ridge * w.sum() / len(w)
        pen[2 * n, 2 * n] = 1e-6          # pas de penalite sur l'avantage terrain
        for j in range(len(self.extra_features)):
            pen[2 * n + 1 + j, 2 * n + 1 + j] = 1e-6
        self.coef_ = np.linalg.solve(A + pen, b)
        self.hfa_ = float(self.coef_[2 * n])

        # residus : ecart-type et correlation entre les deux equipes d'un match
        pred = self.mu_ + X @ self.coef_
        resid = y - pred
        rh, ra = resid[: len(d)], resid[len(d) :]
        self.sigma_ = float(np.sqrt(np.average(resid**2, weights=w)))
        cov = float(np.average(rh * ra, weights=weights))
        self.corr_ = float(np.clip(cov / max(self.sigma_**2, 1e-9), -0.85, 0.85))
        self.fitted_ = True
        self.n_matches_ = len(d)
        return self

    # ------------------------------------------------------------------
    def expected_scores(
        self, home: str, away: str, neutral: bool = False, context: dict | None = None
    ) -> tuple[float, float]:
        if not self.fitted_:
            raise RuntimeError("Modele non entraine")
        n = len(self.teams)
        ih, ia = self._idx.get(home), self._idx.get(away)
        att_h = self.coef_[ih] if ih is not None else 0.0
        def_h = self.coef_[n + ih] if ih is not None else 0.0
        att_a = self.coef_[ia] if ia is not None else 0.0
        def_a = self.coef_[n + ia] if ia is not None else 0.0

        eh = self.mu_ + att_h - def_a + (0.0 if neutral else self.hfa_)
        ea = self.mu_ + att_a - def_h

        ctx = context or {}
        for j, feat in enumerate(self.extra_features):
            c = self.coef_[2 * n + 1 + j]
            eh += c * float(ctx.get(f"home_{feat}", 0.0))
            ea += c * float(ctx.get(f"away_{feat}", 0.0))

        eh += float(ctx.get("home_points_adj", 0.0))
        ea += float(ctx.get("away_points_adj", 0.0))
        mult = float(ctx.get("total_mult", 1.0))
        return max(eh * mult, 1.0), max(ea * mult, 1.0)

    def predict(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        context: dict | None = None,
        allow_draw: bool = False,
    ) -> ScoreDistribution:
        """
        Simule la distribution jointe des scores par tirage gaussien correle.

        allow_draw=False (basket) : les egalites sont resolues par prolongation,
        modelisee par un demi-point aleatoire ajoute au vainqueur du tirage.
        """
        eh, ea = self.expected_scores(home, away, neutral, context)
        ctx = context or {}
        sigma = self.sigma_ * float(ctx.get("sigma_mult", 1.0))

        rng = np.random.default_rng(self.seed)
        cov = np.array([[sigma**2, self.corr_ * sigma**2], [self.corr_ * sigma**2, sigma**2]])
        draws = rng.multivariate_normal([eh, ea], cov, size=self.n_sims)
        draws = np.maximum(draws, 0.0)
        if self.integer_scores:
            draws = np.rint(draws)

        if not allow_draw:
            tie = draws[:, 0] == draws[:, 1]
            if tie.any():
                bump = rng.random(tie.sum()) < 0.5
                draws[tie, 0] += np.where(bump, 1.0, 0.0)
                draws[tie, 1] += np.where(bump, 0.0, 1.0)

        return ScoreDistribution.from_samples(draws)

    # ------------------------------------------------------------------
    def ratings(self) -> pd.DataFrame:
        """
        ATTENTION A LA CONVENTION DE SIGNE, elle differe de celle du modele
        Dixon-Coles du football.

        Ici les points marques s'ecrivent  mu + attaque[soi] - defense[adverse],
        donc une valeur de `defense` ELEVEE signifie une BONNE defense (elle se
        retranche des points de l'adversaire). La note globale est donc la
        SOMME des deux, pas leur difference.

        Dans football_dc.py c'est l'inverse : lambda = exp(att + def_adverse),
        une defense elevee y signifie une defense qui encaisse beaucoup, et la
        note est bien une difference. Confondre les deux inverse le classement
        sans lever la moindre erreur.
        """
        n = len(self.teams)
        return (
            pd.DataFrame(
                {
                    "equipe": self.teams,
                    "attaque": self.coef_[:n],
                    "defense": self.coef_[n : 2 * n],
                    "note_nette": self.coef_[:n] + self.coef_[n : 2 * n],
                }
            )
            .sort_values("note_nette", ascending=False)
            .reset_index(drop=True)
        )

    def params_summary(self) -> dict:
        return {
            "n_equipes": len(self.teams),
            "n_matchs": getattr(self, "n_matches_", 0),
            "points_moyens": round(self.mu_, 2),
            "avantage_terrain_pts": round(self.hfa_, 2),
            "ecart_type_score": round(self.sigma_, 2),
            "correlation_scores": round(self.corr_, 3),
            "ecart_type_marge": round(float(self.sigma_ * np.sqrt(2 * (1 - self.corr_))), 2),
        }
