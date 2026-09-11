"""
Football : modele Dixon-Coles (Poisson bivarie corrige).

Poisson simple sur chaque equipe sous-estime les scores nuls serres (0-0, 1-1)
et surestime les 1-0/0-1 : les buts ne sont pas independants. Dixon & Coles
(1997) corrigent les quatre cellules basses par un parametre rho, ce qui suffit
a rendre le marche "nul" et le "under 2.5" exploitables.

Trois ajouts par rapport a l'article d'origine :

  - decroissance temporelle exponentielle (poids = exp(-xi * jours)) : une
    victoire d'il y a deux ans ne dit presque rien de la forme actuelle ;
  - regularisation ridge sur les forces d'attaque/defense : sans elle, un promu
    avec 3 matchs joues recoit une note aberrante et genere de faux value bets ;
  - ajustements contextuels multiplicatifs (absences, repos, enjeu) appliques
    aux lambdas, alimentes par la couche donnees.

Le modele produit une matrice de scores complete, donc TOUS les marches
derivent d'un objet coherent : impossible d'avoir un 1X2 et un O/U 2.5 qui se
contredisent, ce qui arrive avec des modeles separes par marche.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ..core.scoredist import ScoreDistribution

__all__ = ["DixonColesModel"]

_EPS = 1e-10


class DixonColesModel:
    """
    Parameters
    ----------
    xi : taux de decroissance temporelle par jour. 0.0035 => un match d'il y a
         200 jours pese la moitie d'un match d'hier. 0 desactive la decroissance.
    ridge : force du retrecissement vers l'equipe moyenne du championnat.
         Defaut 20.0, choisi par VALIDATION HORS ECHANTILLON (walk-forward,
         900 matchs de test sur 5 championnats) : la log-loss passe de 1.00677
         a ridge=0.02 a 1.00309 a ridge=20, et remonte a 1.00391 a ridge=40.
         C'est donc un optimum interieur, pas une borne.

         Une valeur faible parait inoffensive ; elle ne l'est pas. Une equipe
         promue ayant encaisse 0 but en 3 matchs pousse sa note de defense vers
         l'infini (probleme de SEPARATION : la vraisemblance n'a pas d'optimum
         fini). Observe reellement : Chelsea donne a 5% de battre Hull chez lui,
         la ou le marche disait 79%.
    max_goals : taille de la matrice de scores (10 suffit, 15 pour les ligues
         a fort volume de buts).
    """

    def __init__(
        self,
        xi: float = 0.0035,
        ridge: float = 20.0,
        max_goals: int = 12,
        rho_bounds: tuple[float, float] = (-0.25, 0.25),
    ):
        self.xi = xi
        self.ridge = ridge
        self.max_goals = max_goals
        self.rho_bounds = rho_bounds

        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_adv: float = 0.25
        self.rho: float = -0.05
        self.fitted_: bool = False
        self.n_matches_: int = 0
        self.league_avg_goals_: float = 2.6

    # ------------------------------------------------------------------
    # Vraisemblance
    # ------------------------------------------------------------------
    @staticmethod
    def _tau(x, y, lam, mu, rho):
        """Correction Dixon-Coles des quatre scores bas."""
        t = np.ones_like(lam, dtype=float)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        t[m00] = 1.0 - lam[m00] * mu[m00] * rho
        t[m01] = 1.0 + lam[m01] * rho
        t[m10] = 1.0 + mu[m10] * rho
        t[m11] = 1.0 - rho
        return np.maximum(t, _EPS)

    def _unpack(self, params):
        """
        Parametrage :  log(lambda) = mu + attaque[soi] - defense[adverse] + gamma

        Deux choix cruciaux, chacun corrigeant un bug observe en production :

        1. UN INTERCEPT EXPLICITE `mu` porte le niveau de scoring du
           championnat. Sans lui, c'est la MOYENNE DES DEFENSES qui l'absorbe,
           et le ridge -- qui retrecit vers zero -- pousse alors chaque equipe
           vers "encaisse exp(0)" au lieu de "encaisse comme la moyenne".
           Symptome : plus on regularisait, MOINS le modele prevoyait de buts.

        2. LA DEFENSE ENTRE AVEC UN SIGNE MOINS : une valeur ELEVEE signifie
           une BONNE defense. C'est la convention de LinearScoreModel
           (basket/rugby), donc les deux modeles se lisent enfin pareil et
           `note_globale` est une somme dans les deux cas.

        Attaque ET defense sont centrees a moyenne nulle, si bien que le ridge
        retrecit vers l'equipe moyenne du championnat -- le seul prior
        defendable pour un promu dont on ne sait presque rien.
        """
        n = len(self.teams)
        att = params[:n]
        att = att - att.mean()
        dfn = params[n : 2 * n]
        dfn = dfn - dfn.mean()
        return att, dfn, params[2 * n], params[2 * n + 1], params[2 * n + 2]

    def _neg_ll(self, params, hi, ai, hg, ag, w, is_home):
        att, dfn, gamma, rho, intercept = self._unpack(params)
        log_lam = intercept + att[hi] - dfn[ai] + gamma * is_home
        log_mu = intercept + att[ai] - dfn[hi]
        lam = np.exp(np.clip(log_lam, -6, 3))
        mu = np.exp(np.clip(log_mu, -6, 3))

        ll = (
            hg * np.log(lam) - lam
            + ag * np.log(mu) - mu
            + np.log(self._tau(hg, ag, lam, mu, rho))
        )
        nll = -float(np.sum(w * ll))
        nll += self.ridge * float(np.sum(att**2) + np.sum(dfn**2))
        return nll

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        """
        df : colonnes date, home, away, home_score, away_score [, neutral]
        as_of : date de reference pour la decroissance. TOUT match posterieur
                est ecarte -- c'est le garde-fou anti-fuite du backtest.
        """
        need = {"date", "home", "away", "home_score", "away_score"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        if as_of is not None:
            d = d[d["date"] < pd.Timestamp(as_of)]
        d = d.dropna(subset=["home_score", "away_score"])
        if len(d) < 20:
            raise ValueError(f"Historique insuffisant : {len(d)} matchs (minimum 20)")

        self.teams = sorted(set(d["home"]) | set(d["away"]))
        self._idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.n_matches_ = len(d)

        hi = d["home"].map(self._idx).to_numpy()
        ai = d["away"].map(self._idx).to_numpy()
        hg = d["home_score"].to_numpy(dtype=float)
        ag = d["away_score"].to_numpy(dtype=float)
        is_home = (
            (~d["neutral"].astype(bool)).to_numpy(dtype=float)
            if "neutral" in d.columns
            else np.ones(len(d))
        )

        ref = pd.Timestamp(as_of) if as_of is not None else d["date"].max()
        age_days = (ref - d["date"]).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-self.xi * np.maximum(age_days, 0.0))
        w = w / w.mean()

        self.league_avg_goals_ = float(np.average(hg + ag, weights=w))

        # L'intercept demarre au niveau de scoring observe : l'optimiseur
        # part ainsi d'un point deja plausible plutot que de 1 but par equipe.
        lam0 = float(np.log(max(np.average(hg, weights=w), 0.2)))
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05], [lam0]])
        bounds = ([(-2.5, 2.5)] * n + [(-2.5, 2.5)] * n
                  + [(-0.5, 1.2), self.rho_bounds, (-1.5, 1.5)])
        res = minimize(
            self._neg_ll,
            x0,
            args=(hi, ai, hg, ag, w, is_home),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        (self.attack, self.defence, self.home_adv, self.rho,
         self.intercept) = self._unpack(res.x)
        self.fitted_ = True
        self.opt_success_ = bool(res.success)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _team_lambda(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Lambdas bruts. Une equipe inconnue est traitee comme moyenne."""
        ih, ia = self._idx.get(home), self._idx.get(away)
        att_h = self.attack[ih] if ih is not None else 0.0
        def_h = self.defence[ih] if ih is not None else 0.0
        att_a = self.attack[ia] if ia is not None else 0.0
        def_a = self.defence[ia] if ia is not None else 0.0
        gamma = 0.0 if neutral else self.home_adv
        # defense soustraite : une note elevee = bonne defense
        return (float(np.exp(self.intercept + att_h - def_a + gamma)),
                float(np.exp(self.intercept + att_a - def_h)))

    def predict(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        context: dict | None = None,
    ) -> ScoreDistribution:
        """
        Retourne la distribution jointe des scores.

        context (optionnel) accepte des multiplicateurs deja calcules par la
        couche donnees :
            home_attack_mult, away_attack_mult   (absences offensives)
            home_defence_mult, away_defence_mult (absences defensives)
            total_mult                           (meteo, enjeu, arbitre)
        Un multiplicateur de defense > 1 signifie une defense AFFAIBLIE, donc
        il augmente les buts encaisses.
        """
        if not self.fitted_:
            raise RuntimeError("Modele non entraine : appeler fit() d'abord")

        lam, mu = self._team_lambda(home, away, neutral)
        ctx = context or {}
        lam *= ctx.get("home_attack_mult", 1.0) * ctx.get("away_defence_mult", 1.0)
        mu *= ctx.get("away_attack_mult", 1.0) * ctx.get("home_defence_mult", 1.0)
        tm = ctx.get("total_mult", 1.0)
        lam *= tm
        mu *= tm

        lam = float(np.clip(lam, 0.05, 8.0))
        mu = float(np.clip(mu, 0.05, 8.0))

        k = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))

        # correction Dixon-Coles sur les 4 cellules basses
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.maximum(mat, 0.0)
        mat /= mat.sum()

        return ScoreDistribution.from_matrix(mat)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def ratings(self) -> pd.DataFrame:
        """
        Classement des forces. `attaque` et `defense` sont en log :
        +0.20 d'attaque = environ +22% de buts marques attendus.
        `note_globale` = attaque - defense (plus haut = meilleur).
        """
        if not self.fitted_:
            raise RuntimeError("Modele non entraine")
        return (
            pd.DataFrame(
                {
                    "equipe": self.teams,
                    "attaque": self.attack,
                    "defense": self.defence,
                    # Somme, pas difference : une defense elevee est BONNE.
                    "note_globale": self.attack + self.defence,
                }
            )
            .sort_values("note_globale", ascending=False)
            .reset_index(drop=True)
        )

    def params_summary(self) -> dict:
        return {
            "n_equipes": len(self.teams),
            "n_matchs": self.n_matches_,
            "avantage_domicile_log": round(float(self.home_adv), 4),
            "avantage_domicile_x": round(float(np.exp(self.home_adv)), 3),
            "rho": round(float(self.rho), 4),
            "intercept_log": round(float(self.intercept), 4),
            "xi": self.xi,
            "buts_moyens_ligue": round(self.league_avg_goals_, 3),
            "convergence": getattr(self, "opt_success_", None),
        }

    @staticmethod
    def tune_xi(
        df: pd.DataFrame,
        candidates=(0.0, 0.0015, 0.0025, 0.0035, 0.005, 0.0075),
        n_folds: int = 4,
        min_train: int = 400,
        ridge: float = 20.0,
    ) -> dict:
        """
        Choisit xi par validation temporelle glissante (jamais aleatoire :
        melanger les dates fait fuiter le futur dans l'entrainement et donne
        des xi trop faibles, donc un modele trop lent a reagir a la forme).
        """
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)
        n = len(d)
        if n < min_train + 100:
            return {"best_xi": 0.0035, "scores": {}, "note": "historique trop court"}

        cuts = np.linspace(min_train, n - 50, n_folds + 1).astype(int)[:-1]
        scores: dict[float, float] = {}
        for xi in candidates:
            lls = []
            for cut in cuts:
                train, test = d.iloc[:cut], d.iloc[cut : cut + 60]
                if len(test) < 10:
                    continue
                try:
                    m = DixonColesModel(xi=xi, ridge=ridge).fit(train, as_of=test["date"].iloc[0])
                except (ValueError, RuntimeError):
                    continue
                for _, row in test.iterrows():
                    sd = m.predict(row["home"], row["away"])
                    p = sd.market_1x2()
                    hs, as_ = row["home_score"], row["away_score"]
                    key = "1" if hs > as_ else ("X" if hs == as_ else "2")
                    lls.append(-np.log(max(p[key], 1e-9)))
            if lls:
                scores[xi] = float(np.mean(lls))
        if not scores:
            return {"best_xi": 0.0035, "scores": {}, "note": "aucun pli exploitable"}
        best = min(scores, key=scores.get)
        return {"best_xi": best, "scores": scores, "logloss": scores[best]}
