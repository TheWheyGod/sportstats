"""
Calibration des probabilites et fusion modele / marche.

Deux operations distinctes, souvent confondues :

1. CALIBRATION : un modele peut bien classer (bon AUC) tout en sortant des
   probabilites fausses. Si tous les matchs annonces a 70% sortent a 62%, chaque
   value bet detecte est un faux positif. On corrige avec une isotonique ou une
   regression logistique (Platt) apprise sur un jeu de validation SEPARE.

2. BLENDING : la ligne du marche contient une information qu'aucun modele
   maison ne reproduit (compos de derniere minute, argent informe, blessures non
   publiques). On fusionne donc en log-odds :

       logit(p_final) = w * logit(p_modele) + (1-w) * logit(p_marche_fair)

   w est appris par minimisation de la log-loss. Un w faible n'est pas un echec :
   il signifie que l'edge du modele est concentre la ou il s'ecarte du marche,
   ce qui est precisement ce qu'on cherche a isoler.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

__all__ = [
    "logit",
    "expit",
    "blend_logit",
    "fit_blend_weight",
    "ProbabilityCalibrator",
    "reliability_table",
    "shrink_to_market",
]

_EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------
# Blending modele <-> marche
# --------------------------------------------------------------------------
def blend_logit(p_model, p_market, w: float) -> np.ndarray:
    """
    Fusion en log-odds. w=1 -> modele pur, w=0 -> marche pur.

    Pour les marches multi-issues (1X2), la fusion se fait composante par
    composante puis on renormalise a 1.
    """
    pm = np.asarray(p_model, dtype=float)
    pk = np.asarray(p_market, dtype=float)
    blended = expit(w * logit(pm) + (1.0 - w) * logit(pk))
    if blended.ndim >= 1 and blended.shape[-1] > 1:
        s = blended.sum(axis=-1, keepdims=True)
        blended = blended / np.where(s == 0, 1.0, s)
    return blended


def fit_blend_weight(
    p_model: np.ndarray,
    p_market: np.ndarray,
    outcomes: np.ndarray,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> dict:
    """
    Trouve le poids w qui minimise la log-loss du melange sur donnees passees.

    p_model, p_market : (n, k) probabilites par issue
    outcomes          : (n,) index de l'issue realisee (0..k-1)

    Retourne w et les log-loss comparees (modele seul / marche seul / melange).
    Si le melange ne bat pas le marche seul, le modele n'apporte rien : il faut
    le savoir avant de miser, pas apres.
    """
    pm = np.atleast_2d(np.asarray(p_model, dtype=float))
    pk = np.atleast_2d(np.asarray(p_market, dtype=float))
    y = np.asarray(outcomes, dtype=int)
    rows = np.arange(len(y))

    def ll(w):
        b = blend_logit(pm, pk, w)
        return float(-np.mean(np.log(np.clip(b[rows, y], _EPS, 1.0))))

    res = minimize_scalar(ll, bounds=bounds, method="bounded")
    w = float(res.x)
    return {
        "w": w,
        "logloss_blend": ll(w),
        "logloss_model": ll(1.0),
        "logloss_market": ll(0.0),
        "gain_vs_market": ll(0.0) - ll(w),
    }


def shrink_to_market(p_model, p_market, max_deviation: float = 0.15, n_iter: int = 40):
    """
    Garde-fou : borne l'ecart absolu entre modele et marche.

    Quand un modele annonce 45% la ou le marche fair est a 20%, dans 95% des cas
    c'est un bug de donnees (mauvais mapping d'equipe, match reporte, ligne sur
    un autre marche), pas une opportunite. On plafonne l'ecart pour que ce type
    d'anomalie ne genere pas une mise maximale.

    SUBTILITE : sur un marche multi-issues, clipper puis renormaliser ne suffit
    PAS. La renormalisation peut faire ressortir une composante de l'intervalle
    qu'on venait de lui imposer. Exemple reel : modele [0.90, 0.05, 0.05] contre
    marche [0.40, 0.30, 0.30] avec une borne de 0.15 donne, apres un seul
    passage, 0.647 pour la premiere issue -- soit un ecart de 0.247, largement
    au-dela de la borne annoncee.

    On alterne donc clip et renormalisation jusqu'a convergence. La suite est
    contractante et converge en quelques iterations vers un point qui respecte
    reellement les deux contraintes.
    """
    pm = np.asarray(p_model, dtype=float)
    pk = np.asarray(p_market, dtype=float)
    lo = np.clip(pk - max_deviation, _EPS, 1 - _EPS)
    hi = np.clip(pk + max_deviation, _EPS, 1 - _EPS)

    out = np.clip(pm, lo, hi)
    if out.ndim == 0 or out.shape[-1] <= 1:
        return out

    for _ in range(n_iter):
        s = out.sum(axis=-1, keepdims=True)
        out = out / np.where(s == 0, 1.0, s)
        clipped = np.clip(out, lo, hi)
        if np.max(np.abs(clipped - out)) < 1e-12:
            return clipped
        out = clipped
    # Si la borne est trop serree pour qu'une distribution normalisee existe,
    # on renvoie le dernier iterate renormalise : la contrainte de somme a 1
    # prime, sinon les probabilites ne seraient plus interpretables.
    s = out.sum(axis=-1, keepdims=True)
    return out / np.where(s == 0, 1.0, s)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
class ProbabilityCalibrator:
    """
    Calibrateur binaire (une issue a la fois), applicable issue par issue sur
    un marche multi-way puis renormalise.

    method : 'isotonic' (souple, exige >= ~1000 obs) ou 'platt' (2 parametres,
             robuste sur petits echantillons).
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt", "none"):
            raise ValueError("method doit etre isotonic, platt ou none")
        self.method = method
        self._model = None
        self.n_fit = 0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p = np.clip(np.asarray(p_raw, dtype=float).ravel(), _EPS, 1 - _EPS)
        y = np.asarray(y, dtype=int).ravel()
        self.n_fit = len(y)
        if self.method == "none" or self.n_fit < 50 or len(np.unique(y)) < 2:
            self._model = None
            return self
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p, y)
        else:
            self._model = LogisticRegression(C=1e6, solver="lbfgs")
            self._model.fit(logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p_raw):
        p = np.clip(np.asarray(p_raw, dtype=float), _EPS, 1 - _EPS)
        if self._model is None:
            return p
        shape = p.shape
        flat = p.ravel()
        if self.method == "isotonic":
            out = self._model.predict(flat)
        else:
            out = self._model.predict_proba(logit(flat).reshape(-1, 1))[:, 1]
        return np.clip(out, _EPS, 1 - _EPS).reshape(shape)

    def fit_transform(self, p_raw, y):
        return self.fit(p_raw, y).transform(p_raw)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """
    Table de fiabilite : pour chaque tranche de probabilite predite, compare la
    frequence observee. C'est le diagnostic numero un a lire avant de miser.

    Un modele utilisable pour du value betting doit avoir |predit - observe|
    inferieur a ~2 points sur les tranches ou l'on parie.
    """
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred = float(p[mask].mean())
        obs = float(y[mask].mean())
        se = float(np.sqrt(max(obs * (1 - obs), 1e-6) / n))
        rows.append(
            {
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n": n,
                "predit": pred,
                "observe": obs,
                "ecart": obs - pred,
                "ecart_en_se": (obs - pred) / se if se > 0 else 0.0,
            }
        )
    return rows
