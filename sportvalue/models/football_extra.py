"""
Football : marches secondaires, derives des memes resultats que le 1X2.

Trois familles, trois modeles volontairement simples :

MI-TEMPS
    Chaque equipe a une part de ses buts marques en premiere periode (et une
    part des buts encaisses). En moyenne 44-46 % des buts tombent avant la
    pause, mais une equipe qui demarre lentement ou qui s'ecroule en fin de
    match s'en ecarte durablement. La part est retrecie vers la moyenne du
    championnat : l'effet est reel mais modeste, et une part estimee sur dix
    matchs ne vaut pas grand-chose seule. Les buts attendus du modele
    Dixon-Coles sont ensuite repartis entre les deux periodes, ce qui donne
    le resultat a la mi-temps, la periode la plus prolifique, les buts par
    periode et le double mi-temps / fin de match.

    Coherence : la table mi-temps / fin de match est recalee sur le 1X2 du
    modele principal. Le total des colonnes "fin de match" vaut EXACTEMENT
    le 1X2 affiche plus haut, jamais une autre estimation.

CORNERS, CARTONS
    Modele multiplicatif classique : moyenne du championnat (domicile ou
    exterieur) x facteur "pour" de l'equipe x facteur "contre" de
    l'adversaire, chaque facteur retreci vers 1. Pour les corners : ce que
    l'equipe obtient, ce qu'elle concede. Pour les cartons : ce qu'elle
    recoit (indiscipline), ce qu'elle fait recevoir (equipes qui provoquent
    les fautes). Distribution de Poisson sur chaque camp.

ARBITRE
    Facteur cartons de l'arbitre (jaunes par match rapportes a la moyenne,
    retreci vers 1), applique aux deux equipes quand l'arbitre du match est
    connu. Seule l'Angleterre fournit l'arbitre dans les resultats libres ;
    ailleurs, pas de facteur -- pas d'invention.

PENALTYS : PAS DE MODELE
    Aucune source libre ne donne les penaltys par equipe (football-data.co.uk
    ne les a pas, API-Football les verrouille hors plan gratuit). Un modele
    qui ne saurait produire que la moyenne du championnat (environ un match
    sur quatre) n'apporterait rien : on n'affiche pas ce qu'on ne sait pas.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.stats import poisson

from ..core.scoredist import ScoreDistribution

__all__ = ["HalfSplitModel", "RateModel", "RefereeEffect", "FootballExtras",
           "half_markets", "count_markets", "first_to_score"]

_N = 11          # taille des matrices par periode (0..10 buts)


def _weights(df: pd.DataFrame, xi: float) -> np.ndarray:
    """Poids exp(-xi * jours), memes conventions que le modele Dixon-Coles."""
    ref = df["date"].max()
    days = (ref - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    return np.exp(-xi * days)


# --------------------------------------------------------------------------
# Mi-temps
# --------------------------------------------------------------------------
class HalfSplitModel:
    """
    Part des buts marques en premiere periode, par equipe.

    k_shrink : nombre de buts "virtuels" a la moyenne du championnat ajoutes a
    chaque equipe. 40 = apres une saison (~50 buts), l'equipe pese un peu plus
    que le prior. L'effet equipe sur la repartition par periode est faible et
    bruite : un retrecissement fort est le bon choix.
    """

    def __init__(self, xi: float = 0.0035, k_shrink: float = 40.0):
        self.xi = xi
        self.k = k_shrink
        self.share_league_ = 0.45
        self.att_ = {}
        self.dfn_ = {}
        self.fitted_ = False

    def fit(self, hist: pd.DataFrame) -> "HalfSplitModel":
        if "ht_home" not in hist.columns:
            return self
        d = hist.dropna(subset=["ht_home", "ht_away", "home_score", "away_score"]).copy()
        if len(d) < 50:
            return self
        w = _weights(d, self.xi)
        ft = (d["home_score"] + d["away_score"]).to_numpy(float)
        ht = (d["ht_home"] + d["ht_away"]).to_numpy(float)
        self.share_league_ = float(np.sum(w * ht) / max(np.sum(w * ft), 1e-9))
        s = self.share_league_
        teams = set(d["home"]) | set(d["away"])
        for t in teams:
            mh = (d["home"] == t).to_numpy()
            ma = (d["away"] == t).to_numpy()
            # marques : a domicile colonne home, a l'exterieur colonne away
            g_for = np.sum(w[mh] * d["home_score"].to_numpy(float)[mh]) + \
                np.sum(w[ma] * d["away_score"].to_numpy(float)[ma])
            g1_for = np.sum(w[mh] * d["ht_home"].to_numpy(float)[mh]) + \
                np.sum(w[ma] * d["ht_away"].to_numpy(float)[ma])
            g_ag = np.sum(w[mh] * d["away_score"].to_numpy(float)[mh]) + \
                np.sum(w[ma] * d["home_score"].to_numpy(float)[ma])
            g1_ag = np.sum(w[mh] * d["ht_away"].to_numpy(float)[mh]) + \
                np.sum(w[ma] * d["ht_home"].to_numpy(float)[ma])
            self.att_[t] = float((g1_for + self.k * s) / (g_for + self.k))
            self.dfn_[t] = float((g1_ag + self.k * s) / (g_ag + self.k))
        self.fitted_ = True
        return self

    def split(self, home: str, away: str, lam_home: float, lam_away: float) -> dict | None:
        """Buts attendus par camp et par periode."""
        if not self.fitted_:
            return None
        s = self.share_league_
        sh = 0.5 * (self.att_.get(home, s) + self.dfn_.get(away, s))
        sa = 0.5 * (self.att_.get(away, s) + self.dfn_.get(home, s))
        sh = float(np.clip(sh, 0.25, 0.65))
        sa = float(np.clip(sa, 0.25, 0.65))
        return {
            "h1": lam_home * sh, "h2": lam_home * (1 - sh),
            "a1": lam_away * sa, "a2": lam_away * (1 - sa),
            "part_1ere_dom": sh, "part_1ere_ext": sa, "part_1ere_ligue": s,
        }


def _ou_rows(lam: float, lignes) -> list[dict]:
    rows = []
    for l in lignes:
        k = int(np.floor(l))
        under = float(poisson.cdf(k, lam))
        rows.append({"ligne": f"{l:g}", "over": round(1 - under, 4), "under": round(under, 4)})
    return rows


def _result_class(mat: np.ndarray) -> dict:
    i, j = np.indices(mat.shape)
    return {"1": float(mat[i > j].sum()), "X": float(mat[i == j].sum()), "2": float(mat[i < j].sum())}


def half_markets(sp: dict, sd: ScoreDistribution) -> dict:
    """
    Marches par periode a partir de la repartition `sp` (voir HalfSplitModel)
    et de la distribution fin de match `sd` du modele principal.
    """
    k = np.arange(_N)
    H1 = np.outer(poisson.pmf(k, sp["h1"]), poisson.pmf(k, sp["a1"]))
    H2 = np.outer(poisson.pmf(k, sp["h2"]), poisson.pmf(k, sp["a2"]))
    H1 /= H1.sum()
    H2 /= H2.sum()

    # periode la plus prolifique : totaux independants
    l1, l2 = sp["h1"] + sp["a1"], sp["h2"] + sp["a2"]
    g = np.arange(16)
    p1, p2 = poisson.pmf(g, l1), poisson.pmf(g, l2)
    joint = np.outer(p1, p2)
    i, j = np.indices(joint.shape)
    prolifique = {"1ere": float(joint[i > j].sum()), "egalite": float(joint[i == j].sum()),
                  "2eme": float(joint[i < j].sum())}

    # mi-temps / fin de match : convolution des deux periodes, puis recalage
    # des colonnes sur le 1X2 du modele principal (coherence garantie).
    ft_model = sd.market_1x2()
    table = {}
    ii, jj = np.indices(H1.shape)
    for r1, mask in (("1", ii > jj), ("X", ii == jj), ("2", ii < jj)):
        part = np.where(mask, H1, 0.0)
        ft = convolve2d(part, H2)            # distribution FT sachant la classe HT
        table[r1] = _result_class(ft)
    for r2 in ("1", "X", "2"):
        col = sum(table[r1][r2] for r1 in table)
        fac = ft_model[r2] / col if col > 1e-12 else 0.0
        for r1 in table:
            table[r1][r2] = round(table[r1][r2] * fac, 4)

    return {
        "buts_par_periode": {
            "dom_1ere": round(sp["h1"], 3), "dom_2eme": round(sp["h2"], 3),
            "ext_1ere": round(sp["a1"], 3), "ext_2eme": round(sp["a2"], 3),
            "part_1ere_ligue": round(sp["part_1ere_ligue"], 3),
        },
        "resultat_mi_temps": {kk: round(v, 4) for kk, v in _result_class(H1).items()},
        "mi_temps_prolifique": {kk: round(v, 4) for kk, v in prolifique.items()},
        "buts_1ere_mt": _ou_rows(l1, (0.5, 1.5, 2.5)),
        "buts_2eme_mt": _ou_rows(l2, (0.5, 1.5, 2.5)),
        "but_chaque_mt": round(float((1 - p1[0]) * (1 - p2[0])), 4),
        "mi_temps_fin": table,
    }


def first_to_score(sd: ScoreDistribution) -> dict:
    """
    Premiere equipe a marquer. Avec des buts en processus de Poisson, la
    probabilite que le premier but soit a domicile vaut lam/(lam+mu) ; le
    0-0 vient directement de la matrice principale.
    """
    lam, mu = sd.expected_home, sd.expected_away
    p00 = float(np.sum(sd.w[(sd.h == 0) & (sd.a == 0)]))
    tot = max(lam + mu, 1e-9)
    return {"domicile": round((1 - p00) * lam / tot, 4),
            "exterieur": round((1 - p00) * mu / tot, 4),
            "aucune": round(p00, 4)}


# --------------------------------------------------------------------------
# Corners, cartons
# --------------------------------------------------------------------------
class RateModel:
    """
    Taux multiplicatif pour une statistique de match comptee par camp.

    lam_dom = moy_dom x pour[dom] x contre[ext]
    lam_ext = moy_ext x pour[ext] x contre[dom]

    `pour` : ce que l'equipe produit (corners obtenus, cartons recus) ;
    `contre` : ce qu'elle laisse produire a l'adversaire. Chaque facteur est
    la moyenne ponderee des ratios observes, retrecie vers 1 avec k_shrink
    matchs virtuels.
    """

    def __init__(self, col_home: str, col_away: str, xi: float = 0.0035,
                 k_shrink: float = 10.0, min_rows: int = 50):
        self.col_home, self.col_away = col_home, col_away
        self.xi, self.k, self.min_rows = xi, k_shrink, min_rows
        self.avg_home_ = self.avg_away_ = None
        self.pour_, self.contre_ = {}, {}
        self.fitted_ = False

    def fit(self, hist: pd.DataFrame) -> "RateModel":
        if self.col_home not in hist.columns or self.col_away not in hist.columns:
            return self
        d = hist.dropna(subset=[self.col_home, self.col_away]).copy()
        if len(d) < self.min_rows:
            return self
        w = _weights(d, self.xi)
        xh = d[self.col_home].to_numpy(float)
        xa = d[self.col_away].to_numpy(float)
        self.avg_home_ = float(np.sum(w * xh) / np.sum(w))
        self.avg_away_ = float(np.sum(w * xa) / np.sum(w))
        if self.avg_home_ <= 0 or self.avg_away_ <= 0:
            return self
        rh, ra = xh / self.avg_home_, xa / self.avg_away_
        for t in set(d["home"]) | set(d["away"]):
            mh = (d["home"] == t).to_numpy()
            ma = (d["away"] == t).to_numpy()
            n = np.sum(w[mh]) + np.sum(w[ma])
            pour = np.sum(w[mh] * rh[mh]) + np.sum(w[ma] * ra[ma])
            contre = np.sum(w[mh] * ra[mh]) + np.sum(w[ma] * rh[ma])
            self.pour_[t] = float((pour + self.k) / (n + self.k))
            self.contre_[t] = float((contre + self.k) / (n + self.k))
        self.fitted_ = True
        return self

    def predict(self, home: str, away: str) -> tuple[float, float] | None:
        if not self.fitted_:
            return None
        lh = self.avg_home_ * self.pour_.get(home, 1.0) * self.contre_.get(away, 1.0)
        la = self.avg_away_ * self.pour_.get(away, 1.0) * self.contre_.get(home, 1.0)
        return float(lh), float(la)


def count_markets(lh: float, la: float, lignes_total, lignes_equipe) -> dict:
    """Totaux, par equipe et esperances pour un couple de Poisson independants."""
    return {
        "attendus": {"domicile": round(float(lh), 2), "exterieur": round(float(la), 2),
                     "total": round(float(lh + la), 2)},
        "total": _ou_rows(lh + la, lignes_total),
        "domicile": _ou_rows(lh, lignes_equipe),
        "exterieur": _ou_rows(la, lignes_equipe),
    }


# --------------------------------------------------------------------------
# Arbitre
# --------------------------------------------------------------------------
def _ref_key(name: str) -> str:
    """'M Oliver', 'M. Oliver', 'Michael Oliver' -> 'm oliver'."""
    s = re.sub(r"[.\-']", " ", str(name or "")).strip().lower()
    parts = [p for p in s.split() if p]
    if not parts:
        return ""
    return f"{parts[0][0]} {parts[-1]}"


class RefereeEffect:
    """Facteur cartons jaunes par arbitre, retreci vers 1 (k matchs virtuels)."""

    def __init__(self, xi: float = 0.0035, k_shrink: float = 20.0):
        self.xi, self.k = xi, k_shrink
        self.factor_, self.n_, self.avg_ = {}, {}, None
        self.fitted_ = False

    def fit(self, hist: pd.DataFrame) -> "RefereeEffect":
        need = {"referee", "yellow_home", "yellow_away"}
        if not need.issubset(hist.columns):
            return self
        d = hist.dropna(subset=["yellow_home", "yellow_away"]).copy()
        d = d[d["referee"].astype(str).str.len() > 1]
        if len(d) < 100:
            return self
        w = _weights(d, self.xi)
        tot = (d["yellow_home"] + d["yellow_away"]).to_numpy(float)
        self.avg_ = float(np.sum(w * tot) / np.sum(w))
        keys = d["referee"].map(_ref_key).to_numpy()
        for kref in set(keys):
            m = keys == kref
            n = float(np.sum(w[m]))
            ratio = np.sum(w[m] * tot[m] / self.avg_)
            self.factor_[kref] = float((ratio + self.k) / (n + self.k))
            self.n_[kref] = int(np.sum(m))
        self.fitted_ = True
        return self

    def lookup(self, name: str | None) -> dict | None:
        if not self.fitted_ or not name:
            return None
        k = _ref_key(name)
        if k not in self.factor_:
            return None
        return {"nom": name, "facteur_cartons": round(float(self.factor_[k]), 3),
                "matchs": self.n_[k], "moyenne_ligue": round(self.avg_, 2)}


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------
class FootballExtras:
    """Tous les modeles secondaires d'un championnat, ajustes une fois."""

    def __init__(self, xi: float = 0.0035):
        self.halves = HalfSplitModel(xi=xi)
        self.corners = RateModel("corners_home", "corners_away", xi=xi, k_shrink=10.0)
        self.yellows = RateModel("yellow_home", "yellow_away", xi=xi, k_shrink=10.0)
        # les rouges sont rares (~0,1 par equipe et par match) : retrecissement fort
        self.reds = RateModel("red_home", "red_away", xi=xi, k_shrink=40.0)
        self.referee = RefereeEffect(xi=xi)

    def fit(self, hist: pd.DataFrame) -> "FootballExtras":
        for m in (self.halves, self.corners, self.yellows, self.reds, self.referee):
            m.fit(hist)
        return self

    def periods(self, home: str, away: str, sd: ScoreDistribution) -> dict:
        """Marches qui dependent du score : mi-temps et premiere equipe a marquer."""
        out = {"premiere_equipe_a_marquer": first_to_score(sd)}
        sp = self.halves.split(home, away, sd.expected_home, sd.expected_away)
        if sp is not None:
            out.update(half_markets(sp, sd))
        return out

    def counts(self, home: str, away: str, arbitre: str | None = None) -> dict:
        """Marches independants du score : corners, cartons, arbitre."""
        out = {}
        c = self.corners.predict(home, away)
        if c is not None:
            out["corners"] = count_markets(c[0], c[1], (7.5, 8.5, 9.5, 10.5, 11.5, 12.5),
                                           (3.5, 4.5, 5.5, 6.5))
        y = self.yellows.predict(home, away)
        if y is not None:
            ref = self.referee.lookup(arbitre)
            fac = ref["facteur_cartons"] if ref else 1.0
            out["cartons"] = count_markets(y[0] * fac, y[1] * fac,
                                           (2.5, 3.5, 4.5, 5.5, 6.5), (1.5, 2.5, 3.5))
            if ref:
                out["cartons"]["arbitre"] = ref
        r = self.reds.predict(home, away)
        if r is not None:
            lh, la = r
            out["carton_rouge"] = {
                "match": round(1 - float(np.exp(-(lh + la))), 4),
                "domicile": round(1 - float(np.exp(-lh)), 4),
                "exterieur": round(1 - float(np.exp(-la)), 4),
            }
        return out
