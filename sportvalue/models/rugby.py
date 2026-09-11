"""
Rugby : modele de points composes, sensible a la meteo.

Pourquoi ne pas se contenter d'une gaussienne comme au basket : le rugby marque
par paliers de 3 (penalite, drop), 5 (essai) et 7 (essai transforme), donc la
distribution des ecarts est DISCRETE et non continue.

Mesure faite sur ce modele (match a ~52 points au total) : le surcroit de masse
sur les ecarts "ronds" est modeste, P(marge=3)/P(marge=4) vaut environ 1.04 --
la convolution de 3 a 5 evenements par equipe lisse largement les paliers. Ne
pas surestimer cet effet.

Ce qui est en revanche materiel et verifie :

  - sur un handicap a ligne ENTIERE (-3, -7), la probabilite de remboursement
    (push) mesuree ici vaut ~1.7 a 2.0%. Un modele gaussien continu la fixe a
    ZERO et surestime donc mecaniquement la probabilite de couvrir. Sur une
    cote a 1.90, ignorer 2% de push deplace l'esperance d'environ 2 points,
    soit l'ordre de grandeur d'un edge entier ;
  - les marches "ecart de victoire par tranche" (1-12, 13+) decoulent
    directement de la distribution simulee, la ou beaucoup de books les cotent
    par regle empirique.

D'ou la simulation explicite des evenements marquants :
    essais ~ Poisson, transformations ~ Binomiale(essais, taux),
    penalites ~ Poisson, drops ~ Poisson.

METEO : c'est le sport ou elle compte le plus. Le vent degrade la reussite au
pied (donc les points de penalite ET les transformations), la pluie degrade le
jeu de mains (donc les essais) et augmente le nombre de fautes. Les coefficients
par defaut ci-dessous sont des PRIORS explicites, pas des valeurs estimees sur
donnees : ils sont volontairement modestes et entierement parametrables. Si vous
disposez d'un historique avec meteo, `calibrate_weather` les reestime.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..core.scoredist import ScoreDistribution
from .linear_score import LinearScoreModel

__all__ = ["WeatherEffect", "RugbyModel", "DEFAULT_WEATHER"]


@dataclass
class WeatherEffect:
    """
    Coefficients d'impact meteo. Tous exprimes en multiplicateurs ou en points
    de pourcentage, et tous documentes pour pouvoir etre discutes et modifies.

    vent_seuil_kmh      : en dessous, aucun effet.
    vent_kick_par_10kmh : perte absolue de reussite au pied par tranche de
                          10 km/h au-dessus du seuil (0.04 = -4 points de %).
    vent_essais_par_10kmh : multiplicateur sur le taux d'essais par tranche.
    pluie_legere_mm     : seuil de pluie legere (mm/h).
    pluie_forte_mm      : seuil de pluie forte.
    pluie_legere_essais / pluie_forte_essais : multiplicateurs du taux d'essais.
    pluie_penalites     : multiplicateur du taux de penalites (le jeu se ferme,
                          davantage de fautes au sol).
    froid_seuil_c / froid_essais : effet du froid sur le jeu de mains.
    """

    vent_seuil_kmh: float = 20.0
    vent_kick_par_10kmh: float = 0.045
    vent_essais_par_10kmh: float = 0.955
    pluie_legere_mm: float = 0.5
    pluie_forte_mm: float = 3.0
    pluie_legere_essais: float = 0.93
    pluie_forte_essais: float = 0.84
    pluie_kick: float = 0.975
    pluie_penalites: float = 1.06
    froid_seuil_c: float = 5.0
    froid_essais: float = 0.97

    def apply(self, weather: dict | None) -> dict:
        """
        weather : {'wind_kmh': float, 'precip_mm': float, 'temp_c': float}
        Retourne {'try_mult', 'kick_delta', 'pen_mult', 'notes': [...]}.
        """
        out = {"try_mult": 1.0, "kick_delta": 0.0, "pen_mult": 1.0, "notes": []}
        if not weather:
            return out

        wind = float(weather.get("wind_kmh", 0.0) or 0.0)
        rain = float(weather.get("precip_mm", 0.0) or 0.0)
        temp = weather.get("temp_c", None)

        if wind > self.vent_seuil_kmh:
            steps = (wind - self.vent_seuil_kmh) / 10.0
            out["kick_delta"] -= self.vent_kick_par_10kmh * steps
            out["try_mult"] *= self.vent_essais_par_10kmh**steps
            out["notes"].append(f"vent {wind:.0f} km/h : reussite au pied et essais en baisse")

        if rain >= self.pluie_forte_mm:
            out["try_mult"] *= self.pluie_forte_essais
            out["kick_delta"] -= 1.0 - self.pluie_kick
            out["pen_mult"] *= self.pluie_penalites
            out["notes"].append(f"pluie forte {rain:.1f} mm/h : jeu ferme, moins d'essais")
        elif rain >= self.pluie_legere_mm:
            out["try_mult"] *= self.pluie_legere_essais
            out["pen_mult"] *= 1.0 + (self.pluie_penalites - 1.0) * 0.5
            out["notes"].append(f"pluie legere {rain:.1f} mm/h : essais en leger retrait")

        if temp is not None and float(temp) < self.froid_seuil_c:
            out["try_mult"] *= self.froid_essais
            out["notes"].append(f"froid {float(temp):.0f} C : jeu de mains degrade")

        return out

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_WEATHER = WeatherEffect()


class RugbyModel:
    """
    Parameters
    ----------
    part_essais : part des points venant des essais + transformations dans le
        total. ~0.58 en Top 14 / Premiership, ~0.62 en Super Rugby (jeu plus
        ouvert), ~0.55 en Coupe du monde (matchs fermes).
    taux_transformation : reussite des transformations par temps sec (~0.72
        au niveau international, un peu moins en club).
    part_drops : part des points venant des drops (faible mais non nulle,
        et elle monte dans les matchs a enjeu et par mauvais temps).
    """

    def __init__(
        self,
        part_essais: float = 0.58,
        taux_transformation: float = 0.72,
        part_drops: float = 0.03,
        half_life_days: float = 150.0,
        ridge: float = 4.0,
        weather_effect: WeatherEffect | None = None,
        n_sims: int = 60000,
        seed: int = 0,
        correction_biais: float = 0.0,
    ):
        self.part_essais = part_essais
        self.taux_transformation = taux_transformation
        self.part_drops = part_drops
        self.weather = weather_effect or DEFAULT_WEATHER
        self.n_sims = n_sims
        self.seed = seed
        self.correction_biais = correction_biais
        self.base = LinearScoreModel(
            half_life_days=half_life_days, ridge=ridge, integer_scores=False
        )
        self.fitted_ = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "RugbyModel":
        self.base.fit(df, as_of=as_of)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    def _event_rates(self, expected_points: float, adj: dict) -> dict:
        """Convertit une esperance de points en taux d'evenements marquants."""
        pts_essais = expected_points * self.part_essais
        pts_drops = expected_points * self.part_drops
        pts_pen = max(expected_points - pts_essais - pts_drops, 0.0)

        kick = float(np.clip(self.taux_transformation + adj["kick_delta"], 0.35, 0.95))
        pts_par_essai = 5.0 + 2.0 * kick
        lam_try = max(pts_essais / pts_par_essai, 0.01) * adj["try_mult"]
        lam_pen = max(pts_pen / 3.0, 0.01) * adj["pen_mult"]
        lam_drop = max(pts_drops / 3.0, 0.0)
        return {"lam_try": lam_try, "lam_pen": lam_pen, "lam_drop": lam_drop, "kick": kick}

    def predict(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        context: dict | None = None,
    ) -> ScoreDistribution:
        """
        context accepte :
            weather = {'wind_kmh','precip_mm','temp_c'}
            home_points_adj / away_points_adj : ajustement direct en points
                (ex. -4 pour l'absence d'un ouvreur buteur titulaire)
        """
        if not self.fitted_:
            raise RuntimeError("Modele non entraine")
        ctx = dict(context or {})
        adj = self.weather.apply(ctx.get("weather"))

        eh, ea = self.base.expected_scores(home, away, neutral, ctx)
        # Correction de biais, repartie a parts egales sur les deux equipes.
        # Un modele retrospectif RETARDE toujours sur une tendance de scoring
        # qui monte : mesure hors echantillon, le biais reste negatif meme avec
        # une demi-vie tres courte (-0.36 pt a 60 jours en NRL). Raccourcir
        # encore la demi-vie ne fait qu'echanger du biais contre de la variance.
        # Mieux vaut donc corriger explicitement l'ecart mesure.
        if self.correction_biais:
            eh += self.correction_biais / 2.0
            ea += self.correction_biais / 2.0
        rh = self._event_rates(eh, adj)
        ra = self._event_rates(ea, adj)

        rng = np.random.default_rng(self.seed)
        n = self.n_sims

        def simulate(r):
            tries = rng.poisson(r["lam_try"], n)
            conv = rng.binomial(tries, r["kick"])
            pens = rng.poisson(r["lam_pen"], n)
            drops = rng.poisson(r["lam_drop"], n) if r["lam_drop"] > 0 else np.zeros(n, int)
            pts = 5 * tries + 2 * conv + 3 * pens + 3 * drops
            return pts, tries

        pts_h, tries_h = simulate(rh)
        pts_a, tries_a = simulate(ra)
        samples = np.column_stack([pts_h, pts_a]).astype(float)
        sd = ScoreDistribution.from_samples(samples)
        sd.weather_notes = adj["notes"]
        sd.event_rates = {"domicile": rh, "exterieur": ra}
        # Distribution des ESSAIS, issue des MEMES tirages que les points.
        # Le total d'essais et le total de points ne peuvent donc pas se
        # contredire : un scenario a 6 essais porte le meme poids dans les
        # deux marches.
        sd.tries = ScoreDistribution.from_samples(
            np.column_stack([tries_h, tries_a]).astype(float)
        )
        sd.expected_tries = (float(tries_h.mean()), float(tries_a.mean()))
        return sd

    # ------------------------------------------------------------------
    def weather_impact_report(self, home: str, away: str, weather: dict, neutral: bool = False) -> dict:
        """
        Compare la prediction avec et sans meteo. C'est le diagnostic a lire
        avant de miser un total : si l'ecart est inferieur a ~1.5 point, la
        meteo n'est pas un argument suffisant pour aller contre la ligne.
        """
        sec = self.predict(home, away, neutral, {"weather": None})
        reel = self.predict(home, away, neutral, {"weather": weather})
        adj = self.weather.apply(weather)
        return {
            "total_sans_meteo": round(sec.expected_total, 2),
            "total_avec_meteo": round(reel.expected_total, 2),
            "delta_points": round(reel.expected_total - sec.expected_total, 2),
            "marge_sans_meteo": round(sec.expected_margin, 2),
            "marge_avec_meteo": round(reel.expected_margin, 2),
            "multiplicateur_essais": round(adj["try_mult"], 3),
            "delta_reussite_pied": round(adj["kick_delta"], 3),
            "notes": adj["notes"],
        }

    def params_summary(self) -> dict:
        s = self.base.params_summary()
        s.update(
            {
                "part_essais": self.part_essais,
                "taux_transformation": self.taux_transformation,
                "meteo": self.weather.to_dict(),
            }
        )
        return s


def calibrate_weather(
    df: pd.DataFrame,
    wind_col: str = "wind_kmh",
    rain_col: str = "precip_mm",
) -> dict:
    """
    Reestime l'effet meteo sur le TOTAL de points par regression simple, quand
    l'historique contient la meteo.

    A n'utiliser qu'avec plusieurs centaines de matchs : l'effet reel est de
    l'ordre de quelques points, largement noye dans une variance d'environ
    20 points par match. Retourne aussi l'erreur-type pour juger si le
    coefficient est distinguable de zero.
    """
    d = df.dropna(subset=[wind_col, rain_col, "home_score", "away_score"]).copy()
    if len(d) < 100:
        return {"note": f"trop peu de matchs avec meteo ({len(d)}), calibration non fiable"}

    total = (d["home_score"] + d["away_score"]).to_numpy(dtype=float)
    wind = np.maximum(d[wind_col].to_numpy(dtype=float) - 20.0, 0.0) / 10.0
    rain = np.minimum(d[rain_col].to_numpy(dtype=float), 10.0)
    X = np.column_stack([np.ones(len(d)), wind, rain])
    coef, *_ = np.linalg.lstsq(X, total, rcond=None)
    resid = total - X @ coef
    dof = max(len(d) - 3, 1)
    cov = np.linalg.pinv(X.T @ X) * float(resid @ resid) / dof
    se = np.sqrt(np.diag(cov))
    return {
        "n": int(len(d)),
        "total_moyen": round(float(coef[0]), 2),
        "effet_vent_par_10kmh": round(float(coef[1]), 3),
        "se_vent": round(float(se[1]), 3),
        "effet_pluie_par_mm": round(float(coef[2]), 3),
        "se_pluie": round(float(se[2]), 3),
        "vent_significatif": bool(abs(coef[1]) > 2 * se[1]),
        "pluie_significative": bool(abs(coef[2]) > 2 * se[2]),
    }
