"""
Backtest en avancee temporelle (walk-forward).

Regles non negociables, chacune correspondant a une facon classique de se
mentir a soi-meme :

  1. AUCUNE FUITE TEMPORELLE. Le modele qui predit le match du 12 mars n'a vu
     que les matchs anterieurs au 12 mars. C'est impose par `as_of` dans fit(),
     pas par une convention de nommage.

  2. LE POIDS DE FUSION EST APPRIS SUR LA PERIODE D'APPRENTISSAGE. L'ajuster
     sur toute la periode revient a choisir le parametre en connaissant le
     resultat : c'est la fuite la plus frequente dans les backtests de paris.

  3. ON PARIE A LA COTE DISPONIBLE AVANT LE MATCH, PAS A LA CLOTURE. La cote de
     cloture n'est connue qu'apres coup ; l'utiliser pour miser gonfle le ROI
     d'environ 2 a 4 points. Ici on mise sur la meilleure cote pre-match
     (colonnes Max) et on garde la cloture Pinnacle UNIQUEMENT pour mesurer le
     CLV.

  4. SEPARATION PREDICTION / SIMULATION. Les predictions sont calculees une
     fois ; on peut ensuite balayer des dizaines de seuils sans re-entrainer.
     Attention : balayer beaucoup de seuils puis retenir le meilleur est du
     sur-apprentissage. C'est pourquoi `threshold_sweep` renvoie AUSSI le
     nombre de paris, pour juger si un seuil flatteur repose sur 12 paris.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.calib import ProbabilityCalibrator, blend_logit, fit_blend_weight, reliability_table
from ..core.kelly import kelly_stake
from ..core.metrics import (
    bootstrap_roi_ci,
    brier_score,
    closing_line_value,
    clv_baseline_control,
    edge_t_stat,
    log_loss_multi,
    max_drawdown,
    roi_summary,
    rps,
)
from ..core.oddsmath import devig
from ..models.football_dc import DixonColesModel
from ..value.scanner import growth_score

__all__ = ["PredictionSet", "run_football_walkforward", "simulate_bets", "threshold_sweep"]

_OUTCOMES = ["1", "X", "2"]


@dataclass
class PredictionSet:
    """Predictions hors echantillon + cotes, pretes pour la simulation."""

    frame: pd.DataFrame
    blend_weight: float
    train_end: pd.Timestamp
    model_params: dict = field(default_factory=dict)

    def quality(self) -> dict:
        """Qualite probabiliste sur la periode de test, modele vs marche."""
        d = self.frame
        y = d["outcome"].to_numpy(dtype=int)
        cols_m = ["p_model_1", "p_model_X", "p_model_2"]
        cols_k = ["p_market_1", "p_market_X", "p_market_2"]
        cols_f = ["p_final_1", "p_final_X", "p_final_2"]
        pm, pk, pf = d[cols_m].to_numpy(), d[cols_k].to_numpy(), d[cols_f].to_numpy()
        return {
            "n_matchs": len(d),
            "logloss_modele": round(log_loss_multi(pm, y), 5),
            "logloss_marche": round(log_loss_multi(pk, y), 5),
            "logloss_fusion": round(log_loss_multi(pf, y), 5),
            "brier_modele": round(brier_score(pm, y), 5),
            "brier_marche": round(brier_score(pk, y), 5),
            "brier_fusion": round(brier_score(pf, y), 5),
            "rps_marche": round(rps(pk, y), 5),
            "rps_fusion": round(rps(pf, y), 5),
            "fusion_bat_marche": bool(log_loss_multi(pf, y) < log_loss_multi(pk, y)),
        }

    def calibration(self, which: str = "final", n_bins: int = 10) -> list[dict]:
        """Table de fiabilite, toutes issues confondues."""
        d = self.frame
        cols = [f"p_{which}_{o}" for o in _OUTCOMES]
        p = d[cols].to_numpy().ravel()
        y = np.zeros((len(d), 3))
        y[np.arange(len(d)), d["outcome"].to_numpy(dtype=int)] = 1
        return reliability_table(p, y.ravel(), n_bins=n_bins)


# --------------------------------------------------------------------------
def _odds_block(df: pd.DataFrame, prefix: str) -> np.ndarray | None:
    cols = [f"{prefix}_{s}" for s in "HDA"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)


def run_football_walkforward(
    df: pd.DataFrame,
    train_fraction: float = 0.45,
    refit_every_days: int = 7,
    xi: float = 0.0035,
    ridge: float = 0.02,
    bet_odds_col: str = "odds_max_open",
    market_odds_col: str = "odds_pinnacle_open",
    closing_odds_col: str = "odds_pinnacle_close",
    devig_method: str = "shin",
    verbose: bool = True,
) -> PredictionSet:
    """
    Deroule un backtest complet sur un ou plusieurs championnats.

    bet_odds_col     : cotes auxquelles on MISE (meilleure cote pre-match).
    market_odds_col  : cotes servant de reference marche (book sharp, pre-match).
    closing_odds_col : cotes de cloture, utilisees UNIQUEMENT pour le CLV.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)

    needed = [f"{market_odds_col}_{s}" for s in "HDA"] + [f"{bet_odds_col}_{s}" for s in "HDA"]
    d = d.dropna(subset=[c for c in needed if c in d.columns]).reset_index(drop=True)
    if len(d) < 300:
        raise ValueError(f"Pas assez de matchs avec cotes completes : {len(d)}")

    split_idx = int(len(d) * train_fraction)
    train_end = d["date"].iloc[split_idx]
    if verbose:
        print(f"  {len(d)} matchs | apprentissage jusqu'au {train_end:%d/%m/%Y} "
              f"({split_idx} matchs), test sur {len(d) - split_idx}")

    # --- 1) predictions hors echantillon sur TOUTE la periode post-demarrage
    start_idx = max(int(len(d) * 0.20), 200)
    rows: list[dict] = []
    leagues = d["league_code"].unique() if "league_code" in d.columns else [None]

    for lg in leagues:
        sub = d if lg is None else d[d["league_code"] == lg]
        sub = sub.reset_index(drop=True)
        if len(sub) < 250:
            continue
        lg_start = max(int(len(sub) * 0.20), 150)
        model = None
        last_fit: pd.Timestamp | None = None
        n_fits = 0

        for i in range(lg_start, len(sub)):
            row = sub.iloc[i]
            date = row["date"]
            if model is None or last_fit is None or (date - last_fit).days >= refit_every_days:
                hist = sub.iloc[:i]
                try:
                    model = DixonColesModel(xi=xi, ridge=ridge).fit(hist, as_of=date)
                    last_fit = date
                    n_fits += 1
                except (ValueError, RuntimeError):
                    continue

            sd = model.predict(row["home"], row["away"], neutral=bool(row.get("neutral", False)))
            p = sd.market_1x2()
            pm = np.array([p["1"], p["X"], p["2"]], dtype=float)

            mkt = np.array([row.get(f"{market_odds_col}_{s}") for s in "HDA"], dtype=float)
            bet = np.array([row.get(f"{bet_odds_col}_{s}") for s in "HDA"], dtype=float)
            close = np.array([row.get(f"{closing_odds_col}_{s}") for s in "HDA"], dtype=float)
            if np.any(~np.isfinite(mkt)) or np.any(mkt <= 1.0):
                continue
            pk = devig(mkt, method=devig_method)

            hs, as_ = row["home_score"], row["away_score"]
            outcome = 0 if hs > as_ else (1 if hs == as_ else 2)

            rows.append(
                {
                    "date": date,
                    "league": lg,
                    "home": row["home"],
                    "away": row["away"],
                    "outcome": outcome,
                    "home_score": hs,
                    "away_score": as_,
                    **{f"p_model_{o}": pm[j] for j, o in enumerate(_OUTCOMES)},
                    **{f"p_market_{o}": pk[j] for j, o in enumerate(_OUTCOMES)},
                    **{f"bet_odds_{o}": bet[j] for j, o in enumerate(_OUTCOMES)},
                    **{f"close_odds_{o}": close[j] for j, o in enumerate(_OUTCOMES)},
                    "xg_home": sd.expected_home,
                    "xg_away": sd.expected_away,
                }
            )
        if verbose and lg is not None:
            print(f"    {lg}: {len(sub) - lg_start} predictions, {n_fits} re-entrainements")

    preds = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if preds.empty:
        raise RuntimeError("Aucune prediction produite")

    # --- 2) poids de fusion appris UNIQUEMENT sur la periode d'apprentissage
    tr = preds[preds["date"] < train_end]
    te = preds[preds["date"] >= train_end]
    if len(tr) < 100:
        w, fit_info = 0.30, {"note": "periode d'apprentissage trop courte, poids par defaut"}
    else:
        fit_info = fit_blend_weight(
            tr[[f"p_model_{o}" for o in _OUTCOMES]].to_numpy(),
            tr[[f"p_market_{o}" for o in _OUTCOMES]].to_numpy(),
            tr["outcome"].to_numpy(dtype=int),
        )
        w = fit_info["w"]
    if verbose:
        print(f"  poids de fusion appris : w = {w:.3f}  "
              f"(1.0 = modele seul, 0.0 = marche seul)")
        if "gain_vs_market" in fit_info:
            print(f"  gain de log-loss vs marche seul sur l'apprentissage : "
                  f"{fit_info['gain_vs_market']:+.5f}")

    pm_all = preds[[f"p_model_{o}" for o in _OUTCOMES]].to_numpy()
    pk_all = preds[[f"p_market_{o}" for o in _OUTCOMES]].to_numpy()
    pf = blend_logit(pm_all, pk_all, w)
    for j, o in enumerate(_OUTCOMES):
        preds[f"p_final_{o}"] = pf[:, j]

    out = PredictionSet(
        frame=te.reset_index(drop=True) if len(te) > 50 else preds,
        blend_weight=w,
        train_end=train_end,
        model_params={"xi": xi, "ridge": ridge, "refit_every_days": refit_every_days,
                      **{k: v for k, v in fit_info.items() if isinstance(v, (int, float))}},
    )
    # recalculer p_final sur le sous-ensemble conserve
    sub = out.frame
    pf2 = blend_logit(
        sub[[f"p_model_{o}" for o in _OUTCOMES]].to_numpy(),
        sub[[f"p_market_{o}" for o in _OUTCOMES]].to_numpy(),
        w,
    )
    for j, o in enumerate(_OUTCOMES):
        out.frame[f"p_final_{o}"] = pf2[:, j]
    return out


# --------------------------------------------------------------------------
def simulate_bets(
    ps: PredictionSet,
    min_probability: float = 0.55,
    min_edge: float = 0.03,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.02,
    flat_stake: float | None = None,
    use_final: bool = True,
    compound: bool = False,
) -> dict:
    """
    Simule la strategie de mise sur les predictions.

    min_probability : le filtre "haute probabilite" demande.
    flat_stake      : si fourni, mise fixe (utile pour comparer la strategie
                      sans l'effet du dimensionnement).
    compound        : si True, la bankroll evolue au fil des paris (croissance
                      geometrique reelle). Sinon, mises calculees sur la
                      bankroll initiale (mesure l'edge pur, sans effet de
                      composition).
    """
    d = ps.frame
    prefix = "p_final" if use_final else "p_model"
    picks = []
    br = bankroll

    for _, r in d.iterrows():
        for j, o in enumerate(_OUTCOMES):
            p = float(r[f"{prefix}_{o}"])
            odds = float(r[f"bet_odds_{o}"])
            if not np.isfinite(odds) or odds <= 1.0:
                continue
            edge = p * odds - 1.0
            if p < min_probability or edge < min_edge:
                continue

            if flat_stake is not None:
                stake = flat_stake
                kf = flat_stake / max(br, 1e-9)
            else:
                info = kelly_stake(p, odds, br if compound else bankroll,
                                   fraction=kelly_fraction, max_stake_pct=max_stake_pct)
                stake, kf = info["stake"], info["kelly_used"]
            if stake <= 0:
                continue

            won = 1.0 if int(r["outcome"]) == j else 0.0
            profit = stake * (odds - 1.0) if won else -stake
            if compound:
                br += profit

            picks.append(
                {
                    "date": r["date"],
                    "match": f"{r['home']} - {r['away']}",
                    "pari": o,
                    "cote": odds,
                    "p": p,
                    "p_marche": float(r[f"p_market_{o}"]),
                    "edge": edge,
                    "mise": stake,
                    "gagne": won,
                    "profit": profit,
                    "score": growth_score(edge, odds),
                    "close_odds": float(r.get(f"close_odds_{o}", np.nan)),
                    "close_H": float(r.get("close_odds_1", np.nan)),
                    "close_D": float(r.get("close_odds_X", np.nan)),
                    "close_A": float(r.get("close_odds_2", np.nan)),
                    "sel_index": j,
                }
            )

    if not picks:
        return {"n": 0, "note": "aucun pari ne passe les filtres"}

    bets = pd.DataFrame(picks)
    summ = roi_summary(bets["mise"].to_numpy(), bets["cote"].to_numpy(), bets["gagne"].to_numpy())
    dd = max_drawdown(summ["profit_series"], starting_bankroll=bankroll)
    ci = bootstrap_roi_ci(summ["profit_series"], bets["mise"].to_numpy())
    t = edge_t_stat(summ["profit_series"], bets["mise"].to_numpy())

    # CLV, corrige du biais mecanique du shopping de cotes
    clv = None
    ok = bets[["close_H", "close_D", "close_A"]].notna().all(axis=1) & (
        bets[["close_H", "close_D", "close_A"]] > 1.0
    ).all(axis=1)
    if ok.any():
        sub = bets[ok]
        clv = closing_line_value(
            sub["cote"].to_numpy(),
            sub[["close_H", "close_D", "close_A"]].to_numpy().tolist(),
            sub["sel_index"].to_numpy(),
        )
        base = clv_baseline_control(
            d[[f"bet_odds_{o}" for o in _OUTCOMES]].to_numpy(),
            d[[f"close_odds_{o}" for o in _OUTCOMES]].to_numpy(),
        )
        clv["clv_aleatoire_pct"] = base["clv_aleatoire_pct"]
        clv["beat_close_aleatoire"] = base["beat_close_aleatoire"]
        # Part du CLV reellement attribuable a la selection du modele
        clv["clv_net_pct"] = clv["clv_mean_pct"] - base["clv_aleatoire_pct"]
        clv["beat_close_net"] = clv["beat_close_rate"] - base["beat_close_aleatoire"]

    return {
        "n": summ["n"],
        "mise_totale": round(summ["turnover"], 2),
        "profit": round(summ["profit"], 2),
        "roi": summ["roi"],
        "taux_reussite": summ["win_rate"],
        "cote_moyenne": round(summ["avg_odds"], 3),
        "proba_moyenne": round(float(bets["p"].mean()), 4),
        "edge_moyen": round(float(bets["edge"].mean()), 4),
        "roi_ic95": (round(ci["lo"], 4), round(ci["hi"], 4)),
        "p_roi_positif": round(ci["p_gt_0"], 3),
        "t_stat": round(t, 2),
        "max_drawdown_%": round(100 * dd["max_dd_pct"], 1),
        "bankroll_finale": round(bankroll + summ["profit"], 2),
        "clv": clv,
        "bets": bets,
    }


def threshold_sweep(
    ps: PredictionSet,
    prob_grid=(0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70),
    edge_grid=(0.01, 0.02, 0.03, 0.05, 0.08),
    **kwargs,
) -> pd.DataFrame:
    """
    Balaye les seuils. A LIRE AVEC PRUDENCE : choisir la case la plus rentable
    apres coup est du sur-apprentissage. La colonne `n` sert justement a
    reperer les cases qui ne reposent que sur une poignee de paris.
    """
    rows = []
    for pmin in prob_grid:
        for emin in edge_grid:
            res = simulate_bets(ps, min_probability=pmin, min_edge=emin, **kwargs)
            if res.get("n", 0) == 0:
                continue
            rows.append(
                {
                    "p_min": pmin,
                    "edge_min": emin,
                    "n": res["n"],
                    "roi_%": round(100 * res["roi"], 2),
                    "profit": res["profit"],
                    "cote_moy": res["cote_moyenne"],
                    "t_stat": res["t_stat"],
                    "clv_%": round(100 * res["clv"]["clv_mean_pct"], 2) if res.get("clv") else None,
                    "bat_cloture_%": round(100 * res["clv"]["beat_close_rate"], 1) if res.get("clv") else None,
                }
            )
    return pd.DataFrame(rows)
