"""
Suite de tests. Lancer : py -m pytest tests/ -q   (ou py tests/test_suite.py)

Priorite donnee aux invariants qui, s'ils cassent, produisent des erreurs
SILENCIEUSES : reglement des paris, coherence des probabilites, absence de
fuite temporelle. Un bug de handicap -0.25 ne leve aucune exception, il fausse
juste tout le backtest.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sportvalue.core.oddsmath import (
    devig, devig_all_methods, overround, margin_pct, best_odds_across_books,
    american_to_decimal, decimal_to_american,
)
from sportvalue.core.scoredist import ScoreDistribution, settle_asian, settle_total
from sportvalue.core.kelly import kelly_fraction, kelly_stake, kelly_simultaneous_exclusive
from sportvalue.core.calib import blend_logit, fit_blend_weight, shrink_to_market
from sportvalue.core.metrics import brier_score, log_loss_multi, rps, roi_summary
from sportvalue.models.football_dc import DixonColesModel
from sportvalue.models.tennis import game_win_prob, tiebreak_win_prob, TennisMatchModel
from sportvalue.models.linear_score import LinearScoreModel
from sportvalue.models.rugby import RugbyModel
from sportvalue.data.arjel import trj, effective_trj_multi_book, is_playable
from sportvalue.data.schema import Fixture, MarketQuote
from sportvalue.value.scanner import ValueScanner, ScannerConfig, growth_score

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("   OK    " if cond else "   ECHEC ") + label)
    if not cond:
        FAILS.append(label)


# ==========================================================================
def test_oddsmath():
    print("\n-- oddsmath")
    o = [2.10, 3.40, 3.60]
    check(abs(overround(o) - sum(1 / x for x in o)) < 1e-12, "overround")
    check(abs(margin_pct(o) - (1 - 1 / overround(o))) < 1e-12, "marge")
    for name, p in devig_all_methods(o).items():
        check(abs(p.sum() - 1) < 1e-9, f"de-vig {name} somme a 1")
        check(np.all(p > 0), f"de-vig {name} strictement positif")

    # Shin ne doit PAS degenerer en proportionnelle (bug corrige)
    ps, pm = devig(o, "shin"), devig(o, "multiplicative")
    check(np.max(np.abs(ps - pm)) > 1e-4, "shin distinct de la proportionnelle")

    # favourite-longshot : shin releve le favori
    big = [1.25, 6.50, 13.0]
    check(devig(big, "shin")[0] > devig(big, "multiplicative")[0],
          "shin releve la proba du favori marque")

    # marche sans marge : de-vig ne change rien
    fair = [2.0, 2.0]
    check(np.allclose(devig(fair, "shin"), [0.5, 0.5], atol=1e-9), "marche deja fair inchange")

    # conservateur : borne superieure des methodes
    cons = devig(o, conservative=True)
    check(abs(cons.sum() - 1) < 1e-9, "de-vig conservateur normalise")

    best, books = best_odds_across_books({"a": [2.0, 3.0], "b": [2.2, 2.9]})
    check(list(best) == [2.2, 3.0] and books == ["b", "a"], "meilleure cote par issue")
    check(abs(american_to_decimal(-110) - 1.909090909) < 1e-6, "cote americaine -> decimale")
    check(abs(decimal_to_american(2.0) - 100) < 1e-9, "decimale -> americaine")


def test_settlement():
    print("\n-- reglement des paris")
    cases = [
        (1.0, -0.5, 1.0), (0.0, -0.5, -1.0), (0.0, 0.0, 0.0), (1.0, -1.0, 0.0),
        (1.0, -0.25, 1.0), (0.0, -0.25, -0.5), (2.0, -1.75, 0.5), (2.0, -1.5, 1.0),
        (-1.0, 0.5, -1.0), (0.0, 0.25, 0.5), (3.0, -2.25, 1.0), (1.0, -1.25, -0.5),
    ]
    for margin, line, expected in cases:
        got = float(settle_asian(np.array([margin]), line)[0])
        check(abs(got - expected) < 1e-12, f"AH marge={margin:+g} ligne={line:+g} -> {expected:+g}")

    for total, line, side, expected in [
        (2, 2.5, "over", -1.0), (3, 2.5, "over", 1.0), (2, 2.0, "over", 0.0),
        (2, 2.25, "under", 0.5), (2, 2.25, "over", -0.5), (3, 2.75, "over", 0.5),
    ]:
        got = float(settle_total(np.array([total]), line, side)[0])
        check(abs(got - expected) < 1e-12, f"O/U total={total} ligne={line} {side} -> {expected:+g}")


def test_scoredist():
    print("\n-- distribution de scores")
    from scipy.stats import poisson
    lh, la = 1.6, 1.2
    k = np.arange(13)
    mat = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    mat /= mat.sum()
    sd = ScoreDistribution.from_matrix(mat)

    m = sd.market_1x2()
    check(abs(sum(m.values()) - 1) < 1e-9, "1X2 somme a 1")
    check(abs(sd.expected_home - lh) < 0.02, "buts attendus domicile")
    check(abs(sd.expected_away - la) < 0.02, "buts attendus exterieur")

    ou = sd.over_under(2.5)
    check(abs(ou["p_over"] + ou["p_under"] - 1) < 1e-9, "O/U 2.5 somme a 1 (ligne demi)")
    ou3 = sd.over_under(3.0)
    check(ou3["p_push"] > 0.01, "O/U 3.0 a un push non nul")
    check(abs(ou3["p_over"] + ou3["p_under"] + ou3["p_push"] - 1) < 1e-9, "O/U 3.0 somme a 1")

    # AH 0.0 doit egaler draw-no-bet
    ah0 = sd.asian_handicap(0.0)
    dnb = sd.draw_no_bet()
    check(abs(ah0["home_fair_odds"] - 1 / dnb["1"]) < 1e-6, "AH 0.0 == draw-no-bet")

    dc = sd.double_chance()
    check(abs(dc["1X"] - (m["1"] + m["X"])) < 1e-12, "double chance coherente")
    check(abs(sum(c["p"] for c in sd.correct_score(200)) - 1) < 0.02, "scores exacts somment a ~1")

    # matrice et echantillons doivent converger
    rng = np.random.default_rng(0)
    s = np.column_stack([rng.poisson(lh, 200000), rng.poisson(la, 200000)])
    sd2 = ScoreDistribution.from_samples(s)
    check(max(abs(m[kk] - sd2.market_1x2()[kk]) for kk in m) < 0.005,
          "coherence matrice / echantillons")


def test_kelly():
    print("\n-- kelly")
    check(abs(kelly_fraction(0.55, 2.0) - 0.10) < 1e-12, "kelly binaire connu")
    check(kelly_fraction(0.40, 2.0) == 0.0, "esperance negative -> mise nulle")
    r = kelly_stake(0.60, 2.0, 1000, fraction=0.25, max_stake_pct=0.02)
    check(r["stake"] == 20.0 and r["capped"], "plafond de mise applique")
    r2 = kelly_stake(0.52, 2.0, 1000, fraction=0.25, max_stake_pct=0.05)
    check(abs(r2["stake"] - 10.0) < 1e-9, "kelly fractionnaire sans plafond")
    f = kelly_simultaneous_exclusive(np.array([0.5, 0.3, 0.2]), np.array([2.5, 4.0, 6.0]))
    check(f.sum() <= 0.101 and np.all(f >= 0), "kelly simultane borne et positif")

    # a edge egal, la cote basse doit avoir un meilleur score de croissance
    check(growth_score(0.08, 1.44) > 15 * growth_score(0.08, 9.0),
          "croissance : cote basse ~18x plus efficace a edge egal")


def test_calib():
    print("\n-- calibration et fusion")
    pm = np.array([[0.5, 0.3, 0.2]])
    pk = np.array([[0.4, 0.35, 0.25]])
    check(np.allclose(blend_logit(pm, pk, 1.0), pm, atol=1e-9), "w=1 -> modele pur")
    check(np.allclose(blend_logit(pm, pk, 0.0), pk, atol=1e-9), "w=0 -> marche pur")
    b = blend_logit(pm, pk, 0.5)
    check(abs(b.sum() - 1) < 1e-9, "fusion renormalisee")

    mkt = np.array([0.4, 0.3, 0.3])
    sh = shrink_to_market(np.array([0.9, 0.05, 0.05]), mkt, 0.15)
    check(abs(sh.sum() - 1) < 1e-9, "shrink renormalise a 1")
    check(np.max(np.abs(sh - mkt)) <= 0.15 + 1e-9,
          f"ecart au marche reellement borne (max {np.max(np.abs(sh - mkt)):.4f})")

    # le blender doit detecter un modele informatif
    rng = np.random.default_rng(0)
    n = 3000
    truth = rng.dirichlet([3, 2, 2], size=n)
    y = np.array([rng.choice(3, p=t) for t in truth])
    noisy = np.clip(truth + rng.normal(0, 0.12, truth.shape), 0.01, 0.98)
    noisy /= noisy.sum(1, keepdims=True)
    res = fit_blend_weight(truth, noisy, y)
    check(res["w"] > 0.7, "poids eleve quand le modele est meilleur que le marche")


def test_metrics():
    print("\n-- metriques")
    p = np.array([[1.0, 0.0, 0.0]])
    check(abs(brier_score(p, np.array([0]))) < 1e-12, "brier parfait = 0")
    check(log_loss_multi(p, np.array([0])) < 1e-9, "logloss parfaite = 0")
    unif = np.full((1, 3), 1 / 3)
    check(abs(brier_score(unif, np.array([0])) - 2 / 3) < 1e-9, "brier uniforme = 2/3")
    # RPS penalise plus une erreur lointaine
    near = rps(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    far = rps(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    check(far > near, "RPS penalise davantage l'erreur lointaine")
    s = roi_summary(np.array([10.0, 10.0]), np.array([2.0, 2.0]), np.array([1.0, 0.0]))
    check(abs(s["profit"]) < 1e-9 and s["n"] == 2, "roi : un gagnant un perdant a cote 2 = 0")


def test_football_model():
    print("\n-- modele football (Dixon-Coles)")
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(12)]
    force = dict(zip(teams, np.linspace(0.45, -0.45, 12)))
    rows = []
    d0 = datetime(2022, 8, 1)
    for w in range(90):
        order = rng.permutation(teams)
        for i in range(0, 12, 2):
            h, a = order[i], order[i + 1]
            lh, la = np.exp(0.25 + force[h] - force[a]), np.exp(0.05 + force[a] - force[h])
            rows.append({"date": d0 + timedelta(days=7 * w), "home": h, "away": a,
                         "home_score": rng.poisson(lh), "away_score": rng.poisson(la),
                         "neutral": False})
    df = pd.DataFrame(rows)

    m = DixonColesModel(xi=0.003).fit(df)
    check(m.fitted_ and m.opt_success_, "convergence de l'optimisation")
    check(0.0 < m.home_adv < 0.8, f"avantage domicile plausible ({m.home_adv:.3f})")

    r = m.ratings()
    from scipy.stats import spearmanr
    vraie = [force[t] for t in r["equipe"]]
    rho = spearmanr(vraie, r["note_globale"]).statistic
    check(rho > 0.85, f"classement foot correle a la vraie force (rho={rho:.3f})")

    sd = m.predict("T0", "T11")
    p = sd.market_1x2()
    check(p["1"] > p["2"], "le favori a la plus forte probabilite")
    check(abs(sum(p.values()) - 1) < 1e-9, "1X2 predit somme a 1")

    # ANTI-FUITE : as_of doit exclure le futur
    mid = df["date"].iloc[len(df) // 2]
    m2 = DixonColesModel().fit(df, as_of=mid)
    check(m2.n_matches_ < len(df), "as_of exclut bien les matchs posterieurs")

    # les ajustements de contexte agissent dans le bon sens
    base = m.predict("T0", "T5").expected_home
    boost = m.predict("T0", "T5", context={"home_attack_mult": 1.2}).expected_home
    check(boost > base, "un multiplicateur d'attaque augmente les buts attendus")


def test_tennis():
    print("\n-- modele tennis")
    check(abs(game_win_prob(0.5) - 0.5) < 1e-12, "jeu equilibre a p=0.5")
    check(game_win_prob(0.7) > game_win_prob(0.6), "monotonie du jeu")
    check(abs(game_win_prob(0.645) - 0.821) < 0.01, "hold ATP dur realiste (~82%)")
    check(abs(tiebreak_win_prob(0.64, 0.64) - 0.5) < 1e-6, "tie-break symetrique")

    m = TennisMatchModel("atp")
    for target in (0.35, 0.5, 0.72):
        pr = m.predict("A", "B", "Hard", 3, p_match=target)
        check(abs(pr.p_a - target) < 2e-3, f"inversion service -> p_match={target}")
    pr = m.predict("A", "B", "Hard", 3, p_match=0.75)
    check(abs(sum(pr.set_scores.values()) - 1) < 1e-9, "scores en sets somment a 1")
    tg = pr.total_games(22.5)
    check(abs(tg["p_over"] + tg["p_under"] - 1) < 1e-9, "total de jeux somme a 1")
    check(20 < pr.expected_games() < 30, "nombre de jeux attendu plausible en BO3")

    # le format long favorise le meilleur joueur
    sa, sb = m.solve_serve_probs(0.65, "Hard", 3)
    from sportvalue.models.tennis import _p_win_from_serve
    check(_p_win_from_serve(sa, sb, 5) > _p_win_from_serve(sa, sb, 3),
          "BO5 favorise le favori")


def test_linear_and_rugby():
    print("\n-- modeles basket / rugby")
    rng = np.random.default_rng(2)
    teams = [f"E{i}" for i in range(10)]
    force = dict(zip(teams, np.linspace(6, -6, 10)))
    rows = []
    d0 = datetime(2023, 10, 1)
    for w in range(60):
        order = rng.permutation(teams)
        for i in range(0, 10, 2):
            h, a = order[i], order[i + 1]
            rows.append({"date": d0 + timedelta(days=3 * w), "home": h, "away": a,
                         "home_score": max(0, int(rng.normal(110 + force[h] - force[a] + 3, 11))),
                         "away_score": max(0, int(rng.normal(110 + force[a] - force[h], 11))),
                         "neutral": False})
    df = pd.DataFrame(rows)

    lm = LinearScoreModel(half_life_days=200).fit(df)
    check(lm.fitted_, "ajustement du modele lineaire")
    check(1.0 < lm.hfa_ < 8.0, f"avantage terrain plausible ({lm.hfa_:.2f} pts)")
    sd = lm.predict("E0", "E9", allow_draw=False)
    check(abs(sd.market_1x2()["X"]) < 1e-9, "pas de nul autorise au basket")
    check(sd.expected_margin > 0, "le favori a une marge attendue positive")

    r = lm.ratings()
    from scipy.stats import spearmanr
    rho_b = spearmanr([force[t] for t in r["equipe"]], r["note_nette"]).statistic
    check(rho_b > 0.85, f"classement basket correle a la vraie force (rho={rho_b:.3f})")

    # rugby + meteo
    rows_r = [{**x, "home_score": max(0, int(x["home_score"] * 0.25)),
               "away_score": max(0, int(x["away_score"] * 0.25))} for x in rows]
    rm = RugbyModel().fit(pd.DataFrame(rows_r))
    sec = rm.predict("E0", "E9", context={"weather": None})
    pluie = rm.predict("E0", "E9", context={"weather": {"wind_kmh": 55, "precip_mm": 6, "temp_c": 3}})
    check(pluie.expected_total < sec.expected_total, "la tempete reduit le total de points")
    rep = rm.weather_impact_report("E0", "E9", {"wind_kmh": 55, "precip_mm": 6, "temp_c": 3})
    check(rep["delta_points"] < 0 and len(rep["notes"]) >= 2, "rapport meteo coherent")
    check(sec.margin_band(3, 3) > 0, "les ecarts discrets ont une masse non nulle")


def test_predict_markets():
    """Coherence des blocs de marches produits par predict.py."""
    print("\n-- blocs de marches (predict)")
    from scipy.stats import poisson
    from sportvalue import predict as P

    k = np.arange(13)
    mat = np.outer(poisson.pmf(k, 1.7), poisson.pmf(k, 1.1)); mat /= mat.sum()
    sd = ScoreDistribution.from_matrix(mat)

    f = P.football_markets(sd)
    # Tolerance a 1e-3 : football_markets ARRONDIT a 4 decimales pour
    # l'affichage, donc la somme des valeurs affichees peut devier de ~1.5e-4.
    # La somme exacte est verifiee dans test_scoredist sur market_1x2().
    check(abs(sum(f["1x2"].values()) - 1) < 1e-3, "football 1X2 somme a 1 (valeurs arrondies)")
    check(abs(f["btts"]["oui"] + f["btts"]["non"] - 1) < 1e-3, "football BTTS somme a 1")
    overs = [r["over"] for r in f["total_buts"]]
    check(all(overs[i] >= overs[i + 1] for i in range(len(overs) - 1)),
          "football : P(over) decroit quand la ligne monte")

    # BASKET : la probabilite du domicile doit DECROITRE quand son handicap
    # devient plus exigeant. Un bug de signe donnait une ligne a p = 1.0.
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal([118, 105], [[120, 10], [10, 120]], size=40000)
    sdb = ScoreDistribution.from_samples(np.rint(draws))
    b = P.basket_markets(sdb)
    lignes = [float(h["ligne_domicile"]) for h in b["handicap"]]
    probas = [h["p_domicile"] for h in b["handicap"]]
    ordre = np.argsort(lignes)[::-1]   # du plus genereux au plus exigeant
    p_ord = [probas[i] for i in ordre]
    check(all(p_ord[i] >= p_ord[i + 1] - 1e-9 for i in range(len(p_ord) - 1)),
          f"basket : handicap monotone ({[round(x,3) for x in p_ord]})")
    check(all(0.0 < p < 1.0 for p in probas),
          f"basket : aucune probabilite degeneree a 0 ou 1 ({probas})")
    check(abs(sum(b["vainqueur"].values()) - 1) < 1e-6, "basket vainqueur somme a 1")


def test_scorers():
    print("\n-- marches joueurs")
    from sportvalue.models.scorers import ScorerModel, PRIORS_FOOT

    squad = pd.DataFrame([
        {"joueur": "Attaquant", "poste": "attaquant", "minutes": 900,
         "buts_hors_penalty": 9, "tireur_penalty": 1},
        {"joueur": "Milieu", "poste": "milieu", "minutes": 900, "buts_hors_penalty": 3,
         "tireur_penalty": 0},
        {"joueur": "Defenseur", "poste": "defenseur", "minutes": 900,
         "buts_hors_penalty": 1, "tireur_penalty": 0},
        {"joueur": "Remplacant", "poste": "attaquant", "minutes": 120,
         "buts_hors_penalty": 1, "tireur_penalty": 0},
    ])
    m = ScorerModel(PRIORS_FOOT, k_shrink=600)
    lam_h, lam_a = 1.8, 1.2
    pred = m.predict_match(squad, squad.copy(), lam_h, lam_a)

    chk = m.coherence_check(pred, lam_h, lam_a)
    check(chk["ok"], f"somme des lambdas joueurs = lambda equipe hors csc ({chk})")
    check(pred["p_marque"].between(0, 1).all(), "probabilites de marquer dans [0,1]")
    check((pred["p_2plus"] <= pred["p_marque"] + 1e-12).all(),
          "P(2 buts et +) <= P(marque)")
    check((pred["p_triple"] <= pred["p_2plus"] + 1e-12).all(),
          "P(triple) <= P(2 buts et +)")

    # somme des premiers buteurs + aucun buteur = 1 - part des csc
    total = float(pred["p_premier_buteur"].sum()) + float(pred["p_aucun_buteur"].iloc[0])
    check(abs(total - 1.0) < 0.05, f"premiers buteurs + aucun buteur ~ 1 ({total:.4f})")

    # l'attaquant tireur de penalty doit dominer
    top = pred.sort_values("lambda_joueur", ascending=False).iloc[0]
    check(top["joueur"] == "Attaquant", "le buteur attitre ressort en tete")

    # un plus gros lambda d'equipe releve tout le monde
    p2 = m.predict_match(squad, squad.copy(), 3.0, 1.2)
    check(float(p2.loc[p2.joueur == "Attaquant", "lambda_joueur"].iloc[0])
          > float(pred.loc[pred.joueur == "Attaquant", "lambda_joueur"].iloc[0]),
          "une equipe plus prolifique releve les lambdas individuels")


def test_arjel():
    print("\n-- specificites ARJEL")
    o = [1.55, 4.30, 5.50]
    check(abs(trj(o) - 1 / overround(o)) < 1e-12, "TRJ = 1/overround")
    check(is_playable([2.0, 2.0]) and not is_playable([1.7, 1.7]),
          "filtre de jouabilite par TRJ")

    quotes = {"winamax": [1.55, 4.30, 5.50], "betclic": [1.57, 4.25, 5.60],
              "unibet": [1.54, 4.40, 5.40]}
    eff = effective_trj_multi_book(quotes)
    check(eff["trj_combine_%"] >= eff["meilleur_book_seul_%"],
          "le shopping ne peut pas degrader le TRJ")
    check(eff["gain_du_shopping_pts"] >= 0, "gain du shopping positif ou nul")
    check(eff["book_par_issue"] == ["betclic", "unibet", "betclic"],
          "attribution du meilleur book par issue")


def test_scanner():
    print("\n-- scanner de value")
    fx = Fixture("football", "Ligue 1", datetime(2026, 9, 13), "PSG", "Lyon")
    # Pinnacle : favori a ~62% fair. Un book ARJEL propose 1.85 sur ce favori.
    ref = [MarketQuote("pinnacle", "1x2", ["1", "X", "2"], [1.60, 4.50, 5.90])]
    p_ref = devig([1.60, 4.50, 5.90], "shin")

    genereux = [
        MarketQuote("winamax", "1x2", ["1", "X", "2"], [1.85, 4.00, 4.40]),
        MarketQuote("betclic", "1x2", ["1", "X", "2"], [1.57, 4.25, 5.60]),
    ]
    sc = ValueScanner(ScannerConfig(min_probability=0.55, min_edge=0.02, blend_weight=0.0))
    bets = sc.find_value(fx, "1x2", np.full(3, 1 / 3), ref, genereux)
    check(len(bets) == 1, "un seul pari detecte")
    if bets:
        b = bets[0]
        check(b.selection == "1", "le pari porte sur l'issue mal chiffree")
        check(b.book == "Winamax", "le book le plus genereux est retenu")
        check(abs(b.odds - 1.85) < 1e-9, "la meilleure cote ARJEL est prise")
        check(abs(b.p_final - p_ref[0]) < 1e-9, "w=0 -> proba finale = reference sharp")
        check(b.edge > 0.02, f"edge coherent ({100*b.edge:.1f}%)")
        check(b.stake > 0, "mise non nulle")

    # aucun edge quand les books francais sont sous la ligne sharp
    serres = [
        MarketQuote("winamax", "1x2", ["1", "X", "2"], [1.55, 4.30, 5.50]),
        MarketQuote("betclic", "1x2", ["1", "X", "2"], [1.54, 4.25, 5.45]),
    ]
    check(sc.find_value(fx, "1x2", np.full(3, 1 / 3), ref, serres) == [],
          "aucune value quand les books ARJEL sont plus chers que la reference")

    # le filtre de haute probabilite ecarte les outsiders genereux
    outsider = [MarketQuote("winamax", "1x2", ["1", "X", "2"], [1.50, 4.20, 9.00])]
    sc2 = ValueScanner(ScannerConfig(min_probability=0.55, min_edge=0.02, blend_weight=0.0))
    b2 = sc2.find_value(fx, "1x2", np.full(3, 1 / 3), ref, outsider)
    check(all(x.p_final >= 0.55 for x in b2),
          "le filtre haute probabilite ecarte les outsiders")

    # marche de reference illiquide : doit etre ecarte AVANT le de-vig.
    # Cas reel : un exchange sans liquidite renvoie ~1.01 sur toutes les issues.
    ref_pourri = [MarketQuote("betfair_ex_eu", "1x2", ["1", "X", "2"], [1.06, 1.06, 1.01])]
    check(sc.find_value(fx, "1x2", np.full(3, 1 / 3), ref_pourri, genereux) == [],
          "reference illiquide (overround 2.88) ecartee")

    # une reference saine parmi des references pourries doit rester utilisable
    ref_mixte = ref_pourri + ref
    b3 = sc.find_value(fx, "1x2", np.full(3, 1 / 3), ref_mixte, genereux)
    check(len(b3) == 1, "reference saine retenue malgre une reference illiquide")

    # marche surtaxe : rejete avant analyse
    surtaxe = [MarketQuote("pmu", "1x2", ["1", "X", "2"], [1.40, 3.60, 4.60])]
    check(sc.find_value(fx, "1x2", np.full(3, 1 / 3), ref, surtaxe) == [],
          "marche sous le TRJ minimal ecarte")


# ==========================================================================
def main() -> int:
    print("=" * 70)
    print("SUITE DE TESTS sportvalue")
    print("=" * 70)
    for fn in [test_oddsmath, test_settlement, test_scoredist, test_kelly,
               test_calib, test_metrics, test_football_model, test_tennis,
               test_linear_and_rugby, test_predict_markets, test_scorers,
               test_arjel, test_scanner]:
        fn()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"{len(FAILS)} ECHEC(S) :")
        for f in FAILS:
            print("   - " + f)
        return 1
    print("TOUS LES TESTS PASSENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
