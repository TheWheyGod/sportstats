"""Interface en ligne de commande."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .core.oddsmath import devig, devig_all_methods, margin_pct
from .data.arjel import ARJEL_BOOKS, effective_trj_multi_book, trj, trj_report
from .data.manual_odds import load_manual_csv, write_template
from .value.scanner import ScannerConfig, ValueScanner

# --------------------------------------------------------------------------
# Rendu tableau (sans dependance externe)
# --------------------------------------------------------------------------
def table(rows: list[dict], cols: list[str] | None = None, max_width: int = 34) -> str:
    if not rows:
        return "   (aucune ligne)"
    cols = cols or list(rows[0])
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)
    widths = {c: min(max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)), max_width) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    head = "  ".join(c[: widths[c]].ljust(widths[c]) for c in cols)
    out = [head, sep]
    for r in rows:
        out.append("  ".join(fmt(r.get(c, ""))[: widths[c]].ljust(widths[c]) for c in cols))
    return "\n".join("   " + line for line in out)


def title(txt: str) -> None:
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# --------------------------------------------------------------------------
def cmd_template(args) -> int:
    p = write_template(args.fichier)
    print(f"Modele CSV ecrit dans {p}")
    print("Remplis-le avec les cotes relevees chez tes books, puis lance :")
    print(f"   py -m sportvalue scan-csv {p}")
    return 0


def cmd_scan_csv(args) -> int:
    markets = load_manual_csv(args.fichier)
    cfg = ScannerConfig(
        min_probability=args.p_min,
        min_edge=args.edge_min,
        blend_weight=args.w,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_stake_pct=args.max_mise,
        min_trj=args.trj_min,
        allow_no_reference=args.sans_reference,
    )
    scanner = ValueScanner(cfg)

    title("PARAMETRES")
    print(table([{"parametre": k, "valeur": v} for k, v in cfg.to_dict().items()],
                ["parametre", "valeur"]))

    all_bets = []
    for m in markets:
        n = len(m["selections"])
        # Sans modele propre, la probabilite du modele EST la reference sharp :
        # avec blend_weight=0 cela ne change rien au resultat, et cela permet
        # d'utiliser l'outil en pur detecteur d'ecart sharp / ARJEL.
        p_model = np.full(n, 1.0 / n)
        bets = scanner.find_value(
            m["fixture"], m["market"], p_model, m["reference"], m["arjel"], line=m["line"]
        )
        all_bets.extend(bets)

    title("ANALYSE TRJ PAR MARCHE")
    for m in markets:
        quotes = {q.book: q.odds for q in m["arjel"]}
        eff = effective_trj_multi_book(quotes)
        lbl = f"{m['fixture'].label()} / {m['market']}"
        print(f"\n   {lbl}")
        print(table(trj_report(quotes), ["book", "trj_%", "marge_%", "jouable"]))
        print(f"   -> meilleur book seul {eff['meilleur_book_seul_%']}%  "
              f"| en shoppant les {len(quotes)} books {eff['trj_combine_%']}%  "
              f"(gain {eff['gain_du_shopping_pts']:+} pts)"
              + ("   *** ARBITRAGE ***" if eff["arbitrage"] else ""))

    title(f"VALUE DETECTEE ({len(all_bets)} pari(s))")
    if not all_bets:
        print("   Aucun pari ne passe les filtres.")
        print("   Pistes : baisser --p-min ou --edge-min, ou ajouter des books.")
        return 0

    ranked = ValueScanner.rank(all_bets, top=args.top)
    print(table([b.to_row() for b in ranked],
                ["match", "marche", "pari", "book", "cote", "p_finale",
                 "p_marche", "edge_%", "mise", "confiance", "score"]))

    if args.details:
        title("JUSTIFICATION DE CHAQUE PARI")
        for b in ranked:
            print(f"\n   {b.fixture.label()} | {b.market} | {b.selection} @ {b.odds:.2f} ({b.book})")
            for r in b.rationale:
                print(f"      . {r}")
            print(f"      diagnostics : {b.diagnostics}")
    return 0


def _render(bloc, indent: int = 3) -> None:
    """Affichage recursif d'un dictionnaire de marches."""
    pad = " " * indent
    if isinstance(bloc, dict):
        for k, v in bloc.items():
            if isinstance(v, (dict, list)):
                print(f"\n{pad}{k.replace('_', ' ').upper()}")
                _render(v, indent + 3)
            else:
                print(f"{pad}{k:<26} {v}")
    elif isinstance(bloc, list):
        if bloc and isinstance(bloc[0], dict):
            print(table(bloc, list(bloc[0].keys()), max_width=26))
        else:
            for x in bloc:
                print(f"{pad}. {x}")


def cmd_predict(args) -> int:
    """
    Probabilites d'un match, calculees UNIQUEMENT a partir des donnees.
    Aucune cote n'entre dans le calcul.
    """
    from .data.results_csv import load_results, load_players as load_players_csv
    from .models.football_dc import DixonColesModel
    from .models.linear_score import LinearScoreModel
    from .models.rugby import RugbyModel
    from .models.tennis import TennisMatchModel
    from .models.elo import EloRating
    from .models.scorers import ScorerModel, PRIORS_FOOT, PRIORS_RUGBY
    from . import predict as P

    sport = args.sport

    # ---------- historique ----------
    if args.historique:
        title("HISTORIQUE")
        hist = load_results(args.historique)
    elif sport == "football":
        from .data import football_data as fd
        title("HISTORIQUE (football-data.co.uk)")
        codes_fit = fd.with_lower_tier([args.league])
        print(f"   ajustement conjoint sur {', '.join(codes_fit)} "
              "(la division inferieure evite les notes aberrantes des promus)")
        hist = fd.load_many(codes_fit, fd.season_codes(args.depuis, args.jusqu))
        if hist.empty:
            print("   Aucune donnee telechargee.")
            return 1
    else:
        print(f"Le sport '{sport}' exige un historique : --historique fichier.csv")
        print(f"   Modele de fichier :  py -m sportvalue template-resultats {sport}.csv")
        return 1

    equipes = sorted(set(hist["home"]) | set(hist["away"]))
    for nom in (args.home, args.away):
        if nom not in equipes:
            proches = [e for e in equipes if nom.lower()[:4] in e.lower()]
            print(f"\n   [!] '{nom}' absent de l'historique.")
            if proches:
                print(f"       Vouliez-vous dire : {', '.join(proches[:6])} ?")
            else:
                print(f"       Equipes disponibles : {', '.join(equipes[:15])}"
                      + ("..." if len(equipes) > 15 else ""))
            return 1

    title(f"{args.home}  -  {args.away}   [{sport}]")

    # ---------- meteo (rugby) ----------
    meteo = None
    if args.meteo and sport == "rugby":
        from .data import weather as wx
        from datetime import datetime as _dt, timedelta as _td
        quand = _dt.now() + _td(days=args.dans_jours)
        meteo = wx.venue_weather(args.home, quand)
        print(f"\n   METEO {args.home} : {wx.describe(meteo)}  [{meteo.get('source')}]")

    # ---------- modele d'equipe + marches ----------
    if sport == "football":
        model = DixonColesModel(xi=args.xi).fit(hist)
        print("\n   MODELE " + str(model.params_summary()))
        sd = model.predict(args.home, args.away, neutral=args.neutre)

        scorers = None
        if args.buteurs:
            squads, lam = {}, {"home": sd.expected_home, "away": sd.expected_away}
            if args.joueurs:
                pl = load_players_csv(args.joueurs, "football")
                squads["home"] = pl[pl["equipe"].str.lower() == args.home.lower()]
                squads["away"] = pl[pl["equipe"].str.lower() == args.away.lower()]
                mh = ma = None
            else:
                from .data import fpl
                print("\n   BUTEURS : statistiques FPL (Premier League)")
                pl = fpl.load_players()
                inv = {v: k for k, v in P.FPL_TO_FOOTBALLDATA.items()}
                squads["home"] = fpl.team_squad(pl, inv.get(args.home, args.home))
                squads["away"] = fpl.team_squad(pl, inv.get(args.away, args.away))
                mh = fpl.estimate_minutes(squads["home"])
                ma = fpl.estimate_minutes(squads["away"])
            if squads["home"].empty or squads["away"].empty:
                print("   [!] effectif introuvable pour une des equipes, buteurs ignores.")
            else:
                sm = ScorerModel(PRIORS_FOOT, k_shrink=args.k_shrink)
                scorers = sm.predict_match(squads["home"], squads["away"],
                                           lam["home"], lam["away"], mh, ma)
                chk = sm.coherence_check(scorers, lam["home"], lam["away"])
                print(f"   coherence buteurs/buts : {'OK' if chk['ok'] else 'ECART'} "
                      f"(somme lambda {chk['somme_lambda_domicile']} vs cible "
                      f"{chk['cible_domicile']})")
        marches = P.football_markets(sd, scorers)
        sd_rapport, scorers_rapport = sd, scorers

    elif sport == "rugby":
        model = RugbyModel(part_essais=args.part_essais,
                           half_life_days=args.demi_vie_rugby).fit(hist)
        print("\n   MODELE " + str(model.base.params_summary()))
        ctx = {"weather": meteo} if meteo else {}
        sd = model.predict(args.home, args.away, neutral=args.neutre, context=ctx)
        ts = None
        if args.joueurs:
            pl = load_players_csv(args.joueurs, "rugby")
            sh = pl[pl["equipe"].str.lower() == args.home.lower()]
            sa = pl[pl["equipe"].str.lower() == args.away.lower()]
            if not sh.empty and not sa.empty:
                sm = ScorerModel(PRIORS_RUGBY, k_shrink=args.k_shrink,
                                 full_match_minutes=80.0, own_goal_share=0.0,
                                 penalty_rate=0.0)
                th, ta = sd.expected_tries
                ts = sm.predict_match(sh, sa, th, ta)
            else:
                print("   [!] effectif introuvable, marqueurs d'essais ignores.")
        marches = P.rugby_markets(sd, ts)
        sd_rapport, scorers_rapport = sd, ts

    elif sport == "basket":
        model = LinearScoreModel(half_life_days=args.demi_vie).fit(hist)
        print("\n   MODELE " + str(model.params_summary()))
        sd = model.predict(args.home, args.away, neutral=args.neutre, allow_draw=False)
        marches = P.basket_markets(sd)
        sd_rapport, scorers_rapport = sd, None

    elif sport == "tennis":
        elo = EloRating(k0=250, k_offset=5, k_shape=0.4,
                        home_advantage=0.0, surface_weight=args.poids_surface)
        surf_col = "surface" if "surface" in hist.columns else None
        elo.fit_stream(hist, surface_col=surf_col, record=False)
        p_match = (args.p_match if args.p_match is not None
                   else elo.predict(args.home, args.away, surface=args.surface, neutral=True))
        print(f"\n   ELO {args.home} {elo.get(args.home, args.surface):.0f} "
              f"vs {args.away} {elo.get(args.away, args.surface):.0f} "
              f"-> p({args.home}) = {p_match:.4f}")
        tm = TennisMatchModel(tour=args.tour)
        pred = tm.predict(args.home, args.away, surface=args.surface,
                          best_of=args.bo, p_match=p_match)
        marches = P.tennis_markets(pred, args.home, args.away)
        # Le tennis n'expose pas de distribution de scores comparable :
        # la heatmap ne s'y applique pas, on ne la transmet donc pas.
        sd_rapport, scorers_rapport = None, None

    else:
        print(f"Sport inconnu : {sport}")
        return 1

    _render(marches)

    if args.json:
        import json
        from pathlib import Path
        Path(args.json).write_text(
            json.dumps(marches, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"\n   Marches ecrits dans {args.json}")

    if args.html:
        from .report import write_report
        chemin = write_report(
            args.html, home=args.home, away=args.away, marches=marches,
            sd=sd_rapport, scorers=scorers_rapport,
            competition=(args.league if sport == "football" and not args.historique else sport),
            sport=sport,
            meta={"modele": type(model).__name__},
        )
        print(f"\n   Rapport HTML : {chemin}")
        print("   Ouvre-le dans un navigateur (double-clic sur le fichier).")
        if args.ouvrir:
            import webbrowser
            webbrowser.open("file:///" + chemin.replace("\\", "/"))
    return 0


def _date_paris(dt) -> str:
    """
    dd/mm/yyyy en heure de Paris a partir d'un horodatage UTC.

    Les coups d'envoi sont stockes en UTC pour les calculs de direct, mais la
    journee est groupee et affichee en heure locale : un match argentin a
    00:30 UTC appartient au meme jour pour un lecteur parisien (02:30), pas
    a la veille.
    """
    if dt is None or pd.isna(dt):
        return ""
    try:
        from datetime import timezone
        from zoneinfo import ZoneInfo
        d = pd.Timestamp(dt).to_pydatetime()
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y")
    except Exception:
        return pd.Timestamp(dt).strftime("%d/%m/%Y")


def cmd_journee(args) -> int:
    """
    Vue journee : tous les matchs a venir, plusieurs sports, une seule page.

    Le modele est ajuste UNE FOIS par championnat puis applique a toutes ses
    rencontres : c'est ce qui rend la commande utilisable sur cinq ligues sans
    attendre cinq minutes.
    """
    import pickle
    from pathlib import Path

    from .data import fixtures as fx
    from .data.results_csv import load_results, load_players as load_players_csv
    from .models.football_dc import DixonColesModel
    from .models.linear_score import LinearScoreModel
    from .models.rugby import RugbyModel
    from .models.tennis import TennisMatchModel
    from .models.elo import EloRating
    from .models.scorers import ScorerModel, PRIORS_FOOT, PRIORS_RUGBY
    from . import predict as P
    from .report import write_slate

    sport = args.sport
    slate = []

    # --- reprise d'une journee existante (pour empiler plusieurs sports)
    sidecar = Path(args.html).with_suffix(".slate")
    if args.ajouter and sidecar.exists():
        slate = pickle.loads(sidecar.read_bytes())
        print(f"   {len(slate)} match(s) deja presents, ajout en cours")

    if sport == "football":
        from .data import football_data as fd

        codes = fx.expand_presets([c for c in args.leagues.split(",") if c.strip()])
        if len(codes) > 6:
            print(f"   {len(codes)} championnats : {', '.join(codes)}")
        title("MATCHS A VENIR")
        a_venir = pd.DataFrame()
        if args.source == "apifootball":
            try:
                a_venir = fx.upcoming_from_apifootball(codes, args.cle_football)
            except Exception as e:
                print(f"   Calendrier API-Football indisponible ({e})")
        elif args.source in ("auto", "api"):
            try:
                a_venir = fx.upcoming_from_oddsapi(codes, args.cle)
            except Exception as e:
                print(f"   Calendrier live indisponible ({e})")
        if a_venir.empty and args.source in ("auto", "csv"):
            print("   Repli sur fixtures.csv de football-data.")
            print("   [!] Ce fichier est mis a jour a la main et peut etre fige :")
            print("       verifie les dates ci-dessous avant de t'y fier.")
            a_venir = fx.upcoming_football(codes)
        if a_venir.empty:
            print("   Rien a predire. Utilise --matchs pour fournir tes propres rencontres.")
            return 1

        squads = None
        if args.buteurs:
            from .data import fpl
            print("\n   Chargement des statistiques joueurs (FPL, Premier League)")
            squads = fpl.load_players()

        for code in codes:
            sous = a_venir[a_venir["league_code"].str.upper() == code]
            if sous.empty:
                continue
            title(f"{code} — {fd.nom_competition(code)}")
            # On ajuste sur le championnat ET sa division inferieure : sans
            # cela, un promu avec 3 matchs recoit une note aberrante.
            codes_fit = (fd.with_lower_tier([code])
                         if fd.has_lower_tier(code) and not args.sans_division_inf
                         else [code])
            hist = fd.load_any(codes_fit, fd.season_codes(args.depuis, args.jusqu),
                               depuis=args.depuis, verbose=False)
            if hist.empty:
                print("   Pas d'historique, championnat ignore.")
                continue
            try:
                model = DixonColesModel(xi=args.xi).fit(hist)
            except (ValueError, RuntimeError) as e:
                print(f"   Modele impossible : {e}")
                continue
            connues = set(hist["home"]) | set(hist["away"])

            for r in sous.itertuples(index=False):
                # Les noms du calendrier et ceux de l'historique ne coincident
                # pas ("Nottingham Forest" contre "Nott'm Forest") : on resout
                # avant de chercher l'equipe dans le modele.
                h = fx.resolve_team(r.home, connues)
                a = fx.resolve_team(r.away, connues)
                if h is None or a is None:
                    inconnu = r.home if h is None else r.away
                    print(f"   [!] {r.home} - {r.away} : '{inconnu}' absent de l'historique, ignore")
                    continue
                sd = model.predict(h, a)
                sc = None
                if squads is not None and code == "E0":
                    inv = {v: k for k, v in P.FPL_TO_FOOTBALLDATA.items()}
                    from .data import fpl
                    sh = fpl.team_squad(squads, inv.get(h, h))
                    sa = fpl.team_squad(squads, inv.get(a, a))
                    if not sh.empty and not sa.empty:
                        sm = ScorerModel(PRIORS_FOOT, k_shrink=args.k_shrink)
                        sc = sm.predict_match(sh, sa, sd.expected_home, sd.expected_away,
                                              fpl.estimate_minutes(sh), fpl.estimate_minutes(sa))
                slate.append({
                    "sport": "football", "competition": fd.nom_competition(code),
                    "date": _date_paris(r.date),
                    "coup_envoi": r.date.to_pydatetime() if hasattr(r.date, "to_pydatetime") else r.date,
                    "home": h, "away": a,
                    "marches": P.football_markets(sd, sc), "sd": sd, "scorers": sc,
                })
                p1 = P.football_markets(sd)["1x2"]
                print(f"   {h} - {a}   1 {100*p1[chr(49)]:.0f}%  "
                      f"X {100*p1[chr(88)]:.0f}%  2 {100*p1[chr(50)]:.0f}%")
    elif sport == "nrl":
        # Seule discipline hors football a disposer d'un historique libre ET de
        # donnees joueurs : tout est automatique, rien a saisir a la main.
        from .data import nrl as NRL

        title("HISTORIQUE NRL")
        hist = NRL.load_results(args.depuis_nrl)
        joueurs = NRL.load_player_stats(hist)
        connues = set(hist["home"]) | set(hist["away"])

        title("CALENDRIER NRL")
        try:
            evs = fx.nrl_fixtures(args.cle)
        except Exception as e:
            print(f"   Calendrier indisponible : {e}")
            return 1
        print(f"   {len(evs)} match(s) a venir")

        # part_essais plus elevee qu'au XV : le rugby a XIII marque davantage
        # par essais et beaucoup moins par penalites.
        model = RugbyModel(part_essais=args.part_essais_nrl,
                           half_life_days=args.demi_vie_rugby).fit(hist)
        sm = ScorerModel(PRIORS_RUGBY, k_shrink=args.k_shrink,
                         full_match_minutes=80.0, own_goal_share=0.0, penalty_rate=0.0)

        for e in evs:
            h = NRL.resolve_team(e["home"], connues)
            a = NRL.resolve_team(e["away"], connues)
            if h is None or a is None:
                inconnu = e["home"] if h is None else e["away"]
                print(f"   [!] {e['home']} - {e['away']} : '{inconnu}' inconnu, ignore")
                continue
            sd = model.predict(h, a)
            sh = joueurs[joueurs["equipe"] == h]
            sa = joueurs[joueurs["equipe"] == a]
            ts = None
            if not sh.empty and not sa.empty:
                th, ta = sd.expected_tries
                ts = sm.predict_match(sh, sa, th, ta)
            marches = P.rugby_markets(sd, ts)
            slate.append({"sport": "rugby", "competition": "NRL",
                          "date": _date_paris(e["date"]),
                          # Horodatage complet : indispensable pour calculer le
                          # temps ecoule lors d'un rafraichissement en direct.
                          "coup_envoi": e["date"],
                          "home": h, "away": a, "marches": marches,
                          "sd": sd, "scorers": ts})
            v = marches["vainqueur"]
            print(f"   {h} - {a}   1 {100*v['1']:.0f}%  2 {100*v['2']:.0f}%   "
                  f"[{sd.expected_home:.0f}-{sd.expected_away:.0f}] "
                  f"{marches['essais_attendus']['total']:.1f} essais")
    elif sport in ("top14", "prod2"):
        # Historique reel via les tableaux croises de Wikipedia. Le calendrier,
        # lui, n'a aucune source libre : il se fournit en CSV.
        from .data import rugby_fr as RF

        comp_tpl = RF.TOP14 if sport == "top14" else RF.PROD2
        title("HISTORIQUE " + sport.upper())
        saisons = tuple(x.strip() for x in args.saisons.split(",") if x.strip())
        hist = (RF.load_top14(saisons) if sport == "top14" else RF.load_prod2(saisons))

        # Saison en cours : ses resultats deja joues comptent double en
        # pertinence (ce sont les plus recents), et ses tableaux par journee
        # sont la seule source libre du calendrier.
        joues, a_venir_wiki = RF.load_current_season(comp_tpl, args.saison_courante)
        if not joues.empty:
            hist = pd.concat([hist, joues], ignore_index=True).sort_values("date")
        if hist.empty:
            print("   Aucune donnee recuperee.")
            return 1
        connues = set(hist["home"]) | set(hist["away"])

        if not args.matchs and not a_venir_wiki.empty:
            # On ecarte d'abord les journees DEJA DISPUTEES, puis on prend la
            # premiere restante. Se contenter du plus petit numero de journee
            # non renseignee ne suffit pas : Wikipedia met les scores a jour
            # avec du retard, donc une journee jouee hier y figure encore comme
            # "a venir" et on afficherait des pronostics sur des matchs finis.
            auj = pd.Timestamp.now().normalize()
            futur = a_venir_wiki[a_venir_wiki["date"] >= auj]
            n_passe = len(a_venir_wiki) - len(futur)
            if n_passe:
                print(f"   {n_passe} match(s) deja dispute(s) ecarte(s) du calendrier")
            if futur.empty:
                print("   Plus aucune rencontre a venir dans le calendrier publie.")
                return 1
            prochaine = futur["journee"].min()
            a_venir = futur[futur["journee"] == prochaine].copy()
            title(f"JOURNEE {int(prochaine)} — {a_venir['date'].iloc[0]:%d/%m/%Y}")
            print(f"   {len(a_venir)} match(s), calendrier Wikipedia")
        elif not args.matchs:
            print("\n   Il manque le calendrier : ni la LNR ni aucune API libre ne le publient.")
            print("   Fournis-le en CSV (home,away[,date]) :")
            print(f"      py -m sportvalue template-matchs matchs_{sport}.csv")
            print("\n   Equipes reconnues :")
            for e in sorted(connues):
                print("      " + e)
            return 1

        else:
            title("MATCHS A PREDIRE")
            a_venir = fx.load_fixtures_csv(args.matchs)

        model = RugbyModel(part_essais=args.part_essais,
                           half_life_days=args.demi_vie_rugby).fit(hist)
        print("\nMODELE " + str(model.base.params_summary()))
        comp = "Top 14" if sport == "top14" else "Pro D2"

        for r in a_venir.itertuples(index=False):
            h = RF.resolve_team(r.home, connues)
            a = RF.resolve_team(r.away, connues)
            if h is None or a is None:
                inconnu = r.home if h is None else r.away
                print(f"   [!] {r.home} - {r.away} : '{inconnu}' inconnu, ignore")
                continue
            ctx = {}
            if args.meteo:
                from .data import weather as wx
                from datetime import datetime as _dt, timedelta as _td
                ctx["weather"] = wx.venue_weather(h, _dt.now() + _td(days=args.dans_jours))
            sd = model.predict(h, a, context=ctx)
            marches = P.rugby_markets(sd)
            # Une journee s'etale sur deux jours et Wikipedia ne donne pas la
            # date match par match : on transporte la PLAGE plutot que
            # d'affirmer un jour precis pour chaque rencontre.
            fin = getattr(r, "date_fin", None)
            slate.append({"sport": "rugby", "competition": comp,
                          "date": r.date.strftime("%d/%m/%Y") if pd.notna(r.date) else "",
                          "date_fin": fin.strftime("%d/%m/%Y")
                                      if fin is not None and pd.notna(fin) else "",
                          "home": h, "away": a, "marches": marches,
                          "sd": sd, "scorers": None})
            v = marches["vainqueur"]
            print(f"   {h} - {a}   1 {100*v['1']:.0f}%  2 {100*v['2']:.0f}%   "
                  f"[{sd.expected_home:.0f}-{sd.expected_away:.0f}] "
                  f"{marches['essais_attendus']['total']:.1f} essais")
    else:
        if not (args.historique and args.matchs):
            print(f"Pour le {sport}, il faut --historique (resultats passes) "
                  "et --matchs (rencontres a predire).")
            print(f"   py -m sportvalue template-matchs matchs_{sport}.csv")
            return 1
        title("HISTORIQUE")
        hist = load_results(args.historique)
        title("MATCHS A PREDIRE")
        a_venir = fx.load_fixtures_csv(args.matchs)
        joueurs = load_players_csv(args.joueurs, sport) if args.joueurs else None
        connues = set(hist["home"]) | set(hist["away"])

        if sport == "rugby":
            model = RugbyModel(part_essais=args.part_essais,
                           half_life_days=args.demi_vie_rugby).fit(hist)
        elif sport == "basket":
            model = LinearScoreModel(half_life_days=args.demi_vie).fit(hist)
        else:
            elo = EloRating(k0=250, k_offset=5, k_shape=0.4,
                            home_advantage=0.0, surface_weight=0.5)
            elo.fit_stream(hist, surface_col="surface" if "surface" in hist.columns else None,
                           record=False)
            model = TennisMatchModel(tour=args.tour)

        for r in a_venir.itertuples(index=False):
            if r.home not in connues or r.away not in connues:
                print(f"   [!] {r.home} - {r.away} : inconnu de l'historique, ignore")
                continue
            date = r.date.strftime("%d/%m/%Y") if pd.notna(r.date) else ""
            if sport == "rugby":
                ctx = {}
                if args.meteo:
                    from .data import weather as wx
                    from datetime import datetime as _dt, timedelta as _td
                    ctx["weather"] = wx.venue_weather(r.home, _dt.now() + _td(days=args.dans_jours))
                sd = model.predict(r.home, r.away, context=ctx)
                ts = None
                if joueurs is not None:
                    sh = joueurs[joueurs["equipe"].str.lower() == r.home.lower()]
                    sa = joueurs[joueurs["equipe"].str.lower() == r.away.lower()]
                    if not sh.empty and not sa.empty:
                        sm = ScorerModel(PRIORS_RUGBY, k_shrink=args.k_shrink,
                                         full_match_minutes=80.0, own_goal_share=0.0,
                                         penalty_rate=0.0)
                        th, ta = sd.expected_tries
                        ts = sm.predict_match(sh, sa, th, ta)
                marches, obj, sco = P.rugby_markets(sd, ts), sd, ts
            elif sport == "basket":
                sd = model.predict(r.home, r.away, allow_draw=False)
                marches, obj, sco = P.basket_markets(sd), sd, None
            else:
                p = elo.predict(r.home, r.away, surface=args.surface, neutral=True)
                pred = model.predict(r.home, r.away, surface=args.surface,
                                     best_of=args.bo, p_match=p)
                marches, obj, sco = P.tennis_markets(pred, r.home, r.away), None, None

            slate.append({"sport": sport, "competition": args.competition or sport,
                          "date": date, "home": r.home, "away": r.away,
                          "marches": marches, "sd": obj, "scorers": sco})
            print(f"   {r.home} - {r.away}")

    # --- absences (blessures, suspensions) ---------------------------------
    if args.absences:
        from .data import injuries as INJ

        title("ABSENCES")
        # API-Football ne couvre QUE le football. Sans ce filtre, les noms
        # d'equipes se telescopent d'un sport a l'autre : les blessures de
        # l'OGC Nice se retrouvaient collees au Nice - Grenoble de Pro D2.
        foot = [m for m in slate if m.get("sport") == "football"]
        equipes = {m["home"] for m in foot} | {m["away"] for m in foot}
        dates_iso = sorted({f"{m['date'][6:]}-{m['date'][3:5]}-{m['date'][:2]}"
                            for m in foot if len(m.get("date", "")) == 10})
        try:
            inj = INJ.fetch_injuries(dates_iso, args.cle_football,
                                     leagues=list(INJ.LEAGUE_IDS))
        except Exception as e:
            print(f"   Indisponible : {e}")
            inj = []

        if inj:
            par_equipe = {}
            for eq, lst in INJ.by_team(inj).items():
                r = INJ.resolve_team(eq, equipes)
                if r:
                    par_equipe.setdefault(r, []).extend(lst)
            n_enrichis = 0
            for m in foot:
                blocs = []
                for cote, eq in (("domicile", m["home"]), ("exterieur", m["away"])):
                    blocs.extend({**x, "cote": cote} for x in par_equipe.get(eq, []))
                if blocs:
                    m["marches"]["absences"] = blocs
                    n_enrichis += 1
            print(f"   {len(par_equipe)} equipe(s) appariee(s), "
                  f"{n_enrichis} match(s) enrichi(s)")

    # Garde-fou final. Il couvre aussi les matchs empiles lors d'executions
    # precedentes (--ajouter), qui ont pu basculer dans le passe depuis : sans
    # lui, la page garde indefiniment des pronostics sur des matchs termines.
    auj = pd.Timestamp.now().normalize()

    def _encore_a_jouer(m):
        try:
            return pd.to_datetime(m.get("date", ""), format="%d/%m/%Y") >= auj
        except (ValueError, TypeError):
            return True   # date illisible : on garde plutot que de perdre le match

    avant = len(slate)
    slate = [m for m in slate if _encore_a_jouer(m)]
    if avant != len(slate):
        print(f"\n   {avant - len(slate)} match(s) deja joue(s) retire(s) de la journee")

    if not slate:
        print("\n   Aucun match predit.")
        return 1

    sidecar.write_bytes(pickle.dumps(slate))
    chemin = write_slate(args.html, slate, titre=args.titre)
    title(f"{len(slate)} MATCH(S)")
    print(f"   Rapport : {chemin}")
    print(f"   Etat conserve dans {sidecar.name} — relance avec --ajouter")
    print("   pour empiler un autre sport sur la meme page.")
    if args.ouvrir:
        import webbrowser
        webbrowser.open("file:///" + chemin.replace("\\", "/"))
    return 0


def cmd_europe(args) -> int:
    """
    Coupes d'Europe. Ajustement CONJOINT sur plusieurs championnats, plus un
    decalage de force par championnat.

    Le decalage est un PRIOR pose a la main, pas une estimation : rien dans les
    donnees domestiques ne permet de comparer deux pays, faute d'equipes jouant
    dans les deux. La commande le rappelle a chaque execution.
    """
    import pickle
    from pathlib import Path

    import numpy as np

    from .data import football_data as fd
    from .data import uefa
    from .models.football_dc import DixonColesModel
    from . import predict as P
    from .report import write_slate

    codes = [c.strip().upper() for c in args.leagues.split(",") if c.strip()]

    title("CALENDRIER EUROPEEN")
    try:
        matchs = uefa.european_fixtures(args.competition, args.cle)
    except Exception as e:
        print(f"   Impossible de recuperer le calendrier : {e}")
        return 1
    nom_comp = uefa.COMPETITIONS_EUROPE.get(args.competition, ("", "Europe"))[1]
    print(f"   {len(matchs)} match(s) — {nom_comp}")
    if not matchs:
        return 1

    title("HISTORIQUE (ajustement conjoint)")
    hist = fd.load_many(codes, fd.season_codes(args.depuis, args.jusqu))
    if hist.empty:
        print("   Aucune donnee.")
        return 1

    # Championnat d'appartenance de chaque equipe : sa derniere division vue.
    league_of = (
        hist.assign(eq=hist["home"])[["eq", "league_code", "date"]]
        .sort_values("date")
        .drop_duplicates("eq", keep="last")
        .set_index("eq")["league_code"]
        .to_dict()
    )
    connues = set(hist["home"]) | set(hist["away"])

    print("\n   Ajustement Dixon-Coles conjoint sur "
          f"{len(codes)} championnats, {len(hist)} matchs...")
    model = DixonColesModel(xi=args.xi).fit(hist)

    title("DECALAGES DE FORCE APPLIQUES  (prior explicite, non estime)")
    print(table(
        [{"championnat": fd.LEAGUES.get(c, c), "code": c,
          "decalage_log_buts": uefa.FORCE_CHAMPIONNAT.get(c, 0.0),
          "effet_sur_les_buts": f"x{np.exp(uefa.FORCE_CHAMPIONNAT.get(c, 0.0)):.3f}"}
         for c in codes],
        ["championnat", "code", "decalage_log_buts", "effet_sur_les_buts"]))
    print("\n   Ces valeurs ne sortent PAS des donnees : aucune equipe ne joue dans")
    print("   deux championnats, donc leur force relative n'est pas estimable.")
    print("   Pour l'estimer vraiment, il faut un historique contenant des matchs")
    print("   europeens : les deux echelles s'y soudent d'elles-memes.")

    slate, ignores = [], []
    sidecar = Path(args.html).with_suffix(".slate")
    if args.ajouter and sidecar.exists():
        slate = pickle.loads(sidecar.read_bytes())

    title("PREDICTIONS")
    for m in matchs:
        h = uefa.resolve_team(m["home"], connues)
        a = uefa.resolve_team(m["away"], connues)
        if h is None or a is None:
            manquant = m["home"] if h is None else m["away"]
            if h is None and a is None:
                manquant = f"{m['home']} et {m['away']}"
            ignores.append((f"{m['home']} - {m['away']}", manquant))
            continue

        lh, la = league_of.get(h), league_of.get(a)
        oh = uefa.FORCE_CHAMPIONNAT.get(lh, 0.0)
        oa = uefa.FORCE_CHAMPIONNAT.get(la, 0.0)
        ctx = {
            "home_attack_mult": float(np.exp(oh)),
            "home_defence_mult": float(np.exp(-oh)),
            "away_attack_mult": float(np.exp(oa)),
            "away_defence_mult": float(np.exp(-oa)),
        }
        sd = model.predict(h, a, neutral=False, context=ctx)
        marches = P.football_markets(sd)
        p = marches["1x2"]
        print(f"   {h} ({lh}) - {a} ({la})   "
              f"1 {100*p['1']:.1f}%  X {100*p['X']:.1f}%  2 {100*p['2']:.1f}%   "
              f"[{sd.expected_home:.2f} - {sd.expected_away:.2f}]")
        slate.append({
            "sport": "football", "competition": nom_comp,
            "date": m["date"].strftime("%d/%m/%Y"),
            "home": h, "away": a, "marches": marches, "sd": sd, "scorers": None,
        })

    if ignores:
        title(f"{len(ignores)} MATCH(S) NON PREDIT(S)")
        print("   Ces clubs ne jouent dans aucun championnat couvert, donc le modele")
        print("   n'a aucune donnee sur eux. Mieux vaut ne rien annoncer.")
        for match, qui in ignores:
            print(f"   {match:<44} — inconnu : {qui}")
        print("\n   Pour les couvrir, ajoute leur championnat a --leagues s'il existe")
        print("   chez football-data, ou fournis un historique via `predict --historique`.")

    if not slate:
        print("\n   Aucun match predit.")
        return 1

    sidecar.write_bytes(pickle.dumps(slate))
    chemin = write_slate(args.html, slate, titre=f"{nom_comp}")
    print(f"\n   Rapport : {chemin}")
    if args.ouvrir:
        import webbrowser
        webbrowser.open("file:///" + chemin.replace("\\", "/"))
    return 0


def cmd_live(args) -> int:
    """
    Rafraichit une journee deja calculee avec les scores en cours.

    La page publiee ne peut pas se rafraichir seule (pas d'appel reseau
    autorise cote navigateur) : on la REGENERE. Les matchs en cours voient
    leurs probabilites recalculees conditionnellement au score et au temps
    restant, les matchs termines recoivent leur score final.
    """
    import pickle
    from datetime import datetime, timezone
    from pathlib import Path

    from .data import scores as SC
    from .data.fixtures import resolve_team
    from . import live as LV
    from . import predict as P
    from .report import write_slate

    sidecar = Path(args.html).with_suffix(".slate")
    if not sidecar.exists():
        print(f"Aucune journee a rafraichir ({sidecar.name} introuvable).")
        print("   Genere-la d'abord avec `journee`.")
        return 1
    slate = pickle.loads(sidecar.read_bytes())
    title(f"RAFRAICHISSEMENT — {len(slate)} match(s)")

    # N'interroger QUE les competitions ayant un match plausiblement en cours.
    # La journee connait l'heure de chaque coup d'envoi : un match est
    # candidat s'il a commence il y a moins de (duree + 30 min). Interroger
    # les six competitions couvertes a chaque appel brulait un credit par
    # competition sans match, soit la majorite du quota pour rien.
    from datetime import timedelta as _td

    maintenant = datetime.now(timezone.utc)
    # Restreindre le live a un sous-ensemble de competitions plafonne le cout :
    # un samedi, vingt championnats peuvent avoir un match en meme temps, soit
    # vingt credits par rafraichissement. Par defaut on suit les cinq grands,
    # la Ligue 2 et la NRL.
    from .data import football_data as fd
    from .data.fixtures import expand_presets
    autorisees = None
    if args.leagues:
        codes = expand_presets([c for c in args.leagues.split(",") if c.strip()])
        autorisees = {fd.nom_competition(c) for c in codes} | {"NRL"}
    cles = set()
    for m in slate:
        comp = m.get("competition")
        ke = m.get("coup_envoi")
        if comp not in SC.SPORT_KEYS_SCORES or ke is None:
            continue
        if autorisees is not None and comp not in autorisees:
            continue
        if ke.tzinfo is None:
            ke = ke.replace(tzinfo=timezone.utc)
        duree = LV.DUREES.get(m.get("sport")) or 90.0
        if ke <= maintenant <= ke + _td(minutes=duree + 30) or args.tout:
            cles.add(SC.SPORT_KEYS_SCORES[comp])
    if not cles:
        print("   Aucun match en cours d'apres les heures de coup d'envoi :")
        print("   aucun credit consomme. (--tout pour forcer toutes les competitions)")
        return 0
    print(f"   {len(cles)} competition(s) avec match en cours : {', '.join(sorted(cles))}")

    try:
        sc = SC.fetch_scores(cles, args.cle)
    except Exception as e:
        print(f"   Scores indisponibles : {e}")
        return 1

    index = {}
    for (h, a), v in sc.items():
        index[(h.lower(), a.lower())] = v
    equipes = {m["home"] for m in slate} | {m["away"] for m in slate}
    for (h, a), v in sc.items():
        rh, ra = resolve_team(h, equipes), resolve_team(a, equipes)
        if rh and ra:
            index[(rh.lower(), ra.lower())] = v

    n_live, n_fin = 0, 0
    maintenant = datetime.now(timezone.utc)
    for m in slate:
        v = index.get((str(m["home"]).lower(), str(m["away"]).lower()))
        if not v:
            continue
        m["live"] = v
        hs, as_ = v.get("home_score"), v.get("away_score")
        if v["statut"] == "termine":
            n_fin += 1
            continue
        if v["statut"] != "en_cours" or hs is None or as_ is None:
            continue

        duree = LV.DUREES.get(m.get("sport"), 90.0)
        debut = m.get("coup_envoi")
        mins = LV.minutes_ecoulees(debut, maintenant, duree or 90.0) if debut else None
        if mins is None:
            # Sans heure de coup d'envoi, on ne peut pas conditionner : on
            # affiche le score sans toucher aux probabilites, plutot que
            # d'inventer un temps ecoule.
            m["live"]["minutes"] = None
            continue

        sd = LV.recalcule_conditionnel(m, int(hs), int(as_), mins)
        if sd is None:
            continue
        if m.get("sport") == "rugby":
            m["marches"] = P.rugby_markets(sd)
        elif m.get("sport") == "football":
            m["marches"] = P.football_markets(sd)
        elif m.get("sport") == "basket":
            m["marches"] = P.basket_markets(sd)
        m["sd"] = sd
        m["live"]["minutes"] = round(mins)
        m["marches"]["conditionnel"] = [
            f"Probabilites recalculees sachant {hs}-{as_} a la {round(mins)}e minute.",
            "Le temps ecoule est deduit des horodatages, pas lu sur un chrono : "
            "une erreur de quelques minutes deplace sensiblement le resultat.",
        ]
        n_live += 1
        print(f"   [en cours] {m['home']} {hs}-{as_} {m['away']}  "
              f"~{round(mins)}e min -> probabilites recalculees")

    print(f"\n{n_live} match(s) recalcule(s), {n_fin} termine(s)")
    sidecar.write_bytes(pickle.dumps(slate))
    chemin = write_slate(args.html, slate, titre=args.titre)
    print(f"   Rapport : {chemin}")
    return 0


def cmd_template_matchs(args) -> int:
    from .data.fixtures import write_fixtures_template

    p = write_fixtures_template(args.fichier)
    print(f"Modele de matchs a predire ecrit dans {p}")
    print("Colonnes : home, away [, date]")
    return 0


def cmd_template_resultats(args) -> int:
    from .data.results_csv import write_results_template, write_players_template

    p = write_results_template(args.fichier)
    print(f"Modele d'historique ecrit dans {p}")
    if args.joueurs:
        p2 = write_players_template(args.joueurs)
        print(f"Modele de statistiques joueurs ecrit dans {p2}")
    print("\nColonnes obligatoires : date, home, away, home_score, away_score")
    print("Optionnelles : neutral (terrain neutre), surface (tennis)")
    return 0


def cmd_sports(args) -> int:
    """Liste les competitions disponibles. Gratuit : ne consomme aucun credit."""
    from .data.oddsapi import OddsAPIClient, SPORT_KEYS

    cli = OddsAPIClient(api_key=args.cle)
    if not cli.configured:
        title("COMPETITIONS COURANTES (liste locale)")
        for sport, keys in SPORT_KEYS.items():
            print(f"\n   {sport}")
            for k in keys:
                print(f"      {k}")
        print("\n   Definis ODDS_API_KEY pour interroger la liste complete et a jour.")
        return 0

    sports = cli.list_sports()
    if args.filtre:
        f = args.filtre.lower()
        sports = [s for s in sports if f in (s["key"] + str(s["titre"])).lower()]
    title(f"COMPETITIONS ACTIVES ({len(sports)})")
    print(table(sports, ["key", "groupe", "titre", "actif"], max_width=42))
    print(f"\n   Quota : {cli.quota_status()}")
    return 0


def cmd_scan(args) -> int:
    """Scan automatique via The Odds API, region `fr` (operateurs ARJEL)."""
    from .data.oddsapi import OddsAPIClient, QuotaExhausted, SPORT_KEYS

    keys = (
        [k.strip() for k in args.competitions.split(",") if k.strip()]
        if args.competitions
        else SPORT_KEYS.get(args.sport, [])
    )
    if not keys:
        print(f"Aucune competition. Utilise --competitions, ou --sport parmi {list(SPORT_KEYS)}.")
        return 1

    regions = ["fr"] if args.sans_reference else ["eu", "fr"]
    cout = len(keys) * len(regions)
    title("COUT EN CREDITS")
    print(f"   {len(keys)} competition(s) x {len(regions)} region(s) = {cout} credits par scan")
    if "eu" in regions:
        print("   la region eu apporte Pinnacle, soit la reference de verite")
    print(f"   Palier gratuit : 500 credits/mois -> a raison d'un scan par jour, "
          f"{cout * 30} credits/mois.")
    if cout * 30 > 500:
        print("   [!] Ce rythme depasse le quota gratuit. Reduis le nombre de competitions")
        print("       ou ne scanne que les jours de match.")
    if not args.oui:
        try:
            rep = input("\n   Lancer le scan ? [o/N] ").strip().lower()
        except EOFError:
            rep = "n"
        if rep not in ("o", "oui", "y", "yes"):
            print("   Annule.")
            return 0

    cli = OddsAPIClient(api_key=args.cle, ttl=args.ttl)
    if not cli.configured:
        print("\n   Aucune cle API configuree. Deux options :")
        print("      1. cle gratuite sur https://the-odds-api.com puis  set ODDS_API_KEY=...")
        print("      2. releve manuel :  py -m sportvalue template cotes.csv")
        return 1

    cfg = ScannerConfig(
        min_probability=args.p_min, min_edge=args.edge_min, blend_weight=args.w,
        bankroll=args.bankroll, kelly_fraction=args.kelly, max_stake_pct=args.max_mise,
        min_trj=args.trj_min, allow_no_reference=args.sans_reference,
    )
    scanner = ValueScanner(cfg)

    all_bets, n_events, erreurs, diagnostics = [], 0, [], []
    for key in keys:
        try:
            events = cli.fixtures_with_quotes(key, args.sport, market=args.marche, regions=regions)
        except QuotaExhausted as e:
            print(f"\n   [!] {e}")
            break
        except Exception as e:
            erreurs.append(f"{key} : {type(e).__name__} - {e}")
            continue
        n_events += len(events)
        for ev in events:
            n = len(ev["arjel"][0].selections)
            p_plat = np.full(n, 1.0 / n)
            all_bets.extend(
                scanner.find_value(
                    ev["fixture"], args.marche, p_plat, ev["reference"], ev["arjel"],
                )
            )
            # Diagnostic : on garde la trace de TOUTES les issues analysees,
            # avec le motif de rejet. Sans cela, un scan sans resultat est
            # indiscernable d'un scan casse.
            scan = scanner.scan_market(p_plat, ev["reference"], ev["arjel"])
            if scan is None:
                continue
            for i, sel in enumerate(scan.selections):
                p_fin = float(scan.p_final[i])
                odds = float(scan.best_odds[i])
                edge = p_fin * odds - 1.0
                if scan.trj_arjel < cfg.min_trj:
                    motif = f"TRJ {100*scan.trj_arjel:.1f}% < {100*cfg.min_trj:.0f}%"
                elif p_fin < cfg.min_probability:
                    motif = f"proba {100*p_fin:.1f}% < {100*cfg.min_probability:.0f}%"
                elif edge < cfg.min_edge:
                    motif = f"edge {100*edge:+.2f}% < {100*cfg.min_edge:.1f}%"
                elif cfg.require_robust_devig and scan.robust_min_edge[i] <= 0:
                    motif = "edge non robuste au de-vig"
                else:
                    motif = "RETENU"
                diagnostics.append({
                    "match": ev["fixture"].label()[:32],
                    "issue": sel[:16],
                    "p": round(p_fin, 3),
                    "cote": odds,
                    "book": scan.best_books[i].replace("_fr", ""),
                    "edge_%": round(100 * edge, 2),
                    "reference": scan.reference_book,
                    "motif": motif,
                })

    title(f"SCAN : {n_events} evenement(s) cotes chez au moins un book ARJEL, "
          f"{len(all_bets)} value detectee(s)")
    for e in erreurs:
        print(f"   [!] {e}")
    if not all_bets:
        print("   Aucun pari ne passe les filtres.")
        print("   C'est le resultat NORMAL la plupart du temps : le backtest donne")
        print("   environ 20 paris pour 1000 matchs aux seuils utilisables.")
    else:
        ranked = ValueScanner.rank(all_bets, top=args.top)
        print(table([b.to_row() for b in ranked],
                    ["date", "match", "pari", "book", "cote", "p_finale",
                     "edge_%", "mise", "confiance", "score"]))
        if args.details:
            for b in ranked:
                print(f"\n   {b.fixture.label()} | {b.selection} @ {b.odds:.2f} ({b.book})")
                for r in b.rationale:
                    print(f"      . {r}")

    # Diagnostic : un scan sans resultat doit rester lisible. Sans cette table,
    # un scan casse et un scan legitimement vide sont indiscernables.
    if diagnostics and (args.diagnostic or not all_bets):
        positifs = [d for d in diagnostics if d["edge_%"] > 0]
        title(f"DIAGNOSTIC : {len(diagnostics)} issues analysees, "
              f"{len(positifs)} a edge positif")
        top = sorted(diagnostics, key=lambda d: -d["edge_%"])[: args.top]
        print(table(top, ["match", "issue", "p", "cote", "book", "edge_%", "motif"],
                    max_width=30))
        if positifs and not all_bets:
            bas = [d for d in positifs if d["motif"].startswith("proba")]
            if len(bas) == len(positifs):
                print("\n   Tous les edges positifs portent sur des issues a FAIBLE probabilite.")
                print("   C'est structurel : les books francais s'ecartent surtout de la ligne")
                print("   sharp sur les outsiders, pas sur les favoris. Le filtre haute")
                print("   probabilite les rejette, et c'est coherent avec le classement par")
                print("   croissance esperee, qui penalise lourdement les cotes elevees.")

    if args.journal and all_bets:
        n = _log_bets(all_bets, args.journal)
        print(f"\n   {n} pari(s) ajoute(s) au journal {args.journal}")
        print("   Complete les colonnes cote_cloture et resultat apres les matchs :")
        print("   le CLV converge bien plus vite que le ROI pour trancher.")

    print(f"\n   Quota : {cli.quota_status()}")
    return 0


def _log_bets(bets, path: str) -> int:
    """
    Journalise les paris detectes pour un suivi ulterieur du CLV.

    Le CLV converge beaucoup plus vite que le ROI : quelques centaines de paris
    suffisent a savoir si la selection a un edge, contre plusieurs milliers pour
    le ROI. Journaliser systematiquement est donc la facon la moins chere de
    distinguer un edge reel d'une serie chanceuse.
    """
    from pathlib import Path

    rows = []
    for b in bets:
        r = b.to_row()
        r["horodatage"] = datetime.now().isoformat(timespec="seconds")
        r["cote_cloture"] = ""      # a completer apres le match
        r["resultat"] = ""          # gagne / perdu / rembourse
        rows.append(r)
    new = pd.DataFrame(rows)
    p = Path(path)
    if p.exists():
        new = pd.concat([pd.read_csv(p), new], ignore_index=True)
    new.to_csv(p, index=False)
    return len(rows)


def cmd_trj(args) -> int:
    odds = [float(x) for x in args.cotes]
    title("ANALYSE D'UN MARCHE")
    print(f"   cotes : {odds}")
    print(f"   overround : {sum(1/o for o in odds):.4f}")
    print(f"   marge du book : {100*margin_pct(odds):.2f}%")
    print(f"   TRJ : {100*trj(odds):.2f}%"
          + ("   (au-dessus du plafond legal moyen de 85%)" if trj(odds) > 0.85 else ""))
    print(f"   jouable (seuil 90%) : {'OUI' if trj(odds) >= 0.90 else 'NON'}")

    title("PROBABILITES FAIR SELON LA METHODE DE DE-VIG")
    rows = []
    for name, p in devig_all_methods(odds).items():
        row = {"methode": name}
        for i, v in enumerate(p):
            row[f"issue_{i+1}"] = round(float(v), 4)
            row[f"cote_fair_{i+1}"] = round(1 / float(v), 3)
        rows.append(row)
    cols = ["methode"] + [f"issue_{i+1}" for i in range(len(odds))] + \
           [f"cote_fair_{i+1}" for i in range(len(odds))]
    print(table(rows, cols))
    print("\n   L'ecart entre methodes donne l'ordre de grandeur de l'incertitude")
    print("   sur toute estimation d'edge. Un edge inferieur a cet ecart n'est pas fiable.")
    return 0


def cmd_backtest(args) -> int:
    from .data import football_data as fd
    from .backtest.engine import run_football_walkforward, simulate_bets, threshold_sweep

    leagues = args.leagues.split(",")
    seasons = fd.season_codes(args.depuis, args.jusqu)
    title("TELECHARGEMENT")
    df = fd.load_many(leagues, seasons)
    if df.empty:
        print("   Aucune donnee.")
        return 1

    title("BACKTEST WALK-FORWARD")
    ps = run_football_walkforward(df, refit_every_days=args.refit, xi=args.xi)

    title("QUALITE PROBABILISTE (hors echantillon)")
    q = ps.quality()
    print(table([{"metrique": k, "valeur": v} for k, v in q.items()], ["metrique", "valeur"]))
    if not q["fusion_bat_marche"]:
        print("\n   [!] Le modele n'ameliore pas la ligne du marche. C'est le cas normal")
        print("       face a un book sharp. Utilise l'outil en detecteur d'ecart")
        print("       sharp/ARJEL plutot qu'en modele predictif autonome.")

    title("CALIBRATION")
    print(table(ps.calibration("final"), ["bin", "n", "predit", "observe", "ecart", "ecart_en_se"]))

    title(f"SIMULATION (p>={args.p_min}, edge>={args.edge_min})")
    res = simulate_bets(ps, min_probability=args.p_min, min_edge=args.edge_min)
    if res.get("n", 0) == 0:
        print("   Aucun pari.")
    else:
        for k in ["n", "profit", "cote_moyenne", "proba_moyenne", "taux_reussite",
                  "t_stat", "max_drawdown_%", "bankroll_finale"]:
            print(f"   {k:<22} {res[k]}")
        print(f"   {'roi':<22} {100*res['roi']:+.2f}%")
        print(f"   {'ic95':<22} [{100*res['roi_ic95'][0]:+.2f}%, {100*res['roi_ic95'][1]:+.2f}%]")
        if res.get("clv"):
            c = res["clv"]
            print(f"   {'clv brut':<22} {100*c['clv_mean_pct']:+.2f}%")
            print(f"   {'clv du hasard':<22} {100*c['clv_aleatoire_pct']:+.2f}%  "
                  "(biais mecanique du shopping)")
            print(f"   {'clv NET':<22} {100*c['clv_net_pct']:+.2f}%  <- seul chiffre interpretable")

    if args.sweep:
        title("BALAYAGE DES SEUILS")
        sw = threshold_sweep(ps)
        print(sw.to_string(index=False))
        print("\n   Choisir la meilleure case apres coup est du sur-apprentissage.")
        print("   Regarder d'abord la colonne n.")
    return 0


def cmd_weather(args) -> int:
    from .data import weather as wx
    from .models.rugby import RugbyModel

    when = datetime.now() + timedelta(days=args.dans_jours)
    w = wx.venue_weather(args.stade, when)
    title(f"METEO : {args.stade}")
    print(f"   {wx.describe(w)}   [source : {w.get('source')}]")

    eff = RugbyModel().weather.apply(w)
    print(f"\n   Impact modelise :")
    print(f"      multiplicateur d'essais : x{eff['try_mult']:.3f}")
    print(f"      reussite au pied        : {eff['kick_delta']:+.3f}")
    print(f"      multiplicateur penalites: x{eff['pen_mult']:.3f}")
    for n in eff["notes"]:
        print(f"      . {n}")
    if not eff["notes"]:
        print("      aucune : conditions neutres, la meteo n'est pas un argument ici.")
    return 0


def cmd_books(args) -> int:
    from .data.arjel import AUTO_BOOKS, MANUAL_ONLY_BOOKS

    title("PERIMETRE AUTOMATIQUE (commande `scan`)")
    print(table(
        [{"cle": ARJEL_BOOKS[k].key, "nom": ARJEL_BOOKS[k].nom,
          "the_odds_api": ARJEL_BOOKS[k].oddsapi_key, "note": ARJEL_BOOKS[k].note}
         for k in AUTO_BOOKS],
        ["cle", "nom", "the_odds_api", "note"], max_width=48))
    print(f"\n   {len(AUTO_BOOKS)} operateurs relevables sans intervention manuelle.")
    print("   Backtest sur un portefeuille equivalent de 5 books grand public :")
    print("   ROI +2.17%, IC 95% [-2.57%, +7.03%] -- positif mais non significatif.")

    title("HORS PERIMETRE AUTOMATIQUE (saisie manuelle uniquement)")
    print(table(
        [{"cle": ARJEL_BOOKS[k].key, "nom": ARJEL_BOOKS[k].nom} for k in MANUAL_ONLY_BOOKS],
        ["cle", "nom"], max_width=48))
    print("\n   Exposes par aucune API. L'extraction automatisee est interdite par leurs")
    print("   CGU et bloquee par leur anti-bot, avec le risque de faire fermer le compte")
    print("   qui sert a miser. Utiliser `template` pour les inclure a la main.")
    print("\n   Liste officielle des agrements : https://anj.fr")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sportvalue",
        description="Detection de value sur books ARJEL : football, rugby, basket, tennis.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="ecrit un modele CSV de saisie des cotes")
    t.add_argument("fichier", nargs="?", default="cotes.csv")
    t.set_defaults(func=cmd_template)

    s = sub.add_parser("scan-csv", help="analyse un CSV de cotes releve a la main")
    s.add_argument("fichier")
    s.add_argument("--p-min", dest="p_min", type=float, default=0.55)
    s.add_argument("--edge-min", dest="edge_min", type=float, default=0.015)
    s.add_argument("--w", type=float, default=0.0, help="poids du modele (0 = reference sharp seule)")
    s.add_argument("--bankroll", type=float, default=1000.0)
    s.add_argument("--kelly", type=float, default=0.25)
    s.add_argument("--max-mise", dest="max_mise", type=float, default=0.02)
    s.add_argument("--trj-min", dest="trj_min", type=float, default=0.90)
    s.add_argument("--sans-reference", action="store_true",
                   help="autoriser l'analyse sans ligne sharp (moins fiable)")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--details", action="store_true")
    s.set_defaults(func=cmd_scan_csv)

    pr = sub.add_parser("predict", help="probabilites d'un match a partir des DONNEES (sans cotes)")
    pr.add_argument("--sport", required=True, choices=["football", "rugby", "basket", "tennis"])
    pr.add_argument("--home", required=True, help="equipe/joueur a domicile")
    pr.add_argument("--away", required=True, help="equipe/joueur a l'exterieur")
    pr.add_argument("--historique", default=None,
                    help="CSV de resultats. Obligatoire hors football.")
    pr.add_argument("--joueurs", default=None, help="CSV de statistiques joueurs")
    pr.add_argument("--neutre", action="store_true", help="terrain neutre")
    pr.add_argument("--json", default=None, help="ecrire les marches dans un fichier JSON")
    pr.add_argument("--html", default=None,
                    help="generer un rapport HTML consultable dans un navigateur")
    pr.add_argument("--ouvrir", action="store_true",
                    help="ouvrir directement le rapport HTML dans le navigateur")
    # football
    pr.add_argument("--league", default="E0", help="code football-data (E0, F1, D1...)")
    pr.add_argument("--depuis", type=int, default=2022)
    pr.add_argument("--jusqu", type=int, default=2026,
                    help="derniere saison incluse. La laisser sur l'annee en cours pour "
                         "que le modele voie la saison qui se joue.")
    pr.add_argument("--xi", type=float, default=0.0035, help="decroissance temporelle par jour")
    pr.add_argument("--buteurs", action="store_true",
                    help="ajouter les probabilites de buteurs (FPL si Premier League)")
    pr.add_argument("--k-shrink", dest="k_shrink", type=float, default=600.0,
                    help="retrecissement des taux joueurs, en minutes")
    # rugby
    pr.add_argument("--meteo", action="store_true", help="integrer la meteo du stade (rugby)")
    pr.add_argument("--dans-jours", dest="dans_jours", type=int, default=2)
    pr.add_argument("--part-essais", dest="part_essais", type=float, default=0.58)
    # basket
    pr.add_argument("--demi-vie", dest="demi_vie", type=float, default=240.0)
    # tennis
    pr.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    pr.add_argument("--bo", type=int, default=3, choices=[3, 5])
    pr.add_argument("--tour", default="atp", choices=["atp", "wta"])
    pr.add_argument("--poids-surface", dest="poids_surface", type=float, default=0.5)
    pr.add_argument("--p-match", dest="p_match", type=float, default=None,
                    help="forcer la probabilite de victoire (court-circuite l'Elo)")
    pr.set_defaults(func=cmd_predict)

    jo = sub.add_parser("journee", help="tous les matchs a venir sur une seule page HTML")
    jo.add_argument("--sport", default="football",
                    choices=["football", "nrl", "top14", "prod2", "rugby",
                             "basket", "tennis"],
                    help="nrl = rugby a XIII australien, entierement automatique")
    jo.add_argument("--leagues", default="top5",
                    help="codes separes par des virgules, ou une preselection : "
                         "top5, top15, d2, europe, monde")
    jo.add_argument("--source", default="auto",
                    choices=["auto", "api", "apifootball", "csv"],
                    help="calendrier : api = The Odds API (live, 1 credit/ligue), "
                         "csv = fixtures.csv football-data (gratuit mais souvent fige)")
    jo.add_argument("--cle", default=None, help="cle The Odds API (calendrier seulement)")
    jo.add_argument("--html", default="journee.html")
    jo.add_argument("--titre", default="Journée")
    jo.add_argument("--ajouter", action="store_true",
                    help="empiler sur la journee existante au lieu de la remplacer")
    jo.add_argument("--ouvrir", action="store_true")
    jo.add_argument("--historique", default=None, help="CSV de resultats (hors football)")
    jo.add_argument("--matchs", default=None, help="CSV des rencontres a predire (hors football)")
    jo.add_argument("--joueurs", default=None, help="CSV de statistiques joueurs")
    jo.add_argument("--competition", default=None, help="nom affiche de la competition")
    jo.add_argument("--buteurs", action="store_true", help="ajouter les buteurs (Premier League)")
    jo.add_argument("--absences", action="store_true",
                    help="ajouter les blessures et suspensions (API-Football). "
                         "Plan gratuit : fenetre de +/- 1 jour autour d'aujourd'hui.")
    jo.add_argument("--cle-football", dest="cle_football", default=None,
                    help="cle API-Football (sinon API_FOOTBALL_KEY)")
    jo.add_argument("--depuis", type=int, default=2022)
    jo.add_argument("--jusqu", type=int, default=2026)
    jo.add_argument("--xi", type=float, default=0.0035)
    jo.add_argument("--sans-division-inf", dest="sans_division_inf",
                    action="store_true",
                    help="ne pas inclure la division inferieure dans l'ajustement "
                         "(deconseille : les promus deviennent aberrants)")
    jo.add_argument("--k-shrink", dest="k_shrink", type=float, default=600.0)
    jo.add_argument("--meteo", action="store_true")
    jo.add_argument("--dans-jours", dest="dans_jours", type=int, default=2)
    jo.add_argument("--part-essais", dest="part_essais", type=float, default=0.58)
    jo.add_argument("--part-essais-nrl", dest="part_essais_nrl", type=float, default=0.70,
                    help="part des points venant des essais au XIII (plus haute qu'au XV)")
    jo.add_argument("--depuis-nrl", dest="depuis_nrl", default="2021-01-01")
    jo.add_argument("--demi-vie-rugby", dest="demi_vie_rugby", type=float, default=150.0,
                    help="demi-vie de la ponderation temporelle au rugby. 150 jours : "
                         "choisi sur le BIAIS hors echantillon, pas sur l'erreur absolue "
                         "(cette derniere ne discrimine pas, noyee dans la variance).")
    jo.add_argument("--saison-courante", dest="saison_courante", default="2026-2027",
                    help="saison en cours : resultats recents + calendrier")
    jo.add_argument("--saisons", default="2023-2024,2024-2025,2025-2026",
                    help="saisons Wikipedia a charger (Top 14 / Pro D2)")
    jo.add_argument("--demi-vie", dest="demi_vie", type=float, default=240.0)
    jo.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    jo.add_argument("--bo", type=int, default=3, choices=[3, 5])
    jo.add_argument("--tour", default="atp", choices=["atp", "wta"])
    jo.set_defaults(func=cmd_journee)

    eu = sub.add_parser("europe", help="Ligue des champions / Europa : predictions cross-championnats")
    eu.add_argument("--competition", default="c1", choices=["c1", "c3", "c4"],
                    help="c1 = Ligue des champions, c3 = Europa, c4 = Conference")
    eu.add_argument("--leagues", default="E0,F1,SP1,I1,D1,P1,N1,B1,T1,G1,SC0",
                    help="championnats a inclure dans l'ajustement conjoint")
    eu.add_argument("--html", default="europe.html")
    eu.add_argument("--cle", default=None, help="cle The Odds API (calendrier uniquement)")
    eu.add_argument("--depuis", type=int, default=2022)
    eu.add_argument("--jusqu", type=int, default=2026)
    eu.add_argument("--xi", type=float, default=0.0035)
    eu.add_argument("--ajouter", action="store_true")
    eu.add_argument("--ouvrir", action="store_true")
    eu.set_defaults(func=cmd_europe)

    lv = sub.add_parser("live", help="rafraichit une journee avec les scores en cours")
    lv.add_argument("--html", default="journee.html")
    lv.add_argument("--titre", default="Journée")
    lv.add_argument("--cle", default=None, help="cle The Odds API")
    lv.add_argument("--leagues", default="top5,F2",
                    help="championnats suivis en direct (plafonne le cout). "
                         "Vide = tous ceux de la journee.")
    lv.add_argument("--tout", action="store_true",
                    help="interroger toutes les competitions, pas seulement celles "
                         "avec un match en cours (coute plus de credits)")
    lv.set_defaults(func=cmd_live)

    tm = sub.add_parser("template-matchs", help="modele CSV des rencontres a predire")
    tm.add_argument("fichier", nargs="?", default="matchs.csv")
    tm.set_defaults(func=cmd_template_matchs)

    tr = sub.add_parser("template-resultats", help="modele CSV d'historique de resultats")
    tr.add_argument("fichier", nargs="?", default="resultats.csv")
    tr.add_argument("--joueurs", default=None, help="ecrire aussi un modele de stats joueurs")
    tr.set_defaults(func=cmd_template_resultats)

    sp = sub.add_parser("sports", help="liste les competitions (gratuit, 0 credit)")
    sp.add_argument("--cle", default=None, help="cle The Odds API (sinon ODDS_API_KEY)")
    sp.add_argument("--filtre", default=None, help="filtre sur le nom, ex. 'soccer'")
    sp.set_defaults(func=cmd_sports)

    sc = sub.add_parser("scan", help="scan automatique des cotes ARJEL via The Odds API")
    sc.add_argument("--sport", default="football", choices=["football", "basket", "rugby", "tennis"])
    sc.add_argument("--competitions", default=None,
                    help="cles de competitions separees par des virgules "
                         "(ex. soccer_france_ligue_one). Defaut : toutes celles du sport.")
    sc.add_argument("--marche", default="h2h", help="h2h (vainqueur) | totals | spreads")
    sc.add_argument("--cle", default=None, help="cle The Odds API (sinon ODDS_API_KEY)")
    sc.add_argument("--ttl", type=float, default=900.0, help="duree du cache en secondes")
    sc.add_argument("--journal", default=None,
                    help="fichier CSV ou journaliser les paris pour suivi du CLV")
    sc.add_argument("--oui", action="store_true", help="ne pas demander confirmation du cout")
    sc.add_argument("--p-min", dest="p_min", type=float, default=0.55)
    sc.add_argument("--edge-min", dest="edge_min", type=float, default=0.015)
    sc.add_argument("--w", type=float, default=0.0)
    sc.add_argument("--bankroll", type=float, default=1000.0)
    sc.add_argument("--kelly", type=float, default=0.25)
    sc.add_argument("--max-mise", dest="max_mise", type=float, default=0.02)
    sc.add_argument("--trj-min", dest="trj_min", type=float, default=0.90)
    sc.add_argument("--sans-reference", action="store_true",
                    help="ne pas interroger la region eu (economise des credits, "
                         "mais prive de la reference Pinnacle : nettement moins fiable)")
    sc.add_argument("--top", type=int, default=20)
    sc.add_argument("--details", action="store_true")
    sc.add_argument("--diagnostic", action="store_true",
                    help="afficher toutes les issues analysees et le motif de rejet "
                         "(affiche automatiquement si aucun pari ne passe)")
    sc.set_defaults(func=cmd_scan)

    tj = sub.add_parser("trj", help="analyse TRJ et de-vig d'un marche")
    tj.add_argument("cotes", nargs="+")
    tj.set_defaults(func=cmd_trj)

    b = sub.add_parser("backtest", help="backtest football sur donnees reelles")
    b.add_argument("--leagues", default="E0,D1,SP1,I1,F1")
    b.add_argument("--depuis", type=int, default=2019)
    b.add_argument("--jusqu", type=int, default=2024)
    b.add_argument("--refit", type=int, default=14)
    b.add_argument("--xi", type=float, default=0.0035)
    b.add_argument("--p-min", dest="p_min", type=float, default=0.55)
    b.add_argument("--edge-min", dest="edge_min", type=float, default=0.02)
    b.add_argument("--sweep", action="store_true")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("meteo", help="meteo d'un stade et impact rugby")
    w.add_argument("stade")
    w.add_argument("--dans-jours", dest="dans_jours", type=int, default=2)
    w.set_defaults(func=cmd_weather)

    bk = sub.add_parser("books", help="liste des operateurs ARJEL")
    bk.set_defaults(func=cmd_books)

    return p


def _console_utf8():
    """
    Force la sortie console en UTF-8.

    Sans cela, sous Windows, la console est en cp1252 et tout affichage d'un
    nom d'equipe turc, portugais ou bresilien fait PLANTER la commande
    (UnicodeEncodeError sur "Eyupspor", "Caykur Rizespor", "Goztepe"). Des lors
    qu'on couvre des championnats hors Europe de l'Ouest, le cas est garanti.
    `errors="replace"` evite en plus qu'un caractere exotique isole ne casse un
    scan de plusieurs centaines de matchs.
    """
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _console_utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompu.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
