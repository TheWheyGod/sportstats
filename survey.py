"""
Etat des lieux : ou les books ARJEL s'ecartent-ils de la ligne sharp ?

L'edge moyen sur un marche vaut approximativement MOINS la marge du book,
tant que l'opinion du book colle a la ligne sharp. Ce qui nous interesse est
donc l'ECART A CETTE ATTENTE : un marche ou l'edge maximal remonte nettement
au-dessus de -marge est un marche ou les books francais ont un avis different,
donc un marche ou chercher.

Cout : 2 credits par competition (regions eu + fr).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from sportvalue.data.oddsapi import OddsAPIClient, QuotaExhausted
from sportvalue.data.arjel import ODDSAPI_FR_KEYS, SHARP_REFERENCES
from sportvalue.core.oddsmath import devig, margin_pct, best_odds_across_books, is_sane_market

COMPETITIONS = [
    ("soccer_france_ligue_one", "Ligue 1"),
    ("soccer_france_ligue_two", "Ligue 2"),
    ("soccer_epl", "Premier League"),
    ("soccer_efl_champ", "Championship (D2 ang)"),
    ("soccer_spain_la_liga", "La Liga"),
    ("soccer_italy_serie_a", "Serie A"),
    ("soccer_germany_bundesliga", "Bundesliga"),
    ("soccer_netherlands_eredivisie", "Eredivisie"),
    ("soccer_portugal_primeira_liga", "Primeira Liga"),
    ("soccer_belgium_first_div", "Jupiler Pro League"),
    ("soccer_turkey_super_league", "Super Lig"),
    ("soccer_brazil_campeonato", "Brasileirao"),
]


def analyse(cli, key, label):
    try:
        raw = cli.get_odds(key, markets=["h2h"], regions=["eu", "fr"])
    except QuotaExhausted:
        raise
    except Exception as e:
        return {"competition": label, "erreur": f"{type(e).__name__}"}

    edges, best_per_match, n_fr_list, marges_fr = [], [], [], []
    for ev in raw:
        ref, arjel = {}, {}
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                names = [x["name"] for x in mk["outcomes"]]
                odds = [x["price"] for x in mk["outcomes"]]
                if len(odds) < 2:
                    continue
                if bm["key"] in ODDSAPI_FR_KEYS:
                    if is_sane_market(odds, 1.00, 1.25)[0]:
                        arjel[bm["key"]] = (names, odds)
                elif bm["key"] in SHARP_REFERENCES:
                    if is_sane_market(odds, 0.98, 1.12)[0]:
                        ref[bm["key"]] = (names, odds)
        if not ref or not arjel:
            continue
        rk = min(ref, key=lambda k: margin_pct(ref[k][1]))
        rnames, rodds = ref[rk]
        p_ref = devig(rodds, "shin")

        aligned = {}
        for b, (nm, od) in arjel.items():
            try:
                aligned[b] = [od[nm.index(n)] for n in rnames]
            except ValueError:
                pass
        if not aligned:
            continue
        marges_fr.extend(margin_pct(v) for v in aligned.values())
        best, _ = best_odds_across_books(aligned)
        e = p_ref * best - 1.0
        edges.extend(e.tolist())
        best_per_match.append(float(e.max()))
        n_fr_list.append(len(aligned))

    if not edges:
        return {"competition": label, "erreur": "aucun match exploitable"}
    a = np.array(edges)
    marge_moy = float(np.mean(marges_fr))
    return {
        "competition": label,
        "matchs": len(best_per_match),
        "books_fr": round(float(np.mean(n_fr_list)), 1),
        "marge_fr_%": round(100 * marge_moy, 2),
        "edge_moy_%": round(100 * a.mean(), 2),
        "edge_max_%": round(100 * a.max(), 2),
        # Ce que l'edge moyen vaudrait si les books collaient parfaitement au
        # sharp : -marge. L'ecart positif mesure leur divergence d'opinion.
        "divergence_pts": round(100 * (a.mean() + marge_moy), 2),
        "n_edge_pos": int((a >= 0).sum()),
        "n_total": len(a),
    }


def main():
    cli = OddsAPIClient()
    if not cli.configured:
        print("ODDS_API_KEY non definie.")
        return 1
    rows = []
    for key, label in COMPETITIONS:
        try:
            r = analyse(cli, key, label)
        except QuotaExhausted as e:
            print(f"[!] {e}")
            break
        rows.append(r)
        if "erreur" in r:
            print(f"   {label:<24} {r['erreur']}")
        else:
            print(f"   {label:<24} {r['matchs']:>3} matchs, {r['books_fr']} books FR, "
                  f"edge moy {r['edge_moy_%']:+.2f}%  max {r['edge_max_%']:+.2f}%  "
                  f"divergence {r['divergence_pts']:+.2f} pts")

    ok = [r for r in rows if "erreur" not in r]
    if not ok:
        return 1
    ok.sort(key=lambda r: -r["divergence_pts"])
    print("\n" + "=" * 108)
    print("CLASSEMENT PAR DIVERGENCE  (ecart entre l'edge moyen constate et -marge)")
    print("=" * 108)
    print("%-24s %7s %8s %10s %11s %11s %12s %10s"
          % ("competition", "matchs", "booksFR", "marge_FR", "edge_moy", "edge_max",
             "divergence", "edges>=0"))
    print("-" * 108)
    for r in ok:
        print("%-24s %7d %8.1f %9.2f%% %10.2f%% %10.2f%% %11.2f pts %5d/%d"
              % (r["competition"], r["matchs"], r["books_fr"], r["marge_fr_%"],
                 r["edge_moy_%"], r["edge_max_%"], r["divergence_pts"],
                 r["n_edge_pos"], r["n_total"]))
    tot_pos = sum(r["n_edge_pos"] for r in ok)
    tot = sum(r["n_total"] for r in ok)
    print(f"\n   TOTAL : {tot_pos} issue(s) a edge positif sur {tot} analysees "
          f"({100*tot_pos/tot:.1f}%)")
    print(f"   Quota : {cli.quota_status()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
