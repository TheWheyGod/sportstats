"""
Scores en direct et resultats recents.

POURQUOI CE N'EST PAS "DU LIVE DANS LA PAGE"
---------------------------------------------
Une page publiee comme Artifact tourne sous une politique de securite qui
interdit tout appel reseau sortant : elle ne peut donc pas aller chercher un
score elle-meme, quelle que soit l'API. Et les connecteurs claude.ai de
l'utilisateur ne couvrent aucune donnee sportive.

Le direct se fait donc en AMONT : on recupere les scores ici, on les fusionne
dans la journee, on regenere la page. Repete toutes les N minutes par une tache
planifiee, cela donne une page qui suit les matchs -- la fraicheur est celle de
la derniere regeneration, affichee explicitement pour ne pas faire croire a du
temps reel.

Cout : 1 credit par competition et par appel. A 15 minutes d'intervalle sur une
journee de 5 championnats, cela represente 20 credits par heure : le palier
gratuit (500/mois) n'y suffit pas sur la duree. Reserver le rafraichissement
rapide aux plages ou des matchs se jouent reellement.
"""
from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["fetch_scores", "fetch_live_apifootball", "SPORT_KEYS_SCORES"]

# Statuts API-Football -> statut interne. Tout ce qui n'est pas la est ignore
# (reporte, annule, a venir).
_EN_COURS = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"}
_TERMINE = {"FT", "AET", "PEN"}


def fetch_live_apifootball(api_key: str | None = None, verbose: bool = True) -> list:
    """
    Tous les matchs de football en cours dans le monde, PLUS ceux termines
    aujourd'hui : deux requetes API-Football, zero credit The Odds API.

    Chaque element : {league_id, home, away, home_score, away_score, statut,
    minute, ht_home, ht_away, maj}. `minute` est le chrono officiel du match
    (fixture.status.elapsed), pas une deduction depuis le coup d'envoi.
    """
    import os

    import requests

    from .cache import get_cache

    key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY absente : live indisponible.")

    def charge(params: dict, ttl: float):
        url = "https://v3.football.api-sports.io/fixtures?" + "&".join(f"{k}={v}" for k, v in params.items())

        def loader():
            r = requests.get("https://v3.football.api-sports.io/fixtures",
                             headers={"x-apisports-key": key}, params=params, timeout=40)
            r.raise_for_status()
            return __import__("json").dumps(r.json())

        return __import__("json").loads(get_cache().get_text(url, ttl, loader))

    maj = datetime.now(timezone.utc)
    out = []
    vus = set()
    # 1) en cours : 4 minutes de cache, la tache tourne toutes les heures
    # 2) termines aujourd'hui (date UTC) : 20 minutes, pour les verdicts
    for params, ttl in ((dict(live="all"), 240), (dict(date=maj.strftime("%Y-%m-%d")), 1200)):
        try:
            j = charge(params, ttl)
        except Exception as e:
            if verbose:
                print(f"   [!] API-Football {params} : {type(e).__name__}")
            continue
        if j.get("errors"):
            if verbose:
                print(f"   [!] API-Football {params} : {j['errors']}")
            continue
        for x in j.get("response", []):
            f = x.get("fixture") or {}
            st = ((f.get("status") or {}).get("short")) or ""
            if st in _EN_COURS:
                statut = "en_cours"
            elif st in _TERMINE:
                statut = "termine"
            else:
                continue
            fid = f.get("id")
            if fid in vus:
                continue
            vus.add(fid)
            g = x.get("goals") or {}
            ht = (x.get("score") or {}).get("halftime") or {}
            out.append({
                "fixture_id": fid,
                "league_id": (x.get("league") or {}).get("id"),
                "home": ((x.get("teams") or {}).get("home") or {}).get("name", ""),
                "away": ((x.get("teams") or {}).get("away") or {}).get("name", ""),
                "home_score": g.get("home"), "away_score": g.get("away"),
                "statut": statut,
                "minute": (f.get("status") or {}).get("elapsed"),
                "ht_home": ht.get("home"), "ht_away": ht.get("away"),
                "maj": maj,
            })
    if verbose:
        n_live = sum(1 for x in out if x["statut"] == "en_cours")
        print(f"   API-Football : {n_live} match(s) en cours, {len(out) - n_live} termine(s) aujourd'hui")
    return out

# Competitions dont on sait recuperer les scores, par libelle interne.
SPORT_KEYS_SCORES = {
    "Angleterre - Premier League": "soccer_epl",
    "France - Ligue 1": "soccer_france_ligue_one",
    "Espagne - La Liga": "soccer_spain_la_liga",
    "Italie - Serie A": "soccer_italy_serie_a",
    "Allemagne - Bundesliga": "soccer_germany_bundesliga",
    "France - Ligue 2": "soccer_france_ligue_two",
    "Angleterre - Championship": "soccer_efl_champ",
    "Espagne - Segunda": "soccer_spain_segunda_division",
    "Italie - Serie B": "soccer_italy_serie_b",
    "Allemagne - Bundesliga 2": "soccer_germany_bundesliga2",
    "Portugal - Primeira Liga": "soccer_portugal_primeira_liga",
    "Pays-Bas - Eredivisie": "soccer_netherlands_eredivisie",
    "NRL": "rugbyleague_nrl",
}


def fetch_scores(sport_keys, api_key: str | None = None, days_from: int = 1,
                 verbose: bool = True) -> dict:
    """
    Retourne {(home, away): {statut, home_score, away_score, maj}}.

    `statut` vaut 'a_venir', 'en_cours' ou 'termine'. Le champ `completed` de
    l'API distingue les deux derniers : un evenement avec des scores mais non
    complete est un match en cours.
    """
    import os

    import requests

    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY absente : scores indisponibles.")

    out, credits = {}, None
    RESERVE = 25   # credits a garder pour le cycle quotidien jusqu'a la fin du mois
    for sk in dict.fromkeys(sport_keys):
        # Garde-fou : des que le quota passe sous la reserve, on cesse
        # d'interroger. Sans lui, une tache planifiee en live viderait le
        # quota mensuel en un week-end, et le cycle quotidien (calendrier NRL,
        # 1 credit) ne pourrait plus tourner jusqu'au mois suivant.
        if credits is not None and int(credits) < RESERVE:
            if verbose:
                print(f"   [!] quota sous la reserve ({credits} < {RESERVE}) : live suspendu")
            break
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sk}/scores",
                params={"apiKey": key, "daysFrom": days_from, "dateFormat": "iso"},
                timeout=30,
            )
            if r.status_code != 200:
                if verbose:
                    print(f"   [!] {sk} : HTTP {r.status_code}")
                continue
            credits = r.headers.get("x-requests-remaining", credits)
        except Exception as e:
            if verbose:
                print(f"   [!] {sk} : {type(e).__name__}")
            continue

        for ev in r.json():
            sc = ev.get("scores")
            if ev.get("completed"):
                statut = "termine"
            elif sc:
                statut = "en_cours"
            else:
                statut = "a_venir"
            hs = as_ = None
            if sc:
                par_nom = {x.get("name"): x.get("score") for x in sc}
                hs = par_nom.get(ev.get("home_team"))
                as_ = par_nom.get(ev.get("away_team"))
            out[(ev.get("home_team", ""), ev.get("away_team", ""))] = {
                "statut": statut,
                "home_score": _num(hs),
                "away_score": _num(as_),
                "maj": ev.get("last_update"),
            }

    if verbose:
        n_live = sum(1 for v in out.values() if v["statut"] == "en_cours")
        n_fin = sum(1 for v in out.values() if v["statut"] == "termine")
        print(f"   {len(out)} evenement(s) : {n_live} en cours, {n_fin} termine(s)"
              + (f" | {credits} credits restants" if credits else ""))
    return out


def _num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def attach_to_slate(slate, scores: dict, resolve=None) -> int:
    """
    Fusionne les scores dans une journee deja calculee.

    L'appariement se fait sur les noms de l'API, qui ne sont PAS ceux de
    l'historique ("Nottingham Forest" contre "Nott'm Forest"). On passe donc la
    meme fonction de resolution que pour le calendrier, sans quoi aucun score
    ne se rattache -- silencieusement.
    """
    index = {}
    for (h, a), v in scores.items():
        index[(h.lower(), a.lower())] = v
        if resolve is not None:
            rh, ra = resolve(h), resolve(a)
            if rh and ra:
                index[(rh.lower(), ra.lower())] = v

    n = 0
    for m in slate:
        cle = (str(m["home"]).lower(), str(m["away"]).lower())
        v = index.get(cle)
        if v is None:
            continue
        m["live"] = v
        n += 1
    return n
