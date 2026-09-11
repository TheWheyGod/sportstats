"""
football-data.org : buts, passes decisives et penaltys par joueur.

Seule source gratuite et propre de statistiques joueurs hors Premier League
(FPL). Le plan gratuit couvre douze competitions -- ici les neuf qu'on suit --
avec un jeton par mail et 10 requetes par minute.

CE QUE LA SOURCE DONNE, ET NE DONNE PAS
---------------------------------------
Le point d'entree `scorers` liste les joueurs AYANT MARQUE dans la saison :
buts, passes, penaltys, matchs joues, poste (Offence/Midfield/Defence).
Consequences :

- pas de minutes : on les approche par matchs joues x 78 (un titulaire
  fait ~85 min, un remplacant ~25 ; 78 est la moyenne des joueurs qui
  marquent). Le retrecissement du modele (k = 600 min) absorbe l'a-peu-pres ;
- un joueur sans but cette saison est absent de la liste courante. On
  complete par la SAISON PASSEE : un attaquant a 21 buts qui n'a pas encore
  marque en septembre reste un candidat evident. Sa presence dans l'effectif
  est incertaine (transfert ?), donc ses minutes attendues sont reduites ;
  s'il reapparait dans la liste courante d'une AUTRE competition, il est
  ecarte -- c'est un transfert detecte ;
- un passeur qui n'a jamais marque n'existe pas pour cette source : les
  probabilites de passe decisive sous-estiment les purs createurs a 0 but.
  Assume, et dit sur la page.

Le "reste de l'equipe" (defenseurs et remplacants jamais listes) recoit une
ligne collective avec le prior de poste, pour que la somme des esperances
individuelles reste egale a celle de l'equipe.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import time
import unicodedata

import pandas as pd

from .cache import get_cache

__all__ = ["FDO_CODES", "load_scorers", "build_squads", "name_key"]

_TTL = 60 * 60 * 20          # une fois par jour suffit : les stats bougent par journee
_MIN_PAR_MATCH = 78.0
_DECAY_PREV = 0.6            # poids d'un match de la saison passee
_PRESENCE_PREV = 0.6         # minutes attendues d'un joueur vu seulement la saison passee
_RESTE_JOUEURS = 4           # joueurs reguliers jamais listes (defenseurs...)

# code football-data.co.uk -> code football-data.org
FDO_CODES = {
    "E0": "PL", "E1": "ELC", "F1": "FL1", "SP1": "PD", "I1": "SA", "D1": "BL1",
    "N1": "DED", "P1": "PPL", "BRA1": "BSA",
}
_SECTION = {"Offence": "attaquant", "Midfield": "milieu", "Defence": "defenseur",
            "Goalkeeper": "gardien"}


def name_key(name: str) -> str:
    """
    Cle de rapprochement de noms de joueurs entre sources :
    'Kylian Mbappé', 'K. Mbappe', 'Mbappé' -> 'mbappe'. Nom de famille seul,
    sans accents : les initiales et prenoms varient trop d'une source a l'autre.
    """
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z ]", " ", s).lower().split()
    return s[-1] if s else ""


def _get(url: str, key: str) -> dict:
    import requests

    def loader():
        for tentative in range(3):
            r = requests.get(url, headers={"X-Auth-Token": key}, timeout=40)
            if r.status_code == 429:
                time.sleep(62)          # 10 requetes/minute sur le plan gratuit
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("football-data.org : limite de debit persistante")

    return get_cache().get_json(url, _TTL, loader)


def load_scorers(fdo_code: str, api_key: str | None = None, season: int | None = None,
                 limit: int = 200) -> tuple[pd.DataFrame, dict]:
    """
    Une saison de buteurs. Renvoie (DataFrame, meta) avec meta = {annee, journee,
    matchs_saison}. `season` = annee de debut ; None = saison en cours.
    """
    key = api_key or os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    if not key:
        raise RuntimeError("FOOTBALL_DATA_ORG_KEY absente.")
    url = f"https://api.football-data.org/v4/competitions/{fdo_code}/scorers?limit={limit}"
    if season:
        url += f"&season={season}"
    j = _get(url, key)
    if "scorers" not in j:
        raise RuntimeError(f"football-data.org {fdo_code} : {j.get('message', j)}")
    s = j.get("season") or {}
    debut = str(s.get("startDate", ""))[:4]
    meta = {"annee": int(debut) if debut.isdigit() else None,
            "journee": int(s.get("currentMatchday") or 0),
            "fin": s.get("endDate")}
    rows = []
    for x in j["scorers"]:
        p, t = x.get("player") or {}, x.get("team") or {}
        rows.append({
            "joueur_id": p.get("id"), "joueur": p.get("name") or "",
            "poste": _SECTION.get(p.get("section") or "", "inconnu"),
            "equipe_fdo": t.get("shortName") or t.get("name") or "",
            "equipe_fdo_long": t.get("name") or "",
            "matchs": int(x.get("playedMatches") or 0),
            "buts": int(x.get("goals") or 0),
            "passes": int(x.get("assists") or 0),
            "penaltys": int(x.get("penalties") or 0),
        })
    return pd.DataFrame(rows), meta


def _matchs_saison(meta: dict, df: pd.DataFrame) -> float:
    """Nombre de matchs d'une saison achevee : le maximum joue par un joueur."""
    return float(df["matchs"].max()) if not df.empty else 34.0


def build_squads(codes: list[str], api_key: str | None = None, verbose: bool = True) -> dict:
    """
    Effectifs au format ScorerModel, par code de championnat :
        joueur, joueur_id, equipe_fdo, poste, minutes, buts_hors_penalty,
        passes, tireur_penalty, minutes_attendues, saison_seule_passee

    Charge la saison en cours et la precedente de chaque competition, et
    utilise l'ensemble des listes courantes pour reperer les transferts.
    """
    key = api_key or os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    if not key:
        raise RuntimeError("FOOTBALL_DATA_ORG_KEY absente.")
    cur, prev, metas = {}, {}, {}
    for code in codes:
        fdo = FDO_CODES.get(code)
        if not fdo:
            continue
        try:
            c, mc = load_scorers(fdo, key)
            p, mp = load_scorers(fdo, key, season=(mc["annee"] or _dt.date.today().year) - 1)
        except Exception as e:
            if verbose:
                print(f"   [!] football-data.org {code} : {e}")
            continue
        cur[code], prev[code], metas[code] = c, p, (mc, mp)
        if verbose:
            print(f"   {code:<5} {fdo:<4} saison {mc['annee']} : {len(c):>3} buteurs (J{mc['journee']}), "
                  f"saison {mp['annee']} : {len(p):>3}")

    # joueur -> competition ou il a marque CETTE saison (detection de transfert)
    ou_joue = {}
    for code, c in cur.items():
        for r in c.itertuples(index=False):
            ou_joue[r.joueur_id] = code

    out = {}
    for code in cur:
        c, p = cur[code], prev[code]
        mc, mp = metas[code]
        journee = max(mc["journee"], 1)
        n_prev = _matchs_saison(mp, p)
        rows = {}
        for r in c.itertuples(index=False):
            rows[r.joueur_id] = {
                "joueur_id": r.joueur_id, "joueur": r.joueur, "poste": r.poste,
                "equipe_fdo": r.equipe_fdo, "equipe_fdo_long": r.equipe_fdo_long,
                "matchs_c": r.matchs, "buts_c": r.buts, "pen_c": r.penaltys, "passes_c": r.passes,
                "matchs_p": 0, "buts_p": 0, "pen_p": 0, "passes_p": 0,
            }
        for r in p.itertuples(index=False):
            if r.joueur_id in rows:
                x = rows[r.joueur_id]
                x.update({"matchs_p": r.matchs, "buts_p": r.buts, "pen_p": r.penaltys,
                          "passes_p": r.passes})
                if x["poste"] == "inconnu":
                    x["poste"] = r.poste
            else:
                if ou_joue.get(r.joueur_id, code) != code:
                    continue                    # a marque ailleurs cette saison : parti
                rows[r.joueur_id] = {
                    "joueur_id": r.joueur_id, "joueur": r.joueur, "poste": r.poste,
                    "equipe_fdo": r.equipe_fdo, "equipe_fdo_long": r.equipe_fdo_long,
                    "matchs_c": 0, "buts_c": 0, "pen_c": 0, "passes_c": 0,
                    "matchs_p": r.matchs, "buts_p": r.buts, "pen_p": r.penaltys,
                    "passes_p": r.passes,
                }
        d = pd.DataFrame(list(rows.values()))
        if d.empty:
            continue
        d["minutes"] = _MIN_PAR_MATCH * (d["matchs_c"] + _DECAY_PREV * d["matchs_p"])
        d["buts_hors_penalty"] = ((d["buts_c"] - d["pen_c"]).clip(lower=0)
                                  + _DECAY_PREV * (d["buts_p"] - d["pen_p"]).clip(lower=0))
        d["passes"] = d["passes_c"] + _DECAY_PREV * d["passes_p"]
        d["saison_seule_passee"] = d["matchs_c"] == 0
        # minutes attendues : part des matchs joues, reduite quand la presence
        # dans l'effectif n'est connue que de la saison passee
        d["minutes_attendues"] = (90.0 * d["matchs_c"] / journee).clip(0, 90)
        seule = d["saison_seule_passee"]
        d.loc[seule, "minutes_attendues"] = (
            90.0 * d.loc[seule, "matchs_p"] / n_prev * _PRESENCE_PREV).clip(0, 90)
        # tireur de penalty : le plus de penaltys marques, saison en cours d'abord
        d["tireur_penalty"] = 0
        for eq, g in d.groupby("equipe_fdo"):
            score = g["pen_c"] * 10 + g["pen_p"]
            if score.max() > 0:
                d.loc[score.idxmax(), "tireur_penalty"] = 1
        # reste de l'equipe : ligne collective au prior de poste
        restes = []
        for eq, g in d.groupby("equipe_fdo"):
            restes.append({
                "joueur_id": -1, "joueur": "Autres joueurs", "poste": "collectif",
                "equipe_fdo": eq, "equipe_fdo_long": g["equipe_fdo_long"].iloc[0],
                "matchs_c": journee, "buts_c": 0, "pen_c": 0, "passes_c": 0,
                "matchs_p": n_prev, "buts_p": 0, "pen_p": 0, "passes_p": 0,
                "minutes": 0.0, "buts_hors_penalty": 0.0, "passes": 0.0,
                "saison_seule_passee": False,
                "minutes_attendues": 90.0 * _RESTE_JOUEURS, "tireur_penalty": 0,
            })
        d = pd.concat([d, pd.DataFrame(restes)], ignore_index=True)
        d["cle_nom"] = d["joueur"].map(name_key)
        out[code] = d
    return out


# football-data.co.uk -> football-data.org, quand le rapprochement automatique echoue
ALIAS_FDO = {
    "Rennes": "Stade Rennais", "Nijmegen": "NEC", "Guimaraes": "Vitória SC",
    "Sp Lisbon": "Sporting CP", "Athletico-PR": "Paranaense", "Atletico-MG": "Mineiro",
    "Vitoria": "Vitória", "Paris SG": "Paris Saint-Germain",
}


def team_squad(squads: pd.DataFrame, equipe_hist: str) -> pd.DataFrame:
    """Effectif d'une equipe nommee a la maniere de football-data.co.uk."""
    from .fixtures import resolve_team

    courts = set(squads["equipe_fdo"])
    longs = dict(zip(squads["equipe_fdo_long"], squads["equipe_fdo"]))
    nom = ALIAS_FDO.get(equipe_hist)
    if nom not in courts:
        nom = resolve_team(equipe_hist, courts)
    if nom is None:
        long = resolve_team(equipe_hist, set(longs))
        nom = longs.get(long) if long else None
    if nom is None:
        return pd.DataFrame()
    return squads[squads["equipe_fdo"] == nom].copy()


def apply_absences(squad: pd.DataFrame, absents: list) -> pd.DataFrame:
    """
    Met a zero (forfait) ou reduit de moitie (incertain) les minutes attendues
    des joueurs signales absents par API-Football. Rapprochement par nom de
    famille : les deux sources n'ecrivent ni les prenoms ni les accents pareil.
    """
    if squad.empty or not absents:
        return squad
    d = squad.copy()
    poids = {}
    for x in absents:
        k = name_key(x.get("joueur", ""))
        if k:
            poids[k] = min(poids.get(k, 1.0), 1.0 - float(x.get("poids", 1.0)))
    if not poids:
        return d
    fac = d["cle_nom"].map(poids)
    d.loc[fac.notna(), "minutes_attendues"] = d.loc[fac.notna(), "minutes_attendues"] * fac[fac.notna()]
    return d
