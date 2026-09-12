"""
Sportradar Rugby Union (v3) : essais et minutes par joueur, match par match.

La seule source qui donne, pour le Top 14 et la Pro D2, QUI a marque et QUI
a joue combien de temps : la chronologie d'un match (`timeline`) liste chaque
essai avec son auteur et chaque remplacement avec la minute ; la feuille de
match (`lineups`) donne les vingt-trois joueurs, titulaires et remplacants,
avec leur numero de maillot -- donc leur poste.

CONTRAINTES, ET CE QU'ON EN FAIT
--------------------------------
- Cle d'ESSAI : un mois, environ mille requetes, une par seconde. Deux
  requetes par match (chronologie + feuille). Une saison de Top 14 et de
  Pro D2 = 428 matchs = 856 requetes : la collecte est etalee, budget par
  passage (60 par defaut), saison en cours d'abord, puis la saison passee
  en remontant le temps.
- Ce qui est collecte est CONSERVE dans un CSV versionne (data/), une ligne
  par joueur et par match. Quand la cle expirera, tout ce qui a ete
  recolte restera exploitable ; seule la mise a jour s'arretera.
- L'API coupe parfois la connexion sans raison : chaque appel est retente.

Noms d'equipes : ceux de Sportradar ("Asm Clermont Auvergne", "RC
Toulonnais"...) sont rapproches des libelles Wikipedia par rugby_fr.ALIAS.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cache import get_cache

__all__ = ["COMPETITIONS", "cle", "seasons", "collect", "season_squads", "STORE"]

_BASE = "https://api.sportradar.com/rugby-union/trial/v3/en"
COMPETITIONS = {"Top 14": "sr:competition:420", "Pro D2": "sr:competition:1147"}
STORE = Path(__file__).resolve().parents[2] / "data" / "rugby_sportradar.csv"
_COLS = ["competition", "saison", "sport_event_id", "date", "equipe", "joueur_id", "joueur",
         "maillot", "titulaire", "minutes", "essais"]
_TTL_LONG = 10 * 365 * 24 * 3600      # un match termine ne change plus
_TTL_SAISON = 6 * 3600

# Numero de maillot -> poste (priors du modele de marqueurs). Le banc
# (16-23) suit l'usage courant : trois premiere ligne, un deuxieme, un
# troisieme, un demi de melee, un ouvreur/centre, un trois-quarts.
POSTE_MAILLOT = {
    1: "premiere_ligne", 2: "premiere_ligne", 3: "premiere_ligne",
    4: "deuxieme_ligne", 5: "deuxieme_ligne",
    6: "troisieme_ligne", 7: "troisieme_ligne", 8: "troisieme_ligne",
    9: "demi_de_melee", 10: "ouverture", 11: "ailier", 12: "centre", 13: "centre",
    14: "ailier", 15: "arriere",
    16: "premiere_ligne", 17: "premiere_ligne", 18: "premiere_ligne",
    19: "deuxieme_ligne", 20: "troisieme_ligne", 21: "demi_de_melee",
    22: "ouverture", 23: "ailier",
}

_derniere_requete = [0.0]


def cle(api_key: str | None = None) -> str:
    return api_key or os.environ.get("SPORTRADAR_KEY", "")


class QuotaEpuise(RuntimeError):
    """Cle expiree ou quota atteint : inutile d'insister pendant ce passage."""


def _get(path: str, ttl: float, api_key: str | None = None) -> dict:
    import json

    import requests

    key = cle(api_key)
    if not key:
        raise RuntimeError("SPORTRADAR_KEY absente.")
    url = f"{_BASE}{path}"

    def loader():
        derniere_erreur = None
        for tentative in range(4):
            # une requete par seconde, contrat de la cle d'essai
            attente = 1.1 - (time.time() - _derniere_requete[0])
            if attente > 0:
                time.sleep(attente)
            _derniere_requete[0] = time.time()
            try:
                r = requests.get(url, headers={"x-api-key": key, "accept": "application/json"},
                                 timeout=60)
            except requests.RequestException as e:      # connexion coupee : on retente
                derniere_erreur = e
                time.sleep(2 + 2 * tentative)
                continue
            if r.status_code == 429:               # cadence depassee : on souffle
                derniere_erreur = RuntimeError("429")
                time.sleep(3 + 3 * tentative)
                continue
            if r.status_code in (401, 403):
                raise QuotaEpuise(f"Sportradar {r.status_code} : cle expiree ou quota atteint")
            if r.status_code == 404:
                raise FileNotFoundError(f"Sportradar 404 : {path}")
            r.raise_for_status()
            return json.dumps(r.json())
        raise RuntimeError(f"Sportradar injoignable ({type(derniere_erreur).__name__})")

    return json.loads(get_cache().get_text(url, ttl, loader))


def seasons(competition: str, api_key: str | None = None) -> list[dict]:
    """Saisons d'une competition : [{id, name, start, end}], la plus recente en dernier."""
    j = _get(f"/competitions/{COMPETITIONS[competition]}/seasons.json", _TTL_SAISON, api_key)
    return [{"id": s["id"], "name": s.get("name", ""), "start": s.get("start_date", ""),
             "end": s.get("end_date", "")} for s in j.get("seasons", [])]


def season_events(season_id: str, api_key: str | None = None) -> list[dict]:
    """Tous les matchs d'une saison avec leur statut (une requete)."""
    j = _get(f"/seasons/{season_id}/summaries.json?start=0&limit=200", _TTL_SAISON, api_key)
    out = []
    for x in j.get("summaries", []):
        se, st = x.get("sport_event") or {}, x.get("sport_event_status") or {}
        comps = se.get("competitors") or []
        out.append({
            "id": se.get("id"), "start": se.get("start_time", ""),
            "status": st.get("status", ""),
            "home": next((c["name"] for c in comps if c.get("qualifier") == "home"), ""),
            "away": next((c["name"] for c in comps if c.get("qualifier") == "away"), ""),
        })
    return out


def _lignes_match(competition: str, saison: str, ev: dict, api_key: str | None) -> list[dict]:
    """Chronologie + feuille de match -> une ligne par joueur ayant joue."""
    tl = _get(f"/sport_events/{ev['id']}/timeline.json", _TTL_LONG, api_key).get("timeline") or []
    lu = _get(f"/sport_events/{ev['id']}/lineups.json", _TTL_LONG, api_key)
    comps = ((lu.get("lineups") or {}).get("competitors")) or []

    entree, sortie, essais = {}, {}, {}
    for e in tl:
        t = e.get("match_time")
        typ = e.get("type")
        if typ == "substitution" and t is not None:
            for p in e.get("players") or []:
                if p.get("type") == "substituted_in":
                    entree[p.get("id")] = float(t)
                elif p.get("type") == "substituted_out":
                    sortie[p.get("id")] = float(t)
        elif typ == "score_change" and e.get("method") == "try":
            for p in e.get("players") or []:
                if p.get("type") == "scorer":
                    essais[p.get("id")] = essais.get(p.get("id"), 0) + 1
        elif typ == "red_card" and t is not None:
            for p in e.get("players") or []:
                sortie.setdefault(p.get("id"), float(t))

    rows = []
    date = str(ev.get("start", ""))[:10]
    for c in comps:
        for p in c.get("players") or []:
            pid = p.get("id")
            starter = bool(p.get("starter"))
            if starter:
                debut = 0.0
            elif pid in entree:
                debut = entree[pid]
            else:
                continue                       # remplacant reste sur le banc
            fin = sortie.get(pid, 80.0)
            rows.append({
                "competition": competition, "saison": saison, "sport_event_id": ev["id"],
                "date": date, "equipe": c.get("name", ""), "joueur_id": pid,
                "joueur": p.get("name", ""), "maillot": p.get("jersey_number"),
                "titulaire": int(starter), "minutes": max(fin - debut, 0.0),
                "essais": essais.get(pid, 0),
            })
    return rows


def _charger_store() -> pd.DataFrame:
    if STORE.exists():
        return pd.read_csv(STORE, encoding="utf-8")
    return pd.DataFrame(columns=_COLS)


def collect(competition: str, budget: int = 60, api_key: str | None = None,
            verbose: bool = True) -> int:
    """
    Recolte les matchs termines non encore stockes, saison en cours d'abord
    puis saison passee, dans la limite de `budget` requetes. Retourne le
    nombre de requetes consommees.
    """
    store = _charger_store()
    deja = set(store["sport_event_id"]) if not store.empty else set()
    utilise, nouvelles = 0, []
    try:
        ss = seasons(competition, api_key)
        utilise += 1
        # saison en cours (derniere) puis precedente
        for s in list(reversed(ss))[:2]:
            evs = season_events(s["id"], api_key)
            utilise += 1
            clos = [e for e in evs if e["status"] == "closed" and e["id"] not in deja]
            clos.sort(key=lambda e: e["start"], reverse=True)
            for ev in clos:
                if utilise + 2 > budget:
                    break
                try:
                    nouvelles.extend(_lignes_match(competition, s["name"], ev, api_key))
                    deja.add(ev["id"])
                except FileNotFoundError:
                    pass                       # pas de chronologie pour ce match
                utilise += 2
            if utilise + 2 > budget:
                break
    except QuotaEpuise as e:
        if verbose:
            print(f"   [!] {e}")
    except Exception as e:
        if verbose:
            print(f"   [!] Sportradar {competition} : {type(e).__name__} : {e}")
    if nouvelles:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([store, pd.DataFrame(nouvelles)], ignore_index=True)[_COLS].to_csv(
            STORE, index=False, encoding="utf-8")
    if verbose:
        n_matchs = len({r["sport_event_id"] for r in nouvelles})
        print(f"   Sportradar {competition} : {n_matchs} match(s) ajoute(s), "
              f"{utilise} requete(s), {len(deja)} match(s) en stock")
    return utilise


def season_squads(competition: str, saison_courante: str, decay_prev: float = 0.6,
                  presence_prev: float = 0.8) -> pd.DataFrame:
    """
    Effectifs au format ScorerModel a partir du stock : une ligne par joueur,
    saison en cours + saison passee (ponderee), poste deduit du maillot le
    plus frequent, minutes attendues = minutes par match de la saison en
    cours (ou de la saison passee, reduites, si pas encore vu cette saison).
    """
    store = _charger_store()
    if store.empty:
        return pd.DataFrame()
    d = store[store["competition"] == competition].copy()
    if d.empty:
        return pd.DataFrame()
    debut = int(str(saison_courante)[:4])
    lib_cur = f"{str(debut)[2:]}/{str(debut + 1)[2:]}"          # "26/27"
    lib_prev = f"{str(debut - 1)[2:]}/{str(debut)[2:]}"
    d["poids"] = d["saison"].map(lambda s: 1.0 if lib_cur in str(s) else
                                 (decay_prev if lib_prev in str(s) else 0.0))
    d = d[d["poids"] > 0]
    if d.empty:
        return pd.DataFrame()
    d["courante"] = d["saison"].map(lambda s: lib_cur in str(s))
    d["maillot"] = pd.to_numeric(d["maillot"], errors="coerce")

    rows = []
    for pid, g in d.groupby("joueur_id"):
        g = g.sort_values("date")
        cur = g[g["courante"]]
        ref = cur if not cur.empty else g
        maillots = g.loc[g["titulaire"] == 1, "maillot"].dropna()
        if maillots.empty:
            maillots = g["maillot"].dropna()
        poste = POSTE_MAILLOT.get(int(maillots.mode().iloc[0]), "inconnu") if not maillots.empty else "inconnu"
        n_ref = max(len(ref), 1)
        min_att = float(ref["minutes"].sum()) / n_ref
        if cur.empty:
            min_att *= presence_prev
        rows.append({
            "joueur_id": pid, "joueur": g["joueur"].iloc[-1],
            "club": ref["equipe"].iloc[-1], "poste": poste,
            "minutes": float((g["minutes"] * g["poids"]).sum()),
            "buts_hors_penalty": float((g["essais"] * g["poids"]).sum()),
            "matchs": int(len(g)), "minutes_attendues": min(min_att, 80.0),
            "saison_seule_passee": bool(cur.empty), "tireur_penalty": 0,
        })
    out = pd.DataFrame(rows)
    # "Nom, Prenom" -> "Prenom Nom"
    out["joueur"] = out["joueur"].map(
        lambda n: " ".join(reversed([x.strip() for x in str(n).split(",", 1)])) if "," in str(n) else n)
    return out
