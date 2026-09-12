"""
Highlightly (highlightly.net) : calendrier et scores du rugby a XV.

Ce que la source apporte, et ce qu'elle n'apporte pas -- verifie sur le
plan BASIC (100 requetes par jour) :

- RUGBY : matchs avec heure de coup d'envoi exacte, etat (a venir, en
  cours, termine) et score, pour le Top 14, la Pro D2, la Champions Cup, le
  Premiership, l'URC, le Super Rugby. AUCUNE donnee joueur : ni marqueurs,
  ni compositions, ni evenements. Pas de NRL (rugby a XIII).
- FOOTBALL : evenements du match (buts, passes, cartons, remplacements, a la
  minute), statistiques d'equipe dont les buts attendus (xG), arbitre. Riche,
  mais une requete par match : hors de portee du quota pour tout collecter.

Ici : le calendrier precis du Top 14 et de la Pro D2 (Wikipedia ne donne
qu'une plage de deux jours par journee) et leurs scores en direct.

Entetes : la doc impose x-rapidapi-key / x-rapidapi-host, meme en acces
direct hors RapidAPI.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from .cache import get_cache

__all__ = ["RUGBY_LEAGUES", "rugby_fixtures", "rugby_scores", "cle"]

_HOST = "sports.highlightly.net"
_BASE = f"https://{_HOST}"
RUGBY_LEAGUES = {"Top 14": 14400, "Pro D2": 15251}
# libelles d'etat vus dans l'API ; tout autre etat avec un score = en cours
_A_VENIR = {"Not started", "Postponed", "Cancelled", "Canceled", "Abandoned"}
_TERMINE = {"Finished", "After Extra Time", "After Penalties"}


def cle(api_key: str | None = None) -> str:
    return api_key or os.environ.get("HIGHLIGHTLY_KEY", "")


def _get(path: str, params: dict, ttl: float, api_key: str | None = None) -> dict:
    import json

    import requests

    key = cle(api_key)
    if not key:
        raise RuntimeError("HIGHLIGHTLY_KEY absente.")
    url = f"{_BASE}{path}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))

    def loader():
        r = requests.get(f"{_BASE}{path}", params=params, timeout=40,
                         headers={"x-rapidapi-key": key, "x-rapidapi-host": _HOST})
        r.raise_for_status()
        return json.dumps(r.json())

    return json.loads(get_cache().get_text(url, ttl, loader))


def _lignes(j: dict) -> list:
    out = []
    for m in j.get("data", []):
        try:
            dt = datetime.fromisoformat(str(m["date"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        st = m.get("state") or {}
        desc = str(st.get("description") or "")
        score = st.get("score")
        if isinstance(score, dict):
            score = score.get("current")
        hs = as_ = None
        if isinstance(score, str) and "-" in score:
            try:
                hs, as_ = (int(x.strip()) for x in score.split("-", 1))
            except ValueError:
                pass
        if desc in _TERMINE:
            statut = "termine"
        elif desc in _A_VENIR or hs is None:
            statut = "a_venir"
        else:
            statut = "en_cours"
        out.append({
            "id": m.get("id"), "date": dt.replace(tzinfo=None), "journee": m.get("week"),
            "home": (m.get("homeTeam") or {}).get("name", ""),
            "away": (m.get("awayTeam") or {}).get("name", ""),
            "statut": statut, "etat": desc, "home_score": hs, "away_score": as_,
        })
    return out


def rugby_fixtures(competition: str, season: int, api_key: str | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Tous les matchs d'une saison (dates UTC naives, statut, score).
    Deux requetes par saison (pages de 100), en cache six heures : le
    calendrier bouge peu, et le quota est de cent requetes par jour.
    """
    lid = RUGBY_LEAGUES[competition]
    rows, offset = [], 0
    while True:
        j = _get("/rugby/matches", {"leagueId": lid, "season": season, "limit": 100,
                                    "offset": offset}, 6 * 3600, api_key)
        rows.extend(_lignes(j))
        pg = j.get("pagination") or {}
        offset += int(pg.get("limit") or 100)
        if offset >= int(pg.get("totalCount") or 0) or not j.get("data"):
            break
    df = pd.DataFrame(rows)
    if verbose:
        n_av = int((df["statut"] == "a_venir").sum()) if not df.empty else 0
        print(f"   Highlightly {competition} {season} : {len(df)} matchs, {n_av} a venir")
    return df


def rugby_scores(competition: str, jour: str, api_key: str | None = None) -> list:
    """Matchs d'une competition a une date (AAAA-MM-JJ), 3 minutes de cache."""
    j = _get("/rugby/matches", {"leagueId": RUGBY_LEAGUES[competition], "date": jour,
                                "limit": 50}, 180, api_key)
    return _lignes(j)
