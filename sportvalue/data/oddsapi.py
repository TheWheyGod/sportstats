"""
Client The Odds API, configure pour le marche francais.

L'API expose une region `fr` qui contient exactement les operateurs agrees ANJ
couverts : winamax_fr, betclic_fr, unibet_fr, pmu_fr, netbet_fr.

STRATEGIE DE QUOTA
------------------
Le palier gratuit donne 500 credits par mois, et un credit est consomme PAR
REGION ET PAR MARCHE. Un appel `regions=eu,fr&markets=h2h,totals` coute donc
4 credits, pas 1. Avec 500 credits :

  - 1 scan quotidien de 2 regions x 2 marches = 4 credits/jour = 120/mois : OK.
  - Un scan toutes les heures epuise le quota en 5 jours.

D'ou le cache disque agressif et le compteur de credits restants lu dans les
en-tetes de reponse. Le client refuse de continuer sous un seuil de securite
plutot que de couper au milieu d'un scan.

On demande `eu,fr` en un seul appel : `eu` apporte Pinnacle (la reference de
verite, non jouable depuis la France) et `fr` les books ou l'on mise vraiment.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from .arjel import ODDSAPI_FR_KEYS, SHARP_REFERENCES
from .cache import get_cache
from .schema import Fixture, MarketQuote

__all__ = ["OddsAPIClient", "SPORT_KEYS", "MARKET_KEYS"]

BASE = "https://api.the-odds-api.com/v4"

# Cles de sport les plus utiles pour les quatre disciplines demandees.
SPORT_KEYS = {
    "football": [
        "soccer_france_ligue_one",
        "soccer_france_ligue_two",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_germany_bundesliga",
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
    ],
    "basket": ["basketball_nba", "basketball_euroleague", "basketball_ncaab"],
    "rugby": ["rugbyleague_nrl", "rugbyunion_six_nations"],
    "tennis": [
        "tennis_atp_aus_open_singles",
        "tennis_atp_french_open",
        "tennis_atp_wimbledon",
        "tennis_atp_us_open",
        "tennis_wta_aus_open_singles",
        "tennis_wta_french_open",
        "tennis_wta_wimbledon",
        "tennis_wta_us_open",
    ],
}

MARKET_KEYS = {
    "1x2": "h2h",          # vainqueur (avec nul au football)
    "ml": "h2h",           # vainqueur sans nul (tennis, basket)
    "ou": "totals",        # total de points/buts
    "ah": "spreads",       # handicap
}


class QuotaExhausted(RuntimeError):
    pass


class OddsAPIClient:
    """
    Parameters
    ----------
    api_key : cle The Odds API. A defaut, lue dans la variable
        d'environnement ODDS_API_KEY.
    ttl : duree de vie du cache en secondes. 900 (15 min) est un bon compromis :
        les lignes bougent, mais pas au point de justifier un appel par minute.
    min_credits : seuil sous lequel le client refuse d'appeler l'API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        ttl: float = 900.0,
        min_credits: int = 10,
    ):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.ttl = ttl
        self.min_credits = min_credits
        self.credits_remaining: int | None = None
        self.credits_used: int | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict) -> list | dict:
        import requests
        from urllib.parse import urlencode

        if not self.configured:
            raise RuntimeError(
                "Aucune cle The Odds API. Definir ODDS_API_KEY, ou utiliser "
                "la saisie manuelle des cotes (voir data/manual_odds.py)."
            )
        if self.credits_remaining is not None and self.credits_remaining < self.min_credits:
            raise QuotaExhausted(
                f"Quota presque epuise ({self.credits_remaining} credits restants). "
                "Le scan est interrompu volontairement."
            )

        full = {**params, "apiKey": self.api_key}
        cache_key = f"{path}?{urlencode(sorted((k, v) for k, v in params.items()))}"

        def loader():
            r = requests.get(f"{BASE}{path}", params=full, timeout=30)
            if r.status_code == 401:
                raise RuntimeError("Cle API refusee (401).")
            if r.status_code == 429:
                raise QuotaExhausted("Quota depasse (429).")
            r.raise_for_status()
            self.credits_remaining = int(r.headers.get("x-requests-remaining", -1))
            self.credits_used = int(r.headers.get("x-requests-used", -1))
            return r.json()

        return get_cache().get_json(cache_key, self.ttl, loader)

    # ------------------------------------------------------------------
    def list_sports(self) -> list[dict]:
        """Gratuit : ne consomme pas de credit."""
        data = self._get("/sports", {"all": "false"})
        return [
            {"key": s["key"], "groupe": s.get("group"), "titre": s.get("title"),
             "actif": s.get("active")}
            for s in data
        ]

    # ------------------------------------------------------------------
    def get_odds(
        self,
        sport_key: str,
        markets: Iterable[str] = ("h2h",),
        regions: Iterable[str] = ("eu", "fr"),
    ) -> list[dict]:
        """
        Renvoie la reponse brute. Cout : len(regions) x len(markets) credits.
        """
        markets = list(markets)
        regions = list(regions)
        return self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )

    # ------------------------------------------------------------------
    def fixtures_with_quotes(
        self,
        sport_key: str,
        sport_label: str,
        market: str = "h2h",
        regions: Iterable[str] = ("eu", "fr"),
    ) -> list[dict]:
        """
        Convertit la reponse en objets du schema interne, en separant
        explicitement les deux roles :

            'reference' : books sharp (Pinnacle...), pour estimer la verite
            'arjel'     : books agrees ANJ, ou l'on peut reellement miser

        Un evenement sans aucune cote ARJEL est ecarte : il n'est pas jouable.
        """
        raw = self.get_odds(sport_key, markets=[market], regions=regions)
        out = []
        for ev in raw:
            try:
                dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue

            fixture = Fixture(
                sport=sport_label,
                competition=ev.get("sport_title", sport_key),
                date=dt.astimezone(timezone.utc).replace(tzinfo=None),
                home=ev.get("home_team", ""),
                away=ev.get("away_team", ""),
                fixture_id=ev.get("id", ""),
            )

            ref_quotes: list[MarketQuote] = []
            fr_quotes: list[MarketQuote] = []
            for bm in ev.get("bookmakers", []):
                for mk in bm.get("markets", []):
                    if mk.get("key") != market:
                        continue
                    outcomes = mk.get("outcomes", [])
                    if len(outcomes) < 2:
                        continue
                    names = [o["name"] for o in outcomes]
                    prices = [float(o["price"]) for o in outcomes]
                    line = outcomes[0].get("point")
                    q = MarketQuote(
                        book=bm["key"],
                        market=market,
                        selections=names,
                        odds=prices,
                        line=float(line) if line is not None else None,
                        timestamp=datetime.now(),
                    )
                    if bm["key"] in ODDSAPI_FR_KEYS:
                        fr_quotes.append(q)
                    elif bm["key"] in SHARP_REFERENCES:
                        ref_quotes.append(q)

            if not fr_quotes:
                continue
            out.append({"fixture": fixture, "reference": ref_quotes, "arjel": fr_quotes})
        return out

    # ------------------------------------------------------------------
    def quota_status(self) -> dict:
        return {
            "credits_restants": self.credits_remaining,
            "credits_utilises": self.credits_used,
            "cle_configuree": self.configured,
        }
