"""
Meteo via Open-Meteo : gratuit, sans cle d'API, sans compte.

Deux points d'attention pratiques :

  - il faut la meteo A L'HEURE DU COUP D'ENVOI, pas la moyenne du jour. Un match
    a 21h en novembre et un match a 15h le meme jour n'ont ni la meme
    temperature ni le meme vent ;
  - un stade COUVERT annule l'effet meteo. Les toits sont signales dans la table
    de stades ci-dessous, sinon le modele appliquerait du vent a un match joue a
    l'abri, ce qui produirait un faux value bet sur le total.

L'API archive sert au backtest (meteo reelle passee), l'API forecast aux matchs
a venir. Les deux sont interrogees avec le meme code.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .cache import get_cache

__all__ = ["get_weather", "geocode", "STADIUMS", "venue_weather"]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

_TTL_FORECAST = 60 * 60 * 3       # 3 h
_TTL_ARCHIVE = 60 * 60 * 24 * 90  # 90 jours (le passe ne change pas)

# Stades principaux. `roof=True` -> meteo neutralisee.
STADIUMS = {
    # France - Top 14
    "Toulouse": (43.6206, 1.4335, False),
    "Stade Francais": (48.8414, 2.2530, False),
    "Racing 92": (48.9276, 2.2299, True),      # Paris La Defense Arena : couvert
    "La Rochelle": (46.1503, -1.1420, False),
    "Bordeaux": (44.8280, -0.5570, False),
    "Clermont": (45.7899, 3.1130, False),
    "Toulon": (43.1290, 5.9330, False),
    "Lyon": (45.7247, 4.8320, False),
    "Montpellier": (43.6220, 3.8120, False),
    "Castres": (43.5990, 2.2400, False),
    "Bayonne": (43.4900, -1.4780, False),
    "Perpignan": (42.7000, 2.8890, False),
    "Pau": (43.3160, -0.3660, False),
    "Vannes": (47.6480, -2.7600, False),
    # Angleterre / Irlande / Galles / Ecosse
    "Twickenham": (51.4560, -0.3417, False),
    "Leicester": (52.6208, -1.1410, False),
    "Bath": (51.3810, -2.3560, False),
    "Saracens": (51.6030, -0.2380, False),
    "Harlequins": (51.4720, -0.2470, False),
    "Northampton": (52.2360, -0.8850, False),
    "Sale": (53.3480, -2.4670, False),
    "Exeter": (50.7420, -3.4680, False),
    "Gloucester": (51.8600, -2.2360, False),
    "Bristol": (51.4400, -2.6200, False),
    "Aviva Stadium": (53.3350, -6.2280, False),
    "Thomond Park": (52.6740, -8.6420, False),
    "RDS Arena": (53.3250, -6.2290, False),
    "Principality Stadium": (51.4780, -3.1826, True),   # toit retractable
    "Murrayfield": (55.9422, -3.2410, False),
    "Stadio Olimpico": (41.9339, 12.4547, False),
    # Hemisphere sud
    "Eden Park": (-36.8748, 174.7450, False),
    "Ellis Park": (-26.1975, 28.0608, False),
    "Newlands": (-33.9710, 18.4680, False),
    "Suncorp Stadium": (-27.4648, 153.0095, False),
    "Estadio Jose Amalfitani": (-34.6350, -58.5200, False),
}


def _http_json(url: str, params: dict, ttl: float) -> dict:
    import requests
    from urllib.parse import urlencode

    key = f"{url}?{urlencode(sorted(params.items()))}"

    def loader():
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        return r.json()

    return get_cache().get_json(key, ttl, loader)


def geocode(name: str) -> tuple[float, float] | None:
    """Resout un nom de ville/stade en coordonnees. None si introuvable."""
    try:
        data = _http_json(
            GEOCODE_URL, {"name": name, "count": 1, "language": "fr", "format": "json"},
            _TTL_ARCHIVE,
        )
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return float(results[0]["latitude"]), float(results[0]["longitude"])


def get_weather(lat: float, lon: float, when: datetime) -> dict:
    """
    Meteo horaire au plus pres de `when`.

    Retourne {'temp_c','precip_mm','wind_kmh','wind_gust_kmh','source'}.
    En cas d'echec reseau, retourne un dict avec source='indisponible' et des
    valeurs None : les modeles traitent alors le match comme meteo neutre
    plutot que de planter le scan entier.
    """
    when = when.replace(minute=0, second=0, microsecond=0)
    day = when.strftime("%Y-%m-%d")
    now = datetime.now()
    is_past = when < now - timedelta(days=1)

    url = ARCHIVE_URL if is_past else FORECAST_URL
    ttl = _TTL_ARCHIVE if is_past else _TTL_FORECAST
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        "start_date": day,
        "end_date": day,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    try:
        data = _http_json(url, params, ttl)
        hourly = data["hourly"]
        times = [datetime.fromisoformat(t) for t in hourly["time"]]
        idx = int(np.argmin([abs((t - when).total_seconds()) for t in times]))

        def pick(key):
            v = hourly.get(key, [None])[idx]
            return float(v) if v is not None else None

        return {
            "temp_c": pick("temperature_2m"),
            "precip_mm": pick("precipitation"),
            "wind_kmh": pick("wind_speed_10m"),
            "wind_gust_kmh": pick("wind_gusts_10m"),
            "heure": times[idx].isoformat(),
            "source": "archive" if is_past else "prevision",
        }
    except Exception as exc:
        return {
            "temp_c": None, "precip_mm": None, "wind_kmh": None, "wind_gust_kmh": None,
            "source": "indisponible", "erreur": f"{type(exc).__name__}: {exc}",
        }


def venue_weather(venue: str, when: datetime) -> dict:
    """
    Meteo pour un stade connu, ou pour un lieu resolu par geocodage.
    Un stade couvert renvoie explicitement une meteo neutre.
    """
    if venue in STADIUMS:
        lat, lon, roof = STADIUMS[venue]
        if roof:
            return {
                "temp_c": None, "precip_mm": 0.0, "wind_kmh": 0.0,
                "source": "stade couvert", "stade": venue,
            }
        w = get_weather(lat, lon, when)
        w["stade"] = venue
        return w

    coords = geocode(venue)
    if coords is None:
        return {"source": "lieu introuvable", "stade": venue,
                "temp_c": None, "precip_mm": None, "wind_kmh": None}
    w = get_weather(coords[0], coords[1], when)
    w["stade"] = venue
    w["source"] += " (geocode)"
    return w


def describe(weather: dict) -> str:
    """Resume lisible pour l'affichage."""
    if not weather or weather.get("wind_kmh") is None:
        return weather.get("source", "meteo indisponible") if weather else "meteo indisponible"
    parts = []
    if weather.get("temp_c") is not None:
        parts.append(f"{weather['temp_c']:.0f}C")
    if weather.get("wind_kmh") is not None:
        parts.append(f"vent {weather['wind_kmh']:.0f} km/h")
    if weather.get("precip_mm"):
        parts.append(f"pluie {weather['precip_mm']:.1f} mm/h")
    else:
        parts.append("sec")
    return ", ".join(parts)
