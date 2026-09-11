"""
Schema canonique. Toutes les sources de donnees sont converties vers ces
structures, ce qui permet aux modeles et au scanner d'ignorer d'ou viennent
les donnees.

Le champ `context` porte tout ce qui est specifique a un sport : meteo,
absences, repos, surface, altitude. Les modeles y piochent ce dont ils ont
besoin et ignorent le reste.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date as date_cls
from typing import Any, Optional

import pandas as pd

__all__ = [
    "Sport",
    "Fixture",
    "MatchResult",
    "OddsQuote",
    "MarketQuote",
    "ValueBet",
    "results_to_frame",
]


class Sport:
    FOOTBALL = "football"
    RUGBY = "rugby"
    BASKET = "basket"
    TENNIS = "tennis"
    ALL = (FOOTBALL, RUGBY, BASKET, TENNIS)


@dataclass
class Fixture:
    """Une rencontre a venir."""

    sport: str
    competition: str
    date: datetime
    home: str
    away: str
    neutral: bool = False
    fixture_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.fixture_id or f"{self.sport}:{self.date:%Y%m%d}:{self.home}:{self.away}"

    def label(self) -> str:
        sep = " vs " if self.neutral else " - "
        return f"{self.home}{sep}{self.away}"


@dataclass
class MatchResult:
    """Une rencontre jouee, avec son score."""

    sport: str
    competition: str
    date: datetime
    home: str
    away: str
    home_score: int
    away_score: int
    neutral: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def outcome_1x2(self) -> int:
        """0 = domicile, 1 = nul, 2 = exterieur."""
        if self.home_score > self.away_score:
            return 0
        if self.home_score == self.away_score:
            return 1
        return 2

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def total(self) -> int:
        return self.home_score + self.away_score


@dataclass
class OddsQuote:
    """Une cote proposee par un book sur une issue precise."""

    book: str
    market: str          # "1x2", "ou", "ah", "ml", "btts", ...
    selection: str       # "1", "X", "2", "over", "under", "home", "away"
    odds: float
    line: Optional[float] = None   # ligne O/U ou handicap
    timestamp: Optional[datetime] = None


@dataclass
class MarketQuote:
    """
    Un marche COMPLET chez un book : toutes les issues mutuellement
    exclusives. C'est l'unite minimale pour pouvoir de-vigger.
    """

    book: str
    market: str
    selections: list[str]
    odds: list[float]
    line: Optional[float] = None
    timestamp: Optional[datetime] = None

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.selections, self.odds))


@dataclass
class ValueBet:
    """Une opportunite detectee, prete a etre affichee ou journalisee."""

    fixture: Fixture
    market: str
    selection: str
    line: Optional[float]
    book: str
    odds: float
    p_model: float            # probabilite du modele seul
    p_market_fair: float      # probabilite du marche, marge retiree
    p_final: float            # probabilite retenue (modele fusionne au marche)
    edge: float               # p_final * odds - 1
    stake: float
    kelly_used: float
    confidence: str           # "haute" / "moyenne" / "faible"
    score: float              # score de classement (confiance x mauvais prix)
    rationale: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "date": self.fixture.date,
            "sport": self.fixture.sport,
            "competition": self.fixture.competition,
            "match": self.fixture.label(),
            "marche": self.market,
            "pari": self.selection + (f" {self.line:+g}" if self.line is not None else ""),
            "book": self.book,
            "cote": round(self.odds, 3),
            "p_modele": round(self.p_model, 4),
            "p_marche": round(self.p_market_fair, 4),
            "p_finale": round(self.p_final, 4),
            "edge_%": round(100 * self.edge, 2),
            "mise": round(self.stake, 2),
            "confiance": self.confidence,
            "score": round(self.score, 3),
        }


def results_to_frame(results: list[MatchResult]) -> pd.DataFrame:
    """Convertit une liste de resultats en DataFrame pour les modeles."""
    if not results:
        return pd.DataFrame(
            columns=["date", "competition", "home", "away", "home_score", "away_score", "neutral"]
        )
    rows = [
        {
            "date": pd.Timestamp(r.date),
            "competition": r.competition,
            "home": r.home,
            "away": r.away,
            "home_score": r.home_score,
            "away_score": r.away_score,
            "neutral": r.neutral,
            **{f"ctx_{k}": v for k, v in r.context.items() if isinstance(v, (int, float, str, bool))},
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
