"""
Saisie manuelle des cotes : le chemin utilisable immediatement, sans cle d'API.

The Odds API ne couvre que cinq operateurs ARJEL (Winamax, Betclic, Unibet,
PMU, NetBet) et son palier gratuit est vite epuise. ParionsSport, ZEbet et
Bwin France n'y sont pas du tout. Or ce sont precisement les books ou les
ecarts de cote peuvent etre les plus larges.

D'ou ce format CSV : on releve les cotes a la main sur les marches qui
interessent, l'outil fait le reste. C'est fastidieux, mais c'est aussi la
seule facon d'inclure TOUS les operateurs agrees et de verifier que la cote
existe vraiment au moment de miser -- une cote d'API vieille de quinze minutes
n'est pas une cote jouable.

FORMAT ATTENDU
--------------
Une ligne par ISSUE, une colonne par book. Colonnes obligatoires :
    match, marche, issue
Colonnes de books : n'importe quel nom (winamax, betclic, ..., pinnacle).
Colonnes optionnelles : date, competition, sport, ligne

    match,competition,marche,issue,winamax,betclic,unibet,pinnacle
    PSG-Lyon,Ligue 1,1x2,1,1.55,1.57,1.54,1.60
    PSG-Lyon,Ligue 1,1x2,X,4.30,4.25,4.40,4.50
    PSG-Lyon,Ligue 1,1x2,2,5.50,5.60,5.40,5.90

Toute colonne dont le nom figure dans ARJEL_BOOKS est traitee comme un lieu
d'execution ; 'pinnacle', 'betfair', 'smarkets', 'matchbook' comme reference
de verite. Les autres sont ignorees, avec un avertissement.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .arjel import ARJEL_BOOKS, SHARP_REFERENCES
from .schema import Fixture, MarketQuote

__all__ = ["load_manual_csv", "write_template", "MANUAL_TEMPLATE"]

_REQUIRED = {"match", "marche", "issue"}

MANUAL_TEMPLATE = """match,competition,sport,date,marche,ligne,issue,pinnacle,winamax,betclic,unibet,pmu,netbet
PSG-Lyon,Ligue 1,football,2026-09-13 21:00,1x2,,1,1.60,1.55,1.57,1.54,1.52,1.56
PSG-Lyon,Ligue 1,football,2026-09-13 21:00,1x2,,X,4.50,4.30,4.25,4.40,4.20,4.35
PSG-Lyon,Ligue 1,football,2026-09-13 21:00,1x2,,2,5.90,5.50,5.60,5.40,5.30,5.45
Toulouse-La Rochelle,Top 14,rugby,2026-09-13 17:00,ah,-6.5,home,1.92,1.85,1.88,1.87,1.83,1.86
Toulouse-La Rochelle,Top 14,rugby,2026-09-13 17:00,ah,-6.5,away,1.92,1.90,1.87,1.88,1.92,1.89
"""


def _classify(col: str) -> str:
    c = col.strip().lower()
    if c in ARJEL_BOOKS:
        return "arjel"
    if any(s in c for s in SHARP_REFERENCES) or c in ("pinnacle", "betfair", "smarkets", "matchbook"):
        return "reference"
    return "ignore"


def write_template(path: str | Path) -> Path:
    p = Path(path)
    p.write_text(MANUAL_TEMPLATE, encoding="utf-8")
    return p


def load_manual_csv(path: str | Path, verbose: bool = True) -> list[dict]:
    """
    Retourne une liste de dicts :
        {'fixture': Fixture, 'market': str, 'line': float|None,
         'reference': [MarketQuote], 'arjel': [MarketQuote]}

    Les issues sont conservees dans l'ordre d'apparition du fichier : c'est
    cet ordre qui doit correspondre a celui des probabilites du modele.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing}")

    book_cols = {c: _classify(c) for c in df.columns if c not in
                 {"match", "competition", "sport", "date", "marche", "ligne", "issue"}}
    ignored = [c for c, k in book_cols.items() if k == "ignore"]
    if ignored and verbose:
        print(f"  [i] colonnes ignorees (book inconnu) : {', '.join(ignored)}")
    arjel_cols = [c for c, k in book_cols.items() if k == "arjel"]
    ref_cols = [c for c, k in book_cols.items() if k == "reference"]
    if not arjel_cols:
        raise ValueError(
            "Aucune colonne de book ARJEL reconnue. Books valides : "
            + ", ".join(sorted(ARJEL_BOOKS))
        )
    if not ref_cols and verbose:
        print("  [!] aucune reference sharp (pinnacle/betfair/smarkets) : "
              "l'analyse comparera des books francais entre eux, ce qui est "
              "nettement moins fiable.")

    out = []
    group_cols = ["match", "marche"] + (["ligne"] if "ligne" in df.columns else [])
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        match_name, marche = keys[0], keys[1]
        ligne = keys[2] if len(keys) > 2 and pd.notna(keys[2]) else None

        selections = [str(s) for s in g["issue"].tolist()]
        first = g.iloc[0]

        date = datetime.now()
        if "date" in g.columns and pd.notna(first.get("date")):
            try:
                date = pd.to_datetime(first["date"]).to_pydatetime()
            except (ValueError, TypeError):
                pass

        home, away = (match_name.split("-", 1) + [""])[:2] if "-" in str(match_name) else (str(match_name), "")
        fixture = Fixture(
            sport=str(first.get("sport", "football")),
            competition=str(first.get("competition", "")),
            date=date,
            home=home.strip(),
            away=away.strip(),
        )

        def quotes_for(cols):
            qs = []
            for c in cols:
                vals = pd.to_numeric(g[c], errors="coerce").tolist()
                if any(pd.isna(v) or v <= 1.0 for v in vals):
                    continue
                qs.append(
                    MarketQuote(
                        book=c, market=str(marche), selections=selections,
                        odds=[float(v) for v in vals],
                        line=float(ligne) if ligne is not None else None,
                    )
                )
            return qs

        arjel_q = quotes_for(arjel_cols)
        if not arjel_q:
            if verbose:
                print(f"  [!] {match_name} / {marche} : aucune cote ARJEL complete, ignore")
            continue

        out.append(
            {
                "fixture": fixture,
                "market": str(marche),
                "line": float(ligne) if ligne is not None else None,
                "selections": selections,
                "reference": quotes_for(ref_cols),
                "arjel": arjel_q,
            }
        )

    if verbose:
        print(f"  {len(out)} marche(s) charge(s) depuis {path}")
    return out
