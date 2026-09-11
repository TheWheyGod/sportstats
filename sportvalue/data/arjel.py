"""
Operateurs agrees ANJ (ex-ARJEL) et contraintes specifiques au marche francais.

POURQUOI CE MODULE EXISTE
-------------------------
Un edge mesure contre Pinnacle n'est pas jouable depuis la France : Pinnacle
n'a pas de licence ANJ. Tout l'outil doit donc separer deux roles qui sont
confondus dans la litterature anglo-saxonne :

  - la REFERENCE DE VERITE : le book le plus sharp du monde (Pinnacle, ou un
    exchange). On ne parie JAMAIS dessus, on s'en sert uniquement pour estimer
    la vraie probabilite. Sa ligne reste consultable meme sans compte.

  - le LIEU D'EXECUTION : un operateur agree ANJ. C'est la seule cote qui
    compte pour calculer un edge reellement encaissable.

Ce montage est en realite PLUS favorable que le montage classique modele-contre-
Pinnacle : au lieu de demander a un modele maison de battre le marche le plus
efficient du monde, on exploite l'ecart entre une ligne sharp et une ligne
francaise plus large.

LA CONTRAINTE DE TRJ (specifique a la France)
---------------------------------------------
La loi francaise plafonne le Taux de Retour au Joueur a 85% en moyenne annuelle
pour les paris sportifs en ligne (controle par l'ANJ). Un operateur peut
depasser 85% sur des paris individuels, a condition que la moyenne tienne.

Consequence strategique, et elle est contre-intuitive : comme les operateurs
affichent 93-94% de TRJ sur les marches vitrines (1X2 des grandes affiches) pour
rester competitifs, ils sont MECANIQUEMENT obliges de se rattraper ailleurs. Le
rattrapage se fait sur les combines, le live et les marches de niche.

En France il faut donc chercher la value sur les marches VITRINES, ce qui est
l'inverse exact du conseil habituel ("les books sont paresseux sur les marches
obscurs"). Ici les marches obscurs sont deliberement surtaxes.

Le module fournit `trj` et `is_playable` pour appliquer ce filtre
automatiquement : un marche a moins de ~90% de TRJ est ecarte avant meme de
regarder le modele.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.oddsmath import overround

__all__ = [
    "ARJEL_BOOKS",
    "ODDSAPI_FR_KEYS",
    "SHARP_REFERENCES",
    "trj",
    "is_playable",
    "trj_report",
    "effective_trj_multi_book",
    "TRJ_LEGAL_CAP",
]

# Plafond legal francais du taux de retour au joueur (moyenne annuelle).
TRJ_LEGAL_CAP = 0.85


@dataclass(frozen=True)
class Bookmaker:
    key: str
    nom: str
    oddsapi_key: str | None      # identifiant chez The Odds API, si couvert
    note: str = ""


# Principaux operateurs agrees ANJ pour les paris sportifs en ligne.
# La liste officielle et a jour est publiee par l'ANJ (https://anj.fr) : la
# verifier avant d'ouvrir un compte, les agrements evoluent.
ARJEL_BOOKS: dict[str, Bookmaker] = {
    "winamax": Bookmaker("winamax", "Winamax", "winamax_fr", "leader FR, cotes souvent competitives sur le foot"),
    "betclic": Bookmaker("betclic", "Betclic", "betclic_fr", "large offre, boosts frequents"),
    "unibet": Bookmaker("unibet", "Unibet France", "unibet_fr", "groupe Kindred"),
    "pmu": Bookmaker("pmu", "PMU", "pmu_fr", "historiquement hippique, lignes sportives plus larges"),
    "netbet": Bookmaker("netbet", "NetBet", "netbet_fr", ""),
    "parionssport": Bookmaker("parionssport", "ParionsSport En Ligne (FDJ)", None, "non couvert par The Odds API"),
    "zebet": Bookmaker("zebet", "ZEbet", None, "non couvert par The Odds API"),
    "bwin": Bookmaker("bwin", "Bwin France", None, "non couvert par The Odds API"),
    "genybet": Bookmaker("genybet", "Genybet", None, "non couvert par The Odds API"),
    "vbet": Bookmaker("vbet", "Vbet France", None, "non couvert par The Odds API"),
}

# Identifiants The Odds API interrogeables avec regions=fr
ODDSAPI_FR_KEYS = tuple(b.oddsapi_key for b in ARJEL_BOOKS.values() if b.oddsapi_key)

# PERIMETRE AUTOMATIQUE : les seuls operateurs relevables sans intervention
# manuelle. C'est le perimetre de travail par defaut de l'outil.
#
# Les autres (ParionsSport, ZEbet, Bwin, Genybet, Vbet) ne sont exposes par
# aucune API. Les recuperer supposerait de l'extraction automatisee, que leurs
# CGU interdisent et que leur protection anti-bot bloque -- avec le risque de
# faire fermer le compte qui sert justement a miser. Ils restent listes dans
# ARJEL_BOOKS pour la saisie manuelle, mais sortent du flux automatique.
AUTO_BOOKS = tuple(k for k, b in ARJEL_BOOKS.items() if b.oddsapi_key)
MANUAL_ONLY_BOOKS = tuple(k for k, b in ARJEL_BOOKS.items() if not b.oddsapi_key)

# References de verite : sharp, hors France, jamais utilisees pour miser.
SHARP_REFERENCES = ("pinnacle", "betfair_ex_eu", "smarkets", "matchbook")


def auto_book_names() -> list[str]:
    """Noms lisibles des operateurs couverts par le flux automatique."""
    return [ARJEL_BOOKS[k].nom for k in AUTO_BOOKS]


# --------------------------------------------------------------------------
def trj(odds) -> float:
    """
    Taux de retour au joueur d'un marche complet : TRJ = 1 / overround.

    Un 1X2 a [2.10, 3.40, 3.60] donne un overround de 1.048, soit un TRJ de
    95.4%. Un marche a TRJ 88% coute 12 points d'esperance : aucun modele
    raisonnable ne rattrape ca.
    """
    return float(1.0 / overround(odds))


def is_playable(odds, seuil_trj: float = 0.90) -> bool:
    """
    Le marche vaut-il la peine d'etre analyse ?

    Filtre a appliquer AVANT le modele : inutile de chercher un edge de 4% sur
    un marche qui en prend 12 au depart. Seuil par defaut 90%, ce qui ecarte la
    plupart des marches de niche francais tout en gardant les vitrines.
    """
    return trj(odds) >= seuil_trj


def trj_report(quotes_par_book: dict) -> list[dict]:
    """
    quotes_par_book : {nom_book: [cotes du marche complet]}
    Classe les books par TRJ decroissant sur ce marche precis.

    A regarder systematiquement avant de miser : le meilleur TRJ ne designe pas
    forcement le book ou l'on va parier (on parie a la meilleure cote sur NOTRE
    issue), mais un book durablement bas en TRJ sur un marche est un book a
    eviter pour ce marche.
    """
    rows = []
    for book, odds in quotes_par_book.items():
        try:
            t = trj(odds)
        except (ValueError, ZeroDivisionError):
            continue
        rows.append(
            {
                "book": book,
                "trj_%": round(100 * t, 2),
                "marge_%": round(100 * (1 - t), 2),
                "au_dessus_du_plafond_legal": t > TRJ_LEGAL_CAP,
                "jouable": t >= 0.90,
            }
        )
    return sorted(rows, key=lambda r: -r["trj_%"])


def effective_trj_multi_book(quotes_par_book: dict) -> dict:
    """
    TRJ EFFECTIF obtenu en prenant la meilleure cote de chaque issue parmi
    plusieurs books.

    C'est le levier numero un du parieur francais, et il est purement
    mecanique : avoir des comptes chez cinq operateurs et prendre a chaque fois
    la meilleure cote fait remonter le TRJ de plusieurs points sans aucun
    modele. Sur certains marches cela suffit a passer au-dessus de 100%
    (situation d'arbitrage, rare et vite corrigee).

    Retourne le TRJ de chaque book, le TRJ combine, et le gain en points.
    """
    if not quotes_par_book:
        return {}
    books = list(quotes_par_book)
    matrix = np.asarray([quotes_par_book[b] for b in books], dtype=float)
    best = matrix.max(axis=0)
    individuels = {b: trj(quotes_par_book[b]) for b in books}
    combine = trj(best)
    meilleur_seul = max(individuels.values())
    return {
        "trj_par_book": {b: round(100 * v, 2) for b, v in individuels.items()},
        "meilleur_book_seul_%": round(100 * meilleur_seul, 2),
        "trj_combine_%": round(100 * combine, 2),
        "gain_du_shopping_pts": round(100 * (combine - meilleur_seul), 2),
        "meilleures_cotes": best.tolist(),
        "book_par_issue": [books[i] for i in matrix.argmax(axis=0)],
        "arbitrage": combine > 1.0,
    }
