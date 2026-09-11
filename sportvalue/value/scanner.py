"""
Detection des paris a HAUTE PROBABILITE et MAL CHIFFRES, cote ARJEL.

ARCHITECTURE : DEUX ROLES SEPARES
---------------------------------
  REFERENCE DE VERITE : la ligne la plus sharp disponible (Pinnacle, exchange).
      Consultable depuis la France, mais NON JOUABLE. Sert uniquement a estimer
      la vraie probabilite.
  LIEU D'EXECUTION : un operateur agree ANJ. Seule sa cote entre dans le calcul
      de l'edge, parce que c'est la seule qu'on peut reellement prendre.

Ce montage est plus favorable que le classique modele-contre-Pinnacle : on ne
demande pas a un modele maison de battre le marche le plus efficient du monde,
on exploite l'ecart entre une ligne sharp et une ligne francaise plus large.

CE QUE MESURE LE BACKTEST (a lire avant de miser)
--------------------------------------------------
Emulation d'un portefeuille de 5 books grand public (Bet365, Betway,
Interwetten, VCBet, William Hill), verite = Pinnacle deviggee, 41 000 matchs de
18 championnats sur 2019-2025, marche 1X2 :

  - le ROI est positif dans les 24 cellules de seuils testees (+1.4% a +7.9%) ;
  - AUCUNE n'est statistiquement significative : tous les IC 95% contiennent 0 ;
  - avec ~30 books (colonne Max, inaccessible en France) le meme filtre donnait
    +4.8% avec t = 3.0. L'edge se degrade donc fortement quand on passe de 30
    books a 5 comptes.

Conclusion honnete : direction encourageante, ampleur non prouvee. L'outil doit
etre utilise d'abord en suivi sans mise reelle, en tracant le CLV.

CLASSEMENT
----------
Le tri se fait sur le TAUX DE CROISSANCE ESPERE (approximation de Kelly)

    g ~= edge^2 / (2 * (cote - 1))

qui privilegie fortement les cotes basses a edge egal. Deux paris a +8% d'edge :

    p = 0.75 a cote 1.44  ->  g = 72.7 points de base par pari
    p = 0.12 a cote 9.00  ->  g =  4.0 points de base par pari

Le premier fait croitre le capital environ 18 fois plus vite pour le meme edge
affiche. C'est la traduction quantitative de "haute probabilite ET mal chiffre".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.calib import blend_logit, shrink_to_market
from ..core.kelly import kelly_stake
from ..core.oddsmath import best_odds_across_books, devig, is_sane_market, margin_pct
from ..data.arjel import ARJEL_BOOKS, ODDSAPI_FR_KEYS, trj
from ..data.schema import Fixture, MarketQuote, ValueBet

__all__ = ["ScannerConfig", "ValueScanner", "growth_score", "MarketScan"]


def growth_score(edge: float, odds: float) -> float:
    """Croissance esperee approchee, en points de base de bankroll par pari."""
    if odds <= 1.0 or edge <= 0:
        return 0.0
    return 10000.0 * (edge**2) / (2.0 * (odds - 1.0))


@dataclass
class ScannerConfig:
    """
    min_probability : seuil de "haute probabilite" (defaut 0.55).
    min_edge : edge minimal apres fusion. NB : le backtest 5 books trouve les
        meilleurs resultats vers 1-2%, pas au-dela. Un seuil eleve reduit
        l'echantillon sans ameliorer le ROI.
    blend_weight : poids du modele face a la reference sharp. MESURE A ZERO
        sur le football 1X2 (le Dixon-Coles n'ajoute rien a Pinnacle) : le
        defaut est donc 0. Le relever uniquement si `fit_blend_weight` le
        justifie sur VOS donnees, ou pour des marches que la reference ne cote
        pas (rugby avec meteo, totaux tennis derives).
    min_trj : TRJ minimal du marche ARJEL. La loi francaise plafonnant le TRJ
        moyen a 85%, les operateurs se rattrapent sur les marches de niche :
        ce filtre les ecarte avant toute analyse.
    """

    min_probability: float = 0.55
    min_edge: float = 0.015
    blend_weight: float = 0.0
    max_deviation: float = 0.15
    require_robust_devig: bool = True
    min_trj: float = 0.90
    devig_method: str = "shin"
    conservative_devig: bool = False
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02
    min_arjel_books: int = 2
    allow_no_reference: bool = False
    # Bornes de sante des marches. Un exchange sans liquidite renvoie des
    # cotes proches de 1.01 sur toutes les issues (overround > 2) : sans ce
    # controle, le de-vig les normalise et fabrique des edges de +200%.
    ref_overround: tuple = (0.98, 1.12)
    arjel_overround: tuple = (1.00, 1.25)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class MarketScan:
    selections: list[str]
    p_model: np.ndarray
    p_reference: np.ndarray
    p_final: np.ndarray
    best_odds: np.ndarray
    best_books: list[str]
    trj_arjel: float
    trj_best: float
    n_arjel_books: int
    robust_min_edge: np.ndarray
    reference_book: str
    reference_margin: float


class ValueScanner:
    def __init__(self, config: ScannerConfig | None = None):
        self.cfg = config or ScannerConfig()

    # ------------------------------------------------------------------
    def _reference_probs(
        self, reference: list[MarketQuote], arjel: list[MarketQuote]
    ) -> tuple[np.ndarray, str, float]:
        """
        Estimation de la vraie probabilite.

        Priorite a la ligne sharp. A defaut, on retombe sur le book ARJEL a la
        marge la plus faible -- solution degradee : on compare alors des books
        francais entre eux, ce qui ne detecte qu'un desaccord interne au marche
        francais, pas une erreur de prix par rapport a la verite.
        """
        if reference:
            best = min(reference, key=lambda q: margin_pct(q.odds))
            return (
                devig(best.odds, self.cfg.devig_method, self.cfg.conservative_devig),
                best.book,
                margin_pct(best.odds),
            )
        if not self.cfg.allow_no_reference:
            raise ValueError("Aucune reference sharp disponible")
        best = min(arjel, key=lambda q: margin_pct(q.odds))
        return (
            devig(best.odds, self.cfg.devig_method, self.cfg.conservative_devig),
            f"{best.book} (degrade)",
            margin_pct(best.odds),
        )

    def _robust_edges(
        self,
        reference: list[MarketQuote],
        arjel: list[MarketQuote],
        p_model: np.ndarray,
        best_odds: np.ndarray,
    ) -> np.ndarray:
        """Edge le plus defavorable sur les trois methodes de de-vig."""
        mins = np.full(len(best_odds), np.inf)
        saved = self.cfg.devig_method
        for method in ("shin", "power", "odds_ratio"):
            self.cfg.devig_method = method
            try:
                p_ref, _, _ = self._reference_probs(reference, arjel)
            except ValueError:
                continue
            p_bl = np.asarray(
                blend_logit(shrink_to_market(p_model, p_ref, self.cfg.max_deviation),
                            p_ref, self.cfg.blend_weight)
            ).ravel()
            mins = np.minimum(mins, p_bl * best_odds - 1.0)
        self.cfg.devig_method = saved
        return mins

    # ------------------------------------------------------------------
    def scan_market(
        self,
        p_model: np.ndarray,
        reference_quotes: list[MarketQuote],
        arjel_quotes: list[MarketQuote],
    ) -> MarketScan | None:
        n = len(p_model)
        self.rejets_sante: list[str] = []

        def garder(quotes, bornes, role):
            out = []
            for q in quotes:
                if not q.odds or len(q.odds) != n:
                    continue
                ok, motif = is_sane_market(q.odds, bornes[0], bornes[1])
                if ok:
                    out.append(q)
                else:
                    self.rejets_sante.append(f"{role} {q.book} ecarte : {motif}")
            return out

        reference = garder(reference_quotes, self.cfg.ref_overround, "reference")
        arjel = garder(arjel_quotes, self.cfg.arjel_overround, "ARJEL")
        if not arjel:
            return None

        p_model = np.asarray(p_model, dtype=float)
        p_model = p_model / p_model.sum()

        try:
            p_ref, ref_book, ref_margin = self._reference_probs(reference, arjel)
        except ValueError:
            return None

        p_shrunk = shrink_to_market(p_model, p_ref, self.cfg.max_deviation)
        p_final = np.asarray(blend_logit(p_shrunk, p_ref, self.cfg.blend_weight)).ravel()

        best_odds, best_books = best_odds_across_books({q.book: q.odds for q in arjel})
        worst_trj = min(trj(q.odds) for q in arjel)
        best_trj_single = max(trj(q.odds) for q in arjel)
        robust = self._robust_edges(reference, arjel, p_model, best_odds)

        return MarketScan(
            selections=list(arjel[0].selections),
            p_model=p_model,
            p_reference=p_ref,
            p_final=p_final,
            best_odds=best_odds,
            best_books=best_books,
            trj_arjel=best_trj_single,
            trj_best=trj(best_odds),
            n_arjel_books=len(arjel),
            robust_min_edge=robust,
            reference_book=ref_book,
            reference_margin=ref_margin,
        )

    # ------------------------------------------------------------------
    def find_value(
        self,
        fixture: Fixture,
        market_name: str,
        p_model: np.ndarray,
        reference_quotes: list[MarketQuote],
        arjel_quotes: list[MarketQuote],
        line: float | None = None,
        extra_rationale: list[str] | None = None,
    ) -> list[ValueBet]:
        scan = self.scan_market(p_model, reference_quotes, arjel_quotes)
        if scan is None:
            return []
        if scan.trj_arjel < self.cfg.min_trj:
            return []          # marche structurellement surtaxe : on n'analyse pas

        bets: list[ValueBet] = []
        for i, sel in enumerate(scan.selections):
            p_fin = float(scan.p_final[i])
            odds = float(scan.best_odds[i])
            edge = p_fin * odds - 1.0

            if p_fin < self.cfg.min_probability:
                continue
            if edge < self.cfg.min_edge:
                continue
            if self.cfg.require_robust_devig and scan.robust_min_edge[i] <= 0:
                continue

            info = kelly_stake(
                p_fin, odds,
                bankroll=self.cfg.bankroll,
                fraction=self.cfg.kelly_fraction,
                max_stake_pct=self.cfg.max_stake_pct,
            )
            # Le book peut arriver sous sa cle courte ("winamax", via CSV) ou
            # sous son identifiant The Odds API ("winamax_fr"). On accepte les
            # deux, sinon les noms ne se resolvent jamais sur le chemin manuel.
            raw_book = scan.best_books[i]
            book_nom = next(
                (b.nom for b in ARJEL_BOOKS.values()
                 if raw_book in (b.key, b.oddsapi_key)),
                raw_book,
            )

            rationale = list(extra_rationale or [])
            rationale.append(
                f"verite ({scan.reference_book}, marge {100*scan.reference_margin:.1f}%) : "
                f"{100*scan.p_reference[i]:.1f}% -> cote fair {1/max(scan.p_reference[i],1e-9):.2f}"
            )
            rationale.append(
                f"meilleure cote ARJEL {odds:.2f} chez {book_nom} "
                f"parmi {scan.n_arjel_books} operateurs -> edge {100*edge:+.2f}%"
            )
            rationale.append(
                f"TRJ du marche : {100*scan.trj_arjel:.1f}% chez le meilleur book seul, "
                f"{100*scan.trj_best:.1f}% en prenant la meilleure cote de chaque issue"
            )
            if self.cfg.blend_weight > 0:
                rationale.append(
                    f"modele {100*scan.p_model[i]:.1f}% fusionne a w={self.cfg.blend_weight:.2f}"
                )
            rationale.append(
                f"edge le plus defavorable sur les 3 methodes de de-vig : "
                f"{100*scan.robust_min_edge[i]:+.2f}%"
            )

            bets.append(
                ValueBet(
                    fixture=fixture,
                    market=market_name,
                    selection=sel,
                    line=line,
                    book=book_nom,
                    odds=odds,
                    p_model=float(scan.p_model[i]),
                    p_market_fair=float(scan.p_reference[i]),
                    p_final=p_fin,
                    edge=edge,
                    stake=info["stake"],
                    kelly_used=info["kelly_used"],
                    confidence=self._confidence(scan, i, p_fin),
                    score=growth_score(edge, odds),
                    rationale=rationale,
                    diagnostics={
                        "trj_meilleur_book_%": round(100 * scan.trj_arjel, 2),
                        "trj_multi_books_%": round(100 * scan.trj_best, 2),
                        "n_books_arjel": scan.n_arjel_books,
                        "reference": scan.reference_book,
                        "marge_reference_%": round(100 * scan.reference_margin, 2),
                        "edge_robuste_%": round(100 * float(scan.robust_min_edge[i]), 2),
                        "kelly_plein": round(info["kelly_full"], 4),
                        "mise_plafonnee": info["capped"],
                        "croissance_pb": round(growth_score(edge, odds), 1),
                    },
                )
            )
        return bets

    # ------------------------------------------------------------------
    def _confidence(self, scan: MarketScan, i: int, p_fin: float) -> str:
        """
        Signaux independants de l'edge lui-meme (sinon on ne ferait que le
        renommer). La presence d'une vraie reference sharp pese le plus lourd :
        sans elle, on ne compare que des books francais entre eux.
        """
        pts = 0
        if "degrade" not in scan.reference_book:
            pts += 2
        if scan.n_arjel_books >= 4:
            pts += 1
        if scan.trj_arjel >= 0.93:
            pts += 1
        if scan.robust_min_edge[i] > 0.01:
            pts += 1
        if p_fin >= 0.65:
            pts += 1
        if pts >= 5:
            return "haute"
        if pts >= 3:
            return "moyenne"
        return "faible"

    @staticmethod
    def rank(bets: list[ValueBet], top: int | None = None) -> list[ValueBet]:
        out = sorted(bets, key=lambda b: -b.score)
        return out[:top] if top else out
