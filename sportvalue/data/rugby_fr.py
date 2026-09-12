"""
Rugby francais : Top 14 et Pro D2, via les tableaux de Wikipedia.

Ni la LNR ni aucune API libre ne publient l'historique des resultats. Les pages
de saison de Wikipedia, en revanche, contiennent un TABLEAU CROISE complet :
lignes = equipe a domicile, colonnes = equipe a l'exterieur, cellules "38-40".
Pour le Top 14 cela fait 182 matchs par saison (14 equipes x 13 adversaires x 2),
240 pour la Pro D2 (16 equipes).

DEUX LIMITES, A CONNAITRE AVANT DE S'EN SERVIR
-----------------------------------------------
1. PAS DE DATES. Le tableau croise donne les scores, pas le calendrier. On
   repartit donc les matchs uniformement sur la fenetre de la saison. La
   decroissance temporelle reste correcte D'UNE SAISON A L'AUTRE -- ce qui est
   l'essentiel -- mais elle ne distingue pas septembre de mai a l'interieur
   d'une meme saison. Avec une demi-vie de 200 jours, la perte est modeste.

2. SOURCE COMMUNAUTAIRE. Un tableau de Wikipedia peut etre incomplet en cours
   de saison (matchs non encore saisis) ou comporter une coquille. Le
   chargeur signale le taux de remplissage : un tableau a 60% signifie que la
   saison est en cours, pas que la source est cassee.

Les phases finales ne figurent pas dans le tableau croise : seule la saison
reguliere est recuperee. C'est sans consequence pour estimer la force des
equipes, et cela evite de compter deux fois des confrontations.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from .cache import get_cache

__all__ = ["load_season", "load_top14", "load_prod2", "TOP14", "PROD2"]

TOP14 = "Championnat de France de rugby à XV {saison}"
PROD2 = "Championnat de France de rugby à XV de 2e division {saison}"

_TTL = 60 * 60 * 24 * 3  # 3 jours
_SCORE = re.compile(r"^\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*$")


def _page_html(titre: str) -> str:
    import requests

    url = "https://fr.wikipedia.org/wiki/" + requests.utils.quote(titre.replace(" ", "_"))

    def loader() -> str:
        r = requests.get(url, timeout=45, headers={"User-Agent": "sportvalue/1.0"})
        r.raise_for_status()
        return r.text

    return get_cache().get_text(url, _TTL, loader)


def _find_cross_table(html: str) -> pd.DataFrame | None:
    """
    Repere le tableau croise parmi les ~80 tableaux de la page.

    Critere : une matrice carree dont beaucoup de cellules ont la forme
    "nombre - nombre". On prend celle qui en contient le plus, ce qui evite de
    confondre avec le classement ou les tableaux de phase finale.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return None
    best, best_n = None, 0
    for t in tables:
        if t.shape[0] < 8 or abs(t.shape[0] - t.shape[1]) > 1:
            continue
        n = sum(1 for v in t.astype(str).values.ravel() if _SCORE.match(str(v)))
        if n > best_n:
            best, best_n = t, n
    return best if best_n >= 20 else None


def load_season(competition: str, saison: str, verbose: bool = True) -> pd.DataFrame:
    """
    competition : TOP14 ou PROD2 (les gabarits de titre ci-dessus)
    saison      : "2025-2026"
    """
    titre = competition.format(saison=saison)
    t = _find_cross_table(_page_html(titre))
    if t is None:
        if verbose:
            print(f"   [!] {titre} : tableau de resultats introuvable")
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score"])

    # Colonne 0 = equipes a domicile ; les colonnes suivent le MEME ordre.
    equipes = [str(x).strip() for x in t.iloc[1:, 0]]
    n = len(equipes)
    lignes = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = str(t.iloc[i + 1, j + 1]).strip()
            m = _SCORE.match(cell)
            if not m:
                continue
            lignes.append({
                "home": equipes[i], "away": equipes[j],
                "home_score": int(m.group(1)), "away_score": int(m.group(2)),
            })

    if not lignes:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score"])

    df = pd.DataFrame(lignes)
    # Dates reparties uniformement : la saison va de septembre a juin.
    an = int(saison.split("-")[0])
    debut, fin = pd.Timestamp(an, 9, 1), pd.Timestamp(an + 1, 6, 15)
    df["date"] = pd.date_range(debut, fin, periods=len(df))
    df["neutral"] = False
    df["saison"] = saison

    attendu = n * (n - 1)
    if verbose:
        print(f"   {titre[:52]:<52} {len(df):>3}/{attendu} matchs "
              f"({100*len(df)/attendu:.0f}% du tableau rempli, {n} equipes)")
    return df[["date", "home", "away", "home_score", "away_score", "neutral", "saison"]]


def _load_many(competition: str, saisons, verbose: bool = True) -> pd.DataFrame:
    frames = [load_season(competition, s, verbose) for s in saisons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score"])
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if verbose:
        print(f"   total : {len(df)} matchs, {df['home'].nunique()} equipes")
    return df


def load_top14(saisons=("2023-2024", "2024-2025", "2025-2026"), verbose: bool = True):
    return _load_many(TOP14, saisons, verbose)


def load_prod2(saisons=("2023-2024", "2024-2025", "2025-2026"), verbose: bool = True):
    return _load_many(PROD2, saisons, verbose)


_MOIS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
         "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
         "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12}
_JOURNEE = re.compile(r"(\d{1,2})\s*(?:re|e|ère)?\s*journ[ée]e\s*(.{0,70})", re.I)
_MOIS_ANNEE = re.compile(r"([a-zéûôàèA-Z]+)\s+(\d{4})")


def _date_journee(texte: str):
    """
    Extrait la PLAGE de dates d'une journee.

    "samedi 5 et dimanche 6 septembre 2026" -> (5 septembre, 6 septembre).

    UNE PLAGE, PAS UN JOUR UNIQUE. Une journee de championnat s'etale sur deux
    jours et Wikipedia ne donne pas la date match par match. En Pro D2,
    Aurillac-Brive se joue le JEUDI quand les sept autres rencontres de la meme
    journee ont lieu le VENDREDI : afficher un jour unique pour toute la
    journee est donc faux pour la plupart des matchs. Mieux vaut annoncer
    "17-18 sept" que d'inventer une precision dont on ne dispose pas.

    Le premier jour se repere en localisant le couple mois+annee puis en
    prenant le premier numero qui le precede. Un regex "jour mois annee" ne
    matcherait que "6 septembre 2026" -- le seul numero colle au mois -- et
    decalerait tout le calendrier de 24 h.
    """
    m = _MOIS_ANNEE.search(texte)
    if not m:
        return None, None
    mois = _MOIS.get(m.group(1).lower())
    if not mois:
        return None, None
    jours = re.findall(r"\b(\d{1,2})\b", texte[: m.start()])
    if not jours:
        return None, None
    try:
        debut = pd.Timestamp(int(m.group(2)), mois, int(jours[0]))
        fin = pd.Timestamp(int(m.group(2)), mois, int(jours[-1]))
        return debut, (fin if fin >= debut else debut)
    except ValueError:
        return None, None


def load_current_season(competition: str, saison: str, verbose: bool = True) -> tuple:
    """
    Saison EN COURS : resultats deja joues et rencontres a venir.

    Le tableau croise ne sert a rien tant que la saison n'est pas avancee : il
    est vide. Wikipedia publie en revanche des tableaux PAR JOURNEE, et c'est
    la seule source libre du calendrier -- ni la LNR ni aucune API ne l'exposent.

    PARTICULARITE DU FORMAT, facile a manquer : chaque tableau contient DEUX
    journees. Les lignes 1 a 7 sont la journee annoncee dans l'en-tete (avec
    les scores si elle est jouee), les lignes 8 a 14 sont la SUIVANTE, sans
    score. Ne lire que la premiere moitie fait disparaitre une journee sur
    deux -- dont, au moment ou j'ecris, celle qui contient Bordeaux-Toulouse.

    La date de la seconde moitie n'est ecrite nulle part : on la deduit en
    ajoutant 7 jours a celle de l'en-tete, ce qui correspond au rythme normal
    du championnat. C'est une INFERENCE, signalee comme telle dans la sortie.
    """
    titre = competition.format(saison=saison)
    try:
        tables = pd.read_html(io.StringIO(_page_html(titre)))
    except (ValueError, Exception):
        if verbose:
            print(f"   [!] {titre} : page illisible")
        return pd.DataFrame(), pd.DataFrame()

    joues, a_venir = [], []
    for tb in tables:
        if tb.shape[0] < 8 or tb.shape[1] < 5:
            continue
        m = _JOURNEE.match(str(tb.iloc[0, 0]).strip())
        if not m:
            continue
        num = int(m.group(1))
        date0, date_fin = _date_journee(m.group(2))
        if date0 is None:
            continue

        n_par_journee = (tb.shape[0] - 1) // 2
        for k in range(1, tb.shape[0]):
            dom, ext = tb.iloc[k, 1], tb.iloc[k, 4]
            if pd.isna(dom) or pd.isna(ext):
                continue
            seconde = k > n_par_journee
            decalage = pd.Timedelta(days=7 if seconde else 0)
            date = date0 + decalage
            fin = (date_fin or date0) + decalage
            sh, sa = pd.to_numeric(tb.iloc[k, 2], errors="coerce"), \
                     pd.to_numeric(tb.iloc[k, 3], errors="coerce")
            # Les tableaux par journee abregent ("Toulouse") la ou le tableau
            # croise ecrit en entier ("Stade toulousain"). Sans cette
            # normalisation, le modele voit DEUX equipes distinctes : 29 au
            # lieu de 14, chacune avec la moitie des donnees, et un Toulouse
            # donne perdant chez lui.
            ligne = {"date": date,
                     "home": ALIAS.get(str(dom).strip().lower(), str(dom).strip()),
                     "away": ALIAS.get(str(ext).strip().lower(), str(ext).strip()),
                     "date_fin": fin,
                     "journee": num + (1 if seconde else 0), "neutral": False}
            if pd.notna(sh) and pd.notna(sa):
                joues.append({**ligne, "home_score": float(sh), "away_score": float(sa)})
            else:
                a_venir.append(ligne)

    dj = pd.DataFrame(joues).drop_duplicates(subset=["home", "away", "journee"])
    dv = pd.DataFrame(a_venir).drop_duplicates(subset=["home", "away", "journee"])
    if not dj.empty:
        dj = dj.sort_values("date").reset_index(drop=True)
    if not dv.empty:
        dv = dv.sort_values("date").reset_index(drop=True)
    if verbose:
        print(f"   {titre[:50]:<50} {len(dj)} joue(s), {len(dv)} a venir")
        noms = set()
        for d_ in (dj, dv):
            if not d_.empty:
                noms |= set(d_["home"]) | set(d_["away"])
        brut = [n for n in noms if n.lower() in ALIAS and ALIAS[n.lower()] != n]
        if brut:
            print(f"   [!] noms non normalises : {brut} -- a ajouter a ALIAS")
        if not dv.empty:
            print(f"   prochaine journee : J{int(dv['journee'].iloc[0])} "
                  f"le {dv['date'].iloc[0]:%d/%m/%Y} "
                  "(date deduite pour les journees paires)")
    return dj, dv


# The Odds API n'a ni Top 14 ni Pro D2 : les noms ci-dessous servent a resoudre
# une saisie manuelle de calendrier vers les libelles de Wikipedia.
ALIAS = {
    "toulouse": "Stade toulousain", "stade toulousain": "Stade toulousain",
    "bordeaux": "Union Bordeaux Bègles", "ubb": "Union Bordeaux Bègles",
    "bordeaux bègles": "Union Bordeaux Bègles",
    "bordeaux begles": "Union Bordeaux Bègles",
    "vannes": "RC Vannes", "rc vannes": "RC Vannes",
    "oyonnax": "Oyonnax Rugby", "montauban": "US Montauban",
    "toulouse": "Stade toulousain",
    "la rochelle": "Stade rochelais", "stade rochelais": "Stade rochelais",
    "toulon": "RC Toulon", "rct": "RC Toulon",
    "racing": "Racing 92", "racing 92": "Racing 92",
    "clermont": "ASM Clermont", "asm": "ASM Clermont",
    "lyon": "Lyon OU", "lou": "Lyon OU",
    "castres": "Castres olympique", "co": "Castres olympique",
    "pau": "Section paloise", "section paloise": "Section paloise",
    "perpignan": "USA Perpignan", "usap": "USA Perpignan",
    "bayonne": "Aviron bayonnais", "aviron bayonnais": "Aviron bayonnais",
    "montpellier": "Montpellier HR", "mhr": "Montpellier HR",
    "stade francais": "Stade français", "stade français": "Stade français",
    "stade français paris": "Stade français", "stade francais paris": "Stade français",
    # Pro D2 : noms courts des tableaux par journee et noms Highlightly ->
    # libelles des tableaux croises (historique)
    "agen": "SU Agen", "su agen": "SU Agen",
    "aurillac": "Stade aurillacois", "stade aurillacois": "Stade aurillacois",
    "biarritz": "Biarritz olympique", "biarritz olympique": "Biarritz olympique",
    "brive": "CA Brive", "ca brive": "CA Brive",
    "béziers": "AS Béziers", "beziers": "AS Béziers", "as béziers": "AS Béziers",
    "colomiers": "Colomiers Rugby", "colomiers rugby": "Colomiers Rugby",
    "dax": "US Dax", "us dax": "US Dax",
    "grenoble": "FC Grenoble", "fc grenoble": "FC Grenoble", "grenoble fc": "FC Grenoble",
    "nevers": "USON Nevers", "uson nevers": "USON Nevers",
    "us oyonnax": "Oyonnax Rugby", "oyonnax rugby": "Oyonnax Rugby",
    "angouleme": "Soyaux Angoulême XV", "angoulême": "Soyaux Angoulême XV",
    "soyaux angoulême": "Soyaux Angoulême XV", "soyaux angouleme": "Soyaux Angoulême XV",
    "valence romans": "Valence Romans DR", "valence romans dr": "Valence Romans DR",
    "provence": "Provence Rugby", "carcassonne": "US Carcassonne",
    "mont-de-marsan": "Stade montois", "mont de marsan": "Stade montois",
    "nissa": "Nice", "stade niçois": "Nice", "stade nicois": "Nice", "nice": "Nice",
    "narbonne": "Narbonne", "rc narbonne": "Narbonne",
    "paris": "Stade français", "montauban": "US Montauban",
}


def resolve_team(nom: str, connues) -> str | None:
    """Apparie un nom saisi a la main aux libelles Wikipedia."""
    connues = set(connues)
    if nom in connues:
        return nom
    a = ALIAS.get(str(nom).strip().lower())
    if a and a in connues:
        return a
    cible = str(nom).lower().strip()
    cands = {c for c in connues if cible in c.lower()}
    if len(cands) == 1:
        return cands.pop()
    return None


# --------------------------------------------------------------------------
# Marqueurs d'essais (tables "Meilleurs marqueurs" des pages de saison)
# --------------------------------------------------------------------------
# Poste Wikipedia (francais, parfois compose "Ailier, arriere") -> prior.
_POSTES_FR = [
    ("ailier", "ailier"), ("arrière", "arriere"), ("arriere", "arriere"),
    ("centre", "centre"), ("troisième ligne", "troisieme_ligne"),
    ("troisieme ligne", "troisieme_ligne"), ("numéro 8", "troisieme_ligne"),
    ("demi de mêlée", "demi_de_melee"), ("demi de melee", "demi_de_melee"),
    ("demi d'ouverture", "ouverture"), ("ouverture", "ouverture"),
    ("deuxième ligne", "deuxieme_ligne"), ("deuxieme ligne", "deuxieme_ligne"),
    ("pilier", "premiere_ligne"), ("talonneur", "premiere_ligne"),
]
_RESTE_JOUEURS_XV = 13       # joueurs d'un XV + banc jamais listes, en moyenne
_DECAY_PREV = 0.6
# Les joueurs listes sont les stars du championnat : ils changent moins de
# club que la moyenne, et Wikipedia n'offre aucun effectif pour verifier.
_PRESENCE_PREV = 0.8


def _poste_fr(texte: str) -> str:
    t = str(texte or "").lower()
    for cle, poste in _POSTES_FR:
        if cle in t:
            return poste
    return "inconnu"


def _table_marqueurs(html: str):
    """La table 'Meilleurs marqueurs d'essais' : Joueur, Club, Poste, Essais, MJ, TJ."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "essais" in cols and "joueur" in cols and "club" in cols and "points" not in cols:
            t = t.copy()
            t.columns = cols
            return t
    return None


def load_try_scorers(competition: str, saison_courante: str, verbose: bool = True) -> pd.DataFrame:
    """
    Meilleurs marqueurs d'essais, saison en cours + saison passee, au format
    ScorerModel : joueur, equipe (nom Wikipedia du club), poste, minutes,
    buts_hors_penalty (= essais), minutes_attendues, saison_seule_passee.

    COUVERTURE VOLONTAIREMENT ASSUMEE COMME FAIBLE : Wikipedia ne liste que
    les 10 a 15 meilleurs marqueurs du championnat, soit environ un joueur
    par club. Tous les autres sont regroupes par une ligne collective par
    club, ajoutee par l'appelant (voir cli), pour que la somme des esperances
    reste celle de l'equipe.
    """
    debut = int(str(saison_courante)[:4])
    frames = []
    for annee, poids, presence in ((debut, 1.0, 1.0), (debut - 1, _DECAY_PREV, _PRESENCE_PREV)):
        saison = f"{annee}-{annee + 1}"
        titre = competition.format(saison=saison)
        try:
            t = _table_marqueurs(_page_html(titre))
        except Exception as e:
            if verbose:
                print(f"   [!] marqueurs {saison} : {type(e).__name__}")
            continue
        if t is None or t.empty:
            if verbose:
                print(f"   [i] marqueurs {saison} : table introuvable")
            continue
        for r in t.itertuples(index=False):
            d = r._asdict()
            try:
                essais = float(d.get("essais") or 0)
                mj = float(d.get("mj") or 0)
                tj = float(d.get("tj") or (mj * 60.0))
            except (TypeError, ValueError):
                continue
            if not d.get("joueur") or mj <= 0:
                continue
            frames.append({
                "joueur": str(d["joueur"]).strip(), "club": str(d.get("club") or "").strip(),
                "poste": _poste_fr(d.get("poste")), "essais": essais * poids,
                "minutes": tj * poids, "min_par_match": min(tj / mj, 80.0) * presence,
                "saison": saison, "poids": poids,
            })
        if verbose:
            print(f"   marqueurs {saison} : {len(t)} joueurs listes")
    if not frames:
        return pd.DataFrame()
    d = pd.DataFrame(frames)
    # un joueur present les deux saisons : cumul, club et minutes attendues
    # de la saison la plus recente
    d = d.sort_values("poids", ascending=False)
    agg = d.groupby("joueur", sort=False).agg(
        club=("club", "first"), poste=("poste", "first"),
        buts_hors_penalty=("essais", "sum"), minutes=("minutes", "sum"),
        minutes_attendues=("min_par_match", "first"), poids=("poids", "max"),
    ).reset_index()
    agg["saison_seule_passee"] = agg["poids"] < 1.0
    agg["tireur_penalty"] = 0
    return agg.drop(columns=["poids"])


def lignes_collectives(equipes, journee_hint: int = 0) -> pd.DataFrame:
    """Une ligne 'Autres joueurs' par club : le reste de l'effectif au prior."""
    return pd.DataFrame([{
        "joueur": "Autres joueurs", "club": e, "equipe": e, "poste": "collectif",
        "buts_hors_penalty": 0.0, "minutes": 0.0,
        "minutes_attendues": 80.0 * _RESTE_JOUEURS_XV,
        "saison_seule_passee": False, "tireur_penalty": 0,
    } for e in equipes])
