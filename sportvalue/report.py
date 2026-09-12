"""
Rapport HTML autonome : la surface de consultation des predictions.

Le terminal convient pour verifier un chiffre, pas pour consulter vingt marches
d'un coup. Ce module produit une page unique, sans dependance externe, qui
s'ouvre dans un navigateur et se partage telle quelle.

Deux choix de fond :

  - CHAQUE PROBABILITE EST DOUBLEE DE SA COTE EQUITABLE (1/p). C'est la seule
    forme sous laquelle une probabilite se compare d'un coup d'oeil a une ligne
    de bookmaker. Afficher 54% oblige a un calcul mental ; afficher 54% / 1.85
    ne l'oblige a rien.

  - LA MATRICE DES SCORES EXACTS est rendue en heatmap plutot qu'en liste. La
    liste des douze scores les plus probables cache la STRUCTURE : ou se
    concentre la masse, si elle penche vers le haut ou le bas du tableau, si
    les nuls forment une diagonale marquee. La grille la montre d'un coup.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

__all__ = ["build_report", "write_report"]

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Instrument+Serif:ital@0;1&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=IBM+Plex+Mono:wght@400;500&display=swap\">"
)

_CSS = """
:root{
  --paper:#F5F3EF; --panel:#FFFFFF; --panel-2:#FAF8F5;
  --ink:#141A21; --ink-2:#5A636E; --ink-3:#8B939D;
  --line:#E2DED6; --line-2:#EFEBE4;
  --home:#A63A2B; --away:#2A5C8A; --draw:#77808D;
  --heat:#A63A2B; --ok:#1F6F4F;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0F1317; --panel:#171C22; --panel-2:#1C222A;
    --ink:#E9E6E0; --ink-2:#9EA7B2; --ink-3:#6C7580;
    --line:#28303A; --line-2:#212830;
    --home:#DE7460; --away:#63A2DA; --draw:#8C95A2;
    --heat:#DE7460; --ok:#4FAE86;
  }
}
:root[data-theme="dark"]{
  --paper:#0F1317; --panel:#171C22; --panel-2:#1C222A;
  --ink:#E9E6E0; --ink-2:#9EA7B2; --ink-3:#6C7580;
  --line:#28303A; --line-2:#212830;
  --home:#DE7460; --away:#63A2DA; --draw:#8C95A2;
  --heat:#DE7460; --ok:#4FAE86;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.5;
}
.wrap{max-width:1120px; margin:0 auto; padding:28px 20px 64px}

/* ---------- bandeau ---------- */
.eyebrow{
  font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-3); font-weight:600;
}
.scoreboard{
  display:grid; grid-template-columns:1fr auto 1fr; gap:20px; align-items:center;
  padding:26px 22px; margin:14px 0 26px;
  background:var(--panel); border:1px solid var(--line); border-radius:3px;
}
.team{font-family:"Instrument Serif",Georgia,serif; font-size:38px; line-height:1.05; text-wrap:balance}
.team.h{color:var(--home)}
.team.a{color:var(--away); text-align:right}
.xg{font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--ink-2); margin-top:6px}
.a .xg{text-align:right}
.vs{
  font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3);
  text-align:center; letter-spacing:.1em;
}
.vs strong{display:block; font-size:26px; color:var(--ink); letter-spacing:0; margin-top:2px}

/* ---------- barres 1X2 ---------- */
.bars{display:flex; height:52px; border-radius:2px; overflow:hidden; border:1px solid var(--line)}
.bar{
  display:flex; flex-direction:column; justify-content:center; align-items:center;
  color:#fff; font-size:12px; min-width:0; padding:0 6px; overflow:hidden;
}
.bar.mini span,.bar.nano b{display:none}
.bar b{font-family:"IBM Plex Mono",monospace; font-size:16px; font-weight:500}
.bar span{opacity:.82; font-size:11px; white-space:nowrap; max-width:100%; overflow:hidden; text-overflow:ellipsis}
.bar.h{background:var(--home)} .bar.d{background:var(--draw)} .bar.a{background:var(--away)}

/* ---------- grille de panneaux ---------- */
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(272px,1fr)); gap:14px; margin-top:14px}
.panel{
  background:var(--panel); border:1px solid var(--line); border-radius:3px;
  padding:16px 18px 18px; min-width:0;
}
.panel h2{
  font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:var(--ink-3);
  margin:0 0 12px; font-weight:600;
}
.panel.wide{grid-column:1/-1}
.panel h3{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3);
          font-weight:600; margin:14px 0 4px}

/* ---------- lignes de marche ---------- */
table{width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums}
th{
  font-size:10px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3);
  font-weight:600; text-align:right; padding:0 0 7px; border-bottom:1px solid var(--line-2);
}
th:first-child{text-align:left}
td{padding:6px 0; border-bottom:1px solid var(--line-2); font-size:14px}
tr:last-child td{border-bottom:none}
td.num{font-family:"IBM Plex Mono",monospace; text-align:right; white-space:nowrap}
td.odds{font-family:"IBM Plex Mono",monospace; text-align:right; color:var(--ink-3); font-size:13px}
.lab{color:var(--ink-2)}
.meter{position:relative; height:4px; background:var(--line-2); border-radius:2px; margin-top:5px}
.meter i{position:absolute; inset:0 auto 0 0; border-radius:2px; background:var(--heat)}

/* ---------- heatmap des scores ---------- */
.matrix{overflow-x:auto}
.matrix table{min-width:420px; border-collapse:separate; border-spacing:2px}
.matrix td, .matrix th{border:none; padding:0}
.matrix th{
  font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--ink-3);
  text-align:center; padding:2px; letter-spacing:0;
}
.cell{
  aspect-ratio:1; min-width:38px; border-radius:2px;
  display:flex; align-items:center; justify-content:center;
  font-family:"IBM Plex Mono",monospace; font-size:11px;
}
.cell.draw{outline:1px solid var(--draw); outline-offset:-1px}
.axis{
  font-size:10px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink-3); font-weight:600;
}

/* ---------- buteurs ---------- */
.scorer{display:grid; grid-template-columns:1fr auto auto; gap:10px; align-items:baseline;
        padding:7px 0; border-bottom:1px solid var(--line-2)}
.scorer:last-child{border-bottom:none}
.scorer .nm{min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; vertical-align:middle}
.dot.h{background:var(--home)} .dot.a{background:var(--away)}
.scorer .pct{font-family:"IBM Plex Mono",monospace; font-size:14px; text-align:right}
.scorer .fo{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3); text-align:right; min-width:52px}
.role{font-size:11px; color:var(--ink-3)}

.note{font-size:12.5px; color:var(--ink-2); margin-top:10px; line-height:1.55}
.foot{margin-top:30px; padding-top:16px; border-top:1px solid var(--line);
      font-size:12px; color:var(--ink-3); display:flex; flex-wrap:wrap; gap:16px}
.tag{font-family:"IBM Plex Mono",monospace}
@media (max-width:640px){
  .scoreboard{grid-template-columns:1fr; gap:10px}
  .team,.team.a{font-size:29px; text-align:left}
  .a .xg{text-align:left}
}
"""


def _paris(dt):
    """Convertit un horodatage UTC (naif ou non) en heure de Paris."""
    if dt is None:
        return None
    try:
        from datetime import timezone
        from zoneinfo import ZoneInfo
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Europe/Paris"))
    except Exception:
        return dt


def _fo(p: float) -> str:
    """Cote equitable. Au-dela de 1000, l'affichage n'apporte plus rien."""
    if p is None or p <= 0:
        return "—"
    v = 1.0 / p
    return f"{v:.2f}" if v < 100 else ("999+" if v > 999 else f"{v:.0f}")


def _pct(p: float) -> str:
    return f"{100*p:.1f}%"


def _esc(s) -> str:
    return html.escape(str(s))


def _rows(pairs, meter_max=None):
    """Lignes libelle / probabilite / cote equitable, avec jauge."""
    mx = meter_max or max((p for _, p in pairs), default=1) or 1
    out = []
    for lab, p in pairs:
        out.append(
            f'<tr><td class="lab">{_esc(lab)}'
            f'<div class="meter"><i style="width:{100*p/mx:.1f}%"></i></div></td>'
            f'<td class="num">{_pct(p)}</td><td class="odds">{_fo(p)}</td></tr>'
        )
    return (
        '<table><thead><tr><th>Issue</th><th>Proba</th><th>Cote éq.</th></tr></thead>'
        f"<tbody>{''.join(out)}</tbody></table>"
    )


def _ou_table(rows, libelle="Plus de") -> str:
    """Tableau Plus/Moins avec cote equitable, pour toute liste de lignes."""
    out = []
    for r in rows:
        po, pu = r.get("over", 0), r.get("under", 0)
        out.append(
            f'<tr><td class="lab">{_esc(libelle)} {_esc(r["ligne"])}'
            f'<div class="meter"><i style="width:{100*po:.1f}%"></i></div></td>'
            f'<td class="num">{_pct(po)}</td><td class="odds">{_fo(po)}</td>'
            f'<td class="num">{_pct(pu)}</td><td class="odds">{_fo(pu)}</td></tr>'
        )
    return ("<table><thead><tr><th>Ligne</th><th>Plus</th><th>Cote éq.</th>"
            f"<th>Moins</th><th>Cote éq.</th></tr></thead><tbody>{''.join(out)}</tbody></table>")


def _two_sides_table(rows_h, rows_a, home, away, cle="over") -> str:
    """Une ligne par seuil, une colonne par camp : 'Plus de 1.5' dom. / ext."""
    out = []
    for rh, ra in zip(rows_h, rows_a):
        ph, pa = rh.get(cle, rh.get("p_over", 0)), ra.get(cle, ra.get("p_over", 0))
        out.append(
            f'<tr><td class="lab">Plus de {_esc(rh["ligne"])}</td>'
            f'<td class="num">{_pct(ph)}</td><td class="odds">{_fo(ph)}</td>'
            f'<td class="num">{_pct(pa)}</td><td class="odds">{_fo(pa)}</td></tr>'
        )
    return (f"<table><thead><tr><th>Ligne</th><th>{_esc(home)}</th><th>Cote éq.</th>"
            f"<th>{_esc(away)}</th><th>Cote éq.</th></tr></thead><tbody>{''.join(out)}</tbody></table>")


def _matrix_panel(sd, home: str, away: str, maxg: int = 6) -> str:
    """Heatmap des scores exacts, construite depuis la distribution elle-meme."""
    grid = np.zeros((maxg + 1, maxg + 1))
    for hh, aa, w in zip(sd.h, sd.a, sd.w):
        i, j = int(min(hh, maxg)), int(min(aa, maxg))
        grid[i, j] += w
    mx = grid.max() or 1.0

    head = "".join(f"<th>{j}</th>" for j in range(maxg + 1))
    body = []
    for i in range(maxg + 1):
        cells = []
        for j in range(maxg + 1):
            p = grid[i, j]
            alpha = (p / mx) ** 0.55
            txt = f"{100*p:.1f}" if p >= 0.006 else ""
            # Texte clair une fois le fond suffisamment dense.
            col = "#fff" if alpha > 0.58 else "var(--ink-2)"
            cls = "cell draw" if i == j else "cell"
            cells.append(
                f'<td><div class="{cls}" style="background:color-mix(in srgb,'
                f'var(--heat) {100*alpha:.0f}%, transparent); color:{col}"'
                f' title="{i}-{j} : {100*p:.2f}%">{txt}</div></td>'
            )
        body.append(f'<th>{i}</th>{"".join(cells)}')
    rows = "".join(f"<tr>{r}</tr>" for r in body)
    return (
        '<div class="panel wide"><h2>Scores exacts — probabilité en %</h2>'
        f'<div class="matrix"><table><thead><tr><th></th>'
        f'<th colspan="{maxg+1}" class="axis">Buts {_esc(away)}</th></tr>'
        f"<tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
        f'<div class="note">Lignes : buts de {_esc(home)}. '
        "La diagonale entourée est celle des matchs nuls. "
        "L'intensité suit la probabilité — elle montre où se concentre la masse, "
        "ce qu'une liste des douze scores les plus probables ne dit pas.</div></div>"
    )


def _scorers_panel(scorers, home: str, away: str) -> str:
    if scorers is None or len(scorers) == 0:
        return ""
    if "poste" in scorers.columns:
        scorers = pd.concat([scorers[scorers["poste"] != "collectif"],
                             scorers[scorers["poste"] == "collectif"]])
    d = scorers.head(18)
    lignes = []
    for r in d.itertuples(index=False):
        cote = "h" if r.equipe_cote == "domicile" else "a"
        poste = getattr(r, "poste", "")
        if poste == "collectif":
            poste = "reste de l'effectif"
        prem = getattr(r, "p_premier_buteur", None)
        extra = f" · 1er {100*prem:.1f}%" if prem is not None else ""
        lignes.append(
            f'<div class="scorer"><div class="nm"><span class="dot {cote}"></span>'
            f'{_esc(r.joueur)} <span class="role">{_esc(poste)}{extra}</span></div>'
            f'<div class="pct">{_pct(r.p_marque)}</div>'
            f'<div class="fo">{_fo(r.p_marque)}</div></div>'
        )
    aucun = float(d["p_aucun_buteur"].iloc[0]) if "p_aucun_buteur" in d.columns else None
    note = (
        f'<div class="note">Aucun buteur : <b>{_pct(aucun)}</b> (cote éq. {_fo(aucun)}). '
        "La somme des espérances individuelles égale celle de l'équipe : les buteurs "
        "ne peuvent pas contredire le marché des buts.</div>"
        if aucun is not None else ""
    )
    return (
        f'<div class="panel wide"><h2>Buteurs — probabilité de marquer</h2>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0 26px">'
        f'{"".join(lignes)}</div>{note}</div>'
    )


def _assists_panel(passeurs: list, home: str, away: str) -> str:
    """Passes decisives : meme presentation que les buteurs."""
    if not passeurs:
        return ""
    # la ligne collective "reste de l'effectif" ferme la liste, quel que soit
    # son rang : un pot commun en tete d'un classement de passeurs ne se lit pas
    ordre = ([r for r in passeurs if r.get("poste") != "collectif"]
             + [r for r in passeurs if r.get("poste") == "collectif"])
    lignes = []
    for r in ordre[:18]:
        cote = "h" if r.get("equipe_cote") == "domicile" else "a"
        poste = r.get("poste", "")
        if poste == "collectif":
            poste = "reste de l'effectif"
        p = r.get("p_passe", 0.0)
        lignes.append(
            f'<div class="scorer"><div class="nm"><span class="dot {cote}"></span>'
            f'{_esc(r.get("joueur", ""))} <span class="role">{_esc(poste)}</span></div>'
            f'<div class="pct">{_pct(p)}</div><div class="fo">{_fo(p)}</div></div>'
        )
    return (
        "<div class=\"panel wide\"><h2>Passeurs — probabilité d'au moins une passe décisive</h2>"
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0 26px">'
        f'{"".join(lignes)}</div>'
        '<div class="note">Environ 72 % des buts sont assistés : les espérances de passes '
        "somment à 72 % des buts attendus de l'équipe. Hors Premier League, la source ne liste "
        'que les joueurs ayant déjà marqué : un pur créateur sans but est sous-estimé.</div></div>'
    )


def _tries_panel(marqueurs: list, p_au_moins_un=None) -> str:
    """Marqueurs d'essais : au moins un essai, deux ou plus, premier essai."""
    if not marqueurs:
        return ""
    ordre = ([r for r in marqueurs if r.get("poste") != "collectif"]
             + [r for r in marqueurs if r.get("poste") == "collectif"])
    lignes = []
    for r in ordre[:18]:
        cote = "h" if r.get("equipe_cote") == "domicile" else "a"
        poste = str(r.get("poste", "")).replace("_", " ")
        if poste == "collectif":
            poste = "reste de l'effectif"
        p = r.get("p_marque", 0.0)
        p2 = r.get("p_2plus")
        p1 = r.get("p_premier_buteur")
        detail = " · ".join(x for x in (
            f"2+ {100*p2:.1f}%" if p2 is not None else "",
            f"1er {100*p1:.1f}%" if p1 is not None else "") if x)
        lignes.append(
            f'<div class="scorer"><div class="nm"><span class="dot {cote}"></span>'
            f'{_esc(r.get("joueur", ""))} <span class="role">{_esc(poste)}{" · " + detail if detail else ""}</span></div>'
            f'<div class="pct">{_pct(p)}</div><div class="fo">{_fo(p)}</div></div>'
        )
    note = ""
    if p_au_moins_un is not None:
        note = f"Au moins un essai dans le match : <b>{_pct(p_au_moins_un)}</b>. "
    return (
        "<div class=\"panel wide\"><h2>Marqueurs d'essais — probabilité de marquer au moins un essai</h2>"
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0 26px">'
        f'{"".join(lignes)}</div>'
        f'<div class="note">{note}La somme des espérances individuelles égale les essais attendus de '
        "l'équipe. Top 14 et Pro D2 : seuls les meilleurs marqueurs du championnat sont connus "
        "(Wikipedia) ; « Autres joueurs » porte tout le reste de l'effectif.</div></div>"
    )


def _bars(marches: dict, home: str, away: str, cls: str = "bars") -> str:
    """Barres de vainqueur, largeur proportionnelle a la probabilite."""
    tri = marches.get("1x2") or marches.get("vainqueur") or marches.get("moneyline") or {}
    if not tri:
        return ""
    lib = {"1": home, "X": "Nul", "2": away}
    classe = {"1": "h", "X": "d", "2": "a"}
    seg = []
    for k, p in tri.items():
        # Un segment etroit ne peut pas porter son texte : a moins de 10 %
        # on retire le libelle, a moins de 5 % le pourcentage aussi. Sinon
        # "2.0%" du nul au rugby debordait sur les segments voisins. Le
        # detail reste lisible au survol et dans les marches.
        taille = " mini" if p < 0.10 else ""
        taille = " mini nano" if p < 0.05 else taille
        seg.append(
            f'<div class="bar {classe.get(k, "a" if k not in classe else classe[k])}{taille}" '
            f'style="flex:{max(p,0.02)}" title="{_esc(lib.get(k,k))} {_pct(p)}">'
            f"<b>{_pct(p)}</b><span>{_esc(lib.get(k,k))} · {_fo(p)}</span></div>"
        )
    return f'<div class="{cls}">{chr(10).join(seg)}</div>'


def _panels(home: str, away: str, marches: dict, sd=None, scorers=None,
            sport: str = "football") -> list:
    """Panneaux de marches. Partages par le rapport d'un match et la vue journee."""
    panels = []

    def panel(titre, contenu, wide=False):
        if contenu:
            panels.append(
                f'<div class="panel{" wide" if wide else ""}"><h2>{_esc(titre)}</h2>{contenu}</div>'
            )

    # --- en direct : rappel de ce sur quoi les probabilites sont conditionnees
    if marches.get("conditionnel"):
        panels.append('<div class="panel wide"><h2>En direct</h2><div class="note">'
                      + " ".join(_esc(x) for x in marches["conditionnel"]) + "</div></div>")

    # --- totaux (buts / points / essais / jeux)
    for cle, titre in [
        ("total_buts", "Nombre de buts"),
        ("total_points", "Nombre de points"),
        ("total_essais", "Nombre d'essais"),
        ("total_jeux", "Nombre de jeux"),
    ]:
        bloc = marches.get(cle)
        if not bloc:
            continue
        lignes = []
        for r in bloc:
            po, pu = r.get("over", r.get("p_over", 0)), r.get("under", r.get("p_under", 0))
            push = r.get("push", 0) or 0
            extra = f' <span class="role">push {100*push:.1f}%</span>' if push > 0.005 else ""
            lignes.append(
                f'<tr><td class="lab">Plus de {_esc(r["ligne"])}{extra}'
                f'<div class="meter"><i style="width:{100*po:.1f}%"></i></div></td>'
                f'<td class="num">{_pct(po)}</td><td class="odds">{_fo(po)}</td>'
                f'<td class="num">{_pct(pu)}</td><td class="odds">{_fo(pu)}</td></tr>'
            )
        panel(
            titre,
            "<table><thead><tr><th>Ligne</th><th>Plus</th><th>Cote éq.</th>"
            f"<th>Moins</th><th>Cote éq.</th></tr></thead><tbody>{''.join(lignes)}</tbody></table>",
        )

    if "btts" in marches:
        b = marches["btts"]
        panel("Les deux équipes marquent", _rows([("Oui", b["oui"]), ("Non", b["non"])], 1.0))

    if "double_chance" in marches:
        dc = marches["double_chance"]
        panel("Double chance", _rows([(f"{home} ou nul", dc["1X"]),
                                      (f"{home} ou {away}", dc["12"]),
                                      (f"Nul ou {away}", dc["X2"])], 1.0))

    if "draw_no_bet" in marches:
        dnb = marches["draw_no_bet"]
        panel("Remboursé si nul", _rows([(home, dnb.get("1", 0)), (away, dnb.get("2", 0))], 1.0))

    if marches.get("total_domicile") and marches.get("total_exterieur"):
        panel("Buts par équipe",
              _two_sides_table(marches["total_domicile"], marches["total_exterieur"], home, away)
              + '<div class="note">« Moins de 0,5 » pour une équipe = cage inviolée pour l\'autre.</div>')

    if "premiere_equipe_a_marquer" in marches:
        f = marches["premiere_equipe_a_marquer"]
        panel("Première équipe à marquer",
              _rows([(home, f["domicile"]), (away, f["exterieur"]), ("Aucun but", f["aucune"])], 1.0))

    if "resultat_mi_temps" in marches:
        ht = marches["resultat_mi_temps"]
        pr = marches["mi_temps_prolifique"]
        bp = marches.get("buts_par_periode", {})
        note = ""
        if bp:
            note = (f'<div class="note">Buts attendus 1<sup>re</sup> / 2<sup>e</sup> période : '
                    f'{_esc(home)} {bp["dom_1ere"]:.2f} / {bp["dom_2eme"]:.2f}, '
                    f'{_esc(away)} {bp["ext_1ere"]:.2f} / {bp["ext_2eme"]:.2f}. '
                    f'Dans ce championnat, {100*bp["part_1ere_ligue"]:.0f} % des buts tombent avant la pause.</div>')
        panel("Mi-temps",
              '<h3>Résultat à la mi-temps</h3>'
              + _rows([(home, ht["1"]), ("Nul", ht["X"]), (away, ht["2"])], 1.0)
              + '<h3>Période la plus prolifique</h3>'
              + _rows([("1re mi-temps", pr["1ere"]), ("Égalité", pr["egalite"]),
                       ("2e mi-temps", pr["2eme"])], 1.0)
              + '<h3>Buts par période</h3>'
              + _two_sides_table(marches["buts_1ere_mt"], marches["buts_2eme_mt"],
                                 "1re MT", "2e MT")
              + _rows([("Un but dans chaque mi-temps", marches.get("but_chaque_mt", 0))], 1.0)
              + note)

    if "mi_temps_fin" in marches:
        t = marches["mi_temps_fin"]
        lib = {"1": home, "X": "Nul", "2": away}
        head = "".join(f"<th>{_esc(lib[c])}</th>" for c in "1X2")
        body = "".join(
            f'<tr><td class="lab">{_esc(lib[r])}</td>'
            + "".join(f'<td class="num">{_pct(t[r][c])}<br><span class="role">{_fo(t[r][c])}</span></td>'
                      for c in "1X2")
            + "</tr>" for r in "1X2")
        panel("Mi-temps / fin de match",
              f'<table><thead><tr><th>MT ↓ &nbsp; FM →</th>{head}</tr></thead><tbody>{body}</tbody></table>'
              '<div class="note">Lignes : résultat à la mi-temps ; colonnes : résultat final. '
              'La somme de chaque colonne est exactement le 1X2 du modèle.</div>')

    if "corners" in marches:
        c = marches["corners"]
        att = c["attendus"]
        panel("Corners",
              f'<div class="note">Attendus : {_esc(home)} {att["domicile"]:.1f}, '
              f'{_esc(away)} {att["exterieur"]:.1f}, total {att["total"]:.1f}.</div>'
              + _ou_table(c["total"])
              + '<h3>Par équipe</h3>'
              + _two_sides_table(c["domicile"], c["exterieur"], home, away))

    if "cartons" in marches:
        c = marches["cartons"]
        att = c["attendus"]
        arb = c.get("arbitre")
        note_arb = ""
        if arb:
            f = arb["facteur_cartons"]
            tendance = "plus sévère" if f > 1.03 else ("plus clément" if f < 0.97 else "dans la moyenne")
            note_arb = (f'<div class="note">Arbitre : <b>{_esc(arb["nom"])}</b>, {tendance} que la moyenne '
                        f'({f:.2f} × sur {arb["matchs"]} matchs, moyenne {arb["moyenne_ligue"]:.1f} jaunes/match). '
                        f'Facteur appliqué aux deux équipes.</div>')
        rouge = marches.get("carton_rouge")
        bloc_rouge = ""
        if rouge:
            bloc_rouge = ('<h3>Carton rouge</h3>'
                          + _rows([("Au moins un dans le match", rouge["match"]),
                                   (home, rouge["domicile"]), (away, rouge["exterieur"])], 1.0))
        panel("Cartons jaunes",
              f'<div class="note">Attendus : {_esc(home)} {att["domicile"]:.1f}, '
              f'{_esc(away)} {att["exterieur"]:.1f}, total {att["total"]:.1f}.</div>'
              + _ou_table(c["total"])
              + '<h3>Par équipe</h3>'
              + _two_sides_table(c["domicile"], c["exterieur"], home, away)
              + bloc_rouge + note_arb)

    if "ecart_par_tranche" in marches:
        panel("Écart de victoire",
              _rows(list(marches["ecart_par_tranche"].items())), )

    if "essais_attendus" in marches:
        e = marches["essais_attendus"]
        panel("Essais attendus",
              f'<table><tbody><tr><td class="lab">{_esc(home)}</td>'
              f'<td class="num">{e["domicile"]:.2f}</td></tr>'
              f'<tr><td class="lab">{_esc(away)}</td><td class="num">{e["exterieur"]:.2f}</td></tr>'
              f'<tr><td class="lab">Total</td><td class="num">{e["total"]:.2f}</td></tr></tbody></table>')

    if "score_en_sets" in marches:
        panel("Score en sets",
              _rows([(s["score"], s["p"]) for s in marches["score_en_sets"]]))

    if "handicap_jeux" in marches or "handicap" in marches:
        bloc = marches.get("handicap_jeux") or marches.get("handicap")
        lignes = []
        for r in bloc:
            ks = [k for k in r if k.startswith("p_")]
            if len(ks) < 2:
                continue
            lignes.append(
                f'<tr><td class="lab">{_esc(r.get("ligne", r.get("ligne_domicile","")))}</td>'
                f'<td class="num">{_pct(r[ks[0]])}</td><td class="odds">{_fo(r[ks[0]])}</td>'
                f'<td class="num">{_pct(r[ks[1]])}</td><td class="odds">{_fo(r[ks[1]])}</td></tr>'
            )
        panel("Handicap",
              "<table><thead><tr><th>Ligne</th><th>Dom.</th><th>Cote éq.</th>"
              f"<th>Ext.</th><th>Cote éq.</th></tr></thead><tbody>{''.join(lignes)}</tbody></table>")

    if sd is not None and sport == "football":
        panels.append(_matrix_panel(sd, home, away))
    if scorers is not None and sport == "football":
        s = _scorers_panel(scorers, home, away)
        if s:
            panels.append(s)
    if marches.get("passeurs"):
        panels.append(_assists_panel(marches["passeurs"], home, away))
    if marches.get("marqueurs_essais"):
        panels.append(_tries_panel(marches["marqueurs_essais"], marches.get("premier_essai_marque")))

    absents = marches.get("absences")
    if absents:
        def bloc(cote, titre):
            lst = [x for x in absents if x.get("cote") == cote]
            if not lst:
                return f'<div class="note">{_esc(titre)} : aucune absence signalée</div>'
            items = "".join(
                f'<div class="abs{" doute" if x.get("poids", 1) < 1 else ""}">'
                f'<span class="nm">{_esc(x["joueur"])}</span>'
                f'<span class="role">{_esc(x.get("raison") or "")}'
                f'{" · incertain" if x.get("poids", 1) < 1 else ""}</span></div>'
                for x in lst[:10])
            return (f'<div class="abs-col"><h3>{_esc(titre)} '
                    f'<span class="role">{len(lst)}</span></h3>{items}</div>')
        panel("Absences",
              f'<div class="abs-grid">{bloc("domicile", home)}{bloc("exterieur", away)}</div>'
              '<div class="note">Forfaits et incertains signalés par API-Football. '
              'Une absence « incertaine » reste possible en jeu : elle est affichée '
              'mais pèse moitié moins dans tout comptage.</div>', True)

    if "meteo" in marches and marches["meteo"]:
        panel("Météo", "".join(f'<div class="note">· {_esc(n)}</div>' for n in marches["meteo"]), True)

    return panels


UNITES = {"football": "buts", "rugby": "points", "basket": "points", "tennis": "jeux"}


def build_report(
    home: str,
    away: str,
    marches: dict,
    sd=None,
    scorers=None,
    competition: str = "",
    sport: str = "football",
    meta: dict | None = None,
    standalone: bool = True,
) -> str:
    """Rapport d'un match. `standalone=False` renvoie un fragment (pour Artifact)."""
    meta = meta or {}
    xg = marches.get("buts_attendus") or marches.get("points_attendus") or {}
    xh, xa = xg.get("domicile", 0), xg.get("exterieur", 0)
    bars = _bars(marches, home, away)
    panels = _panels(home, away, marches, sd, scorers, sport)

    unite = UNITES.get(sport, "points")
    corps = f"""
<div class="wrap">
  <div class="eyebrow">{_esc(competition or sport)} · modèle {_esc(meta.get('modele','')) }</div>
  <div class="scoreboard">
    <div><div class="team h">{_esc(home)}</div><div class="xg">{xh:.2f} {unite} attendus</div></div>
    <div class="vs">Prévision<strong>{xh:.1f} – {xa:.1f}</strong></div>
    <div><div class="team a">{_esc(away)}</div><div class="xg">{xa:.2f} {unite} attendus</div></div>
  </div>
  {bars}
  <div class="grid">{''.join(panels)}</div>
  <div class="foot">
    <span class="tag">généré le {_paris(datetime.now(timezone.utc)):%d/%m/%Y à %H:%M}</span>
    <span>Probabilités issues des données — aucune cote n'entre dans le calcul.</span>
    <span>« Cote éq. » = 1/probabilité, à comparer directement à une ligne de bookmaker.</span>
  </div>
</div>"""

    tete = f"<title>{_esc(home)} – {_esc(away)}</title>{_FONTS}<style>{_CSS}</style>"
    if not standalone:
        return tete + corps
    return (
        "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"{tete}</head><body>{corps}</body></html>"
    )


_CSS_SLATE = """
.filters{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:16px 0 18px}
.chip{
  font:500 12px/1 "IBM Plex Sans",sans-serif; letter-spacing:.03em;
  padding:7px 13px; border:1px solid var(--line); border-radius:999px;
  background:var(--panel); color:var(--ink-2); cursor:pointer;
}
.chip[aria-pressed="true"]{background:var(--ink); color:var(--paper); border-color:var(--ink)}
.chip:focus-visible{outline:2px solid var(--heat); outline-offset:2px}
#q{
  flex:1; min-width:160px; padding:8px 12px; font:400 13px "IBM Plex Sans",sans-serif;
  background:var(--panel); border:1px solid var(--line); border-radius:3px; color:var(--ink);
}
#q:focus-visible{outline:2px solid var(--heat); outline-offset:1px}
.count{font-size:12px; color:var(--ink-3); font-family:"IBM Plex Mono",monospace}
.chip .n{font-family:"IBM Plex Mono",monospace; font-size:11px; opacity:.6; margin-left:3px}
.tri{display:flex; align-items:center; gap:7px; font-size:12px; color:var(--ink-3)}
#comp{max-width:min(100%,320px)}
.tri select{
  font:400 12.5px "IBM Plex Sans",sans-serif; padding:7px 9px; color:var(--ink);
  background:var(--panel); border:1px solid var(--line); border-radius:3px;
}
.tri select:focus-visible{outline:2px solid var(--heat); outline-offset:1px}

.slate{display:flex; flex-direction:column; gap:10px}
.match{background:var(--panel); border:1px solid var(--line); border-radius:3px; overflow:hidden}
.match[hidden]{display:none !important}
.mhead{display:grid; grid-template-columns:132px 1fr 210px; gap:16px; align-items:center; padding:14px 18px}
.mwhen{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3); line-height:1.4}
.mwhen b{display:block; color:var(--ink-2); font-weight:500; font-size:11px;
         letter-spacing:.08em; text-transform:uppercase}
/* Tableau d'affichage : deux colonnes EGALES autour du tiret, nom a domicile
   cale a droite, nom exterieur cale a gauche, chaque nombre de buts attendus
   sous son equipe. Avec une premiere colonne "auto", un nom long a domicile
   (Cronulla Sutherland Sharks) prenait toute la place et poussait
   "North Queensland Cowboys" sur trois lignes, sous les barres. */
.mteams{display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:end;
        font-family:"Instrument Serif",Georgia,serif; font-size:22px; line-height:1.15; min-width:0}
.mteams .h,.mteams .a{overflow-wrap:anywhere; text-wrap:balance}
.mteams .h,.mteams .xh{color:var(--home); text-align:right; justify-self:end}
.mteams .a,.mteams .xa{color:var(--away); text-align:left; justify-self:start}
.mteams .sep{color:var(--ink-3); font-size:15px; padding:0 8px 2px; text-align:center}
.mteams .xg,.mteams .xl{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3);
                        margin-top:3px; line-height:1.3}
.mteams .xl{font-size:10px; text-align:center; white-space:nowrap; padding:0 8px}
.mbars{display:flex; height:34px; border-radius:2px; overflow:hidden}
.mbars .bar b{font-size:13px} .mbars .bar span{font-size:10px}
.mkeys{display:flex; flex-wrap:wrap; gap:6px 18px; padding:0 18px 12px;
       font-size:12.5px; color:var(--ink-2)}
.mkeys b{font-family:"IBM Plex Mono",monospace; color:var(--ink); font-weight:500}
details{border-top:1px solid var(--line-2)}
summary{
  cursor:pointer; padding:10px 18px; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600; list-style:none;
}
summary::-webkit-details-marker{display:none}
summary::after{content:" ▸"; display:inline-block; transition:transform .15s}
details[open] summary::after{transform:rotate(90deg)}
summary:focus-visible{outline:2px solid var(--heat); outline-offset:-2px}
details .grid{padding:0 14px 16px}
.mwhen .h{display:block; font-size:15px; color:var(--ink); font-weight:500; margin-top:2px}
.fermer{
  display:block; width:calc(100% - 28px); margin:4px 14px 14px; padding:9px;
  font:600 11px/1 "IBM Plex Sans",sans-serif; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-3); background:var(--panel-2); border:1px solid var(--line);
  border-radius:3px; cursor:pointer;
}
.fermer:hover{color:var(--ink); border-color:var(--ink-3)}
.fermer:focus-visible{outline:2px solid var(--heat); outline-offset:2px}
.empty{padding:40px 18px; text-align:center; color:var(--ink-3); font-size:14px}

/* separateurs de jour */
.day-sep{
  display:flex; align-items:baseline; gap:12px; margin:22px 2px 2px;
  font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-3); font-weight:600;
}
.day-sep::after{content:""; flex:1; height:1px; background:var(--line)}
.day-sep .n{font-family:"IBM Plex Mono",monospace; letter-spacing:0; opacity:.7}
.day-sep[hidden]{display:none !important}
.day-sep.today{color:var(--heat)}

/* statut de match */
.statut{
  display:inline-flex; align-items:center; gap:6px; font-size:11px;
  font-family:"IBM Plex Mono",monospace; padding:3px 8px; border-radius:2px;
  border:1px solid var(--line); color:var(--ink-2); white-space:nowrap;
}
.statut.live{border-color:var(--home); color:var(--home)}
.statut.live .pt{
  width:6px; height:6px; border-radius:50%; background:var(--home);
  animation:pulse 1.6s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.statut.live .pt{animation:none}}
.statut b{font-weight:500; color:var(--ink)}
.verdict{font-size:11px; padding:2px 7px; border-radius:2px; font-family:"IBM Plex Mono",monospace}
.verdict.ok{background:color-mix(in srgb, var(--ok) 16%, transparent); color:var(--ok)}
.verdict.ko{background:color-mix(in srgb, var(--home) 14%, transparent); color:var(--home)}
.maj{font-size:11px; color:var(--ink-3); font-family:"IBM Plex Mono",monospace}
.abs-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:6px 24px}
.abs-col h3{font-size:12px; margin:0 0 6px; font-weight:600; color:var(--ink-2)}
.abs{display:flex; justify-content:space-between; gap:10px; padding:4px 0;
     border-bottom:1px solid var(--line-2); font-size:13px}
.abs:last-child{border-bottom:none}
.abs .nm{white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.abs.doute{opacity:.62}
.abs.doute .nm::before{content:"? "; color:var(--ink-3)}
@media (prefers-reduced-motion:reduce){summary::after{transition:none}}
@media (max-width:760px){
  .mhead{grid-template-columns:1fr; gap:10px}
  .mteams{font-size:20px}
}
"""


def build_slate_report(
    matchs: list,
    titre: str = "Journée",
    standalone: bool = True,
) -> str:
    """
    Vue journee : plusieurs matchs, plusieurs sports, sur une seule page.

    Chaque entree de `matchs` :
        home, away, marches, sport, competition, date (str), sd, scorers

    La carte resume tout ce qui se lit d'un coup d'oeil -- vainqueur, score
    attendu, deux marches cles -- et le detail complet vit dans un <details>
    replie. Le resume est autonome : on n'a jamais besoin d'ouvrir le detail
    pour savoir de quoi parle le match.
    """
    cartes, sports = [], []
    for i, m in enumerate(matchs):
        sport = m.get("sport", "football")
        marches = m["marches"]
        home, away = m["home"], m["away"]
        if sport not in sports:
            sports.append(sport)

        xg = marches.get("buts_attendus") or marches.get("points_attendus") or {}
        xh, xa = xg.get("domicile", 0), xg.get("exterieur", 0)
        unite = UNITES.get(sport, "points")
        # Sous chaque nom, l'esperance de l'equipe ; la legende au centre.
        # Rien quand le sport n'a pas d'esperance par camp (tennis).
        xg_row = (f'<span class="xg xh">{xh:.2f}</span><span class="xl">{unite} attendus</span>'
                  f'<span class="xg xa">{xa:.2f}</span>') if xg else ""

        # deux marches cles, choisis selon le sport
        keys = []
        tot = (marches.get("total_buts") or marches.get("total_points")
               or marches.get("total_jeux") or [])
        if tot:
            milieu = tot[len(tot) // 2]
            po = milieu.get("over", milieu.get("p_over", 0))
            keys.append((f"Plus de {milieu['ligne']}", po))
        if "btts" in marches:
            keys.append(("Les deux marquent", marches["btts"]["oui"]))
        if "total_essais" in marches and marches["total_essais"]:
            e = marches["total_essais"][len(marches["total_essais"]) // 2]
            keys.append((f"Plus de {e['ligne']} essais", e.get("over", 0)))
        if "score_en_sets" in marches and marches["score_en_sets"]:
            s0 = marches["score_en_sets"][0]
            keys.append((f"Score {s0['score']}", s0["p"]))
        if "scores_exacts" in marches and marches["scores_exacts"]:
            s0 = marches["scores_exacts"][0]
            keys.append((f"Score {s0['score']}", s0["p"]))
        buteurs = [b for b in (marches.get("buteurs") or []) if b.get("poste") != "collectif"]
        if buteurs:
            b0 = buteurs[0]
            keys.append((f"{b0['joueur']} marque", b0["p_marque"]))

        keys_html = "".join(
            f"<span>{_esc(lab)} <b>{_pct(p)}</b> <span class='role'>{_fo(p)}</span></span>"
            for lab, p in keys[:4]
        )

        # Score en direct ou final, quand il a ete rattache a la rencontre.
        live = m.get("live") or {}
        badge = ""
        if live.get("statut") in ("en_cours", "termine"):
            hs, as_ = live.get("home_score"), live.get("away_score")
            score = f"{hs} – {as_}" if hs is not None and as_ is not None else "—"
            if live["statut"] == "en_cours":
                mn = live.get("minutes")
                minute = f" {int(mn)}’" if mn is not None else ""
                badge = (f'<span class="statut live"><span class="pt"></span>'
                         f'EN COURS{minute} <b>{score}</b></span>')
            else:
                # Verdict : l'issue la plus probable du modele s'est-elle
                # realisee ? C'est le retour le plus direct sur sa qualite.
                v = ""
                if hs is not None and as_ is not None and tri:
                    reel = "1" if hs > as_ else ("X" if hs == as_ else "2")
                    prevu = max(tri, key=tri.get)
                    ok = reel == prevu
                    v = (f'<span class="verdict {"ok" if ok else "ko"}">'
                         f'{"prevu" if ok else "rate"}</span>')
                badge = f'<span class="statut">TERMINE <b>{score}</b></span>{v}'
        detail = "".join(_panels(home, away, marches, m.get("sd"), m.get("scorers"), sport))
        cherche = f"{home} {away} {m.get('competition','')} {sport}".lower()

        # Axes de tri. L'HEURE est le seul axe qui traverse reellement les
        # sports : "qu'est-ce qui se joue ce soir" est une question qui a du
        # sens, "compare ce tennis a ce football" n'en a pas.
        # Libelle de date : une plage quand la journee couvre deux jours.
        # Afficher "17/09" pour un match qui se joue le 18 serait faux.
        d_fin = (m.get("date_fin") or "").strip()
        d_deb = (m.get("date") or "").strip()
        if d_fin and d_fin != d_deb and len(d_deb) == 10 and len(d_fin) == 10:
            libelle_date = f"{d_deb[:5]}–{d_fin[:5]}/{d_fin[6:]}"
        else:
            libelle_date = d_deb

        # Tri a la MINUTE, pas au jour. Avec une cle limitee a AAAAMMJJ, tous
        # les matchs d'une meme journee se retrouvaient departages par ordre
        # alphabetique -- un 21h avant un 13h.
        # Les coups d'envoi sont stockes en UTC (indispensable pour calculer
        # le temps ecoule en direct) mais s'AFFICHENT en heure de Paris.
        # Sans conversion, un match de 19h s'affichait a 17h en ete : deux
        # heures d'avance sur tout le calendrier.
        ts = 0
        heure_txt = ""
        ke = _paris(m.get("coup_envoi"))
        if ke is not None:
            try:
                ts = int(ke.strftime("%Y%m%d%H%M"))
                heure_txt = ke.strftime("%H:%M")
            except (AttributeError, ValueError):
                ts = 0
        d = d_deb
        if not ts and d:
            try:
                j, mo, an = d.split("/")
                ts = int(an) * 100000000 + int(mo) * 1000000 + int(j) * 10000
            except ValueError:
                ts = 0
        tri = marches.get("1x2") or marches.get("vainqueur") or marches.get("moneyline") or {}
        net = max(tri.values()) if tri else 0.0

        cartes.append(f"""
<article class="match" data-sport="{_esc(sport)}" data-comp="{_esc(m.get('competition',''))}" data-q="{_esc(cherche)}" data-ts="{ts}" data-net="{net:.4f}" data-day="{str(ts)[:8]}">
  <div class="mhead">
    <div class="mwhen"><b>{_esc(sport)}</b>{_esc(libelle_date)}{f'<b class="h">{heure_txt}</b>' if heure_txt else ''}<br>{_esc(m.get('competition',''))}</div>
    <div>
      <div class="mteams"><span class="h">{_esc(home)}</span><span class="sep">—</span><span class="a">{_esc(away)}</span>{xg_row}</div>
    </div>
    {_bars(marches, home, away, "mbars")}
  </div>
  <div class="mkeys">{badge}{keys_html}</div>
  <details><summary>Tous les marchés</summary><div class="grid">{detail}</div><button class="fermer" type="button">Fermer les marchés ▲</button></details>
</article>""")

    # Onglets a SELECTION UNIQUE, pas des cases a cocher multiples.
    # Cocher les quatre sports revient exactement a n'en cocher aucun : un
    # controle qui offre deux chemins vers le meme etat n'apprend rien et fait
    # douter. On regarde un sport a la fois, ou tout.
    counts = {}
    for m in matchs:
        counts[m.get("sport", "football")] = counts.get(m.get("sport", "football"), 0) + 1
    onglets = [('<button class="chip" data-f="" aria-pressed="true">Tous '
                f'<span class="n">{len(matchs)}</span></button>')]
    for s in sports:
        onglets.append(
            f'<button class="chip" data-f="{_esc(s)}" aria-pressed="false">{_esc(s)} '
            f'<span class="n">{counts[s]}</span></button>'
        )
    chips = "".join(onglets)

    # Competitions par sport, pour le second filtre. Les options sont
    # reconstruites en JS a chaque changement de sport : un <select> ne sait
    # pas cacher une option sur tous les navigateurs mobiles.
    comps = {}
    for m in matchs:
        k = (m.get("sport", "football"), m.get("competition", ""))
        comps[k] = comps.get(k, 0) + 1
    comps_json = json.dumps(
        [{"sport": sp, "nom": nom, "n": n} for (sp, nom), n in sorted(comps.items())],
        ensure_ascii=False)

    js = """
(function(){
  var chips=[].slice.call(document.querySelectorAll('.chip[data-f]'));
  var q=document.getElementById('q'), tri=document.getElementById('tri');
  var comp=document.getElementById('comp');
  var COMPS=__COMPS__;
  function sportActif(){
    var a=chips.filter(function(c){return c.getAttribute('aria-pressed')==='true'})[0];
    return a?a.dataset.f:'';
  }
  // Options du filtre competition : celles du sport choisi, avec effectifs.
  function remplirComps(){
    var f=sportActif(), courant=comp.value;
    comp.innerHTML='';
    var o=document.createElement('option'); o.value=''; o.textContent='Toutes les compétitions';
    comp.appendChild(o);
    COMPS.forEach(function(c){
      if(f!==''&&c.sport!==f) return;
      var x=document.createElement('option'); x.value=c.nom;
      x.textContent=(f===''?c.sport+' · ':'')+c.nom+' ('+c.n+')';
      comp.appendChild(x);
    });
    comp.value=courant;
    if(comp.value!==courant) comp.value='';
  }
  var slate=document.querySelector('.slate');
  var cards=[].slice.call(document.querySelectorAll('.match'));
  var cnt=document.getElementById('cnt'), vide=document.getElementById('vide');
  var JOURS=['dimanche','lundi','mardi','mercredi','jeudi','vendredi','samedi'];
  var MOIS=['janvier','février','mars','avril','mai','juin','juillet','août',
            'septembre','octobre','novembre','décembre'];
  var auj=new Date(); var ajStr=''+auj.getFullYear()+
      ('0'+(auj.getMonth()+1)).slice(-2)+('0'+auj.getDate()).slice(-2);

  function libelle(ts){
    if(!ts||ts==='0') return 'date inconnue';
    var y=+ts.slice(0,4), mo=+ts.slice(4,6), d=+ts.slice(6,8);
    var dt=new Date(y,mo-1,d);
    var lab=JOURS[dt.getDay()]+' '+d+' '+MOIS[mo-1];
    if(ts===ajStr) lab="aujourd'hui — "+lab;
    return lab;
  }
  function purgerSeparateurs(){
    [].slice.call(slate.querySelectorAll('.day-sep')).forEach(function(x){x.remove();});
  }
  function separer(){
    purgerSeparateurs();
    if(tri.value!=='date') return;
    var jour=null;
    [].slice.call(slate.querySelectorAll('.match')).forEach(function(c){
      if(c.hidden) return;
      var ts=(c.dataset.day)||'0';
      if(ts===jour) return;
      jour=ts;
      var sep=document.createElement('div');
      sep.className='day-sep'+(ts===ajStr?' today':'');
      var n=cards.filter(function(x){return !x.hidden && (x.dataset.day||'0')===ts;}).length;
      sep.innerHTML='<span>'+libelle(ts)+'</span><span class="n">'+n+'</span>';
      slate.insertBefore(sep,c);
    });
  }
  function apply(){
    var f=sportActif(), k=comp.value||'';
    var t=(q.value||'').trim().toLowerCase(); var n=0;
    cards.forEach(function(c){
      var ok=(f===''||c.dataset.sport===f)&&(k===''||c.dataset.comp===k)
             &&(t===''||c.dataset.q.indexOf(t)>-1);
      c.hidden=!ok; if(ok)n++;
    });
    cnt.textContent=n+' match'+(n>1?'s':'');
    vide.hidden=n>0;
    separer();
  }
  function trier(){
    var mode=tri.value;
    purgerSeparateurs();
    var l=cards.slice().sort(function(a,b){
      if(mode==='net') return (+b.dataset.net)-(+a.dataset.net);
      var d=(+a.dataset.ts)-(+b.dataset.ts);
      return d!==0?d:a.dataset.q.localeCompare(b.dataset.q);
    });
    l.forEach(function(c){slate.appendChild(c);});
    separer();
  }
  chips.forEach(function(c){c.addEventListener('click',function(){
    chips.forEach(function(o){o.setAttribute('aria-pressed','false');});
    c.setAttribute('aria-pressed','true'); remplirComps(); apply();
  });});
  comp.addEventListener('change', apply);
  // Refermer depuis le BAS du panneau : une fois les marches deplies, le
  // resume qui sert d'interrupteur est souvent sorti de l'ecran, et rien
  // n'indique comment refermer.
  [].slice.call(document.querySelectorAll('.fermer')).forEach(function(b){
    b.addEventListener('click', function(){
      var d=b.closest('details'); if(!d) return;
      d.open=false;
      var c=d.closest('.match'); if(c) c.scrollIntoView({block:'nearest'});
    });
  });
  q.addEventListener('input', apply);
  tri.addEventListener('change', trier);
  remplirComps(); trier(); apply();
})();
""".replace("__COMPS__", comps_json)
    corps = f"""
<div class="wrap">
  <div class="eyebrow">{_esc(titre)}</div>
  <div class="filters">
    {chips}
    <label class="tri"><select id="comp" aria-label="Filtrer par compétition"></select></label>
    <input id="q" type="search" placeholder="Filtrer par équipe…" aria-label="Filtrer">
    <label class="tri"><span>Trier</span>
      <select id="tri" aria-label="Trier les matchs">
        <option value="date">par heure de match</option>
        <option value="net">par pronostic le plus net</option>
      </select>
    </label>
    <span class="count" id="cnt"></span>
  </div>
  <div class="slate">{''.join(cartes)}</div>
  <div class="empty" id="vide" hidden>Aucun match ne correspond à ce filtre.</div>
  <div class="foot">
    <span class="tag">généré le {_paris(datetime.now(timezone.utc)):%d/%m/%Y à %H:%M}</span>
    <span>{len(matchs)} match(s) — probabilités issues des données, aucune cote n'entre dans le calcul.</span>
    <span>« Cote éq. » = 1/probabilité.</span>
  </div>
</div>
<script>{js}</script>"""

    tete = f"<title>{_esc(titre)}</title>{_FONTS}<style>{_CSS}{_CSS_SLATE}</style>"
    if not standalone:
        return tete + corps
    return (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{tete}</head><body>{corps}</body></html>"
    )


def write_report(path, **kwargs) -> str:
    from pathlib import Path

    p = Path(path)
    p.write_text(build_report(**kwargs), encoding="utf-8")
    return str(p.resolve())


def write_slate(path, matchs, titre="Journée") -> str:
    from pathlib import Path

    p = Path(path)
    p.write_text(build_slate_report(matchs, titre), encoding="utf-8")
    return str(p.resolve())
