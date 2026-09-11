"""
Cycle de rafraichissement, concu pour le PLAN GRATUIT des deux APIs.

    py refresh.py complet     # calendrier 48 h, modeles, absences, rugby
    py refresh.py live        # scores des matchs en cours uniquement

BUDGET, ET POURQUOI DEUX CADENCES
----------------------------------
The Odds API : 500 credits/mois, soit ~13/jour. API-Football : 100/jour.

Le cycle COMPLET tire le calendrier proche d'API-Football (2 requetes pour
toutes les competitions sur 48 h) au lieu de The Odds API (1 credit PAR
championnat). Il ne coute donc qu'UN credit Odds API par jour -- pour le
calendrier NRL -- et 4 requetes API-Football. Une fois par jour suffit : les
calendriers bougent peu et les blessures se confirment la veille.

Le cycle LIVE n'interroge que les competitions ayant un match en cours d'apres
les heures de coup d'envoi deja connues, et seulement parmi les championnats
suivis (cinq grands + Ligue 2 + NRL). Il ne coute RIEN quand rien ne se joue.
Un samedi charge : environ 5 credits par passage. A 45 minutes d'intervalle
sur les creneaux de match, on tient dans le budget ; a 10 minutes, non.

Le fragment HTML pour la page publiee est ecrit dans `artifact.html` : c'est
lui que la tache planifiee republie.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "journee.html"
SLATE = ROOT / "journee.slate"
FRAGMENT = ROOT / "artifact.html"
TITRE = "20 championnats + rugby"

# Les cles viennent EXCLUSIVEMENT de l'environnement. Jamais en dur ici :
# ce fichier est versionne, et un depot -- meme prive -- n'est pas un
# coffre-fort. Sur GitHub Actions elles arrivent par les Secrets du depot ;
# en local, par les variables d'environnement ou un fichier .env non suivi.
REQUISES = ("ODDS_API_KEY", "API_FOOTBALL_KEY")


def _charger_env_local() -> None:
    """Lit un eventuel fichier .env (ignore par git) pour l'usage local."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for ligne in f.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        k, v = ligne.split("=", 1)
        # Le .env PRIME sur l'environnement. Cas reel : une variable
        # ODDS_API_KEY=ta_cle_ici trainait dans le registre utilisateur
        # (un placeholder pose par `setx` en suivant un exemple), et un
        # setdefault la laissait gagner sur la vraie cle du .env. Resultat :
        # 401 sur l'API et une journee amputee, sans autre message.
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _verifier_cles() -> None:
    suspectes = [k for k in REQUISES
                 if os.environ.get(k, "").lower() in ("ta_cle_ici", "ta_cle", "xxx", "changeme")]
    if suspectes:
        print(f"[!] placeholder au lieu d'une vraie cle : {', '.join(suspectes)}", flush=True)
        sys.exit(2)
    manquantes = [k for k in REQUISES if not os.environ.get(k)]
    if manquantes:
        print(f"[!] variables manquantes : {', '.join(manquantes)}", flush=True)
        print("    en local : cree un fichier .env (voir .env.example)", flush=True)
        print("    sur GitHub : Settings > Secrets and variables > Actions", flush=True)
        sys.exit(2)


def run(*args) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "sportvalue", *args]
    print(f"$ {' '.join(args)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
    print(tail, flush=True)
    # Toute sortie d'erreur est affichee, quel que soit le code retour.
    # Traiter le code 1 comme "normal" a masque un NameError pendant deux
    # cycles complets : la journee sortait amputee de tout le football sans
    # la moindre trace. Un code 1 legitime ("aucun match a venir") n'ecrit
    # rien sur stderr, donc l'afficher ne genere aucun bruit dans ce cas.
    err = (r.stderr or "").strip()
    if err:
        print("--- stderr ---", flush=True)
        print(err[-1500:], flush=True)
    if "Traceback" in err:
        print(f"[!] ECHEC de la sous-commande (code {r.returncode})", flush=True)
    return r.returncode


DOCS = ROOT / "docs"


def compacter(slate: list, n: int = 3000) -> list:
    """
    Reduit la journee a ce dont le live et la page ont besoin.

    Un match de rugby ou de basket porte 60 000 tirages Monte-Carlo : la
    journee entiere pesait 87 Mo, impossible a faire transiter entre deux
    executions. Les marches sont deja calcules dans `marches` ; la
    distribution ne sert plus qu'au recalcul en direct (esperances et taux
    d'evenements) et a la heatmap du football (matrice 13x13, deja petite).
    On sous-echantillonne donc a 3 000 tirages : suffisant pour un recalcul
    conditionnel, 20 fois plus leger.
    """
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from sportvalue.core.scoredist import ScoreDistribution

    rng = np.random.default_rng(0)
    for m in slate:
        sd = m.get("sd")
        if sd is None or len(sd.w) <= n or m.get("sport") == "football":
            continue
        idx = rng.choice(len(sd.w), size=n, replace=True, p=sd.w)
        petit = ScoreDistribution(sd.h[idx], sd.a[idx], np.ones(n))
        for attr in ("event_rates", "weather_notes", "expected_tries"):
            if hasattr(sd, attr):
                setattr(petit, attr, getattr(sd, attr))
        if hasattr(sd, "tries"):
            t = sd.tries
            j = rng.choice(len(t.w), size=n, replace=True, p=t.w)
            petit.tries = ScoreDistribution(t.h[j], t.a[j], np.ones(n))
        m["sd"] = petit
        m["scorers"] = None   # tableaux deja rendus dans marches["marqueurs_essais"]
    return slate


def fragment() -> None:
    sys.path.insert(0, str(ROOT))
    from sportvalue.report import build_slate_report

    slate = compacter(pickle.loads(SLATE.read_bytes()))
    SLATE.write_bytes(pickle.dumps(slate))
    frag = build_slate_report(slate, titre=TITRE, standalone=False)
    FRAGMENT.write_text(frag, encoding="utf-8")
    # Page complete pour GitHub Pages. Le fragment (sans doctype) est destine
    # a l'Artifact claude.ai ; Pages sert un document HTML entier.
    DOCS.mkdir(exist_ok=True)
    page = build_slate_report(slate, titre=TITRE, standalone=True)
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    # La journee (journee.slate) ne va PAS dans docs/ : trop lourde pour git.
    # Sur GitHub Actions elle transite par le cache d'un run a l'autre.
    n_live = sum(1 for m in slate if (m.get("live") or {}).get("statut") == "en_cours")
    print(f"fragment : {len(slate)} matchs, {n_live} en cours, "
          f"{len(frag)/1024:.0f} Ko -> {FRAGMENT.name}", flush=True)


def complet() -> None:
    print(f"=== CYCLE COMPLET {datetime.now():%d/%m %H:%M} ===", flush=True)
    for f in (SLATE, HTML):
        if f.exists():
            f.unlink()
    # Rugby d'abord : Top 14 et Pro D2 ne coutent rien, la NRL 1 credit.
    for sport in ("nrl", "top14", "prod2"):
        run("journee", "--sport", sport, "--html", str(HTML), "--ajouter",
            "--titre", TITRE)
    # Football sur 48 h via API-Football : 0 credit Odds API.
    run("journee", "--sport", "football", "--leagues", "top15,d2",
        "--source", "apifootball", "--buteurs", "--absences",
        "--html", str(HTML), "--ajouter", "--titre", TITRE)
    fragment()


def live() -> None:
    print(f"=== LIVE {datetime.now():%d/%m %H:%M} ===", flush=True)
    if not SLATE.exists():
        print("pas de journee : lancer d'abord `refresh.py complet`", flush=True)
        return
    run("live", "--html", str(HTML), "--titre", TITRE, "--leagues", "top5,F2")
    fragment()


if __name__ == "__main__":
    _charger_env_local()
    _verifier_cles()
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    {"complet": complet, "live": live}.get(mode, live)()
