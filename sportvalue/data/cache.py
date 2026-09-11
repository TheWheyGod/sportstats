"""
Cache disque pour les appels reseau.

Deux raisons, l'une pratique l'autre critique :
  - les APIs de cotes ont des quotas serres (souvent 500 requetes/mois en
    gratuit) : re-telecharger a chaque essai les epuise en une journee ;
  - un backtest doit etre REPRODUCTIBLE. Si les donnees changent entre deux
    executions, on ne sait plus si une amelioration vient du modele ou du jeu
    de donnees.

Les reponses de cotes en direct utilisent un TTL court (defaut 5 min), les
donnees historiques un TTL long.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["Cache", "default_cache"]


class Cache:
    def __init__(self, root: str | Path = "data_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.root / f"{h}{suffix}"

    # ---- texte / csv ---------------------------------------------------
    def get_text(self, key: str, ttl: float, loader: Callable[[], str]) -> str:
        p = self._path(key, ".txt")
        meta = self._path(key, ".meta")
        if p.exists() and meta.exists():
            try:
                age = time.time() - json.loads(meta.read_text())["ts"]
                if age < ttl:
                    return p.read_text(encoding="utf-8", errors="replace")
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        data = loader()
        p.write_text(data, encoding="utf-8")
        meta.write_text(json.dumps({"ts": time.time(), "key": key}))
        return data

    # ---- json ----------------------------------------------------------
    def get_json(self, key: str, ttl: float, loader: Callable[[], Any]) -> Any:
        p = self._path(key, ".json")
        meta = self._path(key, ".meta")
        if p.exists() and meta.exists():
            try:
                age = time.time() - json.loads(meta.read_text())["ts"]
                if age < ttl:
                    return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        data = loader()
        p.write_text(json.dumps(data), encoding="utf-8")
        meta.write_text(json.dumps({"ts": time.time(), "key": key}))
        return data

    def has(self, key: str, suffix: str = ".txt") -> bool:
        return self._path(key, suffix).exists()

    def clear(self) -> int:
        n = 0
        for f in self.root.glob("*"):
            if f.is_file():
                f.unlink()
                n += 1
        return n

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.root.glob("*") if f.is_file()) / 1e6


default_cache: Optional[Cache] = None


def get_cache(root: str | Path = "data_cache") -> Cache:
    global default_cache
    if default_cache is None:
        default_cache = Cache(root)
    return default_cache
