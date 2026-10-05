"""
Carrega les variables d'entorn de l'arxiu .env de l'arrel del projecte (si
existeix), sense dependències externes. No sobreescriu variables ja definides
a l'entorn. Les claus mai han d'estar escrites al codi -- vegeu .env.example.
"""

import os

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _carrega_env(path: str = _ENV_PATH) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia or linia.startswith("#") or "=" not in linia:
                continue
            clau, valor = linia.split("=", 1)
            os.environ.setdefault(clau.strip(), valor.strip().strip('"').strip("'"))


_carrega_env()
