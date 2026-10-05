"""
Crea data/jornades_recomanacions.xlsx (el fitxer de "memòria") per primer
cop. Cal córrer aquest script UNA SOLA VEGADA abans de fer servir
actualitza_jornada.py.

Sembra la memòria amb el resultat del backtest ja validat d'estrategia.py
(mateixa lògica, mateixes SEASONS):
  - Full "Apostes": totes les apostes històriques del backtest (Value Local
    / Contrarian Visitant), amb el seu resultat ja conegut.
  - Full "Confianca": la confiança per equip acumulada al final del
    backtest -- és el punt de partida real per a les recomanacions en viu,
    no es comença de zero.
  - Full "Resum": bankroll_actual = BANKROLL_INICIAL (cada temporada
    comença amb el mateix pot, mateix criteri que estrategia.py -- vegeu
    el seu docstring; el bankroll NO es compon entre temporades, només la
    confiança per equip ho fa).
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import os

import pandas as pd

from estrategia import SEASONS, SEASON_CONF_DECAY, TEAM_CONF_MIN, TEAM_CONF_MAX, BANKROLL_INICIAL, simular
from memoria import MEMORIA_PATH, escriu_memoria, existeix


def inicialitza(path: str = MEMORIA_PATH, force: bool = False) -> None:
    if existeix(path) and not force:
        raise FileExistsError(f"{path} ja existeix. Passa force=True si vols refer-lo de zero.")

    print(f"Corrent el backtest d'estrategia.py ({', '.join(SEASONS)}) per sembrar la memòria...")
    team_confidence: dict = {}
    resultats = []
    for i, season in enumerate(SEASONS):
        if i > 0:
            for equip in team_confidence:
                team_confidence[equip] = min(max(team_confidence[equip] * SEASON_CONF_DECAY, TEAM_CONF_MIN), TEAM_CONF_MAX)
        resultats.append(simular(season, team_confidence))

    apostes_hist = pd.concat(resultats, ignore_index=True) if resultats else pd.DataFrame()
    if not apostes_hist.empty:
        apostes_hist["confirmada"] = True   # backtest ja validat, resultat ja conegut

    escriu_memoria(apostes_hist, team_confidence, BANKROLL_INICIAL, path)
    print(f"\nMemòria creada a {path}")
    print(f"  Apostes històriques sembrades : {len(apostes_hist)}")
    print(f"  Equips amb confiança calculada: {len(team_confidence)}")
    print(f"  Bankroll de partida (en viu)  : {BANKROLL_INICIAL:.2f} EUR")


if __name__ == "__main__":
    inicialitza()
