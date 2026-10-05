"""
Calendari OFICIAL de partits de La Liga 2026-27 (data/calendari_2026_27.csv,
emparellaments publicats abans de començar la temporada). Permet saber de
manera FIABLE a quina jornada pertany un partit concret, en comptes de
l'estimació precària "partits_jugats // 10 + 1" (jornada_estimada() a
actualitza_historic.py, que ara només queda com a referència).
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import pandas as pd

from actualitza_historic import temporada_actual_understat
from equips import calendari_a_model

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALENDARI_PATH = os.path.join(BASE_DIR, "data", "calendari_2026_27.csv")
HISTORY_PATH   = os.path.join(BASE_DIR, "data", "la_liga_2014_2025_all_matches_final.csv")


def _season_mapped(season: str) -> str:
    return f"{season}-{str(int(season) + 1)[-2:]}"


def carrega_calendari(path: str = CALENDARI_PATH, history_path: str = HISTORY_PATH) -> pd.DataFrame:
    """Calendari amb els noms d'equip ja traduïts als del model."""
    cal = pd.read_csv(path, encoding="utf-8-sig")
    hist = pd.read_csv(history_path)
    noms_model = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    cal = cal.copy()
    cal["home_team"] = cal["home_team"].map(lambda n: calendari_a_model(n, noms_model))
    cal["away_team"] = cal["away_team"].map(lambda n: calendari_a_model(n, noms_model))
    return cal


def partits_jornada(jornada: int, path: str = CALENDARI_PATH, history_path: str = HISTORY_PATH) -> pd.DataFrame:
    """Emparellaments (home_team, away_team, noms del model) d'una jornada
    concreta segons el calendari oficial."""
    cal = carrega_calendari(path, history_path)
    return cal.loc[cal["jornada"] == jornada, ["home_team", "away_team"]].reset_index(drop=True)


def jornada_de_partit(home_team: str, away_team: str, path: str = CALENDARI_PATH,
                       history_path: str = HISTORY_PATH) -> int | None:
    """Jornada a què pertany un partit concret (noms del model), o None si
    no és al calendari."""
    cal = carrega_calendari(path, history_path)
    fila = cal[(cal["home_team"] == home_team) & (cal["away_team"] == away_team)]
    return int(fila.iloc[0]["jornada"]) if not fila.empty else None


def jornada_actual_suggerida(history_path: str = HISTORY_PATH, calendari_path: str = CALENDARI_PATH,
                              season: str | None = None) -> int:
    """Suggereix la propera jornada a jugar-se: la primera (per ordre) que
    encara NO té TOTS els seus partits a l'històric de resultats jugats.
    Fiable perquè es basa en el calendari OFICIAL (publicat abans de
    començar la temporada) en comptes de comptar partits jugats -- només un
    suggeriment per defecte, l'usuari sempre pot triar una altra jornada."""
    season = season or temporada_actual_understat()
    season_mapped = _season_mapped(season)

    hist = pd.read_csv(history_path)
    jugats = hist[hist["season_mapped"] == season_mapped]
    claus_jugats = set(zip(jugats["home_team"], jugats["away_team"]))

    cal = carrega_calendari(calendari_path, history_path)
    for jornada in sorted(cal["jornada"].unique()):
        partits = cal[cal["jornada"] == jornada]
        claus = set(zip(partits["home_team"], partits["away_team"]))
        if not claus <= claus_jugats:
            return int(jornada)
    return int(cal["jornada"].max()) + 1
