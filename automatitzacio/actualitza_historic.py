"""
Pas 2 del flux setmanal: baixa d'Understat els partits de La Liga ja
JUGATS que encara no són a l'històric (data/la_liga_2014_2025_all_matches_final.csv)
i els hi afegeix, amb les mateixes columnes/definicions amb què es va
construir el fitxer originalment (Understat: xg, ppda, deep, shots, sot,
i xpts derivat del forecast w/d/l de cada partit).

Dos passos per partit:
  1. u.league("La_Liga").get_match_data(season=...) -- llista de partits
     de la temporada (resultat, xG, forecast w/d/l), per saber quins
     match_id ja s'han jugat i encara no són a l'històric.
  2. Per cada match_id nou, es fa scraping de https://understat.com/match/<id>
     (variable JS "match_info", JSON escapat amb \\xNN) per treure
     shots/shotsOnTarget/deep/ppda -- dades que NOMÉS són a la pàgina del
     partit, no a la llista de la lliga.

xPts (no ve directe d'Understat): xpts_local = 3*w + d, xpts_visitant = 3*l + d,
amb w/d/l = forecast del MATEIX match_info (comprovat que coincideix
exactament amb els valors ja calculats a l'històric existent).
"""

import os
import re
import time

import pandas as pd
import requests
from understatapi import UnderstatClient

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "data", "la_liga_2014_2025_all_matches_final.csv")

HIST_COLS = ["match_id", "date", "home_team", "away_team", "home_goals", "away_goals",
             "home_xg", "away_xg", "home_shots", "away_shots", "home_sot", "away_sot",
             "home_deep", "away_deep", "home_ppda", "away_ppda", "home_xpts", "away_xpts",
             "season_mapped"]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ProjecteApostesPersonal/1.0)"}
MATCH_INFO_RE = re.compile(r"var match_info\s*=\s*JSON\.parse\('(.+?)'\)")


def jornada_estimada(history_path: str = HISTORY_PATH, season: str | None = None) -> int:
    """ESTIMACIÓ PRECÀRIA (partits_jugats // 10 + 1) que actualitza_jornada.py
    JA NO fa servir: es va escriure perquè data/jornades.csv (el calendari
    OFICIAL que fa servir estrategia.py) no cobreix cap temporada en curs,
    ja que es construeix a partir de resultats ja jugats. Ara hi ha
    automatitzacio/calendari.py (calendari_2026_27.csv, l'emparellament
    OFICIAL publicat abans de començar la temporada), fiable de veritat --
    vegeu calendari.jornada_actual_suggerida(). Es deixa aquí només de
    referència/fallback."""
    season = season or temporada_actual_understat()
    season_mapped = f"{season}-{str(int(season) + 1)[-2:]}"
    hist = pd.read_csv(history_path)
    jugats = int((hist["season_mapped"] == season_mapped).sum())
    return jugats // 10 + 1


def temporada_actual_understat() -> str:
    """Understat identifica cada temporada amb l'any en què COMENÇA
    (p.ex. '2025' per 2025-26). La Liga sol començar a mitjans d'agost,
    així que a partir de juliol ja es considera oberta la temporada nova."""
    now = pd.Timestamp.now()
    return str(now.year if now.month >= 7 else now.year - 1)


def _match_info(match_id: str, session: requests.Session) -> dict:
    resp = session.get(f"https://understat.com/match/{match_id}", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    m = MATCH_INFO_RE.search(resp.text)
    if not m:
        raise ValueError(f"No s'ha trobat match_info a la pàgina del partit {match_id}")
    return __import__("json").loads(m.group(1).encode().decode("unicode_escape"))


def actualitza_historic(history_path: str = HISTORY_PATH, season: str | None = None) -> pd.DataFrame:
    season = season or temporada_actual_understat()
    hist = pd.read_csv(history_path)
    ids_coneguts = set(hist["match_id"].astype(int))

    with UnderstatClient() as u:
        partits_temporada = u.league(league="La_Liga").get_match_data(season=season)

    nous_ids = [m["id"] for m in partits_temporada
                if m["isResult"] and int(m["id"]) not in ids_coneguts]

    if not nous_ids:
        print(f"Cap partit jugat nou a la temporada {season} (Understat) que ja no fos a l'històric.")
        return pd.DataFrame(columns=HIST_COLS)

    print(f"Trobats {len(nous_ids)} partit(s) jugats nous a Understat (temporada {season}): baixant estadístiques...")

    season_mapped = f"{season}-{str(int(season) + 1)[-2:]}"
    files = []
    with requests.Session() as session:
        for mid in nous_ids:
            info = _match_info(mid, session)
            w, d, l = float(info["h_w"]), float(info["h_d"]), float(info["h_l"])
            files.append({
                "match_id":   int(info["id"]),
                "date":       info["date"],
                "home_team":  info["team_h"],
                "away_team":  info["team_a"],
                "home_goals": int(info["h_goals"]),
                "away_goals": int(info["a_goals"]),
                "home_xg":    round(float(info["h_xg"]), 2),
                "away_xg":    round(float(info["a_xg"]), 2),
                "home_shots": int(info["h_shot"]),
                "away_shots": int(info["a_shot"]),
                "home_sot":   int(info["h_shotOnTarget"]),
                "away_sot":   int(info["a_shotOnTarget"]),
                "home_deep":  int(info["h_deep"]),
                "away_deep":  int(info["a_deep"]),
                "home_ppda":  round(float(info["h_ppda"]), 2),
                "away_ppda":  round(float(info["a_ppda"]), 2),
                "home_xpts":  round(3 * w + d, 2),
                "away_xpts":  round(3 * l + d, 2),
                "season_mapped": season_mapped,
            })
            time.sleep(0.4)   # no martiritzem el servidor d'Understat

    df_nou = pd.DataFrame(files)[HIST_COLS]
    combinat = pd.concat([hist, df_nou], ignore_index=True)
    combinat.to_csv(history_path, index=False)
    print(f"Històric actualitzat: {len(df_nou)} partit(s) afegits a {history_path}")
    return df_nou


if __name__ == "__main__":
    df = actualitza_historic()
    if not df.empty:
        print(df[["date", "home_team", "away_team", "home_goals", "away_goals"]].to_string(index=False))
