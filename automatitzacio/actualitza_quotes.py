"""
Pas 1 del flux setmanal: baixa les quotes d'una jornada concreta de La Liga
amb The Odds API (mateixa API que hack_apostes/obtenir_quotes_live.py) i
les afegeix a data/quotes_laliga.csv, en un format compatible amb el que
ja fa servir estrategia.py (football-data.co.uk: Div/Date/Time/HomeTeam/
AwayTeam + una columna <CODI>H/D/A per casa).

La jornada a buscar la tria l'usuari (actualitza_jornada.py li ho pregunta)
i es fa servir el calendari OFICIAL (automatitzacio/calendari.py,
data/calendari_2026_27.csv) per saber exactament quins partits en formen
part -- es filtren els partits que retorna l'API per aquest emparellament,
en comptes de suposar que "la propera jornada" és la que ve abans d'una
finestra de dies des del primer partit pendent.

Bet365, William Hill i 888sport (comprovat en TOTES les regions de l'API:
eu/uk/us/us2/au) NO hi són -- aquestes cases no llicencien les seves
quotes a agregadors. Es fa servir Pinnacle, Betfair Sportsbook, Ladbrokes,
Bet Victor i Betway en el seu lloc (codis PIN/BFD/LAD/BV/BWY), i "Avg"
es calcula com la mitjana de TOTES les cases que retorni l'API (no només
aquestes 5).
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

import numpy as np
import pandas as pd
import requests

import config  # noqa: F401  (carrega .env abans de llegir ODDS_API_KEY)
from calendari import CALENDARI_PATH, partits_jornada
from equips import odds_api_a_model, model_a_sp1

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUOTES_PATH = os.path.join(BASE_DIR, "data", "quotes_laliga.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "la_liga_2014_2025_all_matches_final.csv")

API_KEY  = os.environ.get("ODDS_API_KEY")   # obligatòria -- vegeu .env.example
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_spain_la_liga"

# Odds API bookmaker key -> codi de columna a l'estil football-data.co.uk
LIVE_BOOKMAKERS = {
    "pinnacle":      "PIN",
    "betfair_sb_uk": "BFD",   # casa principal (vegeu recomana_viva.LIVE_CASES/FALLBACK_ORDER)
    "ladbrokes_uk":  "LAD",
    "betvictor":     "BV",
    "betway":        "BWY",   # reserva quan Betfair no torna quota (vegeu recomana_viva.py)
}
LIVE_CODES = list(LIVE_BOOKMAKERS.values())


def _get_la_liga_odds() -> list[dict]:
    if not API_KEY:
        raise RuntimeError("Falta la variable d'entorn ODDS_API_KEY (clau de https://the-odds-api.com). "
                           "Vegeu .env.example.")
    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": API_KEY, "regions": "eu,uk", "markets": "h2h",
        "oddsFormat": "decimal", "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    print(f"[{SPORT_KEY}] Credits usats: {resp.headers.get('x-requests-used')} | "
          f"Restants: {resp.headers.get('x-requests-remaining')}")
    return resp.json()


def _partits_de_la_jornada(matches: list[dict], jornada: int, noms_model: list[str]) -> list[dict]:
    """Filtra els partits que retorna l'API per quedar-se NOMÉS amb els de
    `jornada`, segons el calendari oficial (calendari.partits_jornada)."""
    cal = partits_jornada(jornada)
    if cal.empty:
        print(f"Avís: la jornada {jornada} no és a {os.path.basename(CALENDARI_PATH)}.")
        return []
    claus = set(zip(cal["home_team"], cal["away_team"]))

    seleccionats = []
    for m in matches:
        home_model = odds_api_a_model(m["home_team"], noms_model)
        away_model = odds_api_a_model(m["away_team"], noms_model)
        if (home_model, away_model) in claus:
            seleccionats.append(m)
    return seleccionats


def _fila_partit(match: dict, noms_model: list[str]) -> dict:
    home_model = odds_api_a_model(match["home_team"], noms_model)
    away_model = odds_api_a_model(match["away_team"], noms_model)
    dt = pd.to_datetime(match["commence_time"]).tz_convert(None) if pd.to_datetime(match["commence_time"]).tzinfo \
        else pd.to_datetime(match["commence_time"])

    fila = {
        "Div": "SP1",
        "Date": dt.strftime("%d/%m/%Y"),
        "Time": dt.strftime("%H:%M"),
        "HomeTeam": model_a_sp1(home_model),
        "AwayTeam": model_a_sp1(away_model),
        "_home_model": home_model,
        "_away_model": away_model,
        "_date": dt.normalize(),
    }

    totes_h, totes_d, totes_a = [], [], []
    for bk in match.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            oh = outcomes.get(match["home_team"])
            oa = outcomes.get(match["away_team"])
            od = outcomes.get("Draw")
            if oh is None or oa is None or od is None:
                continue
            totes_h.append(oh); totes_d.append(od); totes_a.append(oa)

            codi = LIVE_BOOKMAKERS.get(bk["key"])
            if codi:
                fila[f"{codi}H"], fila[f"{codi}D"], fila[f"{codi}A"] = oh, od, oa

    if totes_h:
        fila["AvgH"] = round(float(np.mean(totes_h)), 2)
        fila["AvgD"] = round(float(np.mean(totes_d)), 2)
        fila["AvgA"] = round(float(np.mean(totes_a)), 2)

    return fila


def _dedup_mateix_partit(df: pd.DataFrame) -> pd.DataFrame:
    """Mentre l'API encara no té tancada la data oficial d'un partit, el pot
    llistar dues vegades amb dates/hores candidates diferents (mateixos
    equips) -- típic de jornades encara llunyanes. Només ens interessa una
    fila per partit real, així que ens quedem amb la que tingui més cobertura
    de cases d'apostes (sol ser la data que ja s'ha consolidat entre cases;
    l'altra acostuma a portar només 'Avg' o cap quota)."""
    meta_cols = {"Div", "Date", "Time", "HomeTeam", "AwayTeam",
                 "_date", "_home_model", "_away_model"}
    odds_cols = [c for c in df.columns if c not in meta_cols]
    df = df.copy()
    df["_cobertura"] = df[odds_cols].notna().sum(axis=1)
    df = (df.sort_values("_cobertura", ascending=False)
            .drop_duplicates(subset=["_home_model", "_away_model"], keep="first")
            .drop(columns="_cobertura"))
    return df


def actualitza_quotes(jornada: int, quotes_path: str = QUOTES_PATH, history_path: str = HISTORY_PATH) -> pd.DataFrame:
    """Baixa les quotes dels partits de `jornada` (segons el calendari
    oficial, vegeu calendari.py), les afegeix/actualitza a quotes_path
    (upsert per Date+HomeTeam+AwayTeam) i retorna un DataFrame de fixtures
    (date, home_team, away_team, amb els NOMS DEL MODEL) MÉS les columnes
    de quotes (PINH/D/A, BFDH/D/A, LADH/D/A, BVH/D/A, BWYH/D/A, AvgH/D/A),
    per a recomana_viva.py."""
    hist = pd.read_csv(history_path)
    noms_model = sorted(set(hist["home_team"]) | set(hist["away_team"]))

    matches = _partits_de_la_jornada(_get_la_liga_odds(), jornada, noms_model)
    if not matches:
        print(f"Encara no hi ha quotes disponibles a l'API per als partits de la jornada {jornada}.")
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    files = [_fila_partit(m, noms_model) for m in matches]
    df_nou = _dedup_mateix_partit(pd.DataFrame(files))

    # Versió amb noms del model + quotes, per a la predicció/recomanació (no es desa a CSV)
    fixtures_odds = df_nou.rename(columns={"_date": "date", "_home_model": "home_team", "_away_model": "away_team"}) \
        .drop(columns=["Div", "Time", "HomeTeam", "AwayTeam"])

    df_nou = df_nou.drop(columns=["_date", "_home_model", "_away_model"])

    if os.path.exists(quotes_path):
        df_actual = pd.read_csv(quotes_path, encoding="utf-8-sig")
        clau_nova = set(zip(df_nou["Date"], df_nou["HomeTeam"], df_nou["AwayTeam"]))
        ja_hi_es = df_actual.apply(lambda r: (r["Date"], r["HomeTeam"], r["AwayTeam"]) in clau_nova, axis=1)
        df_log = pd.concat([df_actual[~ja_hi_es], df_nou], ignore_index=True)
    else:
        df_log = df_nou

    df_log.to_csv(quotes_path, index=False, encoding="utf-8-sig")
    print(f"quotes_laliga.csv actualitzat: {len(df_nou)} partit(s) de la jornada {jornada} "
          f"({', '.join(LIVE_CODES)} + Avg on hi hagi dades).")
    return fixtures_odds.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import sys as _sys
    _jornada = int(_sys.argv[1]) if len(_sys.argv) > 1 else int(input("Quina jornada? "))
    fx = actualitza_quotes(_jornada)
    print(fx.to_string(index=False))
