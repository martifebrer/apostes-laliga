"""
Prediu, amb el model ja entrenat (model_rf_eval.joblib, generat per
entrenament.py -- el mateix que es fa servir a les simulacions d'estrategia.py,
per tenir sempre la mateixa recepta validada en viu i en backtest), les
probabilitats (Local/Empat/Visitant) dels propers partits de La Liga.

Dos fitxers d'entrada:
  - HISTORY_PATH  : l'històric de partits JUGATS de La Liga, mateix format
                     que data/la_liga_2014_2025_all_matches_final.csv.
                     La idea és que hi vagis afegint files a mesura que es
                     juguen jornades noves.
  - FIXTURES_PATH : un CSV petit (data/next_fixtures.csv) amb els partits
                     encara PENDENTS de jugar: columnes date, home_team,
                     away_team. Normalment només la propera jornada.

Com es calculen les features del partit pendent: s'afegeixen les files
dels fixtures (sense resultat, sense estadístiques) al final de
l'històric i es corre el mateix pipeline de features que a l'entrenament
(build_dual_features, amb ELO + rolling curt/llarg). Així cada fixture
rep l'ELO/rolling "tal com estan avui", calculats amb tot l'històric
real que ja s'ha jugat.

Nota: pensat per predir UNA jornada pendent alhora (cada equip hi apareix
com a màxim un cop). Si el fitxer de fixtures inclou diverses jornades
futures per al mateix equip, la segona en endavant NO es calcula bé: els
gols del fixture anterior són NaN i les comparacions amb NaN donen False,
de manera que l'ELO s'actualitza com si el local hagués PERDUT aquell
partit pendent (no queda congelat), i les rolling reben un NaN que fa que
el partit posterior surti amb features incompletes i es descarti.
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from features import STAT_COLS
from entrenament import build_dual_features, LABELS, MODEL_EVAL_PATH

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH   = os.path.join(BASE_DIR, "data", "la_liga_2014_2025_all_matches_final.csv")
FIXTURES_PATH  = os.path.join(BASE_DIR, "data", "next_fixtures.csv")

COLS = ["date", "league", "season", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg",
        "home_ppda", "away_ppda", "home_deep", "away_deep",
        "home_xpts", "away_xpts"]


def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"season_mapped": "season"})
    df["league"] = "La liga"
    df["date"] = pd.to_datetime(df["date"])
    return df[COLS]


def load_fixtures(path: str) -> pd.DataFrame:
    fx = pd.read_csv(path)
    missing = {"date", "home_team", "away_team"} - set(fx.columns)
    if missing:
        raise ValueError(f"Falten columnes a {path}: {missing}")
    fx["date"] = pd.to_datetime(fx["date"])

    for s in STAT_COLS:
        fx[f"home_{s}"] = np.nan
        fx[f"away_{s}"] = np.nan
    fx["home_goals"] = np.nan
    fx["away_goals"] = np.nan
    fx["league"] = "La liga"
    fx["season"] = "PENDENT"
    return fx[COLS]


def predict(history_path: str = HISTORY_PATH, fixtures_path: str = FIXTURES_PATH) -> pd.DataFrame:
    if not os.path.exists(MODEL_EVAL_PATH):
        raise FileNotFoundError(
            f"No s'ha trobat el model a {MODEL_EVAL_PATH}. Executa entrenament.py primer per entrenar-lo i desar-lo."
        )
    bundle = joblib.load(MODEL_EVAL_PATH)
    model, features = bundle["model"], bundle["features"]

    hist     = load_history(history_path)
    fixtures = load_fixtures(fixtures_path)
    if fixtures.empty:
        raise ValueError(f"No hi ha cap partit pendent a {fixtures_path}")

    combined = pd.concat([hist, fixtures], ignore_index=True)
    feat = build_dual_features(combined)

    fixture_keys = fixtures[["date", "home_team", "away_team"]].copy()
    fixture_keys["date"] = pd.to_datetime(fixture_keys["date"])
    pred_rows = feat.merge(fixture_keys, on=["date", "home_team", "away_team"], how="inner").copy()

    if len(pred_rows) != len(fixtures):
        trobats = set(zip(pred_rows["date"], pred_rows["home_team"], pred_rows["away_team"]))
        esperats = set(zip(fixture_keys["date"], fixture_keys["home_team"], fixture_keys["away_team"]))
        print(f"Avís: no s'han pogut casar {len(esperats - trobats)} partit(s) de {fixtures_path}: "
              f"{esperats - trobats}")

    missing_feat = pred_rows[features].isna().any(axis=1)
    if missing_feat.any():
        equips = pred_rows.loc[missing_feat, ["home_team", "away_team"]]
        print(f"Avís: {missing_feat.sum()} partit(s) amb features incompletes (equip sense prou "
              f"històric?), es descarten:\n{equips}")
        pred_rows = pred_rows[~missing_feat]

    proba = model.predict_proba(pred_rows[features])
    for idx, cls in enumerate(model.classes_):
        pred_rows[f"prob_{LABELS[cls]}"] = (proba[:, idx] * 100).round(1)
    pred_rows["prediccio"] = [LABELS[model.classes_[row.argmax()]] for row in proba]
    pred_rows["home_elo"]  = pred_rows["home_elo"].round().astype(int)
    pred_rows["away_elo"]  = pred_rows["away_elo"].round().astype(int)

    out_cols = ["date", "home_team", "away_team", "home_elo", "away_elo",
                "prob_Local", "prob_Empat", "prob_Visitant", "prediccio"]
    return pred_rows[out_cols].sort_values("date").reset_index(drop=True)


def print_predictions(df: pd.DataFrame) -> None:
    print(f"\n=== Prediccions ({len(df)} partit(s)) ===\n")
    for _, r in df.iterrows():
        data = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        print(f"{data}  {r['home_team']} vs {r['away_team']}  "
              f"(ELO {r['home_elo']} - {r['away_elo']})")
        print(f"   Local {r['prob_Local']:5.1f}%   Empat {r['prob_Empat']:5.1f}%   "
              f"Visitant {r['prob_Visitant']:5.1f}%   -> {r['prediccio']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=HISTORY_PATH, help="CSV amb l'històric de partits jugats")
    parser.add_argument("--fixtures", default=FIXTURES_PATH, help="CSV amb els partits pendents (date, home_team, away_team)")
    args = parser.parse_args()

    preds = predict(args.history, args.fixtures)
    print_predictions(preds)
