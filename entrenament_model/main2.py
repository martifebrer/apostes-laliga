"""
Variant del main amb les features de forma DUPLICADES (dues finestres):
  - meitat curta:  roll_n=5,  decay=0.8   (forma recent, reacciona ràpid)
  - meitat llarga: roll_n=25, decay=0.95  (forma de fons, més estable)

El model rep totes dues versions de cada feature rolling alhora (les curtes
amb sufix _s, les llargues amb el nom original), més l'ELO (compartit).

Treu per pantalla les mètriques del model sobre la temporada de test:
accuracy, precision/recall/F1 per classe, log loss i RPS.

El report HTML agrupat per jornades es genera amb report_jornades.py.
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
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from features import build_features, load_all_leagues, load_la_liga, FEATURES, TEST_LEAGUE
from model import make_model

LABELS = {0: "Local", 1: "Empat", 2: "Visitant"}
INV    = {v: k for k, v in LABELS.items()}
K_ELO  = 25     # ELO (a main.py és 100)

# Dues finestres de forma: curta (recent) i llarga (de fons)
ROLL_SHORT, DECAY_SHORT = 5,  0.8
ROLL_LONG,  DECAY_LONG  = 25, 0.95

# L'ELO depèn només de K (no de roll_n/decay), així que NO es duplica.
ELO_FEATURES  = ["home_elo", "away_elo", "elo_diff"]
ROLL_FEATURES = [f for f in FEATURES if f not in ELO_FEATURES]
# Features finals: rolling llargues (nom original) + rolling curtes (_s) + ELO
FEATURES2     = ROLL_FEATURES + [c + "_s" for c in ROLL_FEATURES] + ELO_FEATURES


def build_dual_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Construeix les features dues vegades (finestra curta i llarga) i les
    combina en un sol DataFrame. Les llargues mantenen el nom original; les
    curtes reben el sufix _s. L'ELO es pren del build llarg (és el mateix)."""
    feat_l = build_features(df_raw, K=K_ELO, roll_n=ROLL_LONG,  decay=DECAY_LONG)
    feat_s = build_features(df_raw, K=K_ELO, roll_n=ROLL_SHORT, decay=DECAY_SHORT)
    # Mateixa entrada → mateix ordre de files; alineació posicional segura
    for c in ROLL_FEATURES:
        feat_l[c + "_s"] = feat_s[c].to_numpy()
    return feat_l


def get_probabilities(test_season: str = None):
    """
    Entrena amb totes les lligues + La Liga fins a test_season (exclosa)
    i retorna un DataFrame amb la probabilitat de cada opció per cada
    partit de la temporada de test, més la predicció i si s'ha encertat.
    """
    df_all    = load_all_leagues()
    df_laliga = load_la_liga()

    df_laliga_feat = build_dual_features(df_laliga)
    seasons = sorted(df_laliga_feat["season"].unique())
    if test_season is None:
        test_season = seasons[-1]

    df_other      = df_all[df_all["league"] != TEST_LEAGUE].copy()
    df_other_feat = build_dual_features(df_other)

    train_ll = df_laliga_feat[df_laliga_feat["season"] != test_season]
    train    = pd.concat([df_other_feat, train_ll], ignore_index=True).dropna(subset=FEATURES2)
    test     = df_laliga_feat[df_laliga_feat["season"] == test_season].dropna(subset=FEATURES2).copy()

    model = make_model()
    model.fit(train[FEATURES2], train["result"])

    proba = model.predict_proba(test[FEATURES2])   # shape: (n_partits, 3)

    # model.classes_ diu l'ordre de les columnes de proba (p.ex. [0, 1, 2])
    for idx, cls in enumerate(model.classes_):
        test[f"prob_{LABELS[cls]}"] = (proba[:, idx] * 100).round(1)

    test["prediccio"]     = [LABELS[model.classes_[row.argmax()]] for row in proba]
    test["resultat_real"] = test["result"].map(LABELS)
    test["correct"]       = (test["prediccio"] == test["resultat_real"]).astype(int)
    test["date"]          = pd.to_datetime(test["date"])
    test["home_elo"]      = test["home_elo"].round().astype(int)
    test["away_elo"]      = test["away_elo"].round().astype(int)

    cols = ["date", "home_team", "away_team", "home_elo", "away_elo",
            "prob_Local", "prob_Empat", "prob_Visitant",
            "prediccio", "resultat_real", "correct"]
    return test[cols], test_season


def rps(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Ranked Probability Score mitjà (resultats ordinals Local<Empat<Visitant)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p[:, :2] - cum_o[:, :2]) ** 2, axis=1) / 2.0))


def print_metrics(df: pd.DataFrame, season: str) -> None:
    """Treu per pantalla accuracy, precision/recall/F1 per classe, log loss i RPS."""
    y    = df["resultat_real"].map(INV).to_numpy()
    pred = df["prediccio"].map(INV).to_numpy()

    P = df[["prob_Local", "prob_Empat", "prob_Visitant"]].to_numpy(float) / 100
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    acc = (pred == y).mean() * 100
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0)
    ll = log_loss(y, P, labels=[0, 1, 2])
    r  = rps(P, y)

    print(f"\n=== Mètriques — La Liga {season}  ({len(df)} partits) ===\n")
    print(f"Accuracy : {acc:6.2f}%")
    print(f"Log loss : {ll:6.4f}")
    print(f"RPS      : {r:6.4f}\n")

    print(f"{'Classe':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Suport':>6}")
    print("-" * 50)
    for i in range(3):
        print(f"{LABELS[i]:>9} | {prec[i]*100:>8.2f}% | {rec[i]*100:>6.2f}% | "
              f"{f1[i]*100:>5.2f}% | {int(sup[i]):>6}")

    # ── Matriu de confusió (files = real, columnes = predit) ──────────────────
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    print(f"\nMatriu de confusió (files = real, columnes = predit):")
    etiq = "real \\\\ pred"   # fora de l'f-string: els backslashs dins d'f-strings peten amb Python < 3.12
    print(f"{etiq:>11} | {'Local':>7} | {'Empat':>7} | {'Visit':>7}")
    print("-" * 44)
    for i in range(3):
        print(f"{LABELS[i]:>11} | {cm[i,0]:>7} | {cm[i,1]:>7} | {cm[i,2]:>7}")


if __name__ == "__main__":
    print("Carregant dades i entrenant el model...")
    df, season = get_probabilities()

    print_metrics(df, season)
    
    # Report HTML per jornades (lògica separada a report_jornades.py)
    from report_jornades import generate_report
    out = generate_report(
        df, season,
        out_path=os.path.join(_ROOT, "prob_report_k50.html"),
    )
    print(f"\nHTML generat: {out}")
