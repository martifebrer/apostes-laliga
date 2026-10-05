"""
Optimització en dues fases per al Random Forest:

Fase 1 — Decay EWM:
  Cerca el millor decay per al rolling exponencial.
  Assumim que decay in (0, 1]: com més baix, més pes als partits recents.
  Es fixa roll_n=30 i els hiperparàmetres del RF base.

Fase 2 — Hiperparàmetres RF:
  Amb el millor decay trobat a la fase 1, fa un RandomizedSearchCV
  sobre n_estimators, max_depth, min_samples_split, min_samples_leaf i max_features.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("", "entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d) if _d else _ROOT)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features import build_features, load_matches, FEATURES

LEAGUE   = "La liga"
ROLL_N   = 30
K_ELO    = 16

DECAY_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

RF_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500, 700],
    "max_depth":         [4, 6, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
}


def load_split(decay):
    df_raw = load_matches()
    df = df_raw[df_raw["league"] == LEAGUE].copy()
    df = build_features(df, K=K_ELO, roll_n=ROLL_N, decay=decay)
    last_season = df["season"].max()
    train = df[df["season"] != last_season].dropna(subset=FEATURES)
    test  = df[df["season"] == last_season].dropna(subset=FEATURES)
    return train[FEATURES], train["result"], test[FEATURES], test["result"]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Optimitzar decay
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  FASE 1 — Optimitzant decay EWM")
print(f"  Valors provats: {DECAY_VALUES}")
print("=" * 55)

base_rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42
)

print(f"\n{'Decay':>7} | {'Accuracy':>9} | {'F1 draw':>8} | {'F1 macro':>9}")
print("-" * 42)

decay_results = []
for decay in DECAY_VALUES:
    X_train, y_train, X_test, y_test = load_split(decay)
    base_rf.fit(X_train, y_train)
    pred = base_rf.predict(X_test)
    acc      = accuracy_score(y_test, pred)
    f1_draw  = f1_score(y_test, pred, labels=[1], average="macro")
    f1_macro = f1_score(y_test, pred, average="macro")
    decay_results.append((decay, acc, f1_draw, f1_macro))
    print(f"{decay:>7.2f} | {acc*100:>8.2f}% | {f1_draw*100:>7.2f}% | {f1_macro*100:>8.2f}%")

best_decay_row = max(decay_results, key=lambda x: x[3])
best_decay = best_decay_row[0]
print(f"\nMillor decay (F1 macro): {best_decay}  ->  accuracy={best_decay_row[1]*100:.2f}%  f1_macro={best_decay_row[3]*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Tuning hiperparàmetres RF amb el millor decay
# ══════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 55)
# print(f"  FASE 2 — Tuning RF  (decay={best_decay})")
# print("=" * 55)
#
# X_train, y_train, X_test, y_test = load_split(best_decay)
#
# search = RandomizedSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_distributions=RF_PARAM_DIST,
#     n_iter=60,
#     scoring="f1_macro",
#     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
#     random_state=42,
#     n_jobs=-1,
#     verbose=2,
# )
#
# search.fit(X_train, y_train)
#
# print("\n=== Millors hiperparàmetres ===")
# for k, v in search.best_params_.items():
#     print(f"  {k}: {v}")
# print(f"\n  F1 macro CV (train): {search.best_score_:.4f}")
#
# pred = search.best_estimator_.predict(X_test)
# print(f"  Accuracy test:       {accuracy_score(y_test, pred)*100:.2f}%")
# print(f"  F1 draw test:        {f1_score(y_test, pred, labels=[1], average='macro')*100:.2f}%")
# print(f"  F1 macro test:       {f1_score(y_test, pred, average='macro')*100:.2f}%")
#
# print("\n=== RESUM FINAL ===")
# print(f"  decay       = {best_decay}")
# for k, v in search.best_params_.items():
#     print(f"  {k} = {v}")
