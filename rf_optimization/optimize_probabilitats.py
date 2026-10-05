"""
Optimització del main PROBABILÍSTIC (main_probabilitats.py).

A diferència de optimize_rf.py, aquí la sortida del model són PROBABILITATS,
així que la mètrica principal és el LOG LOSS (cross-entropy): penalitza estar
molt segur i fallar, i premia probabilitats ben calibrades. Com més BAIX, millor.
També es reporten accuracy i F1 macro de referència.

Mateix split que main_probabilitats.py:
  - Train: totes les altres lligues + La Liga fins a l'última temporada (exclosa)
  - Test:  última temporada de La Liga

Fase 1 — Decay EWM:  cerca el millor decay del rolling exponencial (K fix).
Fase 2 — Tuning RF:  amb el millor decay, RandomizedSearchCV (scoring=neg_log_loss).
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("", "entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d) if _d else _ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features import build_features, load_all_leagues, load_la_liga, FEATURES

TEST_LEAGUE = "La liga"
K_ELO       = 16          # ELO fix durant l'optimització
CLASSES     = [0, 1, 2]

DECAY_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

RF_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500, 700],
    "max_depth":         [4, 6, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
}

# Es carreguen els CSV una sola vegada; build_features es recalcula per cada decay
_df_all    = load_all_leagues()
_df_laliga = load_la_liga()


def load_split(decay, K=K_ELO):
    """Reprodueix el split de main_probabilitats.py per a un decay/K donats."""
    df_laliga_feat = build_features(_df_laliga, K=K, roll_n=30, decay=decay)
    df_other       = _df_all[_df_all["league"] != TEST_LEAGUE].copy()
    df_other_feat  = build_features(df_other, K=K, roll_n=30, decay=decay)

    seasons     = sorted(df_laliga_feat["season"].unique())
    test_season = seasons[-1]

    train_ll = df_laliga_feat[df_laliga_feat["season"] != test_season]
    train    = pd.concat([df_other_feat, train_ll], ignore_index=True).dropna(subset=FEATURES)
    test     = df_laliga_feat[df_laliga_feat["season"] == test_season].dropna(subset=FEATURES)

    return train[FEATURES], train["result"], test[FEATURES], test["result"]


def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    pred  = model.predict(X_test)
    ll       = log_loss(y_test, proba, labels=CLASSES)
    acc      = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro")
    return ll, acc, f1_macro


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Optimitzar decay (mètrica: log loss, més baix millor)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 58)
print("  FASE 1 — Optimitzant decay EWM (probabilitats)")
print(f"  Valors provats: {DECAY_VALUES}")
print("=" * 58)

base_rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42
)

print(f"\n{'Decay':>7} | {'LogLoss':>8} | {'Accuracy':>9} | {'F1 macro':>9}")
print("-" * 44)

decay_results = []
for decay in DECAY_VALUES:
    X_train, y_train, X_test, y_test = load_split(decay)
    ll, acc, f1m = evaluate(base_rf, X_train, y_train, X_test, y_test)
    decay_results.append((decay, ll, acc, f1m))
    print(f"{decay:>7.2f} | {ll:>8.4f} | {acc*100:>8.2f}% | {f1m*100:>8.2f}%")

best_decay_row = min(decay_results, key=lambda x: x[1])   # menor log loss
best_decay = best_decay_row[0]
print(f"\nMillor decay (log loss): {best_decay}  ->  "
      f"logloss={best_decay_row[1]:.4f}  acc={best_decay_row[2]*100:.2f}%  f1_macro={best_decay_row[3]*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Tuning hiperparàmetres RF amb el millor decay (scoring=neg_log_loss)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print(f"  FASE 2 — Tuning RF  (decay={best_decay}, scoring=neg_log_loss)")
print("=" * 58)

X_train, y_train, X_test, y_test = load_split(best_decay)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=RF_PARAM_DIST,
    n_iter=40,
    scoring="neg_log_loss",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

print("\n=== Millors hiperparàmetres ===")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n  Log loss CV (train): {-search.best_score_:.4f}")

best = search.best_estimator_
proba = best.predict_proba(X_test)
pred  = best.predict(X_test)
print(f"  Log loss test:       {log_loss(y_test, proba, labels=CLASSES):.4f}")
print(f"  Accuracy test:       {accuracy_score(y_test, pred)*100:.2f}%")
print(f"  F1 macro test:       {f1_score(y_test, pred, average='macro')*100:.2f}%")

print("\n=== RESUM FINAL — copia això a make_model() / decay de main_probabilitats.py ===")
print(f"  decay = {best_decay}")
for k, v in search.best_params_.items():
    print(f"  {k} = {v}")
