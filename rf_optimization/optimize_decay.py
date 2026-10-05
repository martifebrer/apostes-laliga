"""
Optimitzador RÀPID dels hiperparàmetres del model dual (entrenament.py):
  - decay de la finestra curta  (roll_n=5,  forma recent)
  - decay de la finestra llarga (roll_n=25, forma de fons)
  - K de l'ELO

Cerca SEQÜENCIAL (coordinate descent) en comptes de graella completa: per
cada hiperparàmetre es prova una graella petita deixant els altres dos
fixos al millor valor trobat fins ara. Suma de graelles en lloc de
producte -> molt més ràpid que una 3D completa.

Per ser ràpid:
  - Per cada configuració provada, les features es calculen UN SOL COP
    (no un cop per temporada de test).
  - S'avalua la MITJANA de les últimes N_TEST_SEASONS temporades de La
    Liga (per no sobreajustar-se als peculiaritats d'una sola, com la
    24-25), reaprofitant el mateix DataFrame de features ja calculat --
    només es repeteix el fit (lleuger) + predict per cada temporada.
  - Cada temporada de test s'entrena només amb l'històric ANTERIOR a
    ella (walk-forward real, sense fuita entre els 3 folds de test).
  - Model lleuger (poques arbres) només per a la cerca; el de producció
    és a model.py.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("", "entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d) if _d else _ROOT)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from features import build_features, load_all_leagues, load_la_liga, TEST_LEAGUE
import entrenament as m1

ROLL_SHORT, ROLL_LONG = m1.ROLL_SHORT, m1.ROLL_LONG   # 5, 25 (fixos, no s'optimitzen)
ROLL_FEATURES = m1.ROLL_FEATURES
FEATURES2     = m1.FEATURES2
CLASSES       = [0, 1, 2]

N_TEST_SEASONS = 3   # mitjana de les últimes 3 temporades, no només l'última

# Valors de partida de la cerca i graelles. Ull: decay_short parteix de 0.75,
# però el valor actual de producció (entrenament.py/main2.py) és 0.8.
BASELINE = {"decay_short": 0.75, "decay_long": 0.95, "K": m1.K_ELO}

DECAY_SHORT_GRID = [0.6, 0.7, 0.75, 0.8, 0.9]
DECAY_LONG_GRID  = [0.85, 0.9, 0.95, 0.98]
K_GRID           = [15, 20, 25, 30, 40]


def quick_model():
    return RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)


# CSVs carregats un sol cop
df_all    = load_all_leagues()
df_laliga = load_la_liga()
df_other  = df_all[df_all["league"] != TEST_LEAGUE].copy()


def _build_dual(df_raw: pd.DataFrame, K: float, decay_short: float, decay_long: float) -> pd.DataFrame:
    feat_l = build_features(df_raw, K=K, roll_n=ROLL_LONG,  decay=decay_long)
    feat_s = build_features(df_raw, K=K, roll_n=ROLL_SHORT, decay=decay_short)
    for c in ROLL_FEATURES:
        feat_l[c + "_s"] = feat_s[c].to_numpy()
    return feat_l


def evaluate(K: float, decay_short: float, decay_long: float) -> tuple[float, float]:
    """Mitjana de log loss/accuracy de les últimes N_TEST_SEASONS temporades
    de La Liga. Les features es calculen un sol cop per aquesta configuració;
    cada temporada de test s'entrena només amb l'històric anterior a ella."""
    laliga_feat = _build_dual(df_laliga, K, decay_short, decay_long)
    other_feat  = _build_dual(df_other,  K, decay_short, decay_long).dropna(subset=FEATURES2)

    seasons      = sorted(laliga_feat["season"].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]

    lls, accs = [], []
    for test_season in test_seasons:
        i = seasons.index(test_season)
        train_ll = laliga_feat[laliga_feat["season"].isin(seasons[:i])]
        train = pd.concat([other_feat, train_ll], ignore_index=True).dropna(subset=FEATURES2)
        test  = laliga_feat[laliga_feat["season"] == test_season].dropna(subset=FEATURES2)

        model = quick_model()
        model.fit(train[FEATURES2], train["result"])
        proba = model.predict_proba(test[FEATURES2])
        pred  = model.predict(test[FEATURES2])

        lls.append(log_loss(test["result"], proba, labels=CLASSES))
        accs.append(accuracy_score(test["result"], pred))

    return sum(lls) / len(lls), sum(accs) / len(accs)


def _sweep(nom: str, grid: list, fixed: dict, key: str) -> float:
    """Prova `grid` de valors per a `key`, deixant la resta fixos a `fixed`.
    Retorna el valor guanyador (menor log loss mitjà)."""
    print("=" * 46, flush=True)
    print(f"  {nom}  (fixos: {', '.join(f'{k}={v}' for k, v in fixed.items() if k != key)})", flush=True)
    print(f"{'Valor':>7} | {'LogLoss':>8} | {'Accuracy':>9}", flush=True)
    print("-" * 32, flush=True)

    resultats = []
    for valor in grid:
        params = {**fixed, key: valor}
        ll, acc = evaluate(params["K"], params["decay_short"], params["decay_long"])
        resultats.append((valor, ll, acc))
        print(f"{valor:>7} | {ll:>8.4f} | {acc*100:>8.2f}%", flush=True)

    millor_valor, millor_ll, millor_acc = min(resultats, key=lambda r: r[1])
    print(f"  >> millor {key}: {millor_valor}  (logloss mitjà={millor_ll:.4f}, "
          f"accuracy mitjana={millor_acc*100:.2f}%)\n", flush=True)
    return millor_valor


if __name__ == "__main__":
    print(f"Cerca seqüencial (decay curt -> decay llarg -> K), "
          f"mitjana de les últimes {N_TEST_SEASONS} temporades de La Liga\n", flush=True)

    params = dict(BASELINE)

    params["decay_short"] = _sweep(
        f"DECAY finestra curta (roll_n={ROLL_SHORT})", DECAY_SHORT_GRID, params, "decay_short")

    params["decay_long"] = _sweep(
        f"DECAY finestra llarga (roll_n={ROLL_LONG})", DECAY_LONG_GRID, params, "decay_long")

    params["K"] = _sweep(
        "K de l'ELO", K_GRID, params, "K")

    print("=" * 46, flush=True)
    print("  RESUM: millors hiperparàmetres trobats", flush=True)
    print("=" * 46, flush=True)
    print(f"  decay_short (roll_n={ROLL_SHORT}) : {params['decay_short']}")
    print(f"  decay_long  (roll_n={ROLL_LONG}) : {params['decay_long']}")
    print(f"  K (ELO)                          : {params['K']}")

    ll_final, acc_final = evaluate(params["K"], params["decay_short"], params["decay_long"])
    print(f"\n  LogLoss mitjà final  : {ll_final:.4f}")
    print(f"  Accuracy mitjana final: {acc_final*100:.2f}%")
