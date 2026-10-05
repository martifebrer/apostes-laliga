"""
Walk-forward backtest de La Liga amb finestra mòbil de temporades.

Esquema:
  - A cada fold s'entrena amb WINDOW temporades consecutives de La Liga
    + TOT l'històric de la resta de lligues (Ligue 1, EPL, Bundesliga,
    Serie A), i es testeja amb la temporada immediatament posterior.
  - La finestra llisca una temporada cada fold (s'oblida la més antiga,
    s'hi afegeix la que abans era de test).
  - Cada fold entrena un model NOU (un RandomForest no té memòria entre
    folds), però les features (ELO + rolling curt/llarg) es calculen UN
    SOL COP sobre tot l'històric complet i cronològic -- mai es
    reinicien entre folds. A cada fold només es seleccionen quines files
    (per temporada) van a train i quines a test.

Features de forma DUPLICADES (dues finestres, com a main2.py):
  - meitat curta:  roll_n=5,  decay=0.8  (forma recent)
  - meitat llarga: roll_n=25, decay=0.95  (forma de fons)

Genera un HTML (walkforward_report.html) amb el resum per fold i
l'agregat de tot el backtest.
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

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from features import build_features, load_all_leagues, load_la_liga, FEATURES, TEST_LEAGUE
from model import make_model

LABELS = {0: "Local", 1: "Empat", 2: "Visitant"}
INV    = {v: k for k, v in LABELS.items()}
K_ELO  = 25     # ELO

# Dues finestres de forma: curta (recent) i llarga (de fons)
ROLL_SHORT, DECAY_SHORT = 5,  0.8
ROLL_LONG,  DECAY_LONG  = 25, 0.95

# L'ELO depèn només de K (no de roll_n/decay), així que NO es duplica.
ELO_FEATURES  = ["home_elo", "away_elo", "elo_diff"]
ROLL_FEATURES = [f for f in FEATURES if f not in ELO_FEATURES]
# Features finals: rolling llargues (nom original) + rolling curtes (_s) + ELO
FEATURES2     = ROLL_FEATURES + [c + "_s" for c in ROLL_FEATURES] + ELO_FEATURES

WINDOW = 7  # temporades de La Liga a cada finestra d'entrenament

MODEL_PATH      = os.path.join(_ROOT, "model_rf.joblib")
MODEL_EVAL_PATH = os.path.join(_ROOT, "model_rf_eval.joblib")


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


def get_fold_probabilities() -> list[dict]:
    """
    Walk-forward sobre les temporades de La Liga amb finestra mòbil de
    WINDOW temporades.

    Les features es calculen UN SOL COP sobre tot l'històric cronològic
    (La Liga sencera i la resta de lligues senceres), de manera que
    l'ELO i les rolling mai es reinicien entre folds: a cada fold només
    es filtren les files ja calculades per temporada.

    Retorna una llista de dicts, un per fold:
      {"train_seasons": [...], "test_season": str, "df": DataFrame}
    """
    df_all    = load_all_leagues()
    df_laliga = load_la_liga()

    # Features calculades un sol cop sobre tot l'històric (mai es reinicien)
    df_laliga_feat = build_dual_features(df_laliga)
    df_other       = df_all[df_all["league"] != TEST_LEAGUE].copy()
    df_other_feat  = build_dual_features(df_other).dropna(subset=FEATURES2)

    seasons = sorted(df_laliga_feat["season"].unique())

    folds = []
    for i in range(len(seasons) - WINDOW):
        train_seasons = seasons[i: i + WINDOW]
        test_season   = seasons[i + WINDOW]

        train_ll = df_laliga_feat[df_laliga_feat["season"].isin(train_seasons)]
        train = pd.concat([df_other_feat, train_ll], ignore_index=True).dropna(subset=FEATURES2)
        test  = df_laliga_feat[df_laliga_feat["season"] == test_season].dropna(subset=FEATURES2).copy()

        # Model nou a cada fold (el RandomForest no té memòria entre folds)
        model = make_model()
        model.fit(train[FEATURES2], train["result"])

        proba = model.predict_proba(test[FEATURES2])   # shape: (n_partits, 3)
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
        folds.append({
            "train_seasons": train_seasons,
            "test_season":   test_season,
            "df":            test[cols].reset_index(drop=True),
        })

    return folds


def train_and_save_final_models() -> None:
    """Entrena i guarda DOS models amb joblib:

    - MODEL_PATH (producció): últimes WINDOW temporades de La Liga,
      incloent-hi la més recent + tot l'històric de la resta de lligues.
      És el que fa servir predict_next.py per predir partits futurs.
    - MODEL_EVAL_PATH (avaluació): mateix esquema però EXCLOENT les TRES
      últimes temporades (2023-24, 2024-25 i 2025-26). La 2024-25 es va
      fer servir per optimitzar hiperparàmetres (decay, K), així que
      treure-la també del train evita qualsevol biaix indirecte de la
      selecció d'hiperparàmetres a més del leakage directe. És el que fa
      servir comparar_cases.py / sense_estrategia.py / estrategia.py per
      avaluar les temporades 2023-24 a 2025-26 sense data leakage.
    """
    df_all    = load_all_leagues()
    df_laliga = load_la_liga()

    df_laliga_feat = build_dual_features(df_laliga)
    df_other       = df_all[df_all["league"] != TEST_LEAGUE].copy()
    df_other_feat  = build_dual_features(df_other).dropna(subset=FEATURES2)

    seasons = sorted(df_laliga_feat["season"].unique())

    def _fit_and_dump(train_seasons: list, path: str, nom: str) -> None:
        train_ll = df_laliga_feat[df_laliga_feat["season"].isin(train_seasons)]
        train = pd.concat([df_other_feat, train_ll], ignore_index=True).dropna(subset=FEATURES2)
        model = make_model()
        model.fit(train[FEATURES2], train["result"])
        joblib.dump(
            {"model": model, "features": FEATURES2, "train_seasons": train_seasons},
            path,
        )
        print(f"\nModel {nom} guardat: {path}")
        print(f"  Train: La Liga {train_seasons[0]} -> {train_seasons[-1]} + resta de lligues ({len(train)} partits)")

    _fit_and_dump(seasons[-WINDOW:], MODEL_PATH, "de PRODUCCIÓ (per predict_next.py)")
    _fit_and_dump(seasons[-WINDOW - 3:-3], MODEL_EVAL_PATH,
                  f"d'AVALUACIÓ, sense {seasons[-3]}, {seasons[-2]} ni {seasons[-1]} (per comparar_cases.py / sense_estrategia.py / estrategia.py)")


def rps(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Ranked Probability Score mitjà (resultats ordinals Local<Empat<Visitant)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p[:, :2] - cum_o[:, :2]) ** 2, axis=1) / 2.0))


def compute_metrics(df: pd.DataFrame) -> dict:
    """Accuracy, precision/recall/F1 per classe, log loss, RPS i matriu de
    confusió d'un DataFrame amb columnes prob_*/prediccio/resultat_real."""
    y    = df["resultat_real"].map(INV).to_numpy()
    pred = df["prediccio"].map(INV).to_numpy()

    P = df[["prob_Local", "prob_Empat", "prob_Visitant"]].to_numpy(float) / 100
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0)

    return {
        "n":    len(df),
        "acc":  float((pred == y).mean()) * 100,
        "ll":   log_loss(y, P, labels=[0, 1, 2]),
        "rps":  rps(P, y),
        "prec": prec, "rec": rec, "f1": f1, "sup": sup,
        "cm":   confusion_matrix(y, pred, labels=[0, 1, 2]),
    }


def print_metrics(name: str, m: dict) -> None:
    print(f"\n=== {name}  ({m['n']} partits) ===\n")
    print(f"Accuracy : {m['acc']:6.2f}%")
    print(f"Log loss : {m['ll']:6.4f}")
    print(f"RPS      : {m['rps']:6.4f}\n")

    print(f"{'Classe':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Suport':>6}")
    print("-" * 50)
    for i in range(3):
        print(f"{LABELS[i]:>9} | {m['prec'][i]*100:>8.2f}% | {m['rec'][i]*100:>6.2f}% | "
              f"{m['f1'][i]*100:>5.2f}% | {int(m['sup'][i]):>6}")

    cm = m["cm"]
    print(f"\nMatriu de confusió (files = real, columnes = predit):")
    etiq = "real \\\\ pred"   # fora de l'f-string: els backslashs dins d'f-strings peten amb Python < 3.12
    print(f"{etiq:>11} | {'Local':>7} | {'Empat':>7} | {'Visit':>7}")
    print("-" * 44)
    for i in range(3):
        print(f"{LABELS[i]:>11} | {cm[i,0]:>7} | {cm[i,1]:>7} | {cm[i,2]:>7}")


# ──────────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────────
def _acc_color(a: float) -> str:
    return "#22c55e" if a >= 55 else "#f59e0b" if a >= 45 else "#ef4444"


def _cm_table(cm: np.ndarray) -> str:
    rows = ""
    for i in range(3):
        cells = "".join(f"<td>{cm[i, j]}</td>" for j in range(3))
        rows += f"<tr><th>{LABELS[i]}</th>{cells}</tr>"
    return f"""
    <table class="cm">
      <tr><th></th><th colspan="3">Predicció</th></tr>
      <tr><th>Real</th><th>Local</th><th>Empat</th><th>Visit</th></tr>
      {rows}
    </table>"""


def build_html(folds: list[dict], fold_metrics: list[dict], agg_metrics: dict) -> str:
    summary_rows = ""
    for f, m in zip(folds, fold_metrics):
        rng = f"{f['train_seasons'][0]} → {f['train_seasons'][-1]}"
        summary_rows += (
            "<tr>"
            f"<td>{rng}</td>"
            f"<td><strong>{f['test_season']}</strong></td>"
            f"<td>{m['n']}</td>"
            f"<td><span class='badge' style='background:{_acc_color(m['acc'])}'>{m['acc']:.1f}%</span></td>"
            f"<td>{m['ll']:.4f}</td>"
            f"<td>{m['rps']:.4f}</td>"
            "</tr>"
        )
    summary_rows += (
        "<tr class='agg'>"
        "<td colspan='2'>TOTAL (tots els folds)</td>"
        f"<td>{agg_metrics['n']}</td>"
        f"<td><span class='badge' style='background:{_acc_color(agg_metrics['acc'])}'>{agg_metrics['acc']:.1f}%</span></td>"
        f"<td>{agg_metrics['ll']:.4f}</td>"
        f"<td>{agg_metrics['rps']:.4f}</td>"
        "</tr>"
    )

    bars = ""
    for f, m in zip(folds, fold_metrics):
        bars += f"""
      <div class="bar-row">
        <span class="bar-label">{f['test_season']}</span>
        <div class="bar-track"><div class="bar-fill" style="width:{m['acc']}%;background:{_acc_color(m['acc'])}"></div></div>
        <span class="bar-val">{m['acc']:.1f}%</span>
      </div>"""

    fold_cards = ""
    for f, m in zip(folds, fold_metrics):
        rng = f"{f['train_seasons'][0]} → {f['train_seasons'][-1]}"
        fold_cards += f"""
    <div class="card">
      <h2>Test {f['test_season']} <span class="badge" style="background:{_acc_color(m['acc'])}">{m['acc']:.1f}%</span></h2>
      <p class="subtitle-card">Entrenat amb {rng} + resta de lligues · {m['n']} partits</p>
      {_cm_table(m['cm'])}
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>Walk-forward backtest — La Liga</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding-bottom: 3rem; }}
  h1 {{ text-align: center; padding: 2rem 1rem .3rem; font-size: 1.8rem; color: #f8fafc; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .9rem; }}
  .subtitle-card {{ color: #94a3b8; font-size: .78rem; margin-bottom: .9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.4rem; padding: 0 2rem; max-width: 1100px; margin: 0 auto; }}
  .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem; }}
  .card {{ background: #1e2433; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3748; }}
  .card h2 {{ font-size: 1rem; font-weight: 600; color: #f8fafc; margin-bottom: .3rem; display: flex; align-items: center; gap: .6rem; }}
  .badge {{ font-size: .72rem; font-weight: 700; color: #0f1117; padding: .15rem .55rem; border-radius: 999px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ color: #64748b; font-weight: 600; text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #2d3748; }}
  td {{ padding: .35rem .7rem; border-bottom: 1px solid #1a2030; font-variant-numeric: tabular-nums; }}
  tr.agg td {{ border-top: 2px solid #2d3748; font-weight: 700; color: #a3e635; }}
  table.cm th, table.cm td {{ text-align: center; padding: .4rem .6rem; }}
  table.cm th:first-child, table.cm td:first-child {{ text-align: left; color: #94a3b8; }}
  .bar-row {{ display: flex; align-items: center; gap: .8rem; margin-bottom: .55rem; }}
  .bar-label {{ width: 3.4rem; font-size: .8rem; color: #94a3b8; }}
  .bar-track {{ flex: 1; background: #1a2030; border-radius: 6px; height: 14px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; }}
  .bar-val {{ width: 3.2rem; text-align: right; font-size: .8rem; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>Walk-forward backtest — La Liga</h1>
<p class="subtitle">Finestra mòbil de {WINDOW} temporades · ELO i rolling calculats un sol cop sobre tot l'històric (mai es reinicien) · model nou a cada fold</p>

<div class="grid">

  <div class="card">
    <h2>Resum per fold</h2>
    <table>
      <tr><th>Train (La Liga)</th><th>Test</th><th>n</th><th>Accuracy</th><th>LogLoss</th><th>RPS</th></tr>
      {summary_rows}
    </table>
  </div>

  <div class="card">
    <h2>Accuracy per fold</h2>
    {bars}
  </div>

  <div class="cards-grid">
    {fold_cards}
  </div>

</div>
</body>
</html>"""


if __name__ == "__main__":
    print(f"Carregant dades i executant walk-forward (finestra de {WINDOW} temporades)...")
    folds = get_fold_probabilities()

    fold_metrics = []
    for f in folds:
        m = compute_metrics(f["df"])
        fold_metrics.append(m)
        print_metrics(f"Fold - train {f['train_seasons'][0]}->{f['train_seasons'][-1]}  test {f['test_season']}", m)

    df_agg = pd.concat([f["df"] for f in folds], ignore_index=True)
    agg_metrics = compute_metrics(df_agg)
    print_metrics("TOTAL (tots els folds)", agg_metrics)

    out_path = os.path.join(_ROOT, "walkforward_report.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build_html(folds, fold_metrics, agg_metrics))
    print(f"\nHTML generat: {out_path}")

    train_and_save_final_models()
