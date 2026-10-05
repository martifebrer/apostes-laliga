"""
Compara les probabilitats del model JA ENTRENAT (model_rf_eval.joblib,
generat per entrenament.py SENSE les tres últimes temporades, per evitar
data leakage) amb les probabilitats NORMALITZADES de diverses cases
d'apostes (fitxer data/SP1_2526.csv, football-data.co.uk, La Liga 2025-26).

No es reentrena res: es carrega el model amb joblib i es calculen només
les features de La Liga (ELO + rolling curt/llarg, un sol cop sobre tot
l'històric) per fer predict_proba de la temporada demanada.

Per cada casa, la probabilitat surt de la quota descomptant el marge:
    prob(opció) = (1 / quota_opció) / overround
    overround   = 1/quota_Local + 1/quota_Empat + 1/quota_Visitant

Sortida:
  - Per pantalla: accuracy, log loss i RPS del model i de cada casa.
  - HTML (comparar_report.html): partits agrupats per jornada, i a sota
    de cada partit la probabilitat que li donava cada casa.
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
import webbrowser

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import load_la_liga
from entrenament import build_dual_features, MODEL_EVAL_PATH, LABELS as MODEL_LABELS
from report_jornades import add_jornades

SP1_PATH = os.path.join(_ROOT, "data", "SP1_2526.csv")
SEASON   = "2025-26"
LABELS   = ["Local", "Empat", "Visitant"]          # ordre = índex 0,1,2

# Equivalència noms dels mains  ->  noms a SP1
NAME_MAP = {
    "Athletic Club":   "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Celta Vigo":      "Celta",
    "Espanyol":        "Espanol",
    "Rayo Vallecano":  "Vallecano",
    "Real Betis":      "Betis",
    "Real Sociedad":   "Sociedad",
    "Real Valladolid": "Valladolid",
    "Real Oviedo":     "Oviedo",
}
REV_MAP = {v: k for k, v in NAME_MAP.items()}       # SP1 -> mains

# Cases a comparar: (codi de columna a SP1, nom per mostrar). football-data.co.uk
# canvia quines cases publica d'una temporada a l'altra (p.ex. Betfair passa de
# "BF" a "BFD" i "1XB" desapareix a 2025-26); les que no existeixin al fitxer
# es descarten en temps d'execució (vegeu construir_dades).
CASES = [
    ("B365", "Bet365"),
    ("BFD",  "Betfair"),
    ("WH",   "William Hill"),
    ("BW",   "Bet&Win"),
    ("Avg",  "Mitjana mercat"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Mètriques
# ──────────────────────────────────────────────────────────────────────────────
def normalitza_quotes(odds: np.ndarray) -> np.ndarray:
    """odds shape (n, 3) -> probabilitats (n, 3) descomptant l'overround."""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def rps(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Ranked Probability Score mitjà (resultats ordinals Local<Empat<Visitant).
    probs (n,3) en fraccions; y_idx (n,) amb el resultat real 0/1/2."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    # només els 2 primers cumulatius (el tercer sempre és 1-1=0)
    return float(np.mean(np.sum((cum_p[:, :2] - cum_o[:, :2]) ** 2, axis=1) / 2.0))


def metriques(probs: np.ndarray, y_idx: np.ndarray) -> dict:
    """Accuracy, log loss i RPS d'una matriu de probabilitats (n,3)."""
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred  = probs.argmax(axis=1)
    return {
        "n":   len(y_idx),
        "acc": float((pred == y_idx).mean()) * 100,
        "ll":  log_loss(y_idx, probs, labels=[0, 1, 2]),
        "rps": rps(probs, y_idx),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dades
# ──────────────────────────────────────────────────────────────────────────────
def get_model_predictions_from_saved(season: str = SEASON):
    """
    Prediccions del model d'avaluació JA ENTRENAT (model_rf_eval.joblib,
    que NO ha vist `season` durant l'entrenament) per a `season`,
    sense reentrenar res. Les features de La Liga (ELO + rolling curt/
    llarg) es calculen un sol cop sobre tot l'històric complet -- no cal
    carregar les altres lligues, que només fan falta per entrenar.
    """
    if not os.path.exists(MODEL_EVAL_PATH):
        raise FileNotFoundError(
            f"No s'ha trobat el model a {MODEL_EVAL_PATH}. Executa entrenament.py per entrenar-lo i desar-lo."
        )
    bundle = joblib.load(MODEL_EVAL_PATH)
    model, features = bundle["model"], bundle["features"]

    df_laliga_feat = build_dual_features(load_la_liga())
    test = df_laliga_feat[df_laliga_feat["season"] == season].dropna(subset=features).copy()
    if test.empty:
        raise ValueError(f"No hi ha partits de la temporada {season} amb features completes.")

    proba = model.predict_proba(test[features])   # shape: (n_partits, 3)
    for idx, cls in enumerate(model.classes_):
        test[f"prob_{MODEL_LABELS[cls]}"] = (proba[:, idx] * 100).round(1)

    test["prediccio"]     = [MODEL_LABELS[model.classes_[row.argmax()]] for row in proba]
    test["resultat_real"] = test["result"].map(MODEL_LABELS)
    test["correct"]       = (test["prediccio"] == test["resultat_real"]).astype(int)
    test["date"]          = pd.to_datetime(test["date"])
    test["home_elo"]      = test["home_elo"].round().astype(int)
    test["away_elo"]      = test["away_elo"].round().astype(int)

    cols = ["date", "home_team", "away_team", "home_elo", "away_elo",
            "prob_Local", "prob_Empat", "prob_Visitant",
            "prediccio", "resultat_real", "correct"]
    return test[cols].reset_index(drop=True), season


def construir_dades(season: str = SEASON):
    global CASES

    # Prediccions del model ja entrenat (sense reentrenar)
    df, season = get_model_predictions_from_saved(season)
    df = add_jornades(df)   # columna 'jornada' per agrupar l'HTML

    # Resultat real com a índex 0/1/2
    df["y_idx"] = df["resultat_real"].map({l: i for i, l in enumerate(LABELS)})

    # SP1 amb noms d'equip mapejats als dels mains
    sp = pd.read_csv(SP1_PATH, encoding="utf-8-sig").copy()
    sp = sp.assign(
        home_team=sp["HomeTeam"].replace(REV_MAP),
        away_team=sp["AwayTeam"].replace(REV_MAP),
    ).copy()

    # Descarta cases el codi de columna de les quals no existeixi en aquest SP1
    # (football-data.co.uk canvia quines cases publica d'una temporada a l'altra)
    disponibles = [c for c in CASES if all(f"{c[0]}{s}" in sp.columns for s in "HDA")]
    descartades = [c for c in CASES if c not in disponibles]
    if descartades:
        print(f"Avís: no hi ha quotes de {[n for _, n in descartades]} a {SP1_PATH}, es descarten.")
    CASES = disponibles

    odd_cols = [f"{c}{s}" for c, _ in CASES for s in ("H", "D", "A")]
    sp_small = sp[["home_team", "away_team", *odd_cols]]

    df = df.merge(sp_small, on=["home_team", "away_team"], how="left")

    # Probabilitats normalitzades de cada casa (en %)
    for code, _ in CASES:
        odds = df[[f"{code}H", f"{code}D", f"{code}A"]].to_numpy(dtype=float)
        probs = normalitza_quotes(odds) * 100
        for i, lab in enumerate(LABELS):
            df[f"{code}_{lab}"] = np.round(probs[:, i], 1)

    return df, season


def calcular_metriques(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y_idx"].to_numpy()

    files = []
    # Model
    pm = df[["prob_Local", "prob_Empat", "prob_Visitant"]].to_numpy(dtype=float) / 100
    files.append(("Model", *metriques(pm, y).values()))

    # Cases (només files amb quota disponible per a aquella casa)
    for code, name in CASES:
        cols = [f"{code}_{l}" for l in LABELS]
        mask = df[cols].notna().all(axis=1)
        p = df.loc[mask, cols].to_numpy(dtype=float) / 100
        m = metriques(p, y[mask.to_numpy()])
        files.append((name, m["n"], m["acc"], m["ll"], m["rps"]))

    return pd.DataFrame(files, columns=["Font", "n", "Accuracy", "LogLoss", "RPS"])


# ──────────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────────
def _prob_cells(vals, prefix=""):
    """3 cel·les de probabilitat amb la màxima de la fila en negreta blava."""
    nums = [v if pd.notna(v) else None for v in vals]
    mx = max([v for v in nums if v is not None], default=None)
    out = ""
    for v in nums:
        if v is None:
            out += "<td style='color:#475569'>—</td>"
        else:
            style = " style='color:#60a5fa;font-weight:700'" if v == mx else ""
            out += f"<td{style}>{v}%</td>"
    return out


def build_html(df: pd.DataFrame, season: str, met: pd.DataFrame) -> str:
    acc_global = round(df["correct"].mean() * 100, 1)

    def acc_color(a):
        return "#22c55e" if a >= 60 else "#f59e0b" if a >= 40 else "#ef4444"

    # ── Taula de mètriques (resum) ────────────────────────────────────────────
    met_rows = ""
    for _, r in met.iterrows():
        destacat = " style='color:#a3e635;font-weight:700'" if r["Font"] == "Model" else ""
        met_rows += (
            "<tr>"
            f"<td{destacat}>{r['Font']}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{r['Accuracy']:.1f}%</td>"
            f"<td>{r['LogLoss']:.4f}</td>"
            f"<td>{r['RPS']:.4f}</td>"
            "</tr>"
        )

    # ── Blocs per jornada ─────────────────────────────────────────────────────
    n_src = 1 + len(CASES)   # model + cases (per al rowspan)
    jornada_blocks = ""
    for jornada in sorted(df["jornada"].unique()):
        sub = df[df["jornada"] == jornada]
        jacc = round(sub["correct"].mean() * 100, 1)

        match_rows = ""
        for _, r in sub.iterrows():
            partit = (f"<strong>{r['home_team']}</strong> "
                      f"<span style='color:#64748b'>vs</span> "
                      f"<strong>{r['away_team']}</strong>")
            real = r["resultat_real"]
            ok = ("<span style='color:#22c55e'>OK</span>" if r["correct"]
                  else "<span style='color:#ef4444'>X</span>")

            # Fila del model (porta el Partit, el Real i l'Encert amb rowspan)
            m_probs = _prob_cells([r["prob_Local"], r["prob_Empat"], r["prob_Visitant"]])
            match_rows += (
                "<tr class='model'>"
                f"<td rowspan='{n_src}'>{partit}</td>"
                "<td class='src'>🎯 Model</td>"
                f"{m_probs}"
                f"<td rowspan='{n_src}' style='text-align:center'>{real}</td>"
                f"<td rowspan='{n_src}' style='text-align:center'>{ok}</td>"
                "</tr>"
            )
            # Files de les cases
            for code, name in CASES:
                vals = [r[f"{code}_{l}"] for l in LABELS]
                match_rows += (
                    "<tr class='casa'>"
                    f"<td class='src'>{name}</td>"
                    f"{_prob_cells(vals)}"
                    "</tr>"
                )

        jornada_blocks += f"""
  <div class="card full">
    <h2>Jornada {jornada}
      <span class="badge" style="background:{acc_color(jacc)}">{jacc}% model · {int(sub['correct'].sum())}/{len(sub)}</span>
    </h2>
    <table>
      <tr><th>Partit</th><th>Font</th><th>P(Local)</th><th>P(Empat)</th><th>P(Visitant)</th><th>Real</th><th>Model OK</th></tr>
      {match_rows}
    </table>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>Model vs Cases — La Liga {season}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding-bottom: 3rem; }}
  h1 {{ text-align: center; padding: 2rem 1rem .3rem; font-size: 1.8rem; color: #f8fafc; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.4rem; padding: 0 2rem; max-width: 1100px; margin: 0 auto; }}
  .card {{ background: #1e2433; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3748; }}
  .card h2 {{ font-size: .85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1.1rem; display: flex; align-items: center; gap: .8rem; }}
  .badge {{ font-size: .72rem; font-weight: 700; color: #0f1117; padding: .15rem .55rem; border-radius: 999px; text-transform: none; letter-spacing: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ color: #64748b; font-weight: 600; text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #2d3748; }}
  td {{ padding: .35rem .7rem; border-bottom: 1px solid #1a2030; font-variant-numeric: tabular-nums; }}
  tr.model td {{ border-top: 2px solid #2d3748; }}
  tr.model .src {{ color: #a3e635; font-weight: 600; }}
  tr.casa .src {{ color: #94a3b8; }}
  tr.casa td {{ color: #cbd5e1; }}
  .metrics td, .metrics th {{ text-align: left; }}
</style>
</head>
<body>
<h1>Model vs Cases d'apostes — La Liga {season}</h1>
<p class="subtitle">Probabilitats de les cases normalitzades (descomptat l'overround) · {len(df)} partits</p>

<div class="grid">

  <div class="card">
    <h2>Mètriques globals (menor LogLoss / RPS = millor)</h2>
    <table class="metrics">
      <tr><th>Font</th><th>n</th><th>Accuracy</th><th>LogLoss</th><th>RPS</th></tr>
      {met_rows}
    </table>
  </div>
{jornada_blocks}

</div>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Carregant prediccions del model i quotes de SP1...")
    df, season = construir_dades()
    met = calcular_metriques(df)

    print(f"\n=== Model vs Cases — La Liga {season}  ({len(df)} partits) ===\n")
    print(f"{'Font':>15} | {'n':>4} | {'Accuracy':>9} | {'LogLoss':>8} | {'RPS':>7}")
    print("-" * 56)
    for _, r in met.iterrows():
        print(f"{r['Font']:>15} | {int(r['n']):>4} | {r['Accuracy']:>8.1f}% | "
              f"{r['LogLoss']:>8.4f} | {r['RPS']:>7.4f}")

    out_path = os.path.join(_ROOT, "comparar_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(df, season, met))
    print(f"\nHTML generat: {out_path}")
    webbrowser.open(f"file://{out_path}")
