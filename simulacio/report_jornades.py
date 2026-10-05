"""
Generació del report HTML agrupat per jornades.

Tota la configuració de jornades i la construcció de l'HTML viu aquí, separada
dels mains (que només entrenen, prediuen i treuen mètriques per pantalla).

Funcions públiques:
  - add_jornades(df)              -> afegeix 'jornada' (la OFICIAL, de
                                     data/jornades.csv) i 'ajornat' (partit jugat
                                     lluny de la seva jornada); si el CSV no
                                     cobreix el df, cau a l'heurística antiga
                                     de blocs de 10 partits per data.
  - generate_report(df, season)   -> escriu l'HTML i (opcional) l'obre al navegador.

El DataFrame d'entrada ha de tenir: date, home_team, away_team, home_elo,
away_elo, prob_Local, prob_Empat, prob_Visitant, resultat_real, correct.
"""

import base64
import io
import os
import webbrowser

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

LABELS_INV = {"Local": 0, "Empat": 1, "Visitant": 2}


JORNADES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jornades.csv")
AJORNAT_MAX_DIES = 3   # partit jugat a més d'aquests dies de la mediana de la seva jornada => ajornat


def _add_jornades_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Heurística antiga (només fallback): ordena per data i talla blocs de 10 partits.
    No sap detectar ajornats, així que marca ajornat=False a tot."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["jornada"] = (df.index // 10) + 1
    df["ajornat"] = False
    return df


def add_jornades(df: pd.DataFrame, jornades_path: str = JORNADES_PATH) -> pd.DataFrame:
    """Ordena per data i afegeix dues columnes:

      - jornada: la jornada OFICIAL del calendari de La Liga (data/jornades.csv,
        construït a partir del calendari públic d'openfootball, validat contra
        l'històric del projecte).
      - ajornat: True si el partit es va jugar a més d'AJORNAT_MAX_DIES dies de la
        mediana de dates de la seva jornada (partit ajornat o avançat). Els backtests
        d'apostes descarten aquests partits; als features (ELO/rolling) sí que hi
        compten, perquè es calculen sobre l'històric complet, no sobre aquest df.

    Si data/jornades.csv no existeix o no cobreix els partits del df (p.ex. una
    temporada antiga), es recorre a l'heurística de blocs de 10 partits per data,
    amb ajornat=False i un avís per pantalla."""
    if not os.path.exists(jornades_path):
        print(f"Avís: no s'ha trobat {jornades_path}; jornades per blocs de 10 (heurística).")
        return _add_jornades_heuristic(df)

    jor = pd.read_csv(jornades_path)
    jor["date_oficial"] = pd.to_datetime(jor["date_oficial"])
    mediana = jor.groupby(["season", "jornada"])["date_oficial"].transform("median")
    jor["ajornat"] = (jor["date_oficial"] - mediana).abs() > pd.Timedelta(days=AJORNAT_MAX_DIES)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_row"] = range(len(df))

    # Merge per parella d'equips. La mateixa parella surt a diverses temporades del
    # CSV: es tria la fila amb la data oficial més propera a la data real de joc.
    m = df.merge(jor[["home_team", "away_team", "date_oficial", "jornada", "ajornat"]],
                 on=["home_team", "away_team"], how="left")
    m["_diff"] = (m["date"].dt.normalize() - m["date_oficial"]).abs()
    m = m.sort_values("_diff", na_position="last").drop_duplicates("_row", keep="first")

    # Si la millor candidata és a més de 4 dies (o no n'hi ha), el CSV no cobreix el df
    dolents = m["_diff"].isna() | (m["_diff"] > pd.Timedelta(days=4))
    if dolents.any():
        print(f"Avís: {int(dolents.sum())} partit(s) sense jornada oficial a {jornades_path}; "
              f"jornades per blocs de 10 (heurística).")
        return _add_jornades_heuristic(df.drop(columns="_row"))

    m["jornada"] = m["jornada"].astype(int)
    m = m.sort_values("_row").drop(columns=["_row", "_diff", "date_oficial"])
    return m.sort_values("date").reset_index(drop=True)


def _team_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    for team in teams:
        sub = df[(df["home_team"] == team) | (df["away_team"] == team)]
        rows.append({
            "equip":     team,
            "partits":   len(sub),
            "correctes": int(sub["correct"].sum()),
            "acc":       round(sub["correct"].mean() * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("acc", ascending=False)


def _jornada_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    jdf = (df.groupby("jornada")["correct"]
             .agg(partits="count", correctes="sum")
             .reset_index())
    jdf["acc"] = (jdf["correctes"] / jdf["partits"] * 100).round(1)
    return jdf


def _per_jornada_metrics(df: pd.DataFrame):
    """Retorna (jornades, accuracies %, loglosses) per cada jornada."""
    js, accs, lls = [], [], []
    for j in sorted(df["jornada"].unique()):
        sub = df[df["jornada"] == j]
        y = sub["resultat_real"].map(LABELS_INV).to_numpy()
        P = sub[["prob_Local", "prob_Empat", "prob_Visitant"]].to_numpy(float) / 100
        P = np.clip(P, 1e-12, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        js.append(int(j))
        accs.append(sub["correct"].mean() * 100)
        lls.append(log_loss(y, P, labels=[0, 1, 2]))
    return js, accs, lls


def _jornada_chart_b64(df: pd.DataFrame) -> str:
    """Gràfica per jornada: accuracy (barres) i log loss (línia). PNG en base64."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    js, accs, lls = _per_jornada_metrics(df)

    bg, fg, muted = "#1e2433", "#e2e8f0", "#94a3b8"
    fig, ax1 = plt.subplots(figsize=(11, 4), facecolor=bg)
    ax1.set_facecolor(bg)

    # Accuracy (barres, eix esquerre)
    ax1.bar(js, accs, color="#60a5fa", alpha=0.55, label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color="#60a5fa")
    ax1.set_xlabel("Jornada", color=muted)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", colors=muted)
    ax1.tick_params(axis="y", colors="#60a5fa")

    # Log loss (línia, eix dret)
    ax2 = ax1.twinx()
    ax2.plot(js, lls, color="#a3e635", marker="o", markersize=4, linewidth=1.8, label="Log loss")
    ax2.set_ylabel("Log loss", color="#a3e635")
    ax2.tick_params(axis="y", colors="#a3e635")

    for spine in (*ax1.spines.values(), *ax2.spines.values()):
        spine.set_color("#2d3748")
    ax1.grid(axis="y", color="#2d3748", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(df: pd.DataFrame, season: str) -> str:
    acc_global = round(df["correct"].mean() * 100, 1)
    team_df    = _team_accuracy(df)
    jorn_df    = _jornada_accuracy(df)
    chart_b64  = _jornada_chart_b64(df)

    def acc_color(acc):
        return "#22c55e" if acc >= 60 else "#f59e0b" if acc >= 40 else "#ef4444"

    def prob_cell(val, is_max):
        style = ' style="color:#60a5fa;font-weight:700"' if is_max else ''
        return f'<td{style}>{val}%</td>'

    # ── Partits agrupats per jornada ──────────────────────────────────────────
    jornada_blocks = ""
    for jornada in sorted(df["jornada"].unique()):
        sub = df[df["jornada"] == jornada]
        jacc = round(sub["correct"].mean() * 100, 1)
        match_rows = ""
        for _, r in sub.iterrows():
            probs = {"Local": r["prob_Local"], "Empat": r["prob_Empat"], "Visitant": r["prob_Visitant"]}
            mx = max(probs, key=probs.get)
            ok = ("<span style='color:#22c55e'>OK</span>" if r["correct"]
                  else "<span style='color:#ef4444'>X</span>")
            match_rows += (
                "<tr>"
                f"<td>{r['home_team']}</td>"
                f"<td class='elo'>{r['home_elo']}</td>"
                f"<td>{r['away_team']}</td>"
                f"<td class='elo'>{r['away_elo']}</td>"
                + prob_cell(r["prob_Local"],    mx == "Local")
                + prob_cell(r["prob_Empat"],    mx == "Empat")
                + prob_cell(r["prob_Visitant"], mx == "Visitant")
                + f"<td>{r['resultat_real']}</td>"
                + f"<td style='text-align:center'>{ok}</td>"
                "</tr>"
            )
        jornada_blocks += f"""
  <div class="card full">
    <h2>Jornada {jornada}
      <span class="badge" style="background:{acc_color(jacc)}">{jacc}% encert · {int(sub['correct'].sum())}/{len(sub)}</span>
    </h2>
    <table>
      <tr><th>Local</th><th>ELO</th><th>Visitant</th><th>ELO</th><th>P(Local)</th><th>P(Empat)</th><th>P(Visitant)</th><th>Real</th><th>Encert</th></tr>
      {match_rows}
    </table>
  </div>"""

    # ── Taula resum per equip ─────────────────────────────────────────────────
    team_rows = ""
    for _, r in team_df.iterrows():
        team_rows += (
            "<tr>"
            f"<td>{r['equip']}</td>"
            f"<td>{r['partits']}</td>"
            f"<td>{r['correctes']}</td>"
            f"<td style='color:{acc_color(r['acc'])};font-weight:600'>{r['acc']}%</td>"
            "</tr>"
        )

    # ── Taula resum per jornada ───────────────────────────────────────────────
    jorn_rows = ""
    for _, r in jorn_df.sort_values("acc", ascending=False).iterrows():
        jorn_rows += (
            "<tr>"
            f"<td>J{int(r['jornada'])}</td>"
            f"<td>{int(r['partits'])}</td>"
            f"<td>{int(r['correctes'])}</td>"
            f"<td style='color:{acc_color(r['acc'])};font-weight:600'>{r['acc']}%</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>Probabilitats — La Liga {season}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding-bottom: 3rem; }}
  h1 {{ text-align: center; padding: 2rem 1rem .3rem; font-size: 1.8rem; color: #f8fafc; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; padding: 0 2rem; max-width: 1300px; margin: 0 auto; }}
  .card {{ background: #1e2433; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3748; }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{ font-size: .85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1.1rem; display: flex; align-items: center; gap: .8rem; }}
  .badge {{ font-size: .72rem; font-weight: 700; color: #0f1117; padding: .15rem .55rem; border-radius: 999px; text-transform: none; letter-spacing: 0; }}
  .kpi-row {{ display: flex; gap: 1.2rem; }}
  .kpi {{ flex: 1; text-align: center; background: #252d3f; border-radius: 10px; padding: 1rem; }}
  .kpi .label {{ font-size: .75rem; color: #94a3b8; margin-bottom: .3rem; }}
  .kpi .value {{ font-size: 2.2rem; font-weight: 700; color: #60a5fa; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ color: #64748b; font-weight: 600; text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #2d3748; }}
  td {{ padding: .45rem .7rem; border-bottom: 1px solid #1a2030; }}
  tr:last-child td {{ border-bottom: none; }}
  td.elo {{ color: #a3e635; font-variant-numeric: tabular-nums; font-size: .8rem; }}
</style>
</head>
<body>
<h1>Probabilitats per partit — La Liga {season}</h1>
<p class="subtitle">En negreta blava, l'opció més probable · {len(df)} partits</p>

<div class="grid">

  <div class="card full">
    <h2>Resum</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="label">Accuracy global</div><div class="value">{acc_global}%</div></div>
      <div class="kpi"><div class="label">Encerts</div><div class="value">{int(df['correct'].sum())}/{len(df)}</div></div>
      <div class="kpi"><div class="label">Jornades</div><div class="value">{df['jornada'].nunique()}</div></div>
      <div class="kpi"><div class="label">Equips</div><div class="value">{len(team_df)}</div></div>
    </div>
  </div>

  <div class="card full">
    <h2>Accuracy i Log loss per jornada</h2>
    <img src="data:image/png;base64,{chart_b64}" style="width:100%;display:block;border-radius:8px"
         alt="Accuracy i log loss per jornada">
  </div>

  <div class="card">
    <h2>Encert per equip (millors primers)</h2>
    <table>
      <tr><th>Equip</th><th>Partits</th><th>Correctes</th><th>Accuracy</th></tr>
      {team_rows}
    </table>
  </div>

  <div class="card">
    <h2>Encert per jornada (millors primeres)</h2>
    <table>
      <tr><th>Jornada</th><th>Partits</th><th>Correctes</th><th>Accuracy</th></tr>
      {jorn_rows}
    </table>
  </div>
{jornada_blocks}

</div>
</body>
</html>"""


def generate_report(df: pd.DataFrame, season: str,
                    out_path: str = None, open_browser: bool = True) -> str:
    """Afegeix jornades, construeix l'HTML, el guarda i (opcional) l'obre."""
    df = add_jornades(df)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prob_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(df, season))
    if open_browser:
        webbrowser.open(f"file://{out_path}")
    return out_path
