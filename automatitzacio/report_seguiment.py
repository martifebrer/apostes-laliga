"""
Report HTML de seguiment de les apostes reals (data/jornades_recomanacions.xlsx).

El full "Apostes" barreja dues coses per disseny (vegeu inicialitza_memoria.py):
  - Les apostes del backtest de 3 temporades amb què es va sembrar la memòria
    (resultat ja conegut, no canvien mai).
  - Les apostes recomanades EN VIU per actualitza_jornada.py, que es van
    resolent setmana a setmana a mesura que es juguen els partits.

Com que el bankroll NO es compon entre temporades (cada temporada torna a
començar a BANKROLL_INICIAL -- vegeu estrategia.py), aquest report separa:
  - La temporada en curs: KPIs + evolució del bankroll aposta a aposta, que
    és el que interessa consultar setmana rere setmana.
  - Temporades anteriors: només un resum (ja tancat, no canvia).

Genera seguiment_apostes.html a l'arrel del projecte.
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import base64
import io
import webbrowser

import pandas as pd

from actualitza_historic import temporada_actual_understat
from estrategia import BANKROLL_INICIAL
from memoria import MEMORIA_PATH, llegeix_memoria

OUT_PATH = os.path.join(_ROOT, "seguiment_apostes.html")
RECENTS_N = 25


def _season(date: pd.Timestamp) -> str:
    """Mateixa convenció que la resta del projecte: la temporada s'identifica
    per l'any en què comença (agost-juliol)."""
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def _season_actual() -> str:
    y = int(temporada_actual_understat())
    return f"{y}-{str(y + 1)[-2:]}"


def _bankroll_chart_b64(resoltes: pd.DataFrame) -> str:
    """Evolució del bankroll aposta a aposta (només temporada en curs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resoltes = resoltes.sort_values("date")
    bankroll = BANKROLL_INICIAL + resoltes["profit"].cumsum()
    x = range(1, len(resoltes) + 1)

    bg, muted = "#1e2433", "#94a3b8"
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=bg)
    ax.set_facecolor(bg)
    ax.plot(x, bankroll, color="#60a5fa", linewidth=1.8, marker="o", markersize=3)
    ax.axhline(BANKROLL_INICIAL, color=muted, linewidth=1, linestyle="--", alpha=.6)
    ax.set_ylabel("Bankroll (EUR)", color="#60a5fa")
    ax.set_xlabel("Aposta resolta #", color=muted)
    ax.tick_params(axis="x", colors=muted)
    ax.tick_params(axis="y", colors="#60a5fa")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.grid(axis="y", color="#2d3748", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _kpi_row(bankroll_actual: float, profit: float, roi: float, win_rate: float, pendents: int) -> str:
    profit_color = "#22c55e" if profit >= 0 else "#ef4444"
    roi_color = "#22c55e" if roi >= 0 else "#ef4444"
    return f"""
    <div class="kpi-row">
      <div class="kpi"><div class="label">Bankroll actual</div><div class="value">{bankroll_actual:.2f} €</div></div>
      <div class="kpi"><div class="label">Profit</div><div class="value" style="color:{profit_color}">{profit:+.2f} €</div></div>
      <div class="kpi"><div class="label">ROI</div><div class="value" style="color:{roi_color}">{roi:+.1f}%</div></div>
      <div class="kpi"><div class="label">Encert</div><div class="value">{win_rate:.1f}%</div></div>
      <div class="kpi"><div class="label">Pendents</div><div class="value">{pendents}</div></div>
    </div>"""


def _taula_apostes(df: pd.DataFrame) -> str:
    def result_color(g):
        if pd.isna(g):
            return "#94a3b8"
        return "#22c55e" if g else "#ef4444"

    rows = ""
    for _, r in df.sort_values("date", ascending=False).iterrows():
        data = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        if pd.isna(r["resultat_real"]):
            estat, profit_cell = "<span style='color:#94a3b8'>pendent</span>", "—"
        else:
            guanyada = r["guanyada"] == True   # noqa: E712 (columna object amb NaN barrejats)
            estat = f"<span style='color:{result_color(guanyada)}'>{'guanyada' if guanyada else 'perduda'}</span>"
            profit_cell = f"{r['profit']:+.2f}"
        rows += (
            "<tr>"
            f"<td>{data}</td><td>{r['home_team']} vs {r['away_team']}</td>"
            f"<td>{r['tipus']}</td><td>{r['opcio']}</td><td>{r['quota']}</td>"
            f"<td>{r['stake']:.2f}</td><td>{estat}</td><td>{profit_cell}</td>"
            "</tr>"
        )
    return rows


def build_html(apostes: pd.DataFrame, confidence: dict, bankroll_actual: float) -> str:
    apostes = apostes.copy()
    apostes["season"] = apostes["date"].map(_season)
    season_actual = _season_actual()

    actual = apostes[apostes["season"] == season_actual]
    resoltes_actual = actual[actual["resultat_real"].notna()]
    pendents_actual = actual[actual["resultat_real"].isna()]

    n_resoltes = len(resoltes_actual)
    n_guanyades = int((resoltes_actual["guanyada"] == True).sum()) if n_resoltes else 0  # noqa: E712
    win_rate = 100 * n_guanyades / n_resoltes if n_resoltes else 0.0
    profit_total = round(resoltes_actual["profit"].sum(), 2) if n_resoltes else 0.0
    stake_total = round(resoltes_actual["stake"].sum(), 2) if n_resoltes else 0.0
    roi = 100 * profit_total / stake_total if stake_total else 0.0

    if n_resoltes:
        chart_html = (f'<div class="card full"><h2>Evolució del bankroll ({season_actual})</h2>'
                       f'<img src="data:image/png;base64,{_bankroll_chart_b64(resoltes_actual)}" '
                       f'style="width:100%;display:block;border-radius:8px" alt="Evolució del bankroll"></div>')
    else:
        chart_html = (f'<div class="card full"><h2>Evolució del bankroll ({season_actual})</h2>'
                       f'<p style="color:#94a3b8">Encara no hi ha cap aposta resolta aquesta temporada -- '
                       f'aniran apareixent a mesura que actualitza_jornada.py les recomani i es juguin '
                       f'els partits.</p></div>')

    recents_actual = pd.concat([pendents_actual, resoltes_actual]).sort_values("date", ascending=False)
    taula_actual = _taula_apostes(recents_actual) if not recents_actual.empty else (
        "<tr><td colspan='8' style='color:#94a3b8'>Sense apostes encara aquesta temporada</td></tr>")

    # ── Resum de temporades anteriors (backtest, ja tancades) ──────────────────
    anteriors = apostes[apostes["season"] != season_actual]
    files_anteriors = ""
    for season in sorted(anteriors["season"].unique(), reverse=True):
        sub = anteriors[anteriors["season"] == season]
        n = len(sub)
        guanyades = int((sub["guanyada"] == True).sum())  # noqa: E712
        wr = 100 * guanyades / n if n else 0.0
        profit_s = round(sub["profit"].sum(), 2)
        bankroll_final = BANKROLL_INICIAL + profit_s
        color = "#22c55e" if profit_s >= 0 else "#ef4444"
        files_anteriors += (
            "<tr>"
            f"<td>{season}</td><td>{n}</td><td>{wr:.1f}%</td>"
            f"<td style='color:{color};font-weight:600'>{profit_s:+.2f} €</td>"
            f"<td>{bankroll_final:.2f} €</td>"
            "</tr>"
        )

    conf_rows = ""
    for equip, conf in sorted(confidence.items(), key=lambda kv: -kv[1]):
        color = "#22c55e" if conf > 1.0 else "#ef4444" if conf < 1.0 else "#94a3b8"
        conf_rows += f"<tr><td>{equip}</td><td style='color:{color};font-weight:600'>{conf:.2f}x</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>Seguiment d'apostes</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding-bottom: 3rem; }}
  h1 {{ text-align: center; padding: 2rem 1rem .3rem; font-size: 1.8rem; color: #f8fafc; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; padding: 0 2rem; max-width: 1300px; margin: 0 auto; }}
  .card {{ background: #1e2433; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3748; }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{ font-size: .85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1.1rem; }}
  .kpi-row {{ display: flex; gap: 1.2rem; flex-wrap: wrap; }}
  .kpi {{ flex: 1; min-width: 110px; text-align: center; background: #252d3f; border-radius: 10px; padding: 1rem; }}
  .kpi .label {{ font-size: .75rem; color: #94a3b8; margin-bottom: .3rem; }}
  .kpi .value {{ font-size: 2rem; font-weight: 700; color: #60a5fa; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ color: #64748b; font-weight: 600; text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #2d3748; }}
  td {{ padding: .45rem .7rem; border-bottom: 1px solid #1a2030; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<h1>Seguiment d'apostes</h1>
<p class="subtitle">Temporada en curs: {season_actual} · {len(actual)} aposta(es) ({n_resoltes} resoltes, {len(pendents_actual)} pendents)</p>

<div class="grid">

  <div class="card full">
    <h2>Resum -- temporada {season_actual}</h2>
    {_kpi_row(bankroll_actual, profit_total, roi, win_rate, len(pendents_actual))}
  </div>

  {chart_html}

  <div class="card full">
    <h2>Apostes -- temporada {season_actual}</h2>
    <table>
      <tr><th>Data</th><th>Partit</th><th>Tipus</th><th>Opció</th><th>Quota</th><th>Stake</th><th>Estat</th><th>Profit</th></tr>
      {taula_actual}
    </table>
  </div>

  <div class="card">
    <h2>Temporades anteriors (backtest, tancades)</h2>
    <table>
      <tr><th>Temporada</th><th>Apostes</th><th>Encert</th><th>Profit</th><th>Bankroll final</th></tr>
      {files_anteriors}
    </table>
  </div>

  <div class="card">
    <h2>Confiança per equip (acumulada)</h2>
    <table>
      <tr><th>Equip</th><th>Confiança</th></tr>
      {conf_rows}
    </table>
  </div>

</div>
</body>
</html>"""


def generate_report(path: str = MEMORIA_PATH, out_path: str = OUT_PATH, open_browser: bool = True) -> str:
    apostes, confidence, bankroll = llegeix_memoria(path)
    html = build_html(apostes, confidence, bankroll)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report de seguiment generat: {out_path}")
    if open_browser:
        webbrowser.open(f"file://{out_path}")
    return out_path


if __name__ == "__main__":
    generate_report()
