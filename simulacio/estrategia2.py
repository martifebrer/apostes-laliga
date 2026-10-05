"""
Variant d'estrategia.py: en lloc d'una única confiança per equip (compartida
entre "Value Local" i "Contrarian Visitant"), aquí cada equip té DUES
confiances independents, una per cada tipus d'aposta -- clau
`(equip, tipus)` en lloc de només `equip`. Motiu: la correlació empírica
entre el ROI de Value Local i el de Contrarian Visitant per equip surt
NEGATIVA (~-0.27 a les 3 temporades), és a dir que un equip pot ser fiable
com a favorit a casa i alhora fluix com a sorpresa a fora (o al revés) --
no van "de la mà", així que barrejar-les en una sola confiança dilueix
senyal real de cada tipus per separat.

Tota la resta -- selecció d'apostes, Kelly fraccionat, bankroll compost,
esmorteïment de la confiança entre temporades (SEASON_CONF_DECAY) -- és
idèntica a estrategia.py. Genera el seu propi HTML (estrategia2_report.html)
per no xafar el d'estrategia.py.
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
import os
import webbrowser

import numpy as np
import pandas as pd

from comparar_cases import REV_MAP, get_model_predictions_from_saved, normalitza_quotes
from report_jornades import add_jornades
from staking import kelly_stake

QUOTES_PATH  = os.path.join(_ROOT, "data", "quotes_laliga.csv")
ALL_LABELS   = ["Local", "Empat", "Visitant"]   # ordre = columnes H, D, A (cal per l'overround)

# Rang de dates que separa cada temporada dins quotes_laliga.csv (a >= inclòs, b exclòs)
SEASON_RANGES = {
    "2023-24": ("2023-07-01", "2024-07-01"),
    "2024-25": ("2024-07-01", "2025-07-01"),
    "2025-26": ("2025-07-01", "2026-07-01"),
}
SEASONS = list(SEASON_RANGES.keys())

EDGE_MIN  = 0.05   # edge mínim (prob. model - prob. implícita de la MILLOR casa), per a Value Local
PROB_MIN  = {"Local": 0.43}   # confiança mínima del model, per a Value Local

# Staking: Kelly fraccionat (staking.py), amb bankroll REAL que compon jornada
# a jornada (vegeu docstring del mòdul)
BANKROLL_INICIAL = 100.0   # capital amb què es comença cada temporada
KELLY_FRACTION    = 0.25   # 1/4 de Kelly
MAX_STAKE_PCT     = 0.20   # topall per aposta: 20% del pot ACTUAL (no fix), evita que un edge extrem ho desproporcioni tot
MIN_STAKE         = 0.0

# Aposta "Contrarian Visitant": el mercat sobrevalora el Local respecte al
# meu model en més d'aquest marge, i la meva prob. de Visitant supera el mínim
TRIGGER_LOCAL_MIN  = 0.05
PROB_MIN_VISITANT  = 0.30

# Descarta apostes on la quota de la casa triada s'allunya massa de la
# mitjana de mercat ("Avg") -- sol indicar una quota mal actualitzada o
# errònia, no un edge real (vegeu anàlisi manual de William Hill 2025-26)
MAX_DIFF_VS_AVG = 8.0   # % de diferència màxima permesa

# Confiança per (equip, tipus) (Elo-like): multiplicador del stake que puja o
# baixa segons si les apostes fetes en aquell equip, per a AQUELL tipus
# concret, han anat millor o pitjor del que predeia el model. Comença a 1.0
# i persisteix entre temporades (esmorteïda -- vegeu SEASON_CONF_DECAY).
#
# El senyal és el PROFIT real per euro apostat, no simplement si s'ha
# encertat: es compara el retorn real de l'aposta (quota-1 si guanya, -1
# si perd) amb el retorn ESPERAT segons prob_model i quota. Asimètric a
# propòsit: puja ràpid (K_UP) quan supera l'esperat, baixa lent (K_DOWN)
# quan decep.
TEAM_CONF_INICIAL = 1.0
TEAM_CONF_K_UP    = 0.08   # velocitat de pujada quan el profit supera l'esperat
TEAM_CONF_K_DOWN  = 0.01   # velocitat de baixada quan decep
TEAM_CONF_MIN     = 0.5    # mai baixa de mig Kelly per aquell (equip, tipus)
TEAM_CONF_MAX     = 2.0    # mai puja per sobre de 2x Kelly

# A cada canvi de temporada (la primera no, no hi ha història encara) es
# redueix TOTA la confiança acumulada multiplicant-la per aquest factor,
# perquè no s'hi quedi enganxada al topall (vegeu estrategia.py)
SEASON_CONF_DECAY = 0.5

JORNADA_MIN = 6    # no s'aposta a les jornades 1-5 (s'aposta a partir de la 6)
JORNADA_MAX = 34   # no s'aposta a partir de la jornada 35

# Cases d'apostes reals a comparar (codi de columna a quotes_laliga.csv, nom per mostrar).
# S'exclouen "Avg"/"Max" (agregats de mercat, no cases reals) i "BFE" (Betfair Exchange,
# funciona diferent -- comissió i contrapart d'un altre usuari, no una casa fixa).
# BF/BFD i són codis que football-data.co.uk ha fet servir en temporades diferents
# per Betfair; es deixen tots dos, només un existirà per temporada.
CASES_ROI = [
    ("B365", "Bet365"), ("BW", "BWin"), ("WH", "William Hill"),
    ("BFD", "Betfair"), ("BF", "Betfair"),
]


def carregar_quotes(season: str) -> pd.DataFrame:
    start, end = SEASON_RANGES[season]
    q = pd.read_csv(QUOTES_PATH, encoding="utf-8-sig")
    q["Date"] = pd.to_datetime(q["Date"], dayfirst=True)
    q = q[(q["Date"] >= start) & (q["Date"] < end)].copy()

    q["home_team"] = q["HomeTeam"].replace(REV_MAP)
    q["away_team"] = q["AwayTeam"].replace(REV_MAP)

    keep = ["home_team", "away_team", "FTR"]

    # Mitjana de mercat ("Avg"), per detectar quotes de cases individuals que
    # s'allunyen massa del consens (vegeu MAX_DIFF_VS_AVG a simular())
    if all(f"Avg{s}" in q.columns for s in "HDA"):
        for s, lab in zip("HDA", ALL_LABELS):
            q[f"avg_quota_{lab}"] = q[f"Avg{s}"]
        keep += [f"avg_quota_{lab}" for lab in ALL_LABELS]

    for code, _ in CASES_ROI:
        cols = [f"{code}H", f"{code}D", f"{code}A"]
        if not all(c in q.columns for c in cols):
            continue
        mask = q[cols].notna().all(axis=1)
        odds = q.loc[mask, cols].to_numpy(dtype=float)
        probs = normalitza_quotes(odds)
        for i, lab in enumerate(ALL_LABELS):
            q.loc[mask, f"{code}_quota_{lab}"] = odds[:, i]
            q.loc[mask, f"{code}_implicita_{lab}"] = probs[:, i]
        keep += [f"{code}_quota_{lab}" for lab in ALL_LABELS] + [f"{code}_implicita_{lab}" for lab in ALL_LABELS]

    return q[keep]


def _prop_vs_avg(r: pd.Series, avg_col: str, quota: float) -> bool:
    """True si `quota` no s'allunya més de MAX_DIFF_VS_AVG% de la mitjana de
    mercat (avg_col). Si no hi ha mitjana disponible per a aquesta temporada,
    no es filtra (es deixa passar)."""
    if avg_col not in r or pd.isna(r[avg_col]):
        return True
    diff_pct = abs(quota - r[avg_col]) / r[avg_col] * 100
    return diff_pct <= MAX_DIFF_VS_AVG


def simular(season: str, team_confidence: dict | None = None) -> pd.DataFrame:
    """`team_confidence` (clau (equip, tipus) -> multiplicador) es passa i es muta
    in-place perquè pugui persistir entre crides per a diferents temporades (vegeu
    TEAM_CONF_* i simular_totes()) -- si no se'n passa cap, se'n crea un de nou i
    local a la crida."""
    preds, _ = get_model_predictions_from_saved(season)
    quotes = carregar_quotes(season)

    df = preds.merge(quotes, on=["home_team", "away_team"], how="inner")
    print(f"[{season}] Partits amb prediccio del model i alguna quota disponible: {len(df)} "
          f"(de {len(preds)} amb predicció, {len(quotes)} amb quotes)")

    df = add_jornades(df)
    ajornats = int(df["ajornat"].sum())
    if ajornats:
        print(f"[{season}] Descartats {ajornats} partit(s) ajornat(s) (jugats lluny de la seva jornada oficial); "
              f"segueixen comptant per a l'ELO/rolling, que es calculen sobre l'històric complet")
    df = df[~df["ajornat"]]
    abans = len(df)
    df = df[(df["jornada"] >= JORNADA_MIN) & (df["jornada"] <= JORNADA_MAX)]
    print(f"[{season}] Partits dins la finestra de jornades {JORNADA_MIN}-{JORNADA_MAX}: {len(df)} (de {abans})")

    apostes = []
    bankroll = BANKROLL_INICIAL
    if team_confidence is None:
        team_confidence = {}

    # Jornades en ordre CRONOLÒGIC de joc, no pel número (vegeu estrategia.py)
    ordre_jornades = sorted(df["jornada"].unique(),
                            key=lambda j: df.loc[df["jornada"] == j, "date"].min())
    for jornada in ordre_jornades:
        sub = df[df["jornada"] == jornada]

        # 1) Es decideixen les apostes candidates de la jornada (la selecció NO depèn
        #    del pot, només l'stake -- vegeu pas 2)
        candidats = []
        for _, r in sub.iterrows():
            prob_model_local    = r["prob_Local"] / 100
            prob_model_visitant = r["prob_Visitant"] / 100

            # ── Value Local: line shopping, queda't amb la casa que dona més edge a Local ──
            millor_local = None
            for code, nom in CASES_ROI:
                impl_col = f"{code}_implicita_Local"
                if impl_col not in r or pd.isna(r[impl_col]):
                    continue
                edge = prob_model_local - r[impl_col]
                if millor_local is None or edge > millor_local["edge"]:
                    millor_local = {"nom": nom, "edge": edge,
                                     "quota": r[f"{code}_quota_Local"], "implicita": r[impl_col]}

            if (millor_local is not None and millor_local["edge"] >= EDGE_MIN and prob_model_local >= PROB_MIN["Local"]
                    and _prop_vs_avg(r, "avg_quota_Local", millor_local["quota"])):
                candidats.append({
                    "season": season, "jornada": int(jornada), "date": r["date"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "tipus": "Value Local", "opcio": "Local", "casa": millor_local["nom"],
                    "equip": r["home_team"],
                    "prob": prob_model_local,
                    "prob_model": round(prob_model_local * 100, 1),
                    "prob_implicita": round(millor_local["implicita"] * 100, 1),
                    "edge": round(millor_local["edge"] * 100, 1),
                    "quota": millor_local["quota"], "resultat_real": r["resultat_real"],
                    "guanyada": r["resultat_real"] == "Local",
                })

            # ── Contrarian Visitant: la casa que MÉS sobrevalora el Local respecte a mi ──
            pitjor_local = None
            for code, nom in CASES_ROI:
                impl_col_l = f"{code}_implicita_Local"
                impl_col_v = f"{code}_implicita_Visitant"
                if impl_col_l not in r or pd.isna(r[impl_col_l]) or impl_col_v not in r or pd.isna(r[impl_col_v]):
                    continue
                diff = r[impl_col_l] - prob_model_local
                if pitjor_local is None or diff > pitjor_local["diff"]:
                    pitjor_local = {"nom": nom, "diff": diff,
                                     "quota": r[f"{code}_quota_Visitant"], "implicita": r[impl_col_v]}

            if (pitjor_local is not None and pitjor_local["diff"] > TRIGGER_LOCAL_MIN
                    and prob_model_visitant > PROB_MIN_VISITANT
                    and _prop_vs_avg(r, "avg_quota_Visitant", pitjor_local["quota"])):
                candidats.append({
                    "season": season, "jornada": int(jornada), "date": r["date"],
                    "home_team": r["home_team"], "away_team": r["away_team"],
                    "tipus": "Contrarian Visitant", "opcio": "Visitant", "casa": pitjor_local["nom"],
                    "equip": r["away_team"],
                    "prob": prob_model_visitant,
                    "prob_model": round(prob_model_visitant * 100, 1),
                    "prob_implicita": round(pitjor_local["implicita"] * 100, 1),
                    "edge": round(pitjor_local["diff"] * 100, 1),
                    "quota": pitjor_local["quota"], "resultat_real": r["resultat_real"],
                    "guanyada": r["resultat_real"] == "Visitant",
                })

        # 2) L'stake de cada aposta es calcula amb Kelly sobre el pot que hi ha en
        #    COMENÇAR la jornada -- totes les apostes de la jornada s'aposten "a
        #    cegues" dels resultats d'aquesta mateixa jornada, amb el mateix pot
        bankroll_jornada = bankroll
        max_stake = bankroll_jornada * MAX_STAKE_PCT
        profit_jornada = 0.0
        apostades_jornada = []   # candidats amb stake > 0, per actualitzar team_confidence al pas 4

        for c in candidats:
            clau = (c["equip"], c["tipus"])
            conf = team_confidence.get(clau, TEAM_CONF_INICIAL)
            stake = kelly_stake(c["prob"], c["quota"], bankroll_jornada, KELLY_FRACTION, MIN_STAKE, max_stake)
            stake = min(stake * conf, max_stake)   # confiança de l'(equip, tipus) modula el Kelly, sense trencar el topall
            if stake <= 0:
                continue
            profit = (c["quota"] - 1) * stake if c["guanyada"] else -stake
            profit_jornada += profit
            apostades_jornada.append(c)
            apostes.append({
                **{k: v for k, v in c.items() if k not in ("prob", "equip")},
                "bankroll_abans": round(bankroll_jornada, 2),
                "team_conf": round(conf, 3),
                "stake": round(stake, 2),
                "profit": round(profit, 2),
            })

        # 3) El pot es recalcula un cop es coneixen tots els resultats de la jornada,
        #    i passa a ser la base per a la següent
        bankroll += profit_jornada

        # 4) La confiança de cada (equip, tipus) apostat també es recalcula un cop es
        #    coneixen els resultats -- es mou cap al PROFIT real per euro apostat
        #    comparat amb el que s'esperava (no només si s'ha encertat), amb K
        #    asimètric: puja ràpid si supera l'esperat, baixa lent si decep (mateixa
        #    lògica causal que el bankroll: només afecta la jornada següent)
        for c in apostades_jornada:
            clau = (c["equip"], c["tipus"])
            prev = team_confidence.get(clau, TEAM_CONF_INICIAL)
            retorn_esperat = c["prob"] * c["quota"] - 1.0        # EV per euro apostat, segons el meu model
            retorn_real = (c["quota"] - 1.0) if c["guanyada"] else -1.0
            sorpresa = retorn_real - retorn_esperat
            k = TEAM_CONF_K_UP if sorpresa > 0 else TEAM_CONF_K_DOWN
            nova = prev + k * sorpresa
            team_confidence[clau] = min(max(nova, TEAM_CONF_MIN), TEAM_CONF_MAX)

    return pd.DataFrame(apostes)


def simular_totes() -> pd.DataFrame:
    """Encadena les temporades en ordre cronològic (SEASONS ja hi és) compartint el
    mateix `team_confidence` entre crides, perquè la confiança per (equip, tipus)
    persisteixi d'una temporada a la següent -- però esmorteïda (SEASON_CONF_DECAY)
    a cada canvi de temporada perquè no s'hi quedi enganxada (vegeu estrategia.py)."""
    team_confidence = {}
    resultats = []
    for i, season in enumerate(SEASONS):
        if i > 0:
            for clau in team_confidence:
                team_confidence[clau] = min(max(team_confidence[clau] * SEASON_CONF_DECAY,
                                                 TEAM_CONF_MIN), TEAM_CONF_MAX)
        resultats.append(simular(season, team_confidence))
    return pd.concat(resultats, ignore_index=True)


def print_report(apostes: pd.DataFrame) -> None:
    if apostes.empty:
        print("\nCap aposta ha complert els criteris de cap dels dos tipus.")
        return

    n          = len(apostes)
    staked     = apostes["stake"].sum()
    profit     = apostes["profit"].sum()
    roi        = profit / staked * 100
    encertades = int(apostes["guanyada"].sum())

    print(f"\n=== ROI {' / '.join(SEASONS)} (confiança separada per tipus, stake Kelly 1/{int(1/KELLY_FRACTION)}) ===\n")
    print(f"Apostes fetes   : {n}")
    print(f"Encertades      : {encertades} ({encertades/n*100:.1f}%)")
    print(f"Import apostat  : {staked:.2f} EUR")
    print(f"Guany/perdua net: {profit:+.2f} EUR")
    print(f"ROI             : {roi:+.2f}%")

    print(f"\nPer temporada (pot real: comença cada temporada amb {BANKROLL_INICIAL:.0f}€, Kelly sobre el pot actual):")
    print(f"{'Temporada':>10} | {'Apostes':>7} | {'Encerts':>7} | {'Apostat':>9} | {'ROI':>8} | {'Pot final':>10} | {'Creixement':>10}")
    print("-" * 82)
    for season in sorted(apostes["season"].unique()):
        sub = apostes[apostes["season"] == season]
        sub_staked = sub["stake"].sum()
        sub_profit = sub["profit"].sum()
        sub_roi = sub_profit / sub_staked * 100
        bankroll_final = BANKROLL_INICIAL + sub_profit
        creixement = sub_profit / BANKROLL_INICIAL * 100
        print(f"{season:>10} | {len(sub):>7} | {int(sub['guanyada'].sum()):>7} | {sub_staked:>8.2f}€ | {sub_roi:>7.2f}% | "
              f"{bankroll_final:>9.2f}€ | {creixement:>9.2f}%")

    print(f"\nPer tipus d'aposta:")
    print(f"{'Tipus':>22} | {'Apostes':>7} | {'Encerts':>7} | {'Apostat':>9} | {'ROI':>8}")
    print("-" * 66)
    for tipus in apostes["tipus"].unique():
        sub = apostes[apostes["tipus"] == tipus]
        sub_staked = sub["stake"].sum()
        sub_roi = sub["profit"].sum() / sub_staked * 100
        print(f"{tipus:>22} | {len(sub):>7} | {int(sub['guanyada'].sum()):>7} | {sub_staked:>8.2f}€ | {sub_roi:>7.2f}%")

    print(f"\nDetall de les apostes:")
    print(apostes[["date", "home_team", "away_team", "tipus", "opcio", "casa", "prob_model",
                    "prob_implicita", "edge", "quota", "stake", "resultat_real", "profit"]]
          .to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Evolució del pot (bankroll real, compost)
# ──────────────────────────────────────────────────────────────────────────────
def _bankroll_evolution(apostes: pd.DataFrame, season: str) -> pd.DataFrame:
    """Sèrie (jornada, pot) reconstruïda jornada a jornada per a una temporada,
    incloent les jornades sense cap aposta (el pot es manté igual que a l'anterior)."""
    sub = apostes[apostes["season"] == season]
    profit_per_jornada = sub.groupby("jornada")["profit"].sum()

    bankroll = BANKROLL_INICIAL
    rows = [{"jornada": JORNADA_MIN - 1, "bankroll": bankroll}]
    for j in range(JORNADA_MIN, JORNADA_MAX + 1):
        bankroll += profit_per_jornada.get(j, 0.0)
        rows.append({"jornada": j, "bankroll": bankroll})
    return pd.DataFrame(rows)


def _bankroll_chart_b64(apostes: pd.DataFrame) -> str:
    """Gràfica d'evolució del pot (EUR) per jornada, una línia per temporada. PNG en base64."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg, muted = "#1e2433", "#e2e8f0", "#94a3b8"
    colors = ["#60a5fa", "#f97316", "#a3e635"]

    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=bg)
    ax.set_facecolor(bg)

    for i, season in enumerate(sorted(apostes["season"].unique())):
        evo = _bankroll_evolution(apostes, season)
        ax.plot(evo["jornada"], evo["bankroll"], color=colors[i % len(colors)],
                marker="o", markersize=3, linewidth=1.8, label=season)

    ax.axhline(BANKROLL_INICIAL, color="#64748b", linewidth=1, linestyle="--", label=f"Capital inicial ({BANKROLL_INICIAL:.0f}€)")
    ax.set_xlabel("Jornada", color=muted)
    ax.set_ylabel("Pot (EUR)", color=fg)
    ax.tick_params(axis="x", colors=muted)
    ax.tick_params(axis="y", colors=muted)
    ax.legend(facecolor=bg, edgecolor="#2d3748", labelcolor=fg, loc="best", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.grid(axis="y", color="#2d3748", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────────
def _acc_color(roi: float) -> str:
    return "#22c55e" if roi >= 0 else "#ef4444"


TIPUS_COLOR = {"Value Local": "#3b82f6", "Contrarian Visitant": "#f97316"}


def _bet_row(r: pd.Series) -> str:
    data  = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
    cls   = "pos" if r["profit"] >= 0 else "neg"
    ok    = "<span class='pos'>OK</span>" if r["guanyada"] else "<span class='neg'>X</span>"
    color = TIPUS_COLOR.get(r["tipus"], "#64748b")
    return (
        f"<tr class='tipus-row' style='border-left:3px solid {color}'>"
        f"<td>{data}</td>"
        f"<td><span class='tipus-badge' style='background:{color}'>{r['tipus']}</span></td>"
        f"<td><strong>{r['home_team']}</strong> vs <strong>{r['away_team']}</strong></td>"
        f"<td>{r['opcio']}</td>"
        f"<td>{r['casa']}</td>"
        f"<td>{r['quota']:.2f}</td>"
        f"<td>{r['prob_model']}%</td>"
        f"<td>{r['prob_implicita']}%</td>"
        f"<td>+{r['edge']}%</td>"
        f"<td>{r['stake']:.2f}€</td>"
        f"<td>{r['resultat_real']}</td>"
        f"<td>{ok}</td>"
        f"<td class='{cls}'>{r['profit']:+.2f} EUR</td>"
        "</tr>"
    )


def _team_roi_by_tipus(apostes: pd.DataFrame) -> pd.DataFrame:
    """ROI per equip apostat (Local -> home_team, Visitant -> away_team), desglossat
    per tipus d'aposta (Value Local / Contrarian Visitant), per temporada."""
    df = apostes.copy()
    df["equip"] = np.where(df["opcio"] == "Local", df["home_team"], df["away_team"])
    g = df.groupby(["season", "equip", "tipus"]).agg(
        apostes=("profit", "size"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    g["roi"] = g["profit"] / g["staked"] * 100
    return g


def _team_roi_chart(season: str, g: pd.DataFrame) -> str:
    sub = g[g["season"] == season]
    if sub.empty:
        return ""
    max_abs = max(sub["roi"].abs().max(), 1e-6)

    totals = sub.groupby("equip")["profit"].sum().sort_values(ascending=False)

    blocks = ""
    for equip in totals.index:
        team_rows = sub[sub["equip"] == equip].sort_values("tipus")
        team_staked = team_rows["staked"].sum()
        team_profit = team_rows["profit"].sum()
        team_roi = team_profit / team_staked * 100
        team_n = int(team_rows["apostes"].sum())

        bar_rows = ""
        for _, r in team_rows.iterrows():
            pct = min(abs(r["roi"]) / max_abs * 50, 50)
            pos = r["roi"] >= 0
            color = TIPUS_COLOR.get(r["tipus"], "#64748b")
            opacity = "1" if pos else ".45"
            style = f"left:50%;width:{pct:.2f}%" if pos else f"left:{50 - pct:.2f}%;width:{pct:.2f}%"
            bar_rows += f"""
        <div class="bar-row" title="{int(r['apostes'])} apostes · {r['profit']:+.2f} EUR">
          <div class="bar-label">{r['tipus']} <span class="bar-n">({int(r['apostes'])})</span></div>
          <div class="bar-track"><div class="bar-zero"></div><div class="bar-fill" style="{style};background:{color};opacity:{opacity}"></div></div>
          <div class="bar-value {'pos' if pos else 'neg'}">{r['roi']:+.1f}%</div>
        </div>"""

        blocks += f"""
      <div class="team-block">
        <div class="team-name">{equip}
          <span class="team-total {'pos' if team_roi >= 0 else 'neg'}">{team_roi:+.1f}% total · {team_n} apostes</span>
        </div>{bar_rows}
      </div>"""

    return f"""
  <div class="card">
    <h3>{season}</h3>
    <div class="bar-chart">{blocks}
    </div>
  </div>"""


def build_html(apostes: pd.DataFrame) -> str:
    if apostes.empty:
        resum_html = "<p>Cap aposta ha complert els criteris.</p>"
        jornada_blocks = ""
        team_roi_blocks = ""
        bankroll_chart_html = ""
    else:
        n          = len(apostes)
        staked     = apostes["stake"].sum()
        profit     = apostes["profit"].sum()
        roi        = profit / staked * 100
        encertades = int(apostes["guanyada"].sum())

        resum_html = f"""
    <table class="metrics">
      <tr><th>Apostes fetes</th><td>{n}</td></tr>
      <tr><th>Encertades</th><td>{encertades} ({encertades/n*100:.1f}%)</td></tr>
      <tr><th>Import apostat</th><td>{staked:.2f} EUR</td></tr>
      <tr><th>Guany/perdua net</th><td class="{'pos' if profit >= 0 else 'neg'}">{profit:+.2f} EUR</td></tr>
      <tr><th>ROI</th><td class="{'pos' if roi >= 0 else 'neg'}"><strong>{roi:+.2f}%</strong></td></tr>
    </table>"""

        tipus_rows = ""
        for tipus in apostes["tipus"].unique():
            sub = apostes[apostes["tipus"] == tipus]
            sub_roi = sub["profit"].sum() / sub["stake"].sum() * 100
            color = TIPUS_COLOR.get(tipus, "#64748b")
            tipus_rows += (
                "<tr>"
                f"<td><span class='tipus-badge' style='background:{color}'>{tipus}</span></td>"
                f"<td>{len(sub)}</td>"
                f"<td>{int(sub['guanyada'].sum())}</td>"
                f"<td>{sub['stake'].sum():.2f} EUR</td>"
                f"<td class=\"{'pos' if sub_roi >= 0 else 'neg'}\">{sub_roi:+.2f}%</td>"
                "</tr>"
            )
        resum_html += f"""
    <table style="margin-top:1rem">
      <tr><th>Tipus</th><th>Apostes</th><th>Encerts</th><th>Apostat</th><th>ROI</th></tr>
      {tipus_rows}
    </table>"""

        season_rows = ""
        for season in sorted(apostes["season"].unique()):
            sub = apostes[apostes["season"] == season]
            sub_n           = len(sub)
            sub_staked      = sub["stake"].sum()
            sub_profit      = sub["profit"].sum()
            sub_roi         = sub_profit / sub_staked * 100
            bankroll_final  = BANKROLL_INICIAL + sub_profit
            creixement      = sub_profit / BANKROLL_INICIAL * 100
            season_rows += (
                "<tr>"
                f"<td><strong>{season}</strong></td>"
                f"<td>{sub_n}</td>"
                f"<td>{int(sub['guanyada'].sum())} ({sub['guanyada'].mean()*100:.1f}%)</td>"
                f"<td>{sub_staked:.2f} EUR</td>"
                f"<td>{sub_profit:+.2f} EUR</td>"
                f"<td class=\"{'pos' if sub_roi >= 0 else 'neg'}\">{sub_roi:+.2f}%</td>"
                f"<td>{BANKROLL_INICIAL:.0f}€ → <strong>{bankroll_final:.2f}€</strong></td>"
                f"<td class=\"{'pos' if creixement >= 0 else 'neg'}\"><strong>{creixement:+.2f}%</strong></td>"
                "</tr>"
            )
        resum_html += f"""
    <table style="margin-top:1rem">
      <tr><th>Temporada</th><th>Apostes</th><th>Encerts</th><th>Apostat</th><th>Guany/perdua</th><th>ROI (s/apostat)</th><th>Pot (inici → final)</th><th>Creixement del pot</th></tr>
      {season_rows}
    </table>"""

        bankroll_chart_html = f"""
  <div class="card full">
    <h2>Evolució del pot per jornada (bankroll real, compost)</h2>
    <p class="subtitle" style="margin-bottom:1rem">
      Cada temporada es comença amb {BANKROLL_INICIAL:.0f}€ i l'stake de cada aposta es calcula amb Kelly
      sobre el pot que hi ha en aquell moment, modulat per una confiança INDEPENDENT per
      cada (equip, tipus d'aposta) -- no sobre un import fix.
    </p>
    <img src="data:image/png;base64,{_bankroll_chart_b64(apostes)}" style="width:100%;display:block;border-radius:8px"
         alt="Evolució del pot per jornada">
  </div>"""

        jornada_blocks = ""
        for season in sorted(apostes["season"].unique()):
            sub_season = apostes[apostes["season"] == season]
            jornada_blocks += f'<h2 class="season-title">Temporada {season}</h2>'

            for jornada in sorted(sub_season["jornada"].unique()):
                sub = sub_season[sub_season["jornada"] == jornada]
                j_n        = len(sub)
                j_staked   = sub["stake"].sum()
                j_profit   = sub["profit"].sum()
                j_roi      = j_profit / j_staked * 100
                j_ok       = int(sub["guanyada"].sum())
                j_bankroll = sub["bankroll_abans"].iloc[0]

                rows = "".join(_bet_row(r) for _, r in sub.sort_values("date").iterrows())
                jornada_blocks += f"""
  <div class="card">
    <h3>Jornada {jornada}
      <span class="badge" style="background:{_acc_color(j_roi)}">{j_roi:+.1f}% · {j_ok}/{j_n} · {j_profit:+.2f} EUR</span>
      <span class="badge" style="background:#334155;color:#e2e8f0">Pot {j_bankroll:.2f}€ → {j_bankroll + j_profit:.2f}€</span>
    </h3>
    <table>
      <tr>
        <th>Data</th><th>Tipus</th><th>Partit</th><th>Aposta</th><th>Casa</th><th>Quota</th>
        <th>Prob. meva</th><th>Prob. casa</th><th>Edge</th><th>Stake</th><th>Real</th><th>Encert</th><th>Profit</th>
      </tr>
      {rows}
    </table>
  </div>"""

        team_roi = _team_roi_by_tipus(apostes)
        team_roi_blocks = "".join(_team_roi_chart(season, team_roi) for season in sorted(apostes["season"].unique()))

    return f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>ROI {' / '.join(SEASONS)} — Value Local + Contrarian Visitant (confiança separada per tipus)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding-bottom: 3rem; }}
  h1 {{ text-align: center; padding: 2rem 1rem .3rem; font-size: 1.8rem; color: #f8fafc; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: .9rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1rem; padding: 0 2rem; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #1e2433; border-radius: 14px; padding: 1.4rem 1.6rem; border: 1px solid #2d3748; }}
  .card h2 {{ font-size: .85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1.1rem; }}
  .card h3 {{ font-size: .9rem; font-weight: 600; color: #f8fafc; margin-bottom: .8rem; display: flex; align-items: center; gap: .7rem; }}
  .season-title {{ color: #f8fafc; font-size: 1.2rem; margin: 1.2rem 0 -.2rem; padding-left: .2rem; }}
  .badge {{ font-size: .72rem; font-weight: 700; color: #0f1117; padding: .15rem .55rem; border-radius: 999px; }}
  .tipus-badge {{ font-size: .72rem; font-weight: 600; color: #0f1117; padding: .15rem .5rem; border-radius: 6px; white-space: nowrap; }}
  tr.tipus-row td {{ padding-left: .9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ color: #64748b; font-weight: 600; text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #2d3748; }}
  td {{ padding: .4rem .7rem; border-bottom: 1px solid #1a2030; font-variant-numeric: tabular-nums; }}
  .metrics th {{ color: #94a3b8; width: 40%; }}
  .pos {{ color: #22c55e; font-weight: 600; }}
  .neg {{ color: #ef4444; font-weight: 600; }}
  .bar-chart {{ display: flex; flex-direction: column; gap: .9rem; }}
  .team-block {{ display: flex; flex-direction: column; gap: .3rem; }}
  .team-name {{ font-size: .85rem; font-weight: 600; color: #f1f5f9; display: flex; align-items: baseline; gap: .6rem; }}
  .team-total {{ font-size: .72rem; font-weight: 600; }}
  .bar-row {{ display: grid; grid-template-columns: 190px 1fr 60px; align-items: center; gap: .6rem; font-size: .8rem; }}
  .bar-label {{ color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-n {{ color: #64748b; }}
  .bar-track {{ position: relative; height: 12px; background: #161b28; border-radius: 3px; }}
  .bar-zero {{ position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: #475569; }}
  .bar-fill {{ position: absolute; top: 0; bottom: 0; border-radius: 3px; }}
  .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
</style>
</head>
<body>
<h1>ROI {' / '.join(SEASONS)} — Value Local + Contrarian Visitant (confiança separada per tipus)</h1>
<p class="subtitle">
  <span class="tipus-badge" style="background:{TIPUS_COLOR['Value Local']}">Value Local</span>
  edge Local &gt;= {EDGE_MIN*100:.0f}% i prob. Local &gt;= {PROB_MIN['Local']*100:.0f}% ·
  <span class="tipus-badge" style="background:{TIPUS_COLOR['Contrarian Visitant']}">Contrarian Visitant</span>
  mercat sobrevalora el Local &gt;{TRIGGER_LOCAL_MIN*100:.0f}% i prob. Visitant &gt; {PROB_MIN_VISITANT*100:.0f}%
  <br>line shopping entre {len(CASES_ROI)} cases (descartades si &gt;{MAX_DIFF_VS_AVG:.0f}% vs. mitjana de mercat) ·
  jornades {JORNADA_MIN}-{JORNADA_MAX} · stake Kelly 1/{int(1/KELLY_FRACTION)}
  (pot inicial {BANKROLL_INICIAL:.0f}€/temporada, compost, topall {MAX_STAKE_PCT*100:.0f}% del pot actual/aposta,
  confiança independent per equip I tipus d'aposta, esmorteïda x{SEASON_CONF_DECAY} cada temporada)
</p>

<div class="grid">
  <div class="card">
    <h2>Resum global</h2>
    {resum_html}
  </div>
{bankroll_chart_html}
  <div class="card">
    <h2>ROI per equip i temporada</h2>
    {team_roi_blocks}
  </div>
{jornada_blocks}
</div>
</body>
</html>"""


if __name__ == "__main__":
    apostes = simular_totes()
    print_report(apostes)

    out_path = os.path.join(_ROOT, "estrategia2_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_html(apostes))
    print(f"\nHTML generat: {out_path}")
    webbrowser.open(f"file://{out_path}")
