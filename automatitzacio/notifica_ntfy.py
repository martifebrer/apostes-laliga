"""
Notificacio push al mobil de les apostes recomanades, via ntfy.sh (gratuit,
sense compte, sense obrir ports) -- mateix patró que control_remot_ntfy.py
a l'altre projecte (hack apostes).

Subscriu-te al topic NTFY_TOPIC des de l'app de ntfy al mobil per rebre-les.
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import numpy as np
import pandas as pd
import requests

import config  # noqa: F401  (carrega .env abans de llegir NTFY_TOPIC_APOSTES)
from comparar_cases import normalitza_quotes
from recomana_viva import CASES_NOM, FALLBACK_ORDER

NTFY_BASE = "https://ntfy.sh"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC_APOSTES")   # opcional -- sense topic, no s'envia cap notificació

# Nom de casa (tal com surt a recomanacions["casa"]) -> prefix de columna a
# quotes_laliga.csv. Es reaprofita el mateix mapeig que recomana_viva.py
# (LIVE_CASES/FALLBACK_ORDER), per si la casa principal no té quota i s'ha
# fet servir la reserva -- vegeu docstring de recomana_viva.py.
NOM_A_PREFIX = {nom: code for code, nom in CASES_NOM.items()}
DETALL_COLS_BASE = ["date", "home_team", "away_team",
                     "prob_Local", "prob_Empat", "prob_Visitant"]
# Columnes de quotes de cada casa de reserva: NOMÉS s'afegeixen quan
# actualitza_quotes.py les ha creat (si una casa no ha retornat CAP dada
# aquesta setmana, ni tan sols crea les seves columnes H/D/A -- vegeu
# actualitza_quotes._fila_partit(), que només posa fila[f"{codi}H"] si
# aquella casa apareix als bookmakers d'algun partit).
_DETALL_COLS_CASES = [f"{code}{s}" for code in NOM_A_PREFIX.values() for s in ("H", "D", "A")]


def send_ntfy(message: str, title: str | None = None, priority: str = "default",
              topic: str = NTFY_TOPIC) -> None:
    """Publica una notificacio al topic indicat de ntfy.sh."""
    headers = {"Priority": priority}
    if title:
        headers["Title"] = title.encode("utf-8")
    resp = requests.post(
        f"{NTFY_BASE}/{topic}",
        data=message.encode("utf-8"),
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()


def implicites_casa_principal(row: pd.Series) -> tuple[float, float, float] | None:
    """Probabilitats implícites de la casa principal (FALLBACK_ORDER[0] a
    recomana_viva.py -- actualment Betway) explícitament, s'usi o no
    finalment per a cap recomanació d'aquest partit (s'usa a
    actualitza_jornada.print_resum() per a TOTS els partits, no només els
    recomanats). None si aquesta casa encara no ha publicat quotes per
    aquest partit."""
    return _implicites(row, FALLBACK_ORDER[0])


def _implicites(row: pd.Series, prefix: str) -> tuple[float, float, float] | None:
    """Probabilitats implícites (0-100) de la casa amb aquest prefix de
    columna, descomptant l'overround. None si aquella casa no té quota
    per aquest partit."""
    cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
    if any(pd.isna(row.get(c)) for c in cols):
        return None
    odds = np.array([[row[cols[0]], row[cols[1]], row[cols[2]]]], dtype=float)
    probs = normalitza_quotes(odds)[0] * 100
    return round(float(probs[0]), 1), round(float(probs[1]), 1), round(float(probs[2]), 1)


def implicites_casa_recomanada(row: pd.Series) -> tuple[float, float, float] | None:
    """Com _implicites(), però tria automàticament el prefix de columna
    segons `row['casa']` (el nom que recomana_viva._casa_per_partit() ja
    ha triat per aquesta aposta concreta -- la principal o, si no n'hi
    havia, la de reserva, vegeu FALLBACK_ORDER). Només té sentit per files
    que ja són una recomanació (tenen la columna 'casa'); per a la llista
    general de prediccions (totes, recomanades o no) es fa servir
    implicites_casa_principal."""
    prefix = NOM_A_PREFIX.get(row.get("casa"))
    if prefix is None:
        return None
    return _implicites(row, prefix)


_CASA_PRINCIPAL_NOM = CASES_NOM[FALLBACK_ORDER[0]]


def build_report(jornada: int, recomanacions: pd.DataFrame, df_odds: pd.DataFrame) -> str:
    """Text de la notificacio amb les apostes recomanades de la propera jornada:
    equips, les meves probabilitats i les de la casa realment usada per a
    cada aposta (la principal, o la de reserva si la principal no tenia
    quota -- vegeu recomana_viva._casa_per_partit()/FALLBACK_ORDER), quota
    d'aquella casa, tipus d'aposta i import a jugar."""
    if recomanacions.empty:
        return f"Jornada {jornada}: cap partit compleix els criteris aquesta setmana."

    detall_cols = DETALL_COLS_BASE + [c for c in _DETALL_COLS_CASES if c in df_odds.columns]
    r = recomanacions.merge(df_odds[detall_cols], on=["date", "home_team", "away_team"], how="left")

    linies = [f"Jornada {jornada}: {len(r)} aposta(es) recomanada(es)"]
    for _, row in r.iterrows():
        data = pd.to_datetime(row["date"]).strftime("%d/%m")
        casa = row.get("casa", _CASA_PRINCIPAL_NOM)
        implicites = implicites_casa_recomanada(row)
        if implicites is None:
            casa_line = f"Prob. {casa}: no disponible"
        else:
            local, empat, visitant = implicites
            casa_line = (f"Prob. {casa}: Local {local:.1f}%  Empat {empat:.1f}%  "
                          f"Visitant {visitant:.1f}%")
        avis_fallback = f"  (reserva, {_CASA_PRINCIPAL_NOM} no tenia quota)" if casa != _CASA_PRINCIPAL_NOM else ""

        linies.append(
            f"\n{row['home_team']} vs {row['away_team']} ({data})\n"
            f"Tipus: {row['tipus']} -> {row['opcio']}\n"
            f"La meva prob.: Local {row['prob_Local']:.1f}%  Empat {row['prob_Empat']:.1f}%  "
            f"Visitant {row['prob_Visitant']:.1f}%\n"
            f"{casa_line}\n"
            f"Quota {casa}: {row['quota']}{avis_fallback}\n"
            f"Aposta: {row['stake']:.2f} EUR"
        )
    return "\n".join(linies)


def notifica_recomanacions(jornada: int, recomanacions: pd.DataFrame, df_odds: pd.DataFrame) -> None:
    if not NTFY_TOPIC:
        print("Avis: NTFY_TOPIC_APOSTES no definit, no s'envia cap notificacio ntfy.")
        return
    report = build_report(jornada, recomanacions, df_odds)
    title = (f"La Liga J{jornada}: {len(recomanacions)} aposta(es)"
             if not recomanacions.empty else f"La Liga J{jornada}: sense apostes")
    try:
        send_ntfy(report, title=title)
        print(f"Notificacio enviada a ntfy.sh/{NTFY_TOPIC}")
    except requests.RequestException as e:
        print(f"Avis: no s'ha pogut enviar la notificacio ntfy ({e}).")