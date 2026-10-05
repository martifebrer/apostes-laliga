"""
Llegeix/escriu el fitxer Excel de "memòria" (per defecte
data/jornades_recomanacions.xlsx), amb tres fulls:

  - Apostes    : una fila per aposta recomanada (jugada o pendent). La
                 columna "confirmada" distingeix les apostes recomanades
                 EN VIU que encara no s'ha confirmat si es van jugar de
                 veritat (buida = pendent de confirmar) de les que sí
                 (True) -- vegeu pendents_per_confirmar()/confirma_aposta()/
                 esborra_aposta() i actualitza_jornada.py. Les apostes
                 sembrades pel backtest (inicialitza_memoria.py) sempre
                 porten confirmada=True, ja tenen resultat conegut.
  - Confianca  : confiança acumulada per equip (team_confidence
                 d'estrategia.py, vegeu TEAM_CONF_* i el seu docstring).
  - Resum      : bankroll_actual (el pot real, viu) i quan es va
                 actualitzar per última vegada.

No es crea sol la primera vegada -- cal executar inicialitza_memoria.py
abans, que la genera a partir del backtest d'estrategia.py (mateix punt
de partida que ja s'ha validat amb 3 temporades).
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

from estrategia import (
    BANKROLL_INICIAL, TEAM_CONF_INICIAL, TEAM_CONF_K_UP, TEAM_CONF_K_DOWN,
    TEAM_CONF_MIN, TEAM_CONF_MAX,
)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORIA_PATH = os.path.join(BASE_DIR, "data", "jornades_recomanacions.xlsx")

APOSTES_COLS = ["date", "home_team", "away_team", "tipus", "opcio", "casa",
                 "prob_model", "prob_implicita", "edge", "quota", "team_conf",
                 "stake", "resultat_real", "guanyada", "profit", "confirmada"]
CLAU_APOSTA = ["date", "home_team", "away_team", "tipus"]


def existeix(path: str = MEMORIA_PATH) -> bool:
    return os.path.exists(path)


def llegeix_memoria(path: str = MEMORIA_PATH) -> tuple[pd.DataFrame, dict, float]:
    if not existeix(path):
        raise FileNotFoundError(
            f"No existeix {path}. Cal executar primer inicialitza_memoria.py "
            "per crear-lo (llavor: backtest d'estrategia.py amb 3 temporades)."
        )
    apostes = pd.read_excel(path, sheet_name="Apostes")
    if not apostes.empty:
        apostes["date"] = pd.to_datetime(apostes["date"])
        if "confirmada" not in apostes.columns:
            # Migració: fitxers antics sense aquesta columna -- les apostes
            # ja resoltes es donen per confirmades (ja van comptar en el seu
            # moment), les que encara estan pendents de resultat es deixen
            # pendents de confirmar (vegeu pendents_per_confirmar()).
            apostes["confirmada"] = apostes["resultat_real"].notna().astype(object)
            apostes.loc[apostes["resultat_real"].isna(), "confirmada"] = np.nan
        else:
            # openpyxl/pandas poden llegir aquesta columna com float64 (True
            # -> 1.0, buit -> NaN) un cop ha passat per un cicle
            # escriu_memoria()/llegeix_memoria() -- es normalitza sempre a
            # object perquè confirma_aposta() hi pugui assignar True sense
            # que pandas es queixi de "Invalid value for dtype 'float64'".
            apostes["confirmada"] = apostes["confirmada"].astype(object)
            apostes.loc[apostes["confirmada"] == 1.0, "confirmada"] = True

    conf_df = pd.read_excel(path, sheet_name="Confianca")
    confidence = dict(zip(conf_df["equip"], conf_df["confianca"]))

    resum = pd.read_excel(path, sheet_name="Resum")
    bankroll = float(resum.loc[0, "bankroll_actual"])

    return apostes, confidence, bankroll


def escriu_memoria(apostes: pd.DataFrame, confidence: dict, bankroll: float, path: str = MEMORIA_PATH) -> None:
    apostes = apostes.sort_values("date").reset_index(drop=True) if not apostes.empty else apostes
    conf_df = pd.DataFrame(sorted(confidence.items()), columns=["equip", "confianca"])
    resum = pd.DataFrame([{
        "bankroll_actual": round(bankroll, 2),
        "ultima_actualitzacio": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        apostes[APOSTES_COLS].to_excel(writer, sheet_name="Apostes", index=False)
        conf_df.to_excel(writer, sheet_name="Confianca", index=False)
        resum.to_excel(writer, sheet_name="Resum", index=False)


def afegeix_recomanacions(recomanacions: pd.DataFrame, path: str = MEMORIA_PATH) -> None:
    """Afegeix les noves recomanacions (sortida de recomana_viva.recomana,
    sense resultat encara) al full Apostes, amb upsert per CLAU_APOSTA
    (si es torna a córrer la mateixa setmana amb quotes actualitzades)."""
    apostes, confidence, bankroll = llegeix_memoria(path)

    if recomanacions.empty:
        print("Sense recomanacions noves per apuntar a la memòria.")
        return

    noves = recomanacions.copy()
    noves["resultat_real"] = np.nan
    noves["guanyada"] = np.nan
    noves["profit"] = np.nan
    noves["confirmada"] = np.nan   # pendent de confirmar si s'ha jugat de veritat, vegeu pendents_per_confirmar()
    noves = noves[APOSTES_COLS]

    if not apostes.empty:
        clau_noves = set(zip(*[noves[c] for c in CLAU_APOSTA]))
        ja_hi_es = apostes.apply(lambda r: tuple(r[c] for c in CLAU_APOSTA) in clau_noves, axis=1)
        apostes = pd.concat([apostes[~ja_hi_es], noves], ignore_index=True)
    else:
        apostes = noves

    escriu_memoria(apostes, confidence, bankroll, path)
    print(f"{len(noves)} recomanació(ns) nova(es) apuntada(es) a {path}")


def backfill_resultats(history_path: str, path: str = MEMORIA_PATH) -> None:
    """Per les apostes CONFIRMADES (confirmada=True) i pendents
    (resultat_real buit) que ja s'han jugat (és a dir, ja són a
    l'històric), omple resultat_real/guanyada/profit i actualitza la
    confiança per equip i el bankroll -- MATEIXA fórmula que
    estrategia.simular() (passos 3/4), aplicada quan es coneix el resultat
    en comptes de dins del backtest.

    Les apostes encara SENSE confirmar (confirmada buida) es deixen
    intactes -- vegeu pendents_per_confirmar()/confirma_aposta(): primer
    cal que l'usuari confirmi que les va jugar de veritat."""
    apostes, confidence, bankroll = llegeix_memoria(path)
    if apostes.empty:
        return

    confirmades = apostes["confirmada"].fillna(False).astype(bool)
    pendents = apostes["resultat_real"].isna() & confirmades
    if not pendents.any():
        return

    hist = pd.read_csv(history_path)[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    hist["resultat_real"] = np.select(
        [hist["home_goals"] > hist["away_goals"], hist["home_goals"] == hist["away_goals"]],
        ["Local", "Empat"], default="Visitant",
    )

    apostes = apostes.copy()
    for col in ("resultat_real", "guanyada", "profit"):
        apostes[col] = apostes[col].astype(object)
    apostes["_date_norm"] = pd.to_datetime(apostes["date"]).dt.normalize()
    match = apostes.loc[pendents, ["_date_norm", "home_team", "away_team"]].merge(
        hist[["date", "home_team", "away_team", "resultat_real"]],
        left_on=["_date_norm", "home_team", "away_team"],
        right_on=["date", "home_team", "away_team"], how="left",
    )
    apostes.loc[pendents, "resultat_real"] = match["resultat_real"].to_numpy()
    apostes = apostes.drop(columns="_date_norm")

    resolts = pendents & apostes["resultat_real"].notna()
    if not resolts.any():
        print("Cap aposta pendent s'ha pogut resoldre encara (partits per jugar).")
        return

    apostes.loc[resolts, "guanyada"] = apostes.loc[resolts, "opcio"] == apostes.loc[resolts, "resultat_real"]
    guanyades = resolts & (apostes["guanyada"] == True)   # noqa: E712 (evita ~ sobre columna object amb NaN)
    perdudes  = resolts & (apostes["guanyada"] == False)  # noqa: E712
    apostes.loc[guanyades, "profit"] = (apostes.loc[guanyades, "quota"] - 1) * apostes.loc[guanyades, "stake"]
    apostes.loc[perdudes, "profit"]  = -apostes.loc[perdudes, "stake"]
    apostes["profit"] = apostes["profit"].round(2)

    profit_total = apostes.loc[resolts, "profit"].sum()
    bankroll += profit_total

    for _, r in apostes.loc[resolts].iterrows():
        prob = r["prob_model"] / 100.0
        equip = r["home_team"] if r["opcio"] == "Local" else r["away_team"]
        prev = confidence.get(equip, TEAM_CONF_INICIAL)
        retorn_esperat = prob * r["quota"] - 1.0
        retorn_real = (r["quota"] - 1.0) if r["guanyada"] else -1.0
        sorpresa = retorn_real - retorn_esperat
        k = TEAM_CONF_K_UP if sorpresa > 0 else TEAM_CONF_K_DOWN
        nova = prev + k * sorpresa
        confidence[equip] = min(max(nova, TEAM_CONF_MIN), TEAM_CONF_MAX)

    escriu_memoria(apostes, confidence, bankroll, path)
    print(f"{int(resolts.sum())} aposta(es) resolta(es): profit {profit_total:+.2f} EUR "
          f"-> bankroll {bankroll:.2f} EUR.")


def pendents_per_confirmar(history_path: str, path: str = MEMORIA_PATH) -> pd.DataFrame:
    """Apostes recomanades EN VIU que encara no s'ha confirmat si es van
    jugar de veritat (confirmada buida) i que el partit ja s'ha jugat
    (és a dir, ja té resultat disponible a l'històric -- normalment just
    després de córrer actualitza_historic.py). Són les que
    actualitza_jornada.py pregunta a l'usuari una a una."""
    apostes, _, _ = llegeix_memoria(path)
    if apostes.empty:
        return apostes

    per_confirmar = apostes["confirmada"].isna()
    if not per_confirmar.any():
        return apostes.iloc[0:0]

    hist = pd.read_csv(history_path)[["date", "home_team", "away_team"]].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    ja_jugats = set(zip(hist["date"], hist["home_team"], hist["away_team"]))

    candidates = apostes[per_confirmar].copy()
    candidates["_date_norm"] = pd.to_datetime(candidates["date"]).dt.normalize()
    jugat = candidates.apply(
        lambda r: (r["_date_norm"], r["home_team"], r["away_team"]) in ja_jugats, axis=1
    )
    return candidates[jugat].drop(columns="_date_norm").reset_index(drop=True)


def _localitza_aposta(apostes: pd.DataFrame, date, home_team: str, away_team: str, tipus: str) -> pd.Series:
    mask = (
        (apostes["date"] == pd.to_datetime(date)) & (apostes["home_team"] == home_team)
        & (apostes["away_team"] == away_team) & (apostes["tipus"] == tipus)
    )
    if not mask.any():
        raise ValueError(f"No es troba cap aposta {home_team} vs {away_team} ({tipus}, {date}) a la memòria.")
    return mask


def confirma_aposta(date, home_team: str, away_team: str, tipus: str, path: str = MEMORIA_PATH) -> None:
    """Marca una aposta recomanada com a realment jugada (confirmada=True)
    -- backfill_resultats() ja la pot resoldre normalment (resultat,
    bankroll, confiança per equip)."""
    apostes, confidence, bankroll = llegeix_memoria(path)
    mask = _localitza_aposta(apostes, date, home_team, away_team, tipus)
    apostes.loc[mask, "confirmada"] = True
    escriu_memoria(apostes, confidence, bankroll, path)


def esborra_aposta(date, home_team: str, away_team: str, tipus: str, path: str = MEMORIA_PATH) -> None:
    """Elimina una aposta de la memòria SENSE tocar bankroll ni confiança
    -- per a apostes recomanades que finalment NO es van jugar de veritat.
    Com que backfill_resultats() només actua sobre apostes confirmades,
    aquestes mai han arribat a comptar enlloc: esborrar-les simplement les
    treu de la llista."""
    apostes, confidence, bankroll = llegeix_memoria(path)
    mask = _localitza_aposta(apostes, date, home_team, away_team, tipus)
    escriu_memoria(apostes[~mask].reset_index(drop=True), confidence, bankroll, path)
    print(f"Aposta esborrada: {home_team} vs {away_team} ({tipus}, {pd.to_datetime(date).strftime('%Y-%m-%d')}).")

