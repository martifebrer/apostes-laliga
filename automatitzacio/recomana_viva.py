"""
Pas 4 del flux setmanal: aplica la MATEIXA lògica d'estrategia.py (Value
Local / Contrarian Visitant, filtre vs. mitjana de mercat, stake Kelly
modulat per la confiança per equip) a un partit que ENCARA NO S'HA JUGAT
-- per tant sense guanyada/profit.
 
Es fa servir Betfair (BFD) com a casa real per a l'aposta -- NO hi ha
line-shopping complet entre cases com al backtest d'estrategia.py.
El filtre MAX_DIFF_VS_AVG segueix comparant la quota triada contra la
mitjana de TOTES les cases que baixa actualitza_quotes.py (Avg), com a
control de qualitat perquè no es recomani una quota mal actualitzada.

FALLBACK (setembre 2026): The Odds API ha deixat de retornar el
bookmaker "betfair_sb_uk" per La Liga (comprovat: 0/20 partits durant
dues jornades seguides). Mentre Betfair no torni, `_casa_per_partit()`
fa servir Betway (BWY) NOMÉS per als partits concrets on Betfair no
tingui quota H/D/A completa -- no és line shopping (no es compara edge
entre cases), és una substitució 1-a-1 seguint FALLBACK_ORDER: primer
Betfair, i només si falta, Betway. Quan Betfair torni a publicar quota
amb normalitat, es pot treure "BWY" de FALLBACK_ORDER sense tocar res
més.

Reutilitza directament les constants ja calibrades a estrategia.py
(EDGE_MIN, PROB_MIN, MAX_DIFF_VS_AVG, TEAM_CONF_*, etc.).
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
 
from comparar_cases import normalitza_quotes
from estrategia import (
    ALL_LABELS, EDGE_MIN, PROB_MIN, TRIGGER_LOCAL_MIN, PROB_MIN_VISITANT,
    MAX_DIFF_VS_AVG, KELLY_FRACTION, MAX_STAKE_PCT, MIN_STAKE,
    TEAM_CONF_INICIAL,
)
from staking import kelly_stake
 
LIVE_CASES = [("BFD", "Betfair"), ("BWY", "Betway")]
FALLBACK_ORDER = ["BFD", "BWY"]   # ordre de preferència: Betfair primer, Betway només com a reserva
CASES_NOM = dict(LIVE_CASES)


def _casa_per_partit(r: pd.Series) -> tuple[str, str] | None:
    """Tria la casa a fer servir per a AQUEST partit concret, seguint
    FALLBACK_ORDER: la primera casa de la llista que tingui quota H/D/A
    completa. NO és line shopping -- no compara edge entre Betfair i
    Betway, només substitueix Betfair per Betway quan Betfair no tingui
    dades (vegeu docstring del mòdul). Retorna None si cap de les dues
    té quota per aquest partit."""
    for code in FALLBACK_ORDER:
        cols = [f"{code}_implicita_{lab}" for lab in ALL_LABELS]
        if all(c in r and pd.notna(r[c]) for c in cols):
            return code, CASES_NOM[code]
    return None
 
 
def _afegeix_implicites(df: pd.DataFrame) -> pd.DataFrame:
    """Per cada casa de LIVE_CASES (i 'Avg'), calcula la probabilitat
    implícita descomptant l'overround -- equivalent al que fa
    estrategia.carregar_quotes() per a les cases de football-data.co.uk."""
    df = df.copy()
    for code, _ in LIVE_CASES + [("Avg", "Mitjana mercat")]:
        cols = [f"{code}H", f"{code}D", f"{code}A"]
        if not all(c in df.columns for c in cols):
            continue
        mask = df[cols].notna().all(axis=1)
        if not mask.any():
            continue
        odds = df.loc[mask, cols].to_numpy(dtype=float)
        probs = normalitza_quotes(odds)
        for i, lab in enumerate(ALL_LABELS):
            df.loc[mask, f"{code}_quota_{lab}"] = odds[:, i]
            df.loc[mask, f"{code}_implicita_{lab}"] = probs[:, i]
    return df
 
 
def recomana(df_preds_amb_quotes: pd.DataFrame, team_confidence: dict, bankroll: float) -> pd.DataFrame:
    """`df_preds_amb_quotes`: sortida de predict_next.predict() (date,
    home_team, away_team, prob_Local/Empat/Visitant, prediccio) fusionada
    amb les columnes de quotes en viu (PINH/D/A...) que retorna
    actualitza_quotes.actualitza_quotes(). `team_confidence` NO es muta
    (a diferència d'estrategia.simular(): en viu la confiança només
    s'actualitza quan es coneix el resultat real, vegeu memoria.py).
 
    Retorna un DataFrame amb una fila per APOSTA RECOMANADA (pot ser buit
    si cap partit compleix els criteris)."""
    df = _afegeix_implicites(df_preds_amb_quotes)
    max_stake = bankroll * MAX_STAKE_PCT
 
    recomanacions = []
    n_fallback = 0
    n_sense_casa = 0
    for _, r in df.iterrows():
        prob_model_local    = r["prob_Local"] / 100
        prob_model_visitant = r["prob_Visitant"] / 100
 
        casa_triada = _casa_per_partit(r)
        if casa_triada is None:
            n_sense_casa += 1
        elif casa_triada[0] != FALLBACK_ORDER[0]:
            n_fallback += 1
 
        # ── Value Local ──
        candidat = None
        if casa_triada is not None:
            code, nom = casa_triada
            impl_col = f"{code}_implicita_Local"
            edge = prob_model_local - r[impl_col]
            if edge >= EDGE_MIN and prob_model_local >= PROB_MIN["Local"]:
                quota = r[f"{code}_quota_Local"]
                avg = r.get("Avg_quota_Local", np.nan)
                dins_marge = pd.isna(avg) or abs(quota - avg) / avg * 100 <= MAX_DIFF_VS_AVG
                if dins_marge:
                    candidat = {
                        "date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
                        "tipus": "Value Local", "opcio": "Local", "casa": nom,
                        "equip": r["home_team"], "prob": prob_model_local,
                        "prob_model": round(prob_model_local * 100, 1),
                        "prob_implicita": round(r[impl_col] * 100, 1),
                        "edge": round(edge * 100, 1),
                        "quota": quota,
                    }
 
        # ── Contrarian Visitant ──
        candidat_visitant = None
        if casa_triada is not None:
            code, nom = casa_triada
            impl_col_l = f"{code}_implicita_Local"
            impl_col_v = f"{code}_implicita_Visitant"
            diff = r[impl_col_l] - prob_model_local
            if diff > TRIGGER_LOCAL_MIN and prob_model_visitant > PROB_MIN_VISITANT:
                quota = r[f"{code}_quota_Visitant"]
                avg = r.get("Avg_quota_Visitant", np.nan)
                dins_marge = pd.isna(avg) or abs(quota - avg) / avg * 100 <= MAX_DIFF_VS_AVG
                if dins_marge:
                    candidat_visitant = {
                        "date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
                        "tipus": "Contrarian Visitant", "opcio": "Visitant", "casa": nom,
                        "equip": r["away_team"], "prob": prob_model_visitant,
                        "prob_model": round(prob_model_visitant * 100, 1),
                        "prob_implicita": round(r[impl_col_v] * 100, 1),
                        "edge": round(diff * 100, 1),
                        "quota": quota,
                    }
 
        for c in (candidat, candidat_visitant):
            if c is None:
                continue
            conf = team_confidence.get(c["equip"], TEAM_CONF_INICIAL)
            stake = kelly_stake(c["prob"], c["quota"], bankroll, KELLY_FRACTION, MIN_STAKE, max_stake)
            stake = min(stake * conf, max_stake)
            if stake <= 0:
                continue
            recomanacions.append({
                **{k: v for k, v in c.items() if k not in ("prob", "equip")},
                "equip": c["equip"],
                "team_conf": round(conf, 3),
                "stake": round(stake, 2),
            })
 
    if n_fallback:
        print(f"Avís: {n_fallback} partit(s) sense quota de {CASES_NOM[FALLBACK_ORDER[0]]}, "
              f"s'ha fet servir {CASES_NOM[FALLBACK_ORDER[1]]} com a reserva "
              f"(vegeu docstring de recomana_viva.py).")
    if n_sense_casa:
        print(f"Avís: {n_sense_casa} partit(s) sense quota de cap casa de FALLBACK_ORDER "
              f"({', '.join(CASES_NOM[c] for c in FALLBACK_ORDER)}) -- es descarten d'aquesta jornada.")
 
    return pd.DataFrame(recomanacions)
 