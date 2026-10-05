"""
Script "d'una tecla" per a la jornada. Cal haver corregut UNA VEGADA
inicialitza_memoria.py abans (crea data/jornades_recomanacions.xlsx).

Cada cop que el corres fa, EN AQUEST ORDRE:

  1. Històric (actualitza_historic.py, pas 2) -- fa scraping d'Understat per
     trobar partits ja jugats que encara no són a l'històric
     (la_liga_2014_2025_all_matches_final.csv) i els hi afegeix (xG, PPDA,
     deep, xPts inclosos).

  2. Confirmació de les apostes recomanades -- per a cada aposta que es va
     recomanar i que encara no s'ha confirmat si es va jugar de veritat
     (full "Apostes", columna "confirmada" buida) i que, gràcies al pas 1,
     ja té resultat disponible, es pregunta a l'usuari si la va fer. Les que
     es confirmen passen pel pas 3 (memoria.backfill_resultats: resultat
     real, bankroll, confiança per equip) i pel pas 5 (report_seguiment.py).
     Les que NO es confirmen s'esborren de la memòria (memoria.esborra_aposta)
     -- no compten ni al seguiment ni a la confiança de l'equip en qüestió,
     perquè no s'han jugat de veritat.

  3. Quotes (actualitza_quotes.py, pas 1) -- es pregunta a l'usuari de quina
     jornada vol les quotes (per defecte se'n suggereix una fent servir el
     calendari OFICIAL de la temporada, automatitzacio/calendari.py -- més
     fiable que l'antiga estimació "partits jugats // 10 + 1") i es baixen
     amb The Odds API només els partits d'aquesta jornada concreta.

  4. Recomanació (recomana_viva.py, pas 4) -- amb el mateix model fet servir
     a les simulacions d'estrategia.py (model_rf_eval.joblib, NO es
     reentrena) i les features ja actualitzades, calcula l'edge de cada
     partit de la jornada triada amb la MATEIXA lògica d'estrategia.py
     (Value Local / Contrarian Visitant, Kelly modulat per la confiança de
     cada equip), ho treu per pantalla, ho apunta al full "Apostes" de la
     memòria (pendent de confirmar la setmana vinent) i ho notifica per
     ntfy.sh.

Flux d'ús setmanal: només cal córrer `python actualitza_jornada.py`.
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import pandas as pd

from actualitza_historic import actualitza_historic
from actualitza_quotes import actualitza_quotes
from calendari import jornada_actual_suggerida
from estrategia import JORNADA_MIN, JORNADA_MAX
from memoria import (
    MEMORIA_PATH, afegeix_recomanacions, backfill_resultats, confirma_aposta,
    esborra_aposta, llegeix_memoria, pendents_per_confirmar,
)
from notifica_ntfy import implicites_casa_principal, notifica_recomanacions
from predict_next import HISTORY_PATH, predict
from recomana_viva import CASES_NOM, FALLBACK_ORDER, recomana
from report_seguiment import generate_report as genera_report_seguiment

_CASA_PRINCIPAL_NOM = CASES_NOM[FALLBACK_ORDER[0]]

NEXT_FIXTURES_PATH = os.path.join(_ROOT, "data", "next_fixtures.csv")

_RESPOSTES_SI = {"s", "si", "sí", "y", "yes"}


def print_resum(df_odds: pd.DataFrame, recomanacions: pd.DataFrame) -> None:
    print(f"\n=== Apostes recomanades ({len(recomanacions)}) ===\n")
    if recomanacions.empty:
        print("Cap partit compleix els criteris de Value Local / Contrarian Visitant aquesta jornada.")
    else:
        for _, r in recomanacions.iterrows():
            data = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
            print(f"{data}  {r['home_team']} vs {r['away_team']}  ->  {r['tipus']}: {r['opcio']} "
                  f"@ {r['quota']} ({r['casa']})  edge {r['edge']:+.1f}%  "
                  f"conf.equip {r['team_conf']:.2f}x  stake {r['stake']:.2f} EUR")

    print(f"\n=== Prediccions ({len(df_odds)} partit(s)) ===\n")
    for _, r in df_odds.iterrows():
        data = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        print(f"{data}  {r['home_team']} vs {r['away_team']}  "
              f"Local {r['prob_Local']:5.1f}%  Empat {r['prob_Empat']:5.1f}%  "
              f"Visitant {r['prob_Visitant']:5.1f}%  -> {r['prediccio']}")
        implicites = implicites_casa_principal(r)
        if implicites is None:
            print(f"   Prob. implícita {_CASA_PRINCIPAL_NOM}:  Local   ---  Empat   ---  Visitant   ---  "
                  "(encara no hi ha quotes disponibles per aquest partit)\n")
        else:
            local, empat, visitant = implicites
            print(f"   Prob. implícita {_CASA_PRINCIPAL_NOM}:  Local {local:5.1f}%  Empat {empat:5.1f}%  "
                  f"Visitant {visitant:5.1f}%\n")


def confirma_apostes_jugades() -> None:
    """Pregunta a l'usuari, una a una, quines de les apostes recomanades que
    ja s'han pogut jugar (vegeu memoria.pendents_per_confirmar) va fer de
    veritat. Les confirmades es resolen amb el pas 3 (backfill_resultats) i
    es reflecteixen al pas 5 (report de seguiment); la resta s'esborren de
    la memòria i no compten enlloc."""
    pendents = pendents_per_confirmar(HISTORY_PATH, MEMORIA_PATH)
    if pendents.empty:
        print("Cap aposta pendent de confirmar.")
        return

    for _, r in pendents.iterrows():
        data = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        pregunta = (f"Vas fer l'aposta {r['home_team']} vs {r['away_team']} ({data}) "
                    f"-- {r['tipus']}: {r['opcio']} @ {r['quota']} ({r['casa']})? [s/N] ")
        resposta = input(pregunta).strip().lower()
        if resposta in _RESPOSTES_SI:
            confirma_aposta(r["date"], r["home_team"], r["away_team"], r["tipus"], MEMORIA_PATH)
            print("  -> confirmada, es resoldrà amb el resultat real.")
        else:
            esborra_aposta(r["date"], r["home_team"], r["away_team"], r["tipus"], MEMORIA_PATH)

    print("\n--- Pas 3: memòria (resultats + confiança + bankroll) ---")
    backfill_resultats(HISTORY_PATH)

    print("\n--- Pas 5: report de seguiment ---")
    genera_report_seguiment(open_browser=False)


_RESPOSTES_CAP = {"cap", "no", "n", "0", "-"}


def demana_jornada() -> int | None:
    """None vol dir que l'usuari no vol fer scraping de quotes noves aquesta
    vegada (només confirmar/esborrar les apostes de la setmana anterior)."""
    suggerida = jornada_actual_suggerida()
    resposta = input(f"De quina jornada vols quotes noves? [{suggerida}] "
                      "(\"cap\" per no demanar-ne cap ara) ").strip().lower()
    if resposta in _RESPOSTES_CAP:
        return None
    return int(resposta) if resposta else suggerida


def main() -> None:
    print("--- Pas 2: històric (Understat) ---")
    actualitza_historic()

    print("\n--- Confirmació de les apostes recomanades ---")
    confirma_apostes_jugades()

    print("\n--- Pas 1: quotes de la jornada ---")
    jornada = demana_jornada()
    if jornada is None:
        print("D'acord, no es demanen quotes noves aquesta vegada. Fi.")
        return
    fixtures_odds = actualitza_quotes(jornada)
    if fixtures_odds.empty:
        print("Encara no hi ha quotes disponibles per aquesta jornada. Fi.")
        return
    fixtures_odds[["date", "home_team", "away_team"]].to_csv(NEXT_FIXTURES_PATH, index=False)

    print("\n--- Pas 4: recomanació de la jornada ---")
    preds = predict(HISTORY_PATH, NEXT_FIXTURES_PATH)
    df_odds = preds.merge(fixtures_odds, on=["date", "home_team", "away_team"], how="left")

    if not (JORNADA_MIN <= jornada <= JORNADA_MAX):
        print(f"Jornada {jornada} -- fora de la finestra d'apostes "
              f"({JORNADA_MIN}-{JORNADA_MAX}, vegeu estrategia.py). Es mostren "
              f"les prediccions però no es recomana cap aposta.")
        print_resum(df_odds, pd.DataFrame())
        notifica_recomanacions(jornada, pd.DataFrame(), df_odds)
        return

    _, confidence, bankroll = llegeix_memoria(MEMORIA_PATH)
    recomanacions = recomana(df_odds, confidence, bankroll)

    print_resum(df_odds, recomanacions)
    afegeix_recomanacions(recomanacions, MEMORIA_PATH)
    notifica_recomanacions(jornada, recomanacions, df_odds)


if __name__ == "__main__":
    main()
