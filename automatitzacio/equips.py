"""
Equivalència de noms d'equip entre fonts:

  - "model"   : els que fa servir tot el pipeline (features.py, entrenament.py,
                predict_next.py, la_liga_2014_2025_all_matches_final.csv).
                Vénen d'Understat.
  - "odds_api": els que retorna The Odds API (hack_apostes/obtenir_quotes_live.py).
  - "sp1"     : els codis curts de football-data.co.uk (data/quotes_laliga.csv,
                data/SP1_2526.csv). Ja hi ha un mapeig model->sp1 a
                comparar_cases.NAME_MAP; aquí només s'reexporta.

odds_api_a_model() normalitza (sense accents, minúscules) i prova, per ordre:
un override manual explícit, després una coincidència de substring amb els
noms del model. Si no troba res, retorna el nom original de l'Odds API sense
tocar i n'avisa per pantalla -- millor un partit descartat més tard (sense
històric = sense features = predict_next.py ja el descarta sol) que un
mapeig equivocat i silenciós.
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import unicodedata

from comparar_cases import NAME_MAP as MODEL_A_SP1

SP1_A_MODEL = {v: k for k, v in MODEL_A_SP1.items()}

# Odds API -> model. Només cal on la normalització (sense accents/minúscules)
# no n'hi ha prou (abreviatures, "CF"/"CA" de mig, etc.)
ODDS_API_OVERRIDES = {
    "athletic bilbao":  "Athletic Club",
    "atletico madrid":  "Atletico Madrid",
    "ca osasuna":       "Osasuna",
    "elche cf":         "Elche",
    "deportivo la coruna": "Deportivo La Coruna",
    "sd huesca":        "SD Huesca",
    "sporting gijon":   "Sporting Gijon",
    "real racing club de santander": "Racing Santander",
    "racing santander":  "Racing Santander",
    "cd leganes":       "Leganes",
    "ud las palmas":    "Las Palmas",
}


def _normalitza(nom: str) -> str:
    sense_accents = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return sense_accents.lower().strip()


def _match_per_substring(norm: str, noms_model: list[str]) -> str | None:
    """Nucli comú de matching (normalitzat exacte, després substring parcial
    p.ex. "Alaves" conté/és contingut per "Deportivo Alaves"). `norm` ha
    d'estar ja normalitzat amb `_normalitza()`."""
    for candidat in noms_model:
        if _normalitza(candidat) == norm:
            return candidat
    for candidat in noms_model:
        cn = _normalitza(candidat)
        if norm in cn or cn in norm:
            return candidat
    return None


def odds_api_a_model(nom_odds_api: str, noms_model: list[str]) -> str:
    """Tradueix un nom d'equip de The Odds API al nom que fa servir el
    model. `noms_model` ha de ser la llista de noms d'equip que ja
    coneix el model (p.ex. de l'històric), per intentar el match per
    substring quan no hi ha override."""
    norm = _normalitza(nom_odds_api)

    if norm in ODDS_API_OVERRIDES:
        return ODDS_API_OVERRIDES[norm]

    match = _match_per_substring(norm, noms_model)
    if match is not None:
        return match

    print(f"Avís: no s'ha pogut mapejar l'equip '{nom_odds_api}' (The Odds API) a cap nom del model.")
    return nom_odds_api


def calendari_a_model(nom_calendari: str, noms_model: list[str]) -> str:
    """Tradueix un nom d'equip del calendari OFICIAL de jornades
    (data/calendari_2026_27.csv, vegeu automatitzacio/calendari.py) al nom
    que fa servir el model. No calen overrides manuals com a odds_api_a_model:
    el calendari ja fa servir noms complets (p.ex. "Deportivo Alaves",
    "Deportivo La Coruna") que el match per substring resol sol."""
    norm = _normalitza(nom_calendari)

    match = _match_per_substring(norm, noms_model)
    if match is not None:
        return match

    print(f"Avís: no s'ha pogut mapejar l'equip '{nom_calendari}' (calendari) a cap nom del model.")
    return nom_calendari


def model_a_sp1(nom_model: str) -> str:
    """Nom del model -> codi curt de football-data.co.uk (SP1). Si l'equip
    no és a MODEL_A_SP1, football-data fa servir el mateix nom que el
    model (veure comparar_cases.py)."""
    return MODEL_A_SP1.get(nom_model, nom_model)
