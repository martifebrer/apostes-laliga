"""
Bootstrap de l'edge sobre els resultats de sense_estrategia.py (aposta fixa
1€, sense Kelly ni bankroll compost -- vegeu docstring d'aquell mòdul).

Com que l'stake NO depèn del resultat de les apostes anteriors (a diferència
d'estrategia.py/estrategia2.py, on el Kelly sobre bankroll compost i la
confiança per equip encadenen una aposta amb la següent), cada fila de
`profit` es pot tractar com una observació independent i remostrejar-la
directament (bootstrap no paramètric clàssic), sense necessitat de blocs
per jornada.

Per cada subconjunt (global, i per tipus d'aposta per separat):
  1. Es remostregen amb reemplaçament les N apostes, B cops.
  2. Es calcula el ROI mitjà (%) de cada remostreig.
  3. Es reporta l'interval de confiança percentil 95% i la proporció de
     remostreigs amb ROI <= 0 (estimació bootstrap de "quin % de les vegades
     l'edge no s'hauria distingit de zero amb aquesta mida de mostra").
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

from sense_estrategia import simular_totes

N_BOOT = 10_000
SEED   = 42


def bootstrap_roi(profits: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Retorna un array de mida n_boot amb el ROI mitjà (%) de cada remostreig
    amb reemplaçament de `profits` (stake=1€ per aposta, així que ROI mitjà
    == mean(profit)*100)."""
    rng = np.random.default_rng(seed)
    n = len(profits)
    idx = rng.integers(0, n, size=(n_boot, n))
    return profits[idx].mean(axis=1) * 100


def bootstrap_roi_per_equip(profits_per_equip: list[np.ndarray], n_boot: int = N_BOOT,
                             seed: int = SEED) -> np.ndarray:
    """Bootstrap PER BLOCS D'EQUIP: en lloc de remostrejar apostes individuals,
    remostreja EQUIPS sencers (amb reemplaçament, mateix nombre d'equips que
    n'hi ha) i concatena totes les apostes d'aquell equip a cada remostreig.

    Motiu: si el ROI ve concentrat en pocs equips (p.ex. un equip amb un
    parell d'apostes de quota molt alta que encerten), el bootstrap per
    aposta individual (bootstrap_roi) no ho detecta -- tracta cada aposta
    com intercanviable amb qualsevol altra, vingui de l'equip que vingui.
    Remostrejant per equip, un equip "sortós" pot aparèixer 0, 1 o diverses
    vegades al remostreig, així que si el ROI depèn massa d'ell l'IC
    s'eixamplarà molt més que amb el bootstrap per aposta."""
    rng = np.random.default_rng(seed)
    n_equips = len(profits_per_equip)
    boot_rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_equips, size=n_equips)
        mostra = np.concatenate([profits_per_equip[i] for i in idx])
        boot_rois[b] = mostra.mean() * 100
    return boot_rois


def report(nom: str, profits: np.ndarray) -> None:
    n = len(profits)
    if n == 0:
        print(f"\n=== {nom} === (cap aposta, s'omet)")
        return

    roi_observat = profits.mean() * 100
    boot = bootstrap_roi(profits)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_no_edge = (boot <= 0).mean() * 100

    print(f"\n=== {nom} ===")
    print(f"Apostes                 : {n}")
    print(f"ROI observat            : {roi_observat:+.2f}%")
    print(f"IC 95% bootstrap (ROI)  : [{ci_low:+.2f}%, {ci_high:+.2f}%]")
    print(f"P(ROI <= 0) bootstrap   : {p_no_edge:.1f}%")
    if ci_low > 0:
        print("-> L'IC no inclou el 0: l'edge és estadísticament distingible de zero.")
    elif ci_high < 0:
        print("-> L'IC és tot negatiu: l'estratègia perd de forma distingible de zero.")
    else:
        print("-> L'IC inclou el 0: amb aquesta mida de mostra, no es pot descartar que l'edge sigui soroll.")


def report_per_equip(nom: str, apostes: pd.DataFrame) -> None:
    """Igual que report(), però remostrejant per EQUIP (bootstrap_roi_per_equip)
    en lloc de per aposta individual -- vegeu docstring d'aquella funció."""
    df = apostes.copy()
    df["equip"] = np.where(df["opcio"] == "Local", df["home_team"], df["away_team"])

    taula = df.groupby("equip")["profit"].agg(apostes="size", profit_total="sum")
    taula["roi"] = taula["profit_total"] / taula["apostes"] * 100
    taula = taula.sort_values("profit_total", ascending=False)

    print(f"\n=== {nom} - desglossat per equip ===")
    print(taula.to_string(float_format=lambda x: f"{x:.2f}"))

    profits_per_equip = [g["profit"].to_numpy() for _, g in df.groupby("equip")]
    roi_observat = df["profit"].mean() * 100
    boot = bootstrap_roi_per_equip(profits_per_equip)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_no_edge = (boot <= 0).mean() * 100

    print(f"\nEquips diferents        : {len(profits_per_equip)}")
    print(f"ROI observat            : {roi_observat:+.2f}%")
    print(f"IC 95% bootstrap/equip  : [{ci_low:+.2f}%, {ci_high:+.2f}%]")
    print(f"P(ROI <= 0) bootstrap   : {p_no_edge:.1f}%")
    if ci_low > 0:
        print("-> L'IC no inclou el 0 ni remostrejant per equip: l'edge no sembla dependre de pocs equips concrets.")
    else:
        print("-> L'IC inclou el 0: consistent amb (o encara més ample que) el bootstrap per aposta individual.")


def main() -> None:
    apostes = simular_totes()
    if apostes.empty:
        print("Cap aposta ha complert els criteris de cap dels dos tipus.")
        return

    report("Global (Value Local + Contrarian Visitant)", apostes["profit"].to_numpy())

    for tipus in apostes["tipus"].unique():
        sub = apostes[apostes["tipus"] == tipus]
        report(tipus, sub["profit"].to_numpy())

    for season in sorted(apostes["season"].unique()):
        sub = apostes[apostes["season"] == season]
        report(f"Temporada {season}", sub["profit"].to_numpy())

    report_per_equip("Global (Value Local + Contrarian Visitant)", apostes)
    for tipus in apostes["tipus"].unique():
        sub = apostes[apostes["tipus"] == tipus]
        report_per_equip(tipus, sub)

    # ── Detall per als dos equips amb més guany net acumulat (vegeu taula per
    # equip): aquí ja no té sentit el bootstrap per BLOCS d'equip (només n'hi
    # ha un), així que es fa servir bootstrap_roi (per aposta individual, com
    # a report()) sobre les seves apostes en concret. ──
    df = apostes.copy()
    df["equip"] = np.where(df["opcio"] == "Local", df["home_team"], df["away_team"])

    for equip in ["Atletico Madrid", "Barcelona"]:
        sub = df[df["equip"] == equip]
        report(f"{equip} (tots els tipus)", sub["profit"].to_numpy())
        for tipus in sub["tipus"].unique():
            sub_tipus = sub[sub["tipus"] == tipus]
            report(f"{equip} - {tipus}", sub_tipus["profit"].to_numpy())


if __name__ == "__main__":
    main()
