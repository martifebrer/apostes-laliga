"""
Optimitzador de paràmetres d'estrategia.py per a la temporada 2025-26.

Objectiu: maximitzar el POT FINAL de la 2025-26 (bankroll real i compost,
començant amb BANKROLL_INICIAL, exactament la mateixa mecànica que
estrategia.simular).

Paràmetres que s'exploren (grid search exhaustiu):
  - EDGE_MIN            (Value Local: edge mínim sobre la millor casa)
  - PROB_MIN_LOCAL      (Value Local: prob. mínima del model)
  - TRIGGER_LOCAL_MIN   (Contrarian: sobrevaloració mínima del Local)
  - PROB_MIN_VISITANT   (Contrarian: prob. mínima de Visitant)
  - TEAM_CONF_K_UP      (confiança per equip: velocitat de pujada)
  - TEAM_CONF_K_DOWN    (confiança per equip: velocitat de baixada)

Es mantenen FIXOS: staking (KELLY_FRACTION=0.25, MAX_STAKE_PCT=0.20),
filtres (MAX_DIFF_VS_AVG=8%, jornades 4-34) i límits de confiança (0.5-2.0).

Com que la part cara (features + predict del model + merge de quotes) no
depèn de cap paràmetre, es precomputa UNA vegada; la simulació per combinació
és un bucle lleuger sobre les jornades. Abans d'optimitzar es verifica que la
simulació ràpida reprodueix EXACTAMENT estrategia.simular amb els paràmetres
actuals del fitxer.

AVÍS: optimitzar el pot final d'una sola temporada és molt propens al
sobreajust: els valors "òptims" per a la 25-26 no tenen cap garantia de
generalitzar a temporades futures.

Sortides:
  - optimize_estrategia_results.csv  (totes les combinacions provades)
  - resum per pantalla amb el top 20 i el valor òptim de cada paràmetre
"""

import os
import sys

# El projecte està repartit en carpetes (entrenament_model, simulacio,
# automatitzacio): es posen totes al sys.path perquè els imports entre
# mòduls segueixin funcionant executant l'script des de qualsevol lloc.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("entrenament_model", "simulacio", "automatitzacio"):
    sys.path.insert(0, os.path.join(_ROOT, _d))

import itertools
import time

import numpy as np
import pandas as pd

import estrategia
from staking import kelly_stake

SEASON = "2025-26"

# ── Graella de valors (pas i rang acordats) ───────────────────────────────────
GRID = {
    "EDGE_MIN":          [round(v, 2) for v in np.arange(0.02, 0.101, 0.01)],   # 9
    "PROB_MIN_LOCAL":    [round(v, 2) for v in np.arange(0.35, 0.551, 0.02)],   # 11
    "TRIGGER_LOCAL_MIN": [round(v, 2) for v in np.arange(0.02, 0.101, 0.01)],   # 9
    "PROB_MIN_VISITANT": [round(v, 2) for v in np.arange(0.20, 0.421, 0.02)],   # 12
    "TEAM_CONF_K_UP":    [round(v, 2) for v in np.arange(0.00, 0.201, 0.02)],   # 11
    "TEAM_CONF_K_DOWN":  [round(v, 2) for v in np.arange(0.00, 0.101, 0.01)],   # 11
}

# Constants fixes (les del fitxer estrategia.py)
BANKROLL_INICIAL = estrategia.BANKROLL_INICIAL
KELLY_FRACTION   = estrategia.KELLY_FRACTION
MAX_STAKE_PCT    = estrategia.MAX_STAKE_PCT
MIN_STAKE        = estrategia.MIN_STAKE
CONF_INICIAL     = estrategia.TEAM_CONF_INICIAL
CONF_MIN         = estrategia.TEAM_CONF_MIN
CONF_MAX         = estrategia.TEAM_CONF_MAX


# ──────────────────────────────────────────────────────────────────────────────
# Precomputació (independent dels paràmetres a optimitzar)
# ──────────────────────────────────────────────────────────────────────────────
def precompute() -> list[list[dict]]:
    """Llista de jornades (en ordre cronològic de joc); cada jornada és una
    llista de partits amb tot el que cal per decidir i liquidar les apostes:

      prob_local, prob_visitant,
      millor_local:  (edge_base = -implicita de la millor casa + prob es fa
                      per-combinació? NO: la casa triada per line shopping és
                      argmax de (prob_model - implicita) = argmin d'implicita,
                      així que NO depèn d'EDGE_MIN ni de PROB_MIN -> es fixa aquí)
      pitjor_local:  argmax de (implicita_local - prob_model) = argmax
                     d'implicita_local -> tampoc depèn dels paràmetres.
      El filtre MAX_DIFF_VS_AVG s'aplica a la quota de la casa triada (fix).
    """
    from comparar_cases import get_model_predictions_from_saved

    preds, _ = get_model_predictions_from_saved(SEASON)
    quotes = estrategia.carregar_quotes(SEASON)
    df = preds.merge(quotes, on=["home_team", "away_team"], how="inner")
    df = estrategia.add_jornades(df)
    df = df[~df["ajornat"]]
    df = df[(df["jornada"] >= estrategia.JORNADA_MIN) & (df["jornada"] <= estrategia.JORNADA_MAX)]

    ordre = sorted(df["jornada"].unique(),
                   key=lambda j: df.loc[df["jornada"] == j, "date"].min())

    jornades = []
    for j in ordre:
        partits = []
        for _, r in df[df["jornada"] == j].iterrows():
            p_local    = r["prob_Local"] / 100
            p_visitant = r["prob_Visitant"] / 100

            millor_local, pitjor_local = None, None
            for code, nom in estrategia.CASES_ROI:
                il, iv = f"{code}_implicita_Local", f"{code}_implicita_Visitant"
                if il in r and pd.notna(r[il]):
                    edge = p_local - r[il]
                    if millor_local is None or edge > millor_local["edge"]:
                        millor_local = {"edge": edge, "quota": r[f"{code}_quota_Local"]}
                    if iv in r and pd.notna(r[iv]):
                        diff = r[il] - p_local
                        if pitjor_local is None or diff > pitjor_local["diff"]:
                            pitjor_local = {"diff": diff, "quota": r[f"{code}_quota_Visitant"]}

            if millor_local is not None:
                millor_local["ok_avg"] = estrategia._prop_vs_avg(r, "avg_quota_Local", millor_local["quota"])
            if pitjor_local is not None:
                pitjor_local["ok_avg"] = estrategia._prop_vs_avg(r, "avg_quota_Visitant", pitjor_local["quota"])

            partits.append({
                "home": r["home_team"], "away": r["away_team"],
                "p_local": p_local, "p_visitant": p_visitant,
                "millor": millor_local, "pitjor": pitjor_local,
                "real": r["resultat_real"],
            })
        jornades.append(partits)
    return jornades


# ──────────────────────────────────────────────────────────────────────────────
# Simulació ràpida (mateixa mecànica que estrategia.simular)
# ──────────────────────────────────────────────────────────────────────────────
def simula(jornades, edge_min, prob_min_local, trigger_local_min, prob_min_visitant,
           k_up, k_down) -> dict:
    bankroll = BANKROLL_INICIAL
    conf: dict = {}
    n = staked = profit_tot = wins = 0

    for partits in jornades:
        candidats = []
        for m in partits:
            ml = m["millor"]
            if (ml is not None and ml["edge"] >= edge_min and m["p_local"] >= prob_min_local
                    and ml["ok_avg"]):
                candidats.append((m["home"], m["p_local"], ml["quota"], m["real"] == "Local"))
            pl = m["pitjor"]
            if (pl is not None and pl["diff"] > trigger_local_min
                    and m["p_visitant"] > prob_min_visitant and pl["ok_avg"]):
                candidats.append((m["away"], m["p_visitant"], pl["quota"], m["real"] == "Visitant"))

        bk = bankroll
        max_stake = bk * MAX_STAKE_PCT
        profit_j = 0.0
        apostades = []
        for equip, prob, quota, guanyada in candidats:
            c = conf.get(equip, CONF_INICIAL)
            stake = kelly_stake(prob, quota, bk, KELLY_FRACTION, MIN_STAKE, max_stake)
            stake = min(stake * c, max_stake)
            if stake <= 0:
                continue
            p = (quota - 1) * stake if guanyada else -stake
            profit_j += p
            apostades.append((equip, prob, quota, guanyada))
            n += 1
            staked += stake
            profit_tot += p
            wins += guanyada

        bankroll += profit_j

        for equip, prob, quota, guanyada in apostades:
            prev = conf.get(equip, CONF_INICIAL)
            esperat = prob * quota - 1.0
            real = (quota - 1.0) if guanyada else -1.0
            sorpresa = real - esperat
            k = k_up if sorpresa > 0 else k_down
            conf[equip] = min(max(prev + k * sorpresa, CONF_MIN), CONF_MAX)

    return {"pot_final": bankroll, "apostes": n, "encerts": wins,
            "apostat": staked, "profit": profit_tot,
            "roi": (profit_tot / staked * 100) if staked else 0.0}


# ──────────────────────────────────────────────────────────────────────────────
def verifica(jornades) -> None:
    """La simulació ràpida ha de reproduir EXACTAMENT estrategia.simular
    amb els paràmetres actuals del fitxer (mateix pot final i n. d'apostes).
    Nota: estrategia.py arrodoneix els stakes/profits al report (round 2),
    aquí comparem sobre els valors sense arrodonir del mateix bucle."""
    ap = estrategia.simular(SEASON, {})
    pot_ref = BANKROLL_INICIAL + ap["profit"].sum()

    r = simula(jornades,
               estrategia.EDGE_MIN, estrategia.PROB_MIN["Local"],
               estrategia.TRIGGER_LOCAL_MIN, estrategia.PROB_MIN_VISITANT,
               estrategia.TEAM_CONF_K_UP, estrategia.TEAM_CONF_K_DOWN)

    assert r["apostes"] == len(ap), f"n apostes: {r['apostes']} vs {len(ap)}"
    # estrategia.py arrodoneix cada profit a 2 decimals al report; amb ~100
    # apostes la diferència acumulada pot ser d'uns cèntims
    assert abs(r["pot_final"] - pot_ref) < 0.5, f"pot: {r['pot_final']:.2f} vs {pot_ref:.2f}"
    print(f"[OK] Verificació: simulació ràpida == estrategia.simular "
          f"({r['apostes']} apostes, pot final {r['pot_final']:.2f}€)")


def main() -> None:
    print("Precomputant prediccions, quotes i jornades (un sol cop)...")
    jornades = precompute()
    verifica(jornades)

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"Provant {len(combos):,} combinacions "
          f"({' x '.join(str(len(v)) for v in GRID.values())})...")

    t0 = time.time()
    rows = []
    for i, combo in enumerate(combos):
        r = simula(jornades, *combo)
        rows.append((*combo, r["pot_final"], r["apostes"], r["encerts"], r["apostat"], r["roi"]))
        if (i + 1) % 100_000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,}/{len(combos):,}  ({el:.0f}s, {(i+1)/el:.0f} sims/s)")

    res = pd.DataFrame(rows, columns=[*keys, "pot_final", "apostes", "encerts", "apostat", "roi"])
    res = res.sort_values("pot_final", ascending=False).reset_index(drop=True)
    res.to_csv(os.path.join(_ROOT, "optimize_estrategia_results.csv"), index=False)

    print(f"\nFet en {time.time()-t0:.0f}s. Millors 20 combinacions (per pot final 2025-26):\n")
    print(res.head(20).to_string(index=False,
          float_format=lambda x: f"{x:.2f}"))

    print("\nPer paràmetre, valor de la MILLOR combinació i mitjana del top 1% "
          "(si difereixen molt, el màxim és fràgil):")
    top1pct = res.head(max(len(res) // 100, 20))
    for k in keys:
        print(f"  {k:>18}: millor={res.iloc[0][k]:.2f}  mitjana_top1%={top1pct[k].mean():.3f}")

    base = simula(jornades, estrategia.EDGE_MIN, estrategia.PROB_MIN["Local"],
                  estrategia.TRIGGER_LOCAL_MIN, estrategia.PROB_MIN_VISITANT,
                  estrategia.TEAM_CONF_K_UP, estrategia.TEAM_CONF_K_DOWN)
    print(f"\nBaseline (paràmetres actuals): pot final {base['pot_final']:.2f}€, "
          f"{base['apostes']} apostes, ROI {base['roi']:+.2f}%")
    print(f"Òptim trobat:                  pot final {res.iloc[0]['pot_final']:.2f}€, "
          f"{int(res.iloc[0]['apostes'])} apostes, ROI {res.iloc[0]['roi']:+.2f}%")
    print("\nResultats complets: optimize_estrategia_results.csv")


if __name__ == "__main__":
    main()
