import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_LEAGUES_PATH = os.path.join(_ROOT, "data", "all_leagues_matches.csv")
LA_LIGA_PATH     = os.path.join(_ROOT, "data", "la_liga_2014_2025_all_matches_final.csv")

TEST_LEAGUE = "La liga"

STAT_COLS = ["xg", "deep", "ppda", "xpts"]

FEATURES = [
    # Rolling global EWM (tots els partits anteriors)
    "home_xg_roll",   "away_xg_roll",   "xg_diff_roll",
    "home_deep_roll", "away_deep_roll", "deep_diff_roll",
    "home_ppda_roll", "away_ppda_roll", "ppda_diff_roll",
    "home_xpts_roll", "away_xpts_roll", "xpts_diff_roll",
    # Rolling home-only
    "home_xg_roll_h",   "away_xg_roll_h",
    "home_deep_roll_h", "away_deep_roll_h",
    "home_ppda_roll_h", "away_ppda_roll_h",
    "home_xpts_roll_h", "away_xpts_roll_h",
    # Rolling away-only
    "home_xg_roll_a",   "away_xg_roll_a",
    "home_deep_roll_a", "away_deep_roll_a",
    "home_ppda_roll_a", "away_ppda_roll_a",
    "home_xpts_roll_a", "away_xpts_roll_a",
    # ELO calculat ABANS del partit actual
    "home_elo", "away_elo", "elo_diff",
]


def load_all_leagues(path: str = ALL_LEAGUES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_la_liga(path: str = LA_LIGA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"season_mapped": "season"})
    df["league"] = "La liga"
    return df[["date", "league", "season", "home_team", "away_team",
               "home_goals", "away_goals", "home_xg", "away_xg",
               "home_ppda", "away_ppda", "home_deep", "away_deep",
               "home_xpts", "away_xpts"]]


def _ewm(history: list, roll_n: int, decay: float) -> float:
    """
    Exponential weighted mean dels últims roll_n elements.
    El més recent rep pes 1, el segon decay, el tercer decay^2, etc.
    decay=1.0 equival a la mitjana simple.
    """
    if not history:
        return np.nan
    recent = history[-roll_n:]
    n = len(recent)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, recent) / weights.sum())


def build_features(df_raw: pd.DataFrame, K: int = 16,
                   roll_n: int = 30, decay: float = 0.5) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Resultat ──────────────────────────────────────────────────────────────
    df["result"] = df.apply(
        lambda r: 0 if r["home_goals"] > r["away_goals"]
        else 1 if r["home_goals"] == r["away_goals"]
        else 2,
        axis=1,
    )

    # ── ELO (registrat ABANS del partit, actualitzat DESPRÉS) ─────────────────
    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)
        home_elos.append(eh)
        away_elos.append(ea)

        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        score_h = (
            1.0 if row["home_goals"] > row["away_goals"]
            else 0.5 if row["home_goals"] == row["away_goals"]
            else 0.0
        )
        elo[h] = eh + K * (score_h - exp_h)
        elo[a] = ea + K * ((1.0 - score_h) - (1.0 - exp_h))

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # ── Historials per equip ──────────────────────────────────────────────────
    hist_global = {}  # {equip: {stat: [vals]}}
    hist_home   = {}  # stats quan l'equip juga a casa
    hist_away   = {}  # stats quan l'equip juga fora

    rolls = {k: {s: [] for s in STAT_COLS}
             for k in ("glob_h", "glob_a", "home_h", "home_a", "away_h", "away_a")}

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]

        hg = hist_global.get(h, {s: [] for s in STAT_COLS})
        ag = hist_global.get(a, {s: [] for s in STAT_COLS})
        hh = hist_home.get(h,   {s: [] for s in STAT_COLS})
        ah = hist_home.get(a,   {s: [] for s in STAT_COLS})
        ha = hist_away.get(h,   {s: [] for s in STAT_COLS})
        aa = hist_away.get(a,   {s: [] for s in STAT_COLS})

        for s in STAT_COLS:
            rolls["glob_h"][s].append(_ewm(hg[s], roll_n, decay))
            rolls["glob_a"][s].append(_ewm(ag[s], roll_n, decay))
            rolls["home_h"][s].append(_ewm(hh[s], roll_n, decay))
            rolls["home_a"][s].append(_ewm(ah[s], roll_n, decay))
            rolls["away_h"][s].append(_ewm(ha[s], roll_n, decay))
            rolls["away_a"][s].append(_ewm(aa[s], roll_n, decay))

        # Actualitzem historials DESPRÉS de calcular les features
        for s in STAT_COLS:
            hist_global.setdefault(h, {s2: [] for s2 in STAT_COLS})[s].append(row[f"home_{s}"])
            hist_global.setdefault(a, {s2: [] for s2 in STAT_COLS})[s].append(row[f"away_{s}"])
            hist_home.setdefault(h,   {s2: [] for s2 in STAT_COLS})[s].append(row[f"home_{s}"])
            hist_away.setdefault(a,   {s2: [] for s2 in STAT_COLS})[s].append(row[f"away_{s}"])

    for s in STAT_COLS:
        df[f"home_{s}_roll"]   = rolls["glob_h"][s]
        df[f"away_{s}_roll"]   = rolls["glob_a"][s]
        df[f"{s}_diff_roll"]   = df[f"home_{s}_roll"] - df[f"away_{s}_roll"]
        df[f"home_{s}_roll_h"] = rolls["home_h"][s]
        df[f"away_{s}_roll_h"] = rolls["home_a"][s]
        df[f"home_{s}_roll_a"] = rolls["away_h"][s]
        df[f"away_{s}_roll_a"] = rolls["away_a"][s]

    return df
