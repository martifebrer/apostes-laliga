import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulacio"))

from staking import kelly_fraction, kelly_stake


def test_kelly_fraction_matches_formula():
    # p=0.5, quota 2.5 -> b=1.5 -> f* = (1.5*0.5 - 0.5)/1.5 = 1/6
    assert kelly_fraction(0.5, 2.5) == pytest.approx(1 / 6)


def test_kelly_fraction_zero_without_edge():
    # prob 0.4 amb quota 2.0 (implícita 50%): edge negatiu
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_fraction_rejects_quota_at_or_below_one():
    assert kelly_fraction(0.9, 1.0) == 0.0


def test_kelly_stake_is_fraction_of_full_kelly():
    full = kelly_stake(0.5, 2.5, bankroll=100, fraction=1.0)
    quarter = kelly_stake(0.5, 2.5, bankroll=100, fraction=0.25)
    assert quarter == pytest.approx(full / 4)


def test_kelly_stake_respects_cap_and_floor():
    assert kelly_stake(0.6, 3.0, bankroll=100, fraction=1.0, max_stake=10) == 10
    assert kelly_stake(0.5, 2.5, bankroll=100, fraction=0.01, min_stake=2) == 2


def test_kelly_stake_never_bets_without_edge_even_with_min_stake():
    assert kelly_stake(0.3, 2.0, bankroll=100, min_stake=5) == 0.0
