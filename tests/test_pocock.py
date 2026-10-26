"""Pocock boundary: calibration targets nominal FPR and beats Bonferroni power."""

import numpy as np
import pytest

from peeking import false_positive_rate, peeking_with_effect
from peeking.bonferroni import false_positive_rate_bonferroni, peeking_with_effect_bonferroni
from peeking.effect import critical_z
from peeking.pocock import (
    calibrate_pocock_boundary,
    false_positive_rate_pocock,
    fpr_at_z_critical,
    peeking_with_effect_pocock,
    pocock_z_critical,
)

N = 20_000
P_CONTROL = 0.10
P_TREATMENT = 0.105

# Pocock (1977) table, two-sided overall alpha 0.05, equally spaced looks.
POCOCK_TABLE = {
    2: 2.178,
    5: 2.413,
    10: 2.497,
}


def test_one_look_pocock_is_fixed_sample_critical():
    z = pocock_z_critical(1)
    assert z == pytest.approx(critical_z(0.05), abs=1e-9)


def test_pocock_boundary_matches_published_table():
    """Classical Pocock tables assume normal increments; K=2,5 match our binomial design closely."""
    for n_looks in (2, 5):
        z = calibrate_pocock_boundary(n_looks, n_sims=20_000, seed=1)
        assert z == pytest.approx(POCOCK_TABLE[n_looks], abs=0.025)


def test_classical_pocock_table_near_nominal_on_binomial_design():
    from peeking.simulate import simulate_aa_z

    rng = np.random.default_rng(1)
    z = simulate_aa_z(30_000, 10, 2_000, 0.10, rng)
    fpr = fpr_at_z_critical(z, POCOCK_TABLE[10])
    assert 0.045 <= fpr <= 0.065


def test_pocock_fpr_near_nominal_across_looks():
    for n_looks in (1, 2, 5, 10, 20, 50, 100):
        rate = false_positive_rate_pocock(n_looks=n_looks, n_sims=25_000, seed=2)
        assert rate == pytest.approx(0.05, abs=0.012)


def test_ten_looks_pocock_between_naive_and_bonferroni_fpr():
    naive = false_positive_rate(n_looks=10, n_sims=30_000, seed=0)
    pocock = false_positive_rate_pocock(n_looks=10, n_sims=30_000, seed=0)
    bonf = false_positive_rate_bonferroni(n_looks=10, n_sims=30_000, seed=0)
    assert naive == pytest.approx(0.193, abs=0.015)
    assert pocock == pytest.approx(0.05, abs=0.012)
    assert bonf <= 0.05 + 0.01
    assert bonf < pocock - 0.01 or bonf < naive - 0.10


def test_pocock_more_powerful_than_bonferroni_many_looks():
    common = dict(
        n_sims=15_000,
        n_looks=20,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=4,
    )
    pocock = peeking_with_effect_pocock(**common)
    bonf = peeking_with_effect_bonferroni(**common)
    naive = peeking_with_effect(**common)
    assert pocock.p_correct > bonf.p_correct + 0.05
    assert pocock.p_correct < naive.p_correct + 0.01


def test_pocock_boundary_increases_with_more_looks():
    z_prev = pocock_z_critical(1)
    for n_looks in (2, 5, 10, 20, 50, 100):
        z = pocock_z_critical(n_looks)
        assert z > z_prev
        z_prev = z


@pytest.mark.parametrize("n_looks", [0, -1])
def test_invalid_n_looks_raises(n_looks):
    with pytest.raises(ValueError):
        calibrate_pocock_boundary(n_looks)
