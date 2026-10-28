"""Lan–DeMets OBF alpha spending: theory, equal vs irregular schedules."""

import numpy as np
import pytest
from scipy import stats

from peeking import false_positive_rate, peeking_with_effect
from peeking.effect import critical_z
from peeking.lan_demets import (
    equal_batch_sizes,
    false_positive_rate_lan_demets,
    info_fractions_from_batches,
    lan_demets_obf_z_criticals,
    misindexed_obf_z_criticals,
    obf_alpha_spent,
    peeking_with_effect_lan_demets,
    z_from_cumulative_alpha,
)
from peeking.obrien_fleming import obrien_fleming_z_criticals
from peeking.simulate import ever_significant, simulate_aa_z_batches

N = 20_000
P_CONTROL = 0.10
P_TREATMENT = 0.105
ALPHA = 0.05

# Irregular peek schedule: 10 looks, 20k users per arm (same total as equal batches).
IRREGULAR_BATCHES = np.array([500, 500, 1000, 1500, 2000, 2500, 3000, 4000, 3500, 1500], dtype=int)


def test_obf_spending_at_one_is_nominal_alpha():
    assert obf_alpha_spent(1.0, alpha=ALPHA) == pytest.approx(ALPHA, abs=1e-12)


def test_z_at_full_sample_is_fixed_sample_critical():
    z = lan_demets_obf_z_criticals(np.array([1.0]), alpha=ALPHA)
    assert z[0] == pytest.approx(critical_z(ALPHA), abs=1e-9)


def test_equal_spacing_matches_obf_spending_formula():
    """At t = k/K, Lan–DeMets z equals c·√(K/k) for the calibrated multiplier ``c``."""
    from peeking.lan_demets import _calibrated_obf_c

    n_looks = 10
    c = _calibrated_obf_c(n_looks, total_users_per_arm=N, base_rate=P_CONTROL, alpha=ALPHA)
    frac = np.arange(1, n_looks + 1) / n_looks
    ld = lan_demets_obf_z_criticals(frac, alpha=ALPHA, c=c)
    expected = c * np.sqrt(n_looks / np.arange(1, n_looks + 1))
    assert ld == pytest.approx(expected, rel=1e-5, abs=1e-4)


def test_equal_batches_lan_demets_matches_calibrated_obf():
    n_looks = 10
    batches = equal_batch_sizes(n_looks, N)
    from peeking.lan_demets import lan_demets_z_criticals_for_batches

    ld = lan_demets_z_criticals_for_batches(batches, N, base_rate=P_CONTROL, alpha=ALPHA)
    obf = obrien_fleming_z_criticals(n_looks, total_users_per_arm=N, base_rate=P_CONTROL)
    assert ld == pytest.approx(obf, rel=0.01, abs=0.05)


def test_lan_demets_fpr_nominal_equal_and_irregular():
    equal = equal_batch_sizes(10, N)
    for batches, label in ((equal, "equal"), (IRREGULAR_BATCHES, "irregular")):
        rate = false_positive_rate_lan_demets(
            batches, total_users_per_arm=N, n_sims=25_000, seed=3
        )
        assert rate == pytest.approx(0.05, abs=0.012), label


def test_naive_fpr_inflates_on_irregular_peeks():
    rng = np.random.default_rng(0)
    z = simulate_aa_z_batches(30_000, IRREGULAR_BATCHES, P_CONTROL, rng)
    naive = float(ever_significant(z, alpha=ALPHA).mean())
    ld = false_positive_rate_lan_demets(
        IRREGULAR_BATCHES, total_users_per_arm=N, n_sims=30_000, seed=0
    )
    assert naive > 0.15
    assert ld == pytest.approx(0.05, abs=0.012)


def test_misindexed_boundaries_differ_from_information_fraction():
    """Look index k/K is not the information fraction on an irregular schedule."""
    from peeking.lan_demets import lan_demets_z_criticals_for_batches

    correct = lan_demets_z_criticals_for_batches(
        IRREGULAR_BATCHES, N, base_rate=P_CONTROL, alpha=ALPHA
    )
    wrong = misindexed_obf_z_criticals(
        len(IRREGULAR_BATCHES), total_users_per_arm=N, base_rate=P_CONTROL, alpha=ALPHA
    )
    assert not np.allclose(correct, wrong, rtol=0, atol=0.05)
    # Early peeks use too low a bar if you treat look k as fraction k/K.
    assert wrong[0] < correct[0]


def test_lan_demets_power_matches_obf_on_equal_schedule():
    from peeking.obrien_fleming import peeking_with_effect_obrien_fleming

    batches = equal_batch_sizes(10, N)
    common = dict(
        n_sims=15_000,
        batch_sizes=batches,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=5,
    )
    ld = peeking_with_effect_lan_demets(**common)
    obf = peeking_with_effect_obrien_fleming(
        n_sims=common["n_sims"],
        n_looks=10,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=5,
    )
    assert ld.p_correct == pytest.approx(obf.p_correct, abs=0.01)


def test_cumulative_alpha_monotone_in_t():
    t = np.linspace(0.05, 1.0, 20)
    spent = obf_alpha_spent(t, alpha=ALPHA)
    assert np.all(np.diff(spent) > 0)
    assert spent[-1] == pytest.approx(ALPHA, abs=1e-12)


def test_z_from_cumulative_alpha_inverts_spending():
    t = 0.37
    spent = float(obf_alpha_spent(t, alpha=ALPHA))
    z = float(z_from_cumulative_alpha(spent))
    assert stats.norm.sf(z) * 2 == pytest.approx(spent, rel=1e-5)
