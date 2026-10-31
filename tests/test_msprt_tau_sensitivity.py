"""mSPRT mixing variance tau: analytic monotonicity and simulated sensitivity."""

import numpy as np
import pytest

from peeking.msprt import (
    default_tau,
    false_positive_rate_msprt,
    mixture_likelihood_ratio,
    peeking_with_effect_msprt,
)


def test_null_mixture_ratio_decreases_as_tau_increases():
    """At delta=0 the ratio is sqrt(V/(V+tau^2)), which falls as tau grows."""
    n = 8_000
    p = 0.10
    s = int(n * p)
    taus = np.array([0.002, 0.005, 0.01, 0.02, 0.05])
    ratios = np.array([float(mixture_likelihood_ratio(n, s, s, t)) for t in taus])
    assert np.all(ratios[:-1] > ratios[1:])


def test_msprt_fpr_at_most_nominal_across_tau_grid():
    taus = [0.002, 0.005, 0.01, 0.02, 0.05]
    for tau in taus:
        rate = false_positive_rate_msprt(
            n_looks=100, tau=tau, n_sims=12_000, seed=2
        )
        assert rate <= 0.05 + 0.012


def test_larger_tau_raises_aa_crossings_toward_nominal():
    """Larger tau weights sampling noise more in Lambda, so A/A FPR rises (still <= alpha)."""
    common = dict(n_looks=50, n_sims=20_000, seed=5)
    fpr_small = false_positive_rate_msprt(tau=0.002, **common)
    fpr_default = false_positive_rate_msprt(tau=default_tau(0.10), **common)
    fpr_large = false_positive_rate_msprt(tau=0.05, **common)
    assert fpr_small <= fpr_default + 0.005
    assert fpr_default <= fpr_large + 0.008
    assert fpr_large <= 0.05 + 0.012


def test_power_rises_with_tau_for_detectable_lift():
    """A lift of 0.010 is easier to catch when tau is not far below the effect."""
    common = dict(
        n_sims=15_000,
        n_looks=100,
        total_users_per_arm=20_000,
        rate_control=0.10,
        rate_treatment=0.110,
        seed=6,
    )
    tiny = peeking_with_effect_msprt(tau=0.002, **common)
    mid = peeking_with_effect_msprt(tau=0.01, **common)
    wide = peeking_with_effect_msprt(tau=0.05, **common)
    assert mid.p_correct > tiny.p_correct + 0.05
    assert wide.p_correct > tiny.p_correct + 0.05


def test_default_tau_matches_ten_percent_of_baseline():
    assert default_tau(0.10) == pytest.approx(0.01)
