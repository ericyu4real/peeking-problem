"""mSPRT for two proportions: mixture ratio and always-valid error control."""

import numpy as np
import pytest

from peeking import false_positive_rate, peeking_with_effect
from peeking.msprt import (
    default_tau,
    false_positive_rate_msprt,
    mixture_likelihood_ratio,
    msprt_threshold,
    peeking_with_effect_msprt,
    simulate_aa_counts,
)


def test_threshold_is_inverse_alpha():
    assert msprt_threshold(0.05) == pytest.approx(20.0)
    assert msprt_threshold(0.10) == pytest.approx(10.0)


def test_mixture_ratio_at_null_matches_closed_form():
    """When delta=0 the ratio equals sqrt(V/(V+tau^2)) < 1."""
    n = 5_000
    p = 0.10
    s = int(n * p)
    tau = default_tau(p)
    ratio = float(mixture_likelihood_ratio(n, s, s, tau))
    v = p * (1 - p) / n + p * (1 - p) / n
    expected = np.sqrt(v / (v + tau**2))
    assert ratio == pytest.approx(expected, rel=1e-9)


def test_single_look_fpr_below_alpha():
    """One-shot mSPRT rejection rate is at most alpha under A/A (Monte Carlo)."""
    n_sims = 40_000
    n = 10_000
    p = 0.10
    tau = default_tau(p)
    rng = np.random.default_rng(3)
    s_c = rng.binomial(n, p, size=n_sims)
    s_t = rng.binomial(n, p, size=n_sims)
    ratios = mixture_likelihood_ratio(n, s_c, s_t, tau)
    fpr = float((ratios >= msprt_threshold(0.05)).mean())
    assert fpr <= 0.05 + 0.008


def test_msprt_fpr_at_most_nominal_many_looks():
    for n_looks in (1, 2, 5, 10, 20, 50, 100):
        rate = false_positive_rate_msprt(n_looks=n_looks, n_sims=25_000, seed=1)
        assert rate <= 0.05 + 0.012


def test_ten_looks_msprt_fpr_below_naive():
    naive = false_positive_rate(n_looks=10, n_sims=30_000, seed=0)
    msprt = false_positive_rate_msprt(n_looks=10, n_sims=30_000, seed=0)
    assert naive == pytest.approx(0.193, abs=0.015)
    assert msprt <= 0.05 + 0.012
    assert msprt < naive - 0.10


def test_msprt_more_powerful_than_bonferroni_many_looks():
    from peeking.bonferroni import peeking_with_effect_bonferroni

    common = dict(
        n_sims=15_000,
        n_looks=100,
        total_users_per_arm=20_000,
        rate_control=0.10,
        rate_treatment=0.110,
        seed=4,
    )
    msprt = peeking_with_effect_msprt(**common)
    bonf = peeking_with_effect_bonferroni(**common)
    naive = peeking_with_effect(**common)
    assert msprt.p_correct > bonf.p_correct + 0.10
    assert msprt.p_correct < naive.p_correct - 0.15


def test_default_tau_scales_with_baseline():
    assert default_tau(0.10) == pytest.approx(0.01)
    assert default_tau(0.20) == pytest.approx(0.02)


def test_simulate_aa_counts_shape():
    rng = np.random.default_rng(0)
    a, b = simulate_aa_counts(100, 5, 1000, 0.10, rng)
    assert a.shape == (100, 5)
    assert b.shape == a.shape
