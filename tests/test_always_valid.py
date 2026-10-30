"""Always-valid p-values from mSPRT and continuous monitoring error control."""

import numpy as np
import pytest

from peeking.always_valid import (
    false_positive_rate_always_valid_continuous,
    false_positive_rate_naive_continuous,
    naive_p_path_from_z,
)
from peeking.msprt import (
    always_valid_p_path,
    false_positive_rate_always_valid_msprt,
    false_positive_rate_msprt,
    first_stop_always_valid_p,
    first_stop_msprt,
    lambda_path_from_counts,
    mixture_likelihood_ratio,
    msprt_threshold,
    simulate_aa_counts,
)


def test_always_valid_p_nonincreasing():
    rng = np.random.default_rng(0)
    conv_c, conv_t = simulate_aa_counts(50, 8, 500, 0.10, rng)
    n = 500 * np.arange(1, 9)
    lambdas = lambda_path_from_counts(conv_c, conv_t, n, tau=0.01)
    p = always_valid_p_path(lambdas)
    assert np.all(p[:, 1:] <= p[:, :-1] + 1e-12)


def test_p_stop_equivalent_to_lambda_stop():
    rng = np.random.default_rng(1)
    conv_c, conv_t = simulate_aa_counts(200, 20, 1000, 0.10, rng)
    n = 1000 * np.arange(1, 21)
    lambdas = lambda_path_from_counts(conv_c, conv_t, n, tau=0.01)
    p = always_valid_p_path(lambdas)
    alpha = 0.05
    stop_l = first_stop_msprt(lambdas, alpha)
    stop_p = first_stop_always_valid_p(p, alpha)
    assert np.array_equal(stop_l, stop_p)


def test_always_valid_fpr_matches_msprt_lambda_rule():
    common = dict(n_looks=100, n_sims=12_000, seed=2)
    via_lambda = false_positive_rate_msprt(**common)
    via_p = false_positive_rate_always_valid_msprt(**common)
    assert via_p == pytest.approx(via_lambda, abs=1e-12)


def test_continuous_monitoring_always_valid_at_most_nominal():
    for step_users in (100, 50):
        n_looks = 20_000 // step_users
        rate = false_positive_rate_always_valid_continuous(
            n_looks=n_looks, n_sims=18_000, seed=3
        )
        assert rate <= 0.05 + 0.012


def test_continuous_naive_p_inflates_error():
    n_looks = 200
    naive = false_positive_rate_naive_continuous(n_looks=n_looks, n_sims=15_000, seed=4)
    av = false_positive_rate_always_valid_continuous(n_looks=n_looks, n_sims=15_000, seed=4)
    assert naive > 0.05 + 0.15
    assert av <= 0.05 + 0.012
    assert av < naive - 0.15


def test_naive_p_at_null_single_look_uniform():
    """Standard p-values are uniform at one fixed look (sanity check)."""
    rng = np.random.default_rng(5)
    n_sims = 20_000
    z = rng.normal(size=n_sims)
    p = naive_p_path_from_z(z.reshape(-1, 1))[:, 0]
    frac_low = float((p < 0.05).mean())
    assert frac_low == pytest.approx(0.05, abs=0.012)


def test_mixture_p_at_threshold():
    """When Lambda hits 1/alpha, always-valid p equals alpha."""
    thresh = msprt_threshold(0.05)
    p = float(np.minimum(1.0, 1.0 / thresh))
    assert p == pytest.approx(0.05, rel=1e-9)
