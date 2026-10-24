"""Checks for peeking with a real effect, against truncated-normal formulas."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from peeking.effect import (
    critical_lift,
    first_stop_index,
    fixed_sample_direction_probabilities,
    fixed_sample_mean_lift_given_correct,
    fixed_sample_power,
    mean_given_positive_rejection,
    mean_given_rejection,
    normal_tail_probabilities,
    peeking_with_effect,
    sampling_sd,
    simulate_binomial_looks,
    summarize_stopping,
)

# One planned analysis, 20,000 users per arm, 10% vs 10.5%.
N = 20_000
P_CONTROL = 0.10
P_TREATMENT = 0.105
TRUE_LIFT = P_TREATMENT - P_CONTROL


def test_sampling_sd_hand_calculation():
    variance = 0.1 * 0.9 / N + 0.105 * 0.895 / N
    assert sampling_sd(N, P_CONTROL, P_TREATMENT) == pytest.approx(variance**0.5, rel=1e-12)


def test_critical_lift_hand_calculation():
    # Phi^{-1}(0.975) * sqrt(0.1025 * 0.8975 * 2 / 20000)
    z_crit = 1.959963984540054
    se = (0.1025 * 0.8975 * 2 / N) ** 0.5
    assert critical_lift(N, P_CONTROL, P_TREATMENT) == pytest.approx(z_crit * se, rel=1e-12)


def test_standard_normal_mills_ratio():
    # E[Z | Z > 1.96] = phi(1.96) / (1 - Phi(1.96)) for a standard normal.
    expected = stats.norm.pdf(1.96) / stats.norm.sf(1.96)
    assert mean_given_positive_rejection(0.0, 1.0, 1.96) == pytest.approx(expected, rel=1e-12)


def test_symmetric_rejection_mean_is_zero():
    assert mean_given_rejection(0.0, 1.0, 1.96) == pytest.approx(0.0, abs=1e-12)


def test_positive_conditional_mean_matches_quadrature():
    delta, sigma, threshold = 0.005, 0.003033, 0.005944

    def density(x: float) -> float:
        return float(stats.norm.pdf(x, loc=delta, scale=sigma))

    probability, _ = integrate.quad(density, threshold, np.inf, epsabs=1e-12)
    moment, _ = integrate.quad(lambda x: x * density(x), threshold, np.inf, epsabs=1e-12)
    assert mean_given_positive_rejection(delta, sigma, threshold) == pytest.approx(
        moment / probability, rel=1e-8
    )


def test_two_sided_conditional_mean_matches_quadrature():
    delta, sigma, threshold = 0.002, 0.004, 0.006

    def density(x: float) -> float:
        return float(stats.norm.pdf(x, loc=delta, scale=sigma))

    def moment(x: float) -> float:
        return x * density(x)

    p_hi, _ = integrate.quad(density, threshold, np.inf, epsabs=1e-12)
    p_lo, _ = integrate.quad(density, -np.inf, -threshold, epsabs=1e-12)
    m_hi, _ = integrate.quad(moment, threshold, np.inf, epsabs=1e-12)
    m_lo, _ = integrate.quad(moment, -np.inf, -threshold, epsabs=1e-12)
    assert mean_given_rejection(delta, sigma, threshold) == pytest.approx(
        (m_hi + m_lo) / (p_hi + p_lo), rel=1e-8
    )


def test_fixed_sample_helpers_use_sampling_sd_and_null_threshold():
    delta = TRUE_LIFT
    sigma = sampling_sd(N, P_CONTROL, P_TREATMENT)
    threshold = critical_lift(N, P_CONTROL, P_TREATMENT)
    upper, lower = normal_tail_probabilities(delta, sigma, threshold)
    assert fixed_sample_power(N, P_CONTROL, P_TREATMENT) == pytest.approx(upper + lower)
    assert fixed_sample_direction_probabilities(N, P_CONTROL, P_TREATMENT) == pytest.approx(
        (upper, lower)
    )
    assert fixed_sample_mean_lift_given_correct(N, P_CONTROL, P_TREATMENT) == pytest.approx(
        mean_given_positive_rejection(delta, sigma, threshold)
    )


def test_first_stop_index():
    z = np.array([[0.0, 2.0], [3.0, 4.0], [1.0, -1.0]])
    assert first_stop_index(z).tolist() == [1, 0, -1]


def test_summarize_stopping_hand_built():
    z = np.array(
        [
            [0.5, 2.1, 0.0],
            [-1.0, -1.0, -3.0],
            [1.0, 1.5, 1.9],
        ]
    )
    lift = np.array(
        [
            [0.01, 0.02, 0.03],
            [-0.01, -0.02, -0.04],
            [0.001, 0.002, 0.003],
        ]
    )
    summary = summarize_stopping(z, lift, users_per_look=100, true_lift=0.005)
    assert summary.n_correct == 1
    assert summary.n_wrong == 1
    assert summary.n_inconclusive == 1
    assert summary.mean_lift_at_correct_stop == pytest.approx(0.02)
    assert summary.mean_lift_at_any_stop == pytest.approx(-0.01)
    assert summary.mean_lift_at_final == pytest.approx((0.03 - 0.04 + 0.003) / 3)
    assert summary.mean_lift_at_decision == pytest.approx((0.02 - 0.04 + 0.003) / 3)
    assert summary.mean_users_at_decision == pytest.approx((200 + 300 + 300) / 3)
    assert summary.n_correct_by_look.tolist() == [0, 1, 0]
    assert summary.n_wrong_by_look.tolist() == [0, 0, 1]
    assert summary.mean_lift_correct_by_look[1] == pytest.approx(0.02)
    assert summary.n_final_correct == 0
    assert summary.n_final_wrong == 1

    flipped = summarize_stopping(z, lift, users_per_look=100, true_lift=-0.005)
    assert flipped.n_correct == 1
    assert flipped.n_wrong == 1
    assert flipped.mean_lift_at_correct_stop == pytest.approx(-0.04)
    assert flipped.n_final_correct == 1
    assert flipped.n_final_wrong == 0


def test_summarize_stopping_never_crosses():
    z = np.zeros((4, 3))
    lift = np.full((4, 3), 0.005)
    summary = summarize_stopping(z, lift, users_per_look=10, true_lift=0.005)
    assert summary.n_correct == 0
    assert summary.n_inconclusive == 4
    assert math.isnan(summary.mean_lift_at_correct_stop)
    assert summary.mean_lift_at_final == pytest.approx(0.005)
    assert summary.mean_lift_at_decision == pytest.approx(0.005)
    assert summary.mean_users_at_decision == pytest.approx(30)
    assert summary.n_correct_by_look.tolist() == [0, 0, 0]


def test_z_is_lift_over_estimated_se():
    z, lift = simulate_binomial_looks(30, 5, 400, 0.2, 0.25, np.random.default_rng(0))
    assert z.shape == lift.shape == (30, 5)
    assert np.all(z * lift >= -1e-8)


def test_one_look_normal_matches_truncated_normal():
    summary = peeking_with_effect(
        n_sims=30_000,
        n_looks=1,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=2,
        model="normal",
    )
    p_correct, p_wrong = fixed_sample_direction_probabilities(N, P_CONTROL, P_TREATMENT)
    conditional = fixed_sample_mean_lift_given_correct(N, P_CONTROL, P_TREATMENT)
    assert summary.p_correct == pytest.approx(p_correct, abs=0.01)
    assert summary.p_wrong == pytest.approx(p_wrong, abs=0.002)
    assert summary.mean_lift_at_correct_stop == pytest.approx(conditional, abs=0.0002)
    assert summary.mean_lift_at_final == pytest.approx(TRUE_LIFT, abs=0.00015)
    assert summary.mean_users_at_decision == pytest.approx(float(N))
    assert summary.n_correct + summary.n_wrong + summary.n_inconclusive == summary.n_sims


def test_one_look_binomial_matches_truncated_normal():
    summary = peeking_with_effect(
        n_sims=30_000,
        n_looks=1,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=3,
        model="binomial",
    )
    p_correct, p_wrong = fixed_sample_direction_probabilities(N, P_CONTROL, P_TREATMENT)
    conditional = fixed_sample_mean_lift_given_correct(N, P_CONTROL, P_TREATMENT)
    assert summary.p_correct == pytest.approx(p_correct, abs=0.01)
    assert summary.p_wrong == pytest.approx(p_wrong, abs=0.002)
    assert summary.mean_lift_at_correct_stop == pytest.approx(conditional, abs=0.0003)
    assert summary.mean_lift_at_final == pytest.approx(TRUE_LIFT, abs=0.00015)
    assert summary.n_final_correct / summary.n_sims == pytest.approx(p_correct, abs=0.01)


def test_twenty_looks_binomial_matches_normal_model():
    common = dict(
        n_sims=12_000,
        n_looks=20,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=4,
    )
    binomial = peeking_with_effect(model="binomial", **common)
    normal = peeking_with_effect(model="normal", **common)
    assert binomial.p_correct == pytest.approx(normal.p_correct, abs=0.02)
    assert binomial.mean_lift_at_correct_stop == pytest.approx(
        normal.mean_lift_at_correct_stop, abs=0.0015
    )
    p_final, _ = fixed_sample_direction_probabilities(N, P_CONTROL, P_TREATMENT)
    assert binomial.n_final_correct / binomial.n_sims == pytest.approx(p_final, abs=0.015)
    assert binomial.mean_lift_at_final == pytest.approx(TRUE_LIFT, abs=0.00025)


def test_extra_looks_raise_correct_calls_and_stopped_lift():
    common = dict(
        n_sims=12_000,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=5,
    )
    one = peeking_with_effect(n_looks=1, **common)
    twenty = peeking_with_effect(n_looks=20, **common)
    # The 20-look grid includes the planned end, so it calls a winner at least
    # as often as waiting until the end, and the calls it adds are early.
    assert twenty.p_correct > one.p_correct + 0.05
    assert twenty.mean_lift_at_correct_stop > one.mean_lift_at_correct_stop + 0.001
    assert twenty.mean_users_at_decision < one.mean_users_at_decision
    assert twenty.mean_lift_at_correct_stop > TRUE_LIFT + 0.002


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_sims": 0, "n_looks": 1},
        {"n_sims": 10, "n_looks": 3, "total_users_per_arm": 10},
        {"n_sims": 10, "n_looks": 1, "rate_control": 0.1, "rate_treatment": 0.1},
        {"n_sims": 10, "n_looks": 1, "rate_control": 0.0, "rate_treatment": 0.1},
        {"n_sims": 10, "n_looks": 1, "model": "poisson"},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        peeking_with_effect(**kwargs)
