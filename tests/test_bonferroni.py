"""Bonferroni peeking: union-bound FPR and power versus naive stopping."""

import pytest

from peeking import false_positive_rate, peeking_with_effect
from peeking.bonferroni import (
    bonferroni_power_one_look,
    false_positive_rate_bonferroni,
    peeking_with_effect_bonferroni,
    per_look_alpha,
)
from peeking.effect import fixed_sample_power

N = 20_000
P_CONTROL = 0.10
P_TREATMENT = 0.105


def test_per_look_alpha_divides_by_looks():
    assert per_look_alpha(0.05, 10) == pytest.approx(0.005)
    assert per_look_alpha(0.05, 1) == pytest.approx(0.05)


def test_one_look_bonferroni_matches_naive_fpr():
    naive = false_positive_rate(n_looks=1, n_sims=30_000, seed=2)
    bonf = false_positive_rate_bonferroni(n_looks=1, n_sims=30_000, seed=2)
    assert bonf == pytest.approx(naive, abs=1e-12)
    assert bonf == pytest.approx(0.05, abs=0.006)


def test_bonferroni_fpr_stays_at_or_below_nominal():
    for n_looks in (2, 5, 10, 20, 50, 100):
        rate = false_positive_rate_bonferroni(n_looks=n_looks, n_sims=25_000, seed=1)
        assert rate <= 0.05 + 0.012


def test_ten_looks_naive_inflates_but_bonferroni_does_not():
    naive = false_positive_rate(n_looks=10, n_sims=30_000, seed=0)
    bonf = false_positive_rate_bonferroni(n_looks=10, n_sims=30_000, seed=0)
    assert naive == pytest.approx(0.193, abs=0.015)
    assert bonf <= 0.05 + 0.01
    assert bonf < naive - 0.10


def test_bonferroni_power_formula_matches_fixed_sample():
    for k in (1, 10, 100):
        expected = fixed_sample_power(
            N, P_CONTROL, P_TREATMENT, alpha=per_look_alpha(0.05, k)
        )
        assert bonferroni_power_one_look(N, P_CONTROL, P_TREATMENT, k) == pytest.approx(
            expected
        )


def test_one_look_bonferroni_power_matches_simulation():
    summary = peeking_with_effect_bonferroni(
        n_sims=25_000,
        n_looks=1,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=3,
        model="normal",
    )
    expected = bonferroni_power_one_look(N, P_CONTROL, P_TREATMENT, n_looks=1)
    assert summary.p_correct == pytest.approx(expected, abs=0.012)


def test_bonferroni_loses_power_vs_naive_with_many_looks():
    common = dict(
        n_sims=15_000,
        n_looks=20,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=4,
    )
    naive = peeking_with_effect(**common)
    bonf = peeking_with_effect_bonferroni(**common)
    assert bonf.p_correct < naive.p_correct - 0.05
    assert bonf.p_inconclusive > naive.p_inconclusive + 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0, "n_looks": 5},
        {"alpha": 0.05, "n_looks": 0},
    ],
)
def test_invalid_per_look_alpha_raises(kwargs):
    with pytest.raises(ValueError):
        per_look_alpha(**kwargs)
