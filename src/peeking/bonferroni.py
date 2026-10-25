"""Bonferroni correction when the same test is peeked at K times.

Stop at the first look whose two-sided p-value is below ``alpha / n_looks``.
The union bound gives family-wise error rate at most ``alpha`` under the null,
no matter how correlated the cumulative z-statistics are.
"""

from __future__ import annotations

import numpy as np

from peeking.effect import peeking_with_effect, StoppingSummary
from peeking.simulate import ever_significant, simulate_aa_z


def per_look_alpha(alpha: float, n_looks: int) -> float:
    """Two-sided alpha for each of ``n_looks`` Bonferroni-adjusted tests."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if n_looks < 1:
        raise ValueError("n_looks must be positive")
    return alpha / n_looks


def false_positive_rate_bonferroni(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of A/A tests that cross the Bonferroni threshold at any look."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    z = simulate_aa_z(n_sims, n_looks, users_per_look, base_rate, rng)
    return float(ever_significant(z, per_look_alpha(alpha, n_looks)).mean())


def peeking_with_effect_bonferroni(
    n_sims: int,
    n_looks: int,
    total_users_per_arm: int = 20_000,
    rate_control: float = 0.10,
    rate_treatment: float = 0.105,
    alpha: float = 0.05,
    seed: int = 0,
    model: str = "binomial",
) -> StoppingSummary:
    """Naive peeking with a Bonferroni-adjusted threshold at each look."""
    return peeking_with_effect(
        n_sims=n_sims,
        n_looks=n_looks,
        total_users_per_arm=total_users_per_arm,
        rate_control=rate_control,
        rate_treatment=rate_treatment,
        alpha=per_look_alpha(alpha, n_looks),
        seed=seed,
        model=model,
    )


def bonferroni_power_one_look(
    n: int,
    rate_control: float,
    rate_treatment: float,
    n_looks: int,
    alpha: float = 0.05,
) -> float:
    """Normal approximation to power when only the final look uses alpha / n_looks.

    Bonferroni with K looks but a single analysis at n users per arm is equivalent
    to one two-sided test at ``alpha / K``.
    """
    from peeking.effect import fixed_sample_power

    return fixed_sample_power(
        n, rate_control, rate_treatment, alpha=per_look_alpha(alpha, n_looks)
    )
