"""Always-valid inference from mSPRT and naive p-value peeking under dense monitoring."""

from __future__ import annotations

import numpy as np
from scipy import stats

from peeking.msprt import (
    always_valid_p_path,
    default_tau,
    first_stop_always_valid_p,
    lambda_path_from_counts,
    simulate_aa_counts,
)
from peeking.simulate import simulate_aa_z


def naive_p_path_from_z(z: np.ndarray) -> np.ndarray:
    """Two-sided normal p-value at each look, shape (n_sims, n_looks)."""
    return 2.0 * stats.norm.sf(np.abs(z))


def first_stop_naive_p(p_path: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """First look with naive p < alpha (not valid under optional stopping)."""
    crossed = p_path < alpha
    first = np.argmax(crossed, axis=1).astype(int)
    first[~crossed.any(axis=1)] = -1
    return first


def false_positive_rate_naive_continuous(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of A/A paths that ever cross naive p < alpha (invalid under peeking)."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    z = simulate_aa_z(n_sims, n_looks, users_per_look, base_rate, rng)
    p = naive_p_path_from_z(z)
    return float((first_stop_naive_p(p, alpha) >= 0).mean())


def false_positive_rate_always_valid_continuous(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    tau: float | None = None,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Always-valid mSPRT p-values under the same dense peek schedule."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    if tau is None:
        tau = default_tau(base_rate)
    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    conv_c, conv_t = simulate_aa_counts(n_sims, n_looks, users_per_look, base_rate, rng)
    n = users_per_look * np.arange(1, n_looks + 1)
    lambdas = lambda_path_from_counts(conv_c, conv_t, n, tau)
    p_path = always_valid_p_path(lambdas)
    return float((first_stop_always_valid_p(p_path, alpha) >= 0).mean())
