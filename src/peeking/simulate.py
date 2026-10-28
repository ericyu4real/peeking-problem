"""Simulate A/A tests (no true effect) that are analyzed repeatedly as data arrives."""

from __future__ import annotations

import numpy as np
from scipy import stats


def simulate_aa_z(
    n_sims: int,
    n_looks: int,
    users_per_look: int,
    base_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return two-proportion z statistics with shape (n_sims, n_looks).

    Both arms share the same conversion rate, so every significant result is a
    false positive. Look k analyzes all users that arrived up to and including
    batch k.
    """
    if n_sims < 1 or n_looks < 1 or users_per_look < 1:
        raise ValueError("n_sims, n_looks and users_per_look must be positive")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be strictly between 0 and 1")

    conv_a = rng.binomial(users_per_look, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_b = rng.binomial(users_per_look, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    n = users_per_look * np.arange(1, n_looks + 1)

    pooled = (conv_a + conv_b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    diff = (conv_a - conv_b) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, 0.0)
    return z


def simulate_aa_z_batches(
    n_sims: int,
    batch_sizes: np.ndarray,
    base_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Two-proportion z statistics with shape (n_sims, n_looks) for variable batch sizes."""
    batches = np.asarray(batch_sizes, dtype=int)
    if batches.ndim != 1 or batches.size < 1:
        raise ValueError("batch_sizes must be a one-dimensional array")
    if np.any(batches < 1):
        raise ValueError("batch_sizes must be positive")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be strictly between 0 and 1")
    n_looks = batches.size
    conv_a = rng.binomial(batches, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_b = rng.binomial(batches, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    n = np.cumsum(batches)
    pooled = (conv_a + conv_b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    diff = (conv_a - conv_b) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, 0.0)
    return z


def ever_significant(
    z: np.ndarray,
    alpha: float = 0.05,
    z_critical: float | np.ndarray | None = None,
) -> np.ndarray:
    """Whether each simulated test crosses the two-sided threshold at any look."""
    if z_critical is None:
        z_critical = stats.norm.ppf(1 - alpha / 2)
    crit = np.asarray(z_critical, dtype=float)
    if crit.ndim == 0:
        return (np.abs(z) >= float(crit)).any(axis=1)
    if crit.shape != (z.shape[1],):
        raise ValueError("z_critical must be scalar or length n_looks")
    return (np.abs(z) >= crit).any(axis=1)


def false_positive_rate(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of A/A tests declared significant when stopping at the first significant look.

    The total sample size is fixed and split into ``n_looks`` equal batches, so
    the only thing that changes between calls is how often the analyst peeks.
    """
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    rng = np.random.default_rng(seed)
    z = simulate_aa_z(n_sims, n_looks, total_users_per_arm // n_looks, base_rate, rng)
    return float(ever_significant(z, alpha).mean())
