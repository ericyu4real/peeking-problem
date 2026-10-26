"""Pocock group-sequential boundary: the same two-sided z threshold at every look.

The constant boundary is chosen so that under the null the probability of ever
crossing it in K equally spaced looks is ``alpha``. Calibration uses A/A
simulation and bisection on the critical z value.
"""

from __future__ import annotations

import numpy as np

from peeking.effect import peeking_with_effect, StoppingSummary
from peeking.simulate import ever_significant, simulate_aa_z

# Cached boundaries keyed by (n_looks, alpha, total_users, base_rate, cal_seed).
_BOUNDARY_CACHE: dict[tuple[int, float, int, float, int], float] = {}


def fpr_at_z_critical(
    z: np.ndarray,
    z_critical: float,
) -> float:
    """Share of paths that cross ``z_critical`` at any look."""
    return float(ever_significant(z, z_critical=z_critical).mean())


def calibrate_pocock_boundary(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
    tol: float = 0.002,
    z_hi: float = 4.0,
) -> float:
    """Find the constant two-sided z boundary with overall A/A error rate ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if n_looks < 1:
        raise ValueError("n_looks must be positive")
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    if n_looks == 1:
        from scipy import stats

        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    z = simulate_aa_z(n_sims, n_looks, users_per_look, base_rate, rng)

    from peeking.effect import critical_z

    z_lo = critical_z(alpha)
    fpr_lo = fpr_at_z_critical(z, z_lo)
    fpr_hi = fpr_at_z_critical(z, z_hi)
    if fpr_lo < alpha:
        raise RuntimeError("bisection bracket failed at lower end")
    if fpr_hi > alpha:
        raise RuntimeError("bisection bracket failed at upper end")

    while z_hi - z_lo > tol:
        mid = 0.5 * (z_lo + z_hi)
        if fpr_at_z_critical(z, mid) > alpha:
            z_lo = mid
        else:
            z_hi = mid
    return 0.5 * (z_lo + z_hi)


def pocock_z_critical(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> float:
    """Return a calibrated Pocock boundary, using a module cache for repeat calls."""
    key = (n_looks, alpha, total_users_per_arm, base_rate, cal_seed)
    if key not in _BOUNDARY_CACHE:
        _BOUNDARY_CACHE[key] = calibrate_pocock_boundary(
            n_looks,
            total_users_per_arm=total_users_per_arm,
            base_rate=base_rate,
            alpha=alpha,
            n_sims=n_sims_cal,
            seed=cal_seed,
        )
    return _BOUNDARY_CACHE[key]


def false_positive_rate_pocock(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> float:
    """Share of A/A tests that cross the Pocock boundary at any look."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    z_crit = pocock_z_critical(
        n_looks,
        total_users_per_arm=total_users_per_arm,
        base_rate=base_rate,
        alpha=alpha,
        n_sims_cal=n_sims_cal,
        cal_seed=cal_seed,
    )
    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    z = simulate_aa_z(n_sims, n_looks, users_per_look, base_rate, rng)
    return fpr_at_z_critical(z, z_crit)


def peeking_with_effect_pocock(
    n_sims: int,
    n_looks: int,
    total_users_per_arm: int = 20_000,
    rate_control: float = 0.10,
    rate_treatment: float = 0.105,
    alpha: float = 0.05,
    seed: int = 0,
    model: str = "binomial",
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> StoppingSummary:
    """Peeking with a Pocock constant boundary at each look."""
    z_crit = pocock_z_critical(
        n_looks,
        total_users_per_arm=total_users_per_arm,
        base_rate=rate_control,
        alpha=alpha,
        n_sims_cal=n_sims_cal,
        cal_seed=cal_seed,
    )
    return peeking_with_effect(
        n_sims=n_sims,
        n_looks=n_looks,
        total_users_per_arm=total_users_per_arm,
        rate_control=rate_control,
        rate_treatment=rate_treatment,
        alpha=alpha,
        seed=seed,
        model=model,
        z_critical=z_crit,
    )
