"""O'Brien–Fleming group-sequential boundary: strict early, lenient late.

At look k of K equally spaced looks, stop if |z| exceeds
``c * sqrt(K / k)``, where ``c`` is calibrated so the overall A/A error
rate is ``alpha``. The final look uses threshold ``c`` (below the fixed-sample
1.96 when K > 1).
"""

from __future__ import annotations

import numpy as np

from peeking.effect import critical_z, peeking_with_effect, StoppingSummary
from peeking.pocock import fpr_at_z_critical
from peeking.simulate import ever_significant, simulate_aa_z

# Cached final-look multiplier ``c`` keyed by (n_looks, alpha, total_users, base_rate, cal_seed).
_BOUNDARY_CACHE: dict[tuple[int, float, int, float, int], np.ndarray] = {}


def obf_sqrt_multipliers(n_looks: int) -> np.ndarray:
    """Return sqrt(K/k) for k = 1, …, K."""
    if n_looks < 1:
        raise ValueError("n_looks must be positive")
    k = np.arange(1, n_looks + 1, dtype=float)
    return np.sqrt(n_looks / k)


def z_criticals_from_c(c: float, n_looks: int) -> np.ndarray:
    """Per-look two-sided z thresholds for O'Brien–Fleming with final multiplier ``c``."""
    return c * obf_sqrt_multipliers(n_looks)


def fpr_at_obf_c(
    z: np.ndarray,
    c: float,
    n_looks: int,
) -> float:
    """Share of paths that cross the OBF boundary at any look."""
    return float(ever_significant(z, z_critical=z_criticals_from_c(c, n_looks)).mean())


def calibrate_obf_c(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
    tol: float = 0.002,
    c_lo: float = 1.5,
    c_hi: float = 5.5,
) -> float:
    """Find the final-look multiplier ``c`` with overall A/A error rate ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if n_looks < 1:
        raise ValueError("n_looks must be positive")
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    if n_looks == 1:
        return critical_z(alpha)

    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    z = simulate_aa_z(n_sims, n_looks, users_per_look, base_rate, rng)

    fpr_lo = fpr_at_obf_c(z, c_lo, n_looks)
    fpr_hi = fpr_at_obf_c(z, c_hi, n_looks)
    if fpr_lo < alpha:
        raise RuntimeError("bisection bracket failed at lower end")
    if fpr_hi > alpha:
        raise RuntimeError("bisection bracket failed at upper end")

    while c_hi - c_lo > tol:
        mid = 0.5 * (c_lo + c_hi)
        if fpr_at_obf_c(z, mid, n_looks) > alpha:
            c_lo = mid
        else:
            c_hi = mid
    return 0.5 * (c_lo + c_hi)


def obrien_fleming_z_criticals(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> np.ndarray:
    """Return calibrated per-look O'Brien–Fleming z thresholds."""
    key = (n_looks, alpha, total_users_per_arm, base_rate, cal_seed)
    if key not in _BOUNDARY_CACHE:
        c = calibrate_obf_c(
            n_looks,
            total_users_per_arm=total_users_per_arm,
            base_rate=base_rate,
            alpha=alpha,
            n_sims=n_sims_cal,
            seed=cal_seed,
        )
        _BOUNDARY_CACHE[key] = z_criticals_from_c(c, n_looks)
    return _BOUNDARY_CACHE[key].copy()


def obrien_fleming_final_z(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> float:
    """Threshold at the planned final look (the calibrated multiplier ``c``)."""
    return float(obrien_fleming_z_criticals(
        n_looks,
        total_users_per_arm=total_users_per_arm,
        base_rate=base_rate,
        alpha=alpha,
        n_sims_cal=n_sims_cal,
        cal_seed=cal_seed,
    )[-1])


def false_positive_rate_obrien_fleming(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> float:
    """Share of A/A tests that cross the O'Brien–Fleming boundary at any look."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    z_crits = obrien_fleming_z_criticals(
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
    return float(ever_significant(z, z_critical=z_crits).mean())


def peeking_with_effect_obrien_fleming(
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
    """Peeking with O'Brien–Fleming thresholds at each look."""
    z_crits = obrien_fleming_z_criticals(
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
        z_critical=z_crits,
    )
