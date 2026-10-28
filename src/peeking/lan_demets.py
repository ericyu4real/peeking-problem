"""Lan–DeMets alpha spending with O'Brien–Fleming spending function.

Boundaries depend on the information fraction (share of the planned sample
collected at each look), so peek times need not be equally spaced or fixed in
advance. At fraction ``t`` the cumulative Type I error spent is
``2 - 2 Φ(z_{α/2} / √t)``; the two-sided z threshold is the normal quantile
matching that cumulative alpha.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from peeking.effect import StoppingSummary, critical_z, peeking_with_effect
from peeking.obrien_fleming import calibrate_obf_c
from peeking.simulate import ever_significant, simulate_aa_z_batches

_BOUNDARY_CACHE: dict[tuple[tuple[float, ...], float, int, int, float, int], np.ndarray] = {}
_C_CALIB_CACHE: dict[tuple[int, float, int, float, int], float] = {}


def _calibrated_obf_c(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> float:
    """Final-look OBF multiplier ``c`` calibrated on equal batches (binomial design)."""
    key = (n_looks, alpha, total_users_per_arm, base_rate, cal_seed)
    if key not in _C_CALIB_CACHE:
        _C_CALIB_CACHE[key] = calibrate_obf_c(
            n_looks,
            total_users_per_arm=total_users_per_arm,
            base_rate=base_rate,
            alpha=alpha,
            n_sims=n_sims_cal,
            seed=cal_seed,
        )
    return _C_CALIB_CACHE[key]


def obf_alpha_spent(
    t: float | np.ndarray,
    alpha: float = 0.05,
    c: float | None = None,
) -> np.ndarray:
    """Cumulative alpha spent by O'Brien–Fleming spending at information fraction ``t``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    arr = np.asarray(t, dtype=float)
    if np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError("information fractions must be in (0, 1]")
    if c is None:
        c = critical_z(alpha)
    return 2.0 - 2.0 * stats.norm.cdf(c / np.sqrt(arr))


def z_from_cumulative_alpha(alpha_spent: float | np.ndarray) -> np.ndarray:
    """Two-sided z threshold for cumulative Type I error ``alpha_spent``."""
    arr = np.asarray(alpha_spent, dtype=float)
    if np.any(arr < 0.0) or np.any(arr >= 1.0):
        raise ValueError("alpha_spent must be in [0, 1)")
    z = np.full_like(arr, np.inf, dtype=float)
    positive = arr > 0.0
    z[positive] = stats.norm.ppf(1.0 - arr[positive] / 2.0)
    return z


def info_fractions_from_batches(
    batch_sizes: np.ndarray,
    total_users: int | None = None,
) -> np.ndarray:
    """Information fraction at the end of each batch."""
    batches = np.asarray(batch_sizes, dtype=float)
    if batches.ndim != 1 or batches.size < 1:
        raise ValueError("batch_sizes must be a one-dimensional array")
    if np.any(batches <= 0):
        raise ValueError("batch_sizes must be positive")
    cumulative = np.cumsum(batches)
    total = float(total_users if total_users is not None else cumulative[-1])
    if cumulative[-1] > total + 1e-9:
        raise ValueError("batch_sizes exceed total_users")
    return cumulative / total


def lan_demets_obf_z_criticals(
    info_fractions: np.ndarray,
    alpha: float = 0.05,
    c: float | None = None,
) -> np.ndarray:
    """Per-look two-sided z thresholds from Lan–DeMets OBF spending."""
    frac = np.asarray(info_fractions, dtype=float)
    if frac.ndim != 1 or frac.size < 1:
        raise ValueError("info_fractions must be one-dimensional")
    spent = obf_alpha_spent(frac, alpha=alpha, c=c)
    return z_from_cumulative_alpha(spent)


def lan_demets_z_criticals_for_batches(
    batch_sizes: np.ndarray,
    total_users_per_arm: int,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims_cal: int = 20_000,
    cal_seed: int = 0,
) -> np.ndarray:
    """Return Lan–DeMets OBF z thresholds for a batch schedule."""
    batches = np.asarray(batch_sizes, dtype=int)
    key = (
        tuple(int(x) for x in batches),
        alpha,
        total_users_per_arm,
        base_rate,
        cal_seed,
    )
    if key not in _BOUNDARY_CACHE:
        c = _calibrated_obf_c(
            len(batches),
            total_users_per_arm=total_users_per_arm,
            base_rate=base_rate,
            alpha=alpha,
            n_sims_cal=n_sims_cal,
            cal_seed=cal_seed,
        )
        frac = info_fractions_from_batches(batches, total_users=total_users_per_arm)
        _BOUNDARY_CACHE[key] = lan_demets_obf_z_criticals(frac, alpha=alpha, c=c)
    return _BOUNDARY_CACHE[key].copy()


def false_positive_rate_lan_demets(
    batch_sizes: np.ndarray | list[int],
    total_users_per_arm: int,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of A/A paths crossing the Lan–DeMets boundary at any look."""
    batches = np.asarray(batch_sizes, dtype=int)
    if int(batches.sum()) != total_users_per_arm:
        raise ValueError("batch_sizes must sum to total_users_per_arm")
    z_crits = lan_demets_z_criticals_for_batches(
        batches, total_users_per_arm, base_rate=base_rate, alpha=alpha
    )
    rng = np.random.default_rng(seed)
    z = simulate_aa_z_batches(n_sims, batches, base_rate, rng)
    return float(ever_significant(z, z_critical=z_crits).mean())


def peeking_with_effect_lan_demets(
    n_sims: int,
    batch_sizes: np.ndarray | list[int],
    total_users_per_arm: int,
    rate_control: float = 0.10,
    rate_treatment: float = 0.105,
    alpha: float = 0.05,
    seed: int = 0,
    model: str = "binomial",
) -> StoppingSummary:
    """Peeking with Lan–DeMets OBF boundaries on a (possibly irregular) schedule."""
    batches = np.asarray(batch_sizes, dtype=int)
    if int(batches.sum()) != total_users_per_arm:
        raise ValueError("batch_sizes must sum to total_users_per_arm")
    if model not in ("binomial", "normal"):
        raise ValueError("model must be 'binomial' or 'normal'")
    true_lift = rate_treatment - rate_control
    if true_lift == 0.0:
        raise ValueError("rate_treatment and rate_control must differ")

    z_crits = lan_demets_z_criticals_for_batches(
        batches,
        total_users_per_arm,
        base_rate=rate_control,
        alpha=alpha,
    )
    return peeking_with_effect(
        n_sims=n_sims,
        n_looks=len(batches),
        total_users_per_arm=total_users_per_arm,
        rate_control=rate_control,
        rate_treatment=rate_treatment,
        alpha=alpha,
        seed=seed,
        model=model,
        z_critical=z_crits,
        batch_sizes=batches,
    )


def equal_batch_sizes(n_looks: int, total_users_per_arm: int) -> np.ndarray:
    """Equal batches that sum to ``total_users_per_arm``."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    size = total_users_per_arm // n_looks
    return np.full(n_looks, size, dtype=int)


def misindexed_obf_z_criticals(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    cal_seed: int = 0,
) -> np.ndarray:
    """OBF boundaries at t = k/K instead of true information fractions (wrong for irregular peeks)."""
    c = _calibrated_obf_c(
        n_looks,
        total_users_per_arm=total_users_per_arm,
        base_rate=base_rate,
        alpha=alpha,
        cal_seed=cal_seed,
    )
    k = np.arange(1, n_looks + 1, dtype=float)
    return lan_demets_obf_z_criticals(k / n_looks, alpha=alpha, c=c)
