"""Mixture sequential probability ratio test (mSPRT) for two Bernoulli proportions.

Under a normal mixing prior N(0, tau^2) on the lift, the mixture likelihood ratio
at cumulative sample size n per arm compares treatment to control. Stop the
first time Lambda >= 1/alpha; by Ville's inequality the false-positive rate is
at most alpha under continuous monitoring (Johari et al., 2017).
"""

from __future__ import annotations

import numpy as np

from peeking.effect import StoppingSummary, summarize_stopping

_MIN_N = 100


def msprt_threshold(alpha: float = 0.05) -> float:
    """Reject the null when the mixture likelihood ratio reaches this value."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return 1.0 / alpha


def default_tau(base_rate: float = 0.10) -> float:
    """Default mixing standard deviation on the lift scale (absolute rate difference)."""
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be strictly between 0 and 1")
    return float(base_rate * 0.10)


def mixture_likelihood_ratio(
    n: int | np.ndarray,
    success_control: float | np.ndarray,
    success_treatment: float | np.ndarray,
    tau: float,
    theta: float = 0.0,
    min_n: int = _MIN_N,
) -> np.ndarray:
    """Bernoulli mSPRT mixture likelihood ratio (treatment minus control).

    Closed form from Johari et al. (2017), matching the Bernoulli branch of the
    mixtureSPRT reference implementation with independent streams per arm.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    n_arr = np.asarray(n, dtype=float)
    s_c = np.asarray(success_control, dtype=float)
    s_t = np.asarray(success_treatment, dtype=float)
    if np.any(n_arr < 1):
        raise ValueError("n must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        p_c = s_c / n_arr
        p_t = s_t / n_arr
        delta = p_t - p_c - theta
        v = p_c * (1.0 - p_c) / n_arr + p_t * (1.0 - p_t) / n_arr
        v = np.maximum(v, 1e-15)
        denom = v + tau**2
        log_lambda = 0.5 * np.log(v / denom) + (delta**2 * tau**2) / (2.0 * v * denom)
        log_lambda = np.clip(log_lambda, -700.0, 700.0)
        ratio = np.exp(log_lambda)
        ratio = np.where(n_arr >= min_n, ratio, 0.0)
    return ratio


def lambda_path_from_counts(
    conv_control: np.ndarray,
    conv_treatment: np.ndarray,
    cumulative_n: np.ndarray,
    tau: float,
    theta: float = 0.0,
    min_n: int = _MIN_N,
) -> np.ndarray:
    """Mixture likelihood ratio at each look, shape (n_sims, n_looks)."""
    if conv_control.shape != conv_treatment.shape or conv_control.ndim != 2:
        raise ValueError("conversion arrays must share shape (n_sims, n_looks)")
    n_looks = conv_control.shape[1]
    cum = np.asarray(cumulative_n, dtype=float)
    if cum.shape != (n_looks,):
        raise ValueError("cumulative_n must have length n_looks")
    ratios = np.zeros_like(conv_control, dtype=float)
    for k in range(n_looks):
        ratios[:, k] = mixture_likelihood_ratio(
            int(cum[k]),
            conv_control[:, k],
            conv_treatment[:, k],
            tau,
            theta=theta,
            min_n=min_n,
        )
    return ratios


def first_stop_msprt(
    lambda_path: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Index of the first look with Lambda >= 1/alpha, or -1."""
    if lambda_path.ndim != 2:
        raise ValueError("lambda_path must be two-dimensional")
    threshold = msprt_threshold(alpha)
    crossed = lambda_path >= threshold
    first = np.argmax(crossed, axis=1).astype(int)
    first[~crossed.any(axis=1)] = -1
    return first


def simulate_aa_counts(
    n_sims: int,
    n_looks: int,
    users_per_look: int,
    base_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative conversions per arm under the null, shape (n_sims, n_looks)."""
    if n_sims < 1 or n_looks < 1 or users_per_look < 1:
        raise ValueError("n_sims, n_looks and users_per_look must be positive")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be strictly between 0 and 1")
    conv_a = rng.binomial(users_per_look, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_b = rng.binomial(users_per_look, base_rate, size=(n_sims, n_looks)).cumsum(axis=1)
    return conv_a, conv_b


def simulate_binomial_paths(
    n_sims: int,
    n_looks: int,
    users_per_look: int,
    rate_control: float,
    rate_treatment: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative conversions, z, and lift with shape (n_sims, n_looks)."""
    conv_c = rng.binomial(users_per_look, rate_control, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(users_per_look, rate_treatment, size=(n_sims, n_looks)).cumsum(axis=1)
    n = users_per_look * np.arange(1, n_looks + 1, dtype=float)
    pooled = (conv_c + conv_t) / (2.0 * n)
    se = np.sqrt(pooled * (1.0 - pooled) * (2.0 / n))
    lift = (conv_t - conv_c) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0.0, lift / se, 0.0)
    return conv_c, conv_t, z, lift


def false_positive_rate_msprt(
    n_looks: int,
    total_users_per_arm: int = 20_000,
    base_rate: float = 0.10,
    alpha: float = 0.05,
    tau: float | None = None,
    n_sims: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of A/A paths that cross the mSPRT threshold at any look."""
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    if tau is None:
        tau = default_tau(base_rate)
    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    conv_c, conv_t = simulate_aa_counts(n_sims, n_looks, users_per_look, base_rate, rng)
    n = users_per_look * np.arange(1, n_looks + 1)
    lambdas = lambda_path_from_counts(conv_c, conv_t, n, tau)
    return float((first_stop_msprt(lambdas, alpha) >= 0).mean())


def _msprt_significant_z(
    lambda_path: np.ndarray,
    lift: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Encode mSPRT rejection as |z| >= 1 with the sign of the estimated lift."""
    threshold = msprt_threshold(alpha)
    crossed = lambda_path >= threshold
    z = np.zeros_like(lift)
    z[crossed] = np.sign(lift[crossed]) * 2.0
    return z


def peeking_with_effect_msprt(
    n_sims: int,
    n_looks: int,
    total_users_per_arm: int = 20_000,
    rate_control: float = 0.10,
    rate_treatment: float = 0.105,
    alpha: float = 0.05,
    tau: float | None = None,
    seed: int = 0,
    model: str = "binomial",
) -> StoppingSummary:
    """Peeking with mSPRT stopping at each equally spaced batch."""
    if model != "binomial":
        raise ValueError("mSPRT for proportions is implemented for binomial data only")
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    true_lift = rate_treatment - rate_control
    if true_lift == 0.0:
        raise ValueError("rate_treatment and rate_control must differ")
    if tau is None:
        tau = default_tau(rate_control)

    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    conv_c, conv_t, _z, lift = simulate_binomial_paths(
        n_sims,
        n_looks,
        users_per_look,
        rate_control,
        rate_treatment,
        rng,
    )
    n = users_per_look * np.arange(1, n_looks + 1)
    lambdas = lambda_path_from_counts(conv_c, conv_t, n, tau)
    z_msprt = _msprt_significant_z(lambdas, lift, alpha)
    return summarize_stopping(
        z_msprt,
        lift,
        users_per_look,
        true_lift,
        alpha,
        z_critical=1.0,
    )
