"""Peeking when treatment really differs from control.

The stopping rule is the same naive one as the A/A study: after each equal
batch, run a two-proportion z-test and stop at the first look with p < alpha.
What changes is that the true conversion rates differ, so a stop can be in the
right direction and the estimated lift at that moment can still be wrong.

For one planned look the estimate is approximately normal. Conditional on
crossing the critical value, its expectation is the mean of a truncated
normal; the helpers below compute that directly so the simulation can be
checked against it. With several looks the same normal increment model is
simulated as a second check on the binomial experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def critical_z(alpha: float = 0.05) -> float:
    """Two-sided normal critical value, P(|Z| > c) = alpha under the null."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _validate_rates(rate_control: float, rate_treatment: float) -> None:
    for name, rate in (("rate_control", rate_control), ("rate_treatment", rate_treatment)):
        if not 0.0 < rate < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")


def sampling_sd(n: int, rate_control: float, rate_treatment: float) -> float:
    """Standard deviation of p_hat_treatment - p_hat_control with n users per arm."""
    if n < 1:
        raise ValueError("n must be positive")
    _validate_rates(rate_control, rate_treatment)
    variance = (
        rate_control * (1.0 - rate_control) / n
        + rate_treatment * (1.0 - rate_treatment) / n
    )
    return float(np.sqrt(variance))


def null_se(n: int, rate_control: float, rate_treatment: float) -> float:
    """Large-sample pooled standard error used by the two-proportion z-test.

    The pool is the average of the two true rates, which is where the estimated
    pooled rate sits as n grows.
    """
    if n < 1:
        raise ValueError("n must be positive")
    _validate_rates(rate_control, rate_treatment)
    p_bar = 0.5 * (rate_control + rate_treatment)
    return float(np.sqrt(p_bar * (1.0 - p_bar) * 2.0 / n))


def critical_lift(
    n: int,
    rate_control: float,
    rate_treatment: float,
    alpha: float = 0.05,
) -> float:
    """Smallest absolute estimated lift that the large-sample z-test calls significant."""
    return critical_z(alpha) * null_se(n, rate_control, rate_treatment)


def normal_tail_probabilities(delta: float, sigma: float, threshold: float) -> tuple[float, float]:
    """P(X >= threshold) and P(X <= -threshold) for X ~ Normal(delta, sigma^2)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    upper = float(stats.norm.sf((threshold - delta) / sigma))
    lower = float(stats.norm.cdf((-threshold - delta) / sigma))
    return upper, lower


def rejection_probability(delta: float, sigma: float, threshold: float) -> float:
    """P(|X| >= threshold) for X ~ Normal(delta, sigma^2)."""
    upper, lower = normal_tail_probabilities(delta, sigma, threshold)
    return upper + lower


def mean_given_positive_rejection(delta: float, sigma: float, threshold: float) -> float:
    """E[X | X >= threshold] for X ~ Normal(delta, sigma^2).

    This is delta plus sigma times the mills ratio of the standardized threshold.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    standardized = (threshold - delta) / sigma
    probability = float(stats.norm.sf(standardized))
    if probability == 0.0:
        raise ValueError("positive rejection probability is zero")
    return float(delta + sigma * stats.norm.pdf(standardized) / probability)


def mean_given_rejection(delta: float, sigma: float, threshold: float) -> float:
    """E[X | |X| >= threshold] for X ~ Normal(delta, sigma^2)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    upper_z = (threshold - delta) / sigma
    p_upper = float(stats.norm.sf(upper_z))
    upper_mass = delta * p_upper + sigma * float(stats.norm.pdf(upper_z))

    lower_z = (-threshold - delta) / sigma
    p_lower = float(stats.norm.cdf(lower_z))
    lower_mass = delta * p_lower - sigma * float(stats.norm.pdf(lower_z))

    probability = p_upper + p_lower
    if probability == 0.0:
        raise ValueError("rejection probability is zero")
    return float((upper_mass + lower_mass) / probability)


def fixed_sample_power(
    n: int,
    rate_control: float,
    rate_treatment: float,
    alpha: float = 0.05,
) -> float:
    """Normal approximation to the two-sided rejection rate of one look at n per arm."""
    delta = rate_treatment - rate_control
    return rejection_probability(
        delta,
        sampling_sd(n, rate_control, rate_treatment),
        critical_lift(n, rate_control, rate_treatment, alpha),
    )


def fixed_sample_direction_probabilities(
    n: int,
    rate_control: float,
    rate_treatment: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Normal approximation to (correct-direction, wrong-direction) rejection rates.

    Correct means the significant z has the same sign as the true lift.
    """
    delta = rate_treatment - rate_control
    if delta == 0.0:
        raise ValueError("true lift is zero; there is no correct direction")
    upper, lower = normal_tail_probabilities(
        delta,
        sampling_sd(n, rate_control, rate_treatment),
        critical_lift(n, rate_control, rate_treatment, alpha),
    )
    if delta > 0.0:
        return upper, lower
    return lower, upper


def fixed_sample_mean_lift_given_correct(
    n: int,
    rate_control: float,
    rate_treatment: float,
    alpha: float = 0.05,
) -> float:
    """Normal approximation to E[estimated lift | significant in the true direction]."""
    delta = rate_treatment - rate_control
    if delta == 0.0:
        raise ValueError("true lift is zero; there is no correct direction")
    sigma = sampling_sd(n, rate_control, rate_treatment)
    threshold = critical_lift(n, rate_control, rate_treatment, alpha)
    if delta > 0.0:
        return mean_given_positive_rejection(delta, sigma, threshold)
    return -mean_given_positive_rejection(-delta, sigma, threshold)


def _check_sim_shape(n_sims: int, n_looks: int, users_per_look: int) -> None:
    if n_sims < 1 or n_looks < 1 or users_per_look < 1:
        raise ValueError("n_sims, n_looks and users_per_look must be positive")


def simulate_binomial_looks(
    n_sims: int,
    n_looks: int,
    users_per_look: int,
    rate_control: float,
    rate_treatment: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, lift), each with shape (n_sims, n_looks).

    z is the two-proportion z statistic for treatment minus control, using the
    pooled estimated standard error. lift is p_hat_treatment - p_hat_control.
    Look k is cumulative through batch k.
    """
    _check_sim_shape(n_sims, n_looks, users_per_look)
    _validate_rates(rate_control, rate_treatment)

    conv_c = rng.binomial(users_per_look, rate_control, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(users_per_look, rate_treatment, size=(n_sims, n_looks)).cumsum(axis=1)
    n = users_per_look * np.arange(1, n_looks + 1, dtype=float)

    pooled = (conv_c + conv_t) / (2.0 * n)
    se = np.sqrt(pooled * (1.0 - pooled) * (2.0 / n))
    lift = (conv_t - conv_c) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0.0, lift / se, 0.0)
    return z, lift


def simulate_normal_looks(
    n_sims: int,
    n_looks: int,
    users_per_look: int,
    rate_control: float,
    rate_treatment: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Normal-increment version of :func:`simulate_binomial_looks`.

    Batch increments of the conversion-count difference are independent and
    normal with the binomial mean and variance. The z statistic divides by the
    known pooled null standard error, so a single look lines up with
    :func:`fixed_sample_mean_lift_given_correct`.
    """
    _check_sim_shape(n_sims, n_looks, users_per_look)
    _validate_rates(rate_control, rate_treatment)

    delta = rate_treatment - rate_control
    var_batch = users_per_look * (
        rate_control * (1.0 - rate_control) + rate_treatment * (1.0 - rate_treatment)
    )
    increments = rng.normal(
        loc=users_per_look * delta,
        scale=np.sqrt(var_batch),
        size=(n_sims, n_looks),
    )
    cumulative = increments.cumsum(axis=1)
    n = users_per_look * np.arange(1, n_looks + 1, dtype=float)
    lift = cumulative / n

    p_bar = 0.5 * (rate_control + rate_treatment)
    se = np.sqrt(p_bar * (1.0 - p_bar) * 2.0 / n)
    z = lift / se
    return z, lift


def first_stop_index(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Index of the first |z| past the critical value, or -1 if it never happens.

    z has shape (n_sims, n_looks). The returned array has shape (n_sims,).
    """
    if z.ndim != 2:
        raise ValueError("z must be two-dimensional")
    crit = critical_z(alpha)
    significant = np.abs(z) >= crit
    first = np.argmax(significant, axis=1).astype(int)
    first[~significant.any(axis=1)] = -1
    return first


@dataclass(frozen=True)
class StoppingSummary:
    """What naive peeking decides, and the lift it would report.

    A correct stop is one whose first significant z has the same sign as
    ``true_lift``. ``mean_lift_at_final`` averages the planned-end estimate
    over every simulation, stopped or not. ``mean_lift_at_decision`` averages
    the estimate at the first significant look, using the planned end only
    when the test never crosses the line.
    """

    n_sims: int
    n_looks: int
    users_per_look: int
    true_lift: float
    n_correct: int
    n_wrong: int
    n_inconclusive: int
    mean_lift_at_correct_stop: float
    mean_lift_at_any_stop: float
    mean_lift_at_final: float
    mean_lift_at_decision: float
    mean_users_at_decision: float
    n_final_correct: int
    n_final_wrong: int
    n_correct_by_look: np.ndarray
    n_wrong_by_look: np.ndarray
    mean_lift_correct_by_look: np.ndarray

    @property
    def p_correct(self) -> float:
        return self.n_correct / self.n_sims

    @property
    def p_wrong(self) -> float:
        return self.n_wrong / self.n_sims

    @property
    def p_inconclusive(self) -> float:
        return self.n_inconclusive / self.n_sims


def summarize_stopping(
    z: np.ndarray,
    lift: np.ndarray,
    users_per_look: int,
    true_lift: float,
    alpha: float = 0.05,
) -> StoppingSummary:
    """Reduce per-look z and lift paths to stopping-time summaries.

    z and lift are treatment minus control and share shape (n_sims, n_looks).
    ``true_lift`` must be nonzero so that a correct direction exists.
    """
    if z.shape != lift.shape or z.ndim != 2:
        raise ValueError("z and lift must share a two-dimensional shape")
    if users_per_look < 1:
        raise ValueError("users_per_look must be positive")
    if true_lift == 0.0:
        raise ValueError("true_lift must be non-zero")
    n_sims, n_looks = z.shape
    if n_sims < 1 or n_looks < 1:
        raise ValueError("z must have at least one simulation and one look")

    crit = critical_z(alpha)
    first = first_stop_index(z, alpha)
    crossed = first >= 0
    rows = np.arange(n_sims)

    z_at_stop = np.full(n_sims, np.nan)
    lift_at_stop = np.full(n_sims, np.nan)
    if np.any(crossed):
        z_at_stop[crossed] = z[rows[crossed], first[crossed]]
        lift_at_stop[crossed] = lift[rows[crossed], first[crossed]]

    if true_lift > 0.0:
        correct = crossed & (z_at_stop > 0.0)
        wrong = crossed & (z_at_stop < 0.0)
        n_final_correct = int((z[:, -1] >= crit).sum())
        n_final_wrong = int((z[:, -1] <= -crit).sum())
    else:
        correct = crossed & (z_at_stop < 0.0)
        wrong = crossed & (z_at_stop > 0.0)
        n_final_correct = int((z[:, -1] <= -crit).sum())
        n_final_wrong = int((z[:, -1] >= crit).sum())

    def _mean(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(values.mean())

    reported = lift[:, -1].astype(float).copy()
    reported[crossed] = lift_at_stop[crossed]
    users = np.where(crossed, users_per_look * (first + 1), users_per_look * n_looks)

    n_correct_by_look = np.bincount(first[correct], minlength=n_looks)
    n_wrong_by_look = np.bincount(first[wrong], minlength=n_looks)
    lift_sums = np.bincount(first[correct], weights=lift_at_stop[correct], minlength=n_looks)
    mean_by_look = np.full(n_looks, np.nan)
    np.divide(lift_sums, n_correct_by_look, out=mean_by_look, where=n_correct_by_look > 0)

    return StoppingSummary(
        n_sims=n_sims,
        n_looks=n_looks,
        users_per_look=users_per_look,
        true_lift=float(true_lift),
        n_correct=int(correct.sum()),
        n_wrong=int(wrong.sum()),
        n_inconclusive=int((~crossed).sum()),
        mean_lift_at_correct_stop=_mean(lift_at_stop[correct]),
        mean_lift_at_any_stop=_mean(lift_at_stop[crossed]),
        mean_lift_at_final=float(lift[:, -1].mean()),
        mean_lift_at_decision=float(reported.mean()),
        mean_users_at_decision=float(users.mean()),
        n_final_correct=n_final_correct,
        n_final_wrong=n_final_wrong,
        n_correct_by_look=n_correct_by_look,
        n_wrong_by_look=n_wrong_by_look,
        mean_lift_correct_by_look=mean_by_look,
    )


def peeking_with_effect(
    n_sims: int,
    n_looks: int,
    total_users_per_arm: int = 20_000,
    rate_control: float = 0.10,
    rate_treatment: float = 0.105,
    alpha: float = 0.05,
    seed: int = 0,
    model: str = "binomial",
) -> StoppingSummary:
    """Simulate naive peeking at a fixed sample split into equal looks.

    ``model="binomial"`` draws conversions. ``model="normal"`` draws the
    matching normal increments and is the bridge to the truncated-normal
    formulas. The total sample size does not change with the number of looks.
    """
    if total_users_per_arm % n_looks:
        raise ValueError("total_users_per_arm must be divisible by n_looks")
    if model not in ("binomial", "normal"):
        raise ValueError("model must be 'binomial' or 'normal'")
    true_lift = rate_treatment - rate_control
    if true_lift == 0.0:
        raise ValueError("rate_treatment and rate_control must differ")

    rng = np.random.default_rng(seed)
    users_per_look = total_users_per_arm // n_looks
    simulator = simulate_binomial_looks if model == "binomial" else simulate_normal_looks
    z, lift = simulator(n_sims, n_looks, users_per_look, rate_control, rate_treatment, rng)
    return summarize_stopping(z, lift, users_per_look, true_lift, alpha)
