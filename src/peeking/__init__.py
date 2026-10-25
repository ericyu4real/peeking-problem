"""How much does peeking at an A/B test inflate false positives?"""

from peeking.effect import (
    critical_lift,
    fixed_sample_direction_probabilities,
    fixed_sample_mean_lift_given_correct,
    fixed_sample_power,
    peeking_with_effect,
)
from peeking.bonferroni import (
    false_positive_rate_bonferroni,
    peeking_with_effect_bonferroni,
    per_look_alpha,
)
from peeking.simulate import ever_significant, false_positive_rate, simulate_aa_z

__all__ = [
    "critical_lift",
    "ever_significant",
    "false_positive_rate",
    "false_positive_rate_bonferroni",
    "peeking_with_effect_bonferroni",
    "per_look_alpha",
    "fixed_sample_direction_probabilities",
    "fixed_sample_mean_lift_given_correct",
    "fixed_sample_power",
    "peeking_with_effect",
    "simulate_aa_z",
]
