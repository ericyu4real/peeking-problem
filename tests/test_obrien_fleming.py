"""O'Brien–Fleming boundary: calibration, tables, and power vs Pocock."""

import numpy as np
import pytest

from peeking import false_positive_rate, peeking_with_effect
from peeking.effect import critical_z
from peeking.obrien_fleming import (
    calibrate_obf_c,
    false_positive_rate_obrien_fleming,
    obf_sqrt_multipliers,
    obrien_fleming_final_z,
    obrien_fleming_z_criticals,
    peeking_with_effect_obrien_fleming,
    z_criticals_from_c,
)
from peeking.pocock import peeking_with_effect_pocock, pocock_z_critical
from peeking.simulate import simulate_aa_z

N = 20_000
P_CONTROL = 0.10
P_TREATMENT = 0.105

# Final-look multiplier c for two-sided alpha 0.05 (classical OBF, normal increments).
OBF_FINAL_C = {
    2: 1.987,
    5: 2.039,
    10: 1.992,
}


def test_one_look_obf_is_fixed_sample_critical():
    z = obrien_fleming_z_criticals(1)
    assert z[0] == pytest.approx(critical_z(0.05), abs=1e-9)


def test_obf_multipliers_increase_at_early_looks():
    m = obf_sqrt_multipliers(5)
    assert m[0] > m[-1]
    assert m[-1] == pytest.approx(1.0, abs=1e-9)
    assert z_criticals_from_c(2.0, 5)[0] == pytest.approx(2.0 * np.sqrt(5), abs=1e-9)


def test_obf_final_c_matches_published_table():
    for n_looks in (2, 5):
        c = calibrate_obf_c(n_looks, n_sims=20_000, seed=1)
        assert c == pytest.approx(OBF_FINAL_C[n_looks], abs=0.04)


def test_classical_obf_table_near_nominal_on_binomial_design():
    rng = np.random.default_rng(1)
    z = simulate_aa_z(30_000, 10, 2_000, 0.10, rng)
    crits = z_criticals_from_c(OBF_FINAL_C[10], 10)
    from peeking.simulate import ever_significant

    fpr = float(ever_significant(z, z_critical=crits).mean())
    assert 0.045 <= fpr <= 0.065


def test_obf_fpr_near_nominal_across_looks():
    for n_looks in (1, 2, 5, 10, 20, 50, 100):
        rate = false_positive_rate_obrien_fleming(n_looks=n_looks, n_sims=25_000, seed=2)
        assert rate == pytest.approx(0.05, abs=0.012)


def test_obf_first_look_stricter_than_final():
    for n_looks in (5, 10, 20):
        z = obrien_fleming_z_criticals(n_looks)
        pocock = pocock_z_critical(n_looks)
        assert z[0] > z[-1]
        assert z[-1] < pocock


def test_obf_more_powerful_than_pocock_many_looks():
    common = dict(
        n_sims=15_000,
        n_looks=20,
        total_users_per_arm=N,
        rate_control=P_CONTROL,
        rate_treatment=P_TREATMENT,
        seed=4,
    )
    obf = peeking_with_effect_obrien_fleming(**common)
    pocock = peeking_with_effect_pocock(**common)
    naive = peeking_with_effect(**common)
    assert obf.p_correct > pocock.p_correct + 0.03
    assert obf.p_correct < naive.p_correct + 0.01


def test_ten_looks_obf_between_naive_and_pocock_fpr():
    naive = false_positive_rate(n_looks=10, n_sims=30_000, seed=0)
    obf = false_positive_rate_obrien_fleming(n_looks=10, n_sims=30_000, seed=0)
    assert naive == pytest.approx(0.193, abs=0.015)
    assert obf == pytest.approx(0.05, abs=0.012)


@pytest.mark.parametrize("n_looks", [0, -1])
def test_invalid_n_looks_raises(n_looks):
    with pytest.raises(ValueError):
        calibrate_obf_c(n_looks)
