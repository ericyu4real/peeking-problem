import numpy as np
import pytest

from peeking import ever_significant, false_positive_rate, simulate_aa_z


def test_single_look_matches_nominal_alpha():
    assert false_positive_rate(n_looks=1, n_sims=40_000) == pytest.approx(0.05, abs=0.005)


def test_ten_looks_matches_classical_result():
    # Armitage, McPherson & Rowe (1969): ten equally spaced looks at alpha=0.05 give ~19%.
    assert false_positive_rate(n_looks=10, n_sims=40_000) == pytest.approx(0.193, abs=0.01)


def test_more_looks_never_lowers_false_positives():
    rates = [false_positive_rate(n_looks=k, n_sims=10_000, seed=1) for k in (1, 5, 20, 100)]
    assert rates == sorted(rates)
    assert rates[-1] > 0.3


def test_z_shape_and_cumulative_looks():
    z = simulate_aa_z(7, 4, 100, 0.2, np.random.default_rng(0))
    assert z.shape == (7, 4)
    assert np.isfinite(z).all()


def test_ever_significant_uses_absolute_value():
    z = np.array([[0.0, -2.5], [1.0, 1.5], [1.96, 0.0]])
    assert ever_significant(z).tolist() == [True, False, True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_sims": 0, "n_looks": 1, "users_per_look": 1, "base_rate": 0.1},
        {"n_sims": 1, "n_looks": 1, "users_per_look": 1, "base_rate": 1.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        simulate_aa_z(rng=np.random.default_rng(0), **kwargs)


def test_uneven_split_raises():
    with pytest.raises(ValueError):
        false_positive_rate(n_looks=3, total_users_per_arm=10)
