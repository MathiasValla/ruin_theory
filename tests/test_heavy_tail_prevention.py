import numpy as np
import pytest
from ruin_theory import (
    AnnualHeavyTailModel,
    finite_mean_iso_loss_ruin,
    infinite_mean_prevention_multiplier,
    infinite_mean_ruin_time_scale,
    premium_for_infinite_mean_scale,
    simulate_annual_heavy_tail_ruin_times,
    stable_ruin_time_laplace,
)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_time_scale_inverse(alpha):
    premium = premium_for_infinite_mean_scale(100, alpha, 0.2)
    assert infinite_mean_ruin_time_scale(premium, alpha, 0.2) == pytest.approx(100)


def test_half_index_exact_scale_and_laplace():
    assert AnnualHeavyTailModel().scale(50) == pytest.approx(50 / (np.pi * 0.2**2))
    s = np.array([0, 0.1, 1, 10])
    assert stable_ruin_time_laplace(s, 0.5) == pytest.approx(2 / (1 + np.sqrt(1 + 4 * s)))


def test_prevention_factors():
    base = AnnualHeavyTailModel().scale(100)
    assert AnnualHeavyTailModel(frequency_factor=0.8).scale(100) / base == pytest.approx(0.8**-2)
    assert AnnualHeavyTailModel(heavy_severity_factor=0.8).scale(100) / base == pytest.approx(
        0.8**-1
    )
    assert AnnualHeavyTailModel(light_mean=1, light_severity_factor=0.8).scale(
        100
    ) == pytest.approx(base)
    for mechanism, expected in [
        ("frequency", 0.8**-2),
        ("heavy_severity", 0.8**-1),
        ("light_severity", 1),
    ]:
        assert infinite_mean_prevention_multiplier(0.2, 0.5, mechanism=mechanism) == pytest.approx(
            expected
        )


def test_strict_annual_ruin_and_uncensored_result():
    result = simulate_annual_heavy_tail_ruin_times(
        AnnualHeavyTailModel(heavy_probability=1), premium=0.5, n_simulations=40, seed=1
    )
    assert result.ruin_times.tolist() == [1] * 40
    assert result.mean == 1 and result.standard_error == 0
    with pytest.raises(RuntimeError, match="censored"):
        simulate_annual_heavy_tail_ruin_times(
            AnnualHeavyTailModel(), premium=100, n_simulations=50, seed=1, max_steps=1
        )


def test_reproducible_simulation_and_mean_uncertainty():
    options = dict(premium=2, n_simulations=400, seed=18)
    a = simulate_annual_heavy_tail_ruin_times(AnnualHeavyTailModel(), **options)
    b = simulate_annual_heavy_tail_ruin_times(AnnualHeavyTailModel(), **options)
    assert np.array_equal(a.ruin_times, b.ruin_times)
    assert a.standard_error == pytest.approx(a.ruin_times.std(ddof=1) / 20)


@pytest.mark.parametrize("mechanism", ["frequency", "severity"])
def test_pk_zero_reserve_oracle(mechanism):
    result = finite_mean_iso_loss_ruin([0, 5], n_simulations=30000, seed=18, mechanism=mechanism)
    assert np.all(
        abs(result.probabilities[:, 0] - result.traffic_intensities)
        < 4 * result.standard_errors[:, 0]
    )
    assert result.probabilities[1, 1] < result.probabilities[2, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=1),
        dict(heavy_probability=0),
        dict(frequency_factor=6),
        dict(light_family="normal"),
        dict(heavy_severity_factor=-1),
    ],
)
def test_invalid_annual_model(kwargs):
    with pytest.raises(ValueError):
        AnnualHeavyTailModel(**kwargs)


def test_invalid_iso_loss_and_transform():
    with pytest.raises(ValueError):
        finite_mean_iso_loss_ruin([0], prevention_cost=0.65)
    with pytest.raises(ValueError):
        finite_mean_iso_loss_ruin([0], removed_loss=0.25)
    with pytest.raises(ValueError):
        stable_ruin_time_laplace(-1, 0.5)


def test_zero_loss_event_skipping_first_claim_oracle():
    # Every heavy claim exceeds the growing premium boundary by a huge margin.
    result = simulate_annual_heavy_tail_ruin_times(
        AnnualHeavyTailModel(heavy_scale=1e15),
        premium=1,
        n_simulations=10000,
        seed=27,
        max_steps=1000,
    )
    expected = np.random.default_rng(27).geometric(0.2, 10000)
    np.testing.assert_array_equal(result.ruin_times, expected)


def test_positive_light_component_keeps_annual_crossings():
    result = simulate_annual_heavy_tail_ruin_times(
        AnnualHeavyTailModel(light_family="deterministic", light_mean=100),
        premium=0.1,
        n_simulations=100,
        seed=27,
    )
    np.testing.assert_array_equal(result.ruin_times, np.ones(100))
