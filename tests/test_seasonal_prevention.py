import numpy as np
import pytest

from ruin_theory import (
    PreventionResponseCurve,
    compare_seasonal_prevention,
    deterministic,
    exponential,
    integrate_seasonal_pressure,
    optimize_seasonal_prevention,
    optimize_seasonal_lundberg,
    seasonal_lundberg_coefficient,
    seasonal_exponential_profile_root,
    calibrate_storm_loss_days,
)


@pytest.mark.parametrize("family", ["exponential", "quadratic", "reciprocal"])
@pytest.mark.parametrize("convention", ["amount", "rate"])
@pytest.mark.parametrize("budget", [None, 0.3])
def test_analytic_agrees_with_independent_numerical_solver(family, convention, budget):
    response = PreventionResponseCurve(family, shape=2, offset=2)
    options = dict(
        max_prevention=1.5, durations=[0.2, 0.3, 0.5], budget=budget, budget_convention=convention
    )
    exact = optimize_seasonal_prevention([0.4, 1.2, 0.8], response=response, **options)
    numeric = optimize_seasonal_prevention(
        [0.4, 1.2, 0.8], response=lambda p: response(p), **options
    )
    assert numeric.total_outflow == pytest.approx(exact.total_outflow, abs=2e-7)
    assert numeric.amounts == pytest.approx(exact.amounts, abs=5e-5)


def test_free_budget_closed_form_includes_severity_weights():
    result = optimize_seasonal_prevention(
        [0.1, 0.9], response=PreventionResponseCurve(shape=2), max_prevention=2
    )
    assert result.amounts == pytest.approx([0, np.log(3.6) / 2])
    assert result.budget_spent == pytest.approx(np.log(3.6) / 4)


def test_budget_ceiling_is_not_a_pointwise_cap():
    result = optimize_seasonal_prevention(
        [3, 1], response=PreventionResponseCurve(), max_prevention=2, budget_ceiling=0.1
    )
    assert result.budget_spent == pytest.approx(0.1)
    assert result.amounts == pytest.approx([0.2, 0])


@pytest.mark.parametrize(
    "weights,budget,expected",
    [([0, 0], 0.5, [0.5, 0.5]), ([1, 0], 0.75, [1, 0.5]), ([1, 2], 0, [0, 0]), ([1, 2], 1, [1, 1])],
)
def test_allocation_boundaries(weights, budget, expected):
    result = optimize_seasonal_prevention(
        weights, response=PreventionResponseCurve(), max_prevention=1, budget=budget
    )
    assert result.amounts == pytest.approx(expected)


def test_strong_exponential_fixed_budget_does_not_underflow_multiplier():
    result = optimize_seasonal_prevention(
        [1, 2], response=PreventionResponseCurve(shape=2000), max_prevention=2, budget=1
    )
    assert result.budget_spent == pytest.approx(1)


def test_lag_shifts_benefit_not_cost():
    direct = optimize_seasonal_prevention(
        [1, 4, 2], response=PreventionResponseCurve(), max_prevention=2, budget=0.4
    )
    delayed = optimize_seasonal_prevention(
        [1, 4, 2], response=PreventionResponseCurve(), max_prevention=2, budget=0.4, lag_steps=1
    )
    assert delayed.effective_amounts == pytest.approx(direct.amounts)
    assert delayed.total_outflow == pytest.approx(direct.total_outflow)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(budget=2),
        dict(budget=-1),
        dict(budget_convention="unknown"),
        dict(durations=[0.1, 0.1]),
        dict(lag_steps=1, durations=[0.2, 0.8]),
    ],
)
def test_allocation_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        optimize_seasonal_prevention(
            [1, 2], response=PreventionResponseCurve(), max_prevention=1, **kwargs
        )


def test_quadrature_continuous_profiles():
    weights = integrate_seasonal_pressure(
        lambda t: 2 + np.cos(2 * np.pi * t), breaks=[0, 0.5, 1], severity_mean=lambda t: 3
    )
    assert weights == pytest.approx([3, 3])
    with pytest.raises(ValueError):
        integrate_seasonal_pressure(lambda t: -1, breaks=[0, 1], severity_mean=lambda t: -1)


def test_seasonal_root_reduces_to_homogeneous_exponential():
    result = seasonal_lundberg_coefficient(
        [1, 1],
        [exponential(2), exponential(2)],
        prevention=[0, 0],
        response=lambda p: 1,
        premium_rate=1,
    )
    assert result == pytest.approx(1, abs=1e-8)


def test_exact_drift_crossing_and_phase():
    result = compare_seasonal_prevention(
        [0, 0],
        [deterministic(0)] * 2,
        {"reference": [0, 0], "spent": [2, 0]},
        response=lambda p: 1,
        premium_rate=0.5,
        initial_capital=0.1,
        horizon=1,
        n_simulations=20,
        starting_phase=0,
        seed=10,
    )
    assert result.estimates[0].probability == 0
    assert result.estimates[1].ruin_times == pytest.approx(np.full(20, 0.1 / 1.5))


def test_common_random_numbers_and_poisson_oracle():
    result = compare_seasonal_prevention(
        [2],
        [deterministic(1)],
        {"a": [0], "b": [0]},
        response=lambda p: 1,
        premium_rate=0,
        initial_capital=0,
        horizon=1,
        n_simulations=4000,
        seed=19,
    )
    p = result.estimates[0].probability
    assert abs(p - (1 - np.exp(-2))) < 4 * result.estimates[0].standard_error
    assert np.array_equal(result.ruined[0], result.ruined[1])
    assert result.gain_standard_errors.tolist() == [0, 0]


def test_lagged_simulation_identical_policies():
    result = compare_seasonal_prevention(
        [1, 2],
        [exponential(1)] * 2,
        {"a": [0, 0.2], "b": [0, 0.2]},
        response=PreventionResponseCurve(),
        premium_rate=1,
        initial_capital=0,
        horizon=2,
        n_simulations=50,
        lag_steps=1,
        seed=2,
    )
    assert np.array_equal(result.ruined[0], result.ruined[1])


@pytest.mark.parametrize("profile", ["triangular", "cosine"])
def test_closed_profile_root_against_mesh(profile):
    m = 300
    t = (np.arange(m) + 0.5) / m
    eta = 2 + 0.4 * (np.minimum(t, 1 - t) if profile == "triangular" else np.cos(2 * np.pi * t))
    exact = seasonal_exponential_profile_root(
        base_rate=2, variation=0.4, frequency=1, premium_rate=1, profile=profile
    )
    approximate = seasonal_lundberg_coefficient(
        np.ones(m),
        [exponential(x) for x in eta],
        prevention=np.zeros(m),
        response=lambda p: 1,
        premium_rate=1,
    )
    assert approximate == pytest.approx(exact, abs=2e-6)


def test_storm_calibration_retains_all_positive_days():
    result = calibrate_storm_loss_days(
        [0, 1, 30] + [0] * 9, np.arange(1, 13), [2000] * 12, retained_loss_cap=2, monetary_unit=1
    )
    assert result.frequencies.tolist() == [0, 12, 12] + [0] * 9
    assert result.loss_weights.sum() == 3
    assert result.n_years == 1
    with pytest.raises(ValueError, match="twelve"):
        calibrate_storm_loss_days([1], [1], [2000], retained_loss_cap=2)


def test_generic_zero_cap():
    result = optimize_seasonal_prevention([1, 2], response=lambda p: np.exp(-p), max_prevention=0)
    assert result.total_outflow == 3


def test_recorded_paths_include_claim_jumps_and_terminal_value():
    result = compare_seasonal_prevention(
        [3],
        [deterministic(1)],
        {"policy": [0]},
        response=lambda p: 1,
        premium_rate=1,
        initial_capital=1,
        horizon=2,
        n_simulations=2,
        record_paths=1,
        seed=1,
    )
    times, reserves = result.trajectories["policy"][0]
    assert times[0] == 0 and times[-1] == 2
    assert np.all(np.diff(times) >= 0)
    assert reserves[-1] == result.terminal_reserves[0, 0]


@pytest.mark.parametrize("budget", [None, 0.08])
def test_lundberg_optimum_agrees_with_loss_optimum_for_common_severity(budget):
    rates = np.array([1, 3, 2, 4.0])
    curve = PreventionResponseCurve(shape=2)
    result = optimize_seasonal_lundberg(
        rates,
        [exponential(2)] * 4,
        response=curve,
        premium_rate=3,
        max_prevention=0.4,
        budget=budget,
    )
    assert abs(result.equation_residual) < 1e-9
    assert result.coefficient == pytest.approx(
        seasonal_lundberg_coefficient(
            rates,
            [exponential(2)] * 4,
            prevention=result.allocation.amounts,
            response=curve,
            premium_rate=3,
        )
    )
    if budget is not None:
        expected = optimize_seasonal_prevention(
            rates / 8, response=curve, max_prevention=0.4, budget=budget
        )
        np.testing.assert_allclose(result.allocation.amounts, expected.amounts, atol=1e-9)


def test_lundberg_optimum_heterogeneous_severity_against_dense_grid():
    curve = PreventionResponseCurve(shape=2)
    rates, laws = [2, 2], [exponential(2), exponential(3)]
    result = optimize_seasonal_lundberg(
        rates,
        laws,
        response=curve,
        premium_rate=1.4,
        max_prevention=0.4,
        budget=0.2,
    )
    values = [
        seasonal_lundberg_coefficient(
            rates, laws, prevention=[p, 0.4 - p], response=curve, premium_rate=1.4
        )
        for p in np.linspace(0, 0.4, 81)
    ]
    assert result.coefficient >= max(values) - 1e-9


def test_free_lundberg_ceiling_and_invalid_inputs():
    result = optimize_seasonal_lundberg(
        [2, 2],
        [exponential(2)] * 2,
        response=PreventionResponseCurve(shape=2),
        premium_rate=2,
        max_prevention=0.4,
        budget_ceiling=0.1,
    )
    assert result.allocation.budget_spent <= 0.1 + 1e-10
    with pytest.raises(ValueError, match="net profit"):
        optimize_seasonal_lundberg(
            [20],
            [exponential(1)],
            response=PreventionResponseCurve(),
            premium_rate=1,
            max_prevention=0.1,
        )


def test_minimum_investment_constraint_and_numerical_response():
    weights = np.array([0, 0.1, 0.4, 0.8])
    curve = PreventionResponseCurve(shape=2)
    options = dict(max_prevention=0.25, min_prevention=0.03, budget=0.08)
    result = optimize_seasonal_prevention(weights, response=curve, **options)
    numerical = optimize_seasonal_prevention(weights, response=lambda p: np.exp(-2 * p), **options)
    assert np.all(result.amounts >= 0.03)
    assert result.budget_spent == pytest.approx(0.08)
    np.testing.assert_allclose(result.amounts, numerical.amounts, atol=1e-6)
    with pytest.raises(ValueError, match="cannot fund"):
        optimize_seasonal_prevention(
            weights, response=curve, max_prevention=0.25, min_prevention=0.1, budget=0.08
        )


def test_affine_profile_root_against_mesh():
    eta = 2 + 0.4 * (np.arange(400) + 0.5) / 400
    root = seasonal_lundberg_coefficient(
        np.ones(400),
        [exponential(x) for x in eta],
        prevention=np.zeros(400),
        response=lambda p: 1,
        premium_rate=1,
    )
    exact = seasonal_exponential_profile_root(
        base_rate=2, variation=0.4, frequency=1, premium_rate=1, profile="affine"
    )
    assert root == pytest.approx(exact, abs=1e-7)


def test_free_root_optimum_is_not_free_expected_loss_optimum():
    curve = PreventionResponseCurve(shape=2)
    loss = optimize_seasonal_prevention([0.3], response=curve, max_prevention=0.4)
    root = optimize_seasonal_lundberg(
        [0.6], [exponential(2)], response=curve, premium_rate=0.7, max_prevention=0.4
    )
    assert loss.amounts[0] == 0
    assert root.allocation.amounts[0] == pytest.approx(0.2, abs=1e-9)
    assert root.coefficient == pytest.approx(2 - 0.6 * np.exp(-0.4) / 0.5)


def test_zero_controlled_pressure_has_no_finite_optimal_root():
    with pytest.raises(ValueError, match="no finite"):
        optimize_seasonal_lundberg(
            [1],
            [exponential(2)],
            response=PreventionResponseCurve("quadratic"),
            premium_rate=2,
            max_prevention=1,
            budget=1,
        )
