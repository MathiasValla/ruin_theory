import numpy as np
import pytest

from ruin_theory import (
    AllocationGridResult,
    CramerLundbergProcess,
    PreventionProgram,
    RedTimeCurveResult,
    RedTimeEstimate,
    RedTimePathMetrics,
    ReserveAllocationResult,
    deterministic,
    estimate_red_time_curve,
    estimate_red_time_metrics,
    evaluate_reserve_allocation_grid,
    expected_negative_area_exponential,
    expected_time_in_red_exponential,
    multiline_red_time_metrics_from_paths,
    multirisk_red_time_derivative_sum,
    negative_area_derivative_identity_error,
    optimize_reserve_allocation,
    red_time_derivative,
    red_time_metrics_from_path,
    simplex_reserve_grid,
)
from ruin_theory.results import SimulationPath


def _path(times, reserves):
    return SimulationPath(
        times=np.asarray(times, dtype=float),
        reserves=np.asarray(reserves, dtype=float),
        claim_times=np.empty(0),
        claim_sizes=np.empty(0),
        ruin_time=None,
        horizon=float(times[-1]),
        initial_capital=float(reserves[0]),
        premium_rate=0.0,
    )


def test_red_time_metrics_from_path_are_exact_for_piecewise_linear_segments():
    path = _path([0.0, 1.0, 2.0, 3.0], [1.0, -1.0, -2.0, 2.0])

    metrics = red_time_metrics_from_path(path)

    assert isinstance(metrics, RedTimePathMetrics)
    assert metrics.time_in_red == pytest.approx(2.0)
    assert metrics.negative_area == pytest.approx(2.25)
    assert metrics.minimum_reserve == pytest.approx(-2.0)

    shifted = red_time_metrics_from_path(path, reserve_shift=2.0)
    assert shifted.time_in_red == 0.0
    assert shifted.negative_area == 0.0


def test_negative_area_derivative_identity_matches_loisel_theorem():
    path = _path([0.0, 1.0, 2.0, 3.0], [1.0, -1.0, -2.0, 2.0])

    error = negative_area_derivative_identity_error(path, reserve_shift=0.2, step=1e-5)

    assert error == pytest.approx(0.0, abs=1e-8)


def test_infinite_horizon_exponential_formulas_match_loisel_closed_form():
    u = np.array([0.0, 1.0, 2.0])
    premium = 2.0
    arrival = 1.0
    claim_rate = 1.0
    adjustment = claim_rate - arrival / premium

    red_time = expected_time_in_red_exponential(
        u,
        premium_rate=premium,
        claim_arrival_rate=arrival,
        claim_rate=claim_rate,
    )
    negative_area = expected_negative_area_exponential(
        u,
        premium_rate=premium,
        claim_arrival_rate=arrival,
        claim_rate=claim_rate,
    )

    np.testing.assert_allclose(
        red_time,
        arrival * np.exp(-adjustment * u) / (premium * premium * adjustment**2),
    )
    np.testing.assert_allclose(negative_area, red_time / adjustment)
    derivative = (
        expected_negative_area_exponential(
            1.0 + 1e-5,
            premium_rate=premium,
            claim_arrival_rate=arrival,
            claim_rate=claim_rate,
        )
        - expected_negative_area_exponential(
            1.0 - 1e-5,
            premium_rate=premium,
            claim_arrival_rate=arrival,
            claim_rate=claim_rate,
        )
    ) / 2e-5
    assert derivative == pytest.approx(
        -expected_time_in_red_exponential(
            1.0,
            premium_rate=premium,
            claim_arrival_rate=arrival,
            claim_rate=claim_rate,
        ),
        rel=1e-10,
    )

    with pytest.raises(ValueError, match="positive safety loading"):
        expected_time_in_red_exponential(
            1.0,
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            claim_rate=1.0,
        )


def test_multiline_red_time_metrics_compute_positive_total_penalty():
    line_1 = _path([0.0, 1.0, 2.0], [1.0, -1.0, 1.0])
    line_2 = _path([0.0, 2.0], [2.0, 2.0])

    metrics = multiline_red_time_metrics_from_paths([line_1, line_2])

    assert metrics.time_in_red[0] == pytest.approx(1.0)
    assert metrics.time_in_red[1] == pytest.approx(0.0)
    assert metrics.red_time_with_positive_total[0] == pytest.approx(1.0)
    assert metrics.positive_total_red_time_sum == pytest.approx(1.0)
    assert metrics.aggregate_negative_area == pytest.approx(0.5)


def test_estimate_red_time_metrics_and_curve_are_zero_without_claims():
    model = CramerLundbergProcess(
        initial_capital=0.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        claim_distribution=deterministic(1.0),
        prevention=PreventionProgram(frequency_multiplier=0.0),
    )

    estimate = estimate_red_time_metrics(model, horizon=2.0, n_simulations=4, seed=123)
    curve = estimate_red_time_curve(model, [0.0, 1.0], horizon=2.0, n_simulations=3, seed=123)

    assert isinstance(estimate, RedTimeEstimate)
    assert estimate.expected_time_in_red == 0.0
    assert estimate.expected_negative_area == 0.0
    assert isinstance(curve, RedTimeCurveResult)
    np.testing.assert_allclose(curve.expected_time_in_red, [0.0, 0.0])
    np.testing.assert_allclose(curve.expected_negative_area, [0.0, 0.0])


def test_optimize_reserve_allocation_equalizes_active_red_times():
    red = (lambda u: np.exp(-u), lambda u: np.exp(-u))
    area = (lambda u: np.exp(-u), lambda u: np.exp(-u))

    result = optimize_reserve_allocation(
        total_reserve=4.0,
        red_time_functions=red,
        negative_area_functions=area,
        tolerance=1e-10,
    )

    assert isinstance(result, ReserveAllocationResult)
    np.testing.assert_allclose(result.allocations, [2.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(result.red_times, [np.exp(-2.0), np.exp(-2.0)], atol=1e-6)
    assert result.objective_value == pytest.approx(2.0 * np.exp(-2.0), abs=1e-6)
    assert result.converged


def test_optimize_reserve_allocation_leaves_safe_lines_inactive():
    red = (
        lambda u: 0.1 * np.exp(-u),
        lambda u: np.exp(-u),
        lambda u: np.exp(-u),
    )

    result = optimize_reserve_allocation(
        total_reserve=1.0,
        red_time_functions=red,
        tolerance=1e-10,
    )

    np.testing.assert_allclose(result.allocations, [0.0, 0.5, 0.5], atol=1e-6)
    np.testing.assert_array_equal(result.active, [False, True, True])
    assert result.red_times[1] == pytest.approx(result.red_times[2], abs=1e-6)


def test_simplex_grid_and_allocation_objective_are_vectorized():
    grid = simplex_reserve_grid(total_reserve=2.0, n_lines=3, subdivisions=2)
    functions = (lambda u: np.exp(-u), lambda u: np.exp(-u), lambda u: np.exp(-u))

    result = evaluate_reserve_allocation_grid(
        grid,
        functions,
        red_time_functions=functions,
    )

    assert isinstance(result, AllocationGridResult)
    assert grid.shape == (6, 3)
    np.testing.assert_allclose(np.sum(grid, axis=1), 2.0)
    np.testing.assert_allclose(result.objective_values, np.sum(result.negative_areas, axis=1))
    assert result.red_times is not None


def test_red_time_derivative_helpers_validate_and_compute():
    derivative = red_time_derivative(lambda u: np.exp(-u), 1.0, step=1e-5)
    total = multirisk_red_time_derivative_sum(
        [1.0, 2.0],
        (lambda u: np.exp(-u), lambda u: np.exp(-u)),
        step=1e-5,
    )

    assert derivative == pytest.approx(-np.exp(-1.0), rel=1e-8)
    assert total == pytest.approx(-np.exp(-1.0) - np.exp(-2.0), rel=1e-8)

    with pytest.raises(ValueError, match="red_time_functions"):
        multirisk_red_time_derivative_sum([1.0], ())
