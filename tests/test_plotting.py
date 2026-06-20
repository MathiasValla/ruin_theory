"""Plotting diagnostics tests."""

# ruff: noqa: E402

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from ruin_theory.plotting import (
    plot_barrier_comparison,
    plot_barrier_dividend_distribution,
    plot_barrier_dividend_path,
    plot_barrier_ruin_time_distribution,
    plot_integer_byclaim_counts,
    plot_integer_byclaim_path,
    plot_maximum_before_default_hazard,
    plot_deficit_at_ruin,
    plot_discrete_time_deficit_cdf,
    plot_discrete_time_surplus_cdf,
    plot_finite_time_appell_coefficients,
    plot_finite_time_discrete_boundary,
    plot_finite_time_discrete_computation_set,
    plot_finite_time_discrete_survival,
    plot_finite_time_lundberg_bounds,
    plot_gerber_shiu_scatter,
    plot_path,
    plot_paths,
    plot_periodic_pressure,
    plot_prevention_calendar,
    plot_red_time_allocation,
    plot_red_time_curve,
    plot_ruin_curve,
    plot_ruin_time_histogram,
    plot_simplex_allocation_surface,
    plot_two_line_allocation_curve,
    plot_surplus_before_ruin,
    plot_terminal_reserve_distribution,
    plot_win_first_sensitivity,
    plot_win_first_surface,
)
from ruin_theory import (
    BINARByClaimModel,
    INARByClaimModel,
    RedTimeCurveResult,
    deterministic,
    evaluate_reserve_allocation_grid,
    finite_time_ruin_discrete_appell,
    finite_time_ruin_discrete_boundary,
    finite_time_ruin_discrete_inventory,
    finite_time_ruin_discrete_nonhomogeneous_boundary,
    finite_time_ruin_discrete,
    finite_time_discrete_time_ruin,
    finite_time_lundberg_bounds,
    gerber_shiu_from_paths,
    optimize_reserve_allocation,
    simulate_binar_byclaim_path,
    simulate_barrier_dividend_path,
    simulate_inar_byclaim_path,
    simplex_reserve_grid,
)
from ruin_theory.prevention import optimize_periodic_prevention_calendar
from ruin_theory.results import RuinEstimate, SimulationPath


def _path(*, horizon: float = 3.0, ruin_time: float | None = 1.0) -> SimulationPath:
    return SimulationPath(
        times=np.array([0.0, 1.0, 1.0, 2.0]),
        reserves=np.array([2.0, 3.0, -0.5, 0.5]),
        claim_times=np.array([1.0]),
        claim_sizes=np.array([3.5]),
        ruin_time=ruin_time,
        horizon=horizon,
        initial_capital=2.0,
        premium_rate=1.0,
    )


def _estimate(ruin_times: np.ndarray, *, horizon: float = 3.0) -> RuinEstimate:
    finite = np.isfinite(ruin_times)
    probability = float(np.mean(finite))
    return RuinEstimate(
        probability=probability,
        standard_error=0.0,
        ci_low=probability,
        ci_high=probability,
        n_simulations=ruin_times.size,
        horizon=horizon,
        ruin_times=ruin_times,
    )


def _gerber_shiu_path(deficit: float = 1.0, ruin_time: float = 1.0) -> SimulationPath:
    surplus = 2.0
    return SimulationPath(
        times=np.array([0.0, ruin_time, ruin_time]),
        reserves=np.array([1.0, surplus, -deficit]),
        claim_times=np.array([ruin_time]),
        claim_sizes=np.array([surplus + deficit]),
        ruin_time=ruin_time,
        horizon=3.0,
        initial_capital=1.0,
        premium_rate=1.0,
    )


def test_plot_path_marks_reserve_and_ruin_time():
    fig, ax = plt.subplots()
    try:
        result = plot_path(_path(), ax=ax)

        assert result is ax
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "reserve"
        assert ax.get_title() == "Reserve trajectory"
        assert ax.get_xlim() == pytest.approx((0.0, 3.0))
        assert len(ax.lines) == 3
    finally:
        plt.close(fig)


def test_plot_paths_overlays_paths_and_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        plot_paths([])

    fig, ax = plt.subplots()
    try:
        result = plot_paths([_path(horizon=2.0), _path(horizon=4.0, ruin_time=None)], ax=ax)

        assert result is ax
        assert ax.get_title() == "Simulated reserve trajectories"
        assert ax.get_xlim() == pytest.approx((0.0, 4.0))
        assert len(ax.lines) == 3
    finally:
        plt.close(fig)


def test_plot_ruin_curve_validates_and_labels_probability_curve():
    fig, ax = plt.subplots()
    try:
        result = plot_ruin_curve(
            [0.0, 1.0, 2.0],
            [0.8, 0.5, 0.2],
            ax=ax,
            label="ultimate",
            ci_low=[0.7, 0.4, 0.1],
            ci_high=[0.9, 0.6, 0.3],
        )

        assert result is ax
        assert ax.get_xlabel() == "initial surplus"
        assert ax.get_ylabel() == "ruin probability"
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
        assert ax.get_legend() is not None
        assert len(ax.collections) == 1
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.8, 0.5, 0.2])
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="matching shapes"):
        plot_ruin_curve([0.0, 1.0], [0.5])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_ruin_curve([0.0], [1.2])

    with pytest.raises(ValueError, match="provided together"):
        plot_ruin_curve([0.0], [0.5], ci_low=[0.4])

    with pytest.raises(ValueError, match="less than or equal"):
        plot_ruin_curve([0.0], [0.5], ci_low=[0.6], ci_high=[0.4])


def test_plot_win_first_surface_hazard_and_sensitivity():
    surplus = np.array([0.0, 1.0, 2.0])
    gain = np.array([0.5, 1.0])
    probabilities = np.array([[0.7, 0.5], [0.8, 0.6], [0.9, 0.7]])

    fig, axes = plt.subplots(1, 3)
    try:
        surface = plot_win_first_surface(surplus, gain, probabilities, ax=axes[0])
        hazard = plot_maximum_before_default_hazard(
            surplus,
            [0.3, 0.2, 0.1],
            ax=axes[1],
            label="base",
        )
        sensitivity = plot_win_first_sensitivity(
            [0.0, 0.05, 0.1],
            [0.5, 0.6, 0.7],
            parameter_name="interest force",
            ax=axes[2],
        )

        assert surface is axes[0]
        assert hazard is axes[1]
        assert sensitivity is axes[2]
        assert axes[0].get_ylabel() == "initial surplus"
        assert axes[1].get_title() == "Maximum-before-default hazard"
        assert axes[2].get_xlabel() == "interest force"
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="shape"):
        plot_win_first_surface(surplus, gain, probabilities[:, :1])
    with pytest.raises(ValueError, match="non-negative"):
        plot_maximum_before_default_hazard([0.0], [-0.1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_win_first_sensitivity([1.0], [1.2])


def test_plot_barrier_dividend_diagnostics():
    path = simulate_barrier_dividend_path(
        deterministic(2.0),
        initial_capital=1.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        barrier=1.0,
        seed=123,
    )

    fig, axes = plt.subplots(2, 2)
    try:
        path_axis = plot_barrier_dividend_path(path, ax=axes[0, 0])
        dividend_axis = plot_barrier_dividend_distribution([0.0, 1.0, 2.0], ax=axes[0, 1])
        ruin_axis = plot_barrier_ruin_time_distribution([1.0, 2.0, np.inf], ax=axes[1, 0])
        comparison_axis = plot_barrier_comparison(
            [1.0, 2.0, 3.0],
            [0.5, 0.8, 0.7],
            ax=axes[1, 1],
        )

        assert path_axis is axes[0, 0]
        assert dividend_axis.get_xlabel() == "cumulative dividends"
        assert ruin_axis.get_title() == "Dividend-barrier ruin times"
        assert comparison_axis.get_ylabel() == "expected dividends"
    finally:
        plt.close(fig)

    with pytest.raises(TypeError, match="BarrierDividendPath"):
        plot_barrier_dividend_path(object())
    with pytest.raises(ValueError, match="non-negative"):
        plot_barrier_dividend_distribution([-1.0])
    with pytest.raises(ValueError, match="positive"):
        plot_barrier_comparison([0.0], [1.0])


def test_plot_red_time_and_allocation_diagnostics():
    curve = RedTimeCurveResult(
        initial_capitals=np.array([0.0, 1.0, 2.0]),
        expected_time_in_red=np.array([1.0, 0.5, 0.25]),
        expected_negative_area=np.array([2.0, 0.7, 0.2]),
        time_in_red_standard_error=np.zeros(3),
        negative_area_standard_error=np.zeros(3),
        n_simulations=10,
        horizon=3.0,
    )
    allocation = optimize_reserve_allocation(
        total_reserve=2.0,
        red_time_functions=(lambda u: np.exp(-u), lambda u: np.exp(-u)),
        negative_area_functions=(lambda u: np.exp(-u), lambda u: np.exp(-u)),
    )
    two_line_grid = simplex_reserve_grid(total_reserve=2.0, n_lines=2, subdivisions=4)
    two_line = evaluate_reserve_allocation_grid(
        two_line_grid,
        (lambda u: np.exp(-u), lambda u: np.exp(-u)),
    )
    simplex_grid = simplex_reserve_grid(total_reserve=2.0, n_lines=3, subdivisions=2)
    simplex = evaluate_reserve_allocation_grid(
        simplex_grid,
        (lambda u: np.exp(-u), lambda u: np.exp(-u), lambda u: np.exp(-u)),
    )

    fig, axes = plt.subplots(2, 2)
    try:
        curve_axis = plot_red_time_curve(curve, ax=axes[0, 0])
        allocation_axis = plot_red_time_allocation(allocation, ax=axes[0, 1])
        two_line_axis = plot_two_line_allocation_curve(two_line, ax=axes[1, 0])
        simplex_axis = plot_simplex_allocation_surface(simplex, ax=axes[1, 1])

        assert curve_axis.get_title() == "Time in red"
        assert allocation_axis.get_ylabel() == "allocated reserve"
        assert two_line_axis.get_xlabel() == "line 1 reserve"
        assert simplex_axis.get_title() == "Allocation simplex objective"
    finally:
        plt.close(fig)

    with pytest.raises(TypeError, match="RedTimeCurveResult"):
        plot_red_time_curve(object())
    with pytest.raises(TypeError, match="ReserveAllocationResult"):
        plot_red_time_allocation(object())
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        plot_two_line_allocation_curve(simplex)
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        plot_simplex_allocation_surface(two_line)


def test_plot_terminal_reserve_distribution_marks_zero_and_quantiles():
    fig, ax = plt.subplots()
    try:
        result = plot_terminal_reserve_distribution(
            [-1.0, 0.0, 1.0, 2.0, 4.0],
            ax=ax,
            bins=3,
            quantiles=[0.25, 0.5, 0.75],
        )

        assert result is ax
        assert ax.get_xlabel() == "terminal reserve"
        assert ax.get_ylabel() == "frequency"
        assert ax.get_title() == "Terminal reserve distribution"
        assert ax.get_legend() is not None
        assert len(ax.patches) == 3
        assert len(ax.lines) == 4
        line_positions = [line.get_xdata()[0] for line in ax.lines]
        np.testing.assert_allclose(line_positions, [0.0, 0.0, 1.0, 2.0])
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="bins must be positive"):
        plot_terminal_reserve_distribution([1.0], bins=0)

    with pytest.raises(ValueError, match="finite"):
        plot_terminal_reserve_distribution([1.0, np.nan])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_terminal_reserve_distribution([1.0], quantiles=[1.2])


def test_plot_ruin_time_histogram_handles_ruined_and_unruined_samples():
    fig, ax = plt.subplots()
    try:
        result = plot_ruin_time_histogram(_estimate(np.array([0.5, 1.0, np.inf, 2.0])), ax=ax)

        assert result is ax
        assert ax.get_xlabel() == "time to ruin"
        assert ax.get_ylabel() == "frequency"
        assert ax.get_xlim() == pytest.approx((0.0, 3.0))
        assert len(ax.patches) > 0
    finally:
        plt.close(fig)


def test_plot_finite_time_discrete_diagnostics():
    result = finite_time_ruin_discrete(
        [0.0, 1.0],
        initial_capital=0.5,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=1.0,
        method="inventory",
        return_result=True,
    )

    fig, axes = plt.subplots(1, 2)
    try:
        survival_axis = plot_finite_time_discrete_survival(result, ax=axes[0], label="survival")
        set_axis = plot_finite_time_discrete_computation_set(
            initial_capital=5,
            premium_units=10,
            method="picard-lefevre",
            ax=axes[1],
        )

        assert survival_axis.get_ylabel() == "non-ruin probability"
        assert survival_axis.get_legend() is not None
        assert len(survival_axis.lines) == 1
        assert set_axis.get_xlabel() == "premium units"
        assert set_axis.get_ylabel() == "aggregate claim index"
        assert len(set_axis.collections) == 1
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="inventory recursion"):
        plot_finite_time_discrete_survival(
            finite_time_ruin_discrete(
                [0.0, 1.0],
                initial_capital=0,
                premium_rate=1.0,
                claim_arrival_rate=1.0,
                horizon=1.0,
                method="seal",
                return_result=True,
            )
        )


def test_plot_finite_time_boundary_diagnostics():
    result = finite_time_ruin_discrete_boundary(
        [0.0, 1.0],
        inventory_times=[0.5, 1.0],
        boundary_values=[1.0, 1.5],
        claim_arrival_rate=1.0,
        return_result=True,
    )

    fig, axes = plt.subplots(1, 2)
    try:
        survival_axis = plot_finite_time_discrete_survival(result, ax=axes[0])
        boundary_axis = plot_finite_time_discrete_boundary(result, ax=axes[1], label="h")

        assert survival_axis.get_ylabel() == "non-ruin probability"
        assert boundary_axis.get_ylabel() == "boundary h(t)"
        assert boundary_axis.get_legend() is not None
        assert len(boundary_axis.lines) == 1
    finally:
        plt.close(fig)

    implicit = finite_time_ruin_discrete_inventory(
        [0.0, 1.0],
        inventory_times=[1.0],
        retained_counts=[1],
        claim_arrival_rate=1.0,
        return_result=True,
    )
    with pytest.raises(ValueError, match="explicit boundary"):
        plot_finite_time_discrete_boundary(implicit)

    nonhomogeneous = finite_time_ruin_discrete_nonhomogeneous_boundary(
        [[0.0, 0.4], [0.0, 0.6]],
        inventory_times=[0.5, 1.0],
        boundary_values=[1.0, 1.5],
        return_result=True,
    )
    fig, axes = plt.subplots(1, 2)
    try:
        assert plot_finite_time_discrete_survival(nonhomogeneous, ax=axes[0]) is axes[0]
        assert plot_finite_time_discrete_boundary(nonhomogeneous, ax=axes[1]) is axes[1]
    finally:
        plt.close(fig)


def test_plot_finite_time_appell_coefficients():
    result = finite_time_ruin_discrete_appell(
        [0.0, 1.0],
        boundary=lambda time: 1.0 + time,
        horizon=2.0,
        claim_arrival_rate=1.0,
        return_result=True,
    )

    fig, ax = plt.subplots()
    try:
        axis = plot_finite_time_appell_coefficients(result, ax=ax)

        assert axis.get_xlabel() == "degree"
        assert axis.get_ylabel() == "Appell coefficient"
        assert len(axis.lines) == 2
    finally:
        plt.close(fig)


def test_plot_discrete_time_castaner_diagnostics():
    result = finite_time_discrete_time_ruin(
        [[0.5, 0.5], [0.5, 0.5]],
        premiums=[0.0, 0.0],
        return_result=True,
    )
    bounds = finite_time_lundberg_bounds([2.0, 1.0], initial_capital=1.0)

    fig, axes = plt.subplots(1, 3)
    try:
        surplus_axis = plot_discrete_time_surplus_cdf(result, period=1, ax=axes[0])
        deficit_axis = plot_discrete_time_deficit_cdf(result, period=0, ax=axes[1])
        bound_axis = plot_finite_time_lundberg_bounds(bounds, ax=axes[2], label="bound")

        assert surplus_axis.get_ylabel() == "conditional CDF"
        assert deficit_axis.get_xlabel() == "deficit at ruin"
        assert bound_axis.get_ylabel() == "upper bound"
        assert bound_axis.get_legend() is not None
    finally:
        plt.close(fig)


def test_plot_gerber_shiu_diagnostics():
    result = gerber_shiu_from_paths(
        [_gerber_shiu_path(1.0, 1.0), _gerber_shiu_path(2.0, 2.0)],
    )

    fig, axes = plt.subplots(1, 3)
    try:
        deficit_axis = plot_deficit_at_ruin(result, ax=axes[0], bins=2)
        surplus_axis = plot_surplus_before_ruin(result, ax=axes[1], bins=2)
        scatter_axis = plot_gerber_shiu_scatter(result, ax=axes[2])

        assert deficit_axis.get_title() == "Deficit at ruin"
        assert surplus_axis.get_title() == "Surplus before ruin"
        assert scatter_axis.get_title() == "Gerber-Shiu ruin diagnostics"
        assert len(deficit_axis.patches) == 2
        assert len(surplus_axis.patches) >= 1
        assert len(scatter_axis.collections) == 1
    finally:
        plt.close(fig)


def test_plot_gerber_shiu_diagnostics_handle_no_ruin():
    safe = SimulationPath(
        times=np.array([0.0, 3.0]),
        reserves=np.array([1.0, 4.0]),
        claim_times=np.empty(0),
        claim_sizes=np.empty(0),
        ruin_time=None,
        horizon=3.0,
        initial_capital=1.0,
        premium_rate=1.0,
    )
    result = gerber_shiu_from_paths([safe])

    fig, axes = plt.subplots(1, 3)
    try:
        plot_deficit_at_ruin(result, ax=axes[0])
        plot_surplus_before_ruin(result, ax=axes[1])
        plot_gerber_shiu_scatter(result, ax=axes[2])

        assert [axis.texts[0].get_text() for axis in axes] == ["no ruin observed"] * 3
    finally:
        plt.close(fig)


def test_plot_prevention_calendar_handles_lagged_calendar():
    calendar = optimize_periodic_prevention_calendar(
        [1.0, 4.0, 2.0],
        annual_budget=0.3,
        max_prevention=0.8,
        effectiveness=2.0,
        lag_steps=1,
    )

    fig, ax = plt.subplots()
    try:
        result = plot_prevention_calendar(calendar, ax=ax, labels=["A", "B", "C"])

        assert result is ax
        assert ax.get_ylabel() == "prevention rate"
        assert ax.get_title() == "Periodic prevention calendar"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B", "C"]
        assert len(ax.patches) == 3
        assert len(ax.lines) == 1
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="labels"):
        plot_prevention_calendar(calendar, labels=["A"])


def test_plot_periodic_pressure_shows_controlled_pressure():
    calendar = optimize_periodic_prevention_calendar(
        [1.0, 4.0, 2.0],
        annual_budget=0.3,
        max_prevention=0.8,
        effectiveness=2.0,
    )

    fig, ax = plt.subplots()
    try:
        result = plot_periodic_pressure(calendar, ax=ax, labels=["A", "B", "C"])

        assert result is ax
        assert ax.get_ylabel() == "period pressure"
        assert ax.get_title() == "Periodic risk pressure"
        assert len(ax.patches) == 3
        assert len(ax.lines) == 1
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="labels"):
        plot_periodic_pressure(calendar, labels=["A"])


def test_plot_integer_byclaim_path_and_counts():
    model = INARByClaimModel(
        initial_capital=10.0,
        premium_per_period=2.0,
        primary_count_mean=2.0,
        initial_byclaim_mean=1.0,
        reproduction=0.4,
        primary_distribution=deterministic(1.0),
        byclaim_distribution=deterministic(1.0),
    )
    path = simulate_inar_byclaim_path(
        model,
        periods=4,
        seed=123,
        stop_at_ruin=False,
        ruin_threshold=1.5,
    )

    fig, axes = plt.subplots(1, 2)
    try:
        reserve_axis = plot_integer_byclaim_path(path, ax=axes[0])
        count_axis = plot_integer_byclaim_counts(path, ax=axes[1])

        assert reserve_axis.get_xlabel() == "period"
        assert reserve_axis.get_title() == "Discrete by-claim reserve path"
        np.testing.assert_allclose(reserve_axis.lines[1].get_ydata(), [1.5, 1.5])
        assert count_axis.get_ylabel() == "count"
        assert count_axis.get_title() == "By-claim counts"
        assert len(count_axis.patches) == 4
    finally:
        plt.close(fig)

    binar = BINARByClaimModel(
        initial_capital=20.0,
        premium_per_period=5.0,
        primary_count_means=(1.0, 2.0),
        initial_byclaim_means=(1.0, 1.0),
        reproduction_matrix=((0.2, 0.1), (0.1, 0.2)),
        primary_distributions=(deterministic(1.0), deterministic(1.0)),
        byclaim_distributions=(deterministic(1.0), deterministic(1.0)),
    )
    binar_path = simulate_binar_byclaim_path(binar, periods=3, seed=123, stop_at_ruin=False)
    fig, ax = plt.subplots()
    try:
        primary_axis = plot_integer_byclaim_counts(binar_path, ax=ax, kind="primary")
        assert primary_axis.get_title() == "Primary counts"
        assert primary_axis.get_legend() is not None
        assert len(primary_axis.patches) == 6
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="kind"):
        plot_integer_byclaim_counts(path, kind="secondary")

    fig, ax = plt.subplots()
    try:
        plot_ruin_time_histogram(_estimate(np.array([np.inf, np.inf])), ax=ax)

        assert len(ax.patches) == 0
        assert [text.get_text() for text in ax.texts] == ["no ruin observed"]
        assert list(ax.get_yticks()) == []
    finally:
        plt.close(fig)
