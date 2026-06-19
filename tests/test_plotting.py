"""Plotting diagnostics tests."""

# ruff: noqa: E402

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from ruin_theory.plotting import (
    plot_integer_byclaim_counts,
    plot_integer_byclaim_path,
    plot_deficit_at_ruin,
    plot_finite_time_discrete_computation_set,
    plot_finite_time_discrete_survival,
    plot_gerber_shiu_scatter,
    plot_path,
    plot_paths,
    plot_periodic_pressure,
    plot_prevention_calendar,
    plot_ruin_curve,
    plot_ruin_time_histogram,
    plot_surplus_before_ruin,
    plot_terminal_reserve_distribution,
)
from ruin_theory import (
    BINARByClaimModel,
    INARByClaimModel,
    deterministic,
    finite_time_ruin_discrete,
    gerber_shiu_from_paths,
    simulate_binar_byclaim_path,
    simulate_inar_byclaim_path,
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
