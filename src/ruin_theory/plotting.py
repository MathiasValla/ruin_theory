"""Matplotlib diagnostics for ruin models and simulations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .finite_discrete import (
    FiniteTimeDiscreteAppellResult,
    FiniteTimeDiscreteBoundaryResult,
    FiniteTimeDiscreteMethod,
    FiniteTimeDiscreteNonhomogeneousResult,
    FiniteTimeDiscreteRuinResult,
    finite_time_discrete_computation_set,
)
from .finite_discrete_time import (
    FiniteTimeDependentRuinResult,
    FiniteTimeDiscreteTimeRuinResult,
    FiniteTimeLundbergBoundResult,
    distribution_cdf,
)
from .dividends import BarrierDividendPath
from .integer_byclaims import IntegerByClaimPath
from .markov_modulated import DependenceImpactResult, MarkovModulatedRuinResult, solvency_region
from .prevention import PeriodicPreventionResult
from .red_time import AllocationGridResult, RedTimeCurveResult, ReserveAllocationResult
from .results import GerberShiuResult, RuinEstimate, SimulationPath


def _axis(ax: Axes | None) -> Axes:
    return plt.gca() if ax is None else ax


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _plot_reserve_path(
    axis: Axes,
    path: SimulationPath,
    *,
    alpha: float = 1.0,
    linewidth: float = 1.8,
) -> np.ndarray:
    times = _as_1d_float(path.times, "path.times")
    reserves = _as_1d_float(path.reserves, "path.reserves")
    if times.shape != reserves.shape:
        raise ValueError("path.times and path.reserves must have matching shapes")
    axis.step(times, reserves, where="post", color="#1f77b4", alpha=alpha, linewidth=linewidth)
    return times


def _path_xlim(paths: Iterable[SimulationPath]) -> tuple[float, float] | None:
    upper = 0.0
    for path in paths:
        times = np.asarray(path.times, dtype=float)
        candidates = [float(path.horizon)]
        if times.size:
            candidates.append(float(np.nanmax(times)))
        finite = [value for value in candidates if np.isfinite(value) and value > 0.0]
        if finite:
            upper = max(upper, max(finite))
    if upper <= 0.0:
        return None
    return (0.0, upper)


def plot_path(path: SimulationPath, *, ax: Axes | None = None, show_ruin: bool = True) -> Axes:
    """Plot a simulated reserve path."""

    axis = _axis(ax)
    _plot_reserve_path(axis, path)
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    if show_ruin and path.ruin_time is not None:
        axis.axvline(path.ruin_time, color="#b00020", linewidth=1.2, linestyle=":")
    xlim = _path_xlim((path,))
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.set_xlabel("time")
    axis.set_ylabel("reserve")
    axis.set_title("Reserve trajectory")
    return axis


def plot_paths(
    paths: Iterable[SimulationPath],
    *,
    ax: Axes | None = None,
    alpha: float = 0.25,
) -> Axes:
    """Overlay several reserve trajectories."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    path_list = tuple(paths)
    if not path_list:
        raise ValueError("paths must contain at least one SimulationPath")

    axis = _axis(ax)
    for path in path_list:
        _plot_reserve_path(axis, path, alpha=alpha, linewidth=1.2)
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    xlim = _path_xlim(path_list)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.set_xlabel("time")
    axis.set_ylabel("reserve")
    axis.set_title("Simulated reserve trajectories")
    return axis


def plot_ruin_curve(
    u: ArrayLike,
    probabilities: ArrayLike,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    ci_low: ArrayLike | None = None,
    ci_high: ArrayLike | None = None,
    band_alpha: float = 0.18,
) -> Axes:
    """Plot ruin probability against initial surplus.

    Optional lower and upper confidence limits can be passed through ``ci_low``
    and ``ci_high`` to add a shaded uncertainty band.
    """

    surplus = _as_1d_float(u, "u")
    ruin_probabilities = _as_1d_float(probabilities, "probabilities")
    if surplus.shape != ruin_probabilities.shape:
        raise ValueError("u and probabilities must have matching shapes")
    if np.any((ruin_probabilities < 0.0) | (ruin_probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not 0.0 <= band_alpha <= 1.0:
        raise ValueError("band_alpha must lie in [0, 1]")

    axis = _axis(ax)
    if (ci_low is None) != (ci_high is None):
        raise ValueError("ci_low and ci_high must be provided together")
    if ci_low is not None and ci_high is not None:
        lower = _as_1d_float(ci_low, "ci_low")
        upper = _as_1d_float(ci_high, "ci_high")
        if lower.shape != surplus.shape or upper.shape != surplus.shape:
            raise ValueError("confidence limits must match u shape")
        if np.any((lower < 0.0) | (lower > 1.0) | (upper < 0.0) | (upper > 1.0)):
            raise ValueError("confidence limits must lie in [0, 1]")
        if np.any(lower > upper):
            raise ValueError("ci_low must be less than or equal to ci_high")
        axis.fill_between(surplus, lower, upper, color="#0b6e4f", alpha=band_alpha, linewidth=0)

    axis.plot(surplus, ruin_probabilities, color="#0b6e4f", linewidth=2.0, label=label)
    axis.set_xlabel("initial surplus")
    axis.set_ylabel("ruin probability")
    axis.set_ylim(0.0, 1.0)
    if label:
        axis.legend()
    return axis


def plot_terminal_reserve_distribution(
    terminal_reserves: ArrayLike,
    *,
    ax: Axes | None = None,
    bins: int = 30,
    quantiles: Iterable[float] | None = (0.05, 0.5, 0.95),
    show_zero: bool = True,
) -> Axes:
    """Plot the empirical distribution of terminal reserves."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    reserves = _as_1d_float(terminal_reserves, "terminal_reserves")
    axis = _axis(ax)
    axis.hist(reserves, bins=bins, color="#4c78a8", alpha=0.85)
    if show_zero:
        axis.axvline(0.0, color="#8c1d2d", linewidth=1.2, linestyle="--", label="zero reserve")

    if quantiles is not None:
        levels = _as_1d_float(tuple(quantiles), "quantiles")
        if np.any((levels < 0.0) | (levels > 1.0)):
            raise ValueError("quantiles must lie in [0, 1]")
        values = np.quantile(reserves, levels)
        for level, value in zip(levels, values, strict=True):
            if np.isclose(level, 0.5):
                label = "median"
                color = "#0b6e4f"
                linestyle = "-"
            else:
                label = f"{level:g} quantile"
                color = "#2f3640"
                linestyle = ":"
            axis.axvline(value, color=color, linewidth=1.2, linestyle=linestyle, label=label)

    axis.set_xlabel("terminal reserve")
    axis.set_ylabel("frequency")
    axis.set_title("Terminal reserve distribution")
    if show_zero or quantiles is not None:
        axis.legend()
    return axis


def plot_ruin_time_histogram(
    estimate: RuinEstimate,
    *,
    ax: Axes | None = None,
    bins: int = 30,
) -> Axes:
    """Plot conditional ruin-time histogram from a Monte Carlo estimate."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    axis = _axis(ax)
    ruin_times = np.asarray(estimate.ruin_times, dtype=float)
    finite = ruin_times[np.isfinite(ruin_times)]
    if np.any(finite < 0.0):
        raise ValueError("ruin times must be non-negative")
    if finite.size:
        axis.hist(finite, bins=bins, color="#4c78a8", alpha=0.85)
    else:
        axis.text(
            0.5,
            0.5,
            "no ruin observed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    if estimate.horizon is not None and np.isfinite(estimate.horizon) and estimate.horizon > 0:
        axis.set_xlim(0.0, float(estimate.horizon))
    axis.set_xlabel("time to ruin")
    axis.set_ylabel("frequency")
    axis.set_title("Conditional time to ruin")
    return axis


def plot_finite_time_discrete_survival(
    result: FiniteTimeDiscreteRuinResult
    | FiniteTimeDiscreteBoundaryResult
    | FiniteTimeDiscreteNonhomogeneousResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot exact survival probabilities at finite-time inventory dates."""

    if not isinstance(
        result,
        (
            FiniteTimeDiscreteRuinResult,
            FiniteTimeDiscreteBoundaryResult,
            FiniteTimeDiscreteNonhomogeneousResult,
        ),
    ):
        raise TypeError("result must be a finite-time discrete result")
    if result.inventory_times.size == 0 or result.survival_probabilities.size == 0:
        raise ValueError("result does not contain inventory recursion diagnostics")
    if result.inventory_times.shape != result.survival_probabilities.shape:
        raise ValueError("inventory_times and survival_probabilities must match")

    axis = _axis(ax)
    axis.step(
        result.inventory_times,
        result.survival_probabilities,
        where="post",
        color="#0b6e4f",
        linewidth=2.0,
        label=label,
    )
    axis.scatter(result.inventory_times, result.survival_probabilities, color="#0b6e4f", s=28)
    axis.set_xlabel("time")
    axis.set_ylabel("non-ruin probability")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Finite-time lattice survival")
    if label:
        axis.legend()
    return axis


def plot_finite_time_discrete_boundary(
    result: FiniteTimeDiscreteBoundaryResult | FiniteTimeDiscreteNonhomogeneousResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot a deterministic finite-time lattice boundary."""

    if not isinstance(
        result,
        (FiniteTimeDiscreteBoundaryResult, FiniteTimeDiscreteNonhomogeneousResult),
    ):
        raise TypeError("result must be a finite-time boundary result")
    if result.inventory_times.shape != result.boundary_values.shape:
        raise ValueError("inventory_times and boundary_values must match")
    if np.any(~np.isfinite(result.boundary_values)):
        raise ValueError("result does not contain explicit boundary values")

    axis = _axis(ax)
    axis.step(
        result.inventory_times,
        result.boundary_values,
        where="post",
        color="#8c1d2d",
        linewidth=2.0,
        label=label,
    )
    axis.scatter(result.inventory_times, result.boundary_values, color="#8c1d2d", s=28)
    axis.set_xlabel("time")
    axis.set_ylabel("boundary h(t)")
    axis.set_title("Finite-time lattice boundary")
    if label:
        axis.legend()
    return axis


def plot_finite_time_appell_coefficients(
    result: FiniteTimeDiscreteAppellResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot Picard-Lefevre generalized-Appell coefficients."""

    if not isinstance(result, FiniteTimeDiscreteAppellResult):
        raise TypeError("result must be a FiniteTimeDiscreteAppellResult")
    coefficients = np.asarray(result.appell_coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("result does not contain Appell coefficients")

    axis = _axis(ax)
    degrees = np.arange(coefficients.size)
    axis.axhline(0.0, color="#222222", linewidth=0.8, linestyle=":")
    axis.plot(degrees, coefficients, marker="o", color="#4c78a8", linewidth=1.8)
    axis.set_xlabel("degree")
    axis.set_ylabel("Appell coefficient")
    axis.set_title("Generalized-Appell coefficients")
    return axis


def plot_finite_time_discrete_computation_set(
    *,
    initial_capital: float,
    premium_units: float,
    method: FiniteTimeDiscreteMethod = "seal",
    ax: Axes | None = None,
) -> Axes:
    """Plot Picard-Lefevre or Seal/Takacs computation-set lattice points."""

    points = finite_time_discrete_computation_set(
        initial_capital=initial_capital,
        premium_units=premium_units,
        method=method,
    )
    axis = _axis(ax)
    if points.size:
        axis.scatter(points[:, 0], points[:, 1], s=18, color="#4c78a8", alpha=0.82)
    axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle=":")
    axis.set_xlabel("premium units")
    axis.set_ylabel("aggregate claim index")
    axis.set_title(f"{method.replace('-', ' ').title()} computation set")
    return axis


def _discrete_time_result(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
) -> FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult:
    if not isinstance(result, (FiniteTimeDiscreteTimeRuinResult, FiniteTimeDependentRuinResult)):
        raise TypeError("result must be a finite-time discrete-time result")
    return result


def plot_discrete_time_surplus_cdf(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
    *,
    period: int,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot conditional surplus CDF given non-ruin at a period."""

    checked = _discrete_time_result(result)
    values, probabilities = checked.surplus_distributions[period]
    if values.size == 0:
        raise ValueError("surplus distribution is empty for this period")
    points = np.sort(values)
    cdf = distribution_cdf((values, probabilities), points)
    axis = _axis(ax)
    axis.step(points, cdf, where="post", color="#0b6e4f", linewidth=2.0, label=label)
    axis.set_xlabel("surplus")
    axis.set_ylabel("conditional CDF")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Surplus given non-ruin")
    if label:
        axis.legend()
    return axis


def plot_discrete_time_deficit_cdf(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
    *,
    period: int,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot conditional deficit-at-ruin CDF for a ruin period."""

    checked = _discrete_time_result(result)
    values, probabilities = checked.deficit_distributions[period]
    if values.size == 0:
        raise ValueError("deficit distribution is empty for this period")
    points = np.sort(values)
    cdf = distribution_cdf((values, probabilities), points)
    axis = _axis(ax)
    axis.step(points, cdf, where="post", color="#8c1d2d", linewidth=2.0, label=label)
    axis.set_xlabel("deficit at ruin")
    axis.set_ylabel("conditional CDF")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Deficit conditional on ruin")
    if label:
        axis.legend()
    return axis


def plot_finite_time_lundberg_bounds(
    result: FiniteTimeLundbergBoundResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot finite-time non-homogeneous Lundberg bounds by horizon."""

    if not isinstance(result, FiniteTimeLundbergBoundResult):
        raise TypeError("result must be a FiniteTimeLundbergBoundResult")
    horizons = np.arange(1, result.bounds.size + 1)
    axis = _axis(ax)
    axis.step(horizons, result.bounds, where="post", color="#4c78a8", linewidth=2.0, label=label)
    axis.scatter(horizons, result.bounds, color="#4c78a8", s=28)
    axis.set_xlabel("horizon")
    axis.set_ylabel("upper bound")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Finite-time Lundberg bound")
    if label:
        axis.legend()
    return axis


def _ruined_values(result: GerberShiuResult, values: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(result, GerberShiuResult):
        raise TypeError("result must be a GerberShiuResult")
    selected = np.asarray(values, dtype=float)[result.ruined]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return np.empty(0, dtype=float)
    if np.any(selected < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return selected


def plot_deficit_at_ruin(
    result: GerberShiuResult,
    *,
    ax: Axes | None = None,
    bins: int = 30,
) -> Axes:
    """Plot the conditional distribution of the deficit at ruin."""

    axis = _axis(ax)
    deficits = _ruined_values(result, result.deficits_at_ruin, "deficits_at_ruin")
    if deficits.size == 0:
        axis.text(
            0.5,
            0.5,
            "no ruin observed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    else:
        axis.hist(deficits, bins=bins, color="#b84a39", alpha=0.78, edgecolor="white")
        axis.axvline(float(np.mean(deficits)), color="#222222", linewidth=1.2, linestyle="--")
    axis.set_xlabel("deficit at ruin")
    axis.set_ylabel("count")
    axis.set_title("Deficit at ruin")
    return axis


def plot_surplus_before_ruin(
    result: GerberShiuResult,
    *,
    ax: Axes | None = None,
    bins: int = 30,
) -> Axes:
    """Plot the conditional distribution of the surplus immediately before ruin."""

    axis = _axis(ax)
    surplus = _ruined_values(result, result.surplus_before_ruin, "surplus_before_ruin")
    if surplus.size == 0:
        axis.text(
            0.5,
            0.5,
            "no ruin observed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    else:
        axis.hist(surplus, bins=bins, color="#2f6f9f", alpha=0.78, edgecolor="white")
        axis.axvline(float(np.mean(surplus)), color="#222222", linewidth=1.2, linestyle="--")
    axis.set_xlabel("surplus before ruin")
    axis.set_ylabel("count")
    axis.set_title("Surplus before ruin")
    return axis


def plot_gerber_shiu_scatter(
    result: GerberShiuResult,
    *,
    ax: Axes | None = None,
    alpha: float = 0.7,
) -> Axes:
    """Plot surplus-before-ruin against deficit-at-ruin."""

    if not isinstance(result, GerberShiuResult):
        raise TypeError("result must be a GerberShiuResult")
    axis = _axis(ax)
    mask = (
        result.ruined
        & np.isfinite(result.surplus_before_ruin)
        & np.isfinite(result.deficits_at_ruin)
    )
    if not np.any(mask):
        axis.text(
            0.5,
            0.5,
            "no ruin observed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    else:
        colors = np.asarray(result.ruin_times, dtype=float)[mask]
        scatter = axis.scatter(
            result.surplus_before_ruin[mask],
            result.deficits_at_ruin[mask],
            c=colors,
            cmap="viridis",
            alpha=alpha,
            edgecolors="none",
        )
        axis.figure.colorbar(scatter, ax=axis, label="ruin time")
    axis.set_xlabel("surplus before ruin")
    axis.set_ylabel("deficit at ruin")
    axis.set_title("Gerber-Shiu ruin diagnostics")
    return axis


def plot_prevention_calendar(
    calendar: PeriodicPreventionResult,
    *,
    ax: Axes | None = None,
    labels: Iterable[str] | None = None,
    show_effective: bool = True,
) -> Axes:
    """Plot a periodic prevention spending calendar."""

    if not isinstance(calendar, PeriodicPreventionResult):
        raise TypeError("calendar must be a PeriodicPreventionResult")

    n_periods = calendar.amounts.size
    x = np.arange(n_periods)
    if labels is None:
        tick_labels = [str(i + 1) for i in x]
    else:
        tick_labels = list(labels)
        if len(tick_labels) != n_periods:
            raise ValueError("labels must match the number of calendar periods")

    axis = _axis(ax)
    axis.bar(x, calendar.amounts, color="#4c78a8", alpha=0.88, label="spending")
    if show_effective and calendar.lag_steps:
        axis.plot(
            x,
            calendar.effective_amounts,
            color="#b00020",
            linewidth=1.8,
            marker="o",
            label="effective",
        )
        axis.legend()

    axis.set_xticks(x)
    axis.set_xticklabels(tick_labels)
    axis.set_ylabel("prevention rate")
    axis.set_title("Periodic prevention calendar")
    return axis


def plot_periodic_pressure(
    calendar: PeriodicPreventionResult,
    *,
    ax: Axes | None = None,
    labels: Iterable[str] | None = None,
    show_controlled: bool = True,
) -> Axes:
    """Plot baseline and controlled periodic pressure weights."""

    if not isinstance(calendar, PeriodicPreventionResult):
        raise TypeError("calendar must be a PeriodicPreventionResult")

    n_periods = calendar.weights.size
    x = np.arange(n_periods)
    if labels is None:
        tick_labels = [str(i + 1) for i in x]
    else:
        tick_labels = list(labels)
        if len(tick_labels) != n_periods:
            raise ValueError("labels must match the number of calendar periods")

    axis = _axis(ax)
    axis.bar(x, calendar.weights, color="#4c78a8", alpha=0.35, label="baseline")
    if show_controlled:
        controlled = calendar.weights * calendar.frequency_multipliers
        axis.plot(
            x,
            controlled,
            color="#0b6e4f",
            linewidth=2.0,
            marker="o",
            label="controlled",
        )
        axis.legend()

    axis.set_xticks(x)
    axis.set_xticklabels(tick_labels)
    axis.set_ylabel("period pressure")
    axis.set_title("Periodic risk pressure")
    return axis


def plot_win_first_surface(
    initial_capital: ArrayLike,
    gain: ArrayLike,
    probabilities: ArrayLike,
    *,
    ax: Axes | None = None,
    colorbar: bool = True,
) -> Axes:
    """Plot a win-first probability surface over initial surplus and target gain."""

    surplus = _as_1d_float(initial_capital, "initial_capital")
    target_gain = _as_1d_float(gain, "gain")
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (surplus.size, target_gain.size):
        raise ValueError("probabilities must have shape (len(initial_capital), len(gain))")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must contain values in [0, 1]")

    axis = _axis(ax)
    mesh = axis.pcolormesh(target_gain, surplus, values, shading="auto", cmap="viridis")
    if colorbar:
        axis.figure.colorbar(mesh, ax=axis, label="win-first probability")
    axis.set_xlabel("target gain")
    axis.set_ylabel("initial surplus")
    axis.set_title("Win-first probability")
    return axis


def plot_maximum_before_default_hazard(
    x: ArrayLike,
    hazard: ArrayLike,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot the hazard rate of the maximum-before-default distribution."""

    levels = _as_1d_float(x, "x")
    values = _as_1d_float(hazard, "hazard")
    if values.shape != levels.shape:
        raise ValueError("hazard must match x shape")
    if np.any(values < 0.0):
        raise ValueError("hazard must be non-negative")

    axis = _axis(ax)
    axis.plot(levels, values, color="#b00020", linewidth=2.0, label=label)
    axis.set_xlabel("surplus level")
    axis.set_ylabel("hazard rate")
    axis.set_title("Maximum-before-default hazard")
    if label:
        axis.legend()
    return axis


def plot_win_first_sensitivity(
    parameter_values: ArrayLike,
    probabilities: ArrayLike,
    *,
    parameter_name: str = "parameter",
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot win-first sensitivity to a scalar model parameter."""

    parameters = _as_1d_float(parameter_values, "parameter_values")
    values = _as_1d_float(probabilities, "probabilities")
    if values.shape != parameters.shape:
        raise ValueError("probabilities must match parameter_values shape")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    axis = _axis(ax)
    axis.plot(parameters, values, color="#0b6e4f", linewidth=2.0, marker="o", label=label)
    axis.set_xlabel(parameter_name)
    axis.set_ylabel("win-first probability")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Win-first sensitivity")
    if label:
        axis.legend()
    return axis


def plot_barrier_dividend_path(
    path: BarrierDividendPath,
    *,
    ax: Axes | None = None,
    show_dividends: bool = True,
) -> Axes:
    """Plot a reserve path controlled by a horizontal dividend barrier."""

    if not isinstance(path, BarrierDividendPath):
        raise TypeError("path must be a BarrierDividendPath")
    axis = _axis(ax)
    axis.step(path.times, path.reserves, where="post", color="#1f77b4", linewidth=1.8)
    axis.axhline(path.barrier, color="#0b6e4f", linewidth=1.2, linestyle="--", label="barrier")
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle=":")
    if path.ruin_time is not None:
        axis.axvline(path.ruin_time, color="#b00020", linewidth=1.2, linestyle=":")
    if show_dividends:
        twin = axis.twinx()
        twin.step(
            path.dividend_times,
            path.cumulative_dividends,
            where="post",
            color="#9467bd",
            alpha=0.8,
            linewidth=1.4,
        )
        twin.set_ylabel("cumulative dividends")
    axis.set_xlabel("time")
    axis.set_ylabel("reserve")
    axis.set_title("Dividend-barrier reserve path")
    axis.legend(loc="best")
    return axis


def plot_barrier_dividend_distribution(
    dividends: ArrayLike,
    *,
    ax: Axes | None = None,
    bins: int = 30,
    density: bool = False,
) -> Axes:
    """Plot the empirical distribution of cumulative barrier dividends."""

    values = _as_1d_float(dividends, "dividends")
    if np.any(values < 0.0):
        raise ValueError("dividends must be non-negative")
    if bins <= 0:
        raise ValueError("bins must be positive")
    axis = _axis(ax)
    axis.hist(values, bins=bins, density=density, color="#4c78a8", alpha=0.82)
    axis.set_xlabel("cumulative dividends")
    axis.set_ylabel("density" if density else "frequency")
    axis.set_title("Barrier dividend distribution")
    return axis


def plot_barrier_ruin_time_distribution(
    ruin_times: ArrayLike,
    *,
    ax: Axes | None = None,
    bins: int = 30,
) -> Axes:
    """Plot finite ruin times from dividend-barrier simulations."""

    times = np.asarray(ruin_times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("ruin_times must be a non-empty one-dimensional array")
    if np.any(np.isnan(times)) or np.any(times < 0.0):
        raise ValueError("ruin_times must contain non-negative values or infinity")
    finite = times[np.isfinite(times)]
    if bins <= 0:
        raise ValueError("bins must be positive")
    axis = _axis(ax)
    if finite.size:
        axis.hist(finite, bins=bins, color="#b00020", alpha=0.72)
    else:
        axis.text(0.5, 0.5, "no ruin observed", ha="center", va="center", transform=axis.transAxes)
        axis.set_yticks([])
    axis.set_xlabel("time to ruin")
    axis.set_ylabel("frequency")
    axis.set_title("Dividend-barrier ruin times")
    return axis


def plot_barrier_comparison(
    barriers: ArrayLike,
    expected_dividends: ArrayLike,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Compare expected cumulative dividends across barrier levels."""

    levels = _as_1d_float(barriers, "barriers")
    values = _as_1d_float(expected_dividends, "expected_dividends")
    if levels.shape != values.shape:
        raise ValueError("expected_dividends must match barriers shape")
    if np.any(levels <= 0.0):
        raise ValueError("barriers must be positive")
    if np.any(values < 0.0):
        raise ValueError("expected_dividends must be non-negative")
    axis = _axis(ax)
    axis.plot(levels, values, color="#0b6e4f", linewidth=2.0, marker="o", label=label)
    axis.set_xlabel("barrier")
    axis.set_ylabel("expected dividends")
    axis.set_title("Dividend-barrier comparison")
    if label:
        axis.legend()
    return axis


def plot_markov_modulated_ruin_curves(
    results: MarkovModulatedRuinResult | Iterable[MarkovModulatedRuinResult],
    *,
    ax: Axes | None = None,
    labels: Iterable[str] | None = None,
) -> Axes:
    """Plot finite-time ruin curves for Markov-modulated multirisk results."""

    result_list = (results,) if isinstance(results, MarkovModulatedRuinResult) else tuple(results)
    if not result_list:
        raise ValueError("results must contain at least one MarkovModulatedRuinResult")
    if not all(isinstance(result, MarkovModulatedRuinResult) for result in result_list):
        raise TypeError("results must contain MarkovModulatedRuinResult instances")
    if labels is None:
        label_list = [result.region for result in result_list]
    else:
        label_list = list(labels)
        if len(label_list) != len(result_list):
            raise ValueError("labels must match results")

    axis = _axis(ax)
    for result, label in zip(result_list, label_list):
        periods = np.arange(result.ruin_probabilities.size)
        axis.plot(periods, result.ruin_probabilities, linewidth=2.0, marker="o", label=label)
    axis.set_xlabel("period")
    axis.set_ylabel("ruin probability")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Markov-modulated multirisk ruin")
    if len(result_list) > 1 or labels is not None:
        axis.legend()
    return axis


def plot_environment_state_survival(
    result: MarkovModulatedRuinResult,
    *,
    ax: Axes | None = None,
    normalize: bool = False,
) -> Axes:
    """Plot surviving probability mass by Markov environment state."""

    if not isinstance(result, MarkovModulatedRuinResult):
        raise TypeError("result must be a MarkovModulatedRuinResult")
    values = np.asarray(result.survival_by_state, dtype=float)
    if values.ndim != 2:
        raise ValueError("result.survival_by_state must be two-dimensional")
    if normalize:
        totals = np.sum(values, axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0.0)

    periods = np.arange(values.shape[0])
    axis = _axis(ax)
    for state in range(values.shape[1]):
        axis.plot(periods, values[:, state], linewidth=1.8, marker="o", label=f"state {state + 1}")
    axis.set_xlabel("period")
    axis.set_ylabel("conditional mass" if normalize else "surviving mass")
    axis.set_title("Survival by environment state")
    axis.legend()
    return axis


def plot_dependence_impact(
    impact: DependenceImpactResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot the ruin-probability difference between dependence scenarios."""

    if not isinstance(impact, DependenceImpactResult):
        raise TypeError("impact must be a DependenceImpactResult")
    axis = _axis(ax)
    colors = np.where(impact.difference >= 0.0, "#b00020", "#0b6e4f")
    axis.bar(impact.periods, impact.difference, color=colors, alpha=0.84)
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.set_xlabel("period")
    axis.set_ylabel(f"{impact.comparison_label} - {impact.reference_label}")
    axis.set_title("Dependence impact")
    return axis


def plot_solvency_region_2d(
    initial_capitals: ArrayLike,
    premiums: ArrayLike,
    *,
    period: int,
    region: str = "any_line",
    severity_limit: float | ArrayLike = 0.0,
    ax: Axes | None = None,
    grid_size: int = 160,
) -> Axes:
    """Plot a two-line solvency region in aggregate-claim coordinates."""

    initial = _as_1d_float(initial_capitals, "initial_capitals")
    premium = _as_1d_float(premiums, "premiums")
    if initial.size != 2 or premium.size != 2:
        raise ValueError("initial_capitals and premiums must have length two")
    if np.any(initial < 0.0) or np.any(premium < 0.0):
        raise ValueError("initial_capitals and premiums must be non-negative")
    inventory = int(period)
    if inventory != period or inventory <= 0:
        raise ValueError("period must be a positive integer")
    size = int(grid_size)
    if size <= 1:
        raise ValueError("grid_size must be greater than one")

    boundary = initial + inventory * premium
    upper = max(float(np.max(boundary) * 1.6), 1.0)
    x = np.linspace(0.0, upper, size)
    y = np.linspace(0.0, upper, size)
    xx, yy = np.meshgrid(x, y)
    if region == "any_line":
        mask = (xx <= boundary[0]) & (yy <= boundary[1])
    elif region == "total":
        mask = xx + yy <= np.sum(boundary)
    elif region == "hybrid":
        limit = np.asarray(severity_limit, dtype=float)
        if limit.ndim == 0:
            limit = np.full(2, float(limit))
        if limit.shape != boundary.shape:
            raise ValueError("severity_limit must be scalar or have length two")
        mask = (
            (xx + yy <= np.sum(boundary))
            & (xx <= boundary[0] + limit[0])
            & (yy <= boundary[1] + limit[1])
        )
    else:
        solvency_region(region, severity_limit=severity_limit)
        raise ValueError("region must be 'any_line', 'total' or 'hybrid'")

    axis = _axis(ax)
    axis.pcolormesh(x, y, mask.astype(float), shading="auto", cmap="Greens", alpha=0.45)
    axis.axvline(boundary[0], color="#4c78a8", linewidth=1.4, linestyle="--")
    axis.axhline(boundary[1], color="#4c78a8", linewidth=1.4, linestyle="--")
    axis.plot([0.0, float(np.sum(boundary))], [float(np.sum(boundary)), 0.0], color="#b00020")
    axis.set_xlim(0.0, upper)
    axis.set_ylim(0.0, upper)
    axis.set_xlabel("aggregate claims line 1")
    axis.set_ylabel("aggregate claims line 2")
    axis.set_title(f"Solvency region: {region}")
    return axis


def plot_red_time_curve(
    curve: RedTimeCurveResult,
    *,
    ax: Axes | None = None,
    show_negative_area: bool = True,
) -> Axes:
    """Plot expected time in red and optionally integrated negative area."""

    if not isinstance(curve, RedTimeCurveResult):
        raise TypeError("curve must be a RedTimeCurveResult")
    capital = _as_1d_float(curve.initial_capitals, "initial_capitals")
    red_time = _as_1d_float(curve.expected_time_in_red, "expected_time_in_red")
    if red_time.shape != capital.shape:
        raise ValueError("expected_time_in_red must match initial_capitals")

    axis = _axis(ax)
    axis.plot(capital, red_time, color="#b00020", linewidth=2.0, marker="o", label="E[tau]")
    axis.set_xlabel("initial reserve")
    axis.set_ylabel("expected time in red")
    axis.set_title("Time in red")
    if show_negative_area:
        negative_area = _as_1d_float(curve.expected_negative_area, "expected_negative_area")
        if negative_area.shape != capital.shape:
            raise ValueError("expected_negative_area must match initial_capitals")
        twin = axis.twinx()
        twin.plot(
            capital,
            negative_area,
            color="#4c78a8",
            linewidth=1.8,
            marker="s",
            label="E[I]",
        )
        twin.set_ylabel("expected negative area")
    return axis


def plot_red_time_allocation(
    result: ReserveAllocationResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot reserve allocation and red-time equalization diagnostics."""

    if not isinstance(result, ReserveAllocationResult):
        raise TypeError("result must be a ReserveAllocationResult")
    allocations = _as_1d_float(result.allocations, "allocations")
    red_times = _as_1d_float(result.red_times, "red_times")
    if allocations.shape != red_times.shape:
        raise ValueError("allocations and red_times must have matching shapes")

    x = np.arange(allocations.size)
    axis = _axis(ax)
    colors = np.where(result.active, "#4c78a8", "#b8b8b8")
    axis.bar(x, allocations, color=colors, alpha=0.86)
    axis.set_xticks(x)
    axis.set_xticklabels([f"line {index + 1}" for index in x])
    axis.set_ylabel("allocated reserve")
    axis.set_title("Optimal reserve allocation")
    twin = axis.twinx()
    twin.plot(x, red_times, color="#b00020", linewidth=1.8, marker="o")
    twin.axhline(result.threshold, color="#222222", linewidth=1.0, linestyle=":")
    twin.set_ylabel("expected time in red")
    return axis


def plot_two_line_allocation_curve(
    grid: AllocationGridResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot objective values along a two-line reserve allocation."""

    if not isinstance(grid, AllocationGridResult):
        raise TypeError("grid must be an AllocationGridResult")
    allocations = np.asarray(grid.allocations, dtype=float)
    values = _as_1d_float(grid.objective_values, "objective_values")
    if allocations.ndim != 2 or allocations.shape[1] != 2 or allocations.shape[0] != values.size:
        raise ValueError("grid.allocations must have shape (n, 2)")

    order = np.argsort(allocations[:, 0])
    axis = _axis(ax)
    axis.plot(
        allocations[order, 0],
        values[order],
        color="#0b6e4f",
        linewidth=2.0,
        marker="o",
        label=label,
    )
    axis.set_xlabel("line 1 reserve")
    axis.set_ylabel("sum expected negative areas")
    axis.set_title("Two-line allocation objective")
    if label:
        axis.legend()
    return axis


def plot_simplex_allocation_surface(
    grid: AllocationGridResult,
    *,
    ax: Axes | None = None,
    colorbar: bool = True,
) -> Axes:
    """Plot a three-line allocation objective on simplex coordinates."""

    if not isinstance(grid, AllocationGridResult):
        raise TypeError("grid must be an AllocationGridResult")
    allocations = np.asarray(grid.allocations, dtype=float)
    values = _as_1d_float(grid.objective_values, "objective_values")
    if allocations.ndim != 2 or allocations.shape[1] != 3 or allocations.shape[0] != values.size:
        raise ValueError("grid.allocations must have shape (n, 3)")

    x = allocations[:, 1] + 0.5 * allocations[:, 2]
    y = math.sqrt(3.0) * allocations[:, 2] / 2.0
    axis = _axis(ax)
    scatter = axis.scatter(x, y, c=values, cmap="viridis", s=34, edgecolors="none")
    if colorbar:
        axis.figure.colorbar(scatter, ax=axis, label="sum expected negative areas")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Allocation simplex objective")
    return axis


def plot_integer_byclaim_path(
    path: IntegerByClaimPath,
    *,
    ax: Axes | None = None,
    show_ruin: bool = True,
) -> Axes:
    """Plot a discrete INAR/BINAR by-claim reserve trajectory."""

    if not isinstance(path, IntegerByClaimPath):
        raise TypeError("path must be an IntegerByClaimPath")
    axis = _axis(ax)
    periods = np.arange(path.reserves.size)
    axis.step(periods, path.reserves, where="post", color="#1f77b4", linewidth=1.8)
    axis.axhline(path.ruin_threshold, color="#222222", linewidth=1.0, linestyle="--")
    if show_ruin and path.ruin_time is not None:
        axis.axvline(path.ruin_time, color="#b00020", linewidth=1.2, linestyle=":")
    axis.set_xlabel("period")
    axis.set_ylabel("reserve")
    axis.set_title("Discrete by-claim reserve path")
    return axis


def plot_integer_byclaim_counts(
    path: IntegerByClaimPath,
    *,
    ax: Axes | None = None,
    kind: str = "byclaim",
) -> Axes:
    """Plot primary or by-claim counts by period."""

    if not isinstance(path, IntegerByClaimPath):
        raise TypeError("path must be an IntegerByClaimPath")
    if kind not in {"primary", "byclaim"}:
        raise ValueError("kind must be 'primary' or 'byclaim'")
    counts = path.primary_counts if kind == "primary" else path.byclaim_counts
    axis = _axis(ax)
    periods = np.arange(1, counts.shape[0] + 1)
    bottom = np.zeros(counts.shape[0], dtype=float)
    for index in range(counts.shape[1]):
        label = f"type {index + 1}" if counts.shape[1] > 1 else kind
        axis.bar(periods, counts[:, index], bottom=bottom, label=label)
        bottom += counts[:, index]
    axis.set_xlabel("period")
    axis.set_ylabel("count")
    axis.set_title("Primary counts" if kind == "primary" else "By-claim counts")
    if counts.shape[1] > 1:
        axis.legend()
    return axis
