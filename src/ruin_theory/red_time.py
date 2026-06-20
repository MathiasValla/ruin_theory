"""Time-in-red risk measures and reserve allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .models import CramerLundbergProcess, RiskProcess, SparreAndersenProcess
from .results import SimulationPath
from .simulation import simulate_path


RiskFunctional = Callable[[float], float]


@dataclass(frozen=True)
class RedTimePathMetrics:
    """Exact time-in-red and negative-area metrics for one reserve path."""

    time_in_red: float
    negative_area: float
    minimum_reserve: float
    horizon: float


@dataclass(frozen=True)
class RedTimeEstimate:
    """Monte Carlo estimate of expected time in red and integrated negative part."""

    time_in_red: np.ndarray
    negative_area: np.ndarray
    expected_time_in_red: float
    expected_negative_area: float
    time_in_red_standard_error: float
    negative_area_standard_error: float
    n_simulations: int
    horizon: float


@dataclass(frozen=True)
class RedTimeCurveResult:
    """Expected red-time and negative-area curves over initial reserve levels."""

    initial_capitals: np.ndarray
    expected_time_in_red: np.ndarray
    expected_negative_area: np.ndarray
    time_in_red_standard_error: np.ndarray
    negative_area_standard_error: np.ndarray
    n_simulations: int
    horizon: float


@dataclass(frozen=True)
class MultilineRedTimeMetrics:
    """Pathwise multirisk red-time measures for synchronized business lines."""

    time_in_red: np.ndarray
    negative_area: np.ndarray
    red_time_with_positive_total: np.ndarray
    aggregate_negative_area: float
    positive_total_red_time_sum: float
    horizon: float


@dataclass(frozen=True)
class ReserveAllocationResult:
    """Optimal allocation summary from Loisel's red-time equalization criterion."""

    total_reserve: float
    allocations: np.ndarray
    red_times: np.ndarray
    negative_areas: np.ndarray | None
    objective_value: float | None
    active: np.ndarray
    threshold: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class AllocationGridResult:
    """Objective and diagnostic values evaluated on a reserve-allocation grid."""

    allocations: np.ndarray
    objective_values: np.ndarray
    negative_areas: np.ndarray
    red_times: np.ndarray | None
    total_reserve: float


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _as_1d_nonnegative(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _maybe_scalar(values: np.ndarray, original: ArrayLike) -> float | np.ndarray:
    return float(values.item()) if np.asarray(original).ndim == 0 else values


def _call_nonnegative(function: RiskFunctional, value: float, name: str) -> float:
    result = float(function(float(value)))
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must return finite non-negative values")
    return result


def _standard_error(values: np.ndarray) -> float:
    return 0.0 if values.size == 1 else float(np.std(values, ddof=1) / math.sqrt(values.size))


def _with_initial_capital(model: RiskProcess, initial_capital: float) -> RiskProcess:
    capital = _nonnegative_float(initial_capital, "initial_capital")
    if isinstance(model, CramerLundbergProcess):
        return CramerLundbergProcess(
            initial_capital=capital,
            premium_rate=model.premium_rate,
            claim_arrival_rate=model.frequency.mean_rate(),
            claim_distribution=model.claim_distribution,
            prevention=model.prevention,
            by_claims=model.by_claims,
            capital_injections=model.capital_injections,
            name=model.name,
        )
    if isinstance(model, SparreAndersenProcess):
        if model.frequency.interarrival_distribution is None:
            raise ValueError("renewal model requires an interarrival distribution")
        return SparreAndersenProcess(
            initial_capital=capital,
            premium_rate=model.premium_rate,
            interarrival_distribution=model.frequency.interarrival_distribution,
            claim_distribution=model.claim_distribution,
            prevention=model.prevention,
            by_claims=model.by_claims,
            capital_injections=model.capital_injections,
            name=model.name,
        )
    return replace(model, initial_capital=capital)


def _segment_metrics(
    starts: np.ndarray,
    ends: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    red_time = np.zeros_like(durations)
    negative_area = np.zeros_like(durations)
    positive_duration = durations > 0.0

    both_negative = positive_duration & (starts < 0.0) & (ends < 0.0)
    red_time[both_negative] = durations[both_negative]
    negative_area[both_negative] = -0.5 * (
        starts[both_negative] + ends[both_negative]
    ) * durations[both_negative]

    leaves_red = positive_duration & (starts < 0.0) & (ends >= 0.0)
    if np.any(leaves_red):
        zero_time = durations[leaves_red] * (-starts[leaves_red]) / (
            ends[leaves_red] - starts[leaves_red]
        )
        red_time[leaves_red] = zero_time
        negative_area[leaves_red] = -0.5 * starts[leaves_red] * zero_time

    enters_red = positive_duration & (starts >= 0.0) & (ends < 0.0)
    if np.any(enters_red):
        zero_time = durations[enters_red] * (-starts[enters_red]) / (
            ends[enters_red] - starts[enters_red]
        )
        negative_duration = durations[enters_red] - zero_time
        red_time[enters_red] = negative_duration
        negative_area[enters_red] = -0.5 * ends[enters_red] * negative_duration

    return red_time, negative_area


def red_time_metrics_from_path(
    path: SimulationPath,
    *,
    reserve_shift: float = 0.0,
) -> RedTimePathMetrics:
    """Compute exact red-time metrics for a piecewise-linear simulated path."""

    if not isinstance(path, SimulationPath):
        raise TypeError("path must be a SimulationPath")
    shift = _finite_float(reserve_shift, "reserve_shift")
    times = np.asarray(path.times, dtype=float)
    reserves = np.asarray(path.reserves, dtype=float) + shift
    if times.ndim != 1 or reserves.ndim != 1 or times.shape != reserves.shape:
        raise ValueError("path times and reserves must be one-dimensional arrays with same shape")
    if times.size < 2:
        return RedTimePathMetrics(
            time_in_red=0.0,
            negative_area=0.0,
            minimum_reserve=float(np.min(reserves)),
            horizon=float(path.horizon),
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(reserves)):
        raise ValueError("path times and reserves must be finite")
    durations = np.diff(times)
    if np.any(durations < 0.0):
        raise ValueError("path times must be non-decreasing")

    red_time, negative_area = _segment_metrics(reserves[:-1], reserves[1:], durations)
    return RedTimePathMetrics(
        time_in_red=float(np.sum(red_time)),
        negative_area=float(np.sum(negative_area)),
        minimum_reserve=float(np.min(reserves)),
        horizon=float(path.horizon),
    )


def _path_value_at(path: SimulationPath, time: float) -> float:
    times = np.asarray(path.times, dtype=float)
    reserves = np.asarray(path.reserves, dtype=float)
    if time < times[0]:
        return float(reserves[0])
    index = int(np.searchsorted(times, time, side="right") - 1)
    index = max(0, min(index, times.size - 1))
    if index >= times.size - 1:
        return float(reserves[-1])
    start = float(times[index])
    end = float(times[index + 1])
    if end <= start:
        return float(reserves[index])
    weight = (time - start) / (end - start)
    return float(reserves[index] + weight * (reserves[index + 1] - reserves[index]))


def _linear_zero(start: float, end: float, left: float, right: float) -> float | None:
    if start == end or left == right or left * right >= 0.0:
        return None
    return start + (end - start) * (-left) / (right - left)


def multiline_red_time_metrics_from_paths(
    paths: list[SimulationPath] | tuple[SimulationPath, ...],
) -> MultilineRedTimeMetrics:
    """Compute multirisk red-time measures from synchronized reserve paths."""

    path_list = tuple(paths)
    if not path_list:
        raise ValueError("paths must contain at least one SimulationPath")
    if not all(isinstance(path, SimulationPath) for path in path_list):
        raise TypeError("paths must contain SimulationPath instances")
    horizon = min(float(path.horizon) for path in path_list)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("paths must have positive finite horizons")

    base_times = {0.0, horizon}
    for path in path_list:
        times = np.asarray(path.times, dtype=float)
        base_times.update(float(time) for time in times if 0.0 <= time <= horizon)
    ordered = np.array(sorted(base_times), dtype=float)
    red_time = np.zeros(len(path_list), dtype=float)
    positive_total_red = np.zeros(len(path_list), dtype=float)

    for start, end in zip(ordered[:-1], ordered[1:]):
        duration = float(end - start)
        if duration <= 0.0:
            continue
        left = np.array([_path_value_at(path, float(start)) for path in path_list])
        right = np.array([_path_value_at(path, float(end)) for path in path_list])
        cuts = [float(start), float(end)]
        for line_start, line_end in zip(left, right):
            zero = _linear_zero(float(start), float(end), float(line_start), float(line_end))
            if zero is not None:
                cuts.append(zero)
        total_zero = _linear_zero(
            float(start),
            float(end),
            float(np.sum(left)),
            float(np.sum(right)),
        )
        if total_zero is not None:
            cuts.append(total_zero)
        cuts = sorted(set(cuts))
        for sub_start, sub_end in zip(cuts[:-1], cuts[1:]):
            sub_duration = sub_end - sub_start
            if sub_duration <= 0.0:
                continue
            midpoint = 0.5 * (sub_start + sub_end)
            values = np.array([_path_value_at(path, midpoint) for path in path_list])
            red_time += sub_duration * (values < 0.0)
            if float(np.sum(values)) > 0.0:
                positive_total_red += sub_duration * (values < 0.0)

    individual = [red_time_metrics_from_path(path) for path in path_list]
    negative_area = np.array([item.negative_area for item in individual])
    return MultilineRedTimeMetrics(
        time_in_red=red_time,
        negative_area=negative_area,
        red_time_with_positive_total=positive_total_red,
        aggregate_negative_area=float(np.sum(negative_area)),
        positive_total_red_time_sum=float(np.sum(positive_total_red)),
        horizon=horizon,
    )


def expected_time_in_red_exponential(
    initial_capital: ArrayLike,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
) -> float | np.ndarray:
    """Infinite-horizon ``E[tau(u)]`` for Cramer-Lundberg exponential claims."""

    surplus = _as_1d_nonnegative(np.atleast_1d(initial_capital), "initial_capital")
    premium = _positive_float(premium_rate, "premium_rate")
    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    severity_rate = _positive_float(claim_rate, "claim_rate")
    adjustment = severity_rate - arrival / premium
    if adjustment <= 0.0:
        raise ValueError("positive safety loading is required")
    values = arrival * np.exp(-adjustment * surplus) / (premium * premium * adjustment**2)
    return _maybe_scalar(values, initial_capital)


def expected_negative_area_exponential(
    initial_capital: ArrayLike,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
) -> float | np.ndarray:
    """Infinite-horizon ``E[I_infinity(u)]`` for exponential claims."""

    surplus = _as_1d_nonnegative(np.atleast_1d(initial_capital), "initial_capital")
    premium = _positive_float(premium_rate, "premium_rate")
    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    severity_rate = _positive_float(claim_rate, "claim_rate")
    adjustment = severity_rate - arrival / premium
    if adjustment <= 0.0:
        raise ValueError("positive safety loading is required")
    values = arrival * np.exp(-adjustment * surplus) / (premium * premium * adjustment**3)
    return _maybe_scalar(values, initial_capital)


def estimate_red_time_metrics(
    model: RiskProcess,
    *,
    horizon: float,
    n_simulations: int = 10_000,
    seed: int | np.random.Generator | None = None,
    max_events: int = 1_000_000,
) -> RedTimeEstimate:
    """Estimate ``E[tau_T(u)]`` and ``E[I_T(u)]`` by Monte Carlo simulation."""

    if not isinstance(model, RiskProcess):
        raise TypeError("model must be a RiskProcess")
    time_horizon = _positive_float(horizon, "horizon")
    n_paths = _positive_int(n_simulations, "n_simulations")
    maximum_events = _positive_int(max_events, "max_events")
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
    times = np.empty(n_paths, dtype=float)
    areas = np.empty(n_paths, dtype=float)
    for index in range(n_paths):
        path = simulate_path(
            model,
            time_horizon,
            seed=rng,
            max_events=maximum_events,
            stop_at_ruin=False,
        )
        metrics = red_time_metrics_from_path(path)
        times[index] = metrics.time_in_red
        areas[index] = metrics.negative_area

    return RedTimeEstimate(
        time_in_red=times,
        negative_area=areas,
        expected_time_in_red=float(np.mean(times)),
        expected_negative_area=float(np.mean(areas)),
        time_in_red_standard_error=_standard_error(times),
        negative_area_standard_error=_standard_error(areas),
        n_simulations=n_paths,
        horizon=time_horizon,
    )


def estimate_red_time_curve(
    model: RiskProcess,
    initial_capitals: ArrayLike,
    *,
    horizon: float,
    n_simulations: int = 10_000,
    seed: int | np.random.Generator | None = None,
    max_events: int = 1_000_000,
) -> RedTimeCurveResult:
    """Estimate red-time curves over several initial reserve levels."""

    capitals = _as_1d_nonnegative(initial_capitals, "initial_capitals")
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
    estimates = [
        estimate_red_time_metrics(
            _with_initial_capital(model, float(capital)),
            horizon=horizon,
            n_simulations=n_simulations,
            seed=rng,
            max_events=max_events,
        )
        for capital in capitals
    ]
    return RedTimeCurveResult(
        initial_capitals=capitals,
        expected_time_in_red=np.array([item.expected_time_in_red for item in estimates]),
        expected_negative_area=np.array([item.expected_negative_area for item in estimates]),
        time_in_red_standard_error=np.array(
            [item.time_in_red_standard_error for item in estimates],
        ),
        negative_area_standard_error=np.array(
            [item.negative_area_standard_error for item in estimates],
        ),
        n_simulations=int(n_simulations),
        horizon=_positive_float(horizon, "horizon"),
    )


def negative_area_derivative_identity_error(
    path: SimulationPath,
    *,
    reserve_shift: float = 0.0,
    step: float = 1e-4,
) -> float:
    """Finite-difference error in ``d I_T(u) / du = -tau_T(u)`` for one path."""

    shift = _finite_float(reserve_shift, "reserve_shift")
    h = _positive_float(step, "step")
    area_plus = red_time_metrics_from_path(path, reserve_shift=shift + h).negative_area
    area_minus = red_time_metrics_from_path(path, reserve_shift=shift - h).negative_area
    derivative = (area_plus - area_minus) / (2.0 * h)
    time_in_red = red_time_metrics_from_path(path, reserve_shift=shift).time_in_red
    return float(derivative + time_in_red)


def red_time_derivative(
    red_time_function: RiskFunctional,
    initial_capital: float,
    *,
    step: float = 1e-4,
) -> float:
    """Central finite-difference derivative of an expected red-time curve."""

    capital = _nonnegative_float(initial_capital, "initial_capital")
    h = _positive_float(step, "step")
    lower = max(0.0, capital - h)
    upper = capital + h
    if lower == capital:
        return (
            _call_nonnegative(red_time_function, upper, "red_time_function")
            - _call_nonnegative(red_time_function, capital, "red_time_function")
        ) / h
    return (
        _call_nonnegative(red_time_function, upper, "red_time_function")
        - _call_nonnegative(red_time_function, lower, "red_time_function")
    ) / (upper - lower)


def multirisk_red_time_derivative_sum(
    allocations: ArrayLike,
    red_time_functions: list[RiskFunctional] | tuple[RiskFunctional, ...],
    *,
    step: float = 1e-4,
) -> float:
    """Compute ``sum_k d E[tau_k(u_k)] / du_k`` by finite differences."""

    reserves = _as_1d_nonnegative(allocations, "allocations")
    if len(red_time_functions) != reserves.size:
        raise ValueError("red_time_functions must match the number of allocations")
    return float(
        sum(
            red_time_derivative(function, reserve, step=step)
            for function, reserve in zip(red_time_functions, reserves)
        )
    )


def _validate_bounds(
    total_reserve: float,
    n_lines: int,
    lower_bounds: ArrayLike | None,
    upper_bounds: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    total = _nonnegative_float(total_reserve, "total_reserve")
    lower = np.zeros(n_lines, dtype=float) if lower_bounds is None else _as_1d_nonnegative(
        lower_bounds,
        "lower_bounds",
    )
    if lower.size != n_lines:
        raise ValueError("lower_bounds must match the number of branches")
    if upper_bounds is None:
        upper = lower + total - float(np.sum(lower))
    else:
        upper = _as_1d_nonnegative(upper_bounds, "upper_bounds")
        if upper.size != n_lines:
            raise ValueError("upper_bounds must match the number of branches")
    if np.any(upper < lower):
        raise ValueError("upper_bounds must be greater than or equal to lower_bounds")
    if np.sum(lower) - total > 1e-12 or total - np.sum(upper) > 1e-12:
        raise ValueError("total_reserve must lie between the sums of lower and upper bounds")
    return lower, upper


def _allocation_at_threshold(
    threshold: float,
    red_time_functions: list[RiskFunctional] | tuple[RiskFunctional, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    allocations = np.empty(lower.size, dtype=float)
    for index, function in enumerate(red_time_functions):
        low = float(lower[index])
        high = float(upper[index])
        red_low = _call_nonnegative(function, low, "red_time_function")
        red_high = _call_nonnegative(function, high, "red_time_function")
        if red_high > red_low + tolerance:
            raise ValueError("red_time_functions must be non-increasing on the bounds")
        if red_low <= threshold:
            allocations[index] = low
            continue
        if red_high >= threshold:
            allocations[index] = high
            continue
        left = low
        right = high
        for _ in range(max_iterations):
            midpoint = 0.5 * (left + right)
            value = _call_nonnegative(function, midpoint, "red_time_function")
            if abs(value - threshold) <= tolerance or right - left <= tolerance:
                left = right = midpoint
                break
            if value > threshold:
                left = midpoint
            else:
                right = midpoint
        allocations[index] = 0.5 * (left + right)
    return allocations


def optimize_reserve_allocation(
    *,
    total_reserve: float,
    red_time_functions: list[RiskFunctional] | tuple[RiskFunctional, ...],
    negative_area_functions: list[RiskFunctional] | tuple[RiskFunctional, ...] | None = None,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ReserveAllocationResult:
    """Allocate reserves by equalizing expected times in red on active branches."""

    if not red_time_functions:
        raise ValueError("red_time_functions must contain at least one callable")
    if not all(callable(function) for function in red_time_functions):
        raise TypeError("red_time_functions must be callable")
    if negative_area_functions is not None:
        if len(negative_area_functions) != len(red_time_functions):
            raise ValueError("negative_area_functions must match red_time_functions")
        if not all(callable(function) for function in negative_area_functions):
            raise TypeError("negative_area_functions must be callable")
    total = _nonnegative_float(total_reserve, "total_reserve")
    tol = _positive_float(tolerance, "tolerance")
    maximum = _positive_int(max_iterations, "max_iterations")
    lower, upper = _validate_bounds(total, len(red_time_functions), lower_bounds, upper_bounds)

    high = max(
        _call_nonnegative(function, float(bound), "red_time_function")
        for function, bound in zip(red_time_functions, lower)
    )
    low = 0.0
    allocations = lower.copy()
    converged = False
    iteration = 0
    for iteration in range(1, maximum + 1):
        threshold = 0.5 * (low + high)
        allocations = _allocation_at_threshold(
            threshold,
            red_time_functions,
            lower,
            upper,
            tolerance=tol,
            max_iterations=maximum,
        )
        allocated = float(np.sum(allocations))
        if abs(allocated - total) <= tol:
            converged = True
            break
        if allocated > total:
            low = threshold
        else:
            high = threshold

    threshold = 0.5 * (low + high)
    red_times = np.array(
        [
            _call_nonnegative(function, float(allocation), "red_time_function")
            for function, allocation in zip(red_time_functions, allocations)
        ],
    )
    negative_areas = None
    objective = None
    if negative_area_functions is not None:
        negative_areas = np.array(
            [
                _call_nonnegative(function, float(allocation), "negative_area_function")
                for function, allocation in zip(negative_area_functions, allocations)
            ],
        )
        objective = float(np.sum(negative_areas))

    return ReserveAllocationResult(
        total_reserve=total,
        allocations=allocations,
        red_times=red_times,
        negative_areas=negative_areas,
        objective_value=objective,
        active=allocations > lower + math.sqrt(tol),
        threshold=float(threshold),
        converged=converged,
        iterations=iteration,
    )


def simplex_reserve_grid(
    *,
    total_reserve: float,
    n_lines: int,
    subdivisions: int = 20,
    lower_bounds: ArrayLike | None = None,
) -> np.ndarray:
    """Build a regular lattice on the reserve-allocation simplex."""

    total = _nonnegative_float(total_reserve, "total_reserve")
    lines = _positive_int(n_lines, "n_lines")
    steps = _positive_int(subdivisions, "subdivisions")
    lower = np.zeros(lines, dtype=float) if lower_bounds is None else _as_1d_nonnegative(
        lower_bounds,
        "lower_bounds",
    )
    if lower.size != lines:
        raise ValueError("lower_bounds must match n_lines")
    remainder = total - float(np.sum(lower))
    if remainder < -1e-12:
        raise ValueError("lower_bounds cannot sum above total_reserve")
    if lines == 1:
        return np.array([[total]], dtype=float)

    rows: list[list[int]] = []

    def append_compositions(remaining: int, parts: int, prefix: list[int]) -> None:
        if parts == 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            append_compositions(remaining - value, parts - 1, [*prefix, value])

    append_compositions(steps, lines, [])
    weights = np.asarray(rows, dtype=float) / steps
    return lower + remainder * weights


def evaluate_reserve_allocation_grid(
    allocations: ArrayLike,
    negative_area_functions: list[RiskFunctional] | tuple[RiskFunctional, ...],
    *,
    red_time_functions: list[RiskFunctional] | tuple[RiskFunctional, ...] | None = None,
) -> AllocationGridResult:
    """Evaluate multirisk criteria on a grid of reserve allocations."""

    grid = np.asarray(allocations, dtype=float)
    if grid.ndim != 2 or grid.shape[0] == 0 or grid.shape[1] == 0:
        raise ValueError("allocations must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise ValueError("allocations must contain finite non-negative values")
    n_lines = grid.shape[1]
    if len(negative_area_functions) != n_lines:
        raise ValueError("negative_area_functions must match allocation columns")
    if not all(callable(function) for function in negative_area_functions):
        raise TypeError("negative_area_functions must be callable")
    if red_time_functions is not None:
        if len(red_time_functions) != n_lines:
            raise ValueError("red_time_functions must match allocation columns")
        if not all(callable(function) for function in red_time_functions):
            raise TypeError("red_time_functions must be callable")

    negative_areas = np.column_stack(
        [
            [
                _call_nonnegative(function, float(value), "negative_area_function")
                for value in column
            ]
            for function, column in zip(negative_area_functions, grid.T)
        ],
    )
    red_times = None
    if red_time_functions is not None:
        red_times = np.column_stack(
            [
                [_call_nonnegative(function, float(value), "red_time_function") for value in column]
                for function, column in zip(red_time_functions, grid.T)
            ],
        )

    return AllocationGridResult(
        allocations=grid,
        objective_values=np.sum(negative_areas, axis=1),
        negative_areas=negative_areas,
        red_times=red_times,
        total_reserve=float(np.mean(np.sum(grid, axis=1))),
    )
