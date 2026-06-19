"""Exact finite-time ruin formulas for integer-valued claims."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import operator
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


FiniteTimeDiscreteMethod = Literal["seal", "takacs", "picard-lefevre", "inventory"]
BoundaryRuinConvention = Literal["negative", "nonpositive"]
BoundaryKind = Literal["value", "crossing"]


@dataclass(frozen=True)
class FiniteTimeDiscreteRuinResult:
    """Exact finite-time ruin result for a lattice Cramer-Lundberg model."""

    initial_capital: float
    horizon: float
    premium_rate: float
    claim_arrival_rate: float
    claim_pmf: np.ndarray
    method: str
    survival_probability: float
    ruin_probability: float
    premium_units: float
    inventory_times: np.ndarray
    survival_probabilities: np.ndarray
    state_probabilities: np.ndarray
    convention: str

    @property
    def ruin_probabilities_by_time(self) -> np.ndarray:
        """Ruin probabilities at the returned inventory dates."""

        return 1.0 - self.survival_probabilities


@dataclass(frozen=True)
class FiniteTimeDiscreteBoundaryResult:
    """Exact finite-time survival result for an increasing lattice boundary."""

    horizon: float
    claim_arrival_rate: float | None
    arrival_means: np.ndarray
    claim_pmf: np.ndarray
    survival_probability: float
    ruin_probability: float
    inventory_times: np.ndarray
    retained_counts: np.ndarray
    boundary_values: np.ndarray
    survival_probabilities: np.ndarray
    state_probabilities: np.ndarray
    convention: str

    @property
    def ruin_probabilities_by_time(self) -> np.ndarray:
        """Ruin probabilities at the returned inventory dates."""

        return 1.0 - self.survival_probabilities


@dataclass(frozen=True)
class FiniteTimeDiscreteBoundaryGrid:
    """Inventory grid generated from an increasing boundary function."""

    horizon: float
    inventory_times: np.ndarray
    boundary_values: np.ndarray


@dataclass(frozen=True)
class FiniteTimeDiscreteAppellResult:
    """Picard-Lefevre generalized-Appell finite-time ruin result."""

    horizon: float
    claim_arrival_rate: float
    effective_claim_arrival_rate: float
    claim_pmf: np.ndarray
    survival_probability: float
    ruin_probability: float
    boundary_grid: FiniteTimeDiscreteBoundaryGrid
    appell_coefficients: np.ndarray
    state_probabilities: np.ndarray
    convention: str


@dataclass(frozen=True)
class FiniteTimeDiscreteNonhomogeneousResult:
    """Exact finite-time result for non-stationary discrete claim increments."""

    horizon: float
    claim_size_intensities: np.ndarray
    survival_probability: float
    ruin_probability: float
    inventory_times: np.ndarray
    retained_counts: np.ndarray
    boundary_values: np.ndarray
    survival_probabilities: np.ndarray
    state_probabilities: np.ndarray
    convention: str

    @property
    def ruin_probabilities_by_time(self) -> np.ndarray:
        """Ruin probabilities at the returned inventory dates."""

        return 1.0 - self.survival_probabilities


def _nonnegative_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _claim_pmf(values: ArrayLike) -> np.ndarray:
    pmf = np.asarray(values, dtype=float)
    if pmf.ndim != 1 or pmf.size == 0:
        raise ValueError("claim_pmf must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(pmf)) or np.any(pmf < 0.0):
        raise ValueError("claim_pmf must contain finite non-negative probabilities")
    total = float(np.sum(pmf))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("claim_pmf must sum to one for exact finite-time formulas")
    return pmf.copy()


def _method(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("method must be a string")
    method = value.lower().replace("_", "-")
    aliases = {
        "direct": "inventory",
        "recursive": "inventory",
        "picard-lefevre-recursion": "inventory",
        "pl": "picard-lefevre",
        "seal-takacs": "seal",
    }
    method = aliases.get(method, method)
    if method not in {"seal", "takacs", "picard-lefevre", "inventory"}:
        raise ValueError("method must be 'seal', 'takacs', 'picard-lefevre' or 'inventory'")
    return method


def _boundary_convention(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("convention must be a string")
    convention = value.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "strict": "negative",
        "strict-negative": "negative",
        "negative-reserve": "negative",
        "below-zero": "negative",
        "ruin-below-zero": "negative",
        "non-positive": "nonpositive",
        "non-positive-reserve": "nonpositive",
        "at-or-below-zero": "nonpositive",
        "ruin-at-zero": "nonpositive",
        "zero": "nonpositive",
    }
    convention = aliases.get(convention, convention)
    if convention not in {"negative", "nonpositive"}:
        raise ValueError("convention must be 'negative' or 'nonpositive'")
    return convention


def _boundary_kind(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("boundary_kind must be a string")
    kind = value.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "values": "value",
        "level": "value",
        "levels": "value",
        "inverse": "crossing",
        "inverse-times": "crossing",
        "crossings": "crossing",
    }
    kind = aliases.get(kind, kind)
    if kind not in {"value", "crossing"}:
        raise ValueError("boundary_kind must be 'value' or 'crossing'")
    return kind


def _floor_nonnegative(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return int(nearest)
    return int(math.floor(value))


def _ceil_nonnegative(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return int(nearest)
    return int(math.ceil(value))


def _capital_parts(initial_capital: float) -> tuple[int, float]:
    floor_value = _floor_nonnegative(initial_capital)
    epsilon = initial_capital - floor_value
    if math.isclose(epsilon, 1.0, rel_tol=0.0, abs_tol=1e-12):
        floor_value += 1
        epsilon = 0.0
    if math.isclose(epsilon, 0.0, rel_tol=0.0, abs_tol=1e-12):
        epsilon = 0.0
    return floor_value, epsilon


def _as_finite_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _inventory_times(values: ArrayLike) -> np.ndarray:
    times = _as_finite_1d(values, "inventory_times")
    if np.any(times < 0.0):
        raise ValueError("inventory_times must be non-negative")
    if np.any(np.diff(times) < -1e-12):
        raise ValueError("inventory_times must be non-decreasing")
    return times


def _retained_counts(values: ArrayLike, expected_size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError("retained_counts must match inventory_times length")
    counts = np.empty(array.size, dtype=int)
    for index, value in enumerate(array):
        counts[index] = _nonnegative_int(value, "retained_counts")
    return counts


def _claim_size_intensities(values: ArrayLike, name: str = "claim_size_intensities") -> np.ndarray:
    intensities = np.asarray(values, dtype=float)
    if intensities.ndim != 1 or intensities.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(intensities)) or np.any(intensities < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return intensities.copy()


def _claim_size_intensity_matrix(values: ArrayLike, expected_size: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != expected_size or matrix.shape[1] == 0:
        raise ValueError("claim_size_intensities must have one row per inventory interval")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("claim_size_intensities must contain finite non-negative values")
    return matrix.copy()


def _claim_size_intensity_matrix_from_intervals(
    claim_size_intensity_integrals: Callable[[float, float], ArrayLike],
    times: np.ndarray,
) -> np.ndarray:
    if not callable(claim_size_intensity_integrals):
        raise TypeError("claim_size_intensity_integrals must be callable")

    rows: list[np.ndarray] = []
    width = 1
    previous = 0.0
    for time in times:
        current = float(time)
        row = _claim_size_intensities(
            claim_size_intensity_integrals(previous, current),
            "claim_size_intensity_integrals",
        )
        rows.append(row)
        width = max(width, row.size)
        previous = current

    matrix = np.zeros((len(rows), width), dtype=float)
    for index, row in enumerate(rows):
        matrix[index, : row.size] = row
    return matrix


def _arrival_means(
    times: np.ndarray,
    *,
    claim_arrival_rate: float | None,
    arrival_means: ArrayLike | None,
) -> tuple[float | None, np.ndarray]:
    if (claim_arrival_rate is None) == (arrival_means is None):
        raise ValueError("provide exactly one of claim_arrival_rate or arrival_means")
    if arrival_means is not None:
        means = _as_finite_1d(arrival_means, "arrival_means")
        if means.size != times.size:
            raise ValueError("arrival_means must match inventory_times length")
        if np.any(means < 0.0):
            raise ValueError("arrival_means must be non-negative")
        return None, means
    rate = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    elapsed = np.diff(np.concatenate(([0.0], times)))
    elapsed[np.abs(elapsed) <= 1e-12] = 0.0
    return rate, rate * elapsed


def _arrival_means_from_cumulative(
    cumulative_arrival_mean: Callable[[float], float],
    times: np.ndarray,
) -> np.ndarray:
    if not callable(cumulative_arrival_mean):
        raise TypeError("cumulative_arrival_mean must be callable")
    values: list[float] = []
    for time in np.concatenate(([0.0], times)):
        try:
            value = float(cumulative_arrival_mean(float(time)))
        except (TypeError, ValueError) as exc:
            raise TypeError("cumulative_arrival_mean must return numeric values") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("cumulative_arrival_mean must return finite non-negative values")
        values.append(value)
    cumulative = np.asarray(values, dtype=float)
    means = np.diff(cumulative)
    if np.any(means < -1e-12):
        raise ValueError("cumulative_arrival_mean must be non-decreasing")
    means[np.abs(means) <= 1e-12] = 0.0
    return means


def _retained_count_from_boundary(boundary_value: float, convention: str) -> int:
    boundary = _finite_nonnegative(boundary_value, "boundary_values")
    if convention == "negative":
        return _floor_nonnegative(boundary) + 1
    return _ceil_nonnegative(boundary)


def _retained_count_from_crossing(boundary_value: float) -> int:
    boundary = _finite_nonnegative(boundary_value, "boundary_values")
    return _ceil_nonnegative(boundary)


def _retained_counts_from_boundary_values(
    boundaries: np.ndarray,
    *,
    convention: str,
    boundary_kind: str,
) -> np.ndarray:
    if boundary_kind == "value":
        return np.fromiter(
            (_retained_count_from_boundary(value, convention) for value in boundaries),
            dtype=int,
            count=boundaries.size,
        )
    return np.fromiter(
        (_retained_count_from_crossing(value) for value in boundaries),
        dtype=int,
        count=boundaries.size,
    )


def _positive_claim_process(
    claim_pmf: np.ndarray,
    claim_arrival_rate: float,
) -> tuple[np.ndarray, float]:
    nonzero_probability = float(np.sum(claim_pmf[1:]))
    if nonzero_probability <= 0.0:
        return claim_pmf.copy(), 0.0
    positive = np.zeros_like(claim_pmf, dtype=float)
    positive[1:] = claim_pmf[1:] / nonzero_probability
    return positive, claim_arrival_rate * nonzero_probability


def compound_poisson_appell_base(
    claim_pmf: ArrayLike,
    *,
    claim_arrival_rate: float,
    time: float,
    max_degree: int,
) -> np.ndarray:
    """Evaluate Picard-Lefevre base polynomials ``e_n(t)`` up to ``max_degree``."""

    pmf = _claim_pmf(claim_pmf)
    rate = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    horizon = float(time)
    if not math.isfinite(horizon):
        raise ValueError("time must be finite")
    max_index = _nonnegative_int(max_degree, "max_degree")
    positive_pmf, effective_rate = _positive_claim_process(pmf, rate)
    if math.isclose(effective_rate, 0.0, rel_tol=0.0, abs_tol=1e-15):
        values = np.zeros(max_index + 1, dtype=float)
        values[0] = 1.0
        return values
    if horizon < 0.0:
        raise ValueError("time must be non-negative for numerical Appell evaluation")
    aggregate = _compound_poisson_lattice_pmf(
        positive_pmf,
        effective_rate * horizon,
        max_index,
    )
    return math.exp(effective_rate * horizon) * aggregate


def _call_boundary(boundary: Callable[[float], float], time: float) -> float:
    try:
        value = float(boundary(float(time)))
    except (TypeError, ValueError) as exc:
        raise TypeError("boundary must return numeric values") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("boundary must return finite non-negative values")
    return value


def _boundary_crossing_time(
    boundary: Callable[[float], float],
    *,
    level: int,
    low: float,
    high: float,
    tol: float,
    max_iter: int,
) -> float:
    if _call_boundary(boundary, low) >= level:
        return low
    if _call_boundary(boundary, high) < level:
        raise ValueError("boundary does not reach all required integer levels")
    left = low
    right = high
    for _ in range(max_iter):
        middle = 0.5 * (left + right)
        if right - left <= tol:
            break
        if _call_boundary(boundary, middle) >= level:
            right = middle
        else:
            left = middle
    return right


def finite_time_discrete_boundary_crossings(
    boundary: Callable[[float], float],
    *,
    horizon: float,
    root_tol: float = 1e-10,
    max_bisection: int = 80,
) -> FiniteTimeDiscreteBoundaryGrid:
    """Build inverse crossing dates for an increasing deterministic boundary.

    The returned grid contains the integer levels crossed before ``horizon``,
    plus the terminal horizon when the boundary does not end exactly on the
    last crossed integer level.
    """

    if not callable(boundary):
        raise TypeError("boundary must be callable")
    time = _finite_nonnegative(horizon, "horizon")
    tol = _finite_positive(root_tol, "root_tol")
    iterations = _nonnegative_int(max_bisection, "max_bisection")
    if iterations == 0:
        raise ValueError("max_bisection must be positive")

    start_value = _call_boundary(boundary, 0.0)
    end_value = _call_boundary(boundary, time)
    if end_value + 1e-12 < start_value:
        raise ValueError("boundary must be non-decreasing on the horizon")

    first_level = _floor_nonnegative(start_value) + 1
    last_level = _floor_nonnegative(end_value)
    times: list[float] = []
    values: list[float] = []
    low = 0.0
    for level in range(first_level, last_level + 1):
        crossing = _boundary_crossing_time(
            boundary,
            level=level,
            low=low,
            high=time,
            tol=tol,
            max_iter=iterations,
        )
        times.append(crossing)
        values.append(float(level))
        low = crossing

    if not times or not math.isclose(times[-1], time, rel_tol=0.0, abs_tol=tol):
        times.append(time)
        values.append(end_value)

    return FiniteTimeDiscreteBoundaryGrid(
        horizon=time,
        inventory_times=np.asarray(times, dtype=float),
        boundary_values=np.asarray(values, dtype=float),
    )


def finite_time_discrete_computation_set(
    *,
    initial_capital: float,
    premium_units: float,
    method: FiniteTimeDiscreteMethod = "seal",
) -> np.ndarray:
    """Return the ``(tau, j)`` lattice points used by a finite-time formula.

    This diagnostic is useful for reproducing the computation-set figures in
    Loisel's presentation of the Picard-Lefevre and Seal/Takacs formulas.
    """

    u = _finite_nonnegative(initial_capital, "initial_capital")
    x = _finite_nonnegative(premium_units, "premium_units")
    selected = _method(method)
    u_floor, eps_u = _capital_parts(u)
    x_floor, eps_x = _capital_parts(x)
    nu = int(math.floor(eps_u + eps_x + 1e-12))
    points: set[tuple[float, int]] = set()

    if selected in {"seal", "takacs"}:
        final_index = _floor_nonnegative(u + x)
        points.update((x, index) for index in range(final_index + 1))
        if selected == "takacs":
            return np.asarray(sorted(points), dtype=float)
        for k in range(1, x_floor + nu + 1):
            points.add((k - eps_u, u_floor + k))
            tail_time = x - k + eps_u
            tail_index = x_floor - k + nu
            points.update((tail_time, index) for index in range(tail_index + 1))
    elif selected == "picard-lefevre":
        for j in range(u_floor + 1):
            points.add((j - u, j))
            tail_time = x + u - j
            tail_index = _floor_nonnegative(tail_time)
            points.update((tail_time, index) for index in range(tail_index + 1))
    else:
        first_boundary = u_floor + 1
        final_boundary = _floor_nonnegative(u + x) + 1
        for boundary in range(first_boundary, final_boundary + 1):
            inventory_time = x if boundary == final_boundary else boundary - u
            points.update((inventory_time, index) for index in range(boundary))
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(sorted(points), dtype=float)


def compound_poisson_lattice_pmf(
    claim_pmf: ArrayLike,
    *,
    mean: float,
    max_aggregate: int,
) -> np.ndarray:
    """Return ``P(S=j)`` for a compound Poisson sum with integer severities.

    ``claim_pmf[k]`` is the probability of a claim of amount ``k``. The
    returned array is exact on indices ``0, ..., max_aggregate``; probability
    mass above ``max_aggregate`` is intentionally not returned.
    """

    pmf = _claim_pmf(claim_pmf)
    poisson_mean = _finite_nonnegative(mean, "mean")
    max_index = _nonnegative_int(max_aggregate, "max_aggregate")
    return _compound_poisson_lattice_pmf(pmf, poisson_mean, max_index)


def nonhomogeneous_compound_poisson_lattice_pmf(
    claim_size_intensities: ArrayLike,
    *,
    max_aggregate: int,
) -> np.ndarray:
    """Return aggregate masses from integrated claim-size intensities.

    ``claim_size_intensities[k]`` is the integrated intensity of claims of
    size ``k`` over an interval. Index 0 is ignored because zero-size claims do
    not change the aggregate claim amount.
    """

    intensities = _claim_size_intensities(claim_size_intensities)
    max_index = _nonnegative_int(max_aggregate, "max_aggregate")
    return _nonhomogeneous_compound_poisson_lattice_pmf(intensities, max_index)


def _compound_poisson_lattice_pmf(
    claim_pmf: np.ndarray,
    mean: float,
    max_aggregate: int,
) -> np.ndarray:
    aggregate = np.zeros(max_aggregate + 1, dtype=float)
    aggregate[0] = math.exp(-mean * (1.0 - float(claim_pmf[0])))
    if max_aggregate == 0:
        return aggregate
    support_max = min(claim_pmf.size - 1, max_aggregate)
    for j in range(1, max_aggregate + 1):
        upper = min(j, support_max)
        if upper == 0:
            continue
        indices = np.arange(1, upper + 1)
        weighted_previous = indices * claim_pmf[1 : upper + 1] * aggregate[j - indices]
        aggregate[j] = mean * float(np.sum(weighted_previous)) / j
    return aggregate


def _nonhomogeneous_compound_poisson_lattice_pmf(
    claim_size_intensities: np.ndarray,
    max_aggregate: int,
) -> np.ndarray:
    aggregate = np.zeros(max_aggregate + 1, dtype=float)
    aggregate[0] = math.exp(-float(np.sum(claim_size_intensities[1:])))
    if max_aggregate == 0:
        return aggregate
    support_max = min(claim_size_intensities.size - 1, max_aggregate)
    for total in range(1, max_aggregate + 1):
        upper = min(total, support_max)
        if upper == 0:
            continue
        sizes = np.arange(1, upper + 1)
        aggregate[total] = (
            float(np.sum(sizes * claim_size_intensities[1 : upper + 1] * aggregate[total - sizes]))
            / total
        )
    return aggregate


def _h_values(
    claim_pmf: np.ndarray,
    premium_units: float,
    claim_arrival_rate: float,
    premium_rate: float,
    max_index: int,
) -> np.ndarray:
    mean = claim_arrival_rate * premium_units / premium_rate
    return _compound_poisson_lattice_pmf(claim_pmf, mean, max_index)


def _h_tilde(values: np.ndarray, premium_units: float, max_index: int) -> float:
    if max_index < 0:
        return 0.0
    if math.isclose(premium_units, 0.0, abs_tol=1e-14):
        return 1.0
    indices = np.arange(max_index + 1, dtype=float)
    return float(math.fsum((1.0 - indices / premium_units) * values[: max_index + 1]))


def _seal_survival(
    claim_pmf: np.ndarray,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    u_floor, eps_u = _capital_parts(initial_capital)
    x_floor, eps_x = _capital_parts(x)
    nu = int(math.floor(eps_u + eps_x + 1e-12))
    final_index = _floor_nonnegative(initial_capital + x)
    final = _h_values(claim_pmf, x, claim_arrival_rate, premium_rate, final_index)
    survival = math.fsum(final[: final_index + 1])
    correction = 0.0
    for k in range(1, x_floor + nu + 1):
        hit = _h_values(
            claim_pmf,
            k - eps_u,
            claim_arrival_rate,
            premium_rate,
            u_floor + k,
        )[u_floor + k]
        tail_time = x - k + eps_u
        tail_index = x_floor - k + nu
        tail = _h_values(claim_pmf, tail_time, claim_arrival_rate, premium_rate, tail_index)
        correction += hit * _h_tilde(tail, tail_time, tail_index)
    return survival - correction


def _takacs_survival(
    claim_pmf: np.ndarray,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    if math.isclose(x, 0.0, abs_tol=1e-14):
        return 1.0
    n = _floor_nonnegative(x)
    values = _h_values(claim_pmf, x, claim_arrival_rate, premium_rate, n)
    return _h_tilde(values, x, n)


def _picard_lefevre_survival(
    claim_pmf: np.ndarray,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> float:
    x = premium_rate * horizon
    if math.isclose(x, 0.0, abs_tol=1e-14):
        return 1.0
    u_floor, _ = _capital_parts(initial_capital)
    survival = 0.0
    for j in range(u_floor + 1):
        formal = _h_values(
            claim_pmf,
            float(j - initial_capital),
            claim_arrival_rate,
            premium_rate,
            j,
        )[j]
        tail_time = x + initial_capital - j
        tail_index = _floor_nonnegative(tail_time)
        values = _h_values(claim_pmf, tail_time, claim_arrival_rate, premium_rate, tail_index)
        survival += formal * _h_tilde(values, tail_time, tail_index)
    return survival


def _inventory_result(
    claim_pmf: np.ndarray,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    final_boundary = _floor_nonnegative(initial_capital + premium_rate * horizon) + 1
    first_boundary = _floor_nonnegative(initial_capital) + 1
    inventory_times: list[float] = []
    retained_counts: list[int] = []
    for boundary in range(first_boundary, final_boundary + 1):
        if boundary == final_boundary:
            inventory_time = horizon
        else:
            inventory_time = (boundary - initial_capital) / premium_rate
        inventory_times.append(inventory_time)
        retained_counts.append(boundary)
    rate, means = _arrival_means(
        np.asarray(inventory_times, dtype=float),
        claim_arrival_rate=claim_arrival_rate,
        arrival_means=None,
    )
    if rate is None:
        raise RuntimeError("homogeneous inventory recursion did not build an arrival rate")
    return _inventory_from_counts(
        claim_pmf,
        inventory_times=np.asarray(inventory_times, dtype=float),
        retained_counts=np.asarray(retained_counts, dtype=int),
        arrival_means=means,
    )


def _inventory_from_counts(
    claim_pmf: np.ndarray,
    *,
    inventory_times: np.ndarray,
    retained_counts: np.ndarray,
    arrival_means: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    max_count = max(int(np.max(retained_counts)) if retained_counts.size else 0, 1)
    increment_pmfs = np.zeros((arrival_means.size, max_count), dtype=float)
    for index, mean in enumerate(arrival_means):
        if math.isclose(float(mean), 0.0, rel_tol=0.0, abs_tol=1e-15):
            increment_pmfs[index, 0] = 1.0
        else:
            increment_pmfs[index] = _compound_poisson_lattice_pmf(
                claim_pmf,
                float(mean),
                max_count - 1,
            )
    return _inventory_from_increment_pmfs(
        inventory_times=inventory_times,
        retained_counts=retained_counts,
        increment_pmfs=increment_pmfs,
    )


def _inventory_from_increment_pmfs(
    *,
    inventory_times: np.ndarray,
    retained_counts: np.ndarray,
    increment_pmfs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    max_count = max(int(np.max(retained_counts)) if retained_counts.size else 0, 1)
    state = np.zeros(max_count, dtype=float)
    state[0] = 1.0
    survivals: list[float] = []

    for retained, increment in zip(retained_counts, increment_pmfs, strict=True):
        if math.isclose(float(increment[0]), 1.0, rel_tol=0.0, abs_tol=1e-15) and np.allclose(
            increment[1:],
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):
            convolved = state
        else:
            convolved = np.convolve(state, increment)[:max_count]
        next_state = np.zeros_like(state)
        retained = min(int(retained), max_count)
        if retained > 0:
            next_state[:retained] = convolved[:retained]
        state = next_state
        survivals.append(float(np.sum(state)))

    return float(np.sum(state)), inventory_times.copy(), np.asarray(survivals), state.copy()


def _nonhomogeneous_result_from_counts(
    claim_size_intensities: np.ndarray,
    *,
    inventory_times: np.ndarray,
    retained_counts: np.ndarray,
    boundary_values: np.ndarray,
    convention: str,
    return_result: bool,
) -> float | FiniteTimeDiscreteNonhomogeneousResult:
    max_count = max(int(np.max(retained_counts)) if retained_counts.size else 0, 1)
    increment_pmfs = np.vstack(
        [
            _nonhomogeneous_compound_poisson_lattice_pmf(row, max_count - 1)
            for row in claim_size_intensities
        ],
    )
    survival, times, survival_grid, state = _inventory_from_increment_pmfs(
        inventory_times=inventory_times,
        retained_counts=retained_counts,
        increment_pmfs=increment_pmfs,
    )
    survival = float(np.clip(survival, 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteNonhomogeneousResult(
        horizon=float(times[-1]),
        claim_size_intensities=claim_size_intensities,
        survival_probability=survival,
        ruin_probability=ruin,
        inventory_times=times,
        retained_counts=retained_counts,
        boundary_values=boundary_values,
        survival_probabilities=survival_grid,
        state_probabilities=state,
        convention=convention,
    )


def finite_time_ruin_discrete_inventory(
    claim_pmf: ArrayLike,
    *,
    inventory_times: ArrayLike,
    retained_counts: ArrayLike,
    claim_arrival_rate: float | None = None,
    arrival_means: ArrayLike | None = None,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteBoundaryResult:
    """Exact finite-time ruin by Rulliere-Loisel inventory recursion.

    ``retained_counts[i]`` is the number of aggregate-claim lattice states
    retained as safe at ``inventory_times[i]``. For example, a value of ``k``
    keeps aggregate amounts ``0, ..., k - 1``.
    """

    pmf = _claim_pmf(claim_pmf)
    times = _inventory_times(inventory_times)
    counts = _retained_counts(retained_counts, times.size)
    rate, means = _arrival_means(
        times,
        claim_arrival_rate=claim_arrival_rate,
        arrival_means=arrival_means,
    )
    survival, times, survival_grid, state = _inventory_from_counts(
        pmf,
        inventory_times=times,
        retained_counts=counts,
        arrival_means=means,
    )
    survival = float(np.clip(survival, 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteBoundaryResult(
        horizon=float(times[-1]),
        claim_arrival_rate=rate,
        arrival_means=means,
        claim_pmf=pmf,
        survival_probability=survival,
        ruin_probability=ruin,
        inventory_times=times,
        retained_counts=counts,
        boundary_values=np.full(times.size, np.nan),
        survival_probabilities=survival_grid,
        state_probabilities=state,
        convention=(
            "inventory recursion; retained_counts[k] keeps aggregate states "
            "0, ..., retained_counts[k] - 1"
        ),
    )


def finite_time_ruin_discrete_boundary(
    claim_pmf: ArrayLike,
    *,
    inventory_times: ArrayLike,
    boundary_values: ArrayLike,
    claim_arrival_rate: float | None = None,
    arrival_means: ArrayLike | None = None,
    convention: BoundaryRuinConvention = "negative",
    boundary_kind: BoundaryKind = "value",
    return_result: bool = False,
) -> float | FiniteTimeDiscreteBoundaryResult:
    """Exact finite-time ruin for an increasing integer-claim boundary.

    ``boundary_values`` are values of the deterministic upper boundary
    ``h(t)`` at the supplied inventory dates. With ``convention="negative"``,
    ruin occurs when the reserve is strictly negative, so aggregate claims
    ``S_t <= h(t)`` survive. With ``convention="nonpositive"``, ruin occurs
    when the reserve is non-positive, so only ``S_t < h(t)`` survives.
    Use ``boundary_kind="crossing"`` when the dates are inverse crossing
    times ``v_n`` of a continuous increasing boundary, as in Picard-Lefevre
    and Rulliere-Loisel inventory formulas.
    """

    pmf = _claim_pmf(claim_pmf)
    times = _inventory_times(inventory_times)
    boundaries = _as_finite_1d(boundary_values, "boundary_values")
    if boundaries.size != times.size:
        raise ValueError("boundary_values must match inventory_times length")
    if np.any(boundaries < 0.0):
        raise ValueError("boundary_values must be non-negative")
    if np.any(np.diff(boundaries) < -1e-12):
        raise ValueError("boundary_values must be non-decreasing")
    selected_convention = _boundary_convention(convention)
    selected_kind = _boundary_kind(boundary_kind)
    counts = _retained_counts_from_boundary_values(
        boundaries,
        convention=selected_convention,
        boundary_kind=selected_kind,
    )
    rate, means = _arrival_means(
        times,
        claim_arrival_rate=claim_arrival_rate,
        arrival_means=arrival_means,
    )
    survival, times, survival_grid, state = _inventory_from_counts(
        pmf,
        inventory_times=times,
        retained_counts=counts,
        arrival_means=means,
    )
    survival = float(np.clip(survival, 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteBoundaryResult(
        horizon=float(times[-1]),
        claim_arrival_rate=rate,
        arrival_means=means,
        claim_pmf=pmf,
        survival_probability=survival,
        ruin_probability=ruin,
        inventory_times=times,
        retained_counts=counts,
        boundary_values=boundaries,
        survival_probabilities=survival_grid,
        state_probabilities=state,
        convention=f"{selected_convention}; boundary_kind={selected_kind}",
    )


def finite_time_ruin_discrete_boundary_function(
    claim_pmf: ArrayLike,
    *,
    boundary: Callable[[float], float],
    horizon: float,
    claim_arrival_rate: float | None = None,
    cumulative_arrival_mean: Callable[[float], float] | None = None,
    convention: BoundaryRuinConvention = "negative",
    root_tol: float = 1e-10,
    max_bisection: int = 80,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteBoundaryResult:
    """Exact finite-time ruin from an increasing boundary function ``h(t)``."""

    selected_convention = _boundary_convention(convention)
    grid = finite_time_discrete_boundary_crossings(
        boundary,
        horizon=horizon,
        root_tol=root_tol,
        max_bisection=max_bisection,
    )
    if (claim_arrival_rate is None) == (cumulative_arrival_mean is None):
        raise ValueError("provide exactly one of claim_arrival_rate or cumulative_arrival_mean")
    if (
        selected_convention == "nonpositive"
        and math.isclose(_call_boundary(boundary, 0.0), 0.0, rel_tol=0.0, abs_tol=root_tol)
    ):
        pmf = _claim_pmf(claim_pmf)
        empty = np.array([], dtype=float)
        if not return_result:
            return 1.0
        return FiniteTimeDiscreteBoundaryResult(
            horizon=grid.horizon,
            claim_arrival_rate=claim_arrival_rate,
            arrival_means=empty,
            claim_pmf=pmf,
            survival_probability=0.0,
            ruin_probability=1.0,
            inventory_times=empty,
            retained_counts=np.array([], dtype=int),
            boundary_values=empty,
            survival_probabilities=empty,
            state_probabilities=empty,
            convention=f"{selected_convention}; boundary_kind=crossing; initial ruin",
        )
    arrival_means = (
        None
        if cumulative_arrival_mean is None
        else _arrival_means_from_cumulative(cumulative_arrival_mean, grid.inventory_times)
    )
    return finite_time_ruin_discrete_boundary(
        claim_pmf,
        inventory_times=grid.inventory_times,
        boundary_values=grid.boundary_values,
        claim_arrival_rate=claim_arrival_rate,
        arrival_means=arrival_means,
        convention=selected_convention,
        boundary_kind="crossing",
        return_result=return_result,
    )


def finite_time_ruin_discrete_nonhomogeneous_inventory(
    claim_size_intensities: ArrayLike,
    *,
    inventory_times: ArrayLike,
    retained_counts: ArrayLike,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteNonhomogeneousResult:
    """Exact finite-time recursion with interval claim-size intensity measures.

    ``claim_size_intensities[i, k]`` is the integrated Poisson intensity of
    claims of size ``k`` over the interval ending at ``inventory_times[i]``.
    Index 0 is ignored because zero-size claims do not affect aggregate claims.
    """

    times = _inventory_times(inventory_times)
    counts = _retained_counts(retained_counts, times.size)
    matrix = _claim_size_intensity_matrix(claim_size_intensities, times.size)
    return _nonhomogeneous_result_from_counts(
        matrix,
        inventory_times=times,
        retained_counts=counts,
        boundary_values=np.full(times.size, np.nan),
        convention=(
            "non-stationary inventory recursion; retained_counts[k] keeps "
            "aggregate states 0, ..., retained_counts[k] - 1"
        ),
        return_result=return_result,
    )


def finite_time_ruin_discrete_nonhomogeneous_boundary(
    claim_size_intensities: ArrayLike,
    *,
    inventory_times: ArrayLike,
    boundary_values: ArrayLike,
    convention: BoundaryRuinConvention = "negative",
    boundary_kind: BoundaryKind = "value",
    return_result: bool = False,
) -> float | FiniteTimeDiscreteNonhomogeneousResult:
    """Exact finite-time ruin for a boundary and non-stationary claim sizes."""

    times = _inventory_times(inventory_times)
    boundaries = _as_finite_1d(boundary_values, "boundary_values")
    if boundaries.size != times.size:
        raise ValueError("boundary_values must match inventory_times length")
    if np.any(boundaries < 0.0):
        raise ValueError("boundary_values must be non-negative")
    if np.any(np.diff(boundaries) < -1e-12):
        raise ValueError("boundary_values must be non-decreasing")
    selected_convention = _boundary_convention(convention)
    selected_kind = _boundary_kind(boundary_kind)
    counts = _retained_counts_from_boundary_values(
        boundaries,
        convention=selected_convention,
        boundary_kind=selected_kind,
    )
    matrix = _claim_size_intensity_matrix(claim_size_intensities, times.size)
    return _nonhomogeneous_result_from_counts(
        matrix,
        inventory_times=times,
        retained_counts=counts,
        boundary_values=boundaries,
        convention=(
            f"{selected_convention}; boundary_kind={selected_kind}; "
            "non-stationary claim-size intensities"
        ),
        return_result=return_result,
    )


def finite_time_ruin_discrete_nonhomogeneous_boundary_function(
    claim_size_intensity_integrals: Callable[[float, float], ArrayLike],
    *,
    boundary: Callable[[float], float],
    horizon: float,
    convention: BoundaryRuinConvention = "negative",
    root_tol: float = 1e-10,
    max_bisection: int = 80,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteNonhomogeneousResult:
    """Exact finite-time ruin from a boundary and interval intensity callback.

    The callback receives ``(start, end)`` and returns integrated intensities
    ``Lambda_k(start, end)`` by claim size ``k`` for that inventory interval.
    """

    if not callable(claim_size_intensity_integrals):
        raise TypeError("claim_size_intensity_integrals must be callable")
    selected_convention = _boundary_convention(convention)
    grid = finite_time_discrete_boundary_crossings(
        boundary,
        horizon=horizon,
        root_tol=root_tol,
        max_bisection=max_bisection,
    )
    if (
        selected_convention == "nonpositive"
        and math.isclose(_call_boundary(boundary, 0.0), 0.0, rel_tol=0.0, abs_tol=root_tol)
    ):
        empty = np.array([], dtype=float)
        if not return_result:
            return 1.0
        return FiniteTimeDiscreteNonhomogeneousResult(
            horizon=grid.horizon,
            claim_size_intensities=np.empty((0, 0), dtype=float),
            survival_probability=0.0,
            ruin_probability=1.0,
            inventory_times=empty,
            retained_counts=np.array([], dtype=int),
            boundary_values=empty,
            survival_probabilities=empty,
            state_probabilities=empty,
            convention=f"{selected_convention}; boundary_kind=crossing; initial ruin",
        )

    matrix = _claim_size_intensity_matrix_from_intervals(
        claim_size_intensity_integrals,
        grid.inventory_times,
    )
    return finite_time_ruin_discrete_nonhomogeneous_boundary(
        matrix,
        inventory_times=grid.inventory_times,
        boundary_values=grid.boundary_values,
        convention=selected_convention,
        boundary_kind="crossing",
        return_result=return_result,
    )


def _crossing_time_map(grid: FiniteTimeDiscreteBoundaryGrid) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for time, value in zip(grid.inventory_times, grid.boundary_values, strict=True):
        nearest = round(value)
        if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-10):
            mapping[int(nearest)] = float(time)
    return mapping


def finite_time_discrete_appell_coefficients(
    claim_pmf: ArrayLike,
    *,
    claim_arrival_rate: float,
    boundary: Callable[[float], float],
    horizon: float,
    root_tol: float = 1e-10,
    max_bisection: int = 80,
) -> np.ndarray:
    """Return generalized-Appell coefficients for a boundary up to ``horizon``."""

    pmf = _claim_pmf(claim_pmf)
    rate = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    grid = finite_time_discrete_boundary_crossings(
        boundary,
        horizon=horizon,
        root_tol=root_tol,
        max_bisection=max_bisection,
    )
    max_degree = _retained_count_from_crossing(_call_boundary(boundary, grid.horizon)) - 1
    coefficients = np.zeros(max_degree + 1, dtype=float)
    coefficients[0] = 1.0
    initial_boundary = _call_boundary(boundary, 0.0)
    first_constrained = _floor_nonnegative(initial_boundary) + 1
    crossing_times = _crossing_time_map(grid)

    for degree in range(1, max_degree + 1):
        if degree < first_constrained:
            continue
        crossing_time = crossing_times.get(degree)
        if crossing_time is None:
            continue
        base = compound_poisson_appell_base(
            pmf,
            claim_arrival_rate=rate,
            time=crossing_time,
            max_degree=degree,
        )
        coefficients[degree] = -float(np.dot(coefficients[:degree], base[degree:0:-1]))
    return coefficients


def finite_time_ruin_discrete_appell(
    claim_pmf: ArrayLike,
    *,
    boundary: Callable[[float], float],
    horizon: float,
    claim_arrival_rate: float,
    convention: BoundaryRuinConvention = "negative",
    root_tol: float = 1e-10,
    max_bisection: int = 80,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteAppellResult:
    """Exact finite-time ruin via Picard-Lefevre generalized-Appell polynomials."""

    pmf = _claim_pmf(claim_pmf)
    rate = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    selected_convention = _boundary_convention(convention)
    grid = finite_time_discrete_boundary_crossings(
        boundary,
        horizon=horizon,
        root_tol=root_tol,
        max_bisection=max_bisection,
    )
    if (
        selected_convention == "nonpositive"
        and math.isclose(_call_boundary(boundary, 0.0), 0.0, rel_tol=0.0, abs_tol=root_tol)
    ):
        empty = np.array([], dtype=float)
        if not return_result:
            return 1.0
        _, effective_rate = _positive_claim_process(pmf, rate)
        return FiniteTimeDiscreteAppellResult(
            horizon=grid.horizon,
            claim_arrival_rate=rate,
            effective_claim_arrival_rate=effective_rate,
            claim_pmf=pmf,
            survival_probability=0.0,
            ruin_probability=1.0,
            boundary_grid=grid,
            appell_coefficients=empty,
            state_probabilities=empty,
            convention=f"{selected_convention}; initial ruin",
        )

    max_degree = _retained_count_from_crossing(_call_boundary(boundary, grid.horizon)) - 1
    coefficients = finite_time_discrete_appell_coefficients(
        pmf,
        claim_arrival_rate=rate,
        boundary=boundary,
        horizon=grid.horizon,
        root_tol=root_tol,
        max_bisection=max_bisection,
    )
    base = compound_poisson_appell_base(
        pmf,
        claim_arrival_rate=rate,
        time=grid.horizon,
        max_degree=max_degree,
    )
    _, effective_rate = _positive_claim_process(pmf, rate)
    state_polynomials = np.convolve(coefficients, base)[: max_degree + 1]
    state = math.exp(-effective_rate * grid.horizon) * state_polynomials
    survival = float(np.clip(np.sum(state), 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteAppellResult(
        horizon=grid.horizon,
        claim_arrival_rate=rate,
        effective_claim_arrival_rate=effective_rate,
        claim_pmf=pmf,
        survival_probability=survival,
        ruin_probability=ruin,
        boundary_grid=grid,
        appell_coefficients=coefficients,
        state_probabilities=state,
        convention=f"{selected_convention}; generalized Appell",
    )


def finite_time_ruin_discrete(
    claim_pmf: ArrayLike,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    horizon: float,
    method: FiniteTimeDiscreteMethod = "seal",
    return_result: bool = False,
) -> float | FiniteTimeDiscreteRuinResult:
    """Exact finite-time ruin probability for integer-valued claim sizes.

    The model is ``R_t = u + c t - S_t`` with a homogeneous compound Poisson
    aggregate claim process and integer claim sizes.
    """

    pmf = _claim_pmf(claim_pmf)
    u = _finite_nonnegative(initial_capital, "initial_capital")
    c = _finite_positive(premium_rate, "premium_rate")
    lam = _finite_nonnegative(claim_arrival_rate, "claim_arrival_rate")
    time = _finite_nonnegative(horizon, "horizon")
    selected = _method(method)

    inventory_times = np.array([], dtype=float)
    survival_grid = np.array([], dtype=float)
    state = np.array([], dtype=float)

    if time == 0.0 or lam == 0.0 or float(np.sum(pmf[1:])) == 0.0:
        survival = 1.0
    elif selected == "inventory":
        survival, inventory_times, survival_grid, state = _inventory_result(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    elif selected == "takacs":
        if not math.isclose(u, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("method='takacs' is the zero-initial-capital Takacs formula")
        survival = _takacs_survival(
            pmf,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    elif selected == "picard-lefevre":
        survival = _picard_lefevre_survival(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )
    else:
        survival = _seal_survival(
            pmf,
            initial_capital=u,
            premium_rate=c,
            claim_arrival_rate=lam,
            horizon=time,
        )

    survival = float(np.clip(survival, 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteRuinResult(
        initial_capital=u,
        horizon=time,
        premium_rate=c,
        claim_arrival_rate=lam,
        claim_pmf=pmf,
        method=selected,
        survival_probability=survival,
        ruin_probability=ruin,
        premium_units=c * time,
        inventory_times=inventory_times,
        survival_probabilities=survival_grid,
        state_probabilities=state,
        convention=(
            "integer claim amounts; ruin occurs when S_t > u + c t; "
            "inventory states keep aggregate claims strictly below the active integer boundary"
        ),
    )
