"""Markov-modulated multirisk models with common shocks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


Vector = tuple[int, ...]
VectorPmf = dict[Vector, float]
SolvencyRegion = Callable[[np.ndarray, np.ndarray, int], bool]
SolvencyRegionName = Literal["any_line", "total", "hybrid"]


@dataclass(frozen=True)
class MarkovEnvironment:
    """Finite Markov environment observed at inventory dates."""

    initial_distribution: np.ndarray
    transition_matrix: np.ndarray

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_distribution, dtype=float)
        transition = np.asarray(self.transition_matrix, dtype=float)
        if initial.ndim != 1 or initial.size == 0:
            raise ValueError("initial_distribution must be a non-empty vector")
        if transition.shape != (initial.size, initial.size):
            raise ValueError("transition_matrix must be square and match initial_distribution")
        if np.any(~np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial_distribution must contain finite non-negative values")
        if np.any(~np.isfinite(transition)) or np.any(transition < 0.0):
            raise ValueError("transition_matrix must contain finite non-negative values")
        if not np.isclose(np.sum(initial), 1.0):
            raise ValueError("initial_distribution must sum to one")
        if not np.allclose(np.sum(transition, axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to one")
        object.__setattr__(self, "initial_distribution", initial)
        object.__setattr__(self, "transition_matrix", transition)

    @property
    def n_states(self) -> int:
        return int(self.initial_distribution.size)


@dataclass(frozen=True)
class CommonShock:
    """One common-shock type with state-dependent intensity and claim PMF."""

    intensities: np.ndarray
    claim_pmfs: Mapping[Sequence[int], float] | Sequence[Mapping[Sequence[int], float]]
    name: str = "shock"

    def __post_init__(self) -> None:
        intensities = np.asarray(self.intensities, dtype=float)
        if intensities.ndim != 1 or intensities.size == 0:
            raise ValueError("intensities must be a non-empty vector")
        if np.any(~np.isfinite(intensities)) or np.any(intensities < 0.0):
            raise ValueError("intensities must contain finite non-negative values")
        pmf_sources = (
            (self.claim_pmfs,) if isinstance(self.claim_pmfs, Mapping) else self.claim_pmfs
        )
        pmfs = tuple(_clean_vector_pmf(pmf, name="claim_pmf") for pmf in pmf_sources)
        if len(pmfs) == 1 and intensities.size > 1:
            pmfs = pmfs * int(intensities.size)
        if len(pmfs) != intensities.size:
            raise ValueError("claim_pmfs must contain one PMF or one PMF per state")
        n_lines = {len(next(iter(pmf))) for pmf in pmfs if pmf}
        if len(n_lines) != 1:
            raise ValueError("all claim PMFs must have the same vector dimension")
        object.__setattr__(self, "intensities", intensities)
        object.__setattr__(self, "claim_pmfs", pmfs)

    @property
    def n_states(self) -> int:
        return int(self.intensities.size)

    @property
    def n_lines(self) -> int:
        return len(next(iter(self.claim_pmfs[0])))


@dataclass(frozen=True)
class CommonShockIncrementResult:
    """State-dependent lattice increments built from common shocks."""

    increment_pmfs: tuple[VectorPmf, ...]
    truncation_error_bounds: np.ndarray


@dataclass(frozen=True)
class MarkovModulatedRuinResult:
    """Finite-horizon multirisk ruin probabilities with environment states."""

    ruin_probabilities: np.ndarray
    survival_probabilities: np.ndarray
    period_ruin_probabilities: np.ndarray
    survival_by_state: np.ndarray
    terminal_distribution: dict[tuple[Vector, int], float]
    initial_capitals: np.ndarray
    premiums: np.ndarray
    horizon: int
    region: str
    truncation_error_bound: float = 0.0


@dataclass(frozen=True)
class DependenceImpactResult:
    """Difference between two multirisk ruin results under different dependence."""

    reference_label: str
    comparison_label: str
    periods: np.ndarray
    reference_ruin: np.ndarray
    comparison_ruin: np.ndarray
    difference: np.ndarray

    @property
    def final_difference(self) -> float:
        return float(self.difference[-1])


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
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
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _clean_vector(vector: Sequence[int], name: str) -> Vector:
    cleaned = tuple(int(value) for value in vector)
    if len(cleaned) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0 for value in cleaned):
        raise ValueError(f"{name} must contain non-negative integer coordinates")
    if any(float(raw) != value for raw, value in zip(vector, cleaned)):
        raise ValueError(f"{name} must contain integer coordinates")
    return cleaned


def _clean_vector_pmf(pmf: Mapping[Sequence[int], float], *, name: str) -> VectorPmf:
    if not pmf:
        raise ValueError(f"{name} must not be empty")
    cleaned: VectorPmf = {}
    dimension: int | None = None
    for raw_vector, raw_probability in pmf.items():
        vector = _clean_vector(raw_vector, name)
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(f"{name} vectors must have a common dimension")
        probability = float(raw_probability)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{name} probabilities must be finite and non-negative")
        if probability:
            cleaned[vector] = cleaned.get(vector, 0.0) + probability
    total = math.fsum(cleaned.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"{name} probabilities must sum to one")
    return {vector: probability / total for vector, probability in cleaned.items()}


def transition_matrix_from_generator(generator: ArrayLike, step: float = 1.0) -> np.ndarray:
    """Discretize a continuous-time Markov generator over one inventory step."""

    matrix = np.asarray(generator, dtype=float)
    dt = _positive_float(step, "step")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("generator must be a non-empty square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("generator must contain finite values")
    if np.any(matrix - np.diag(np.diag(matrix)) < 0.0):
        raise ValueError("generator off-diagonal entries must be non-negative")
    if not np.allclose(np.sum(matrix, axis=1), 0.0):
        raise ValueError("generator rows must sum to zero")
    transition = linalg.expm(matrix * dt)
    transition = np.clip(transition, 0.0, 1.0)
    return transition / np.sum(transition, axis=1, keepdims=True)


def independent_common_shock_pmf(
    activation_probabilities: ArrayLike,
    severity_pmfs: Sequence[Mapping[int, float]],
) -> VectorPmf:
    """Build a joint claim PMF from independent activations and lattice severities."""

    probabilities = _as_1d_nonnegative(activation_probabilities, "activation_probabilities")
    if np.any(probabilities > 1.0):
        raise ValueError("activation_probabilities must lie in [0, 1]")
    if len(severity_pmfs) != probabilities.size:
        raise ValueError("severity_pmfs must match activation_probabilities")

    pmf: VectorPmf = {tuple([0] * probabilities.size): 1.0}
    for branch, (activation, raw_severity) in enumerate(zip(probabilities, severity_pmfs)):
        severity = _clean_univariate_pmf(raw_severity)
        branch_pmf: VectorPmf = {tuple([0] * probabilities.size): 1.0 - float(activation)}
        for amount, probability in severity.items():
            vector = [0] * probabilities.size
            vector[branch] = amount
            branch_pmf[tuple(vector)] = branch_pmf.get(tuple(vector), 0.0) + float(
                activation,
            ) * probability
        pmf = convolve_vector_pmfs(pmf, branch_pmf)
    return pmf


def _clean_univariate_pmf(pmf: Mapping[int, float]) -> dict[int, float]:
    if not pmf:
        raise ValueError("severity_pmf must not be empty")
    cleaned: dict[int, float] = {}
    for raw_amount, raw_probability in pmf.items():
        amount = int(raw_amount)
        if amount != raw_amount or amount < 0:
            raise ValueError("severity_pmf keys must be non-negative integers")
        probability = float(raw_probability)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError("severity_pmf probabilities must be finite and non-negative")
        if probability:
            cleaned[amount] = cleaned.get(amount, 0.0) + probability
    total = math.fsum(cleaned.values())
    if not np.isclose(total, 1.0):
        raise ValueError("severity_pmf probabilities must sum to one")
    return {amount: probability / total for amount, probability in cleaned.items()}


def convolve_vector_pmfs(
    first: Mapping[Vector, float],
    second: Mapping[Vector, float],
) -> VectorPmf:
    """Convolve two finite multivariate lattice PMFs."""

    left = _clean_vector_pmf(first, name="first")
    right = _clean_vector_pmf(second, name="second")
    dimension = len(next(iter(left)))
    if len(next(iter(right))) != dimension:
        raise ValueError("PMF dimensions must match")
    return _convolve_clean_vector_pmfs(left, right, normalize=True)


def _convolve_clean_vector_pmfs(
    first: Mapping[Vector, float],
    second: Mapping[Vector, float],
    *,
    normalize: bool,
) -> VectorPmf:
    result: VectorPmf = {}
    for vector_a, probability_a in first.items():
        for vector_b, probability_b in second.items():
            vector = tuple(a + b for a, b in zip(vector_a, vector_b))
            result[vector] = result.get(vector, 0.0) + probability_a * probability_b
    return _normalize_pmf(result) if normalize else result


def _normalize_pmf(pmf: Mapping[Vector, float]) -> VectorPmf:
    total = math.fsum(float(value) for value in pmf.values())
    if total <= 0.0:
        raise ValueError("PMF mass must be positive")
    return {vector: float(value) / total for vector, value in pmf.items() if value > 0.0}


def compound_poisson_vector_pmf(
    event_pmf: Mapping[Sequence[int], float],
    mean: float,
    *,
    max_count: int = 32,
    tail_tolerance: float = 1e-12,
) -> tuple[VectorPmf, float]:
    """Compound-Poisson lattice PMF truncated by the number of events."""

    event = _clean_vector_pmf(event_pmf, name="event_pmf")
    intensity = _nonnegative_float(mean, "mean")
    maximum = _positive_int(max_count, "max_count")
    tolerance = _positive_float(tail_tolerance, "tail_tolerance")
    zero = tuple([0] * len(next(iter(event))))
    if intensity == 0.0:
        return {zero: 1.0}, 0.0

    result: VectorPmf = {zero: math.exp(-intensity)}
    power: VectorPmf = {zero: 1.0}
    weight = math.exp(-intensity)
    cumulative_weight = weight
    for count in range(1, maximum + 1):
        power = _convolve_clean_vector_pmfs(power, event, normalize=False)
        weight *= intensity / count
        cumulative_weight += weight
        for vector, probability in power.items():
            result[vector] = result.get(vector, 0.0) + weight * probability
        if 1.0 - cumulative_weight <= tolerance:
            break
    return _normalize_pmf(result), max(0.0, 1.0 - cumulative_weight)


def common_shock_increment_pmfs(
    shocks: Sequence[CommonShock],
    *,
    period_length: float = 1.0,
    max_count: int = 32,
    tail_tolerance: float = 1e-12,
) -> CommonShockIncrementResult:
    """Build one-period aggregate-claim PMFs for each Markov environment state."""

    shock_tuple = tuple(shocks)
    if not shock_tuple:
        raise ValueError("shocks must contain at least one CommonShock")
    n_states = shock_tuple[0].n_states
    n_lines = shock_tuple[0].n_lines
    if any(shock.n_states != n_states for shock in shock_tuple):
        raise ValueError("all shocks must have the same number of states")
    if any(shock.n_lines != n_lines for shock in shock_tuple):
        raise ValueError("all shocks must have the same number of lines")
    dt = _positive_float(period_length, "period_length")
    bounds = np.zeros(n_states, dtype=float)
    increment_pmfs: list[VectorPmf] = []
    zero = tuple([0] * n_lines)
    for state in range(n_states):
        pmf: VectorPmf = {zero: 1.0}
        for shock in shock_tuple:
            compound, tail = compound_poisson_vector_pmf(
                shock.claim_pmfs[state],
                shock.intensities[state] * dt,
                max_count=max_count,
                tail_tolerance=tail_tolerance,
            )
            pmf = _convolve_clean_vector_pmfs(pmf, compound, normalize=False)
            bounds[state] += tail
        increment_pmfs.append(_normalize_pmf(pmf))
    return CommonShockIncrementResult(tuple(increment_pmfs), bounds)


def solvency_region(
    kind: SolvencyRegionName = "any_line",
    *,
    severity_limit: float | ArrayLike = 0.0,
) -> SolvencyRegion:
    """Return a standard solvency-region predicate."""

    if kind not in {"any_line", "total", "hybrid"}:
        raise ValueError("kind must be 'any_line', 'total' or 'hybrid'")
    raw_limit = np.asarray(severity_limit, dtype=float)
    if np.any(~np.isfinite(raw_limit)) or np.any(raw_limit < 0.0):
        raise ValueError("severity_limit must be finite and non-negative")

    def predicate(claims: np.ndarray, boundary: np.ndarray, period: int) -> bool:
        del period
        if kind == "any_line":
            return bool(np.all(claims <= boundary))
        if kind == "total":
            return bool(np.sum(claims) <= np.sum(boundary))
        limit = raw_limit if raw_limit.ndim else np.full(boundary.size, float(raw_limit))
        if limit.shape != boundary.shape:
            raise ValueError("severity_limit must be scalar or match boundary dimension")
        return bool(np.sum(claims) <= np.sum(boundary) and np.all(claims <= boundary + limit))

    return predicate


def finite_time_markov_modulated_ruin(
    increment_pmfs: Sequence[Mapping[Sequence[int], float]],
    environment: MarkovEnvironment,
    *,
    initial_capitals: ArrayLike,
    premiums: ArrayLike,
    horizon: int,
    region: SolvencyRegionName | SolvencyRegion = "any_line",
    severity_limit: float | ArrayLike = 0.0,
    truncation_error_bounds: ArrayLike | None = None,
) -> MarkovModulatedRuinResult:
    """Recursive finite-time ruin probability for a Markov-modulated multirisk model."""

    pmfs = tuple(_clean_vector_pmf(pmf, name="increment_pmf") for pmf in increment_pmfs)
    if len(pmfs) != environment.n_states:
        raise ValueError("increment_pmfs must contain one PMF per environment state")
    n_lines = len(next(iter(pmfs[0])))
    if any(len(next(iter(pmf))) != n_lines for pmf in pmfs):
        raise ValueError("all increment PMFs must have the same vector dimension")
    initial = _as_1d_nonnegative(initial_capitals, "initial_capitals")
    premium = _as_1d_nonnegative(premiums, "premiums")
    if initial.size != n_lines or premium.size != n_lines:
        raise ValueError("initial_capitals and premiums must match claim-vector dimension")
    periods = _positive_int(horizon, "horizon")
    predicate = solvency_region(region, severity_limit=severity_limit) if isinstance(
        region,
        str,
    ) else region
    if not callable(predicate):
        raise TypeError("region must be a known region name or a callable")
    if truncation_error_bounds is None:
        truncation = np.zeros(environment.n_states, dtype=float)
    else:
        truncation = _as_1d_nonnegative(truncation_error_bounds, "truncation_error_bounds")
        if truncation.size != environment.n_states:
            raise ValueError("truncation_error_bounds must match number of states")

    distribution: dict[tuple[Vector, int], float] = {
        (tuple([0] * n_lines), state): float(probability)
        for state, probability in enumerate(environment.initial_distribution)
        if probability > 0.0
    }
    survival = np.empty(periods + 1, dtype=float)
    ruin = np.empty(periods + 1, dtype=float)
    period_ruin = np.zeros(periods + 1, dtype=float)
    by_state = np.zeros((periods + 1, environment.n_states), dtype=float)
    survival[0] = math.fsum(distribution.values())
    ruin[0] = 1.0 - survival[0]
    for (_, state), probability in distribution.items():
        by_state[0, state] += probability

    truncation_bound = 0.0
    for period in range(1, periods + 1):
        boundary = initial + period * premium
        next_distribution: dict[tuple[Vector, int], float] = {}
        current_ruin = 0.0
        for (claims, state), mass in distribution.items():
            truncation_bound += mass * truncation[state]
            for increment, increment_probability in pmfs[state].items():
                updated_claims = tuple(a + b for a, b in zip(claims, increment))
                branch_claims = np.asarray(updated_claims, dtype=float)
                probability = mass * increment_probability
                if predicate(branch_claims, boundary, period):
                    for next_state, transition in enumerate(environment.transition_matrix[state]):
                        if transition:
                            key = (updated_claims, next_state)
                            next_distribution[key] = (
                                next_distribution.get(key, 0.0) + probability * transition
                            )
                else:
                    current_ruin += probability
        distribution = next_distribution
        survival[period] = math.fsum(distribution.values())
        by_state[period] = 0.0
        for (_, state), probability in distribution.items():
            by_state[period, state] += probability
        ruin[period] = 1.0 - survival[period]
        period_ruin[period] = current_ruin

    return MarkovModulatedRuinResult(
        ruin_probabilities=np.clip(ruin, 0.0, 1.0),
        survival_probabilities=np.clip(survival, 0.0, 1.0),
        period_ruin_probabilities=np.clip(period_ruin, 0.0, 1.0),
        survival_by_state=by_state,
        terminal_distribution=distribution,
        initial_capitals=initial,
        premiums=premium,
        horizon=periods,
        region=region if isinstance(region, str) else "custom",
        truncation_error_bound=float(truncation_bound),
    )


def dependence_impact(
    reference: MarkovModulatedRuinResult,
    comparison: MarkovModulatedRuinResult,
    *,
    reference_label: str = "reference",
    comparison_label: str = "comparison",
) -> DependenceImpactResult:
    """Compare two finite-time ruin curves under different dependence assumptions."""

    if not isinstance(reference, MarkovModulatedRuinResult):
        raise TypeError("reference must be a MarkovModulatedRuinResult")
    if not isinstance(comparison, MarkovModulatedRuinResult):
        raise TypeError("comparison must be a MarkovModulatedRuinResult")
    if reference.ruin_probabilities.shape != comparison.ruin_probabilities.shape:
        raise ValueError("ruin probability curves must have matching shapes")
    periods = np.arange(reference.ruin_probabilities.size)
    return DependenceImpactResult(
        reference_label=reference_label,
        comparison_label=comparison_label,
        periods=periods,
        reference_ruin=reference.ruin_probabilities,
        comparison_ruin=comparison.ruin_probabilities,
        difference=comparison.ruin_probabilities - reference.ruin_probabilities,
    )
