"""Finite-horizon discrete-time ruin recursions and diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad


@dataclass(frozen=True)
class FiniteTimeDiscreteTimeRuinResult:
    """Exact finite-horizon result for period aggregate-claim distributions."""

    initial_capital: float
    grid_step: float
    premiums: np.ndarray
    cumulative_premiums: np.ndarray
    boundaries: np.ndarray
    increment_pmfs: np.ndarray
    survival_probability: float
    ruin_probability: float
    survival_probabilities: np.ndarray
    ruin_probabilities: np.ndarray
    ruin_time_probabilities: np.ndarray
    state_probabilities: np.ndarray
    surplus_distributions: tuple[tuple[np.ndarray, np.ndarray], ...]
    deficit_distributions: tuple[tuple[np.ndarray, np.ndarray], ...]
    convention: str


@dataclass(frozen=True)
class FiniteTimeDiscreteTimeBoundsResult:
    """Lower and upper finite-horizon ruin bounds from stochastic discretizations."""

    lower: FiniteTimeDiscreteTimeRuinResult
    upper: FiniteTimeDiscreteTimeRuinResult

    @property
    def ruin_probability_interval(self) -> tuple[float, float]:
        """Final-horizon lower and upper ruin probabilities."""

        return (self.lower.ruin_probability, self.upper.ruin_probability)


@dataclass(frozen=True)
class FiniteTimeDependentRuinResult:
    """Exact finite-horizon result from a joint scenario law for period claims."""

    initial_capital: float
    premiums: np.ndarray
    cumulative_premiums: np.ndarray
    boundaries: np.ndarray
    claim_scenarios: np.ndarray
    scenario_probabilities: np.ndarray
    survival_probability: float
    ruin_probability: float
    survival_probabilities: np.ndarray
    ruin_probabilities: np.ndarray
    ruin_time_probabilities: np.ndarray
    surplus_distributions: tuple[tuple[np.ndarray, np.ndarray], ...]
    deficit_distributions: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class FiniteTimeLundbergBoundResult:
    """Periodwise non-homogeneous Lundberg roots and finite-horizon bounds."""

    initial_capital: float
    period_roots: np.ndarray
    adjustment_coefficients: np.ndarray
    bounds: np.ndarray


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _as_nonnegative_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = _as_1d_float(values, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _positive_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _pmf_matrix(values: ArrayLike, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    totals = matrix.sum(axis=1)
    if np.any(totals > 1.0 + 1e-10):
        raise ValueError(f"{name} rows must sum to at most one")
    return matrix.copy()


def _premium_vector(values: ArrayLike, expected_size: int) -> np.ndarray:
    premiums = _as_nonnegative_1d(values, "premiums")
    if premiums.size != expected_size:
        raise ValueError("premiums must have one value per period")
    return premiums


def _probabilities(values: ArrayLike, expected_size: int) -> np.ndarray:
    probabilities = _as_nonnegative_1d(values, "scenario_probabilities")
    if probabilities.size != expected_size:
        raise ValueError("scenario_probabilities must match the number of scenarios")
    total = float(probabilities.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("scenario_probabilities must sum to one")
    return probabilities / total


def _weighted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    unique, inverse = np.unique(values, return_inverse=True)
    probabilities = np.zeros(unique.size, dtype=float)
    np.add.at(probabilities, inverse, weights)
    total = float(probabilities.sum())
    if total > 0.0:
        probabilities /= total
    return unique, probabilities


def _cdf_from_distribution(
    values: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(thresholds.size, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(probabilities[order])
    indices = np.searchsorted(sorted_values, thresholds, side="right") - 1
    result = np.zeros(thresholds.size, dtype=float)
    valid = indices >= 0
    result[valid] = cumulative[indices[valid]]
    return np.clip(result, 0.0, 1.0)


def discount_factors_from_interest(interest_rates: ArrayLike) -> np.ndarray:
    """Return accumulation factors ``a(t)=prod_{j<=t}(1+i_j)`` with ``a(0)=1``."""

    rates = _as_1d_float(interest_rates, "interest_rates")
    if np.any(rates <= -1.0):
        raise ValueError("interest_rates must be greater than -1")
    return np.concatenate(([1.0], np.cumprod(1.0 + rates)))


def discounted_premiums(
    premiums: ArrayLike,
    interest_rates: ArrayLike,
    *,
    timing: str = "beginning",
) -> np.ndarray:
    """Discount period premiums at time 0 under Castaner-style timing conventions."""

    nominal = _as_nonnegative_1d(premiums, "premiums")
    rates = _as_1d_float(interest_rates, "interest_rates")
    if rates.size != nominal.size:
        raise ValueError("interest_rates must have one value per premium period")
    factors = discount_factors_from_interest(rates)
    selected = timing.lower().replace("_", "-").replace(" ", "-")
    if selected in {"beginning", "start"}:
        denominators = factors[:-1]
    elif selected in {"end", "ending"}:
        denominators = factors[1:]
    elif selected in {"middle", "mid", "mid-period"}:
        denominators = factors[:-1] * np.sqrt(1.0 + rates)
    else:
        raise ValueError("timing must be 'beginning', 'middle' or 'end'")
    return nominal / denominators


def claim_size_intensities_from_functions(
    arrival_rate: float | Callable[[float], float],
    severity_pmf: ArrayLike | Callable[[float], ArrayLike],
    inventory_times: ArrayLike,
    *,
    max_claim_size: int,
    epsabs: float = 1e-10,
    epsrel: float = 1e-8,
) -> np.ndarray:
    """Integrate ``lambda(t) p_k(t)`` over inventory intervals for sizes ``k``."""

    times = _as_nonnegative_1d(inventory_times, "inventory_times")
    if np.any(np.diff(times) < -1e-12):
        raise ValueError("inventory_times must be non-decreasing")
    max_size = int(max_claim_size)
    if max_size < 0:
        raise ValueError("max_claim_size must be non-negative")
    absolute = _positive_float(epsabs, "epsabs")
    relative = _positive_float(epsrel, "epsrel")

    def rate(time: float) -> float:
        value = arrival_rate(time) if callable(arrival_rate) else arrival_rate
        result = _nonnegative_float(value, "arrival_rate")
        return result

    if callable(severity_pmf):

        def probability(time: float, index: int) -> float:
            pmf = np.asarray(severity_pmf(time), dtype=float)
            if pmf.ndim != 1 or pmf.size <= index:
                return 0.0
            if np.any(~np.isfinite(pmf)) or np.any(pmf < 0.0):
                raise ValueError("severity_pmf must return finite non-negative probabilities")
            total = float(pmf.sum())
            if total > 1.0 + 1e-10:
                raise ValueError("severity_pmf must return probabilities summing to at most one")
            return float(pmf[index])

    else:
        pmf = _as_nonnegative_1d(severity_pmf, "severity_pmf")
        if float(pmf.sum()) > 1.0 + 1e-10:
            raise ValueError("severity_pmf must sum to at most one")

        def probability(time: float, index: int) -> float:
            del time
            return float(pmf[index]) if index < pmf.size else 0.0

    matrix = np.zeros((times.size, max_size + 1), dtype=float)
    previous = 0.0
    for row, current in enumerate(times):
        start = previous
        end = float(current)
        if math.isclose(start, end, rel_tol=0.0, abs_tol=1e-14):
            previous = end
            continue
        for size in range(max_size + 1):
            value, _ = quad(
                lambda x, k=size: rate(x) * probability(x, k),
                start,
                end,
                epsabs=absolute,
                epsrel=relative,
            )
            matrix[row, size] = max(0.0, float(value))
        previous = end
    return matrix


def finite_time_discrete_time_ruin(
    increment_pmfs: ArrayLike,
    *,
    premiums: ArrayLike,
    initial_capital: float = 0.0,
    grid_step: float = 1.0,
    return_result: bool = False,
) -> float | FiniteTimeDiscreteTimeRuinResult:
    """Exact finite-horizon ruin for independent non-stationary period increments."""

    increments = _pmf_matrix(increment_pmfs, "increment_pmfs")
    step = _positive_float(grid_step, "grid_step")
    initial = _nonnegative_float(initial_capital, "initial_capital")
    premium_vector = _premium_vector(premiums, increments.shape[0])
    cumulative = np.cumsum(premium_vector)
    boundaries = initial + cumulative
    retained_counts = np.floor(boundaries / step + 1e-12).astype(int) + 1
    max_count = max(int(retained_counts.max()) if retained_counts.size else 0, 1)
    rows = increments

    state = np.zeros(max_count, dtype=float)
    state[0] = 1.0
    survival_grid = np.zeros(increments.shape[0], dtype=float)
    ruin_times = np.zeros(increments.shape[0], dtype=float)
    surplus_distributions: list[tuple[np.ndarray, np.ndarray]] = []
    deficit_distributions: list[tuple[np.ndarray, np.ndarray]] = []

    for period, (boundary, retained, increment) in enumerate(
        zip(boundaries, retained_counts, rows, strict=True),
    ):
        convolved = np.convolve(state, increment)
        safe = convolved[: min(retained, convolved.size)]
        next_state = np.zeros_like(state)
        next_state[: safe.size] = safe
        unsafe = convolved[retained:]
        ruin_mass = float(unsafe.sum())
        missing_tail = max(0.0, float(state.sum()) * (1.0 - float(increment.sum())))
        ruin_times[period] = ruin_mass + missing_tail

        safe_indices = np.arange(safe.size, dtype=float)
        surplus_values = boundary - safe_indices * step
        surplus_distributions.append(_weighted_distribution(surplus_values, safe))

        unsafe_indices = np.arange(retained, retained + unsafe.size, dtype=float)
        deficit_values = unsafe_indices * step - boundary
        deficit_distributions.append(_weighted_distribution(deficit_values, unsafe))

        state = next_state
        survival_grid[period] = float(state.sum())

    survival = float(np.clip(survival_grid[-1], 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDiscreteTimeRuinResult(
        initial_capital=initial,
        grid_step=step,
        premiums=premium_vector,
        cumulative_premiums=cumulative,
        boundaries=boundaries,
        increment_pmfs=increments,
        survival_probability=survival,
        ruin_probability=ruin,
        survival_probabilities=survival_grid,
        ruin_probabilities=1.0 - survival_grid,
        ruin_time_probabilities=ruin_times,
        state_probabilities=state.copy(),
        surplus_distributions=tuple(surplus_distributions),
        deficit_distributions=tuple(deficit_distributions),
        convention="negative reserve; period aggregate claims on an arithmetic lattice",
    )


def finite_time_discrete_time_bounds(
    lower_increment_pmfs: ArrayLike,
    upper_increment_pmfs: ArrayLike,
    *,
    premiums: ArrayLike,
    initial_capital: float = 0.0,
    grid_step: float = 1.0,
) -> FiniteTimeDiscreteTimeBoundsResult:
    """Return lower and upper ruin bounds from lower/upper claim discretizations."""

    lower = finite_time_discrete_time_ruin(
        lower_increment_pmfs,
        premiums=premiums,
        initial_capital=initial_capital,
        grid_step=grid_step,
        return_result=True,
    )
    upper = finite_time_discrete_time_ruin(
        upper_increment_pmfs,
        premiums=premiums,
        initial_capital=initial_capital,
        grid_step=grid_step,
        return_result=True,
    )
    if lower.ruin_probability > upper.ruin_probability + 1e-10:
        raise ValueError(
            "lower_increment_pmfs produced a larger ruin probability than upper_increment_pmfs",
        )
    return FiniteTimeDiscreteTimeBoundsResult(lower=lower, upper=upper)


def finite_time_dependent_discrete_time_ruin(
    claim_scenarios: ArrayLike,
    scenario_probabilities: ArrayLike,
    *,
    premiums: ArrayLike,
    initial_capital: float = 0.0,
    return_result: bool = False,
) -> float | FiniteTimeDependentRuinResult:
    """Exact finite-time ruin from an arbitrary joint law of period claim totals."""

    scenarios = np.asarray(claim_scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[0] == 0 or scenarios.shape[1] == 0:
        raise ValueError("claim_scenarios must be a non-empty two-dimensional array")
    if np.any(~np.isfinite(scenarios)) or np.any(scenarios < 0.0):
        raise ValueError("claim_scenarios must contain finite non-negative claims")
    probabilities = _probabilities(scenario_probabilities, scenarios.shape[0])
    premium_vector = _premium_vector(premiums, scenarios.shape[1])
    initial = _nonnegative_float(initial_capital, "initial_capital")
    cumulative = np.cumsum(premium_vector)
    boundaries = initial + cumulative
    partial_sums = np.cumsum(scenarios, axis=1)
    ruined_by = partial_sums > boundaries[None, :]
    first_ruin = np.full(scenarios.shape[0], -1, dtype=int)
    for index, row in enumerate(ruined_by):
        hits = np.flatnonzero(row)
        if hits.size:
            first_ruin[index] = int(hits[0])

    survival_grid = np.asarray(
        [float(probabilities[~ruined_by[:, period]].sum()) for period in range(scenarios.shape[1])],
        dtype=float,
    )
    ruin_times = np.asarray(
        [float(probabilities[first_ruin == period].sum()) for period in range(scenarios.shape[1])],
        dtype=float,
    )
    surplus_distributions: list[tuple[np.ndarray, np.ndarray]] = []
    deficit_distributions: list[tuple[np.ndarray, np.ndarray]] = []
    for period, boundary in enumerate(boundaries):
        survived = ~ruined_by[:, period]
        surplus_distributions.append(
            _weighted_distribution(
                boundary - partial_sums[survived, period],
                probabilities[survived],
            ),
        )
        ruined_now = first_ruin == period
        deficit_distributions.append(
            _weighted_distribution(
                partial_sums[ruined_now, period] - boundary,
                probabilities[ruined_now],
            ),
        )

    survival = float(np.clip(survival_grid[-1], 0.0, 1.0))
    ruin = float(np.clip(1.0 - survival, 0.0, 1.0))
    if not return_result:
        return ruin
    return FiniteTimeDependentRuinResult(
        initial_capital=initial,
        premiums=premium_vector,
        cumulative_premiums=cumulative,
        boundaries=boundaries,
        claim_scenarios=scenarios,
        scenario_probabilities=probabilities,
        survival_probability=survival,
        ruin_probability=ruin,
        survival_probabilities=survival_grid,
        ruin_probabilities=1.0 - survival_grid,
        ruin_time_probabilities=ruin_times,
        surplus_distributions=tuple(surplus_distributions),
        deficit_distributions=tuple(deficit_distributions),
    )


def exchangeable_bernoulli_claim_scenarios(
    success_count_pmf: ArrayLike,
    *,
    claim_amount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand an exchangeable Bernoulli sum law into equally likely scenarios."""

    count_pmf = _as_nonnegative_1d(success_count_pmf, "success_count_pmf")
    total = float(count_pmf.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("success_count_pmf must sum to one")
    amount = _nonnegative_float(claim_amount, "claim_amount")
    periods = count_pmf.size - 1
    scenarios: list[np.ndarray] = []
    probabilities: list[float] = []
    for count, mass in enumerate(count_pmf):
        if mass <= 0.0:
            continue
        denominator = math.comb(periods, count)
        for selected in combinations(range(periods), count):
            row = np.zeros(periods, dtype=float)
            row[list(selected)] = amount
            scenarios.append(row)
            probabilities.append(float(mass) / denominator)
    return np.vstack(scenarios), np.asarray(probabilities, dtype=float)


def distribution_cdf(
    distribution: tuple[np.ndarray, np.ndarray],
    thresholds: ArrayLike,
) -> np.ndarray:
    """Evaluate the CDF of a discrete diagnostic distribution."""

    values, probabilities = distribution
    points = _as_1d_float(thresholds, "thresholds")
    return _cdf_from_distribution(
        np.asarray(values, dtype=float),
        np.asarray(probabilities),
        points,
    )


def ruin_deficit_cdf(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
    *,
    period: int,
    thresholds: ArrayLike,
) -> np.ndarray:
    """Conditional CDF of the deficit at ruin in a given period."""

    return distribution_cdf(result.deficit_distributions[period], thresholds)


def surplus_cdf_given_survival(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
    *,
    period: int,
    thresholds: ArrayLike,
) -> np.ndarray:
    """Conditional CDF of surplus at a period given non-ruin up to that period."""

    return distribution_cdf(result.surplus_distributions[period], thresholds)


def ruin_deficit_quantile(
    result: FiniteTimeDiscreteTimeRuinResult | FiniteTimeDependentRuinResult,
    *,
    period: int,
    probability: float,
) -> float:
    """Conditional deficit quantile at ruin in a given period."""

    q = _nonnegative_float(probability, "probability")
    if q > 1.0:
        raise ValueError("probability must lie in [0, 1]")
    values, masses = result.deficit_distributions[period]
    if values.size == 0:
        return math.nan
    order = np.argsort(values)
    cumulative = np.cumsum(masses[order])
    return float(values[order[np.searchsorted(cumulative, q, side="left")]])


def period_lundberg_roots_from_pmf(
    increment_pmfs: ArrayLike,
    *,
    premiums: ArrayLike,
    grid_step: float = 1.0,
    upper: float | None = None,
    tol: float = 1e-12,
    max_bisection: int = 100,
) -> np.ndarray:
    """Solve ``E exp(r(X_t-c_t)) = 1`` for lattice period increments."""

    increments = _pmf_matrix(increment_pmfs, "increment_pmfs")
    premium_vector = _premium_vector(premiums, increments.shape[0])
    step = _positive_float(grid_step, "grid_step")
    tolerance = _positive_float(tol, "tol")
    max_iter = int(max_bisection)
    if max_iter <= 0:
        raise ValueError("max_bisection must be positive")
    claim_values = np.arange(increments.shape[1], dtype=float) * step
    roots = np.full(increments.shape[0], np.nan, dtype=float)
    for period, (pmf, premium) in enumerate(zip(increments, premium_vector, strict=True)):
        drift = float(np.dot(pmf, claim_values) - premium)
        if drift >= 0.0:
            continue

        def objective(root: float) -> float:
            return float(np.dot(pmf, np.exp(root * (claim_values - premium))) - 1.0)

        high = 1.0 if upper is None else _positive_float(upper, "upper")
        while objective(high) <= 0.0 and high < 1e6:
            high *= 2.0
        if objective(high) <= 0.0:
            continue
        low = 0.0
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            if objective(mid) <= 0.0:
                low = mid
            else:
                high = mid
            if high - low <= tolerance * max(1.0, high):
                break
        roots[period] = 0.5 * (low + high)
    return roots


def finite_time_lundberg_bounds(
    period_roots: ArrayLike,
    *,
    initial_capital: float,
) -> FiniteTimeLundbergBoundResult:
    """Return Castaner-style finite-time bounds ``exp(-R(t) u)``."""

    roots = _as_1d_float(period_roots, "period_roots")
    initial = _nonnegative_float(initial_capital, "initial_capital")
    finite_roots = np.where(np.isfinite(roots) & (roots > 0.0), roots, np.inf)
    adjustments = np.minimum.accumulate(finite_roots)
    bounds = np.where(np.isfinite(adjustments), np.exp(-adjustments * initial), 1.0)
    return FiniteTimeLundbergBoundResult(
        initial_capital=initial,
        period_roots=roots,
        adjustment_coefficients=adjustments,
        bounds=bounds,
    )


def exponential_lundberg_roots(
    claim_arrival_rates: ArrayLike,
    premiums: ArrayLike,
    claim_means: ArrayLike,
) -> np.ndarray:
    """Explicit roots for compound-Poisson exponential period increments."""

    rates = _as_nonnegative_1d(claim_arrival_rates, "claim_arrival_rates")
    premium_vector = _premium_vector(premiums, rates.size)
    means = _as_nonnegative_1d(claim_means, "claim_means")
    if means.size != rates.size:
        raise ValueError("claim_means must match claim_arrival_rates")
    roots = np.full(rates.size, np.nan, dtype=float)
    positive = (means > 0.0) & (premium_vector > rates * means)
    roots[positive] = 1.0 / means[positive] - rates[positive] / premium_vector[positive]
    return roots


def normal_lundberg_roots(
    claim_means: ArrayLike,
    claim_sds: ArrayLike,
    premiums: ArrayLike,
) -> np.ndarray:
    """Explicit normal-approximation roots ``2(c_t-m_t)/s_t^2``."""

    means = _as_nonnegative_1d(claim_means, "claim_means")
    sds = _as_nonnegative_1d(claim_sds, "claim_sds")
    premiums_array = _premium_vector(premiums, means.size)
    if sds.size != means.size:
        raise ValueError("claim_sds must match claim_means")
    roots = np.full(means.size, np.nan, dtype=float)
    positive = (sds > 0.0) & (premiums_array > means)
    roots[positive] = 2.0 * (premiums_array[positive] - means[positive]) / (sds[positive] ** 2)
    return roots


def castaner_exponential_principle_roots(
    interest_rates: ArrayLike,
    claim_means: ArrayLike,
    safety_loadings: ArrayLike,
    *,
    claim_arrival_rates: ArrayLike | None = None,
    principle: str = "expected",
) -> np.ndarray:
    """Explicit Castaner exponential roots under common premium principles."""

    rates = _as_1d_float(interest_rates, "interest_rates")
    if np.any(rates <= -1.0):
        raise ValueError("interest_rates must be greater than -1")
    means = _as_nonnegative_1d(claim_means, "claim_means")
    loadings = _as_nonnegative_1d(safety_loadings, "safety_loadings")
    if means.size != rates.size or loadings.size != rates.size:
        raise ValueError("claim_means and safety_loadings must match interest_rates")
    accumulation = discount_factors_from_interest(rates)[1:]
    selected = principle.lower().replace("_", "-").replace(" ", "-")
    roots = np.full(rates.size, np.nan, dtype=float)
    positive_mean = means > 0.0
    if selected in {"expected", "expected-value"}:
        roots[positive_mean] = (
            accumulation[positive_mean]
            * loadings[positive_mean]
            / ((1.0 + loadings[positive_mean]) * means[positive_mean])
        )
    elif selected in {"standard-deviation", "sd"}:
        if claim_arrival_rates is None:
            raise ValueError(
                "claim_arrival_rates are required for the standard-deviation principle",
            )
        lambdas = _as_nonnegative_1d(claim_arrival_rates, "claim_arrival_rates")
        if lambdas.size != rates.size:
            raise ValueError("claim_arrival_rates must match interest_rates")
        response = loadings * np.sqrt(2.0 * lambdas)
        denominator = lambdas + response
        valid = positive_mean & (denominator > 0.0)
        roots[valid] = accumulation[valid] * response[valid] / (denominator[valid] * means[valid])
    elif selected in {"variance", "var"}:
        valid = positive_mean
        roots[valid] = (
            accumulation[valid]
            * 2.0
            * loadings[valid]
            / (1.0 + rates[valid] + 2.0 * loadings[valid] * means[valid])
        )
    else:
        raise ValueError("principle must be 'expected', 'standard-deviation' or 'variance'")
    return roots
