"""Optimization helpers for prevention strategies."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize

from .distributions import ClaimDistribution
from .formulas import adjustment_coefficient
from .models import CramerLundbergProcess, PreventionProgram


FrequencyFunction = Callable[[float], float]
PreventionResponse = Callable[[float], float]


class PreventionResponseWarning(UserWarning):
    """Warning emitted when a prevention response fails numerical shape checks."""


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _frequency_value(function: FrequencyFunction, amount: float) -> float:
    value = float(function(float(amount)))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("frequency_function must return finite positive values")
    return value


def _response_value(
    function: PreventionResponse,
    amount: float,
    name: str,
    *,
    strictly_positive: bool = False,
) -> float:
    value = float(function(float(amount)))
    invalid = not np.isfinite(value) or value < 0.0 or (strictly_positive and value <= 0.0)
    if invalid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must return finite {qualifier} values")
    return value


def _response_values(
    function: PreventionResponse,
    amounts: np.ndarray,
    name: str,
    *,
    strictly_positive: bool = False,
) -> np.ndarray:
    try:
        vector_values = np.asarray(function(amounts.astype(float, copy=False)), dtype=float)
    except (TypeError, ValueError, AttributeError):
        vector_values = None
    if vector_values is not None and vector_values.shape == amounts.shape:
        invalid = (
            not np.all(np.isfinite(vector_values))
            or np.any(vector_values < 0.0)
            or (strictly_positive and np.any(vector_values <= 0.0))
        )
        if invalid:
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must return finite {qualifier} values")
        return vector_values

    values = [
        _response_value(
            function,
            amount,
            name,
            strictly_positive=strictly_positive,
        )
        for amount in amounts.ravel()
    ]
    return np.asarray(values, dtype=float).reshape(amounts.shape)


def validate_prevention_response(
    prevention_response: PreventionResponse,
    *,
    max_prevention: float,
    grid_size: int = 129,
    tolerance: float = 1e-8,
    name: str = "prevention_response",
    strictly_positive: bool = False,
) -> np.ndarray:
    """Numerically check that a prevention response is decreasing, convex and C2-like.

    A callable cannot be certified analytically from samples, so this helper emits
    warnings for apparent violations on an evenly spaced grid over the admissible
    interval. It raises when the callable itself is invalid or returns impossible
    response values.
    """

    if not callable(prevention_response):
        raise TypeError(f"{name} must be callable")
    cap = _positive_float(max_prevention, "max_prevention")
    n_grid = int(grid_size)
    if n_grid != grid_size or n_grid < 5:
        raise ValueError("grid_size must be an integer greater than or equal to 5")
    tol = _positive_float(tolerance, "tolerance")
    grid = np.linspace(0.0, cap, n_grid)
    values = _response_values(
        prevention_response,
        grid,
        name,
        strictly_positive=strictly_positive,
    )
    scale = max(1.0, float(np.max(np.abs(values))))
    atol = tol * scale

    if np.any(np.diff(values) > atol):
        warnings.warn(
            f"{name} does not appear to be decreasing on [0, {cap:g}]",
            PreventionResponseWarning,
            stacklevel=2,
        )
    second = np.diff(values, n=2)
    if np.any(second < -atol):
        warnings.warn(
            f"{name} does not appear to be convex on [0, {cap:g}]",
            PreventionResponseWarning,
            stacklevel=2,
        )
    step = cap / (n_grid - 1)
    curvature = second / (step * step)
    if curvature.size > 2:
        curvature_scale = max(1.0, float(np.max(np.abs(curvature))))
        if float(np.max(np.abs(np.diff(curvature)))) > 0.5 * curvature_scale:
            warnings.warn(
                f"{name} does not appear to be C2 on [0, {cap:g}]",
                PreventionResponseWarning,
                stacklevel=2,
            )
    return values


def frequency_function_from_response(
    baseline_frequency: float,
    prevention_response: PreventionResponse,
    *,
    max_prevention: float | None = None,
    grid_size: int = 129,
    tolerance: float = 1e-8,
) -> FrequencyFunction:
    """Build ``lambda(p)=lambda0*f(p)`` from a prevention response function."""

    baseline = _positive_float(baseline_frequency, "baseline_frequency")
    if max_prevention is not None:
        validate_prevention_response(
            prevention_response,
            max_prevention=max_prevention,
            grid_size=grid_size,
            tolerance=tolerance,
        )

    def frequency(amount: float) -> float:
        return baseline * _response_value(prevention_response, amount, "prevention_response")

    return frequency


@dataclass(frozen=True)
class ConstantPreventionResult:
    """Optimal constant prevention amount and induced risk-process quantities."""

    amount: float
    gross_premium_rate: float
    net_premium_rate: float
    baseline_frequency: float
    claim_arrival_rate: float
    expected_claim_amount: float
    loss_ratio: float
    safety_loading: float
    non_ruin_probability_at_zero: float
    adjustment_coefficient: float | None
    boundary: str
    prevention_program: PreventionProgram
    model: CramerLundbergProcess


@dataclass(frozen=True)
class ExpectedSurplusPreventionResult:
    """Optimal constant prevention amount for expected surplus at a horizon."""

    amount: float
    horizon: float
    initial_capital: float
    gross_premium_rate: float
    net_premium_rate: float
    baseline_frequency: float
    claim_arrival_rate: float
    expected_claim_amount: float
    net_drift: float
    expected_surplus: float
    boundary: str
    prevention_program: PreventionProgram
    model: CramerLundbergProcess


@dataclass(frozen=True)
class PeriodicPreventionResult:
    """Discrete periodic KKT prevention calendar."""

    amounts: np.ndarray
    effective_amounts: np.ndarray
    weights: np.ndarray
    durations: np.ndarray
    annual_budget: float
    budget_spent: float
    max_prevention: float
    effectiveness: float | None
    prevention_response: PreventionResponse | None
    tau: float | None
    lag_steps: int
    baseline_pressure: float
    controlled_pressure: float
    constant_pressure: float
    pressure_reduction: float

    @property
    def frequency_multipliers(self) -> np.ndarray:
        """Frequency multipliers induced by the effective calendar."""

        if self.prevention_response is not None:
            return _response_values(
                self.prevention_response,
                self.effective_amounts,
                "prevention_response",
            )
        if self.effectiveness is None:
            raise ValueError("effectiveness is required for exponential calendars")
        return np.exp(-self.effectiveness * self.effective_amounts)

    def frequency_windows(
        self,
        *,
        start: float = 0.0,
        period: float = 1.0,
    ) -> tuple[tuple[float, float, float], ...]:
        """Return `(start, end, multiplier)` windows for `PreventionProgram`."""

        start_value = float(start)
        if not np.isfinite(start_value):
            raise ValueError("start must be finite")
        period_value = _positive_float(period, "period")
        edges = start_value + period_value * np.r_[0.0, np.cumsum(self.durations)]
        return tuple(
            (float(edges[i]), float(edges[i + 1]), float(multiplier))
            for i, multiplier in enumerate(self.frequency_multipliers)
        )


@dataclass(frozen=True)
class HeavyTailPreventionResult:
    """Heavy-tail periodic prevention calendar and ruin-time asymptotic."""

    calendar: PeriodicPreventionResult
    tail_index: float
    frequency_effectiveness: float
    severity_effectiveness: float
    annual_capacity: float | None
    net_annual_capacity: float | None
    expected_time_to_ruin_asymptotic: float | None

    @property
    def amounts(self) -> np.ndarray:
        """Spending calendar."""

        return self.calendar.amounts

    @property
    def controlled_tail_pressure(self) -> float:
        """Controlled annual tail constant."""

        return self.calendar.controlled_pressure


@dataclass(frozen=True)
class DynamicPreventionResult:
    """Finite-horizon dynamic seasonal prevention program."""

    amounts: np.ndarray
    weights: np.ndarray
    durations: np.ndarray
    initial_budget: float
    remaining_budget: np.ndarray
    budget_grid: np.ndarray
    value_function: np.ndarray
    max_prevention: float
    effectiveness: float | None
    prevention_response: PreventionResponse | None
    baseline_pressure: float
    controlled_pressure: float
    pressure_reduction: float

    @property
    def effective_amounts(self) -> np.ndarray:
        return self.amounts

    @property
    def frequency_multipliers(self) -> np.ndarray:
        if self.prevention_response is not None:
            return _response_values(
                self.prevention_response,
                self.amounts,
                "prevention_response",
            )
        if self.effectiveness is None:
            raise ValueError("effectiveness is required for exponential calendars")
        return np.exp(-self.effectiveness * self.amounts)

    def frequency_windows(
        self,
        *,
        start: float = 0.0,
        period: float = 1.0,
    ) -> tuple[tuple[float, float, float], ...]:
        start_value = float(start)
        if not np.isfinite(start_value):
            raise ValueError("start must be finite")
        period_value = _positive_float(period, "period")
        edges = start_value + period_value * np.r_[0.0, np.cumsum(self.durations)]
        return tuple(
            (float(edges[i]), float(edges[i + 1]), float(multiplier))
            for i, multiplier in enumerate(self.frequency_multipliers)
        )


@dataclass(frozen=True)
class TwoClaimPreventionResult:
    """Optimal prevention when only large-claim frequency is controlled."""

    amount: float
    objective: str
    gross_premium_rate: float
    net_premium_rate: float
    small_claim_arrival_rate: float
    large_claim_arrival_rate: float
    total_claim_arrival_rate: float
    small_claim_mean: float
    large_claim_mean: float
    loss_ratio: float
    non_ruin_probability_at_zero: float
    adjustment_coefficient: float | None
    prevention_is_useful_at_zero: bool
    boundary: str
    model: CramerLundbergProcess


def _finite_positive_claim_mean(distribution: ClaimDistribution) -> float:
    if not isinstance(distribution, ClaimDistribution):
        raise TypeError("claim_distribution must be a ClaimDistribution")
    mean_claim = float(distribution.mean())
    if not np.isfinite(mean_claim) or mean_claim <= 0.0:
        raise ValueError("claim_distribution must have finite positive mean")
    return mean_claim


def _prevention_bounds(
    *,
    premium_rate: float,
    max_prevention: float | None,
    activation_threshold: float | None,
) -> tuple[float, float]:
    upper = premium_rate if max_prevention is None else _nonnegative_float(
        max_prevention,
        "max_prevention",
    )
    upper = min(upper, np.nextafter(premium_rate, 0.0))
    if upper <= 0.0:
        raise ValueError("admissible prevention interval must have positive length")

    threshold = 0.0 if activation_threshold is None else _nonnegative_float(
        activation_threshold,
        "activation_threshold",
    )
    return upper, min(threshold, upper)


def _minimize_with_candidates(
    objective: Callable[[float], float],
    *,
    upper: float,
    threshold: float,
    tol: float,
) -> tuple[float, str, float]:
    candidates: list[tuple[float, str]] = [(0.0, "zero"), (upper, "upper")]
    if threshold > 0.0:
        candidates.append((threshold, "threshold"))

    lower = threshold if threshold > 0.0 else 0.0
    if upper - lower > tol:
        optimum = optimize.minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": tol},
        )
        if not optimum.success:
            raise ValueError(f"prevention optimization failed: {optimum.message}")
        candidates.append((float(optimum.x), "interior"))

    values = np.array([objective(amount) for amount, _ in candidates], dtype=float)
    best = int(np.argmin(values))
    amount, boundary = candidates[best]
    value = float(values[best])
    if not np.isfinite(value):
        raise ValueError("could not evaluate a finite prevention objective")
    return float(amount), boundary, value


def _constant_prevention_model(
    *,
    claim_distribution: ClaimDistribution,
    initial_capital: float,
    net_premium: float,
    baseline_frequency: float,
    claim_arrival_rate: float,
) -> tuple[PreventionProgram, CramerLundbergProcess]:
    prevention_program = PreventionProgram(
        frequency_multiplier=claim_arrival_rate / baseline_frequency,
    )
    model = CramerLundbergProcess(
        initial_capital=initial_capital,
        premium_rate=net_premium,
        claim_arrival_rate=baseline_frequency,
        claim_distribution=claim_distribution,
        prevention=prevention_program,
    )
    return prevention_program, model


def _as_1d_float_array(
    values: np.ndarray | list[float] | tuple[float, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(float, copy=True)


def _periodic_inputs(
    weights: np.ndarray | list[float] | tuple[float, ...],
    durations: np.ndarray | list[float] | tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = _as_1d_float_array(weights, "weights")
    if np.any(pressure < 0.0):
        raise ValueError("weights must be non-negative")

    if durations is None:
        period_lengths = np.full(pressure.size, 1.0 / pressure.size)
    else:
        period_lengths = _as_1d_float_array(durations, "durations")
        if period_lengths.shape != pressure.shape:
            raise ValueError("durations must have the same shape as weights")
        if np.any(period_lengths <= 0.0):
            raise ValueError("durations must be positive")
        if not np.isclose(period_lengths.sum(), 1.0, rtol=1e-8, atol=1e-12):
            raise ValueError("durations must sum to one year")

    return pressure, period_lengths


def _allocate_flat_budget(durations: np.ndarray, budget: float, cap: float) -> np.ndarray:
    amounts = np.zeros_like(durations)
    remaining = float(budget)
    for index in np.argsort(-durations):
        if remaining <= 0.0:
            break
        amount = min(cap, remaining / durations[index])
        amounts[index] = amount
        remaining -= durations[index] * amount
    return amounts


def _projected_log_calendar(
    *,
    weights: np.ndarray,
    durations: np.ndarray,
    annual_budget: float,
    max_prevention: float,
    effectiveness: float,
    tol: float,
) -> tuple[np.ndarray, float | None]:
    maximum_budget = max_prevention * durations.sum()
    if annual_budget > maximum_budget + tol:
        raise ValueError("annual_budget exceeds the instantaneous prevention cap")
    if annual_budget <= tol:
        return np.zeros_like(weights), None
    if annual_budget >= maximum_budget - tol:
        return np.full_like(weights, max_prevention), 0.0

    positive = weights > 0.0
    if not np.any(positive):
        return _allocate_flat_budget(durations, annual_budget, max_prevention), None

    positive_capacity = max_prevention * durations[positive].sum()
    amounts = np.zeros_like(weights)
    if annual_budget >= positive_capacity - tol:
        amounts[positive] = max_prevention
        remaining = annual_budget - float(np.dot(durations, amounts))
        if remaining > tol:
            zero_fill = _allocate_flat_budget(durations[~positive], remaining, max_prevention)
            amounts[~positive] = zero_fill
        return amounts, 0.0

    scores = weights[positive] / durations[positive]
    high = float(scores.max())
    low = 0.0
    for _ in range(160):
        tau = 0.5 * (low + high)
        raw = np.log(scores / tau) / effectiveness
        trial = np.clip(raw, 0.0, max_prevention)
        spent = float(np.dot(durations[positive], trial))
        if spent > annual_budget:
            low = tau
        else:
            high = tau

    tau = high
    amounts[positive] = np.clip(np.log(scores / tau) / effectiveness, 0.0, max_prevention)
    return amounts, float(tau)


def _generic_periodic_calendar(
    *,
    weights: np.ndarray,
    durations: np.ndarray,
    annual_budget: float,
    max_prevention: float,
    prevention_response: PreventionResponse,
    tol: float,
) -> np.ndarray:
    maximum_budget = max_prevention * durations.sum()
    if annual_budget > maximum_budget + tol:
        raise ValueError("annual_budget exceeds the instantaneous prevention cap")
    if annual_budget <= tol:
        return np.zeros_like(weights)
    if annual_budget >= maximum_budget - tol:
        return np.full_like(weights, max_prevention)
    if not np.any(weights > 0.0):
        return _allocate_flat_budget(durations, annual_budget, max_prevention)

    x0 = np.full_like(weights, annual_budget / durations.sum())

    def objective(amounts: np.ndarray) -> float:
        return float(
            np.dot(
                weights,
                _response_values(prevention_response, amounts, "prevention_response"),
            )
        )

    result = optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_prevention)] * weights.size,
        constraints={
            "type": "eq",
            "fun": lambda amounts: float(np.dot(durations, amounts) - annual_budget),
        },
        options={"ftol": tol, "maxiter": 1000},
    )
    if not result.success:
        raise ValueError(f"periodic prevention optimization failed: {result.message}")

    amounts = np.clip(np.asarray(result.x, dtype=float), 0.0, max_prevention)
    if abs(float(np.dot(durations, amounts)) - annual_budget) > max(1e-7, 100.0 * tol):
        raise ValueError("periodic prevention optimization did not satisfy the budget")
    return amounts


def _budget_spent(amounts: np.ndarray, durations: np.ndarray) -> float:
    return float(np.dot(durations, amounts))


def _numerical_derivative(
    function: FrequencyFunction,
    point: float,
    *,
    upper: float,
) -> float:
    h = max(1e-6, upper * 1e-6)
    x = float(point)
    if x - h < 0.0:
        return (_frequency_value(function, x + h) - _frequency_value(function, x)) / h
    return (_frequency_value(function, x + h) - _frequency_value(function, x - h)) / (2.0 * h)


def _mixture_claim_distribution(
    distributions: tuple[ClaimDistribution, ClaimDistribution],
    rates: tuple[float, float],
) -> ClaimDistribution:
    weights = np.asarray(rates, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("claim rates must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("at least one claim rate must be positive")
    weights = weights / total
    first, second = distributions
    means = np.array([first.mean(), second.mean()], dtype=float)
    variances = np.array(
        [
            np.nan if first.variance() is None else first.variance(),
            np.nan if second.variance() is None else second.variance(),
        ],
        dtype=float,
    )
    mean = float(weights @ means)
    variance = None
    if np.all(np.isfinite(variances)):
        second_moment = float(weights @ (variances + means**2))
        variance = max(second_moment - mean**2, 0.0)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        choices = rng.choice(2, size=n, p=weights)
        values = np.empty(n, dtype=float)
        for index, distribution in enumerate(distributions):
            mask = choices == index
            if np.any(mask):
                values[mask] = distribution.sample(int(np.count_nonzero(mask)), rng=rng)
        return values

    def weighted(method: str, x: np.ndarray) -> np.ndarray:
        return weights[0] * getattr(first, method)(x) + weights[1] * getattr(second, method)(x)

    def mgf(t: float) -> float:
        return float(weights[0] * first.mgf(t) + weights[1] * second.mgf(t))

    def laplace(s: float) -> float:
        return float(weights[0] * first.laplace(s) + weights[1] * second.laplace(s))

    return ClaimDistribution(
        name="two_claim_mixture",
        mean_value=mean,
        variance_value=variance,
        sampler=sampler,
        cdf_function=lambda x: weighted("cdf", np.asarray(x, dtype=float)),
        survival_function=lambda x: weighted("survival", np.asarray(x, dtype=float)),
        pdf_function=lambda x: weighted("pdf", np.asarray(x, dtype=float)),
        mgf_function=mgf,
        laplace_function=laplace,
        metadata={
            "weights": weights.copy(),
            "component_names": (first.name, second.name),
        },
    )


def _two_claim_adjustment_coefficient(
    small_claim_distribution: ClaimDistribution,
    large_claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    small_claim_arrival_rate: float,
    large_claim_arrival_rate: float,
    tol: float,
) -> float:
    premium = _positive_float(premium_rate, "premium_rate")
    lambda1 = _nonnegative_float(small_claim_arrival_rate, "small_claim_arrival_rate")
    lambda2 = _nonnegative_float(large_claim_arrival_rate, "large_claim_arrival_rate")
    mean_drift = premium - lambda1 * small_claim_distribution.mean()
    mean_drift -= lambda2 * large_claim_distribution.mean()
    if mean_drift <= 0.0:
        raise ValueError("net profit condition must hold")

    def equation(r: float) -> float:
        return (
            lambda1 * (small_claim_distribution.mgf(r) - 1.0)
            + lambda2 * (large_claim_distribution.mgf(r) - 1.0)
            - premium * r
        )

    lower = tol
    lower_value = equation(lower)
    for _ in range(20):
        if np.isfinite(lower_value) and lower_value < 0.0:
            break
        lower *= 0.1
        lower_value = equation(lower)
    else:
        raise ValueError("could not bracket the adjustment coefficient near zero")

    upper = 1.0
    bracket_low = lower
    for _ in range(80):
        value = equation(upper)
        if np.isfinite(value) and value > 0.0:
            break
        if np.isposinf(value):
            upper = 0.5 * (bracket_low + upper)
            continue
        if np.isfinite(value):
            bracket_low = upper
        upper *= 2.0
    else:
        raise ValueError("could not bracket the adjustment coefficient")
    return float(optimize.brentq(equation, lower, upper, xtol=tol, rtol=tol))


def _exponential_response(effectiveness: float) -> PreventionResponse:
    response = _positive_float(effectiveness, "effectiveness")

    def function(amount: float) -> float:
        return math.exp(-response * amount)

    return function


def _periodic_response_function(
    *,
    effectiveness: float | None,
    prevention_response: PreventionResponse | None,
    max_prevention: float,
    validate_response: bool,
    response_grid_size: int,
    response_tolerance: float,
) -> tuple[PreventionResponse, float | None]:
    if prevention_response is None:
        if effectiveness is None:
            raise ValueError("either effectiveness or prevention_response must be provided")
        response = _positive_float(effectiveness, "effectiveness")
        return _exponential_response(response), response
    if effectiveness is not None:
        raise ValueError("provide either effectiveness or prevention_response, not both")
    if validate_response:
        validate_prevention_response(
            prevention_response,
            max_prevention=max_prevention,
            grid_size=response_grid_size,
            tolerance=response_tolerance,
        )
    else:
        _response_value(prevention_response, 0.0, "prevention_response")
    return prevention_response, None


def periodic_pressure_weights(
    frequency_rates: np.ndarray | list[float] | tuple[float, ...],
    *,
    severity_weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    durations: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Integrate period rates into annual pressure weights.

    Use `severity_weights=None` for annual frequency weights, mean retained
    severities for expected-loss pressure, `M_i(r)-1` for seasonal Lundberg
    pressure, or tail constants for heavy-tail pressure.
    """

    rates, period_lengths = _periodic_inputs(frequency_rates, durations)
    if severity_weights is None:
        multipliers = np.ones_like(rates)
    else:
        multipliers = _as_1d_float_array(severity_weights, "severity_weights")
        if multipliers.shape != rates.shape:
            raise ValueError("severity_weights must have the same shape as frequency_rates")
        if np.any(multipliers < 0.0):
            raise ValueError("severity_weights must be non-negative")
    return rates * multipliers * period_lengths


def periodic_controlled_pressure(
    weights: np.ndarray | list[float] | tuple[float, ...],
    amounts: np.ndarray | list[float] | tuple[float, ...],
    *,
    effectiveness: float | None = None,
    prevention_response: PreventionResponse | None = None,
    lag_steps: int = 0,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
) -> float:
    """Evaluate the controlled periodic pressure for a fixed calendar."""

    pressure = _as_1d_float_array(weights, "weights")
    if np.any(pressure < 0.0):
        raise ValueError("weights must be non-negative")
    prevention = _as_1d_float_array(amounts, "amounts")
    if prevention.shape != pressure.shape:
        raise ValueError("amounts must have the same shape as weights")
    if np.any(prevention < 0.0):
        raise ValueError("amounts must be non-negative")
    lag = int(lag_steps)
    if lag != lag_steps:
        raise ValueError("lag_steps must be an integer")
    effective = np.roll(prevention, lag)
    if prevention_response is None:
        response = _positive_float(effectiveness, "effectiveness")
        multipliers = np.exp(-response * effective)
    else:
        if effectiveness is not None:
            raise ValueError("provide either effectiveness or prevention_response, not both")
        cap = float(prevention.max(initial=0.0))
        if validate_response and cap > 0.0:
            validate_prevention_response(
                prevention_response,
                max_prevention=cap,
                grid_size=response_grid_size,
                tolerance=response_tolerance,
            )
        multipliers = _response_values(prevention_response, effective, "prevention_response")
    return float(np.dot(pressure, multipliers))


def periodic_net_profit(
    *,
    premium_rate: float,
    annual_budget: float,
    claim_mean: float,
    controlled_frequency: float,
) -> float:
    """Annual periodic net profit margin `c - B(p) - m A(p)`."""

    premium = _positive_float(premium_rate, "premium_rate")
    budget = _nonnegative_float(annual_budget, "annual_budget")
    mean = _positive_float(claim_mean, "claim_mean")
    frequency = _nonnegative_float(controlled_frequency, "controlled_frequency")
    return float(premium - budget - mean * frequency)


def periodic_lundberg_coefficient(
    claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    annual_budget: float,
    controlled_frequency: float,
    upper: float | None = None,
    tol: float = 1e-12,
) -> float:
    """Solve the one-year periodic Lundberg equation.

    The equation is `rho * (c - B(p)) = A(p) * (M_X(rho) - 1)`, where `A(p)`
    is the controlled annual frequency and `B(p)` is the annual prevention
    budget.
    """

    mean_claim = _finite_positive_claim_mean(claim_distribution)
    premium = _positive_float(premium_rate, "premium_rate")
    budget = _nonnegative_float(annual_budget, "annual_budget")
    frequency = _nonnegative_float(controlled_frequency, "controlled_frequency")
    tol = _positive_float(tol, "tol")
    net_premium = premium - budget
    if net_premium <= 0.0:
        raise ValueError("premium_rate must exceed annual_budget")
    if periodic_net_profit(
        premium_rate=premium,
        annual_budget=budget,
        claim_mean=mean_claim,
        controlled_frequency=frequency,
    ) <= 0.0:
        raise ValueError("periodic net profit condition must hold")
    if frequency == 0.0:
        raise ValueError("controlled_frequency must be positive")

    def kappa(r: float) -> float:
        return frequency * (claim_distribution.mgf(r) - 1.0) - net_premium * r

    lower = tol
    lower_value = kappa(lower)
    for _ in range(20):
        if np.isfinite(lower_value) and lower_value < 0.0:
            break
        lower *= 0.1
        lower_value = kappa(lower)
    else:
        raise ValueError("could not bracket the periodic Lundberg coefficient near zero")

    def finite_positive_upper(low: float, high: float) -> float:
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            midpoint_value = kappa(midpoint)
            if np.isfinite(midpoint_value):
                if midpoint_value > 0.0:
                    return midpoint
                low = midpoint
            else:
                high = midpoint
        raise ValueError("could not bracket the periodic Lundberg coefficient")

    if upper is None:
        high = 1.0
        bracket_low = lower
        for _ in range(80):
            value = kappa(high)
            if np.isfinite(value) and value > 0.0:
                break
            if np.isposinf(value):
                high = finite_positive_upper(bracket_low, high)
                break
            if np.isfinite(value):
                bracket_low = high
            high *= 2.0
        else:
            raise ValueError("could not bracket the periodic Lundberg coefficient")
    else:
        high = _positive_float(upper, "upper")
        value = kappa(high)
        if np.isposinf(value):
            high = finite_positive_upper(lower, high)
        elif not np.isfinite(value) or value <= 0.0:
            raise ValueError("upper does not bracket the periodic Lundberg coefficient")
    return float(optimize.brentq(kappa, lower, high, xtol=tol, rtol=tol))


def optimize_constant_prevention(
    claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    frequency_function: FrequencyFunction,
    max_prevention: float | None = None,
    activation_threshold: float | None = None,
    initial_capital: float = 0.0,
    compute_adjustment: bool = True,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
    tol: float = 1e-10,
) -> ConstantPreventionResult:
    """Optimize a constant prevention spend in the Gauchon et al. model.

    The surplus model is ``U(t,p) = u + (c-p)t - sum_{i<=N_p(t)} X_i`` with
    arrival intensity ``lambda(p)``. Following Gauchon et al. (2020), the
    prevention amount maximizing the infinite-time non-ruin probability also
    maximizes the adjustment coefficient. Numerically this is the minimizer of
    ``lambda(p) * E[X] / (c-p)`` over the admissible interval.
    """

    gross_premium = _positive_float(premium_rate, "premium_rate")
    initial_capital = _nonnegative_float(initial_capital, "initial_capital")
    tol = _positive_float(tol, "tol")
    mean_claim = _finite_positive_claim_mean(claim_distribution)
    if not callable(frequency_function):
        raise TypeError("frequency_function must be callable")

    upper, threshold = _prevention_bounds(
        premium_rate=gross_premium,
        max_prevention=max_prevention,
        activation_threshold=activation_threshold,
    )
    if validate_response:
        validate_prevention_response(
            frequency_function,
            max_prevention=upper,
            grid_size=response_grid_size,
            tolerance=response_tolerance,
            name="frequency_function",
            strictly_positive=True,
        )

    baseline_frequency = _frequency_value(frequency_function, 0.0)

    def loss_ratio(amount: float) -> float:
        net_premium = gross_premium - amount
        if net_premium <= 0.0:
            return np.inf
        return _frequency_value(frequency_function, amount) * mean_claim / net_premium

    amount, boundary, value = _minimize_with_candidates(
        loss_ratio,
        upper=upper,
        threshold=threshold,
        tol=tol,
    )

    net_premium = gross_premium - amount
    claim_arrival_rate = _frequency_value(frequency_function, amount)
    safety = 1.0 / value - 1.0
    non_ruin_zero = max(1.0 - value, 0.0)
    prevention_program, model = _constant_prevention_model(
        claim_distribution=claim_distribution,
        initial_capital=initial_capital,
        net_premium=net_premium,
        baseline_frequency=baseline_frequency,
        claim_arrival_rate=claim_arrival_rate,
    )

    coefficient: float | None = None
    if compute_adjustment and value < 1.0:
        try:
            coefficient = adjustment_coefficient(model)
        except (NotImplementedError, ValueError, OverflowError):
            coefficient = None

    return ConstantPreventionResult(
        amount=float(amount),
        gross_premium_rate=gross_premium,
        net_premium_rate=float(net_premium),
        baseline_frequency=baseline_frequency,
        claim_arrival_rate=claim_arrival_rate,
        expected_claim_amount=mean_claim,
        loss_ratio=value,
        safety_loading=float(safety),
        non_ruin_probability_at_zero=float(non_ruin_zero),
        adjustment_coefficient=coefficient,
        boundary=boundary,
        prevention_program=prevention_program,
        model=model,
    )


def optimize_expected_surplus_prevention(
    claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    frequency_function: FrequencyFunction,
    horizon: float,
    max_prevention: float | None = None,
    activation_threshold: float | None = None,
    initial_capital: float = 0.0,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
    tol: float = 1e-10,
) -> ExpectedSurplusPreventionResult:
    """Optimize constant prevention for expected surplus at a fixed horizon.

    Gauchon et al. (2020) show that
    ``E[U(t,p)] = u + (c - p - lambda(p) * E[X]) t``. Hence this optimizer
    maximizes the net drift ``c - p - lambda(p) * E[X]`` over the admissible
    prevention interval. The resulting amount generally differs from the
    long-run ruin-probability optimizer.
    """

    gross_premium = _positive_float(premium_rate, "premium_rate")
    horizon_value = _positive_float(horizon, "horizon")
    initial_capital = _nonnegative_float(initial_capital, "initial_capital")
    tol = _positive_float(tol, "tol")
    mean_claim = _finite_positive_claim_mean(claim_distribution)
    if not callable(frequency_function):
        raise TypeError("frequency_function must be callable")

    upper, threshold = _prevention_bounds(
        premium_rate=gross_premium,
        max_prevention=max_prevention,
        activation_threshold=activation_threshold,
    )
    if validate_response:
        validate_prevention_response(
            frequency_function,
            max_prevention=upper,
            grid_size=response_grid_size,
            tolerance=response_tolerance,
            name="frequency_function",
            strictly_positive=True,
        )
    baseline_frequency = _frequency_value(frequency_function, 0.0)

    def negative_net_drift(amount: float) -> float:
        net_premium = gross_premium - amount
        return -(net_premium - _frequency_value(frequency_function, amount) * mean_claim)

    amount, boundary, objective = _minimize_with_candidates(
        negative_net_drift,
        upper=upper,
        threshold=threshold,
        tol=tol,
    )

    net_premium = gross_premium - amount
    claim_arrival_rate = _frequency_value(frequency_function, amount)
    net_drift = -objective
    expected_surplus = initial_capital + net_drift * horizon_value
    prevention_program, model = _constant_prevention_model(
        claim_distribution=claim_distribution,
        initial_capital=initial_capital,
        net_premium=net_premium,
        baseline_frequency=baseline_frequency,
        claim_arrival_rate=claim_arrival_rate,
    )

    return ExpectedSurplusPreventionResult(
        amount=float(amount),
        horizon=horizon_value,
        initial_capital=initial_capital,
        gross_premium_rate=gross_premium,
        net_premium_rate=float(net_premium),
        baseline_frequency=baseline_frequency,
        claim_arrival_rate=claim_arrival_rate,
        expected_claim_amount=mean_claim,
        net_drift=float(net_drift),
        expected_surplus=float(expected_surplus),
        boundary=boundary,
        prevention_program=prevention_program,
        model=model,
    )


def optimize_periodic_prevention_calendar(
    weights: np.ndarray | list[float] | tuple[float, ...],
    *,
    annual_budget: float,
    max_prevention: float,
    effectiveness: float | None = None,
    prevention_response: PreventionResponse | None = None,
    durations: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lag_steps: int = 0,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
    tol: float = 1e-12,
) -> PeriodicPreventionResult:
    """Optimize a discrete periodic prevention calendar.

    The objective is
    ``sum_i weights[i] * f(effective_p[i])`` subject to
    ``sum_i durations[i] * p[i] = annual_budget`` and
    ``0 <= p[i] <= max_prevention``. Use `effectiveness` for the default
    exponential response ``f(p)=exp(-effectiveness*p)``, or pass a custom
    `prevention_response`. With `lag_steps > 0`, spending in period `i` affects
    pressure in period `i + lag_steps`.
    """

    pressure, period_lengths = _periodic_inputs(weights, durations)
    budget = _nonnegative_float(annual_budget, "annual_budget")
    cap = _positive_float(max_prevention, "max_prevention")
    tol = _positive_float(tol, "tol")
    lag = int(lag_steps)
    if lag != lag_steps:
        raise ValueError("lag_steps must be an integer")
    response_function, exponential_effectiveness = _periodic_response_function(
        effectiveness=effectiveness,
        prevention_response=prevention_response,
        max_prevention=cap,
        validate_response=validate_response,
        response_grid_size=response_grid_size,
        response_tolerance=response_tolerance,
    )

    shifted_pressure = np.roll(pressure, -lag)
    if exponential_effectiveness is None:
        amounts = _generic_periodic_calendar(
            weights=shifted_pressure,
            durations=period_lengths,
            annual_budget=budget,
            max_prevention=cap,
            prevention_response=response_function,
            tol=tol,
        )
        tau = None
    else:
        amounts, tau = _projected_log_calendar(
            weights=shifted_pressure,
            durations=period_lengths,
            annual_budget=budget,
            max_prevention=cap,
            effectiveness=exponential_effectiveness,
            tol=tol,
        )
    effective_amounts = np.roll(amounts, lag)
    controlled = float(
        np.dot(
            pressure,
            _response_values(response_function, effective_amounts, "prevention_response"),
        )
    )
    baseline = float(
        pressure.sum() * _response_value(response_function, 0.0, "prevention_response")
    )
    constant = float(
        pressure.sum() * _response_value(response_function, budget, "prevention_response")
    )
    reduction = 0.0 if baseline == 0.0 else 1.0 - controlled / baseline

    return PeriodicPreventionResult(
        amounts=amounts.copy(),
        effective_amounts=effective_amounts.copy(),
        weights=pressure.copy(),
        durations=period_lengths.copy(),
        annual_budget=budget,
        budget_spent=_budget_spent(amounts, period_lengths),
        max_prevention=cap,
        effectiveness=exponential_effectiveness,
        prevention_response=prevention_response,
        tau=tau,
        lag_steps=lag,
        baseline_pressure=baseline,
        controlled_pressure=controlled,
        constant_pressure=constant,
        pressure_reduction=float(reduction),
    )


def optimize_dynamic_prevention_calendar(
    weights: np.ndarray | list[float] | tuple[float, ...],
    *,
    initial_budget: float,
    max_prevention: float,
    effectiveness: float | None = None,
    prevention_response: PreventionResponse | None = None,
    durations: np.ndarray | list[float] | tuple[float, ...] | None = None,
    n_cycles: int = 1,
    budget_grid_size: int = 101,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
) -> DynamicPreventionResult:
    """Dynamic finite-horizon seasonal prevention by backward induction.

    Unlike `optimize_periodic_prevention_calendar`, this does not force a fixed
    repeating calendar. It allocates a finite prevention budget over
    `n_cycles * len(weights)` periods and may save budget for later seasons.
    """

    base_weights, base_durations = _periodic_inputs(weights, durations)
    cycles = int(n_cycles)
    if cycles != n_cycles or cycles <= 0:
        raise ValueError("n_cycles must be a positive integer")
    budget = _nonnegative_float(initial_budget, "initial_budget")
    cap = _positive_float(max_prevention, "max_prevention")
    grid_count = int(budget_grid_size)
    if grid_count != budget_grid_size or grid_count < 2:
        raise ValueError("budget_grid_size must be an integer greater than one")
    response_function, exponential_effectiveness = _periodic_response_function(
        effectiveness=effectiveness,
        prevention_response=prevention_response,
        max_prevention=cap,
        validate_response=validate_response,
        response_grid_size=response_grid_size,
        response_tolerance=response_tolerance,
    )

    pressure = np.tile(base_weights, cycles)
    period_lengths = np.tile(base_durations, cycles)
    budget_grid = np.linspace(0.0, budget, grid_count)
    values = np.zeros((pressure.size + 1, grid_count), dtype=float)
    decisions = np.zeros((pressure.size, grid_count), dtype=float)

    for period in range(pressure.size - 1, -1, -1):
        duration = period_lengths[period]
        for budget_index, remaining in enumerate(budget_grid):
            max_spend = min(remaining, cap * duration)
            feasible = budget_grid[budget_grid <= max_spend + 1e-12]
            if feasible.size == 0:
                feasible = np.array([0.0])
            amounts = feasible / duration
            response = _response_values(response_function, amounts, "prevention_response")
            future_budget = remaining - feasible
            future = np.interp(future_budget, budget_grid, values[period + 1])
            objective = pressure[period] * response + future
            best = int(np.argmin(objective))
            values[period, budget_index] = float(objective[best])
            decisions[period, budget_index] = float(feasible[best])

    remaining = budget
    spends = np.zeros(pressure.size, dtype=float)
    remaining_path = np.empty(pressure.size + 1, dtype=float)
    remaining_path[0] = remaining
    for period in range(pressure.size):
        spend = float(np.interp(remaining, budget_grid, decisions[period]))
        spend = min(max(spend, 0.0), remaining, cap * period_lengths[period])
        spends[period] = spend
        remaining -= spend
        remaining_path[period + 1] = remaining

    amounts = np.divide(
        spends,
        period_lengths,
        out=np.zeros_like(spends),
        where=period_lengths > 0.0,
    )
    controlled = float(
        np.dot(pressure, _response_values(response_function, amounts, "prevention_response"))
    )
    baseline = float(
        pressure.sum() * _response_value(response_function, 0.0, "prevention_response")
    )
    reduction = 0.0 if baseline == 0.0 else 1.0 - controlled / baseline
    return DynamicPreventionResult(
        amounts=amounts,
        weights=pressure,
        durations=period_lengths,
        initial_budget=budget,
        remaining_budget=remaining_path,
        budget_grid=budget_grid,
        value_function=values,
        max_prevention=cap,
        effectiveness=exponential_effectiveness,
        prevention_response=prevention_response,
        baseline_pressure=baseline,
        controlled_pressure=controlled,
        pressure_reduction=float(reduction),
    )


def two_claim_prevention_useful_at_zero(
    *,
    premium_rate: float,
    small_claim_arrival_rate: float,
    large_claim_frequency_function: FrequencyFunction,
    small_claim_mean: float,
    large_claim_mean: float,
) -> bool:
    """Check Gauchon et al. (2021) condition for positive prevention at `u=0`."""

    c = _positive_float(premium_rate, "premium_rate")
    lambda1 = _nonnegative_float(small_claim_arrival_rate, "small_claim_arrival_rate")
    mu1 = _positive_float(small_claim_mean, "small_claim_mean")
    mu2 = _positive_float(large_claim_mean, "large_claim_mean")
    lambda20 = _frequency_value(large_claim_frequency_function, 0.0)
    derivative = _numerical_derivative(large_claim_frequency_function, 0.0, upper=c)
    threshold = (lambda1 * mu1 + lambda20 * mu2) / (mu2 * c)
    return bool(-derivative > threshold)


def optimize_two_claim_prevention(
    small_claim_distribution: ClaimDistribution,
    large_claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    small_claim_arrival_rate: float,
    large_claim_frequency_function: FrequencyFunction,
    objective: str = "zero_surplus",
    max_prevention: float | None = None,
    initial_capital: float = 0.0,
    validate_response: bool = True,
    response_grid_size: int = 129,
    response_tolerance: float = 1e-8,
    tol: float = 1e-10,
) -> TwoClaimPreventionResult:
    """Optimize prevention that reduces only large-claim frequency."""

    if not isinstance(small_claim_distribution, ClaimDistribution):
        raise TypeError("small_claim_distribution must be a ClaimDistribution")
    if not isinstance(large_claim_distribution, ClaimDistribution):
        raise TypeError("large_claim_distribution must be a ClaimDistribution")
    c = _positive_float(premium_rate, "premium_rate")
    lambda1 = _nonnegative_float(small_claim_arrival_rate, "small_claim_arrival_rate")
    initial = _nonnegative_float(initial_capital, "initial_capital")
    tol = _positive_float(tol, "tol")
    mu1 = _finite_positive_claim_mean(small_claim_distribution)
    mu2 = _finite_positive_claim_mean(large_claim_distribution)
    upper, _ = _prevention_bounds(
        premium_rate=c,
        max_prevention=max_prevention,
        activation_threshold=None,
    )
    if validate_response:
        validate_prevention_response(
            large_claim_frequency_function,
            max_prevention=upper,
            grid_size=response_grid_size,
            tolerance=response_tolerance,
            name="large_claim_frequency_function",
            strictly_positive=True,
        )

    objective_name = objective.lower()

    def lambda2(amount: float) -> float:
        return _frequency_value(large_claim_frequency_function, amount)

    def loss_ratio(amount: float) -> float:
        net = c - amount
        return (lambda1 * mu1 + lambda2(amount) * mu2) / net

    def heavy_tail_constant(amount: float) -> float:
        denominator = c - amount - lambda1 * mu1 - lambda2(amount) * mu2
        if denominator <= 0.0:
            return np.inf
        return lambda2(amount) / denominator

    def adjustment(amount: float) -> float:
        return _two_claim_adjustment_coefficient(
            small_claim_distribution,
            large_claim_distribution,
            premium_rate=c - amount,
            small_claim_arrival_rate=lambda1,
            large_claim_arrival_rate=lambda2(amount),
            tol=tol,
        )

    if objective_name == "zero_surplus":
        objective_function = loss_ratio
    elif objective_name == "adjustment_coefficient":
        objective_function = lambda amount: -adjustment(amount)
    elif objective_name == "heavy_tail_large":
        objective_function = heavy_tail_constant
    else:
        raise ValueError(
            "objective must be 'zero_surplus', 'adjustment_coefficient' or 'heavy_tail_large'",
        )

    amount, boundary, _ = _minimize_with_candidates(
        objective_function,
        upper=upper,
        threshold=0.0,
        tol=tol,
    )
    selected_lambda2 = lambda2(amount)
    net_premium = c - amount
    total_lambda = lambda1 + selected_lambda2
    mixture = _mixture_claim_distribution(
        (small_claim_distribution, large_claim_distribution),
        (lambda1, selected_lambda2),
    )
    selected_loss_ratio = loss_ratio(amount)
    coefficient: float | None = None
    if selected_loss_ratio < 1.0:
        try:
            coefficient = adjustment(amount)
        except (ValueError, OverflowError, NotImplementedError):
            coefficient = None

    return TwoClaimPreventionResult(
        amount=float(amount),
        objective=objective_name,
        gross_premium_rate=c,
        net_premium_rate=float(net_premium),
        small_claim_arrival_rate=lambda1,
        large_claim_arrival_rate=float(selected_lambda2),
        total_claim_arrival_rate=float(total_lambda),
        small_claim_mean=mu1,
        large_claim_mean=mu2,
        loss_ratio=float(selected_loss_ratio),
        non_ruin_probability_at_zero=float(max(1.0 - selected_loss_ratio, 0.0)),
        adjustment_coefficient=coefficient,
        prevention_is_useful_at_zero=two_claim_prevention_useful_at_zero(
            premium_rate=c,
            small_claim_arrival_rate=lambda1,
            large_claim_frequency_function=large_claim_frequency_function,
            small_claim_mean=mu1,
            large_claim_mean=mu2,
        ),
        boundary=boundary,
        model=CramerLundbergProcess(
            initial_capital=initial,
            premium_rate=net_premium,
            claim_arrival_rate=total_lambda,
            claim_distribution=mixture,
        ),
    )


def heavy_tail_expected_ruin_time_asymptotic(
    *,
    tail_index: float,
    annual_capacity: float,
    tail_constant: float,
    annual_budget: float = 0.0,
) -> float:
    """Large-budget expected ruin-time asymptotic for infinite-mean tails."""

    alpha = _positive_float(tail_index, "tail_index")
    if alpha >= 1.0:
        raise ValueError("tail_index must lie in (0, 1)")
    capacity = _positive_float(annual_capacity, "annual_capacity")
    budget = _nonnegative_float(annual_budget, "annual_budget")
    tail = _positive_float(tail_constant, "tail_constant")
    net_capacity = capacity - budget
    if net_capacity <= 0.0:
        raise ValueError("annual_capacity must exceed annual_budget")

    capacity_power = alpha / (1.0 - alpha)
    tail_power = -1.0 / (1.0 - alpha)
    return float(net_capacity**capacity_power * (tail * math.gamma(1.0 - alpha)) ** tail_power)


def optimize_heavy_tail_prevention_calendar(
    tail_pressures: np.ndarray | list[float] | tuple[float, ...],
    *,
    tail_index: float,
    annual_budget: float,
    max_prevention: float,
    frequency_effectiveness: float,
    severity_effectiveness: float = 0.0,
    durations: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lag_steps: int = 0,
    annual_capacity: float | None = None,
    tol: float = 1e-12,
) -> HeavyTailPreventionResult:
    """Optimize the heavy-tail periodic tail-pressure calendar.

    For regularly varying severities with index `tail_index`, exponential
    frequency response `exp(-a p)` and multiplicative severity response
    `exp(-b p)`, the controlled tail pressure uses effectiveness
    `a + tail_index * b`.
    """

    alpha = _positive_float(tail_index, "tail_index")
    if alpha >= 1.0:
        raise ValueError("tail_index must lie in (0, 1)")
    a = _nonnegative_float(frequency_effectiveness, "frequency_effectiveness")
    b = _nonnegative_float(severity_effectiveness, "severity_effectiveness")
    effectiveness = a + alpha * b
    if effectiveness <= 0.0:
        raise ValueError("at least one prevention effectiveness must be positive")

    calendar = optimize_periodic_prevention_calendar(
        tail_pressures,
        annual_budget=annual_budget,
        max_prevention=max_prevention,
        effectiveness=effectiveness,
        durations=durations,
        lag_steps=lag_steps,
        tol=tol,
    )

    net_capacity: float | None = None
    expected_time: float | None = None
    if annual_capacity is not None:
        capacity = _positive_float(annual_capacity, "annual_capacity")
        net_capacity = capacity - calendar.budget_spent
        if net_capacity <= 0.0:
            raise ValueError("annual_capacity must exceed the prevention budget")
        expected_time = heavy_tail_expected_ruin_time_asymptotic(
            tail_index=alpha,
            annual_capacity=capacity,
            annual_budget=calendar.budget_spent,
            tail_constant=calendar.controlled_pressure,
        )

    return HeavyTailPreventionResult(
        calendar=calendar,
        tail_index=alpha,
        frequency_effectiveness=a,
        severity_effectiveness=b,
        annual_capacity=None if annual_capacity is None else float(annual_capacity),
        net_annual_capacity=net_capacity,
        expected_time_to_ruin_asymptotic=expected_time,
    )


def heavy_tail_one_big_jump_ruin_probability(
    calendar: PeriodicPreventionResult,
    *,
    tail_index: float,
    initial_capital: float,
    annual_capacity: float,
    horizon: float,
    start_period: int = 0,
    steps_per_period: int = 32,
) -> float:
    """One-big-jump finite-horizon ruin-probability heuristic.

    The approximation discretizes the periodic integral
    ``int W_p(s+t) * (u + c t - int_0^t p(s+r)dr)^(-alpha) dt``.
    """

    if not isinstance(calendar, PeriodicPreventionResult):
        raise TypeError("calendar must be a PeriodicPreventionResult")
    alpha = _positive_float(tail_index, "tail_index")
    if alpha >= 1.0:
        raise ValueError("tail_index must lie in (0, 1)")
    reserve = _nonnegative_float(initial_capital, "initial_capital")
    capacity = _positive_float(annual_capacity, "annual_capacity")
    horizon_value = _positive_float(horizon, "horizon")
    steps = int(steps_per_period)
    if steps != steps_per_period or steps <= 0:
        raise ValueError("steps_per_period must be a positive integer")

    n_periods = calendar.amounts.size
    period = int(start_period) % n_periods
    time_remaining = horizon_value
    elapsed = 0.0
    approximation = 0.0
    controlled_density = calendar.weights * calendar.frequency_multipliers / calendar.durations

    while time_remaining > 0.0:
        duration = float(calendar.durations[period])
        dt = min(duration / steps, time_remaining)
        for _ in range(steps):
            if time_remaining <= 0.0:
                break
            step = min(dt, time_remaining)
            midpoint = elapsed + 0.5 * step
            spent = reserve + capacity * midpoint
            if spent <= 0.0:
                return 1.0
            approximation += controlled_density[period] * spent ** (-alpha) * step
            reserve -= calendar.amounts[period] * step
            elapsed += step
            time_remaining -= step
        period = (period + 1) % n_periods

    return float(min(max(approximation, 0.0), 1.0))
