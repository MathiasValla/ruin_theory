"""Optimization helpers for prevention strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize

from .distributions import ClaimDistribution
from .formulas import adjustment_coefficient
from .models import CramerLundbergProcess, PreventionProgram


FrequencyFunction = Callable[[float], float]


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
    effectiveness: float
    tau: float | None
    lag_steps: int
    baseline_pressure: float
    controlled_pressure: float
    constant_pressure: float
    pressure_reduction: float

    @property
    def frequency_multipliers(self) -> np.ndarray:
        """Frequency multipliers induced by the effective calendar."""

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


def _budget_spent(amounts: np.ndarray, durations: np.ndarray) -> float:
    return float(np.dot(durations, amounts))


def optimize_constant_prevention(
    claim_distribution: ClaimDistribution,
    *,
    premium_rate: float,
    frequency_function: FrequencyFunction,
    max_prevention: float | None = None,
    activation_threshold: float | None = None,
    initial_capital: float = 0.0,
    compute_adjustment: bool = True,
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
    effectiveness: float,
    durations: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lag_steps: int = 0,
    tol: float = 1e-12,
) -> PeriodicPreventionResult:
    """Optimize a discrete periodic prevention calendar with exponential response.

    The objective is
    ``sum_i weights[i] * exp(-effectiveness * effective_p[i])`` subject to
    ``sum_i durations[i] * p[i] = annual_budget`` and
    ``0 <= p[i] <= max_prevention``. With `lag_steps > 0`, spending in period
    `i` affects pressure in period `i + lag_steps`.
    """

    pressure, period_lengths = _periodic_inputs(weights, durations)
    budget = _nonnegative_float(annual_budget, "annual_budget")
    cap = _positive_float(max_prevention, "max_prevention")
    response = _positive_float(effectiveness, "effectiveness")
    tol = _positive_float(tol, "tol")
    lag = int(lag_steps)
    if lag != lag_steps:
        raise ValueError("lag_steps must be an integer")

    shifted_pressure = np.roll(pressure, -lag)
    amounts, tau = _projected_log_calendar(
        weights=shifted_pressure,
        durations=period_lengths,
        annual_budget=budget,
        max_prevention=cap,
        effectiveness=response,
        tol=tol,
    )
    effective_amounts = np.roll(amounts, lag)
    controlled = float(np.dot(pressure, np.exp(-response * effective_amounts)))
    baseline = float(pressure.sum())
    constant = float(pressure.sum() * math.exp(-response * budget))
    reduction = 0.0 if baseline == 0.0 else 1.0 - controlled / baseline

    return PeriodicPreventionResult(
        amounts=amounts.copy(),
        effective_amounts=effective_amounts.copy(),
        weights=pressure.copy(),
        durations=period_lengths.copy(),
        annual_budget=budget,
        budget_spent=_budget_spent(amounts, period_lengths),
        max_prevention=cap,
        effectiveness=response,
        tau=tau,
        lag_steps=lag,
        baseline_pressure=baseline,
        controlled_pressure=controlled,
        constant_pressure=constant,
        pressure_reduction=float(reduction),
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
    controlled_density = (
        calendar.weights
        * np.exp(-calendar.effectiveness * calendar.effective_amounts)
        / calendar.durations
    )

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
