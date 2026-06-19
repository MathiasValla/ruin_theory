"""Optimization helpers for prevention strategies."""

from __future__ import annotations

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

    if not isinstance(claim_distribution, ClaimDistribution):
        raise TypeError("claim_distribution must be a ClaimDistribution")
    gross_premium = _positive_float(premium_rate, "premium_rate")
    initial_capital = _nonnegative_float(initial_capital, "initial_capital")
    tol = _positive_float(tol, "tol")
    mean_claim = float(claim_distribution.mean())
    if not np.isfinite(mean_claim) or mean_claim <= 0.0:
        raise ValueError("claim_distribution must have finite positive mean")
    if not callable(frequency_function):
        raise TypeError("frequency_function must be callable")

    upper = gross_premium if max_prevention is None else _nonnegative_float(
        max_prevention,
        "max_prevention",
    )
    upper = min(upper, np.nextafter(gross_premium, 0.0))
    if upper <= 0.0:
        raise ValueError("admissible prevention interval must have positive length")

    threshold = 0.0 if activation_threshold is None else _nonnegative_float(
        activation_threshold,
        "activation_threshold",
    )
    if threshold >= upper:
        threshold = upper

    baseline_frequency = _frequency_value(frequency_function, 0.0)

    def loss_ratio(amount: float) -> float:
        net_premium = gross_premium - amount
        if net_premium <= 0.0:
            return np.inf
        return _frequency_value(frequency_function, amount) * mean_claim / net_premium

    candidates: list[tuple[float, str]] = [(0.0, "zero"), (upper, "upper")]
    if threshold > 0.0:
        candidates.append((threshold, "threshold"))

    lower = threshold if threshold > 0.0 else 0.0
    if upper - lower > tol:
        optimum = optimize.minimize_scalar(
            loss_ratio,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": tol},
        )
        if not optimum.success:
            raise ValueError(f"prevention optimization failed: {optimum.message}")
        candidates.append((float(optimum.x), "interior"))

    values = np.array([loss_ratio(amount) for amount, _ in candidates], dtype=float)
    best = int(np.argmin(values))
    amount, boundary = candidates[best]
    value = float(values[best])
    if not np.isfinite(value):
        raise ValueError("could not evaluate a finite prevention objective")

    net_premium = gross_premium - amount
    claim_arrival_rate = _frequency_value(frequency_function, amount)
    safety = 1.0 / value - 1.0
    non_ruin_zero = max(1.0 - value, 0.0)
    multiplier = claim_arrival_rate / baseline_frequency
    prevention_program = PreventionProgram(frequency_multiplier=multiplier)
    model = CramerLundbergProcess(
        initial_capital=initial_capital,
        premium_rate=net_premium,
        claim_arrival_rate=baseline_frequency,
        claim_distribution=claim_distribution,
        prevention=prevention_program,
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
