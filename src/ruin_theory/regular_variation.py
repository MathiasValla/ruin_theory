"""Infinite-mean ruin tools for regularly varying tails."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, special


@dataclass(frozen=True)
class PremiumPowerCondition:
    """Condition `beta > 1 / alpha` from the infinite-mean KLR theorem."""

    tail_index: float
    premium_power: float
    threshold: float
    margin: float

    @property
    def holds(self) -> bool:
        return self.margin > 0.0


@dataclass(frozen=True)
class RegularlyVaryingTail:
    """Regularly varying claim tail with index `0 < alpha <= 1`."""

    tail_index: float
    survival_function: Callable[[ArrayLike], ArrayLike] | None = None
    scale: float = 1.0
    tail_constant: float = 1.0
    name: str = "regularly varying"

    def __post_init__(self) -> None:
        alpha = _positive_float(self.tail_index, "tail_index")
        if alpha > 1.0:
            raise ValueError("tail_index must be less than or equal to one")
        _positive_float(self.scale, "scale")
        _positive_float(self.tail_constant, "tail_constant")

    def survival(self, amount: ArrayLike) -> np.ndarray:
        x = _nonnegative_array(amount, "amount")
        if self.survival_function is None:
            values = self.tail_constant * (1.0 + x / self.scale) ** (-self.tail_index)
            values = np.minimum(values, 1.0)
        else:
            values = np.asarray(self.survival_function(x), dtype=float)
            values = np.broadcast_to(values, x.shape)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("survival_function must return values in [0, 1]")
        return values


@dataclass(frozen=True)
class PolynomialPremiumGrowth:
    """Cumulative premium function `p(t) = coefficient * t**power`."""

    coefficient: float
    power: float

    def __post_init__(self) -> None:
        _positive_float(self.coefficient, "coefficient")
        _positive_float(self.power, "power")

    def cumulative(self, time: ArrayLike) -> np.ndarray:
        t = _nonnegative_array(time, "time")
        return self.coefficient * t**self.power

    def inverse(self, amount: ArrayLike) -> np.ndarray:
        x = _nonnegative_array(amount, "amount")
        return (x / self.coefficient) ** (1.0 / self.power)


@dataclass(frozen=True)
class InfiniteMeanRuinModel:
    """Infinite-mean risk process with Poisson arrivals and increasing premiums."""

    claim_arrival_rate: float
    tail: RegularlyVaryingTail
    premium: PolynomialPremiumGrowth
    name: str = "infinite-mean ruin model"

    def __post_init__(self) -> None:
        _positive_float(self.claim_arrival_rate, "claim_arrival_rate")
        condition = premium_power_condition(self.tail.tail_index, self.premium.power)
        if not condition.holds:
            warnings.warn(
                "premium.power should be greater than 1 / tail.tail_index for "
                "finite infinite-horizon asymptotics",
                UserWarning,
                stacklevel=2,
            )

    @property
    def tail_index(self) -> float:
        return self.tail.tail_index

    @property
    def premium_power(self) -> float:
        return self.premium.power

    def cumulative_premium(self, time: ArrayLike) -> np.ndarray:
        return self.premium.cumulative(time)

    def premium_inverse(self, amount: ArrayLike) -> np.ndarray:
        return self.premium.inverse(amount)

    def survival(self, amount: ArrayLike) -> np.ndarray:
        return self.tail.survival(amount)


@dataclass(frozen=True)
class InfiniteMeanRuinCurve:
    """Ruin asymptotic or one-big-jump integral over several initial capitals."""

    initial_capitals: np.ndarray
    probabilities: np.ndarray
    method: str
    tail_index: float
    premium_power: float


@dataclass(frozen=True)
class PremiumCalibrationResult:
    """Minimum polynomial premium coefficient for a target ruin probability."""

    target_probability: float
    premium_power: float
    tail_index: float
    initial_capitals: np.ndarray
    required_coefficients: np.ndarray
    required_coefficient: float
    binding_initial_capital: float
    achieved_asymptotic: float
    condition: PremiumPowerCondition


@dataclass(frozen=True)
class PremiumPowerGrid:
    """Calibration over a grid of polynomial premium powers."""

    premium_powers: np.ndarray
    required_coefficients: np.ndarray
    condition_holds: np.ndarray
    target_probability: float
    threshold: float
    binding_initial_capitals: np.ndarray


@dataclass(frozen=True)
class RegularVariationDiagnostic:
    """Tail-ratio diagnostic for regular variation."""

    thresholds: np.ndarray
    multipliers: np.ndarray
    ratios: np.ndarray
    targets: np.ndarray
    relative_errors: np.ndarray
    max_relative_error: float
    tail_index: float


def premium_power_condition(tail_index: float, premium_power: float) -> PremiumPowerCondition:
    """Check the KLR infinite-mean condition `beta > 1 / alpha`."""

    alpha = _positive_float(tail_index, "tail_index")
    if alpha > 1.0:
        raise ValueError("tail_index must be less than or equal to one")
    beta = _positive_float(premium_power, "premium_power")
    threshold = 1.0 / alpha
    return PremiumPowerCondition(
        tail_index=alpha,
        premium_power=beta,
        threshold=threshold,
        margin=beta - threshold,
    )


def infinite_mean_constant(tail_index: float, premium_power: float) -> float:
    """Return `int_0^inf (1 + t**beta)**(-alpha) dt`."""

    condition = premium_power_condition(tail_index, premium_power)
    if not condition.holds:
        raise ValueError("premium_power must be greater than 1 / tail_index")
    alpha = condition.tail_index
    beta = condition.premium_power
    return float(special.beta(1.0 / beta, alpha - 1.0 / beta) / beta)


def infinite_mean_one_big_jump_integral(
    model: InfiniteMeanRuinModel,
    initial_capital: float,
    *,
    epsabs: float = 1e-10,
) -> float:
    """Compute `lambda * int_0^inf Fbar(u + p(t)) dt` numerically."""

    u = _positive_float(initial_capital, "initial_capital")

    def integrand(time: float) -> float:
        return float(model.survival(u + model.cumulative_premium(time)))

    value, _ = integrate.quad(integrand, 0.0, math.inf, epsabs=epsabs, limit=200)
    return model.claim_arrival_rate * value


def infinite_mean_one_big_jump_asymptotic(
    model: InfiniteMeanRuinModel,
    initial_capital: float,
) -> float:
    """KLR asymptotic equivalent for an infinite-mean regularly varying model."""

    u = _positive_float(initial_capital, "initial_capital")
    constant = infinite_mean_constant(model.tail_index, model.premium_power)
    return (
        model.claim_arrival_rate
        * float(model.premium_inverse(u))
        * float(model.survival(u))
        * constant
    )


def infinite_mean_ruin_curve(
    model: InfiniteMeanRuinModel,
    initial_capitals: ArrayLike,
    *,
    method: str = "asymptotic",
    epsabs: float = 1e-10,
) -> InfiniteMeanRuinCurve:
    """Evaluate infinite-mean ruin approximations on an initial-capital grid."""

    capital = _positive_array(initial_capitals, "initial_capitals")
    if method == "asymptotic":
        values = np.array([infinite_mean_one_big_jump_asymptotic(model, u) for u in capital])
    elif method == "integral":
        values = np.array(
            [infinite_mean_one_big_jump_integral(model, u, epsabs=epsabs) for u in capital],
        )
    else:
        raise ValueError("method must be 'asymptotic' or 'integral'")
    return InfiniteMeanRuinCurve(
        initial_capitals=capital,
        probabilities=values,
        method=method,
        tail_index=model.tail_index,
        premium_power=model.premium_power,
    )


def calibrate_polynomial_premium_coefficient(
    tail: RegularlyVaryingTail,
    initial_capitals: ArrayLike,
    *,
    target_probability: float,
    claim_arrival_rate: float = 1.0,
    premium_power: float,
) -> PremiumCalibrationResult:
    """Choose the smallest `a` in `p(t)=a t**beta` meeting a target asymptotic."""

    target = _probability(target_probability, "target_probability")
    rate = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    capital = _positive_array(initial_capitals, "initial_capitals")
    condition = premium_power_condition(tail.tail_index, premium_power)
    if not condition.holds:
        raise ValueError("premium_power must be greater than 1 / tail_index")

    constant = infinite_mean_constant(tail.tail_index, premium_power)
    tail_values = tail.survival(capital)
    required = capital * (rate * tail_values * constant / target) ** premium_power
    index = int(np.argmax(required))
    coefficient = float(required[index])
    achieved = np.max(
        rate
        * (capital / coefficient) ** (1.0 / premium_power)
        * tail_values
        * constant,
    )
    return PremiumCalibrationResult(
        target_probability=target,
        premium_power=condition.premium_power,
        tail_index=tail.tail_index,
        initial_capitals=capital,
        required_coefficients=required,
        required_coefficient=coefficient,
        binding_initial_capital=float(capital[index]),
        achieved_asymptotic=float(achieved),
        condition=condition,
    )


def premium_power_calibration_grid(
    tail: RegularlyVaryingTail,
    initial_capitals: ArrayLike,
    premium_powers: ArrayLike,
    *,
    target_probability: float,
    claim_arrival_rate: float = 1.0,
) -> PremiumPowerGrid:
    """Calibrate polynomial premium coefficients across candidate powers."""

    powers = _positive_array(premium_powers, "premium_powers")
    capital = _positive_array(initial_capitals, "initial_capitals")
    target = _probability(target_probability, "target_probability")
    rate = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    tail_values = tail.survival(capital)
    coefficients = np.full_like(powers, np.nan, dtype=float)
    bindings = np.full_like(powers, np.nan, dtype=float)
    valid = np.zeros_like(powers, dtype=bool)
    for index, beta in enumerate(powers):
        condition = premium_power_condition(tail.tail_index, float(beta))
        valid[index] = condition.holds
        if condition.holds:
            constant = infinite_mean_constant(tail.tail_index, float(beta))
            required = capital * (rate * tail_values * constant / target) ** float(beta)
            binding_index = int(np.argmax(required))
            coefficients[index] = float(required[binding_index])
            bindings[index] = float(capital[binding_index])
    return PremiumPowerGrid(
        premium_powers=powers,
        required_coefficients=coefficients,
        condition_holds=valid,
        target_probability=target,
        threshold=1.0 / tail.tail_index,
        binding_initial_capitals=bindings,
    )


def regular_variation_tail_diagnostic(
    tail: RegularlyVaryingTail,
    thresholds: ArrayLike,
    multipliers: ArrayLike,
) -> RegularVariationDiagnostic:
    """Check `Fbar(k x) / Fbar(x) -> k**(-alpha)` on a finite grid."""

    x = _positive_array(thresholds, "thresholds")
    k = _positive_array(multipliers, "multipliers")
    base = tail.survival(x)
    if np.any(base <= 0.0):
        raise ValueError("tail survival must be positive on thresholds")
    ratios = np.empty((k.size, x.size), dtype=float)
    targets = k ** (-tail.tail_index)
    for index, multiplier in enumerate(k):
        ratios[index] = tail.survival(multiplier * x) / base
    relative = np.abs(ratios - targets[:, None]) / targets[:, None]
    return RegularVariationDiagnostic(
        thresholds=x,
        multipliers=k,
        ratios=ratios,
        targets=targets,
        relative_errors=relative,
        max_relative_error=float(np.max(relative)),
        tail_index=tail.tail_index,
    )


def pareto_infinite_mean_model(
    *,
    claim_arrival_rate: float,
    tail_index: float,
    pareto_scale: float = 1.0,
    premium_coefficient: float,
    premium_power: float,
) -> InfiniteMeanRuinModel:
    """Convenience constructor for the Pareto-II infinite-mean KLR example."""

    return InfiniteMeanRuinModel(
        claim_arrival_rate=claim_arrival_rate,
        tail=RegularlyVaryingTail(tail_index=tail_index, scale=pareto_scale),
        premium=PolynomialPremiumGrowth(
            coefficient=premium_coefficient,
            power=premium_power,
        ),
        name="Pareto-II infinite-mean ruin model",
    )


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _nonnegative_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _positive_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array
