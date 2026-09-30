"""Orthogonal-polynomial approximations for ultimate ruin probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
from numpy.typing import ArrayLike

from .distributions import ClaimDistribution
from .formulas import _as_array, _primary_claim_formula_check, adjustment_coefficient
from .losses import _raw_moment
from .models import CramerLundbergProcess


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


def _nonnegative_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _scaled_raw_moments(
    distribution: ClaimDistribution,
    order: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    moments = np.empty(order + 1, dtype=float)
    moments[0] = 1.0
    for degree in range(1, order + 1):
        moments[degree] = scale**degree * _raw_moment(distribution, degree)
        if not np.isfinite(moments[degree]):
            raise ValueError(f"finite raw moment of order {degree} is required")
    return moments


def compound_geometric_moments_from_severity(
    distribution: ClaimDistribution,
    rho: float,
    order: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Moments of the Pollaczek-Khinchine compound-geometric maximum.

    The summands are the equilibrium severity law of ``scale * distribution``.
    The count has ``P(N=n) = (1-rho) rho**n``.
    """

    order = _nonnegative_int(order, "order")
    rho = float(rho)
    if not np.isfinite(rho) or not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    scale = _nonnegative_float(scale, "scale")
    if scale == 0.0:
        moments = np.zeros(order + 1, dtype=float)
        moments[0] = 1.0
        return moments

    raw = _scaled_raw_moments(distribution, order + 1, scale=scale)
    mean = raw[1]
    if mean <= 0.0:
        moments = np.zeros(order + 1, dtype=float)
        moments[0] = 1.0
        return moments

    equilibrium = np.empty(order + 1, dtype=float)
    equilibrium[0] = 1.0
    for degree in range(1, order + 1):
        equilibrium[degree] = raw[degree + 1] / ((degree + 1.0) * mean)

    denominator = np.empty(order + 1, dtype=float)
    denominator[0] = 1.0 - rho
    for degree in range(1, order + 1):
        denominator[degree] = -rho * equilibrium[degree] / math.factorial(degree)

    coefficients = np.zeros(order + 1, dtype=float)
    coefficients[0] = 1.0
    for degree in range(1, order + 1):
        total = 0.0
        for split in range(1, degree + 1):
            total += denominator[split] * coefficients[degree - split]
        coefficients[degree] = -total / denominator[0]

    moments = np.array(
        [math.factorial(degree) * coefficients[degree] for degree in range(order + 1)],
        dtype=float,
    )
    moments[0] = 1.0
    return moments


@dataclass(frozen=True)
class LaguerreRuinExpansion:
    """Reusable Laguerre expansion for ultimate Cramer-Lundberg ruin."""

    coefficients: np.ndarray
    xi: float
    rho: float
    order: int
    adjustment: float | None = None

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("coefficients must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients.copy())
        object.__setattr__(self, "xi", _positive_float(self.xi, "xi"))
        rho = float(self.rho)
        if not np.isfinite(rho) or not 0.0 <= rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        object.__setattr__(self, "rho", rho)
        order = _nonnegative_int(self.order, "order")
        if order != coefficients.size - 1:
            raise ValueError("order must equal len(coefficients)-1")
        object.__setattr__(self, "order", order)
        if self.adjustment is not None:
            object.__setattr__(self, "adjustment", _positive_float(self.adjustment, "adjustment"))

    def tail_integrals(self, u: ArrayLike) -> np.ndarray:
        """Return ``int_u^inf L_n(xi*x) xi exp(-xi*x) dx`` for all ``n``."""

        surplus = _as_array(u)
        z = self.xi * surplus.ravel()
        integrals = np.zeros((self.order + 1, z.size), dtype=float)
        finite = np.isfinite(z)
        x = z[finite]
        previous = np.exp(-x)
        integrals[0, finite] = previous
        if self.order:
            current = -x * previous
            integrals[1, finite] = current
            # Integral = exp(-x) L_n^(-1)(x), avoiding alternating binomial sums.
            for degree in range(2, self.order + 1):
                following = ((2 * degree - 2 - x) * current - (degree - 2) * previous) / degree
                integrals[degree, finite] = following
                previous, current = current, following
        return integrals.reshape((self.order + 1,) + surplus.shape)

    def evaluate(self, u: ArrayLike, *, clip: bool = True) -> np.ndarray:
        """Evaluate the truncated ultimate ruin approximation."""

        surplus = _as_array(u)
        integrals = self.tail_integrals(surplus).reshape(self.order + 1, -1)
        values = self.coefficients @ integrals
        values = values.reshape(surplus.shape)
        return np.clip(values, 0.0, 1.0) if clip else values

    __call__ = evaluate


def fit_ultimate_ruin_polynomial_expansion(
    model: CramerLundbergProcess,
    order: int,
    *,
    xi: float | None = None,
    validate_xi: bool = True,
) -> LaguerreRuinExpansion:
    """Fit the Laguerre ultimate-ruin expansion for a light-tailed CL model."""

    _primary_claim_formula_check(model)
    order = _nonnegative_int(order, "order")
    if model.premium_rate <= 0.0:
        raise ValueError("premium_rate must be positive")
    scale = float(model.prevention.severity_multiplier)
    mean = scale * model.claim_distribution.mean()
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("finite positive claim mean is required")
    rho = model.claim_arrival_rate * mean / model.premium_rate
    if not 0.0 <= rho < 1.0:
        raise ValueError("net profit condition is required: rho must lie in [0, 1)")

    adjustment = adjustment_coefficient(model)
    xi_value = adjustment if xi is None else _positive_float(xi, "xi")
    if validate_xi and xi_value >= 2.0 * adjustment:
        raise ValueError("the Laguerre integrability condition requires xi < 2 * adjustment")

    maximum_moments = compound_geometric_moments_from_severity(
        model.claim_distribution,
        rho,
        order,
        scale=scale,
    )
    continuous_moments = maximum_moments.copy()
    continuous_moments[0] = rho

    coefficients = np.empty(order + 1, dtype=float)
    for degree in range(order + 1):
        total = 0.0
        for power in range(degree + 1):
            total += (
                math.comb(degree, power)
                * ((-xi_value) ** power)
                * continuous_moments[power]
                / math.factorial(power)
            )
        coefficients[degree] = total

    return LaguerreRuinExpansion(
        coefficients=coefficients,
        xi=xi_value,
        rho=rho,
        order=order,
        adjustment=adjustment,
    )


def ultimate_ruin_polynomial_expansion(
    model: CramerLundbergProcess,
    u: ArrayLike,
    *,
    order: int = 20,
    xi: float | None = None,
    validate_xi: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """Evaluate the Laguerre ultimate-ruin approximation."""

    expansion = fit_ultimate_ruin_polynomial_expansion(
        model,
        order,
        xi=xi,
        validate_xi=validate_xi,
    )
    return expansion.evaluate(u, clip=clip)
