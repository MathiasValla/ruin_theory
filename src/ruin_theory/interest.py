"""Interest-force and double-barrier ruin identities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import special


NonRuinFunction = Callable[[np.ndarray], ArrayLike]


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


def _nonnegative_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _maybe_scalar(values: np.ndarray, original: ArrayLike) -> float | np.ndarray:
    return float(values) if np.asarray(original).ndim == 0 else values


def _log_upper_gamma(shape: float, x: np.ndarray | float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return special.gammaln(shape) + np.log(special.gammaincc(shape, x))


def _function_values(function: NonRuinFunction, points: np.ndarray, name: str) -> np.ndarray:
    if not callable(function):
        raise TypeError(f"{name} must be callable")
    try:
        values = np.asarray(function(points), dtype=float)
    except (TypeError, ValueError, AttributeError):
        values = np.asarray([function(float(point)) for point in points.ravel()], dtype=float)
        values = values.reshape(points.shape)
    if values.shape != points.shape:
        values = np.asarray([function(float(point)) for point in points.ravel()], dtype=float)
        values = values.reshape(points.shape)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must return finite positive values")
    return values


def ultimate_ruin_exponential_interest_force(
    initial_capital: ArrayLike,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
) -> float | np.ndarray:
    """Ultimate ruin probability with exponential claims and constant interest force.

    The surplus follows ``dR_t = c dt - dS_t + delta R_t dt`` with Poisson
    claim intensity ``lambda`` and exponential claim rate ``mu``. For
    ``delta > 0`` this is Segerdahl's incomplete-gamma formula; for
    ``delta = 0`` it reduces to the classical Cramer-Lundberg expression.
    """

    surplus = _nonnegative_array(initial_capital, "initial_capital")
    premium = _positive_float(premium_rate, "premium_rate")
    arrival = _nonnegative_float(claim_arrival_rate, "claim_arrival_rate")
    severity_rate = _positive_float(claim_rate, "claim_rate")
    force = _nonnegative_float(interest_force, "interest_force")

    if arrival == 0.0:
        return _maybe_scalar(np.zeros_like(surplus, dtype=float), initial_capital)

    if force == 0.0:
        rho = arrival / (premium * severity_rate)
        if rho >= 1.0:
            return _maybe_scalar(np.ones_like(surplus, dtype=float), initial_capital)
        exponent = severity_rate - arrival / premium
        ruin = rho * np.exp(-exponent * surplus)
        return _maybe_scalar(np.clip(ruin, 0.0, 1.0), initial_capital)

    shape = arrival / force
    boundary = premium * severity_rate / force
    x = boundary + severity_rate * surplus
    log_upper = _log_upper_gamma(shape, x)
    log_upper_zero = float(_log_upper_gamma(shape, boundary))
    log_claim_term = (
        np.log(arrival)
        + (shape - 1.0) * np.log(force)
        + log_upper
    )
    log_claim_term_zero = np.log(arrival) + (shape - 1.0) * np.log(force) + log_upper_zero
    log_boundary_term = shape * (np.log(severity_rate) + np.log(premium)) - boundary
    log_denominator = np.logaddexp(log_boundary_term, log_claim_term_zero)
    ruin = np.exp(log_claim_term - log_denominator)
    return _maybe_scalar(np.clip(ruin, 0.0, 1.0), initial_capital)


def non_ruin_exponential_interest_force(
    initial_capital: ArrayLike,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
) -> float | np.ndarray:
    """Ultimate non-ruin probability for exponential claims under interest force."""

    ruin = ultimate_ruin_exponential_interest_force(
        initial_capital,
        premium_rate=premium_rate,
        claim_arrival_rate=claim_arrival_rate,
        claim_rate=claim_rate,
        interest_force=interest_force,
    )
    return 1.0 - ruin


def win_first_probability_from_non_ruin(
    initial_capital: ArrayLike,
    gain: ArrayLike,
    non_ruin_function: NonRuinFunction,
) -> float | np.ndarray:
    """Compute ``WF(u,v)=phi(u)/phi(u+v)`` from an ultimate non-ruin function."""

    surplus, target_gain = np.broadcast_arrays(
        _nonnegative_array(initial_capital, "initial_capital"),
        _nonnegative_array(gain, "gain"),
    )
    lower = _function_values(non_ruin_function, surplus, "non_ruin_function")
    upper = _function_values(non_ruin_function, surplus + target_gain, "non_ruin_function")
    ratio = lower / upper
    if np.any(ratio < -1e-10) or np.any(ratio > 1.0 + 1e-10):
        raise ValueError("non_ruin_function must be non-decreasing on the evaluated points")
    return _maybe_scalar(np.clip(ratio, 0.0, 1.0), ratio)


def win_first_probability_exponential_interest_force(
    initial_capital: ArrayLike,
    gain: ArrayLike,
    *,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
) -> float | np.ndarray:
    """Win-first probability for exponential claims under constant interest force."""

    def non_ruin(points: np.ndarray) -> np.ndarray:
        return non_ruin_exponential_interest_force(
            points,
            premium_rate=premium_rate,
            claim_arrival_rate=claim_arrival_rate,
            claim_rate=claim_rate,
            interest_force=interest_force,
        )

    return win_first_probability_from_non_ruin(initial_capital, gain, non_ruin)


def maximum_before_default_survival(
    x: ArrayLike,
    non_ruin_function: NonRuinFunction,
) -> float | np.ndarray:
    """Survival function ``S(x)=P(theta >= x)=phi(0)/phi(x)``."""

    levels = _nonnegative_array(x, "x")
    zero = np.zeros_like(levels, dtype=float)
    baseline = _function_values(non_ruin_function, zero, "non_ruin_function")
    values = _function_values(non_ruin_function, levels, "non_ruin_function")
    survival = baseline / values
    if np.any(survival < -1e-10) or np.any(survival > 1.0 + 1e-10):
        raise ValueError("non_ruin_function must be non-decreasing on the evaluated points")
    return _maybe_scalar(np.clip(survival, 0.0, 1.0), x)


def maximum_before_default_hazard(
    x: ArrayLike,
    non_ruin_function: NonRuinFunction,
    *,
    step: float | None = None,
) -> float | np.ndarray:
    """Numerical hazard rate of the maximum-before-default distribution."""

    levels = _nonnegative_array(x, "x")
    if step is None:
        h = np.maximum(1e-5, np.sqrt(np.finfo(float).eps) * (1.0 + levels))
    else:
        h = np.full_like(levels, _positive_float(step, "step"))

    def log_phi(points: np.ndarray) -> np.ndarray:
        return np.log(_function_values(non_ruin_function, points, "non_ruin_function"))

    lower = np.maximum(0.0, levels - h)
    upper = levels + h
    denominator = upper - lower
    hazard = (log_phi(upper) - log_phi(lower)) / denominator
    return _maybe_scalar(np.maximum(hazard, 0.0), x)


def win_first_time_bound(
    initial_capital: ArrayLike,
    gain: ArrayLike,
    *,
    premium_rate: float,
    interest_force: float = 0.0,
) -> float | np.ndarray:
    """Minimal deterministic time needed to earn ``gain`` without claims."""

    surplus, target_gain = np.broadcast_arrays(
        _nonnegative_array(initial_capital, "initial_capital"),
        _nonnegative_array(gain, "gain"),
    )
    premium = _positive_float(premium_rate, "premium_rate")
    force = _nonnegative_float(interest_force, "interest_force")
    if force == 0.0:
        bound = target_gain / premium
    else:
        bound = np.log1p(target_gain / (surplus + premium / force)) / force
    return _maybe_scalar(bound, bound)
