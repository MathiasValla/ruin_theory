"""Loss moments, coverage transformations and lattice discretization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .distributions import ClaimDistribution


def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN")
    return array


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def raw_moment(distribution: ClaimDistribution, order: int = 1) -> float:
    """Return ``E[X**order]`` for supported severity distributions."""

    if not isinstance(distribution, ClaimDistribution):
        raise TypeError("distribution must be a ClaimDistribution")
    order = _nonnegative_integer(order, "order")
    return _raw_moment(distribution, order)


def _raw_moment(distribution: ClaimDistribution, order: int) -> float:
    name = distribution.name
    if name == "exponential":
        rate = float(distribution.metadata["rate"])
        return math.factorial(order) / rate**order
    if name == "gamma":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        return scale**order * math.gamma(shape + order) / math.gamma(shape)
    if name == "erlang":
        shape = int(distribution.metadata["shape"])
        rate = float(distribution.metadata["rate"])
        scale = 1.0 / rate
        return scale**order * math.gamma(shape + order) / math.gamma(shape)
    if name == "mixture_exponential":
        rates = np.asarray(distribution.metadata["rates"], dtype=float)
        weights = np.asarray(distribution.metadata["weights"], dtype=float)
        return float(np.sum(weights * math.factorial(order) / rates**order))
    if name == "deterministic":
        return float(distribution.metadata["value"]) ** order
    if name == "pareto":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        if shape <= order:
            return float("inf")
        return shape * scale**order / (shape - order)
    if name == "lognormal":
        meanlog = float(distribution.metadata["meanlog"])
        sdlog = float(distribution.metadata["sdlog"])
        return float(math.exp(order * meanlog + 0.5 * order**2 * sdlog**2))
    if name == "weibull":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        return scale**order * math.gamma(1.0 + order / shape)
    if name == "empirical":
        values = np.asarray(distribution.metadata["values"], dtype=float)
        return float(np.mean(values**order))
    raise NotImplementedError(f"raw moments are not implemented for {name}")


def limited_moment(
    distribution: ClaimDistribution,
    limit: ArrayLike,
    *,
    order: int = 1,
    epsabs: float = 1e-10,
) -> np.ndarray:
    """Return the limited moment ``E[min(X, limit)**order]``.

    This is the standard stop-loss/limited-expectation convention for
    non-negative severities. The implementation uses exact empirical evaluation
    when possible and otherwise the survival-integral identity
    ``E[min(X,d)^k] = integral_0^d k x^(k-1) S(x) dx``.
    """

    if not isinstance(distribution, ClaimDistribution):
        raise TypeError("distribution must be a ClaimDistribution")
    order = _nonnegative_integer(order, "order")
    epsabs = _positive_float(epsabs, "epsabs")
    limits = _as_float_array(limit, "limit")
    if np.any(~np.isfinite(limits)):
        raise ValueError("limit must contain only finite values")
    if np.any(limits < 0):
        raise ValueError("limit must be non-negative")
    if order == 0:
        return np.ones_like(limits, dtype=float)
    if distribution.name == "empirical":
        return empirical_limited_moment(
            np.asarray(distribution.metadata["values"], dtype=float),
            limits,
            order=order,
        )

    flat = limits.ravel()
    values = np.empty_like(flat, dtype=float)
    for idx, endpoint in enumerate(flat):
        if endpoint == 0:
            values[idx] = 0.0
            continue

        def integrand(x: float) -> float:
            return order * x ** (order - 1) * float(distribution.survival(x))

        values[idx] = integrate.quad(integrand, 0.0, float(endpoint), epsabs=epsabs)[0]
    return values.reshape(limits.shape)


def empirical_moment(data: ArrayLike, order: int = 1) -> float:
    """Empirical raw moment of non-negative observations."""

    order = _nonnegative_integer(order, "order")
    values = _as_float_array(data, "data")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("data must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if np.any(values < 0):
        raise ValueError("data must be non-negative")
    return float(np.mean(values**order))


def empirical_limited_moment(
    data: ArrayLike,
    limit: ArrayLike,
    *,
    order: int = 1,
) -> np.ndarray:
    """Empirical limited moment ``mean(min(data, limit)**order)``."""

    order = _nonnegative_integer(order, "order")
    values = _as_float_array(data, "data")
    limits = _as_float_array(limit, "limit")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("data must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if np.any(values < 0):
        raise ValueError("data must be non-negative")
    if np.any(~np.isfinite(limits)):
        raise ValueError("limit must contain only finite values")
    if np.any(limits < 0):
        raise ValueError("limit must be non-negative")
    if order == 0:
        return np.ones_like(limits, dtype=float)
    flat = limits.ravel()
    moments = np.array([np.mean(np.minimum(values, endpoint) ** order) for endpoint in flat])
    return moments.reshape(limits.shape)


def coverage_transform(
    distribution: ClaimDistribution | ArrayLike,
    *,
    deductible: float = 0.0,
    limit: float | None = None,
    coinsurance: float = 1.0,
    inflation: float = 1.0,
    franchise: bool = False,
    franchise_deductible: float | None = None,
    name: str | None = None,
) -> ClaimDistribution | np.ndarray | float:
    """Apply common insurance coverage changes to losses or a distribution.

    For raw loss arrays this returns transformed payments. For a
    :class:`ClaimDistribution`, this returns the corresponding payment
    distribution. Let ``Z = inflation * X``. For ordinary coverage the payment is
    ``coinsurance * min((Z - deductible)_+, limit)``. For franchise coverage it
    is ``coinsurance * min(Z, limit)`` when ``Z > deductible`` and zero otherwise.
    ``franchise_deductible`` is an alias for setting ``deductible`` and
    ``franchise=True``.
    """

    if franchise_deductible is not None:
        franchise_value = _nonnegative_float(franchise_deductible, "franchise_deductible")
        if float(deductible) != 0.0:
            raise ValueError("ordinary and franchise deductibles are mutually exclusive")
        deductible = franchise_value
        franchise = True

    deductible = _nonnegative_float(deductible, "deductible")
    coinsurance = _positive_float(coinsurance, "coinsurance")
    inflation = _positive_float(inflation, "inflation")
    if limit is not None:
        limit = _positive_float(limit, "limit")

    if not isinstance(distribution, ClaimDistribution):
        losses = _as_float_array(distribution, "losses")
        if np.any(~np.isfinite(losses)):
            raise ValueError("losses must contain only finite values")
        if np.any(losses < 0):
            raise ValueError("losses must be non-negative")
        ground_up = inflation * losses
        if franchise:
            retained = np.where(ground_up > deductible, ground_up, 0.0)
        else:
            retained = np.maximum(ground_up - deductible, 0.0)
        if limit is not None:
            retained = np.minimum(retained, limit)
        payments = coinsurance * retained
        return float(payments.item()) if payments.ndim == 0 else payments

    max_payment = np.inf if limit is None else coinsurance * limit

    def transformed_sample(rng: np.random.Generator, n: int) -> np.ndarray:
        ground_up = inflation * distribution.sample(n, rng=rng)
        if franchise:
            retained = np.where(ground_up > deductible, ground_up, 0.0)
        else:
            retained = np.maximum(ground_up - deductible, 0.0)
        if limit is not None:
            retained = np.minimum(retained, limit)
        return coinsurance * retained

    def survival(y: ArrayLike) -> np.ndarray:
        values = _as_float_array(y, "y")
        result = np.ones_like(values, dtype=float)
        nonnegative = values >= 0
        result[~nonnegative] = 1.0
        active = nonnegative.copy()
        if np.isfinite(max_payment):
            result[values >= max_payment] = 0.0
            active &= values < max_payment
        if np.any(active):
            y_active = values[active] / coinsurance
            if franchise:
                threshold = np.maximum(deductible, y_active)
            else:
                threshold = deductible + y_active
            result[active] = distribution.survival(threshold / inflation)
        return result

    def cdf(y: ArrayLike) -> np.ndarray:
        values = _as_float_array(y, "y")
        return np.where(values < 0, 0.0, 1.0 - survival(values))

    def payment_moment(order: int) -> float:
        if max_payment == 0:
            return 0.0

        def integrand(y: float) -> float:
            return order * y ** (order - 1) * float(survival(y))

        upper = np.inf if not np.isfinite(max_payment) else max_payment
        value = integrate.quad(integrand, 0.0, upper, epsabs=1e-9)[0]
        return float(value)

    mean = payment_moment(1)
    second = payment_moment(2)
    variance = max(second - mean**2, 0.0) if np.isfinite(second) else np.inf
    label = name or f"coverage_{distribution.name}"
    return ClaimDistribution(
        name=label,
        mean_value=mean,
        variance_value=variance,
        sampler=transformed_sample,
        cdf_function=cdf,
        survival_function=survival,
        metadata={
            "base": distribution.name,
            "deductible": deductible,
            "limit": limit,
            "coinsurance": coinsurance,
            "inflation": inflation,
            "franchise": bool(franchise),
        },
    )


@dataclass(frozen=True)
class DiscretizedDistribution:
    """Arithmetic severity PMF returned by :func:`discretize`."""

    support: np.ndarray
    pmf: np.ndarray
    step: float
    method: str

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=float)
        pmf = np.asarray(self.pmf, dtype=float)
        if support.ndim != 1 or pmf.ndim != 1 or support.size != pmf.size:
            raise ValueError("support and pmf must be one-dimensional arrays of equal length")
        if support.size == 0:
            raise ValueError("support and pmf must not be empty")
        if np.any(~np.isfinite(support)) or np.any(np.diff(support) <= 0):
            raise ValueError("support must be finite and strictly increasing")
        if np.any(~np.isfinite(pmf)) or np.any(pmf < -1e-12):
            raise ValueError("pmf must contain finite non-negative probabilities")
        clean = np.maximum(pmf, 0.0)
        if clean.sum() > 1.0 + 1e-8:
            raise ValueError("pmf must sum to at most one")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "pmf", clean)
        object.__setattr__(self, "step", _positive_float(self.step, "step"))

    @property
    def total_mass(self) -> float:
        return float(self.pmf.sum())

    @property
    def mean(self) -> float:
        return float(np.dot(self.support, self.pmf))

    def cdf(self, x: ArrayLike) -> np.ndarray:
        values = _as_float_array(x, "x")
        cumulative = np.cumsum(self.pmf)
        idx = np.searchsorted(self.support, values, side="right") - 1
        result = np.zeros_like(values, dtype=float)
        valid = idx >= 0
        result[valid] = cumulative[np.minimum(idx[valid], cumulative.size - 1)]
        return result


def _discretization_grid(from_: float, to: float, step: float) -> np.ndarray:
    start = _nonnegative_float(from_, "from_")
    end = _positive_float(to, "to")
    span = _positive_float(step, "step")
    if end <= start:
        raise ValueError("to must be greater than from_")
    count = round((end - start) / span)
    if not np.isclose(start + count * span, end, rtol=1e-10, atol=1e-12):
        raise ValueError("to - from_ must be an integer multiple of step")
    return start + span * np.arange(count + 1)


def discretize(
    distribution: ClaimDistribution,
    *,
    from_: float = 0.0,
    to: float,
    step: float,
    method: str = "upper",
) -> DiscretizedDistribution:
    """Discretize a severity distribution on an arithmetic grid.

    Available methods are ``upper`` (forward difference), ``lower`` (backward
    difference), ``rounding`` (midpoint), and ``unbiased`` (local first-moment
    matching).
    """

    if not isinstance(distribution, ClaimDistribution):
        raise TypeError("distribution must be a ClaimDistribution")
    grid = _discretization_grid(from_, to, step)
    method_key = method.lower()
    if method_key not in {"upper", "lower", "rounding", "unbiased"}:
        raise ValueError("method must be 'upper', 'lower', 'rounding', or 'unbiased'")

    cdf = distribution.cdf
    if method_key == "upper":
        support = grid[:-1]
        pmf = cdf(support + step) - cdf(support)
    elif method_key == "lower":
        support = grid
        pmf = np.empty_like(support)
        pmf[0] = cdf(support[0])
        pmf[1:] = cdf(support[1:]) - cdf(support[:-1])
    elif method_key == "rounding":
        support = grid[:-1]
        pmf = np.empty_like(support)
        pmf[0] = cdf(support[0] + 0.5 * step)
        if support.size > 1:
            centers = support[1:]
            pmf[1:] = cdf(centers + 0.5 * step) - cdf(centers - 0.5 * step)
    else:
        support = grid
        lev = limited_moment(distribution, support, order=1)
        pmf = np.empty_like(support)
        pmf[0] = (lev[0] - limited_moment(distribution, support[0] + step, order=1)) / step
        pmf[0] += 1.0 - float(cdf(support[0]))
        if support.size > 2:
            pmf[1:-1] = (
                2.0 * lev[1:-1]
                - limited_moment(distribution, support[1:-1] - step, order=1)
                - limited_moment(distribution, support[1:-1] + step, order=1)
            ) / step
        pmf[-1] = (lev[-1] - lev[-2]) / step - 1.0 + float(cdf(support[-1]))

    return DiscretizedDistribution(support=support, pmf=pmf, step=step, method=method_key)
