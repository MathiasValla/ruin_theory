"""Claim severity distributions used by the risk-process layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, stats


ArrayFunction = Callable[[ArrayLike], np.ndarray]
RandomFunction = Callable[[np.random.Generator, int], np.ndarray]
MomentFunction = Callable[[float], float]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _sample_size(n: int) -> int:
    try:
        size = operator.index(n)
    except TypeError as exc:
        raise TypeError("n must be an integer") from exc
    if size < 0:
        raise ValueError("n must be non-negative")
    return size


def _finite_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = _as_array(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _phase_type_inputs(
    initial_probabilities: ArrayLike,
    subgenerator: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = _finite_1d(initial_probabilities, "initial_probabilities")
    if np.any(initial < 0.0):
        raise ValueError("initial_probabilities must be non-negative")
    if not np.isclose(float(initial.sum()), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("initial_probabilities must sum to one")

    matrix = np.asarray(subgenerator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("subgenerator must be a non-empty square matrix")
    if matrix.shape[0] != initial.size:
        raise ValueError("initial_probabilities and subgenerator dimensions must match")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("subgenerator must contain only finite values")

    diagonal = np.diag(matrix)
    if np.any(diagonal >= 0.0):
        raise ValueError("subgenerator diagonal entries must be negative")
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal < -1e-12):
        raise ValueError("subgenerator off-diagonal entries must be non-negative")

    row_sums = matrix.sum(axis=1)
    if np.any(row_sums > 1e-10):
        raise ValueError("subgenerator row sums must be non-positive")
    exit_rates = -row_sums
    if np.any(exit_rates < -1e-10):
        raise ValueError("subgenerator exit rates must be non-negative")

    eigenvalues = linalg.eigvals(matrix)
    if np.max(np.real(eigenvalues)) >= -1e-12:
        raise ValueError("subgenerator must be transient with negative spectral abscissa")
    return initial.copy(), matrix.copy(), np.maximum(exit_rates, 0.0)


def _sample_phase_type(
    initial: np.ndarray,
    subgenerator: np.ndarray,
    exit_rates: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    size = _sample_size(n)
    samples = np.zeros(size, dtype=float)
    rates = -np.diag(subgenerator)
    transition_rates = np.maximum(subgenerator.copy(), 0.0)
    np.fill_diagonal(transition_rates, 0.0)

    for sample_index in range(size):
        phase = int(rng.choice(initial.size, p=initial))
        elapsed = 0.0
        while True:
            rate = rates[phase]
            elapsed += float(rng.exponential(1.0 / rate))
            threshold = float(rng.random() * rate)
            cumulative = 0.0
            next_phase = -1
            for candidate, transition_rate in enumerate(transition_rates[phase]):
                cumulative += float(transition_rate)
                if threshold < cumulative:
                    next_phase = candidate
                    break
            if next_phase < 0:
                samples[sample_index] = elapsed
                break
            phase = next_phase
    return samples


@dataclass(frozen=True)
class ClaimDistribution:
    """Small distribution wrapper with the operations ruin formulas need.

    Parameters are intentionally explicit instead of relying on duck-typing a
    frozen SciPy distribution everywhere. This keeps custom distributions,
    empirical laws and mixtures on the same footing.
    """

    name: str
    mean_value: float
    variance_value: float | None
    sampler: RandomFunction
    cdf_function: ArrayFunction | None = None
    survival_function: ArrayFunction | None = None
    pdf_function: ArrayFunction | None = None
    mgf_function: MomentFunction | None = None
    laplace_function: MomentFunction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not callable(self.sampler):
            raise TypeError("sampler must be callable")
        mean = float(self.mean_value)
        if np.isnan(mean) or mean < 0:
            raise ValueError("mean_value must be non-negative or infinity")
        object.__setattr__(self, "mean_value", mean)
        if self.variance_value is not None:
            variance = float(self.variance_value)
            if np.isnan(variance) or variance < 0:
                raise ValueError("variance_value must be non-negative, infinity, or None")
            object.__setattr__(self, "variance_value", variance)
        for attr in (
            "cdf_function",
            "survival_function",
            "pdf_function",
            "mgf_function",
            "laplace_function",
        ):
            function = getattr(self, attr)
            if function is not None and not callable(function):
                raise TypeError(f"{attr} must be callable or None")

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` non-negative claim sizes."""

        size = _sample_size(n)
        generator = np.random.default_rng() if rng is None else rng
        values = np.asarray(self.sampler(generator, size), dtype=float).reshape(-1)
        if values.size != size:
            raise ValueError(f"{self.name} sampler returned {values.size} values for n={size}")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{self.name} produced non-finite claim sizes")
        if np.any(values < 0):
            raise ValueError(f"{self.name} produced negative claim sizes")
        return values

    def mean(self) -> float:
        return float(self.mean_value)

    def variance(self) -> float | None:
        return None if self.variance_value is None else float(self.variance_value)

    def second_moment(self) -> float | None:
        if self.variance_value is None:
            return None
        return float(self.variance_value + self.mean_value**2)

    def cdf(self, x: ArrayLike) -> np.ndarray:
        if self.cdf_function is None:
            raise NotImplementedError(f"{self.name} does not provide a CDF")
        return self.cdf_function(x)

    def survival(self, x: ArrayLike) -> np.ndarray:
        if self.survival_function is not None:
            return self.survival_function(x)
        return 1.0 - self.cdf(x)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        if self.pdf_function is None:
            raise NotImplementedError(f"{self.name} does not provide a density")
        return self.pdf_function(x)

    def mgf(self, t: float) -> float:
        if self.mgf_function is None:
            raise NotImplementedError(f"{self.name} does not provide a finite MGF")
        argument = _finite_float(t, "t")
        value = float(self.mgf_function(argument))
        if np.isnan(value):
            raise ValueError(f"{self.name} MGF returned NaN")
        return value

    def laplace(self, s: float) -> float:
        argument = _nonnegative_float(s, "s")
        if self.laplace_function is not None:
            value = float(self.laplace_function(argument))
            if np.isnan(value):
                raise ValueError(f"{self.name} Laplace transform returned NaN")
            return value
        return self.mgf(-argument)


def exponential(rate: float) -> ClaimDistribution:
    """Exponential claim sizes with density ``rate * exp(-rate*x)``."""

    rate = _positive_float(rate, "rate")
    frozen = stats.expon(scale=1.0 / rate)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.exponential(scale=1.0 / rate, size=n)

    return ClaimDistribution(
        name="exponential",
        mean_value=1.0 / rate,
        variance_value=1.0 / rate**2,
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=lambda t: rate / (rate - t) if t < rate else np.inf,
        laplace_function=lambda s: rate / (rate + s),
        metadata={"rate": float(rate)},
    )


def gamma(shape: float, rate: float | None = None, scale: float | None = None) -> ClaimDistribution:
    """Gamma claim sizes; pass either ``rate`` or ``scale``."""

    shape = _positive_float(shape, "shape")
    if (rate is None) == (scale is None):
        raise ValueError("pass exactly one of rate or scale")
    if rate is not None:
        scale_value = 1.0 / _positive_float(rate, "rate")
    else:
        scale_value = _positive_float(scale, "scale")
    frozen = stats.gamma(a=shape, scale=scale_value)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.gamma(shape=shape, scale=scale_value, size=n)

    return ClaimDistribution(
        name="gamma",
        mean_value=shape * scale_value,
        variance_value=shape * scale_value**2,
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=lambda t: (1.0 - scale_value * t) ** (-shape)
        if t < 1.0 / scale_value
        else np.inf,
        laplace_function=lambda s: (1.0 + scale_value * s) ** (-shape),
        metadata={"shape": float(shape), "scale": scale_value},
    )


def erlang(shape: int, rate: float) -> ClaimDistribution:
    """Erlang claim sizes, a gamma law with integer shape."""

    shape_value = float(shape)
    if not np.isfinite(shape_value) or not shape_value.is_integer() or shape_value <= 0:
        raise ValueError("shape must be a positive integer")
    rate_value = _positive_float(rate, "rate")
    distribution = gamma(shape=int(shape_value), rate=rate_value)
    return ClaimDistribution(
        **{
            **distribution.__dict__,
            "name": "erlang",
            "metadata": {"shape": int(shape_value), "rate": rate_value},
        }
    )


def deterministic(value: float) -> ClaimDistribution:
    """Degenerate claim size equal to ``value``."""

    value = _nonnegative_float(value, "value")

    def sampler(_: np.random.Generator, n: int) -> np.ndarray:
        return np.full(n, value, dtype=float)

    return ClaimDistribution(
        name="deterministic",
        mean_value=float(value),
        variance_value=0.0,
        sampler=sampler,
        cdf_function=lambda x: (_as_array(x) >= value).astype(float),
        survival_function=lambda x: (_as_array(x) < value).astype(float),
        mgf_function=lambda t: float(np.exp(t * value)),
        laplace_function=lambda s: float(np.exp(-s * value)),
        metadata={"value": float(value)},
    )


def mixture_exponential(rates: ArrayLike, weights: ArrayLike | None = None) -> ClaimDistribution:
    """Hyperexponential mixture with component rates and probabilities."""

    rate_array = _finite_1d(rates, "rates")
    if np.any(rate_array <= 0):
        raise ValueError("all rates must be positive")
    if weights is None:
        weight_array = np.full(rate_array.size, 1.0 / rate_array.size)
    else:
        weight_array = _finite_1d(weights, "weights")
    if rate_array.shape != weight_array.shape:
        raise ValueError("rates and weights must have the same shape")
    total_weight = float(weight_array.sum())
    if np.any(weight_array < 0) or not np.isclose(total_weight, 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    weight_array = weight_array / total_weight

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.choice(rate_array.size, size=n, p=weight_array)
        return rng.exponential(scale=1.0 / rate_array[idx])

    def survival(x: ArrayLike) -> np.ndarray:
        arr = _as_array(x)
        clipped = np.maximum(arr, 0.0)
        values = np.sum(
            weight_array[:, None] * np.exp(-rate_array[:, None] * clipped.ravel()),
            axis=0,
        ).reshape(arr.shape)
        return np.where(arr < 0, 1.0, values)

    def pdf(x: ArrayLike) -> np.ndarray:
        arr = _as_array(x)
        flat = arr.ravel()
        density = np.zeros_like(flat, dtype=float)
        mask = flat >= 0
        if np.any(mask):
            density[mask] = np.sum(
                weight_array[:, None]
                * rate_array[:, None]
                * np.exp(-rate_array[:, None] * flat[mask]),
                axis=0,
            )
        return density.reshape(arr.shape)

    mean_value = float(np.sum(weight_array / rate_array))
    second = float(np.sum(weight_array * 2.0 / rate_array**2))
    return ClaimDistribution(
        name="mixture_exponential",
        mean_value=mean_value,
        variance_value=second - mean_value**2,
        sampler=sampler,
        cdf_function=lambda x: 1.0 - survival(x),
        survival_function=survival,
        pdf_function=pdf,
        mgf_function=lambda t: float(np.sum(weight_array * rate_array / (rate_array - t)))
        if t < float(rate_array.min())
        else np.inf,
        laplace_function=lambda s: float(np.sum(weight_array * rate_array / (rate_array + s))),
        metadata={"rates": rate_array.copy(), "weights": weight_array.copy()},
    )


def phase_type(initial_probabilities: ArrayLike, subgenerator: ArrayLike) -> ClaimDistribution:
    """Continuous phase-type severity distribution.

    ``initial_probabilities`` is the initial probability row vector and
    ``subgenerator`` is the transient Markov-chain generator ``T``. The density
    is ``alpha exp(T x) t`` with exit vector ``t = -T 1``.
    """

    initial, matrix, exit_rates = _phase_type_inputs(initial_probabilities, subgenerator)
    dimension = initial.size
    ones = np.ones(dimension, dtype=float)
    minus_generator = -matrix
    first = linalg.solve(minus_generator, ones, assume_a="gen")
    mean_value = float(initial @ first)
    second = float(2.0 * initial @ linalg.solve(minus_generator, first, assume_a="gen"))
    variance = max(second - mean_value**2, 0.0)
    spectral_bound = float(-np.max(np.real(linalg.eigvals(matrix))))

    def survival(x: ArrayLike) -> np.ndarray:
        values = _as_array(x)
        flat = values.ravel()
        result = np.zeros_like(flat, dtype=float)
        result[flat < 0.0] = 1.0
        finite = np.isfinite(flat) & (flat >= 0.0)
        for index in np.where(finite)[0]:
            result[index] = float(initial @ linalg.expm(matrix * flat[index]) @ ones)
        return np.clip(result.reshape(values.shape), 0.0, 1.0)

    def cdf(x: ArrayLike) -> np.ndarray:
        values = _as_array(x)
        return np.where(values < 0.0, 0.0, 1.0 - survival(values))

    def pdf(x: ArrayLike) -> np.ndarray:
        values = _as_array(x)
        flat = values.ravel()
        result = np.zeros_like(flat, dtype=float)
        finite = np.isfinite(flat) & (flat >= 0.0)
        for index in np.where(finite)[0]:
            result[index] = float(initial @ linalg.expm(matrix * flat[index]) @ exit_rates)
        return np.maximum(result.reshape(values.shape), 0.0)

    def mgf(t: float) -> float:
        if t >= spectral_bound:
            return np.inf
        system = -t * np.eye(dimension) - matrix
        return float(initial @ linalg.solve(system, exit_rates, assume_a="gen"))

    def laplace(s: float) -> float:
        system = s * np.eye(dimension) - matrix
        return float(initial @ linalg.solve(system, exit_rates, assume_a="gen"))

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return _sample_phase_type(initial, matrix, exit_rates, rng, n)

    return ClaimDistribution(
        name="phase_type",
        mean_value=mean_value,
        variance_value=variance,
        sampler=sampler,
        cdf_function=cdf,
        survival_function=survival,
        pdf_function=pdf,
        mgf_function=mgf,
        laplace_function=laplace,
        metadata={
            "initial_probabilities": initial.copy(),
            "subgenerator": matrix.copy(),
            "exit_rates": exit_rates.copy(),
            "spectral_bound": spectral_bound,
        },
    )


def pareto(shape: float, scale: float) -> ClaimDistribution:
    """Pareto type I severity with support ``[scale, infinity)``."""

    shape = _positive_float(shape, "shape")
    scale = _positive_float(scale, "scale")
    frozen = stats.pareto(b=shape, scale=scale)
    mean_value = np.inf if shape <= 1 else shape * scale / (shape - 1)
    variance = None if shape <= 2 else shape * scale**2 / ((shape - 1) ** 2 * (shape - 2))

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return scale * (1.0 + rng.pareto(shape, size=n))

    return ClaimDistribution(
        name="pareto",
        mean_value=float(mean_value),
        variance_value=None if variance is None else float(variance),
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=None,
        laplace_function=None,
        metadata={"shape": float(shape), "scale": float(scale)},
    )


def lomax(shape: float, scale: float) -> ClaimDistribution:
    """Lomax severity with survival ``(1 + x / scale)**(-shape)``."""

    shape = _positive_float(shape, "shape")
    scale = _positive_float(scale, "scale")
    frozen = stats.lomax(c=shape, scale=scale)
    mean_value = np.inf if shape <= 1 else scale / (shape - 1)
    variance = None if shape <= 2 else shape * scale**2 / ((shape - 1) ** 2 * (shape - 2))

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return scale * rng.pareto(shape, size=n)

    return ClaimDistribution(
        name="lomax",
        mean_value=float(mean_value),
        variance_value=None if variance is None else float(variance),
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=None,
        laplace_function=None,
        metadata={"shape": float(shape), "scale": float(scale)},
    )


def lognormal(meanlog: float, sdlog: float) -> ClaimDistribution:
    """Lognormal severity parameterized as in R by meanlog and sdlog."""

    meanlog = _finite_float(meanlog, "meanlog")
    sdlog = _positive_float(sdlog, "sdlog")
    frozen = stats.lognorm(s=sdlog, scale=np.exp(meanlog))
    mean_value = float(np.exp(meanlog + 0.5 * sdlog**2))
    variance = float((np.exp(sdlog**2) - 1.0) * np.exp(2.0 * meanlog + sdlog**2))

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.lognormal(mean=meanlog, sigma=sdlog, size=n)

    return ClaimDistribution(
        name="lognormal",
        mean_value=mean_value,
        variance_value=variance,
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=None,
        laplace_function=None,
        metadata={"meanlog": float(meanlog), "sdlog": float(sdlog)},
    )


def weibull(shape: float, scale: float) -> ClaimDistribution:
    """Weibull severity with CDF ``1 - exp(-(x / scale)**shape)``."""

    shape = _positive_float(shape, "shape")
    scale = _positive_float(scale, "scale")
    frozen = stats.weibull_min(c=shape, scale=scale)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return scale * rng.weibull(shape, size=n)

    return ClaimDistribution(
        name="weibull",
        mean_value=float(frozen.mean()),
        variance_value=float(frozen.var()),
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        mgf_function=None,
        laplace_function=None,
        metadata={"shape": float(shape), "scale": float(scale)},
    )


def empirical(data: ArrayLike) -> ClaimDistribution:
    """Empirical claim distribution with sampling from observed severities."""

    values = _finite_1d(data, "data")
    if np.any(values < 0):
        raise ValueError("claim data must be non-negative")
    sorted_values = np.sort(values)

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(values, size=n, replace=True)

    def cdf(x: ArrayLike) -> np.ndarray:
        arr = _as_array(x)
        return np.searchsorted(sorted_values, arr, side="right") / sorted_values.size

    return ClaimDistribution(
        name="empirical",
        mean_value=float(np.mean(values)),
        variance_value=float(np.var(values, ddof=0)),
        sampler=sampler,
        cdf_function=cdf,
        survival_function=lambda x: 1.0 - cdf(x),
        mgf_function=lambda t: float(np.mean(np.exp(t * values))),
        laplace_function=lambda s: float(np.mean(np.exp(-s * values))),
        metadata={"n": int(values.size), "values": values.copy()},
    )


def scipy_distribution(name: str, **params: Any) -> ClaimDistribution:
    """Wrap a non-negative SciPy continuous distribution by name."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    try:
        rv = getattr(stats, name)
    except AttributeError as exc:
        raise ValueError(f"unknown SciPy distribution {name!r}") from exc
    frozen = rv(**params)
    support_low = float(frozen.support()[0])
    if np.isnan(support_low) or support_low < 0:
        raise ValueError("only non-negative distributions are valid claim severities")
    variance = float(frozen.var())

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return frozen.rvs(size=n, random_state=rng)

    return ClaimDistribution(
        name=name,
        mean_value=float(frozen.mean()),
        variance_value=None if np.isnan(variance) else variance,
        sampler=sampler,
        cdf_function=lambda x: frozen.cdf(_as_array(x)),
        survival_function=lambda x: frozen.sf(_as_array(x)),
        pdf_function=lambda x: frozen.pdf(_as_array(x)),
        metadata=dict(params),
    )
