"""Discrete aggregate-loss distributions and Panjer recursion."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
import operator
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike


_PMF_ATOL = 1e-10


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: float, name: str, *, include_one: bool = True) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if include_one else result < 1.0
    if not np.isfinite(result) or result < 0.0 or not upper_ok:
        bound = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must be in {bound}")
    return result


def _pmf(values: ArrayLike, name: str, *, normalize: bool = False) -> np.ndarray:
    masses = _as_1d_float(values, name)
    if np.any(masses < -_PMF_ATOL):
        raise ValueError(f"{name} must not contain negative probabilities")
    masses = np.maximum(masses, 0.0)
    total = float(masses.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total probability")
    if normalize:
        return masses / total
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PMF_ATOL):
        raise ValueError(f"{name} must sum to 1 within {_PMF_ATOL:g}; got {total:g}")
    return masses / total


def _scalar_or_array(values: np.ndarray, scalar: bool) -> float | np.ndarray:
    return float(values.item()) if scalar else values


@dataclass(frozen=True)
class AggregateDistribution:
    """Finite lattice approximation to an aggregate-loss distribution.

    The ``pmf`` may sum to less than one when it represents a truncation of an
    unbounded aggregate distribution. Quantiles and TVaR require the requested
    level to be covered by the computed mass.
    """

    grid: ArrayLike
    pmf: ArrayLike
    name: str = "aggregate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = _as_1d_float(self.grid, "grid")
        masses = _as_1d_float(self.pmf, "pmf")
        if grid.size != masses.size:
            raise ValueError("grid and pmf must have the same length")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("grid must be strictly increasing")
        if np.any(masses < -_PMF_ATOL):
            raise ValueError("pmf must not contain negative probabilities")
        masses = np.maximum(masses, 0.0)
        total = float(masses.sum())
        if total <= 0.0:
            raise ValueError("pmf must have positive total probability")
        if total > 1.0 + _PMF_ATOL:
            raise ValueError(f"pmf must sum to at most 1; got {total:g}")
        if np.isclose(total, 1.0, rtol=0.0, atol=_PMF_ATOL):
            masses = masses / total
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "pmf", masses)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def support(self) -> np.ndarray:
        return self.grid.copy()

    @property
    def total_mass(self) -> float:
        return float(self.pmf.sum())

    @property
    def is_truncated(self) -> bool:
        return not np.isclose(self.total_mass, 1.0, rtol=0.0, atol=_PMF_ATOL)

    def cdf_values(self) -> np.ndarray:
        return np.cumsum(self.pmf)

    def cdf(self, x: ArrayLike) -> float | np.ndarray:
        values = np.asarray(x, dtype=float)
        if np.any(np.isnan(values)):
            raise ValueError("x must not contain NaN")
        scalar = values.ndim == 0
        flat = values.reshape(-1)
        indices = np.searchsorted(self.grid, flat, side="right") - 1
        cdf = self.cdf_values()
        result = np.where(indices < 0, 0.0, cdf[np.clip(indices, 0, cdf.size - 1)])
        return _scalar_or_array(result.reshape(values.shape), scalar)

    def survival(self, x: ArrayLike) -> float | np.ndarray:
        values = np.asarray(self.cdf(x), dtype=float)
        result = 1.0 - values
        if np.asarray(x).ndim == 0:
            return float(result)
        return result

    def ppf(self, q: ArrayLike) -> float | np.ndarray:
        probabilities = np.asarray(q, dtype=float)
        scalar = probabilities.ndim == 0
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("q must contain probabilities in [0, computed mass]")
        total = self.total_mass
        if np.any(probabilities > total + _PMF_ATOL):
            raise ValueError(
                "q exceeds the computed aggregate mass; increase max_aggregate for "
                "truncated distributions"
            )
        clipped = np.minimum(probabilities.reshape(-1), total)
        indices = np.searchsorted(self.cdf_values(), clipped, side="left")
        result = self.grid[np.clip(indices, 0, self.grid.size - 1)]
        return _scalar_or_array(result.reshape(probabilities.shape), scalar)

    quantile = ppf
    value_at_risk = ppf

    def mean(self) -> float:
        return float(np.dot(self.grid, self.pmf))

    def variance(self) -> float:
        mean = self.mean()
        return float(np.dot((self.grid - mean) ** 2, self.pmf))

    def tail_value_at_risk(self, level: float, *, allow_truncated: bool = False) -> float:
        alpha = _probability(level, "level", include_one=False)
        if self.is_truncated and not allow_truncated:
            raise ValueError(
                "TVaR requires the computed pmf to sum to 1; pass "
                "allow_truncated=True for a finite-grid approximation"
            )
        if alpha >= self.total_mass - _PMF_ATOL:
            raise ValueError(
                "level must be below the computed aggregate mass; increase max_aggregate "
                "for truncated distributions"
            )
        var = float(self.value_at_risk(alpha))
        cdf_at_var = float(self.cdf(var))
        strict_tail = self.grid > var
        numerator = var * (cdf_at_var - alpha) + float(
            np.dot(self.grid[strict_tail], self.pmf[strict_tail])
        )
        return numerator / (1.0 - alpha)


@dataclass(frozen=True)
class _PanjerFrequency:
    model: str
    a: float
    b: float
    p0: float
    pgf: Any
    mean: float
    variance: float
    max_count: int | None
    parameters: dict[str, float | int]


def _mapping_get(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"missing frequency parameter: one of {', '.join(names)}")


def _frequency_from_mapping(
    frequency: str | Mapping[str, Any],
    params: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if isinstance(frequency, str):
        return frequency.lower().replace("-", "_"), dict(params or {})
    if not isinstance(frequency, Mapping):
        raise TypeError("frequency must be a string or mapping")
    merged = dict(frequency)
    if params:
        merged.update(params)
    model = merged.pop("model", merged.pop("name", merged.pop("distribution", None)))
    if model is None:
        raise ValueError("frequency mapping must include 'model', 'name', or 'distribution'")
    return str(model).lower().replace("-", "_"), merged


def _panjer_frequency(
    frequency: str | Mapping[str, Any],
    params: Mapping[str, Any] | None,
) -> _PanjerFrequency:
    model, values = _frequency_from_mapping(frequency, params)
    if model in {"poisson", "pois"}:
        lam = _nonnegative_float(
            _mapping_get(values, "lambda", "lambda_", "mean", "rate"),
            "lambda",
        )
        return _PanjerFrequency(
            model="poisson",
            a=0.0,
            b=lam,
            p0=exp(-lam),
            pgf=lambda z: exp(lam * (z - 1.0)),
            mean=lam,
            variance=lam,
            max_count=None,
            parameters={"lambda": lam},
        )
    if model in {"binomial", "binom"}:
        n = _nonnegative_integer(_mapping_get(values, "n", "size", "trials"), "n")
        p = _probability(
            _mapping_get(values, "p", "prob", "probability"),
            "p",
            include_one=False,
        )
        odds = p / (1.0 - p)
        return _PanjerFrequency(
            model="binomial",
            a=-odds,
            b=(n + 1.0) * odds,
            p0=(1.0 - p) ** n,
            pgf=lambda z: (1.0 - p + p * z) ** n,
            mean=n * p,
            variance=n * p * (1.0 - p),
            max_count=n,
            parameters={"n": n, "p": p},
        )
    if model in {"negative_binomial", "negativebinomial", "nbinom", "neg_binomial"}:
        r = _positive_float(_mapping_get(values, "r", "size", "number"), "r")
        p = _probability(_mapping_get(values, "p", "prob", "probability"), "p")
        if p == 0.0:
            raise ValueError("p must be positive")
        q = 1.0 - p
        return _PanjerFrequency(
            model="negative_binomial",
            a=q,
            b=q * (r - 1.0),
            p0=p**r,
            pgf=lambda z: (p / (1.0 - q * z)) ** r,
            mean=r * q / p,
            variance=r * q / p**2,
            max_count=None,
            parameters={"r": r, "p": p},
        )
    if model in {"geometric", "geom"}:
        p = _probability(_mapping_get(values, "p", "prob", "probability"), "p")
        if p == 0.0:
            raise ValueError("p must be positive")
        q = 1.0 - p
        return _PanjerFrequency(
            model="geometric",
            a=q,
            b=0.0,
            p0=p,
            pgf=lambda z: p / (1.0 - q * z),
            mean=q / p,
            variance=q / p**2,
            max_count=None,
            parameters={"p": p},
        )
    raise ValueError(
        "frequency model must be one of 'poisson', 'binomial', "
        "'negative_binomial', or 'geometric'"
    )


def _severity_grid(
    severity_pmf: ArrayLike,
    *,
    support: ArrayLike | None,
    grid_step: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    masses = _pmf(severity_pmf, "severity_pmf", normalize=normalize)
    step = _positive_float(grid_step, "grid_step")
    if support is None:
        grid = np.arange(masses.size, dtype=float) * step
    else:
        grid = _as_1d_float(support, "support")
        if grid.size != masses.size:
            raise ValueError("support and severity_pmf must have the same length")
        if not np.isclose(grid[0], 0.0, rtol=0.0, atol=_PMF_ATOL):
            raise ValueError("support must start at zero")
        differences = np.diff(grid)
        if np.any(differences <= 0.0):
            raise ValueError("support must be strictly increasing")
        step = float(differences[0]) if differences.size else step
        if differences.size and not np.allclose(
            differences,
            step,
            rtol=0.0,
            atol=_PMF_ATOL,
        ):
            raise ValueError("support must be an equally spaced lattice")
        lattice = grid / step
        if not np.allclose(lattice, np.round(lattice), rtol=0.0, atol=1e-8):
            raise ValueError("support must lie on integer multiples of its grid step")
    return masses, grid, step


def panjer_recursion(
    severity_pmf: ArrayLike,
    frequency: str | Mapping[str, Any],
    *,
    frequency_params: Mapping[str, Any] | None = None,
    max_aggregate: int | None = None,
    support: ArrayLike | None = None,
    grid_step: float = 1.0,
    normalize_severity: bool = False,
    name: str | None = None,
) -> AggregateDistribution:
    """Compute a compound aggregate PMF by Panjer's ``(a, b, 0)`` recursion.

    ``severity_pmf[j]`` is the probability of a claim amount ``j * grid_step``
    unless an explicit equally spaced ``support`` starting at zero is supplied.
    Unbounded frequencies are returned on the finite grid ``0:max_aggregate``.
    """

    masses, _, step = _severity_grid(
        severity_pmf,
        support=support,
        grid_step=grid_step,
        normalize=normalize_severity,
    )
    freq = _panjer_frequency(frequency, frequency_params)
    severity_max_index = masses.size - 1
    if max_aggregate is None:
        if freq.max_count is not None:
            max_index = freq.max_count * severity_max_index
        else:
            max_index = severity_max_index
    else:
        max_index = _nonnegative_integer(max_aggregate, "max_aggregate")

    aggregate = np.zeros(max_index + 1, dtype=float)
    f0 = float(masses[0])
    denominator = 1.0 - freq.a * f0
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("invalid Panjer denominator; check frequency and severity p0")
    aggregate[0] = float(freq.pgf(f0))

    for k in range(1, max_index + 1):
        upper = min(k, severity_max_index)
        if upper == 0:
            continue
        j = np.arange(1, upper + 1, dtype=float)
        weights = freq.a + freq.b * j / k
        previous = aggregate[k - np.arange(1, upper + 1)]
        aggregate[k] = float(np.dot(weights * masses[1 : upper + 1], previous))
        aggregate[k] /= denominator

    aggregate = np.maximum(aggregate, 0.0)
    grid = np.arange(max_index + 1, dtype=float) * step
    metadata = {
        "method": "panjer",
        "frequency": freq.model,
        "frequency_parameters": freq.parameters,
        "severity_pmf": masses.copy(),
        "grid_step": step,
    }
    return AggregateDistribution(
        grid=grid,
        pmf=aggregate,
        name=name or f"compound_{freq.model}",
        metadata=metadata,
    )


def compound_poisson_distribution(
    severity_pmf: ArrayLike,
    *,
    rate: float | None = None,
    mean: float | None = None,
    max_aggregate: int | None = None,
    support: ArrayLike | None = None,
    grid_step: float = 1.0,
    normalize_severity: bool = False,
    name: str = "compound_poisson",
) -> AggregateDistribution:
    """Convenience wrapper for a compound Poisson aggregate distribution."""

    if (rate is None) == (mean is None):
        raise ValueError("pass exactly one of rate or mean")
    lam = rate if rate is not None else mean
    return panjer_recursion(
        severity_pmf,
        "poisson",
        frequency_params={"lambda": lam},
        max_aggregate=max_aggregate,
        support=support,
        grid_step=grid_step,
        normalize_severity=normalize_severity,
        name=name,
    )


__all__ = [
    "AggregateDistribution",
    "compound_poisson_distribution",
    "panjer_recursion",
]
