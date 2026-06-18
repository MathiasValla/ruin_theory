"""Lattice Pollaczek-Khinchine approximations for ultimate ruin."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
from numpy.typing import ArrayLike

from .distributions import ClaimDistribution
from .formulas import integrated_tail_survival
from .models import CramerLundbergProcess


@dataclass(frozen=True)
class PanjerRuinResult:
    """Numerical Pollaczek-Khinchine result on a fixed lattice."""

    surplus: np.ndarray
    ruin_probabilities: np.ndarray
    aggregate_pmf: np.ndarray
    aggregate_cdf: np.ndarray
    ladder_height_pmf: np.ndarray
    rho: float
    step: float
    convention: str


def _positive_step(step: float) -> float:
    value = float(step)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("step must be finite and positive")
    return value


def _surplus_array(surplus: ArrayLike) -> np.ndarray:
    values = np.asarray(surplus, dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError("surplus values must not contain NaN")
    if np.any(values < 0.0):
        raise ValueError("surplus values must be non-negative")
    return values


def _rho_value(
    *,
    rho: float | None = None,
    safety_loading: float | None = None,
) -> float:
    if (rho is None) == (safety_loading is None):
        raise ValueError("pass exactly one of rho or safety_loading")
    if rho is None:
        loading = float(safety_loading)
        if not np.isfinite(loading) or loading <= 0.0:
            raise ValueError("safety_loading must be finite and positive")
        rho_value = 1.0 / (1.0 + loading)
    else:
        rho_value = float(rho)
    if not np.isfinite(rho_value) or not 0.0 <= rho_value < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return rho_value


def _severity_pmf(values: ArrayLike) -> np.ndarray:
    pmf = np.asarray(values, dtype=float)
    if pmf.ndim != 1 or pmf.size == 0:
        raise ValueError("ladder_height_pmf must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(pmf)) or np.any(pmf < 0.0):
        raise ValueError("ladder_height_pmf must contain finite non-negative masses")
    total = float(np.sum(pmf))
    if total <= 0.0:
        raise ValueError("ladder_height_pmf must have positive total mass")
    if total > 1.0 + 1e-10:
        raise ValueError("ladder_height_pmf total mass must not exceed one")
    return pmf.copy()


def _nonnegative_index(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def compound_geometric_pmf(
    ladder_height_pmf: ArrayLike,
    *,
    rho: float,
    max_aggregate: int | None = None,
) -> np.ndarray:
    """Return the lattice aggregate PMF for a geometric sum of ladder heights.

    ``ladder_height_pmf[k]`` is the mass at lattice amount ``k * step``; the
    step does not enter the recursion itself. The geometric count has
    ``P(N = n) = (1 - rho) * rho**n`` on ``n = 0, 1, ...``. ``max_aggregate``
    controls the largest aggregate lattice index returned.
    """

    rho_value = _rho_value(rho=rho)
    severity = _severity_pmf(ladder_height_pmf)
    max_index = severity.size - 1 if max_aggregate is None else _nonnegative_index(
        max_aggregate,
        "max_aggregate",
    )
    aggregate = np.zeros(max_index + 1, dtype=float)
    denominator = 1.0 - rho_value * severity[0]
    if denominator <= 0.0:
        raise ValueError("invalid zero-mass denominator in compound-geometric recursion")

    aggregate[0] = (1.0 - rho_value) / denominator
    scale = rho_value / denominator
    severity_max = severity.size - 1
    for k in range(1, aggregate.size):
        upper = min(k, severity_max)
        if upper > 0:
            indices = np.arange(1, upper + 1)
            aggregate[k] = scale * float(np.dot(severity[indices], aggregate[k - indices]))
    return aggregate


def discrete_pollaczek_khinchine_ultimate_ruin(
    ladder_height_pmf: ArrayLike,
    surplus: ArrayLike,
    *,
    step: float = 1.0,
    rho: float | None = None,
    safety_loading: float | None = None,
    max_aggregate: int | None = None,
    return_result: bool = False,
) -> np.ndarray | PanjerRuinResult:
    """Approximate ultimate ruin from a discretized integrated-tail PMF.

    The lattice convention is explicit: ``ladder_height_pmf[k]`` and the
    resulting aggregate PMF place mass at amount ``k * step``. For a surplus
    value ``u``, this returns ``P(M > u)`` by using
    ``floor(u / step)`` as the CDF index.
    """

    step_value = _positive_step(step)
    rho_value = _rho_value(rho=rho, safety_loading=safety_loading)
    surplus_values = _surplus_array(surplus)
    flat = surplus_values.ravel()
    indices = np.floor(flat / step_value).astype(int)
    needed_index = int(indices.max()) if indices.size else 0
    severity = _severity_pmf(ladder_height_pmf)
    if max_aggregate is None:
        max_index = max(severity.size - 1, needed_index)
    else:
        max_index = max(_nonnegative_index(max_aggregate, "max_aggregate"), needed_index)
    aggregate = compound_geometric_pmf(severity, rho=rho_value, max_aggregate=max_index)
    cdf = np.clip(np.cumsum(aggregate), 0.0, 1.0)

    clipped = np.minimum(indices, cdf.size - 1)
    ruin = 1.0 - cdf[clipped]
    ruin = np.clip(ruin.reshape(surplus_values.shape), 0.0, 1.0)

    if not return_result:
        return ruin
    return PanjerRuinResult(
        surplus=surplus_values.copy(),
        ruin_probabilities=ruin,
        aggregate_pmf=aggregate,
        aggregate_cdf=cdf,
        ladder_height_pmf=severity,
        rho=rho_value,
        step=step_value,
        convention="mass[k] is at k*step; surplus uses floor(u/step); ruin=P(M>u)",
    )


def equilibrium_severity_pmf(
    distribution: ClaimDistribution,
    *,
    step: float,
    max_value: float,
    method: str = "upper",
    scale: float = 1.0,
) -> np.ndarray:
    """Discretize the equilibrium, or integrated-tail, severity distribution.

    ``method="upper"`` allocates interval mass ``(k*h, (k+1)*h]`` to ``k*h``.
    ``method="lower"`` allocates interval mass ``((k-1)*h, k*h]`` to ``k*h``.
    These names follow the usual lower/upper endpoint convention used for
    aggregate-loss approximations. Tail mass beyond ``max_value`` is truncated.
    """

    if not isinstance(distribution, ClaimDistribution):
        raise TypeError("distribution must be a ClaimDistribution")
    step_value = _positive_step(step)
    maximum = float(max_value)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_value must be finite and positive")
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be finite and positive")

    method_value = method.lower()
    if method_value not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'")

    quotient = maximum / step_value
    max_index = round(quotient)
    if max_index <= 0 or not math.isclose(
        max_index * step_value,
        maximum,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError("max_value must be a positive integer multiple of step")
    grid = step_value * np.arange(max_index + 1, dtype=float)

    def cdf(points: np.ndarray) -> np.ndarray:
        tail = integrated_tail_survival(distribution, points, scale=scale_value)
        return np.clip(1.0 - np.asarray(tail, dtype=float), 0.0, 1.0)

    masses = np.zeros(max_index + 1, dtype=float)
    if method_value == "lower":
        values = cdf(grid)
        masses[0] = values[0]
        if max_index > 0:
            masses[1:] = values[1:] - values[:-1]
    else:
        right = cdf(grid[1:])
        left = cdf(grid[:-1])
        if max_index > 0:
            masses[:-1] = right - left
        masses[0] += cdf(np.asarray([0.0]))[0]

    masses = np.maximum(masses, 0.0)
    total = float(np.sum(masses))
    if total > 1.0 + 1e-10:
        masses /= total
    return masses


def ladder_height_pmf_from_severity(
    distribution: ClaimDistribution,
    *,
    step: float,
    max_value: float,
    method: str = "upper",
    scale: float = 1.0,
) -> np.ndarray:
    """Alias for the non-defective ladder-step law used by PK recursion."""

    return equilibrium_severity_pmf(
        distribution,
        step=step,
        max_value=max_value,
        method=method,
        scale=scale,
    )


def ultimate_ruin_panjer(
    model: CramerLundbergProcess,
    surplus: ArrayLike | None = None,
    *,
    step: float,
    max_value: float,
    discretization: str = "upper",
    return_result: bool = False,
) -> np.ndarray | PanjerRuinResult:
    """Approximate CL ultimate ruin by Panjer recursion on ladder heights."""

    if not isinstance(model, CramerLundbergProcess):
        raise ValueError("ultimate_ruin_panjer requires a CramerLundbergProcess")
    if model.prevention.frequency_windows:
        raise ValueError("ultimate_ruin_panjer requires stationary frequency prevention")
    if model.prevention.severity_transform is not None:
        raise ValueError("ultimate_ruin_panjer requires linear severity scaling")
    if model.by_claims:
        raise ValueError("ultimate_ruin_panjer does not currently include by-claims")
    if model.capital_injections:
        raise ValueError("ultimate_ruin_panjer does not support capital injections")
    if model.premium_rate <= 0.0:
        raise ValueError("premium_rate must be positive")

    target_surplus = model.initial_capital if surplus is None else surplus
    surplus_values = _surplus_array(target_surplus)
    if model.claim_arrival_rate == 0.0 or model.prevention.severity_multiplier == 0.0:
        zeros = np.zeros_like(surplus_values, dtype=float)
        if not return_result:
            return zeros
        empty = np.array([1.0])
        return PanjerRuinResult(
            surplus=surplus_values.copy(),
            ruin_probabilities=zeros,
            aggregate_pmf=empty.copy(),
            aggregate_cdf=empty.copy(),
            ladder_height_pmf=empty.copy(),
            rho=0.0,
            step=_positive_step(step),
            convention="mass[k] is at k*step; surplus uses floor(u/step); ruin=P(M>u)",
        )

    mean = model.expected_claim_amount
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("finite positive claim mean is required")
    rho = model.claim_intensity / model.premium_rate
    if rho >= 1.0:
        raise ValueError("net profit condition fails: rho must be less than one")

    ladder_pmf = ladder_height_pmf_from_severity(
        model.claim_distribution,
        step=step,
        max_value=max_value,
        method=discretization,
        scale=model.prevention.severity_multiplier,
    )
    return discrete_pollaczek_khinchine_ultimate_ruin(
        ladder_pmf,
        surplus_values,
        step=step,
        rho=rho,
        return_result=return_result,
    )
