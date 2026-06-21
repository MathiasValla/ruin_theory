"""Gerber-Shiu discounted penalty diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import operator

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .models import CramerLundbergProcess, RiskProcess
from .results import GerberShiuResult, SimulationPath
from .simulation import simulate_path

PenaltyFunction = Callable[[float, float], float]


@dataclass(frozen=True)
class GerberShiuExponentialClosedForm:
    """Closed Gerber-Shiu transform for exponential Cramer-Lundberg claims."""

    initial_capitals: np.ndarray
    values: np.ndarray
    discount_rate: float
    deficit_moment_order: int
    decay_rate: float
    prefactor: float


def _rng(seed: int | None | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_common(discount_rate: float, ci_level: float) -> tuple[float, float]:
    discount = float(discount_rate)
    if not math.isfinite(discount) or discount < 0.0:
        raise ValueError("discount_rate must be finite and non-negative")
    level = float(ci_level)
    if not 0.0 < level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")
    return discount, level


def _positive_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _penalty_function(penalty: PenaltyFunction | None) -> PenaltyFunction:
    if penalty is None:
        return lambda surplus, deficit: 1.0
    if not callable(penalty):
        raise TypeError("penalty must be callable or None")
    return penalty


def _surplus_array(values: ArrayLike, fallback: float) -> np.ndarray:
    array = np.asarray(fallback if values is None else values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("initial surplus values must be finite and non-negative")
    return array


def gerber_shiu_exponential_closed_form(
    model: CramerLundbergProcess,
    u: ArrayLike | None = None,
    *,
    discount_rate: float = 0.0,
    deficit_moment_order: int = 0,
    return_result: bool = False,
) -> np.ndarray | GerberShiuExponentialClosedForm:
    """Closed form for `E[e^{-delta tau} D_tau^k; tau < inf]`.

    Claims must be exponential and primary-only. For `k=0`, this is the
    discounted ruin transform. For `k>0`, the exponential memoryless property
    multiplies the transform by the `k`-th moment of the deficit at ruin.
    """

    if not isinstance(model, CramerLundbergProcess):
        raise TypeError("model must be a CramerLundbergProcess")
    if model.claim_distribution.name != "exponential":
        raise ValueError("requires exponential claim sizes")
    if model.by_claims or model.capital_injections:
        raise ValueError("closed Gerber-Shiu formula requires primary claims only")
    if model.prevention.frequency_windows or model.prevention.severity_transform is not None:
        raise ValueError("closed Gerber-Shiu formula requires stationary linear prevention")
    if model.premium_rate <= 0.0:
        raise ValueError("premium_rate must be positive")
    discount, _ = _validate_common(discount_rate, 0.95)
    order = _positive_int(deficit_moment_order + 1, "deficit_moment_order + 1") - 1
    surplus = _surplus_array(u, model.initial_capital)

    scale = float(model.prevention.severity_multiplier)
    if scale == 0.0 or model.claim_arrival_rate == 0.0:
        values = np.zeros_like(surplus, dtype=float)
        result = GerberShiuExponentialClosedForm(
            initial_capitals=surplus.copy(),
            values=values,
            discount_rate=discount,
            deficit_moment_order=order,
            decay_rate=np.inf,
            prefactor=0.0,
        )
        return result if return_result else values
    claim_rate = float(model.claim_distribution.metadata["rate"]) / scale
    arrival = float(model.claim_arrival_rate)
    premium = float(model.premium_rate)
    deficit_moment = math.factorial(order) / claim_rate**order

    rho = arrival / (premium * claim_rate)
    if discount == 0.0 and rho >= 1.0:
        values = np.full_like(surplus, deficit_moment, dtype=float)
        result = GerberShiuExponentialClosedForm(
            initial_capitals=surplus.copy(),
            values=values,
            discount_rate=discount,
            deficit_moment_order=order,
            decay_rate=0.0,
            prefactor=deficit_moment,
        )
        return result if return_result else values

    linear = premium * claim_rate - arrival - discount
    root = (linear + math.sqrt(linear * linear + 4.0 * premium * claim_rate * discount))
    root /= 2.0 * premium
    prefactor = (claim_rate - root) / claim_rate * deficit_moment
    values = prefactor * np.exp(-root * surplus)
    values = np.maximum(values, 0.0)
    result = GerberShiuExponentialClosedForm(
        initial_capitals=surplus.copy(),
        values=values,
        discount_rate=discount,
        deficit_moment_order=order,
        decay_rate=float(root),
        prefactor=float(prefactor),
    )
    return result if return_result else values


def _normal_interval(values: np.ndarray, ci_level: float) -> tuple[float, float, float]:
    n = values.size
    estimate = float(values.mean())
    if n <= 1:
        return estimate, 0.0, 0.0
    standard_error = float(values.std(ddof=1) / math.sqrt(n))
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    return estimate, standard_error, z


def gerber_shiu_from_paths(
    paths: Iterable[SimulationPath],
    *,
    penalty: PenaltyFunction | None = None,
    discount_rate: float = 0.0,
    ci_level: float = 0.95,
    horizon: float | None = None,
) -> GerberShiuResult:
    """Estimate a Gerber-Shiu discounted penalty from simulated paths.

    The penalty is evaluated as ``w(surplus_before_ruin, deficit_at_ruin)`` on
    ruined paths and zero on non-ruined paths. With ``penalty=None`` and
    ``discount_rate=0``, the estimate is the finite-horizon ruin probability.
    """

    path_list = list(paths)
    if not path_list:
        raise ValueError("paths must contain at least one SimulationPath")
    if not all(isinstance(path, SimulationPath) for path in path_list):
        raise TypeError("paths must contain only SimulationPath instances")
    discount, level = _validate_common(discount_rate, ci_level)
    penalty_function = _penalty_function(penalty)

    n = len(path_list)
    ruin_times = np.full(n, np.inf)
    surplus = np.full(n, np.nan)
    deficits = np.full(n, np.nan)
    ruin_claims = np.full(n, np.nan)
    penalty_values = np.zeros(n, dtype=float)
    discounted = np.zeros(n, dtype=float)

    for index, path in enumerate(path_list):
        if path.ruin_time is None:
            continue
        pre_ruin = path.surplus_before_ruin
        deficit = path.deficit_at_ruin
        if pre_ruin is None or deficit is None:
            raise ValueError("ruined paths must expose surplus and deficit at ruin")

        value = float(penalty_function(float(pre_ruin), float(deficit)))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("penalty must return a finite non-negative value")
        ruin_time = float(path.ruin_time)
        ruin_times[index] = ruin_time
        surplus[index] = float(pre_ruin)
        deficits[index] = float(deficit)
        claim = path.claim_causing_ruin
        if claim is not None:
            ruin_claims[index] = float(claim)
        penalty_values[index] = value
        discounted[index] = math.exp(-discount * ruin_time) * value

    estimate, standard_error, z = _normal_interval(discounted, level)
    ci_low = max(0.0, estimate - z * standard_error)
    ci_high = estimate + z * standard_error
    if horizon is None:
        finite_horizons = [float(path.horizon) for path in path_list if math.isfinite(path.horizon)]
        horizon_value = max(finite_horizons) if finite_horizons else None
    else:
        horizon_value = float(horizon)
        if not math.isfinite(horizon_value) or horizon_value <= 0.0:
            raise ValueError("horizon must be positive and finite")

    return GerberShiuResult(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=n,
        horizon=horizon_value,
        discount_rate=discount,
        penalty_values=penalty_values,
        discounted_penalties=discounted,
        ruin_times=ruin_times,
        surplus_before_ruin=surplus,
        deficits_at_ruin=deficits,
        claim_causing_ruin=ruin_claims,
    )


def estimate_gerber_shiu(
    model: RiskProcess,
    horizon: float,
    *,
    n_simulations: int = 10_000,
    penalty: PenaltyFunction | None = None,
    discount_rate: float = 0.0,
    ci_level: float = 0.95,
    seed: int | None | np.random.Generator = None,
    max_events: int = 1_000_000,
    return_paths: bool = False,
) -> GerberShiuResult | tuple[GerberShiuResult, list[SimulationPath]]:
    """Estimate a finite-horizon Gerber-Shiu discounted penalty by simulation."""

    if not isinstance(model, RiskProcess):
        raise TypeError("model must be a RiskProcess")
    horizon_value = float(horizon)
    if not math.isfinite(horizon_value) or horizon_value <= 0.0:
        raise ValueError("horizon must be positive and finite")
    simulation_count = _positive_int(n_simulations, "n_simulations")
    event_count = _positive_int(max_events, "max_events")
    _validate_common(discount_rate, ci_level)
    _penalty_function(penalty)

    rng = _rng(seed)
    paths = [
        simulate_path(
            model,
            horizon_value,
            seed=rng,
            max_events=event_count,
            stop_at_ruin=True,
        )
        for _ in range(simulation_count)
    ]
    result = gerber_shiu_from_paths(
        paths,
        penalty=penalty,
        discount_rate=discount_rate,
        ci_level=ci_level,
        horizon=horizon_value,
    )
    if return_paths:
        return result, paths
    return result
