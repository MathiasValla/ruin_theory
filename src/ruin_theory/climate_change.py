"""Worsening-risk and climate-change ruin models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, special, stats

from .results import RuinEstimate, SimulationPath


WorseningMode = Literal["shape", "scale"]


@dataclass(frozen=True)
class WorseningParetoModel:
    """Pareto risk model with worsening claim severities as in Kortschak-Loisel-Ribereau."""

    initial_capital: float
    claim_arrival_rate: float
    pareto_scale: float
    initial_shape: float
    worsening_speed: float
    safety_loading: float
    mode: WorseningMode = "shape"

    def __post_init__(self) -> None:
        _positive_float(self.initial_capital, "initial_capital")
        _positive_float(self.claim_arrival_rate, "claim_arrival_rate")
        _positive_float(self.pareto_scale, "pareto_scale")
        if _positive_float(self.initial_shape, "initial_shape") <= 1.0:
            raise ValueError("initial_shape must be greater than one")
        _nonnegative_float(self.worsening_speed, "worsening_speed")
        _positive_float(self.safety_loading, "safety_loading")
        if self.mode not in {"shape", "scale"}:
            raise ValueError("mode must be 'shape' or 'scale'")

    @property
    def initial_mean_claim(self) -> float:
        return self.pareto_scale / (self.initial_shape - 1.0)

    def shape_at(self, time: ArrayLike) -> np.ndarray:
        t = _time_array(time)
        if self.mode == "shape":
            return (self.initial_shape - 1.0) / (1.0 + self.worsening_speed * t) + 1.0
        return np.full_like(t, self.initial_shape, dtype=float)

    def scale_at(self, time: ArrayLike) -> np.ndarray:
        t = _time_array(time)
        if self.mode == "scale":
            return self.pareto_scale * (1.0 + self.worsening_speed * t)
        return np.full_like(t, self.pareto_scale, dtype=float)

    def mean_claim_at(self, time: ArrayLike) -> np.ndarray:
        return self.scale_at(time) / (self.shape_at(time) - 1.0)

    def premium_rate_at(self, time: ArrayLike) -> np.ndarray:
        return (1.0 + self.safety_loading) * self.claim_arrival_rate * self.mean_claim_at(time)

    def cumulative_premium(self, time: ArrayLike) -> np.ndarray:
        t = _time_array(time)
        factor = (
            (1.0 + self.safety_loading)
            * self.claim_arrival_rate
            * self.pareto_scale
            / (self.initial_shape - 1.0)
        )
        return factor * (t + 0.5 * self.worsening_speed * t * t)

    def survival_at(self, amount: ArrayLike, time: ArrayLike) -> np.ndarray:
        x = np.asarray(amount, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < 0.0):
            raise ValueError("amount must contain finite non-negative values")
        return (1.0 + x / self.scale_at(time)) ** (-self.shape_at(time))

    def uninsurability_time(self, premium_rate_max: float) -> float:
        maximum = _positive_float(premium_rate_max, "premium_rate_max")
        initial_rate = float(self.premium_rate_at(0.0))
        if maximum < initial_rate:
            return 0.0
        if self.worsening_speed == 0.0:
            return math.inf
        return (maximum / initial_rate - 1.0) / self.worsening_speed


@dataclass(frozen=True)
class ClimateChangeRuinTable:
    """Scenario table comparing KLR shape-drift and scale-drift models."""

    worsening_speeds: np.ndarray
    horizons: np.ndarray
    shape_finite_ruin: np.ndarray
    scale_finite_ruin: np.ndarray
    shape_asymptotic: np.ndarray
    scale_asymptotic: np.ndarray
    premium_rate_max: float
    n_simulations: int


@dataclass(frozen=True)
class InfiniteMeanPremiumModel:
    """Infinite-mean regularly varying model with increasing cumulative premium."""

    claim_arrival_rate: float
    tail_index: float
    pareto_scale: float
    premium_coefficient: float
    premium_power: float

    def __post_init__(self) -> None:
        _positive_float(self.claim_arrival_rate, "claim_arrival_rate")
        alpha = _positive_float(self.tail_index, "tail_index")
        if alpha > 1.0:
            raise ValueError("tail_index must be less than or equal to one")
        _positive_float(self.pareto_scale, "pareto_scale")
        _positive_float(self.premium_coefficient, "premium_coefficient")
        if _positive_float(self.premium_power, "premium_power") <= 1.0 / alpha:
            raise ValueError("premium_power must be greater than 1 / tail_index")

    def cumulative_premium(self, time: ArrayLike) -> np.ndarray:
        t = _time_array(time)
        return self.premium_coefficient * t**self.premium_power

    def survival(self, amount: ArrayLike) -> np.ndarray:
        x = np.asarray(amount, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < 0.0):
            raise ValueError("amount must contain finite non-negative values")
        return (1.0 + x / self.pareto_scale) ** (-self.tail_index)


def klr_shape_asymptotic(model: WorseningParetoModel | None = None, **kwargs: float) -> float:
    """KLR large-surplus asymptotic for the Pareto shape-drift model."""

    m = _model_or_kwargs("shape", model, kwargs)
    if m.worsening_speed <= 0.0:
        raise ValueError("worsening_speed must be positive")
    scale = math.sqrt(
        2.0
        * m.initial_capital
        * (m.initial_shape - 1.0)
        / ((1.0 + m.safety_loading) * m.claim_arrival_rate * m.worsening_speed),
    )
    return (
        m.claim_arrival_rate
        * scale
        * (1.0 + m.initial_capital / m.pareto_scale) ** -1.0
        * (math.pi / 2.0)
    )


def klr_scale_asymptotic(
    model: WorseningParetoModel | None = None,
    *,
    epsabs: float = 1e-10,
    **kwargs: float,
) -> float:
    """KLR large-surplus asymptotic for the Pareto scale-drift model."""

    m = _model_or_kwargs("scale", model, kwargs)
    if m.worsening_speed <= 0.0:
        raise ValueError("worsening_speed must be positive")
    mu = m.initial_mean_claim
    coefficient = m.safety_loading * m.claim_arrival_rate * mu / (2.0 * m.worsening_speed)
    root_u = math.sqrt(m.initial_capital)

    def integrand(t: float) -> float:
        return (
            1.0
            + root_u / m.pareto_scale * (1.0 / t + coefficient * t)
        ) ** (-m.initial_shape)

    integral, _ = integrate.quad(integrand, 0.0, math.inf, epsabs=epsabs, limit=200)
    return m.claim_arrival_rate * root_u / m.worsening_speed * integral


def infinite_mean_ruin_asymptotic(
    model: InfiniteMeanPremiumModel,
    initial_capital: float,
) -> float:
    """Asymptotic ruin probability for infinite-mean Pareto-type claims."""

    u = _positive_float(initial_capital, "initial_capital")
    beta = model.premium_power
    alpha = model.tail_index
    beta_factor = special.beta(1.0 / beta, alpha - 1.0 / beta) / beta
    inverse_premium = (u / model.premium_coefficient) ** (1.0 / beta)
    return model.claim_arrival_rate * inverse_premium * float(model.survival(u)) * beta_factor


def infinite_mean_ruin_integral(
    model: InfiniteMeanPremiumModel,
    initial_capital: float,
    *,
    epsabs: float = 1e-10,
) -> float:
    """Numerically compute `lambda int Fbar(u + p(t)) dt` from KLR Theorem 4.1."""

    u = _positive_float(initial_capital, "initial_capital")

    def integrand(time: float) -> float:
        return float(model.survival(u + model.cumulative_premium(time)))

    integral, _ = integrate.quad(integrand, 0.0, math.inf, epsabs=epsabs, limit=200)
    return model.claim_arrival_rate * integral


def simulate_worsening_pareto_path(
    model: WorseningParetoModel,
    horizon: float,
    *,
    seed: int | np.random.Generator | None = None,
    max_events: int = 1_000_000,
    stop_at_ruin: bool = True,
) -> SimulationPath:
    """Simulate one KLR worsening-Pareto surplus trajectory."""

    end = _positive_float(horizon, "horizon")
    maximum = int(max_events)
    if maximum <= 0:
        raise ValueError("max_events must be positive")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_claims = int(rng.poisson(model.claim_arrival_rate * end))
    if n_claims > maximum:
        raise RuntimeError("max_events reached before horizon")
    claim_times = np.sort(rng.uniform(0.0, end, size=n_claims)) if n_claims else np.empty(0)
    claim_sizes = _sample_worsening_claims(model, claim_times, rng)

    times = [0.0]
    reserves = [float(model.initial_capital)]
    cumulative_claims = 0.0
    ruin_time: float | None = None
    retained_claim_times: list[float] = []
    retained_claim_sizes: list[float] = []
    for time, claim in zip(claim_times, claim_sizes):
        reserve_before = float(
            model.initial_capital + model.cumulative_premium(time) - cumulative_claims,
        )
        if times[-1] != time:
            times.append(float(time))
            reserves.append(reserve_before)
        cumulative_claims += float(claim)
        reserve_after = float(
            model.initial_capital + model.cumulative_premium(time) - cumulative_claims,
        )
        times.append(float(time))
        reserves.append(reserve_after)
        retained_claim_times.append(float(time))
        retained_claim_sizes.append(float(claim))
        if reserve_after <= 0.0 and ruin_time is None:
            ruin_time = float(time)
            if stop_at_ruin:
                break
    if not stop_at_ruin or ruin_time is None:
        terminal = float(model.initial_capital + model.cumulative_premium(end) - cumulative_claims)
        if times[-1] != end:
            times.append(end)
            reserves.append(terminal)

    return SimulationPath(
        times=np.asarray(times, dtype=float),
        reserves=np.asarray(reserves, dtype=float),
        claim_times=np.asarray(retained_claim_times, dtype=float),
        claim_sizes=np.asarray(retained_claim_sizes, dtype=float),
        ruin_time=ruin_time,
        horizon=end,
        initial_capital=float(model.initial_capital),
        premium_rate=float(model.premium_rate_at(0.0)),
    )


def estimate_worsening_pareto_ruin_probability(
    model: WorseningParetoModel,
    horizon: float,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
    seed: int | None = None,
    max_events: int = 1_000_000,
    return_paths: bool = False,
) -> RuinEstimate | tuple[RuinEstimate, list[SimulationPath]]:
    """Estimate finite-time ruin probability for a worsening-Pareto model."""

    end = _positive_float(horizon, "horizon")
    simulations = int(n_simulations)
    if simulations <= 0:
        raise ValueError("n_simulations must be positive")
    maximum = int(max_events)
    if maximum <= 0:
        raise ValueError("max_events must be positive")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")
    method = ci_method.lower()
    if method not in {"wilson", "normal"}:
        raise ValueError("ci_method must be 'wilson' or 'normal'")
    rng = np.random.default_rng(seed)
    ruined = np.zeros(simulations, dtype=bool)
    ruin_times = np.full(simulations, np.inf)
    paths: list[SimulationPath] = []
    for index in range(simulations):
        if return_paths:
            path = simulate_worsening_pareto_path(model, end, seed=rng, max_events=maximum)
            ruined[index] = path.ruined
            if path.ruin_time is not None:
                ruin_times[index] = path.ruin_time
            paths.append(path)
        else:
            ruin_time = _simulate_worsening_ruin_time(model, end, rng, maximum)
            if ruin_time is not None:
                ruined[index] = True
                ruin_times[index] = ruin_time

    estimate = _ruin_estimate_from_flags(
        ruined,
        ruin_times,
        horizon=end,
        ci_level=ci_level,
        ci_method=method,
    )
    if return_paths:
        return estimate, paths
    return estimate


def climate_change_ruin_table(
    worsening_speeds: ArrayLike,
    *,
    initial_capital: float = 500.0,
    claim_arrival_rate: float = 1.0,
    pareto_scale: float = 1.0,
    initial_shape: float = 1.5,
    safety_loading: float = 1.0,
    premium_rate_max: float | None = None,
    reference_speed: float = 0.1,
    reference_horizon: float = 20.0,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> ClimateChangeRuinTable:
    """Build a KLR-style table comparing shape and scale worsening scenarios."""

    speeds = _positive_array(worsening_speeds, "worsening_speeds")
    simulations = int(n_simulations)
    if simulations <= 0:
        raise ValueError("n_simulations must be positive")
    base = WorseningParetoModel(
        initial_capital=initial_capital,
        claim_arrival_rate=claim_arrival_rate,
        pareto_scale=pareto_scale,
        initial_shape=initial_shape,
        worsening_speed=reference_speed,
        safety_loading=safety_loading,
        mode="shape",
    )
    maximum = (
        float(base.premium_rate_at(reference_horizon))
        if premium_rate_max is None
        else _positive_float(premium_rate_max, "premium_rate_max")
    )
    rng = np.random.default_rng(seed)
    horizons = np.empty_like(speeds)
    shape_finite = np.empty_like(speeds)
    scale_finite = np.empty_like(speeds)
    shape_asymptotic = np.empty_like(speeds)
    scale_asymptotic = np.empty_like(speeds)
    for index, speed in enumerate(speeds):
        shape_model = WorseningParetoModel(
            initial_capital=initial_capital,
            claim_arrival_rate=claim_arrival_rate,
            pareto_scale=pareto_scale,
            initial_shape=initial_shape,
            worsening_speed=float(speed),
            safety_loading=safety_loading,
            mode="shape",
        )
        scale_model = WorseningParetoModel(
            initial_capital=initial_capital,
            claim_arrival_rate=claim_arrival_rate,
            pareto_scale=pareto_scale,
            initial_shape=initial_shape,
            worsening_speed=float(speed),
            safety_loading=safety_loading,
            mode="scale",
        )
        horizon = shape_model.uninsurability_time(maximum)
        horizons[index] = horizon
        shape_finite[index] = estimate_worsening_pareto_ruin_probability(
            shape_model,
            horizon,
            n_simulations=simulations,
            seed=rng,
        ).probability
        scale_finite[index] = estimate_worsening_pareto_ruin_probability(
            scale_model,
            horizon,
            n_simulations=simulations,
            seed=rng,
        ).probability
        shape_asymptotic[index] = klr_shape_asymptotic(shape_model)
        scale_asymptotic[index] = klr_scale_asymptotic(scale_model)

    return ClimateChangeRuinTable(
        worsening_speeds=speeds,
        horizons=horizons,
        shape_finite_ruin=shape_finite,
        scale_finite_ruin=scale_finite,
        shape_asymptotic=shape_asymptotic,
        scale_asymptotic=scale_asymptotic,
        premium_rate_max=maximum,
        n_simulations=simulations,
    )


def _model_or_kwargs(
    mode: WorseningMode,
    model: WorseningParetoModel | None,
    kwargs: dict[str, float],
) -> WorseningParetoModel:
    if model is not None:
        if model.mode != mode:
            return WorseningParetoModel(
                initial_capital=model.initial_capital,
                claim_arrival_rate=model.claim_arrival_rate,
                pareto_scale=model.pareto_scale,
                initial_shape=model.initial_shape,
                worsening_speed=model.worsening_speed,
                safety_loading=model.safety_loading,
                mode=mode,
            )
        return model
    return WorseningParetoModel(mode=mode, **kwargs)


def _sample_worsening_claims(
    model: WorseningParetoModel,
    times: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if times.size == 0:
        return np.empty(0)
    uniforms = rng.random(times.size)
    shape = model.shape_at(times)
    scale = model.scale_at(times)
    return scale * ((1.0 - uniforms) ** (-1.0 / shape) - 1.0)


def _simulate_worsening_ruin_time(
    model: WorseningParetoModel,
    horizon: float,
    rng: np.random.Generator,
    max_events: int,
) -> float | None:
    n_claims = int(rng.poisson(model.claim_arrival_rate * horizon))
    if n_claims > max_events:
        raise RuntimeError("max_events reached before horizon")
    if n_claims == 0:
        return None
    claim_times = np.sort(rng.uniform(0.0, horizon, size=n_claims))
    claim_sizes = _sample_worsening_claims(model, claim_times, rng)
    cumulative_claims = np.cumsum(claim_sizes)
    reserves = model.initial_capital + model.cumulative_premium(claim_times) - cumulative_claims
    ruin_indices = np.flatnonzero(reserves <= 0.0)
    if ruin_indices.size == 0:
        return None
    return float(claim_times[ruin_indices[0]])


def _ruin_estimate_from_flags(
    ruined: np.ndarray,
    ruin_times: np.ndarray,
    *,
    horizon: float,
    ci_level: float,
    ci_method: str,
) -> RuinEstimate:
    n = ruined.size
    probability = float(np.mean(ruined))
    standard_error = math.sqrt(max(probability * (1.0 - probability), 0.0) / n)
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    if ci_method == "wilson":
        denominator = 1.0 + z**2 / n
        center = (probability + z**2 / (2.0 * n)) / denominator
        half_width = (
            z
            * math.sqrt(probability * (1.0 - probability) / n + z**2 / (4.0 * n**2))
            / denominator
        )
        ci_low = max(0.0, center - half_width)
        ci_high = min(1.0, center + half_width)
    else:
        ci_low = max(0.0, probability - z * standard_error)
        ci_high = min(1.0, probability + z * standard_error)
    return RuinEstimate(
        probability=probability,
        standard_error=standard_error,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=int(n),
        horizon=float(horizon),
        ruin_times=ruin_times,
        ci_method=ci_method,
    )


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


def _time_array(time: ArrayLike) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("time must contain finite non-negative values")
    return values


def _positive_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array
