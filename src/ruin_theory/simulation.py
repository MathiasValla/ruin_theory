"""Path simulation and Monte Carlo estimators."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .models import CapitalInjectionModel, RiskProcess
from .results import RuinEstimate, SimulationPath


def _rng(seed: int | None | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _next_claim_time(
    model: RiskProcess,
    current_time: float,
    rng: np.random.Generator,
) -> float:
    interarrival = float(model.frequency.sample_interarrival(rng))
    if math.isnan(interarrival) or interarrival < 0:
        raise ValueError("claim interarrival times must be non-negative")
    if math.isinf(interarrival):
        return math.inf
    return _advance_claim_clock(model, current_time, interarrival)


def _advance_claim_clock(model: RiskProcess, current_time: float, interarrival: float) -> float:
    t = float(current_time)
    remaining = float(interarrival)

    for _ in range(2 * len(model.prevention.frequency_windows) + 2):
        frequency_multiplier = float(model.prevention.frequency_multiplier_at(t))
        if not math.isfinite(frequency_multiplier):
            raise ValueError("frequency_multiplier must be finite")

        next_change = float(model.prevention.next_frequency_change_after(t))
        if frequency_multiplier > 0.0:
            if math.isinf(next_change):
                return t + remaining / frequency_multiplier
            available = (next_change - t) * frequency_multiplier
            if remaining <= available:
                return t + remaining / frequency_multiplier
            remaining -= available

        if math.isinf(next_change):
            return math.inf
        t = next_change

    raise RuntimeError("could not advance claim clock across prevention windows")


def _sample_claim_amount(model: RiskProcess, rng: np.random.Generator) -> float:
    primary = np.asarray(model.claim_distribution.sample(1, rng=rng), dtype=float)
    primary = np.asarray(model.prevention.apply_severity(primary), dtype=float)
    if primary.shape != (1,):
        raise ValueError("claim distribution must return one value for n=1")
    if math.isnan(float(primary[0])) or primary[0] < 0:
        raise ValueError("claim severity must be non-negative")

    by_total = 0.0
    for by_claim in model.by_claims:
        secondary = np.asarray(by_claim.sample_total(1, rng=rng), dtype=float)
        if secondary.shape != (1,):
            raise ValueError("by-claim model must return one total for one primary claim")
        if math.isnan(float(secondary[0])) or secondary[0] < 0:
            raise ValueError("by-claim severity must be non-negative")
        by_total += float(secondary[0])

    return float(primary[0] + by_total)


def simulate_path(
    model: RiskProcess,
    horizon: float,
    *,
    seed: int | None | np.random.Generator = None,
    max_events: int = 1_000_000,
    stop_at_ruin: bool = True,
) -> SimulationPath:
    """Simulate one reserve trajectory up to ``horizon``."""

    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    rng = _rng(seed)
    t = 0.0
    reserve = float(model.initial_capital)
    times = [0.0]
    reserves = [reserve]
    claim_times: list[float] = []
    claim_sizes: list[float] = []
    applied_injection_times: list[float] = []
    applied_injection_sizes: list[float] = []
    ruin_time: float | None = None

    injection_times, injection_sizes = _sample_injections(model.capital_injections, horizon, rng)
    injection_index = 0
    next_claim = _next_claim_time(model, t, rng)

    for _ in range(max_events):
        next_injection = (
            injection_times[injection_index] if injection_index < injection_times.size else math.inf
        )
        next_event_time = min(next_claim, next_injection)
        if next_event_time > horizon:
            reserve += model.premium_rate * (horizon - t)
            t = float(horizon)
            if times[-1] != t:
                times.append(t)
                reserves.append(reserve)
            break

        next_time = next_event_time
        reserve += model.premium_rate * (next_time - t)
        t = next_time
        if times[-1] != t:
            times.append(t)
            reserves.append(reserve)

        while injection_index < injection_times.size and injection_times[injection_index] == t:
            injection_size = float(injection_sizes[injection_index])
            reserve += injection_size
            applied_injection_times.append(float(t))
            applied_injection_sizes.append(injection_size)
            injection_index += 1
            times.append(t)
            reserves.append(reserve)

        if next_claim == t:
            claim = _sample_claim_amount(model, rng)
            reserve -= claim
            claim_times.append(t)
            claim_sizes.append(claim)
            times.append(t)
            reserves.append(reserve)
            next_claim = _next_claim_time(model, t, rng)
            if reserve < 0 and ruin_time is None:
                ruin_time = t
                if stop_at_ruin:
                    break

        if t >= horizon:
            break
    else:
        raise RuntimeError("max_events reached before horizon")

    return SimulationPath(
        times=np.asarray(times, dtype=float),
        reserves=np.asarray(reserves, dtype=float),
        claim_times=np.asarray(claim_times, dtype=float),
        claim_sizes=np.asarray(claim_sizes, dtype=float),
        ruin_time=ruin_time,
        horizon=float(horizon),
        initial_capital=float(model.initial_capital),
        premium_rate=float(model.premium_rate),
        injection_times=np.asarray(applied_injection_times, dtype=float),
        injection_sizes=np.asarray(applied_injection_sizes, dtype=float),
    )


def _sample_injections(
    injections: tuple[CapitalInjectionModel, ...],
    horizon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    for injection in injections:
        if not math.isfinite(injection.rate):
            raise ValueError("capital injection rate must be finite")
        if injection.rate == 0:
            continue
        n = rng.poisson(injection.rate * horizon)
        if n == 0:
            continue
        times.append(np.sort(rng.uniform(0.0, horizon, size=n)))
        sampled_sizes = np.asarray(injection.distribution.sample(n, rng=rng), dtype=float)
        if np.any(np.isnan(sampled_sizes)) or np.any(sampled_sizes < 0):
            raise ValueError("capital injections must be non-negative")
        sizes.append(sampled_sizes)
    if not times:
        return np.empty(0), np.empty(0)
    all_times = np.concatenate(times)
    all_sizes = np.concatenate(sizes)
    order = np.argsort(all_times)
    return all_times[order], all_sizes[order]


def estimate_ruin_probability(
    model: RiskProcess,
    horizon: float,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
    seed: int | None = None,
    return_paths: bool = False,
) -> RuinEstimate | tuple[RuinEstimate, list[SimulationPath]]:
    """Estimate finite-time ruin probability by crude Monte Carlo."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie in (0, 1)")
    method = ci_method.lower()
    if method not in {"wilson", "normal"}:
        raise ValueError("ci_method must be 'wilson' or 'normal'")
    rng = np.random.default_rng(seed)
    ruined = np.zeros(n_simulations, dtype=bool)
    ruin_times = np.full(n_simulations, np.inf)
    paths: list[SimulationPath] = []
    for i in range(n_simulations):
        path = simulate_path(model, horizon, seed=rng, stop_at_ruin=True)
        ruined[i] = path.ruined
        if path.ruin_time is not None:
            ruin_times[i] = path.ruin_time
        if return_paths:
            paths.append(path)

    probability = float(np.mean(ruined))
    standard_error = math.sqrt(max(probability * (1.0 - probability), 0.0) / n_simulations)
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    if method == "wilson":
        denominator = 1.0 + z**2 / n_simulations
        center = (probability + z**2 / (2.0 * n_simulations)) / denominator
        half_width = (
            z
            * math.sqrt(
                probability * (1.0 - probability) / n_simulations
                + z**2 / (4.0 * n_simulations**2)
            )
            / denominator
        )
        ci_low = max(0.0, center - half_width)
        ci_high = min(1.0, center + half_width)
    else:
        ci_low = max(0.0, probability - z * standard_error)
        ci_high = min(1.0, probability + z * standard_error)
    estimate = RuinEstimate(
        probability=probability,
        standard_error=standard_error,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=int(n_simulations),
        horizon=float(horizon),
        ruin_times=ruin_times,
        ci_method=method,
    )
    if return_paths:
        return estimate, paths
    return estimate


def simulate_terminal_reserves(
    model: RiskProcess,
    horizon: float,
    *,
    n_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return terminal reserves for stress testing and diagnostics."""

    rng = np.random.default_rng(seed)
    return np.array(
        [
            simulate_path(model, horizon, seed=rng, stop_at_ruin=False).terminal_reserve
            for _ in range(n_simulations)
        ]
    )
