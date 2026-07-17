"""Path simulation and Monte Carlo estimators."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .models import CapitalInjectionModel, CramerLundbergProcess, RiskProcess
from .results import RuinEstimate, RuinSensitivityEstimate, SimulationPath


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
    if not model.prevention.frequency_windows:
        frequency_multiplier = float(model.prevention.frequency_multiplier)
        if not math.isfinite(frequency_multiplier):
            raise ValueError("frequency_multiplier must be finite")
        return math.inf if frequency_multiplier == 0.0 else t + remaining / frequency_multiplier

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
    return float(_sample_effective_claim_amounts(model, 1, rng)[0])


def _sample_effective_claim_amounts(
    model: RiskProcess,
    n_claims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample primary claims after prevention plus event-level by-claims."""

    if n_claims < 0:
        raise ValueError("n_claims must be non-negative")
    if n_claims == 0:
        return np.empty(0)
    primary = _validate_sample(
        model.prevention.apply_severity(model.claim_distribution.sample(n_claims, rng=rng)),
        n_claims,
        "claim distribution",
    )
    totals = primary.copy()
    for by_claim in model.by_claims:
        secondary = _validate_sample(
            by_claim.sample_total(n_claims, rng=rng),
            n_claims,
            "by-claim model",
        )
        totals += secondary
    return totals


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


def _aggregate_by_counts(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    totals = np.zeros(counts.size, dtype=float)
    if values.size == 0:
        return totals
    starts = np.r_[0, np.cumsum(counts[:-1])]
    nonzero = counts > 0
    totals[nonzero] = np.add.reduceat(values, starts[nonzero])
    return totals


def _poisson_frequency_exposure(model: CramerLundbergProcess, horizon: float) -> float:
    base_rate = model.frequency.mean_rate()
    base_multiplier = float(model.prevention.frequency_multiplier)
    exposure = base_multiplier * horizon
    for start, end, multiplier in model.prevention.frequency_windows:
        overlap = max(0.0, min(end, horizon) - start)
        exposure += overlap * (multiplier - base_multiplier)
    return base_rate * exposure


def _validate_sample(values: np.ndarray, expected_size: int, name: str) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    if samples.shape != (expected_size,):
        raise ValueError(f"{name} must return one value per sampled event")
    if np.any(~np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return samples


def _simulate_terminal_reserves_cramer_lundberg(
    model: CramerLundbergProcess,
    horizon: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = rng.poisson(_poisson_frequency_exposure(model, horizon), size=n_simulations)
    total_claims = int(counts.sum())
    claim_totals = np.zeros(n_simulations, dtype=float)
    if total_claims > 0:
        claim_amounts = _sample_effective_claim_amounts(model, total_claims, rng)
        claim_totals += _aggregate_by_counts(claim_amounts, counts)

    injection_totals = np.zeros(n_simulations, dtype=float)
    for injection in model.capital_injections:
        injection_counts = rng.poisson(injection.rate * horizon, size=n_simulations)
        total_injections = int(injection_counts.sum())
        if total_injections == 0:
            continue
        sampled_sizes = _validate_sample(
            injection.distribution.sample(total_injections, rng=rng),
            total_injections,
            "capital injection distribution",
        )
        injection_totals += _aggregate_by_counts(sampled_sizes, injection_counts)

    return (
        model.initial_capital
        + model.premium_rate * horizon
        - claim_totals
        + injection_totals
    )


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


def _as_surplus_array(u: ArrayLike) -> np.ndarray:
    surplus = np.asarray(u, dtype=float)
    if np.any(~np.isfinite(surplus)):
        raise ValueError("u must contain only finite values")
    if np.any(surplus < 0.0):
        raise ValueError("u must be non-negative")
    return surplus


def _premium_income_values(
    premium_income: Callable[[np.ndarray], ArrayLike] | None,
    times: np.ndarray,
    premium_rate: float,
) -> np.ndarray:
    if premium_income is None:
        return premium_rate * times
    values = np.asarray(premium_income(times), dtype=float)
    if values.shape == () and times.shape == (1,):
        values = values.reshape(1)
    if values.shape != times.shape:
        raise ValueError("premium_income must preserve the input shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("premium_income must return finite values")
    if np.any(values < 0.0):
        raise ValueError("premium_income must be non-negative")
    return values


def _validate_loisel_privault_process(model: RiskProcess) -> CramerLundbergProcess:
    if not isinstance(model, CramerLundbergProcess):
        raise ValueError("Loisel-Privault IBP estimation requires a CramerLundbergProcess")
    if model.capital_injections:
        raise ValueError("Loisel-Privault IBP estimation does not support capital injections")
    if model.prevention.frequency_windows:
        raise ValueError("Loisel-Privault IBP estimation requires homogeneous claim arrivals")
    return model


def estimate_finite_time_ruin_sensitivity_ibp(
    model: CramerLundbergProcess,
    u: ArrayLike,
    horizon: float,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    seed: int | None = None,
    premium_income: Callable[[np.ndarray], ArrayLike] | None = None,
    conditional_on_claim: bool = False,
    return_pathwise_density: bool = False,
) -> RuinSensitivityEstimate:
    """Estimate ``d psi(u,T) / du`` by the Loisel-Privault IBP formula.

    The estimator targets the density of
    ``M_[0,T] = inf_{0<=t<=T} (f(t) - S(t))`` at ``-u``. For the usual
    finite-time ruin probability ``psi(u,T) = P(M_[0,T] < -u)``, this density
    is ``-d psi(u,T) / du``. The default premium income is the linear income
    ``model.premium_rate * t``; a deterministic increasing ``premium_income``
    callable can be supplied to reproduce the paper's more general ``f(t)``.
    """

    cl_model = _validate_loisel_privault_process(model)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")

    surplus = _as_surplus_array(u)
    flat_surplus = surplus.ravel()
    infimum_points = -flat_surplus
    if flat_surplus.size == 0:
        raise ValueError("u must contain at least one value")

    f0 = float(_premium_income_values(premium_income, np.array([0.0]), cl_model.premium_rate)[0])
    if abs(f0) > 1e-10:
        raise ValueError("premium_income must satisfy f(0) = 0")
    final_income = float(
        _premium_income_values(premium_income, np.array([horizon]), cl_model.premium_rate)[0]
    )

    claim_arrival_rate = cl_model.claim_arrival_rate
    if claim_arrival_rate == 0.0:
        if conditional_on_claim:
            raise ValueError("conditional density is undefined when the claim arrival rate is zero")
        zeros = np.zeros_like(surplus, dtype=float)
        return RuinSensitivityEstimate(
            surplus=surplus,
            infimum_points=-surplus,
            density=zeros,
            standard_error=zeros,
            ci_low=zeros,
            ci_high=zeros,
            n_simulations=int(n_simulations),
            horizon=float(horizon),
            claim_arrival_rate=0.0,
            conditional_on_claim=False,
        )

    rng = np.random.default_rng(seed)
    counts = rng.poisson(claim_arrival_rate * horizon, size=n_simulations)
    actual_offsets = np.r_[0, np.cumsum(counts)]
    formula_claim_counts = counts + 1
    claim_offsets = np.r_[0, np.cumsum(formula_claim_counts)]
    all_times = rng.uniform(0.0, horizon, size=int(counts.sum()))
    all_claims = _sample_effective_claim_amounts(
        cl_model,
        int(formula_claim_counts.sum()),
        rng,
    )

    contributions = np.zeros((n_simulations, flat_surplus.size), dtype=float)
    for index, n_claims_raw in enumerate(counts):
        n_claims = int(n_claims_raw)
        claim_start = int(claim_offsets[index])
        claim_end = int(claim_offsets[index + 1])
        cumulative_claims = np.cumsum(all_claims[claim_start:claim_end])
        terminal_claim_sum = float(cumulative_claims[n_claims])

        if n_claims:
            time_start = int(actual_offsets[index])
            time_end = int(actual_offsets[index + 1])
            claim_times = np.sort(all_times[time_start:time_end])
            income = _premium_income_values(premium_income, claim_times, cl_model.premium_rate)
            if np.any(np.diff(income) < -1e-10) or income[-1] > final_income + 1e-10:
                raise ValueError("premium_income must be increasing on [0, horizon]")

            running_values = income - cumulative_claims[:n_claims]
            prefix_min = np.minimum.accumulate(running_values)
            shifted_values = income - cumulative_claims[1 : n_claims + 1]
            suffix_min = np.minimum.accumulate(shifted_values[::-1])[::-1]
            previous_income = np.r_[0.0, income[:-1]]
            lower = previous_income - cumulative_claims[:n_claims]
            upper = np.minimum(prefix_min, suffix_min)
            contributions[index] += np.sum(
                (lower[:, None] < infimum_points[None, :])
                & (infimum_points[None, :] <= upper[:, None]),
                axis=0,
            )
            minimum_running_value = float(prefix_min[-1])
            last_income = float(income[-1])
        else:
            minimum_running_value = math.inf
            last_income = 0.0

        terminal_lower = max(-terminal_claim_sum, last_income - terminal_claim_sum)
        terminal_upper = min(final_income - terminal_claim_sum, minimum_running_value)
        contributions[index] += (terminal_lower < infimum_points) & (
            infimum_points < terminal_upper
        )

    scale = claim_arrival_rate
    if conditional_on_claim:
        probability_at_least_one_claim = 1.0 - math.exp(-claim_arrival_rate * horizon)
        scale /= probability_at_least_one_claim
    contributions *= scale
    density_flat = np.mean(contributions, axis=0)
    if n_simulations > 1:
        standard_error_flat = np.std(contributions, axis=0, ddof=1) / math.sqrt(n_simulations)
    else:
        standard_error_flat = np.zeros_like(density_flat)
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    ci_low_flat = np.maximum(density_flat - z * standard_error_flat, 0.0)
    ci_high_flat = density_flat + z * standard_error_flat

    output_shape = surplus.shape
    stored_pathwise = contributions if return_pathwise_density else np.empty((0, 0))
    return RuinSensitivityEstimate(
        surplus=surplus,
        infimum_points=(-surplus),
        density=density_flat.reshape(output_shape),
        standard_error=standard_error_flat.reshape(output_shape),
        ci_low=ci_low_flat.reshape(output_shape),
        ci_high=ci_high_flat.reshape(output_shape),
        n_simulations=int(n_simulations),
        horizon=float(horizon),
        claim_arrival_rate=float(claim_arrival_rate),
        conditional_on_claim=bool(conditional_on_claim),
        pathwise_density=stored_pathwise,
    )


def simulate_terminal_reserves(
    model: RiskProcess,
    horizon: float,
    *,
    n_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return terminal reserves for stress testing and diagnostics."""

    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    rng = np.random.default_rng(seed)
    if isinstance(model, CramerLundbergProcess):
        return _simulate_terminal_reserves_cramer_lundberg(model, horizon, n_simulations, rng)
    return np.array(
        [
            simulate_path(model, horizon, seed=rng, stop_at_ruin=False).terminal_reserve
            for _ in range(n_simulations)
        ]
    )
