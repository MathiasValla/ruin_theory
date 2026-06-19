"""Horizontal dividend-barrier tools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .distributions import ClaimDistribution
from .interest import win_first_probability_exponential_interest_force


DividendMeanConvention = Literal["renewal", "loisel"]


@dataclass(frozen=True)
class BarrierDividendAnalyticResult:
    """Analytic barrier-dividend quantities from win-first probabilities."""

    initial_capital: float
    barrier: float
    hit_probability: float
    continuation_probability: float
    claim_arrival_rate: float
    interest_force: float
    dividend_rate: float
    period_mean: float
    expected_dividends: float
    loisel_expected_dividends: float


@dataclass(frozen=True)
class BarrierDividendPath:
    """One simulated path under a horizontal dividend barrier."""

    times: np.ndarray
    reserves: np.ndarray
    dividend_times: np.ndarray
    cumulative_dividends: np.ndarray
    claim_times: np.ndarray
    claim_sizes: np.ndarray
    ruin_time: float | None
    horizon: float
    initial_capital: float
    barrier: float
    premium_rate: float
    claim_arrival_rate: float
    interest_force: float

    @property
    def ruined(self) -> bool:
        """Whether the simulated path crossed below zero."""

        return self.ruin_time is not None

    @property
    def total_dividends(self) -> float:
        """Total dividends paid on the simulated path."""

        return float(self.cumulative_dividends[-1])


@dataclass(frozen=True)
class BarrierDividendEstimate:
    """Monte Carlo estimate for barrier-dividend totals and ruin times."""

    total_dividends: np.ndarray
    ruin_times: np.ndarray
    mean_dividends: float
    standard_error: float
    probability_ruin: float
    n_simulations: int
    horizon: float
    barrier: float


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


def _probability(value: float, name: str, *, strict_upper: bool = False) -> float:
    result = float(value)
    upper_ok = result < 1.0 if strict_upper else result <= 1.0
    if not np.isfinite(result) or result < 0.0 or not upper_ok:
        interval = "[0, 1)" if strict_upper else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _nonnegative_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _maybe_scalar(values: np.ndarray, original: ArrayLike) -> float | np.ndarray:
    return float(values) if np.asarray(original).ndim == 0 else values


def _horizon_value(horizon: float) -> float:
    value = float(horizon)
    if np.isnan(value) or value <= 0.0:
        raise ValueError("horizon must be positive")
    return value


def _growth(reserve: float, duration: float, premium: float, force: float) -> float:
    if force == 0.0:
        return reserve + premium * duration
    return (reserve + premium / force) * math.exp(force * duration) - premium / force


def _time_to_barrier(reserve: float, barrier: float, premium: float, force: float) -> float:
    if reserve >= barrier:
        return 0.0
    if force == 0.0:
        return (barrier - reserve) / premium
    return math.log((barrier + premium / force) / (reserve + premium / force)) / force


def barrier_dividend_payment_mean(
    *,
    claim_arrival_rate: float,
    interest_force: float = 0.0,
    dividend_rate: float = 1.0,
) -> float:
    """Mean of one barrier payment period.

    The normalized Loisel payment is ``D = int_0^Delta exp(delta s) ds`` with
    ``Delta`` exponentially distributed with parameter ``lambda``. Multiplying
    by `dividend_rate` recovers the classical no-interest barrier amount
    ``c Delta`` when ``delta=0`` and `dividend_rate=c`.
    """

    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    force = _nonnegative_float(interest_force, "interest_force")
    rate = _nonnegative_float(dividend_rate, "dividend_rate")
    if rate == 0.0:
        return 0.0
    if force == 0.0:
        return rate / arrival
    if arrival <= force:
        raise ValueError("claim_arrival_rate must exceed interest_force for finite mean")
    return rate / (arrival - force)


def barrier_dividend_payment_cdf(
    x: ArrayLike,
    *,
    claim_arrival_rate: float,
    interest_force: float = 0.0,
    dividend_rate: float = 1.0,
) -> float | np.ndarray:
    """Distribution of one normalized dividend payment period."""

    values = _nonnegative_array(x, "x")
    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    force = _nonnegative_float(interest_force, "interest_force")
    rate = _nonnegative_float(dividend_rate, "dividend_rate")
    if rate == 0.0:
        return _maybe_scalar(np.ones_like(values), x)
    if force == 0.0:
        cdf = 1.0 - np.exp(-arrival * values / rate)
    else:
        cdf = 1.0 - (1.0 + force * values / rate) ** (-arrival / force)
    return _maybe_scalar(np.clip(cdf, 0.0, 1.0), x)


def barrier_dividend_period_count_pmf(
    max_periods: int,
    *,
    hit_probability: float,
    continuation_probability: float,
) -> np.ndarray:
    """PMF of the number of dividend payment periods under the renewal model."""

    maximum = int(max_periods)
    if maximum != max_periods or maximum < 0:
        raise ValueError("max_periods must be a non-negative integer")
    hit = _probability(hit_probability, "hit_probability")
    continuation = _probability(
        continuation_probability,
        "continuation_probability",
        strict_upper=True,
    )
    pmf = np.zeros(maximum + 1, dtype=float)
    pmf[0] = 1.0 - hit
    if maximum:
        periods = np.arange(maximum, dtype=float)
        pmf[1:] = hit * (1.0 - continuation) * continuation**periods
    return pmf


def expected_cumulative_barrier_dividends(
    *,
    hit_probability: float,
    continuation_probability: float,
    claim_arrival_rate: float,
    interest_force: float = 0.0,
    dividend_rate: float = 1.0,
    convention: DividendMeanConvention = "renewal",
) -> float:
    """Expected cumulative dividends from the geometric barrier-cycle structure."""

    hit = _probability(hit_probability, "hit_probability")
    continuation = _probability(
        continuation_probability,
        "continuation_probability",
        strict_upper=True,
    )
    period_mean = barrier_dividend_payment_mean(
        claim_arrival_rate=claim_arrival_rate,
        interest_force=interest_force,
        dividend_rate=dividend_rate,
    )
    if convention == "renewal":
        return hit * period_mean / (1.0 - continuation)
    if convention == "loisel":
        return hit * continuation * period_mean / (1.0 - continuation)
    raise ValueError("convention must be 'renewal' or 'loisel'")


def barrier_dividend_compound_geometric_cdf(
    x: ArrayLike,
    *,
    hit_probability: float,
    continuation_probability: float,
    claim_arrival_rate: float,
    dividend_rate: float = 1.0,
) -> float | np.ndarray:
    """CDF of cumulative dividends for exponential no-interest payment periods."""

    values = _nonnegative_array(x, "x")
    hit = _probability(hit_probability, "hit_probability")
    continuation = _probability(
        continuation_probability,
        "continuation_probability",
        strict_upper=True,
    )
    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    rate = _nonnegative_float(dividend_rate, "dividend_rate")
    if rate == 0.0:
        return _maybe_scalar(np.ones_like(values), x)

    scaled = arrival * values / rate
    cdf = 1.0 - hit * np.exp(-(1.0 - continuation) * scaled)
    return _maybe_scalar(np.clip(cdf, 0.0, 1.0), x)


def barrier_hit_probability_exponential_interest_force(
    initial_capital: ArrayLike,
    *,
    barrier: float,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
) -> float | np.ndarray:
    """Probability of reaching a horizontal barrier before ruin."""

    surplus = _nonnegative_array(initial_capital, "initial_capital")
    level = _positive_float(barrier, "barrier")
    gain = np.maximum(level - surplus, 0.0)
    probability = win_first_probability_exponential_interest_force(
        surplus,
        gain,
        premium_rate=premium_rate,
        claim_arrival_rate=claim_arrival_rate,
        claim_rate=claim_rate,
        interest_force=interest_force,
    )
    probability = np.where(surplus >= level, 1.0, probability)
    return _maybe_scalar(np.asarray(probability, dtype=float), initial_capital)


def barrier_continuation_probability_exponential_interest_force(
    *,
    barrier: float,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> float:
    """Compute ``E[WF(b-W, W)]`` for exponential claims."""

    level = _nonnegative_float(barrier, "barrier")
    premium = _positive_float(premium_rate, "premium_rate")
    arrival = _nonnegative_float(claim_arrival_rate, "claim_arrival_rate")
    severity_rate = _positive_float(claim_rate, "claim_rate")
    force = _nonnegative_float(interest_force, "interest_force")
    if level == 0.0:
        return 0.0

    def integrand(claim: float) -> float:
        return (
            win_first_probability_exponential_interest_force(
                level - claim,
                claim,
                premium_rate=premium,
                claim_arrival_rate=arrival,
                claim_rate=severity_rate,
                interest_force=force,
            )
            * severity_rate
            * math.exp(-severity_rate * claim)
        )

    value = integrate.quad(
        integrand,
        0.0,
        level,
        epsabs=_positive_float(epsabs, "epsabs"),
        epsrel=_positive_float(epsrel, "epsrel"),
        limit=100,
    )[0]
    return float(np.clip(value, 0.0, 1.0))


def barrier_dividend_analytic_exponential_interest_force(
    *,
    initial_capital: float,
    barrier: float,
    premium_rate: float,
    claim_arrival_rate: float,
    claim_rate: float,
    interest_force: float = 0.0,
    dividend_rate: float = 1.0,
) -> BarrierDividendAnalyticResult:
    """Analytic barrier-dividend quantities for exponential claims."""

    surplus = _nonnegative_float(initial_capital, "initial_capital")
    level = _positive_float(barrier, "barrier")
    if surplus > level:
        raise ValueError("initial_capital must not exceed barrier")
    arrival = _positive_float(claim_arrival_rate, "claim_arrival_rate")
    force = _nonnegative_float(interest_force, "interest_force")
    rate = _nonnegative_float(dividend_rate, "dividend_rate")
    hit = float(
        barrier_hit_probability_exponential_interest_force(
            surplus,
            barrier=level,
            premium_rate=premium_rate,
            claim_arrival_rate=arrival,
            claim_rate=claim_rate,
            interest_force=force,
        )
    )
    continuation = barrier_continuation_probability_exponential_interest_force(
        barrier=level,
        premium_rate=premium_rate,
        claim_arrival_rate=arrival,
        claim_rate=claim_rate,
        interest_force=force,
    )
    period_mean = barrier_dividend_payment_mean(
        claim_arrival_rate=arrival,
        interest_force=force,
        dividend_rate=rate,
    )
    expected = expected_cumulative_barrier_dividends(
        hit_probability=hit,
        continuation_probability=continuation,
        claim_arrival_rate=arrival,
        interest_force=force,
        dividend_rate=rate,
        convention="renewal",
    )
    loisel = expected_cumulative_barrier_dividends(
        hit_probability=hit,
        continuation_probability=continuation,
        claim_arrival_rate=arrival,
        interest_force=force,
        dividend_rate=rate,
        convention="loisel",
    )
    return BarrierDividendAnalyticResult(
        initial_capital=surplus,
        barrier=level,
        hit_probability=hit,
        continuation_probability=continuation,
        claim_arrival_rate=arrival,
        interest_force=force,
        dividend_rate=rate,
        period_mean=period_mean,
        expected_dividends=expected,
        loisel_expected_dividends=loisel,
    )


def simulate_barrier_dividend_path(
    claim_distribution: ClaimDistribution,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    barrier: float,
    interest_force: float = 0.0,
    horizon: float = math.inf,
    max_claims: int = 1_000_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> BarrierDividendPath:
    """Simulate one Cramer-Lundberg path with a horizontal dividend barrier."""

    if not isinstance(claim_distribution, ClaimDistribution):
        raise TypeError("claim_distribution must be a ClaimDistribution")
    reserve = _nonnegative_float(initial_capital, "initial_capital")
    premium = _positive_float(premium_rate, "premium_rate")
    arrival = _nonnegative_float(claim_arrival_rate, "claim_arrival_rate")
    level = _positive_float(barrier, "barrier")
    if reserve > level:
        raise ValueError("initial_capital must not exceed barrier")
    force = _nonnegative_float(interest_force, "interest_force")
    time_horizon = _horizon_value(horizon)
    maximum_claims = int(max_claims)
    if maximum_claims != max_claims or maximum_claims <= 0:
        raise ValueError("max_claims must be a positive integer")
    if arrival == 0.0 and math.isinf(time_horizon):
        raise ValueError("infinite horizon requires positive claim_arrival_rate")
    generator = np.random.default_rng(seed) if rng is None else rng

    t = 0.0
    total_dividends = 0.0
    times = [0.0]
    reserves = [reserve]
    dividend_times = [0.0]
    cumulative = [0.0]
    claim_times: list[float] = []
    claim_sizes: list[float] = []
    ruin_time: float | None = None

    def append_state(time: float, value: float) -> None:
        times.append(float(time))
        reserves.append(float(value))

    def append_dividend(time: float) -> None:
        dividend_times.append(float(time))
        cumulative.append(float(total_dividends))

    for _ in range(maximum_claims):
        if t >= time_horizon:
            break
        interarrival = generator.exponential(1.0 / arrival) if arrival > 0.0 else math.inf
        remaining_horizon = time_horizon - t
        if interarrival > remaining_horizon:
            hit_time = _time_to_barrier(reserve, level, premium, force)
            if hit_time < remaining_horizon:
                reserve = level
                total_dividends += (premium + force * level) * (remaining_horizon - hit_time)
            else:
                reserve = min(level, _growth(reserve, remaining_horizon, premium, force))
            t = time_horizon
            append_state(t, reserve)
            append_dividend(t)
            break

        hit_time = _time_to_barrier(reserve, level, premium, force)
        if interarrival <= hit_time:
            reserve = _growth(reserve, interarrival, premium, force)
            t += interarrival
            append_state(t, reserve)
        else:
            if hit_time > 0.0:
                t += hit_time
                reserve = level
                append_state(t, reserve)
            dividend_duration = interarrival - hit_time
            total_dividends += (premium + force * level) * dividend_duration
            t += dividend_duration
            reserve = level
            append_state(t, reserve)
            append_dividend(t)

        claim = float(claim_distribution.sample(1, rng=generator)[0])
        claim_times.append(t)
        claim_sizes.append(claim)
        reserve -= claim
        append_state(t, reserve)
        if reserve < 0.0:
            ruin_time = float(t)
            append_dividend(t)
            break

    else:
        raise RuntimeError("maximum number of claims reached before ruin or horizon")

    return BarrierDividendPath(
        times=np.asarray(times, dtype=float),
        reserves=np.asarray(reserves, dtype=float),
        dividend_times=np.asarray(dividend_times, dtype=float),
        cumulative_dividends=np.asarray(cumulative, dtype=float),
        claim_times=np.asarray(claim_times, dtype=float),
        claim_sizes=np.asarray(claim_sizes, dtype=float),
        ruin_time=ruin_time,
        horizon=time_horizon,
        initial_capital=float(initial_capital),
        barrier=level,
        premium_rate=premium,
        claim_arrival_rate=arrival,
        interest_force=force,
    )


def estimate_barrier_dividends(
    claim_distribution: ClaimDistribution,
    *,
    initial_capital: float,
    premium_rate: float,
    claim_arrival_rate: float,
    barrier: float,
    interest_force: float = 0.0,
    horizon: float = math.inf,
    n_simulations: int = 10_000,
    max_claims: int = 1_000_000,
    seed: int | None = None,
) -> BarrierDividendEstimate:
    """Monte Carlo estimate of cumulative dividends under a horizontal barrier."""

    n_paths = int(n_simulations)
    if n_paths != n_simulations or n_paths <= 0:
        raise ValueError("n_simulations must be a positive integer")
    generator = np.random.default_rng(seed)
    totals = np.empty(n_paths, dtype=float)
    ruin_times = np.full(n_paths, np.inf, dtype=float)
    for index in range(n_paths):
        path = simulate_barrier_dividend_path(
            claim_distribution,
            initial_capital=initial_capital,
            premium_rate=premium_rate,
            claim_arrival_rate=claim_arrival_rate,
            barrier=barrier,
            interest_force=interest_force,
            horizon=horizon,
            max_claims=max_claims,
            rng=generator,
        )
        totals[index] = path.total_dividends
        if path.ruin_time is not None:
            ruin_times[index] = path.ruin_time

    mean = float(np.mean(totals))
    standard_error = 0.0 if n_paths == 1 else float(np.std(totals, ddof=1) / math.sqrt(n_paths))
    return BarrierDividendEstimate(
        total_dividends=totals,
        ruin_times=ruin_times,
        mean_dividends=mean,
        standard_error=standard_error,
        probability_ruin=float(np.mean(np.isfinite(ruin_times))),
        n_simulations=n_paths,
        horizon=_horizon_value(horizon),
        barrier=_positive_float(barrier, "barrier"),
    )
