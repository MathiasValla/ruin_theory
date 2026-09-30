"""Order-statistic point processes and ordered/dual risk formulas."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
import operator
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, special, stats

from .distributions import ClaimDistribution


FloatFunction = Callable[[float], float]


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


def _nonnegative_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class OrderStatisticPointProcess:
    """Counting process with the order-statistic property."""

    count_pmf_function: Callable[[int, float], float]
    conditional_cdf_function: Callable[[float, float], float]
    mean_function: Callable[[float], float]
    name: str = "ospp"
    finite_max_count: int | None = None

    def count_pmf(self, n: int, t: float) -> float:
        count = _nonnegative_int(n, "n")
        horizon = _nonnegative_float(t, "t")
        if self.finite_max_count is not None and count > self.finite_max_count:
            return 0.0
        value = float(self.count_pmf_function(count, horizon))
        if not np.isfinite(value) or value < -1e-14:
            raise ValueError("count_pmf_function returned an invalid probability")
        return max(value, 0.0)

    def conditional_cdf(self, s: float, t: float) -> float:
        horizon = _positive_float(t, "t")
        value = _nonnegative_float(s, "s")
        if value <= 0.0:
            return 0.0
        if value >= horizon:
            return 1.0
        result = float(self.conditional_cdf_function(value, horizon))
        if not np.isfinite(result):
            raise ValueError("conditional_cdf_function returned a non-finite value")
        return float(np.clip(result, 0.0, 1.0))

    def mean(self, t: float) -> float:
        return _nonnegative_float(self.mean_function(_nonnegative_float(t, "t")), "mean(t)")

    def count_probabilities(
        self,
        t: float,
        *,
        max_count: int | None = None,
        tail_tol: float = 1e-12,
    ) -> np.ndarray:
        horizon = _nonnegative_float(t, "t")
        tolerance = _positive_float(tail_tol, "tail_tol")
        if tolerance >= 1.0:
            raise ValueError("tail_tol must lie in (0, 1)")
        if max_count is None:
            if self.finite_max_count is not None:
                max_count = self.finite_max_count
            else:
                probabilities: list[float] = []
                total = 0.0
                mean = self.mean(horizon)
                # A Poisson-width cutoff is not valid for overdispersed count laws.
                limit = max(100_000, int(math.ceil(mean + 12.0 * math.sqrt(mean + 1.0))))
                for count in range(limit + 1):
                    probability = self.count_pmf(count, horizon)
                    probabilities.append(probability)
                    total += probability
                    if total > 1.0 + 1e-10:
                        raise ValueError("count probabilities sum to more than one")
                    if count >= mean and 1.0 - total <= tolerance:
                        break
                else:
                    raise ValueError("count tail tolerance not reached; provide explicit max_count")
                return np.asarray(probabilities, dtype=float)
        max_count = _nonnegative_int(max_count, "max_count")
        return np.array([self.count_pmf(count, horizon) for count in range(max_count + 1)], dtype=float)


def poisson_ospp(rate: float) -> OrderStatisticPointProcess:
    """Homogeneous Poisson OSPP."""

    rate = _positive_float(rate, "rate")
    return OrderStatisticPointProcess(
        count_pmf_function=lambda n, t: stats.poisson.pmf(n, rate * t),
        conditional_cdf_function=lambda s, t: s / t,
        mean_function=lambda t: rate * t,
        name="poisson_ospp",
    )


def inhomogeneous_poisson_ospp(
    cumulative_intensity: FloatFunction,
) -> OrderStatisticPointProcess:
    """Inhomogeneous Poisson OSPP with cumulative intensity ``mu(t)``."""

    def mean(t: float) -> float:
        value = float(cumulative_intensity(t))
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("cumulative_intensity must be finite and non-negative")
        return value

    return OrderStatisticPointProcess(
        count_pmf_function=lambda n, t: stats.poisson.pmf(n, mean(t)),
        conditional_cdf_function=lambda s, t: mean(s) / mean(t) if mean(t) > 0.0 else 0.0,
        mean_function=mean,
        name="inhomogeneous_poisson_ospp",
    )


def negative_binomial_ospp(gamma: float, beta: float) -> OrderStatisticPointProcess:
    """Mixed Poisson OSPP with Gamma(``gamma``, ``beta``) mixing variable."""

    gamma = _positive_float(gamma, "gamma")
    beta = _positive_float(beta, "beta")

    def count_pmf(n: int, t: float) -> float:
        p = beta / (t + beta)
        return stats.nbinom.pmf(n, gamma, p)

    return OrderStatisticPointProcess(
        count_pmf_function=count_pmf,
        conditional_cdf_function=lambda s, t: s / t,
        mean_function=lambda t: gamma * t / beta,
        name="negative_binomial_ospp",
    )


def linear_birth_immigration_ospp(
    immigration_rate: float,
    birth_rate: float,
) -> OrderStatisticPointProcess:
    """Linear birth process with immigration, represented as an OSPP."""

    immigration_rate = _positive_float(immigration_rate, "immigration_rate")
    birth_rate = _positive_float(birth_rate, "birth_rate")
    shape = immigration_rate / birth_rate

    def count_pmf(n: int, t: float) -> float:
        p = math.exp(-birth_rate * t)
        return stats.nbinom.pmf(n, shape, p)

    return OrderStatisticPointProcess(
        count_pmf_function=count_pmf,
        conditional_cdf_function=lambda s, t: math.exp(-birth_rate * (t - s))
        * math.expm1(-birth_rate * s) / math.expm1(-birth_rate * t),
        mean_function=lambda t: shape * math.expm1(birth_rate * t),
        name="linear_birth_immigration_ospp",
    )


def linear_death_ospp(initial_size: int, death_rate: float) -> OrderStatisticPointProcess:
    """Linear death counting process, represented as an OSPP."""

    size = _nonnegative_int(initial_size, "initial_size")
    if size == 0:
        raise ValueError("initial_size must be positive")
    death_rate = _positive_float(death_rate, "death_rate")

    def count_pmf(n: int, t: float) -> float:
        p = -math.expm1(-death_rate * t)
        return stats.binom.pmf(n, size, p)

    return OrderStatisticPointProcess(
        count_pmf_function=count_pmf,
        conditional_cdf_function=lambda s, t: math.expm1(-death_rate * s)
        / math.expm1(-death_rate * t),
        mean_function=lambda t: -size * math.expm1(-death_rate * t),
        name="linear_death_ospp",
        finite_max_count=size,
    )


def uniform_order_stat_rect_probability(lower: ArrayLike, upper: ArrayLike) -> float:
    """Probability that uniform order statistics fall in coordinate rectangles."""

    lower_values = np.asarray(lower, dtype=float).ravel()
    upper_values = np.asarray(upper, dtype=float).ravel()
    if lower_values.shape != upper_values.shape:
        raise ValueError("lower and upper must have the same shape")
    n = lower_values.size
    if n == 0:
        return 1.0
    if np.any(~np.isfinite(lower_values)) or np.any(~np.isfinite(upper_values)):
        raise ValueError("bounds must be finite")
    lower_values = np.clip(lower_values, 0.0, 1.0)
    upper_values = np.clip(upper_values, 0.0, 1.0)
    if np.any(lower_values > upper_values):
        return 0.0
    if np.any(np.diff(lower_values) < -1e-12) or np.any(np.diff(upper_values) < -1e-12):
        raise ValueError("lower and upper bounds must be non-decreasing")

    if n >= 20:
        # The same recurrence loses all useful digits through alternating
        # binomial terms at high order. Preserve guard digits before clipping.
        with localcontext() as context:
            context.prec = n + 30
            lower_decimal = [Decimal(float(value)) for value in lower_values]
            upper_decimal = [Decimal(float(value)) for value in upper_values]
            masses = [Decimal(1)]
            for degree in range(1, n + 1):
                total = Decimal(0)
                for split in range(degree):
                    width = max(upper_decimal[split] - lower_decimal[degree - 1], Decimal(0))
                    total += math.comb(degree, split) * (-width) ** (degree - split) * masses[split]
                masses.append(-total)
            return float(np.clip(float(masses[n]), 0.0, 1.0))

    probabilities = np.zeros(n + 1, dtype=float)
    probabilities[0] = 1.0
    for degree in range(1, n + 1):
        total = 0.0
        u_degree = lower_values[degree - 1]
        for split in range(degree):
            width = max(upper_values[split] - u_degree, 0.0)
            total += (
                math.comb(degree, split)
                * width ** (degree - split)
                * ((-1.0) ** (degree - split))
                * probabilities[split]
            )
        probabilities[degree] = -total
    return float(np.clip(probabilities[n], 0.0, 1.0))


@dataclass(frozen=True)
class PremiumBoundary:
    """Nondecreasing premium accumulation boundary ``h(t)``."""

    function: FloatFunction
    inverse: FloatFunction
    derivative: FloatFunction | None = None
    name: str = "premium_boundary"

    @classmethod
    def linear(cls, rate: float) -> "PremiumBoundary":
        rate = _positive_float(rate, "rate")
        return cls(
            function=lambda t: rate * max(float(t), 0.0),
            inverse=lambda x: max(float(x), 0.0) / rate,
            derivative=lambda _t: rate,
            name="linear_premium_boundary",
        )

    def value(self, t: float) -> float:
        return _nonnegative_float(self.function(_nonnegative_float(t, "t")), "h(t)")

    def inverse_value(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return _nonnegative_float(self.inverse(float(x)), "h_inverse(x)")

    def derivative_value(self, t: float) -> float:
        if self.derivative is None:
            step = max(1e-5, abs(t) * 1e-5)
            return (self.value(t + step) - self.value(max(0.0, t - step))) / (
                t + step - max(0.0, t - step)
            )
        return _nonnegative_float(self.derivative(_nonnegative_float(t, "t")), "h'(t)")


def _ordered_rect_for_partial_sums(
    partial_sums: np.ndarray,
    ospp: OrderStatisticPointProcess,
    boundary: PremiumBoundary,
    initial_capital: float,
    upper_barrier: float,
    horizon: float,
) -> float:
    n = partial_sums.size
    if n == 0:
        return 1.0
    previous = np.r_[0.0, partial_sums[:-1]]
    lower = np.array(
        [
            ospp.conditional_cdf(
                boundary.inverse_value(max(partial_sums[index] - initial_capital, 0.0)),
                horizon,
            )
            for index in range(n)
        ],
        dtype=float,
    )
    upper = np.array(
        [
            ospp.conditional_cdf(
                boundary.inverse_value(previous[index] + upper_barrier - initial_capital),
                horizon,
            )
            for index in range(n)
        ],
        dtype=float,
    )
    return uniform_order_stat_rect_probability(lower, upper)


def ordered_two_sided_survival(
    ospp: OrderStatisticPointProcess,
    claim_distribution: ClaimDistribution,
    boundary: PremiumBoundary,
    initial_capital: float,
    upper_barrier: float,
    horizon: float,
    *,
    max_claims: int | None = None,
    n_simulations: int = 0,
    tail_tol: float = 1e-12,
    seed: int | None = None,
) -> float:
    """Finite-time survival inside two absorbing barriers in the ordered model."""

    initial = _nonnegative_float(initial_capital, "initial_capital")
    barrier = _positive_float(upper_barrier, "upper_barrier")
    if initial >= barrier:
        raise ValueError("initial_capital must be below upper_barrier")
    time = _nonnegative_float(horizon, "horizon")
    if not isinstance(ospp, OrderStatisticPointProcess):
        raise TypeError("ospp must be an OrderStatisticPointProcess")
    if not isinstance(boundary, PremiumBoundary):
        raise TypeError("boundary must be a PremiumBoundary")
    if not isinstance(claim_distribution, ClaimDistribution):
        raise TypeError("claim_distribution must be a ClaimDistribution")
    simulations = _nonnegative_int(n_simulations, "n_simulations")
    probabilities = ospp.count_probabilities(time, max_count=max_claims, tail_tol=tail_tol)
    rng = np.random.default_rng(seed)
    h_t = boundary.value(time)
    lower_terminal = h_t - (barrier - initial)
    upper_terminal = initial + h_t

    total = 0.0
    for n, count_probability in enumerate(probabilities):
        if count_probability == 0.0:
            continue
        if n == 0:
            indicator = lower_terminal <= 0.0 <= upper_terminal
            total += count_probability * float(indicator)
            continue

        if claim_distribution.name == "deterministic":
            value = float(claim_distribution.metadata["value"])
            partial_sums = value * np.arange(1, n + 1, dtype=float)
            if not (lower_terminal <= partial_sums[-1] <= upper_terminal):
                continue
            total += count_probability * _ordered_rect_for_partial_sums(
                partial_sums,
                ospp,
                boundary,
                initial,
                barrier,
                time,
            )
            continue

        if simulations <= 0:
            raise ValueError(
                "n_simulations must be positive for non-deterministic claim distributions"
            )
        samples = claim_distribution.sample(n * simulations, rng=rng).reshape(simulations, n)
        partial = np.cumsum(samples, axis=1)
        terminal = partial[:, -1]
        mask = (terminal >= lower_terminal) & (terminal <= upper_terminal)
        if not np.any(mask):
            continue
        terms = [
            _ordered_rect_for_partial_sums(row, ospp, boundary, initial, barrier, time)
            for row in partial[mask]
        ]
        total += count_probability * float(np.sum(terms) / simulations)
    return float(np.clip(total, 0.0, 1.0))


def _gamma_sum_density_and_dirichlet_shape(
    distribution: ClaimDistribution,
    n: int,
    x: float,
) -> tuple[float, float]:
    if distribution.name == "exponential":
        rate = float(distribution.metadata["rate"])
        return float(stats.gamma.pdf(x, a=n, scale=1.0 / rate)), 1.0
    if distribution.name == "gamma":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        return float(stats.gamma.pdf(x, a=n * shape, scale=scale)), shape
    if distribution.name == "erlang":
        shape = float(distribution.metadata["shape"])
        scale = 1.0 / float(distribution.metadata["rate"])
        return float(stats.gamma.pdf(x, a=n * shape, scale=scale)), shape
    raise NotImplementedError("win-first density currently supports exponential/gamma severities")


def ordered_win_first_time_density(
    ospp: OrderStatisticPointProcess,
    claim_distribution: ClaimDistribution,
    boundary: PremiumBoundary,
    initial_capital: float,
    upper_barrier: float,
    t: float,
    *,
    max_claims: int | None = None,
    n_conditional_samples: int = 2048,
    tail_tol: float = 1e-12,
    seed: int | None = None,
) -> float:
    """Density of the ordered win-first time through the upper barrier."""

    initial = _nonnegative_float(initial_capital, "initial_capital")
    barrier = _positive_float(upper_barrier, "upper_barrier")
    time = _nonnegative_float(t, "t")
    target = boundary.value(time) - (barrier - initial)
    if target <= 0.0 or time == 0.0:
        return 0.0
    probabilities = ospp.count_probabilities(time, max_count=max_claims, tail_tol=tail_tol)
    n_samples = max(1, _nonnegative_int(n_conditional_samples, "n_conditional_samples"))
    rng = np.random.default_rng(seed)

    total = 0.0
    for n in range(1, probabilities.size):
        count_probability = probabilities[n]
        if count_probability == 0.0:
            continue
        density, shape = _gamma_sum_density_and_dirichlet_shape(claim_distribution, n, target)
        if density == 0.0:
            continue
        if n == 1:
            partial_sums = np.array([target], dtype=float)
            expectation = _ordered_rect_for_partial_sums(
                partial_sums,
                ospp,
                boundary,
                initial,
                barrier,
                time,
            )
        else:
            proportions = rng.dirichlet(np.full(n, shape), size=n_samples)
            partial = np.cumsum(target * proportions, axis=1)
            values = [
                _ordered_rect_for_partial_sums(row, ospp, boundary, initial, barrier, time)
                for row in partial
            ]
            expectation = float(np.mean(values))
        total += count_probability * expectation * boundary.derivative_value(time) * density
    return float(max(total, 0.0))


@dataclass(frozen=True)
class DualRiskProcess:
    """Dual risk process ``W_t = v - a t + sum_{i<=M_t} Y_i``."""

    initial_capital: float
    cost_rate: float
    profit_arrival_rate: float
    profit_distribution: ClaimDistribution
    name: str = "dual_risk"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_capital",
            _positive_float(self.initial_capital, "initial_capital"),
        )
        object.__setattr__(self, "cost_rate", _positive_float(self.cost_rate, "cost_rate"))
        object.__setattr__(
            self,
            "profit_arrival_rate",
            _nonnegative_float(self.profit_arrival_rate, "profit_arrival_rate"),
        )
        if not isinstance(self.profit_distribution, ClaimDistribution):
            raise TypeError("profit_distribution must be a ClaimDistribution")

    @property
    def earliest_ruin_time(self) -> float:
        return self.initial_capital / self.cost_rate


def dual_poisson_exponential_ruin_time_atom(process: DualRiskProcess) -> float:
    """Atom of the dual Poisson-exponential ruin time at ``v/a``."""

    if process.profit_distribution.name != "exponential":
        raise ValueError("requires exponential profit sizes")
    return float(math.exp(-process.profit_arrival_rate * process.earliest_ruin_time))


def dual_poisson_exponential_ruin_time_density(
    process: DualRiskProcess,
    t: ArrayLike,
) -> np.ndarray:
    """Kendall-style density for Poisson arrivals and exponential profits."""

    if process.profit_distribution.name != "exponential":
        raise ValueError("requires exponential profit sizes")
    times = np.asarray(t, dtype=float)
    if np.any(np.isnan(times)) or np.any(times < 0.0):
        raise ValueError("t must contain non-negative values")
    rate = process.profit_arrival_rate
    mu = float(process.profit_distribution.metadata["rate"])
    v = process.initial_capital
    a = process.cost_rate
    values = np.zeros_like(times, dtype=float)
    mask = (times > process.earliest_ruin_time) & np.isfinite(times)
    if np.any(mask) and rate > 0.0:
        tm = times[mask]
        x = a * tm - v
        arrival_root = np.sqrt(rate * tm)
        severity_root = np.sqrt(mu * x)
        z = 2.0 * arrival_root * severity_root
        values[mask] = (
            v
            / tm
            * np.exp(-(arrival_root - severity_root) ** 2)
            * np.sqrt(rate * tm * mu / x)
            * special.ive(1, z)
        )
    return values


def dual_poisson_exponential_ruin_time_cdf(
    process: DualRiskProcess,
    t: ArrayLike,
    *,
    epsabs: float = 1e-9,
) -> np.ndarray:
    """Distribution function of the dual Poisson-exponential ruin time."""

    times = np.asarray(t, dtype=float)
    if np.any(np.isnan(times)) or np.any(times < 0.0):
        raise ValueError("t must contain non-negative values")
    flat = times.ravel()
    values = np.zeros_like(flat, dtype=float)
    atom_time = process.earliest_ruin_time
    atom = dual_poisson_exponential_ruin_time_atom(process)
    for index, time in enumerate(flat):
        if time < atom_time:
            values[index] = 0.0
        elif time == atom_time:
            values[index] = atom
        else:
            integral, _ = integrate.quad(
                lambda x: float(dual_poisson_exponential_ruin_time_density(process, x)),
                atom_time,
                float(time),
                epsabs=epsabs,
            )
            values[index] = atom + float(integral)
    return np.clip(values.reshape(times.shape), 0.0, 1.0)


def dual_mixed_poisson_ruin_time_density(
    count_pmf: Callable[[int, float], float],
    profit_convolution_density: Callable[[int, float], float],
    initial_capital: float,
    cost_rate: float,
    t: float,
    *,
    max_count: int,
) -> float:
    """Mixed-Poisson ordered-dual density using the simplified ``v/t`` formula."""

    v = _positive_float(initial_capital, "initial_capital")
    a = _positive_float(cost_rate, "cost_rate")
    time = _positive_float(t, "t")
    max_n = _nonnegative_int(max_count, "max_count")
    if time <= v / a:
        return 0.0
    x = a * time - v
    total = 0.0
    for n in range(1, max_n + 1):
        total += float(count_pmf(n, time)) * float(profit_convolution_density(n, x))
    return float(max(v / time * total, 0.0))
