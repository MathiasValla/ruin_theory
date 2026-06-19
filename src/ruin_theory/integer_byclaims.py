"""Discrete INAR/BINAR by-claim simulation models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
from scipy import stats

from .distributions import ClaimDistribution
from .results import RuinEstimate


def _rng(seed: int | None | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_int(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_vector(values: object, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _claim_distributions(
    values: tuple[ClaimDistribution, ...],
    size: int,
) -> tuple[ClaimDistribution, ...]:
    distributions = tuple(values)
    if len(distributions) != size:
        raise ValueError(f"expected {size} claim distributions")
    if not all(isinstance(distribution, ClaimDistribution) for distribution in distributions):
        raise TypeError("claim distributions must be ClaimDistribution instances")
    return distributions


@dataclass(frozen=True)
class IntegerByClaimPath:
    """One simulated discrete reserve path with dependent by-claim counts."""

    reserves: np.ndarray
    primary_counts: np.ndarray
    byclaim_counts: np.ndarray
    primary_losses: np.ndarray
    byclaim_losses: np.ndarray
    ruin_time: int | None
    initial_capital: float
    premium_per_period: float

    @property
    def ruined(self) -> bool:
        return self.ruin_time is not None

    @property
    def terminal_reserve(self) -> float:
        return float(self.reserves[-1])

    @property
    def minimum_reserve(self) -> float:
        return float(np.min(self.reserves))


@dataclass(frozen=True)
class INARByClaimModel:
    """Univariate INAR(1) by-claim reserve model.

    Primary counts are iid Poisson with mean `primary_count_mean`. By-claim
    counts satisfy `M_k = rho o M_{k-1} + N_k`, with initialization
    `M_0 = rho o Q_0 + N_0` and `Q_0 ~ Poisson(initial_byclaim_mean)`.
    """

    initial_capital: float
    premium_per_period: float
    primary_count_mean: float
    initial_byclaim_mean: float
    reproduction: float
    primary_distribution: ClaimDistribution
    byclaim_distribution: ClaimDistribution
    name: str = "inar_byclaim"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_capital",
            _nonnegative_float(self.initial_capital, "initial_capital"),
        )
        object.__setattr__(
            self,
            "premium_per_period",
            _nonnegative_float(self.premium_per_period, "premium_per_period"),
        )
        object.__setattr__(
            self,
            "primary_count_mean",
            _nonnegative_float(self.primary_count_mean, "primary_count_mean"),
        )
        object.__setattr__(
            self,
            "initial_byclaim_mean",
            _nonnegative_float(self.initial_byclaim_mean, "initial_byclaim_mean"),
        )
        object.__setattr__(self, "reproduction", _probability(self.reproduction, "reproduction"))
        if not isinstance(self.primary_distribution, ClaimDistribution):
            raise TypeError("primary_distribution must be a ClaimDistribution")
        if not isinstance(self.byclaim_distribution, ClaimDistribution):
            raise TypeError("byclaim_distribution must be a ClaimDistribution")

    @property
    def dimension(self) -> int:
        return 1

    @property
    def primary_distributions(self) -> tuple[ClaimDistribution, ...]:
        return (self.primary_distribution,)

    @property
    def byclaim_distributions(self) -> tuple[ClaimDistribution, ...]:
        return (self.byclaim_distribution,)

    def expected_byclaim_counts(self, periods: int) -> np.ndarray:
        period_count = _positive_int(periods, "periods")
        values = np.empty(period_count, dtype=float)
        previous = self.reproduction * self.initial_byclaim_mean + self.primary_count_mean
        values[0] = previous
        for index in range(1, period_count):
            previous = self.reproduction * previous + self.primary_count_mean
            values[index] = previous
        return values

    def expected_terminal_reserve(self, periods: int) -> float:
        period_count = _positive_int(periods, "periods")
        primary_cost = period_count * self.primary_count_mean * self.primary_distribution.mean()
        byclaim_cost = (
            self.expected_byclaim_counts(period_count).sum()
            * self.byclaim_distribution.mean()
        )
        return float(
            self.initial_capital
            + period_count * self.premium_per_period
            - primary_cost
            - byclaim_cost,
        )


@dataclass(frozen=True)
class BINARByClaimModel:
    """Bivariate INAR(1) by-claim reserve model."""

    initial_capital: float
    premium_per_period: float
    primary_count_means: tuple[float, float]
    initial_byclaim_means: tuple[float, float]
    reproduction_matrix: tuple[tuple[float, float], tuple[float, float]]
    primary_distributions: tuple[ClaimDistribution, ClaimDistribution]
    byclaim_distributions: tuple[ClaimDistribution, ClaimDistribution]
    name: str = "binar_byclaim"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_capital",
            _nonnegative_float(self.initial_capital, "initial_capital"),
        )
        object.__setattr__(
            self,
            "premium_per_period",
            _nonnegative_float(self.premium_per_period, "premium_per_period"),
        )
        primary_means = _nonnegative_vector(self.primary_count_means, "primary_count_means", 2)
        initial_means = _nonnegative_vector(self.initial_byclaim_means, "initial_byclaim_means", 2)
        matrix = np.asarray(self.reproduction_matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("reproduction_matrix must have shape (2, 2)")
        if np.any(~np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
            raise ValueError("reproduction_matrix entries must lie in [0, 1]")
        object.__setattr__(self, "primary_count_means", tuple(float(x) for x in primary_means))
        object.__setattr__(self, "initial_byclaim_means", tuple(float(x) for x in initial_means))
        object.__setattr__(
            self,
            "reproduction_matrix",
            tuple(tuple(float(x) for x in row) for row in matrix),
        )
        object.__setattr__(
            self,
            "primary_distributions",
            _claim_distributions(self.primary_distributions, 2),
        )
        object.__setattr__(
            self,
            "byclaim_distributions",
            _claim_distributions(self.byclaim_distributions, 2),
        )

    @property
    def dimension(self) -> int:
        return 2

    def reproduction_array(self) -> np.ndarray:
        return np.asarray(self.reproduction_matrix, dtype=float)

    def expected_byclaim_counts(self, periods: int) -> np.ndarray:
        period_count = _positive_int(periods, "periods")
        matrix = self.reproduction_array()
        innovations = np.asarray(self.primary_count_means, dtype=float)
        previous = matrix @ np.asarray(self.initial_byclaim_means, dtype=float) + innovations
        values = np.empty((period_count, 2), dtype=float)
        values[0] = previous
        for index in range(1, period_count):
            previous = matrix @ previous + innovations
            values[index] = previous
        return values

    def expected_terminal_reserve(self, periods: int) -> float:
        period_count = _positive_int(periods, "periods")
        primary_means = np.asarray(self.primary_count_means, dtype=float)
        primary_severities = np.array(
            [distribution.mean() for distribution in self.primary_distributions],
        )
        byclaim_severities = np.array(
            [distribution.mean() for distribution in self.byclaim_distributions],
        )
        primary_cost = period_count * float(np.dot(primary_means, primary_severities))
        byclaim_cost = float(
            (self.expected_byclaim_counts(period_count) @ byclaim_severities).sum(),
        )
        return float(
            self.initial_capital
            + period_count * self.premium_per_period
            - primary_cost
            - byclaim_cost,
        )


def _compound_sums(
    distribution: ClaimDistribution,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    integer_counts = np.asarray(counts, dtype=int)
    if np.any(integer_counts < 0):
        raise ValueError("counts must be non-negative")
    result = np.zeros(integer_counts.size, dtype=float)
    positive = integer_counts > 0
    if not np.any(positive):
        return result

    if distribution.name == "deterministic":
        return integer_counts.astype(float) * float(distribution.metadata["value"])
    if distribution.name == "exponential":
        rate = float(distribution.metadata["rate"])
        result[positive] = rng.gamma(shape=integer_counts[positive], scale=1.0 / rate)
        return result
    if distribution.name in {"gamma", "erlang"}:
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata.get("scale", 1.0 / distribution.metadata["rate"]))
        result[positive] = rng.gamma(shape=integer_counts[positive] * shape, scale=scale)
        return result

    total = int(integer_counts.sum())
    samples = np.asarray(distribution.sample(total, rng=rng), dtype=float)
    if samples.shape != (total,):
        raise ValueError("claim distribution must return one value per sampled claim")
    if np.any(~np.isfinite(samples)) or np.any(samples < 0.0):
        raise ValueError("claim severities must be finite and non-negative")
    starts = np.r_[0, np.cumsum(integer_counts[:-1])]
    result[positive] = np.add.reduceat(samples, starts[positive])
    return result


def _sample_initial_counts(
    model: INARByClaimModel | BINARByClaimModel,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, INARByClaimModel):
        primary = rng.poisson(model.primary_count_mean, size=(n_paths, 1))
        ancestors = rng.poisson(model.initial_byclaim_mean, size=n_paths)
        byclaims = rng.binomial(ancestors, model.reproduction).reshape(n_paths, 1) + primary
        return primary.astype(int), byclaims.astype(int)

    primary_means = np.asarray(model.primary_count_means, dtype=float)
    primary = rng.poisson(primary_means, size=(n_paths, 2))
    ancestors = rng.poisson(np.asarray(model.initial_byclaim_means, dtype=float), size=(n_paths, 2))
    matrix = model.reproduction_array()
    byclaims = np.empty((n_paths, 2), dtype=int)
    for target in range(2):
        byclaims[:, target] = primary[:, target]
        for source in range(2):
            byclaims[:, target] += rng.binomial(ancestors[:, source], matrix[target, source])
    return primary.astype(int), byclaims


def _sample_next_counts(
    model: INARByClaimModel | BINARByClaimModel,
    previous_byclaims: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_paths = previous_byclaims.shape[0]
    if isinstance(model, INARByClaimModel):
        primary = rng.poisson(model.primary_count_mean, size=(n_paths, 1))
        byclaims = (
            rng.binomial(previous_byclaims[:, 0], model.reproduction).reshape(n_paths, 1)
            + primary
        )
        return primary.astype(int), byclaims.astype(int)

    primary_means = np.asarray(model.primary_count_means, dtype=float)
    primary = rng.poisson(primary_means, size=(n_paths, 2))
    matrix = model.reproduction_array()
    byclaims = np.empty((n_paths, 2), dtype=int)
    for target in range(2):
        byclaims[:, target] = primary[:, target]
        for source in range(2):
            byclaims[:, target] += rng.binomial(
                previous_byclaims[:, source],
                matrix[target, source],
            )
    return primary.astype(int), byclaims


def _losses_by_type(
    distributions: tuple[ClaimDistribution, ...],
    counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    losses = np.empty(counts.shape, dtype=float)
    for index, distribution in enumerate(distributions):
        losses[:, index] = _compound_sums(distribution, counts[:, index], rng)
    return losses


def _ruined(values: np.ndarray, threshold: float, inclusive: bool) -> np.ndarray:
    return values <= threshold if inclusive else values < threshold


def simulate_integer_byclaim_path(
    model: INARByClaimModel | BINARByClaimModel,
    periods: int,
    *,
    seed: int | None | np.random.Generator = None,
    stop_at_ruin: bool = True,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> IntegerByClaimPath:
    """Simulate one INAR/BINAR by-claim reserve trajectory."""

    period_count = _positive_int(periods, "periods")
    threshold = _finite_float(ruin_threshold, "ruin_threshold")
    rng = _rng(seed)
    reserves = [float(model.initial_capital)]
    primary_counts: list[np.ndarray] = []
    byclaim_counts: list[np.ndarray] = []
    primary_losses: list[np.ndarray] = []
    byclaim_losses: list[np.ndarray] = []
    previous_byclaims: np.ndarray | None = None
    ruin_time: int | None = None

    for period in range(period_count):
        if period == 0:
            primary, byclaims = _sample_initial_counts(model, 1, rng)
        else:
            assert previous_byclaims is not None
            primary, byclaims = _sample_next_counts(model, previous_byclaims, rng)
        previous_byclaims = byclaims
        primary_loss = _losses_by_type(model.primary_distributions, primary, rng)
        byclaim_loss = _losses_by_type(model.byclaim_distributions, byclaims, rng)
        reserve = (
            reserves[-1]
            + model.premium_per_period
            - float(primary_loss.sum() + byclaim_loss.sum())
        )

        primary_counts.append(primary[0].copy())
        byclaim_counts.append(byclaims[0].copy())
        primary_losses.append(primary_loss[0].copy())
        byclaim_losses.append(byclaim_loss[0].copy())
        reserves.append(float(reserve))
        if ruin_time is None and _ruined(np.array([reserve]), threshold, ruin_inclusive)[0]:
            ruin_time = period + 1
            if stop_at_ruin:
                break

    return IntegerByClaimPath(
        reserves=np.asarray(reserves, dtype=float),
        primary_counts=np.asarray(primary_counts, dtype=int),
        byclaim_counts=np.asarray(byclaim_counts, dtype=int),
        primary_losses=np.asarray(primary_losses, dtype=float),
        byclaim_losses=np.asarray(byclaim_losses, dtype=float),
        ruin_time=ruin_time,
        initial_capital=float(model.initial_capital),
        premium_per_period=float(model.premium_per_period),
    )


def simulate_inar_byclaim_path(
    model: INARByClaimModel,
    periods: int,
    *,
    seed: int | None | np.random.Generator = None,
    stop_at_ruin: bool = True,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> IntegerByClaimPath:
    """Simulate one univariate INAR by-claim reserve trajectory."""

    if not isinstance(model, INARByClaimModel):
        raise TypeError("model must be an INARByClaimModel")
    return simulate_integer_byclaim_path(
        model,
        periods,
        seed=seed,
        stop_at_ruin=stop_at_ruin,
        ruin_threshold=ruin_threshold,
        ruin_inclusive=ruin_inclusive,
    )


def simulate_binar_byclaim_path(
    model: BINARByClaimModel,
    periods: int,
    *,
    seed: int | None | np.random.Generator = None,
    stop_at_ruin: bool = True,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> IntegerByClaimPath:
    """Simulate one bivariate INAR by-claim reserve trajectory."""

    if not isinstance(model, BINARByClaimModel):
        raise TypeError("model must be a BINARByClaimModel")
    return simulate_integer_byclaim_path(
        model,
        periods,
        seed=seed,
        stop_at_ruin=stop_at_ruin,
        ruin_threshold=ruin_threshold,
        ruin_inclusive=ruin_inclusive,
    )


def simulate_integer_byclaim_terminal_reserves(
    model: INARByClaimModel | BINARByClaimModel,
    periods: int,
    *,
    n_simulations: int,
    seed: int | None | np.random.Generator = None,
) -> np.ndarray:
    """Vectorized terminal reserves for INAR/BINAR by-claim models."""

    period_count = _positive_int(periods, "periods")
    simulation_count = _positive_int(n_simulations, "n_simulations")
    rng = _rng(seed)
    reserves = np.full(simulation_count, float(model.initial_capital), dtype=float)
    previous_byclaims: np.ndarray | None = None
    for period in range(period_count):
        if period == 0:
            primary, byclaims = _sample_initial_counts(model, simulation_count, rng)
        else:
            assert previous_byclaims is not None
            primary, byclaims = _sample_next_counts(model, previous_byclaims, rng)
        previous_byclaims = byclaims
        primary_losses = _losses_by_type(model.primary_distributions, primary, rng)
        byclaim_losses = _losses_by_type(model.byclaim_distributions, byclaims, rng)
        reserves += (
            model.premium_per_period
            - primary_losses.sum(axis=1)
            - byclaim_losses.sum(axis=1)
        )
    return reserves


def simulate_inar_byclaim_terminal_reserves(
    model: INARByClaimModel,
    periods: int,
    *,
    n_simulations: int,
    seed: int | None | np.random.Generator = None,
) -> np.ndarray:
    """Vectorized terminal reserves for an INAR by-claim model."""

    if not isinstance(model, INARByClaimModel):
        raise TypeError("model must be an INARByClaimModel")
    return simulate_integer_byclaim_terminal_reserves(
        model,
        periods,
        n_simulations=n_simulations,
        seed=seed,
    )


def simulate_binar_byclaim_terminal_reserves(
    model: BINARByClaimModel,
    periods: int,
    *,
    n_simulations: int,
    seed: int | None | np.random.Generator = None,
) -> np.ndarray:
    """Vectorized terminal reserves for a BINAR by-claim model."""

    if not isinstance(model, BINARByClaimModel):
        raise TypeError("model must be a BINARByClaimModel")
    return simulate_integer_byclaim_terminal_reserves(
        model,
        periods,
        n_simulations=n_simulations,
        seed=seed,
    )


def estimate_integer_byclaim_ruin_probability(
    model: INARByClaimModel | BINARByClaimModel,
    periods: int,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
    seed: int | None | np.random.Generator = None,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> RuinEstimate:
    """Vectorized finite-horizon ruin estimate for INAR/BINAR models.

    Returns a `RuinEstimate` with discrete ruin times in period units.
    """

    period_count = _positive_int(periods, "periods")
    simulation_count = _positive_int(n_simulations, "n_simulations")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")
    method = ci_method.lower()
    if method not in {"wilson", "normal"}:
        raise ValueError("ci_method must be 'wilson' or 'normal'")
    threshold = _finite_float(ruin_threshold, "ruin_threshold")
    rng = _rng(seed)
    reserves = np.full(simulation_count, float(model.initial_capital), dtype=float)
    ruined = np.zeros(simulation_count, dtype=bool)
    ruin_times = np.full(simulation_count, np.inf)
    previous_byclaims: np.ndarray | None = None

    for period in range(period_count):
        if period == 0:
            primary, byclaims = _sample_initial_counts(model, simulation_count, rng)
        else:
            assert previous_byclaims is not None
            primary, byclaims = _sample_next_counts(model, previous_byclaims, rng)
        previous_byclaims = byclaims
        primary_losses = _losses_by_type(model.primary_distributions, primary, rng)
        byclaim_losses = _losses_by_type(model.byclaim_distributions, byclaims, rng)
        reserves += (
            model.premium_per_period
            - primary_losses.sum(axis=1)
            - byclaim_losses.sum(axis=1)
        )
        newly_ruined = ~ruined & _ruined(reserves, threshold, ruin_inclusive)
        ruin_times[newly_ruined] = period + 1
        ruined |= newly_ruined

    probability = float(np.mean(ruined))
    standard_error = math.sqrt(max(probability * (1.0 - probability), 0.0) / simulation_count)
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    if method == "wilson":
        denominator = 1.0 + z**2 / simulation_count
        center = (probability + z**2 / (2.0 * simulation_count)) / denominator
        half_width = (
            z
            * math.sqrt(
                probability * (1.0 - probability) / simulation_count
                + z**2 / (4.0 * simulation_count**2)
            )
            / denominator
        )
        ci_low = max(0.0, center - half_width)
        ci_high = min(1.0, center + half_width)
    else:
        ci_low = max(0.0, probability - z * standard_error)
        ci_high = min(1.0, probability + z * standard_error)
    return RuinEstimate(
        probability=probability,
        standard_error=float(standard_error),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=simulation_count,
        horizon=float(period_count),
        ruin_times=ruin_times,
        ci_method=method,
    )


def estimate_inar_byclaim_ruin_probability(
    model: INARByClaimModel,
    periods: int,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
    seed: int | None | np.random.Generator = None,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> RuinEstimate:
    """Estimate finite-horizon ruin probability for an INAR by-claim model."""

    if not isinstance(model, INARByClaimModel):
        raise TypeError("model must be an INARByClaimModel")
    return estimate_integer_byclaim_ruin_probability(
        model,
        periods,
        n_simulations=n_simulations,
        ci_level=ci_level,
        ci_method=ci_method,
        seed=seed,
        ruin_threshold=ruin_threshold,
        ruin_inclusive=ruin_inclusive,
    )


def estimate_binar_byclaim_ruin_probability(
    model: BINARByClaimModel,
    periods: int,
    *,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    ci_method: str = "wilson",
    seed: int | None | np.random.Generator = None,
    ruin_threshold: float = 0.0,
    ruin_inclusive: bool = True,
) -> RuinEstimate:
    """Estimate finite-horizon ruin probability for a BINAR by-claim model."""

    if not isinstance(model, BINARByClaimModel):
        raise TypeError("model must be a BINARByClaimModel")
    return estimate_integer_byclaim_ruin_probability(
        model,
        periods,
        n_simulations=n_simulations,
        ci_level=ci_level,
        ci_method=ci_method,
        seed=seed,
        ruin_threshold=ruin_threshold,
        ruin_inclusive=ruin_inclusive,
    )
