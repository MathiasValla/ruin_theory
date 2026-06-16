"""Risk-process model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import operator
from typing import Callable

import numpy as np

from .distributions import ClaimDistribution, exponential


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


def _sample_size(n: int, name: str) -> int:
    try:
        size = operator.index(n)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if size < 0:
        raise ValueError(f"{name} must be non-negative")
    return size


@dataclass(frozen=True)
class FrequencyModel:
    """Arrival process abstraction.

    ``poisson`` models the Cramer-Lundberg claim count. ``renewal`` models
    Sparre-Andersen arrivals with iid interarrival times.
    """

    kind: str
    rate: float | None = None
    interarrival_distribution: ClaimDistribution | None = None

    def __post_init__(self) -> None:
        if self.kind == "poisson":
            if self.rate is None:
                raise ValueError("poisson frequency requires rate")
            if self.interarrival_distribution is not None:
                raise ValueError("poisson frequency cannot also define interarrival_distribution")
            object.__setattr__(self, "rate", _positive_float(self.rate, "rate"))
            return
        if self.kind == "renewal":
            if self.interarrival_distribution is None:
                raise ValueError("renewal frequency requires interarrival_distribution")
            if self.rate is not None:
                raise ValueError("renewal frequency cannot also define rate")
            mean = self.interarrival_distribution.mean()
            if not np.isfinite(mean) or mean <= 0:
                raise ValueError("interarrival distribution must have finite positive mean")
            return
        raise ValueError("kind must be 'poisson' or 'renewal'")

    @classmethod
    def poisson(cls, rate: float) -> "FrequencyModel":
        return cls(kind="poisson", rate=rate)

    @classmethod
    def renewal(cls, interarrival_distribution: ClaimDistribution) -> "FrequencyModel":
        return cls(kind="renewal", interarrival_distribution=interarrival_distribution)

    def mean_rate(self) -> float:
        if self.kind == "poisson":
            assert self.rate is not None
            return self.rate
        if self.kind == "renewal":
            assert self.interarrival_distribution is not None
            return 1.0 / self.interarrival_distribution.mean()
        raise ValueError(f"unknown frequency model {self.kind!r}")

    def sample_interarrival(self, rng: np.random.Generator) -> float:
        if self.kind == "poisson":
            assert self.rate is not None
            return float(rng.exponential(1.0 / self.rate))
        if self.kind == "renewal":
            assert self.interarrival_distribution is not None
            value = float(self.interarrival_distribution.sample(1, rng=rng)[0])
            if value <= 0:
                raise ValueError("renewal interarrival samples must be positive")
            return value
        raise ValueError(f"unknown frequency model {self.kind!r}")


SeverityTransform = Callable[[np.ndarray], np.ndarray]
FrequencyWindow = tuple[float, float, float]


@dataclass(frozen=True)
class PreventionProgram:
    """Prevention acting on claim frequency and claim intensity.

    Multipliers below one reduce the corresponding risk driver. A custom
    ``severity_transform`` can encode deductibles, caps, inflation mitigation or
    engineering controls that are not simple multipliers. ``frequency_windows``
    can override the base frequency multiplier on finite intervals, encoded as
    ``(start, end, multiplier)`` with intervals interpreted as ``[start, end)``.
    """

    frequency_multiplier: float = 1.0
    severity_multiplier: float = 1.0
    severity_transform: SeverityTransform | None = None
    name: str = "baseline"
    frequency_windows: tuple[FrequencyWindow, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frequency_multiplier",
            _nonnegative_float(self.frequency_multiplier, "frequency_multiplier"),
        )
        object.__setattr__(
            self,
            "severity_multiplier",
            _nonnegative_float(self.severity_multiplier, "severity_multiplier"),
        )
        if self.severity_transform is not None and not callable(self.severity_transform):
            raise TypeError("severity_transform must be callable or None")
        windows = tuple(self.frequency_windows)
        cleaned_windows: list[FrequencyWindow] = []
        previous_end = 0.0
        for index, window in enumerate(windows):
            if len(window) != 3:
                raise ValueError("frequency_windows must contain (start, end, multiplier) tuples")
            start = _nonnegative_float(window[0], "frequency window start")
            end = _nonnegative_float(window[1], "frequency window end")
            multiplier = _nonnegative_float(window[2], "frequency window multiplier")
            if end <= start:
                raise ValueError("frequency window end must be greater than start")
            if index > 0 and start < previous_end:
                raise ValueError("frequency_windows must not overlap")
            cleaned_windows.append((start, end, multiplier))
            previous_end = end
        object.__setattr__(self, "frequency_windows", tuple(cleaned_windows))

    def frequency_multiplier_at(self, time: float) -> float:
        current_time = _nonnegative_float(time, "time")
        for start, end, multiplier in self.frequency_windows:
            if start <= current_time < end:
                return multiplier
        return self.frequency_multiplier

    def next_frequency_change_after(self, time: float) -> float:
        current_time = _nonnegative_float(time, "time")
        changes = [
            boundary
            for start, end, _ in self.frequency_windows
            for boundary in (start, end)
            if boundary > current_time
        ]
        if not changes:
            return np.inf
        return min(changes)

    def apply_frequency(self, rate: float, time: float | None = None) -> float:
        multiplier = self.frequency_multiplier if time is None else self.frequency_multiplier_at(time)
        return _nonnegative_float(rate, "rate") * multiplier

    def apply_severity(self, claims: np.ndarray) -> np.ndarray:
        input_values = np.asarray(claims, dtype=float)
        if np.any(~np.isfinite(input_values)) or np.any(input_values < 0):
            raise ValueError("claims must contain finite non-negative values")
        values = input_values * self.severity_multiplier
        if self.severity_transform is not None:
            values = np.asarray(self.severity_transform(values), dtype=float)
            if values.shape == () and input_values.shape == (1,):
                values = values.reshape(1)
            if values.shape != input_values.shape:
                raise ValueError("severity_transform must preserve the claim array shape")
        if np.any(~np.isfinite(values)):
            raise ValueError("prevention produced non-finite claim sizes")
        if np.any(values < 0):
            raise ValueError("prevention produced negative claim sizes")
        return values


@dataclass(frozen=True)
class ByClaimModel:
    """Secondary claims triggered by each primary claim."""

    probability: float
    distribution: ClaimDistribution
    count_mean: float = 1.0
    name: str = "by_claim"
    count_distribution: str = "poisson"

    def __post_init__(self) -> None:
        probability = _finite_float(self.probability, "probability")
        count_mean = _nonnegative_float(self.count_mean, "count_mean")
        if not isinstance(self.count_distribution, str):
            raise TypeError("count_distribution must be a string")
        count_distribution = self.count_distribution.lower()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if not isinstance(self.distribution, ClaimDistribution):
            raise TypeError("distribution must be a ClaimDistribution")
        if count_distribution not in {"poisson", "geometric"}:
            raise ValueError("count_distribution must be 'poisson' or 'geometric'")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "count_mean", count_mean)
        object.__setattr__(self, "count_distribution", count_distribution)

    def sample_total(self, n_primary: int, rng: np.random.Generator) -> np.ndarray:
        size = _sample_size(n_primary, "n_primary")
        triggered = rng.binomial(1, self.probability, size=size).astype(bool)
        totals = np.zeros(size, dtype=float)
        if not np.any(triggered) or self.count_mean == 0:
            return totals
        counts = self.sample_counts(int(triggered.sum()), rng)
        total_secondary = int(counts.sum())
        if total_secondary == 0:
            return totals
        sizes = self.distribution.sample(total_secondary, rng=rng)
        offset = 0
        for target, count in zip(np.flatnonzero(triggered), counts):
            if count:
                totals[target] = float(np.sum(sizes[offset : offset + count]))
                offset += count
        return totals

    def sample_counts(self, n_triggered: int, rng: np.random.Generator) -> np.ndarray:
        size = _sample_size(n_triggered, "n_triggered")
        if size == 0 or self.count_mean == 0:
            return np.zeros(size, dtype=int)
        if self.count_distribution == "poisson":
            counts = rng.poisson(self.count_mean, size=size)
        elif self.count_distribution == "geometric":
            probability = 1.0 / (1.0 + self.count_mean)
            counts = rng.geometric(probability, size=size) - 1
        else:
            raise ValueError(f"unknown count distribution {self.count_distribution!r}")
        counts = np.asarray(counts)
        if counts.shape != (size,):
            raise ValueError("count distribution must return one count per triggered claim")
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("secondary claim counts must be finite and non-negative")
        if np.any(counts != np.floor(counts)):
            raise ValueError("secondary claim counts must be integers")
        return counts.astype(int)

    def count_pgf(self, z: float) -> float:
        """Probability-generating function of the secondary claim count."""

        argument = _finite_float(z, "z")
        if self.count_mean == 0.0:
            return 1.0
        if self.count_distribution == "poisson":
            return float(math.exp(self.count_mean * (argument - 1.0)))
        if self.count_distribution == "geometric":
            probability = 1.0 / (1.0 + self.count_mean)
            denominator = 1.0 - (1.0 - probability) * argument
            return float(probability / denominator) if denominator > 0.0 else np.inf
        raise ValueError(f"unknown count distribution {self.count_distribution!r}")

    def expected_amount_per_primary(self) -> float:
        return self.probability * self.count_mean * self.distribution.mean()


@dataclass(frozen=True)
class CapitalInjectionModel:
    """Independent positive jumps, useful for solvency support scenarios."""

    rate: float
    distribution: ClaimDistribution
    name: str = "capital_injection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _nonnegative_float(self.rate, "rate"))
        if not isinstance(self.distribution, ClaimDistribution):
            raise TypeError("distribution must be a ClaimDistribution")


@dataclass(frozen=True)
class RiskProcess:
    initial_capital: float
    premium_rate: float
    claim_distribution: ClaimDistribution
    frequency: FrequencyModel
    prevention: PreventionProgram = field(default_factory=PreventionProgram)
    by_claims: tuple[ByClaimModel, ...] = ()
    capital_injections: tuple[CapitalInjectionModel, ...] = ()
    name: str = "risk_process"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_capital",
            _nonnegative_float(self.initial_capital, "initial_capital"),
        )
        object.__setattr__(self, "premium_rate", _nonnegative_float(self.premium_rate, "premium_rate"))
        if not isinstance(self.claim_distribution, ClaimDistribution):
            raise TypeError("claim_distribution must be a ClaimDistribution")
        if not isinstance(self.frequency, FrequencyModel):
            raise TypeError("frequency must be a FrequencyModel")
        if not isinstance(self.prevention, PreventionProgram):
            raise TypeError("prevention must be a PreventionProgram")
        by_claims = tuple(self.by_claims)
        if not all(isinstance(by_claim, ByClaimModel) for by_claim in by_claims):
            raise TypeError("by_claims must contain ByClaimModel instances")
        object.__setattr__(self, "by_claims", by_claims)
        capital_injections = tuple(self.capital_injections)
        if not all(isinstance(injection, CapitalInjectionModel) for injection in capital_injections):
            raise TypeError("capital_injections must contain CapitalInjectionModel instances")
        object.__setattr__(self, "capital_injections", capital_injections)

    @property
    def claim_arrival_rate(self) -> float:
        return self.prevention.apply_frequency(self.frequency.mean_rate())

    @property
    def expected_claim_amount(self) -> float:
        if self.prevention.severity_transform is not None:
            raise NotImplementedError(
                "expected_claim_amount is unavailable with a custom severity_transform"
            )
        base = self.claim_distribution.mean() * self.prevention.severity_multiplier
        by_claims = sum(by_claim.expected_amount_per_primary() for by_claim in self.by_claims)
        return base + by_claims

    @property
    def claim_intensity(self) -> float:
        return self.claim_arrival_rate * self.expected_claim_amount

    @property
    def safety_loading(self) -> float:
        if self.claim_intensity == 0:
            return np.inf
        return self.premium_rate / self.claim_intensity - 1.0


@dataclass(frozen=True)
class CramerLundbergProcess(RiskProcess):
    """Classical reserve process ``R_t = u + c t - sum_{i<=N_t} X_i``."""

    frequency: FrequencyModel = field(default_factory=lambda: FrequencyModel.poisson(1.0))
    name: str = "cramer_lundberg"

    def __init__(
        self,
        initial_capital: float = 0.0,
        premium_rate: float = 1.0,
        claim_arrival_rate: float = 1.0,
        claim_distribution: ClaimDistribution | None = None,
        prevention: PreventionProgram | None = None,
        by_claims: tuple[ByClaimModel, ...] = (),
        capital_injections: tuple[CapitalInjectionModel, ...] = (),
        name: str = "cramer_lundberg",
    ) -> None:
        object.__setattr__(self, "initial_capital", float(initial_capital))
        object.__setattr__(self, "premium_rate", float(premium_rate))
        object.__setattr__(
            self,
            "claim_distribution",
            exponential(1.0) if claim_distribution is None else claim_distribution,
        )
        object.__setattr__(self, "frequency", FrequencyModel.poisson(claim_arrival_rate))
        object.__setattr__(
            self,
            "prevention",
            PreventionProgram() if prevention is None else prevention,
        )
        object.__setattr__(self, "by_claims", tuple(by_claims))
        object.__setattr__(self, "capital_injections", tuple(capital_injections))
        object.__setattr__(self, "name", name)
        RiskProcess.__post_init__(self)


@dataclass(frozen=True)
class SparreAndersenProcess(RiskProcess):
    """Renewal-arrival reserve process."""

    frequency: FrequencyModel = field(default_factory=lambda: FrequencyModel.renewal(exponential(1.0)))
    name: str = "sparre_andersen"

    def __init__(
        self,
        initial_capital: float = 0.0,
        premium_rate: float = 1.0,
        interarrival_distribution: ClaimDistribution | None = None,
        claim_distribution: ClaimDistribution | None = None,
        prevention: PreventionProgram | None = None,
        by_claims: tuple[ByClaimModel, ...] = (),
        capital_injections: tuple[CapitalInjectionModel, ...] = (),
        name: str = "sparre_andersen",
    ) -> None:
        object.__setattr__(self, "initial_capital", float(initial_capital))
        object.__setattr__(self, "premium_rate", float(premium_rate))
        object.__setattr__(
            self,
            "claim_distribution",
            exponential(1.0) if claim_distribution is None else claim_distribution,
        )
        object.__setattr__(
            self,
            "frequency",
            FrequencyModel.renewal(
                exponential(1.0)
                if interarrival_distribution is None
                else interarrival_distribution
            ),
        )
        object.__setattr__(
            self,
            "prevention",
            PreventionProgram() if prevention is None else prevention,
        )
        object.__setattr__(self, "by_claims", tuple(by_claims))
        object.__setattr__(self, "capital_injections", tuple(capital_injections))
        object.__setattr__(self, "name", name)
        RiskProcess.__post_init__(self)
