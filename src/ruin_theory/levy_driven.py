"""Level-dependent Levy-driven and jump-diffusion ruin diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

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


def _call_or_value(value: float | FloatFunction, x: float) -> float:
    return float(value(x)) if callable(value) else float(value)


@dataclass(frozen=True)
class CompoundPoissonSubordinator:
    """Compound-Poisson Levy liability process."""

    rate: float
    jump_distribution: ClaimDistribution
    name: str = "compound_poisson"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _nonnegative_float(self.rate, "rate"))
        if not isinstance(self.jump_distribution, ClaimDistribution):
            raise TypeError("jump_distribution must be a ClaimDistribution")

    @property
    def mean_rate(self) -> float:
        return self.rate * self.jump_distribution.mean()

    @property
    def exponential_moment_upper(self) -> float:
        metadata = self.jump_distribution.metadata
        if self.jump_distribution.name == "exponential":
            return float(metadata["rate"])
        if self.jump_distribution.name == "gamma":
            return 1.0 / float(metadata["scale"])
        if self.jump_distribution.name == "erlang":
            return float(metadata["rate"])
        if self.jump_distribution.name == "mixture_exponential":
            return float(np.min(np.asarray(metadata["rates"], dtype=float)))
        if self.jump_distribution.name in {"phase_type", "matrix_exponential"}:
            return float(metadata["spectral_bound"])
        return np.inf

    def levy_exponent(self, lam: float) -> float:
        argument = _nonnegative_float(lam, "lam")
        if self.rate == 0.0:
            return 0.0
        return self.rate * (self.jump_distribution.mgf(argument) - 1.0)


@dataclass(frozen=True)
class GammaSubordinator:
    """Gamma-process liability with exponent ``alpha log(beta/(beta-lam))``."""

    alpha: float
    beta: float
    name: str = "gamma_process"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _positive_float(self.alpha, "alpha"))
        object.__setattr__(self, "beta", _positive_float(self.beta, "beta"))

    @property
    def mean_rate(self) -> float:
        return self.alpha / self.beta

    @property
    def exponential_moment_upper(self) -> float:
        return self.beta

    def levy_exponent(self, lam: float) -> float:
        argument = _nonnegative_float(lam, "lam")
        if argument >= self.beta:
            return np.inf
        return self.alpha * math.log(self.beta / (self.beta - argument))


@dataclass(frozen=True)
class InverseGaussianSubordinator:
    """Inverse-Gaussian Levy liability with ``E[L_1] = 1 / gamma``."""

    gamma: float
    name: str = "inverse_gaussian_process"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _positive_float(self.gamma, "gamma"))

    @property
    def mean_rate(self) -> float:
        return 1.0 / self.gamma

    @property
    def exponential_moment_upper(self) -> float:
        return 0.5 * self.gamma**2

    def levy_exponent(self, lam: float) -> float:
        argument = _nonnegative_float(lam, "lam")
        upper = self.exponential_moment_upper
        if argument >= upper:
            return np.inf
        return self.gamma - math.sqrt(self.gamma**2 - 2.0 * argument)


LevySubordinatorLike = (
    CompoundPoissonSubordinator | GammaSubordinator | InverseGaussianSubordinator
)


@dataclass(frozen=True)
class LevelDependentLevyRiskProcess:
    """Risk process ``dX_t = p(X_t)dt + sigma(X_t)dW_t - dL_t``."""

    premium_rate: float | FloatFunction
    liability: LevySubordinatorLike
    diffusion: float | FloatFunction = 0.0
    diffusion_derivative: float | FloatFunction | None = 0.0
    initial_capital: float = 0.0
    name: str = "level_dependent_levy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_capital",
            _nonnegative_float(self.initial_capital, "initial_capital"),
        )
        if not hasattr(self.liability, "levy_exponent") or not hasattr(self.liability, "mean_rate"):
            raise TypeError("liability must provide levy_exponent and mean_rate")
        if not callable(self.premium_rate):
            object.__setattr__(self, "premium_rate", _positive_float(self.premium_rate, "premium_rate"))
        if not callable(self.diffusion):
            object.__setattr__(self, "diffusion", _nonnegative_float(self.diffusion, "diffusion"))
        if self.diffusion_derivative is not None and not callable(self.diffusion_derivative):
            object.__setattr__(self, "diffusion_derivative", float(self.diffusion_derivative))

    def premium_at(self, level: float) -> float:
        return _positive_float(_call_or_value(self.premium_rate, level), "premium_rate(level)")

    def diffusion_at(self, level: float) -> float:
        return _nonnegative_float(_call_or_value(self.diffusion, level), "diffusion(level)")

    def diffusion_derivative_at(self, level: float) -> float:
        if self.diffusion_derivative is None:
            step = max(1e-5, abs(level) * 1e-5)
            left = max(0.0, level - step)
            right = level + step
            if right == left:
                right = step
            return (self.diffusion_at(right) - self.diffusion_at(left)) / (right - left)
        return float(_call_or_value(self.diffusion_derivative, level))

    @property
    def net_profit_margin_at_zero(self) -> float:
        return self.premium_at(0.0) - self.liability.mean_rate


@dataclass(frozen=True)
class ExponentialConvergenceResult:
    """Exponential convergence-rate diagnostic for finite-to-ultimate ruin."""

    lambda_star: float
    rate: float
    process: LevelDependentLevyRiskProcess
    level_grid: np.ndarray
    lambda_upper: float

    def factor(self, horizon: ArrayLike) -> np.ndarray:
        values = np.asarray(horizon, dtype=float)
        if np.any(values < 0.0) or np.any(~np.isfinite(values)):
            raise ValueError("horizon must contain finite non-negative values")
        return np.exp(-self.rate * values)

    def gap_bound(
        self,
        horizon: ArrayLike,
        stationary_exponential_moment: float,
    ) -> np.ndarray:
        moment = _positive_float(stationary_exponential_moment, "stationary_exponential_moment")
        return (1.0 + moment) * self.factor(horizon)


def _convergence_rate_objective(
    process: LevelDependentLevyRiskProcess,
    lam: float,
    levels: np.ndarray,
) -> float:
    kappa = process.liability.levy_exponent(lam)
    if not np.isfinite(kappa):
        return -np.inf
    values = np.empty(levels.size, dtype=float)
    for index, level in enumerate(levels):
        sigma = process.diffusion_at(float(level))
        sigma_prime = process.diffusion_derivative_at(float(level))
        premium = process.premium_at(float(level))
        values[index] = (
            premium * lam
            + sigma * sigma_prime * lam
            - 0.5 * sigma**2 * lam**2
            - kappa
        )
    return float(np.min(values))


def exponential_convergence_rate(
    process: LevelDependentLevyRiskProcess,
    *,
    lambda_upper: float | None = None,
    level_grid: ArrayLike | None = None,
    grid_size: int = 128,
    tol: float = 1e-10,
) -> ExponentialConvergenceResult:
    """Compute ``k = max_lambda Phi(lambda)`` for a Levy-driven risk process."""

    if not isinstance(process, LevelDependentLevyRiskProcess):
        raise TypeError("process must be a LevelDependentLevyRiskProcess")
    tol = _positive_float(tol, "tol")
    if level_grid is None:
        if callable(process.premium_rate) or callable(process.diffusion):
            levels = np.linspace(0.0, max(1.0, process.initial_capital + 10.0), grid_size)
        else:
            levels = np.array([0.0])
    else:
        levels = np.asarray(level_grid, dtype=float).ravel()
        if levels.size == 0:
            raise ValueError("level_grid must contain at least one value")
        if np.any(~np.isfinite(levels)) or np.any(levels < 0.0):
            raise ValueError("level_grid must contain finite non-negative values")

    upper = (
        float(process.liability.exponential_moment_upper)
        if lambda_upper is None
        else _positive_float(lambda_upper, "lambda_upper")
    )
    if np.isfinite(upper):
        upper *= 1.0 - 1e-10
    else:
        upper = 1.0
        while _convergence_rate_objective(process, upper, levels) > 0.0 and upper < 1e6:
            upper *= 2.0
    if upper <= tol:
        raise ValueError("lambda search interval is empty")

    objective = lambda lam: -_convergence_rate_objective(process, lam, levels)
    result = optimize.minimize_scalar(
        objective,
        bounds=(tol, upper),
        method="bounded",
        options={"xatol": tol},
    )
    if not result.success:
        raise ValueError("could not optimize the exponential convergence rate")
    rate = -float(result.fun)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("no positive exponential convergence rate was found")
    return ExponentialConvergenceResult(
        lambda_star=float(result.x),
        rate=rate,
        process=process,
        level_grid=levels.copy(),
        lambda_upper=upper,
    )


def finite_to_ultimate_ruin_gap_bound(
    result: ExponentialConvergenceResult,
    horizon: ArrayLike,
    stationary_exponential_moment: float,
) -> np.ndarray:
    """Bound ``psi(u)-psi(u,T)`` by the exponential convergence factor."""

    if not isinstance(result, ExponentialConvergenceResult):
        raise TypeError("result must be an ExponentialConvergenceResult")
    return result.gap_bound(horizon, stationary_exponential_moment)
