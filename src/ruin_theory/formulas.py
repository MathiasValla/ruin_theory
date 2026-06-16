"""Closed forms and approximations for ruin probabilities."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, optimize, special

from .distributions import ClaimDistribution
from .models import CramerLundbergProcess, RiskProcess


def _as_array(u: ArrayLike) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    if np.any(values < 0):
        raise ValueError("initial surplus values must be non-negative")
    return values


def safety_loading(model: RiskProcess) -> float:
    """Return ``c / (lambda * E[X]) - 1`` for the model."""

    return model.safety_loading


def _closed_form_process_check(model: RiskProcess) -> None:
    if not isinstance(model, CramerLundbergProcess):
        raise ValueError("closed-form CL formulas require a CramerLundbergProcess")
    if model.prevention.frequency_windows:
        raise ValueError("closed-form CL formulas require stationary frequency prevention")
    if model.capital_injections:
        raise ValueError("closed-form CL formulas do not support capital injections")
    if model.prevention.severity_transform is not None:
        raise ValueError("closed-form CL formulas require linear severity scaling")


def _primary_claim_formula_check(model: RiskProcess) -> None:
    _closed_form_process_check(model)
    if model.by_claims:
        raise ValueError("this closed form does not include by-claims")


def _severity_scale(model: RiskProcess) -> float:
    return float(model.prevention.severity_multiplier)


def _net_profit_check(model: RiskProcess) -> None:
    _closed_form_process_check(model)
    if model.premium_rate <= model.claim_intensity:
        raise ValueError("net profit condition fails: premium_rate must exceed claim intensity")


def _aggregate_claim_mgf(model: CramerLundbergProcess, r: float) -> float:
    _closed_form_process_check(model)
    scale = _severity_scale(model)
    total = model.claim_distribution.mgf(scale * r)
    for by_claim in model.by_claims:
        by_mgf = by_claim.distribution.mgf(r)
        total *= (1.0 - by_claim.probability) + by_claim.probability * by_claim.count_pgf(by_mgf)
    return float(total)


def _raw_moment(distribution: ClaimDistribution, order: int) -> float:
    if order < 0:
        raise ValueError("order must be non-negative")
    name = distribution.name
    if name == "exponential":
        rate = float(distribution.metadata["rate"])
        return math.factorial(order) / rate**order
    if name == "gamma":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        return scale**order * math.gamma(shape + order) / math.gamma(shape)
    if name == "erlang":
        shape = int(distribution.metadata["shape"])
        rate = float(distribution.metadata["rate"])
        scale = 1.0 / rate
        return scale**order * math.gamma(shape + order) / math.gamma(shape)
    if name == "mixture_exponential":
        rates = np.asarray(distribution.metadata["rates"], dtype=float)
        weights = np.asarray(distribution.metadata["weights"], dtype=float)
        return float(np.sum(weights * math.factorial(order) / rates**order))
    if name == "deterministic":
        return float(distribution.metadata["value"]) ** order
    if name == "pareto":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        if shape <= order:
            return float("inf")
        return shape * scale**order / (shape - order)
    if name == "lognormal":
        meanlog = float(distribution.metadata["meanlog"])
        sdlog = float(distribution.metadata["sdlog"])
        return float(math.exp(order * meanlog + 0.5 * order**2 * sdlog**2))
    if name == "weibull":
        shape = float(distribution.metadata["shape"])
        scale = float(distribution.metadata["scale"])
        return scale**order * math.gamma(1.0 + order / shape)
    if name == "empirical":
        values = np.asarray(distribution.metadata["values"], dtype=float)
        return float(np.mean(values**order))
    raise NotImplementedError(f"raw moments are not implemented for {name}")


def _sample_integrated_tail(
    distribution: ClaimDistribution,
    size: int,
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample the equilibrium distribution of a scaled severity."""

    if scale == 0:
        return np.zeros(size)
    name = distribution.name
    if name == "exponential":
        rate = float(distribution.metadata["rate"]) / scale
        return rng.exponential(1.0 / rate, size=size)
    if name == "mixture_exponential":
        rates = np.asarray(distribution.metadata["rates"], dtype=float) / scale
        weights = np.asarray(distribution.metadata["weights"], dtype=float)
        mean = float(np.sum(weights / rates))
        tail_weights = weights / rates / mean
        idx = rng.choice(rates.size, size=size, p=tail_weights)
        return rng.exponential(1.0 / rates[idx])
    if name == "deterministic":
        value = scale * float(distribution.metadata["value"])
        return rng.uniform(0.0, value, size=size)
    if name == "gamma":
        shape = float(distribution.metadata["shape"])
        scale_value = scale * float(distribution.metadata["scale"])
        return rng.random(size) * rng.gamma(shape=shape + 1.0, scale=scale_value, size=size)
    if name == "erlang":
        shape = int(distribution.metadata["shape"])
        rate = float(distribution.metadata["rate"]) / scale
        return rng.random(size) * rng.gamma(shape=shape + 1.0, scale=1.0 / rate, size=size)
    if name == "lognormal":
        meanlog = math.log(scale) + float(distribution.metadata["meanlog"])
        sdlog = float(distribution.metadata["sdlog"])
        return rng.random(size) * rng.lognormal(mean=meanlog + sdlog**2, sigma=sdlog, size=size)
    if name == "pareto":
        shape = float(distribution.metadata["shape"])
        if shape <= 1:
            raise ValueError("Pareto integrated-tail sampling requires finite mean")
        scaled_xm = scale * float(distribution.metadata["scale"])
        return rng.random(size) * scaled_xm * (1.0 + rng.pareto(shape - 1.0, size=size))
    if name == "weibull":
        shape = float(distribution.metadata["shape"])
        scale_value = scale * float(distribution.metadata["scale"])
        size_biased_power = rng.gamma(shape=1.0 + 1.0 / shape, scale=1.0, size=size)
        return rng.random(size) * scale_value * size_biased_power ** (1.0 / shape)
    if name == "empirical":
        values = scale * np.asarray(distribution.metadata["values"], dtype=float)
        if float(np.mean(values)) <= 0:
            return np.zeros(size)
        idx = rng.choice(values.size, size=size, p=values / values.sum())
        return rng.random(size) * values[idx]
    raise NotImplementedError(f"integrated-tail sampling is not implemented for {name}")


def _normal_survival(x: np.ndarray) -> np.ndarray:
    return 0.5 * special.erfc(x / math.sqrt(2.0))


def _integrated_tail_survival(
    distribution: ClaimDistribution,
    u: ArrayLike,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")

    surplus = _as_array(u)
    if np.any(np.isnan(surplus)):
        raise ValueError("initial surplus values must not contain NaN")
    values = np.zeros_like(surplus.ravel(), dtype=float)
    finite = np.isfinite(surplus.ravel())
    x = surplus.ravel()[finite]
    if scale == 0.0 or x.size == 0:
        return values.reshape(surplus.shape)

    name = distribution.name
    tail = np.zeros_like(x, dtype=float)
    if name == "exponential":
        rate = float(distribution.metadata["rate"]) / scale
        tail = np.exp(-rate * x)
    elif name == "mixture_exponential":
        rates = np.asarray(distribution.metadata["rates"], dtype=float) / scale
        weights = np.asarray(distribution.metadata["weights"], dtype=float)
        mean = float(np.sum(weights / rates))
        tail_weights = weights / rates / mean
        tail = np.sum(tail_weights[:, None] * np.exp(-rates[:, None] * x), axis=0)
    elif name == "deterministic":
        value = scale * float(distribution.metadata["value"])
        if value > 0.0:
            tail = np.maximum(1.0 - x / value, 0.0)
    elif name == "gamma":
        shape = float(distribution.metadata["shape"])
        scale_value = scale * float(distribution.metadata["scale"])
        z = x / scale_value
        tail = special.gammaincc(shape + 1.0, z) - z / shape * special.gammaincc(shape, z)
    elif name == "erlang":
        shape = float(distribution.metadata["shape"])
        scale_value = scale / float(distribution.metadata["rate"])
        z = x / scale_value
        tail = special.gammaincc(shape + 1.0, z) - z / shape * special.gammaincc(shape, z)
    elif name == "pareto":
        shape = float(distribution.metadata["shape"])
        if shape <= 1.0:
            raise ValueError("Pareto integrated-tail survival requires finite mean")
        threshold = scale * float(distribution.metadata["scale"])
        mean = shape * threshold / (shape - 1.0)
        below = x < threshold
        tail[below] = 1.0 - x[below] / mean
        tail[~below] = (threshold / x[~below]) ** (shape - 1.0) / shape
    elif name == "lognormal":
        meanlog = math.log(scale) + float(distribution.metadata["meanlog"])
        sdlog = float(distribution.metadata["sdlog"])
        log_mean = meanlog + 0.5 * sdlog**2
        positive = x > 0.0
        tail[~positive] = 1.0
        if np.any(positive):
            xpos = x[positive]
            d0 = (np.log(xpos) - meanlog) / sdlog
            d1 = d0 - sdlog
            sf0 = _normal_survival(d0)
            sf1 = _normal_survival(d1)
            weighted_sf0 = np.zeros_like(xpos)
            positive_sf = sf0 > 0.0
            weighted_sf0[positive_sf] = np.exp(
                np.log(xpos[positive_sf]) - log_mean + np.log(sf0[positive_sf])
            )
            tail[positive] = sf1 - weighted_sf0
    elif name == "weibull":
        shape = float(distribution.metadata["shape"])
        scale_value = scale * float(distribution.metadata["scale"])
        tail = special.gammaincc(1.0 / shape, (x / scale_value) ** shape)
    elif name == "empirical":
        samples = scale * np.asarray(distribution.metadata["values"], dtype=float)
        mean = float(np.mean(samples))
        if mean > 0.0:
            tail = np.mean(np.maximum(samples[:, None] - x, 0.0), axis=0) / mean
    else:
        raise NotImplementedError(f"integrated-tail survival is not implemented for {name}")

    values[finite] = np.clip(tail, 0.0, 1.0)
    return values.reshape(surplus.shape)


def integrated_tail_survival(
    distribution: ClaimDistribution,
    u: ArrayLike,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Survival of the equilibrium, or integrated-tail, severity law.

    For a scaled claim ``Y = scale * X`` with finite positive mean, this returns
    ``bar F_I(u) = E[(Y - u)_+] / E[Y]``. These tails are the distributional
    input in the Pollaczek-Khinchine formula and in subexponential ruin
    asymptotics.
    """

    return _integrated_tail_survival(distribution, u, scale=scale)


def _effective_exponential_rate(model: CramerLundbergProcess) -> float:
    if model.claim_distribution.name != "exponential":
        raise ValueError("requires exponential claim sizes")
    _primary_claim_formula_check(model)
    scale = _severity_scale(model)
    if scale == 0.0:
        return math.inf
    return float(model.claim_distribution.metadata["rate"]) / scale


def _effective_mixture_exponential_parameters(
    model: CramerLundbergProcess,
) -> tuple[np.ndarray, np.ndarray]:
    if model.claim_distribution.name != "mixture_exponential":
        raise ValueError("requires mixture_exponential claim sizes")
    _primary_claim_formula_check(model)
    scale = _severity_scale(model)
    if scale == 0.0:
        return np.empty(0), np.empty(0)

    rates = np.asarray(model.claim_distribution.metadata["rates"], dtype=float) / scale
    weights = np.asarray(model.claim_distribution.metadata["weights"], dtype=float)
    positive = weights > 0.0
    rates = rates[positive]
    weights = weights[positive]
    order = np.argsort(rates)
    rates = rates[order]
    weights = weights[order]

    unique_rates: list[float] = []
    unique_weights: list[float] = []
    for rate, weight in zip(rates, weights, strict=True):
        if unique_rates and np.isclose(rate, unique_rates[-1], rtol=1e-12, atol=1e-14):
            unique_weights[-1] += float(weight)
        else:
            unique_rates.append(float(rate))
            unique_weights.append(float(weight))
    weight_array = np.asarray(unique_weights, dtype=float)
    return np.asarray(unique_rates, dtype=float), weight_array / weight_array.sum()


def ultimate_ruin_exponential(
    model: CramerLundbergProcess,
    u: ArrayLike | None = None,
) -> np.ndarray:
    """Ultimate ruin probability for exponential claims in the CL model.

    For claim rate ``nu``, arrival rate ``lambda`` and premium rate ``c``,
    ``psi(u) = rho * exp(-(nu - lambda / c) * u)`` with
    ``rho = lambda / (c * nu)``.
    """

    rate = _effective_exponential_rate(model)
    surplus = _as_array(model.initial_capital if u is None else u)
    if not np.isfinite(rate):
        return np.zeros_like(surplus, dtype=float)
    lam = model.claim_arrival_rate
    if lam == 0.0:
        return np.zeros_like(surplus, dtype=float)
    c = model.premium_rate
    if c <= 0.0:
        return np.ones_like(surplus, dtype=float)
    rho = lam / (c * rate)
    if rho >= 1.0:
        return np.ones_like(surplus, dtype=float)
    gamma = rate - lam / c
    return rho * np.exp(-gamma * surplus)


def adjustment_coefficient(
    model: CramerLundbergProcess,
    upper: float | None = None,
    *,
    tol: float = 1e-12,
) -> float:
    """Solve the Lundberg equation ``lambda (M_X(r)-1) - c r = 0``."""

    _net_profit_check(model)
    if model.expected_claim_amount <= 0.0:
        raise ValueError("a positive aggregate claim amount is required")
    lam = model.claim_arrival_rate
    c = model.premium_rate

    def kappa(r: float) -> float:
        return lam * (_aggregate_claim_mgf(model, r) - 1.0) - c * r

    lower = tol
    lower_value = kappa(lower)
    for _ in range(20):
        if np.isfinite(lower_value) and lower_value < 0.0:
            break
        lower *= 0.1
        lower_value = kappa(lower)
    else:
        raise ValueError("could not bracket the adjustment coefficient near zero")

    def finite_positive_upper(low: float, high: float) -> float:
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            midpoint_value = kappa(midpoint)
            if np.isfinite(midpoint_value):
                if midpoint_value > 0.0:
                    return midpoint
                low = midpoint
            else:
                high = midpoint
        raise ValueError("could not bracket the adjustment coefficient")

    if upper is None:
        upper = 1.0
        bracket_low = lower
        for _ in range(80):
            value = kappa(upper)
            if np.isfinite(value) and value > 0:
                break
            if np.isposinf(value):
                upper = finite_positive_upper(bracket_low, upper)
                break
            if np.isfinite(value):
                bracket_low = upper
            upper *= 2.0
        else:
            raise ValueError("could not bracket the adjustment coefficient")
    else:
        value = kappa(upper)
        if np.isposinf(value):
            upper = finite_positive_upper(lower, upper)
        elif not np.isfinite(value) or value <= 0.0:
            raise ValueError("upper does not bracket the adjustment coefficient")
    return float(optimize.brentq(kappa, lower, upper, xtol=tol, rtol=tol))


def lundberg_bound(
    model: CramerLundbergProcess,
    u: ArrayLike,
    gamma: float | None = None,
) -> np.ndarray:
    """Lundberg inequality ``psi(u) <= exp(-gamma u)``."""

    coefficient = adjustment_coefficient(model) if gamma is None else float(gamma)
    if coefficient <= 0.0:
        raise ValueError("gamma must be positive")
    return np.exp(-coefficient * _as_array(u))


def _aggregate_claim_mgf_derivative(model: CramerLundbergProcess, r: float) -> float:
    h = max(1e-6, abs(r) * 1e-5)
    for _ in range(30):
        left = _aggregate_claim_mgf(model, r - h)
        right = _aggregate_claim_mgf(model, r + h)
        if np.isfinite(left) and np.isfinite(right):
            return (right - left) / (2.0 * h)
        h *= 0.5
    raise ValueError("could not evaluate MGF derivative at the adjustment coefficient")


def cramer_lundberg_asymptotic(
    model: CramerLundbergProcess,
    u: ArrayLike,
    gamma: float | None = None,
) -> np.ndarray:
    """Cramer-Lundberg light-tail asymptotic ``C exp(-gamma u)``."""

    _net_profit_check(model)
    coefficient = adjustment_coefficient(model) if gamma is None else float(gamma)
    if coefficient <= 0.0:
        raise ValueError("gamma must be positive")
    lam = model.claim_arrival_rate
    c = model.premium_rate
    rho = model.claim_intensity / c
    derivative = _aggregate_claim_mgf_derivative(model, coefficient)
    denominator = lam * derivative / c - 1.0
    if denominator <= 0.0:
        raise ValueError("invalid Cramer-Lundberg asymptotic constant")

    constant = (1.0 - rho) / denominator
    return constant * np.exp(-coefficient * _as_array(u))


def ultimate_ruin_hyperexponential(model: CramerLundbergProcess, u: ArrayLike) -> np.ndarray:
    """Exact ultimate ruin probability for hyperexponential severities.

    The implementation uses the Pollaczek-Khinchine Laplace transform and
    numerical residues. It covers mixtures of exponentials, including the
    Gerber/actuar example ``0.5 Exp(3) + 0.5 Exp(7)``.
    """

    surplus = _as_array(u)
    rates, weights = _effective_mixture_exponential_parameters(model)
    if rates.size == 0:
        return np.zeros_like(surplus, dtype=float)
    lam = model.claim_arrival_rate
    c = model.premium_rate
    if c <= 0.0:
        return np.ones_like(surplus, dtype=float)
    mean = float(np.sum(weights / rates))
    rho = lam * mean / c
    if rho >= 1.0:
        return np.ones_like(surplus, dtype=float)
    tail_weights = weights / rates / mean

    def integrated_tail_laplace(s: complex) -> complex:
        return np.sum(tail_weights * rates / (rates + s))

    def denominator(s: complex) -> complex:
        return 1.0 - rho * integrated_tail_laplace(s)

    def denominator_at_exponent(alpha: float) -> float:
        return float(np.real(denominator(-alpha)))

    def root_between_poles(left: float, right: float) -> float:
        width = right - left
        eps = max(1e-12, width * 1e-10)
        a = 0.0 if left == 0.0 else left + eps
        b = right - eps
        fa = denominator_at_exponent(a)
        fb = denominator_at_exponent(b)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            return float(optimize.brentq(denominator_at_exponent, a, b))

        grid = np.linspace(a, b, 1000)
        values = np.array([denominator_at_exponent(point) for point in grid])
        finite = np.isfinite(values)
        changes = finite[:-1] & finite[1:] & (
            np.signbit(values[:-1]) != np.signbit(values[1:])
        )
        for idx in np.where(changes)[0]:
            return float(optimize.brentq(denominator_at_exponent, grid[idx], grid[idx + 1]))
        raise ValueError("could not identify a hyperexponential root between poles")

    poles = np.r_[0.0, rates]
    roots = [root_between_poles(poles[i], poles[i + 1]) for i in range(rates.size)]
    if len(roots) != rates.size:
        raise ValueError("could not identify hyperexponential roots")

    def denominator_derivative(s: complex) -> complex:
        return rho * np.sum(tail_weights * rates / (rates + s) ** 2)

    coefficients = np.array(
        [
            float(
                np.real(
                    rho
                    * (1.0 - integrated_tail_laplace(-root))
                    / (-root * denominator_derivative(-root))
                )
            )
            for root in roots
        ]
    )
    values = np.sum(coefficients[:, None] * np.exp(-np.outer(roots, surplus.ravel())), axis=0)
    return np.clip(values.reshape(surplus.shape), 0.0, 1.0)


def expected_time_to_ruin_exponential(
    model: CramerLundbergProcess,
    u: ArrayLike | None = None,
) -> np.ndarray:
    """Conditional mean time to ruin for exponential CL claims."""

    _net_profit_check(model)
    rate = _effective_exponential_rate(model)
    if not np.isfinite(rate):
        return np.full_like(_as_array(model.initial_capital if u is None else u), np.inf)
    beta = model.claim_arrival_rate / model.premium_rate
    surplus = _as_array(model.initial_capital if u is None else u)
    return (beta * surplus + 1.0) / (rate - beta)


def finite_time_ruin_exponential(
    model: CramerLundbergProcess,
    u: float,
    horizon: float,
    *,
    epsabs: float = 1e-9,
) -> float:
    """Finite-time ruin probability for exponential claims via numerical quadrature.

    This implements the Asmussen-Albrecher formula for unit claim rate after
    scaling: ``psi_{beta,nu}(u,T) = psi_{beta/nu,1}(nu*u, nu*c*T)``.
    """

    if u < 0 or horizon < 0:
        raise ValueError("u and horizon must be non-negative")
    rate = _effective_exponential_rate(model)
    if not np.isfinite(rate) or horizon == 0.0:
        return 0.0
    if model.premium_rate <= 0.0:
        raise ValueError("premium_rate must be positive")
    beta = model.claim_arrival_rate / model.premium_rate
    beta_scaled = beta / rate
    u_scaled = rate * u
    t_scaled = rate * model.premium_rate * horizon
    if beta_scaled >= 1:
        raise ValueError("finite-time formula currently assumes positive safety loading")

    sqrt_beta = math.sqrt(beta_scaled)

    def integrand(theta: float) -> float:
        f1 = beta_scaled * math.exp(
            2.0 * sqrt_beta * t_scaled * math.cos(theta)
            - (1.0 + beta_scaled) * t_scaled
            + u_scaled * (sqrt_beta * math.cos(theta) - 1.0)
        )
        f2 = math.cos(u_scaled * sqrt_beta * math.sin(theta)) - math.cos(
            u_scaled * sqrt_beta * math.sin(theta) + 2.0 * theta
        )
        f3 = 1.0 + beta_scaled - 2.0 * sqrt_beta * math.cos(theta)
        return f1 * f2 / f3

    integral, _ = integrate.quad(integrand, 0.0, math.pi, epsabs=epsabs)
    ultimate = beta_scaled * math.exp(-(1.0 - beta_scaled) * u_scaled)
    return float(np.clip(ultimate - integral / math.pi, 0.0, 1.0))


def pollaczek_khinchine_monte_carlo(
    model: CramerLundbergProcess,
    u: ArrayLike,
    *,
    n_simulations: int = 50_000,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo estimator based on the Pollaczek-Khinchine geometric sum."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    _primary_claim_formula_check(model)
    _net_profit_check(model)
    scale = _severity_scale(model)
    surplus = _as_array(u)
    if scale == 0.0:
        return np.zeros_like(surplus, dtype=float)
    mean = scale * model.claim_distribution.mean()
    if not np.isfinite(mean):
        raise ValueError("finite claim mean is required")
    rho = model.claim_intensity / model.premium_rate
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    rng = np.random.default_rng(seed)

    counts = rng.geometric(1.0 - rho, size=n_simulations) - 1
    sums = np.zeros(n_simulations, dtype=float)
    total = int(counts.sum())
    if total > 0:
        samples = _sample_integrated_tail(model.claim_distribution, total, rng, scale=scale)
        starts = np.r_[0, np.cumsum(counts[:-1])]
        nonzero = counts > 0
        sums[nonzero] = np.add.reduceat(samples, starts[nonzero])
    estimates = np.array([np.mean(sums > level) for level in surplus.ravel()])
    return estimates.reshape(surplus.shape)


def de_vylder_approximation(model: CramerLundbergProcess, u: ArrayLike) -> np.ndarray:
    """Three-moment De Vylder exponential approximation."""

    mean = model.claim_distribution.mean()
    second = _raw_moment(model.claim_distribution, 2)
    if not np.isfinite(second):
        raise ValueError("finite second moment is required")
    unsupported_extensions = (
        model.by_claims
        or model.capital_injections
        or model.prevention.severity_transform is not None
    )
    if unsupported_extensions:
        raise NotImplementedError(
            "De Vylder approximation currently supports scaled primary claims only"
        )
    scale = _severity_scale(model)
    m1 = scale * mean
    m2 = scale**2 * second
    m3 = scale**3 * _raw_moment(model.claim_distribution, 3)
    if not np.isfinite(m3):
        raise ValueError("finite third moment is required")
    lam_tilde = 9.0 * model.claim_arrival_rate * m2**3 / (2.0 * m3**2)
    mean_tilde = 2.0 * m3 / (3.0 * m2)
    c_tilde = model.premium_rate - model.claim_arrival_rate * m1 + lam_tilde * mean_tilde
    rate_tilde = 1.0 / mean_tilde
    approx_model = CramerLundbergProcess(
        initial_capital=model.initial_capital,
        premium_rate=c_tilde,
        claim_arrival_rate=lam_tilde,
        claim_distribution=ClaimDistribution(
            name="exponential",
            mean_value=mean_tilde,
            variance_value=mean_tilde**2,
            sampler=lambda rng_, n: rng_.exponential(mean_tilde, size=n),
            mgf_function=lambda t: rate_tilde / (rate_tilde - t) if t < rate_tilde else np.inf,
            laplace_function=lambda s: rate_tilde / (rate_tilde + s),
            metadata={"rate": rate_tilde},
        ),
    )
    return ultimate_ruin_exponential(approx_model, u)


def heavy_tail_integrated_tail_asymptotic(
    model: CramerLundbergProcess,
    u: ArrayLike,
    integrated_tail_survival: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Subexponential approximation ``rho/(1-rho) * bar F_I(u)``.

    Pass ``integrated_tail_survival`` for a custom equilibrium tail. When it is
    omitted, the built-in helper is used for the model's scaled primary severity.
    """

    surplus = _as_array(u)
    if integrated_tail_survival is None:
        _primary_claim_formula_check(model)
    rho = model.claim_intensity / model.premium_rate
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if integrated_tail_survival is None:
        tail = _integrated_tail_survival(
            model.claim_distribution,
            surplus,
            scale=_severity_scale(model),
        )
    else:
        tail = integrated_tail_survival(surplus)
    return rho / (1.0 - rho) * tail
