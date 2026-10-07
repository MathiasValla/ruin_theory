"""Annual infinite-mean ruin times and finite-mean iso-loss comparisons.

Computational counterparts of Valla (2026), 'How long can premiums compensate
infinite-mean claims?'. Annual random walks are not compound-Poisson models.
"""

from dataclasses import dataclass
import operator

import numpy as np
from scipy import optimize, special


def _positive(value, name, *, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


@dataclass(frozen=True)
class AnnualHeavyTailModel:
    """Annual mixture: Pareto-I heavy claim or exponential/deterministic light claim.

    frequency_factor changes the heavy mixing probability q to q*factor;
    removed mass moves to the light component, not to an independent stream.
    This convention is the one used in the paper's infinite-mean experiments.
    """

    alpha: float = 0.5
    heavy_probability: float = 0.2
    heavy_scale: float = 1.0
    light_mean: float = 0.0
    light_family: str = "exponential"
    frequency_factor: float = 1.0
    heavy_severity_factor: float = 1.0
    light_severity_factor: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        for name in (
            "heavy_probability",
            "heavy_scale",
            "frequency_factor",
            "heavy_severity_factor",
        ):
            _positive(getattr(self, name), name)
        for name in ("light_mean", "light_severity_factor"):
            _positive(getattr(self, name), name, allow_zero=True)
        if self.heavy_probability > 1 or self.heavy_probability * self.frequency_factor > 1:
            raise ValueError("the effective heavy probability must not exceed one")
        if self.light_family not in {"exponential", "deterministic"}:
            raise ValueError("light_family must be exponential or deterministic")

    @property
    def tail_constant(self):
        return (
            self.heavy_probability
            * self.frequency_factor
            * (self.heavy_scale * self.heavy_severity_factor) ** self.alpha
        )

    def scale(self, premium):
        """b_c; first-order mean ruin time, not an exact finite-premium mean."""
        return infinite_mean_ruin_time_scale(premium, self.alpha, self.tail_constant)

    def sample(self, n, rng):
        heavy = rng.random(n) < self.heavy_probability * self.frequency_factor
        light = self.light_mean * self.light_severity_factor
        draws = (
            rng.exponential(light, n) if self.light_family == "exponential" else np.full(n, light)
        )
        count = int(heavy.sum())
        # Capping beyond floating-point range cannot alter a finite-reserve crossing.
        log_claims = (
            np.log(self.heavy_scale * self.heavy_severity_factor)
            - np.log1p(-rng.random(count)) / self.alpha
        )
        draws[heavy] = np.exp(np.minimum(log_claims, np.log(np.finfo(float).max) - 1))
        return draws


def infinite_mean_ruin_time_scale(premium, alpha, tail_constant):
    premium = _positive(premium, "premium")
    constant = _positive(tail_constant, "tail_constant")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    exponent = (alpha * np.log(premium) - np.log(constant) - special.gammaln(1 - alpha)) / (
        1 - alpha
    )
    if exponent > np.log(np.finfo(float).max):
        raise OverflowError("ruin-time scale exceeds floating-point range")
    return float(np.exp(exponent))


def premium_for_infinite_mean_scale(target_scale, alpha, tail_constant):
    """Inverse b_c, useful for comparing tail indices on a common time scale."""
    target = _positive(target_scale, "target_scale")
    constant = _positive(tail_constant, "tail_constant")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    exponent = (
        np.log(constant) + special.gammaln(1 - alpha) + (1 - alpha) * np.log(target)
    ) / alpha
    if exponent > np.log(np.finfo(float).max):
        raise OverflowError("premium exceeds floating-point range")
    return float(np.exp(exponent))


def stable_ruin_time_laplace(s, alpha):
    """E[exp(-s*T_alpha)] = Phi(s)**(alpha-1), Phi-Phi**alpha=s.

    The positive root is >=1; this form avoids cancellation in 1-s/Phi.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    values = np.asarray(s, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("s must be finite and nonnegative")

    def transform(value):
        if value == 0:
            return 1.0
        high = max(2.0, 2 * value)
        while high - high**alpha < value:
            high *= 2
        root = optimize.brentq(lambda x: x - x**alpha - value, 1.0, high)
        return root ** (alpha - 1)

    result = np.array([transform(float(x)) for x in values.ravel()]).reshape(values.shape)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class AnnualRuinTimeEstimate:
    ruin_times: np.ndarray
    mean: float
    standard_error: float
    asymptotic_mean: float
    mean_ratio: float


def simulate_annual_heavy_tail_ruin_times(
    model: AnnualHeavyTailModel,
    *,
    premium: float,
    initial_capital: float = 0,
    n_simulations: int = 1000,
    seed=None,
    max_steps: int = 1_000_000,
):
    """Simulate strict annual ruin without silently censoring survivors.

    Raises RuntimeError if max_steps is reached; returned means always use
    complete trajectories. No diffusion, one-big-jump or continuous-time proxy.
    """
    if not isinstance(model, AnnualHeavyTailModel):
        raise TypeError("model must be AnnualHeavyTailModel")
    premium = _positive(premium, "premium")
    capital = _positive(initial_capital, "initial_capital", allow_zero=True)
    n, limit = operator.index(n_simulations), operator.index(max_steps)
    if n < 2 or limit < 1:
        raise ValueError("n_simulations >=2 and max_steps >=1 are required")
    rng = np.random.default_rng(seed)
    sums = np.zeros(n)
    times = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    light = model.light_mean * model.light_severity_factor
    if light == 0:
        # Between heavy claims the loss is constant and the ruin boundary grows.
        clock = np.zeros(n, dtype=np.int64)
        probability = model.heavy_probability * model.frequency_factor
        while alive.size:
            waits = rng.geometric(probability, alive.size)
            if np.any(waits > limit - clock[alive]):
                raise RuntimeError(f"trajectories censored at max_steps={limit}; no mean returned")
            clock[alive] += waits
            log_claims = (
                np.log(model.heavy_scale * model.heavy_severity_factor)
                - np.log1p(-rng.random(alive.size)) / model.alpha
            )
            sums[alive] += np.exp(np.minimum(log_claims, np.log(np.finfo(float).max) - 1))
            crossed = sums[alive] > capital + premium * clock[alive]
            times[alive[crossed]] = clock[alive[crossed]]
            alive = alive[~crossed]
    else:
        for step in range(1, limit + 1):
            sums[alive] += model.sample(alive.size, rng)
            crossed = sums[alive] > capital + premium * step
            times[alive[crossed]] = step
            alive = alive[~crossed]
            if not alive.size:
                break
        if alive.size:
            raise RuntimeError(
                f"{alive.size} trajectories censored at max_steps={limit}; no mean returned"
            )
    mean = float(times.mean())
    scale = model.scale(premium)
    return AnnualRuinTimeEstimate(
        times, mean, float(times.std(ddof=1) / np.sqrt(n)), scale, mean / scale
    )


def infinite_mean_prevention_multiplier(strength, alpha, *, mechanism="frequency"):
    """First-order mean-time ratio for frequency, heavy severity or light severity."""
    values = np.asarray(strength, dtype=float)
    if not np.isfinite(alpha) or not 0 < alpha < 1 or np.any(~np.isfinite(values)):
        raise ValueError("finite strengths and 0<alpha<1 are required")
    if np.any(values < 0) or np.any(values >= 1):
        raise ValueError("strength must lie in [0,1)")
    exponents = {
        "frequency": -1 / (1 - alpha),
        "heavy_severity": -alpha / (1 - alpha),
        "light_severity": 0.0,
    }
    if mechanism not in exponents:
        raise ValueError("unknown prevention mechanism")
    return (1 - values) ** exponents[mechanism]


@dataclass(frozen=True)
class IsoMeanRuinComparison:
    capital: np.ndarray
    labels: tuple[str, ...]
    probabilities: np.ndarray
    standard_errors: np.ndarray
    traffic_intensities: np.ndarray


def finite_mean_iso_loss_ruin(
    capital,
    *,
    mu=1.0,
    beta=2.5,
    heavy_frequency=0.25,
    light_frequency=0.25,
    premium=0.65,
    removed_loss=0.08,
    prevention_cost=0.0,
    mechanism="frequency",
    n_simulations=100_000,
    seed=None,
):
    """Exact PK Monte Carlo for equal-mean Lomax/exponential counterfactuals.

    Compare baseline, heavy-targeted and light-targeted prevention, removing
    the same annual mean loss d. Both prevented policies pay the same cost.
    Severity prevention scales the stationary excess distribution as well as
    the annual loss weight. Monte Carlo noise need not preserve exact ordering.
    """
    u = np.atleast_1d(np.asarray(capital, dtype=float))
    if u.ndim != 1 or not u.size or np.any(~np.isfinite(u)) or np.any(u < 0):
        raise ValueError("capital must be a finite nonnegative vector")
    mu = _positive(mu, "mu")
    beta = _positive(beta, "beta")
    if beta <= 1:
        raise ValueError("finite-mean Lomax requires beta>1")
    lh, ll = (
        _positive(heavy_frequency, "heavy_frequency"),
        _positive(light_frequency, "light_frequency"),
    )
    premium = _positive(premium, "premium")
    cost = _positive(prevention_cost, "prevention_cost", allow_zero=True)
    d = _positive(removed_loss, "removed_loss", allow_zero=True)
    if d >= min(lh * mu, ll * mu):
        raise ValueError("removed_loss must be smaller than each component's annual mean")
    if mechanism not in {"frequency", "severity"}:
        raise ValueError("mechanism must be frequency or severity")
    n = operator.index(n_simulations)
    if n < 2:
        raise ValueError("n_simulations must be at least two")
    annual = np.array([[lh * mu, ll * mu], [lh * mu - d, ll * mu], [lh * mu, ll * mu - d]])
    premiums = np.array([premium, premium - cost, premium - cost])
    if np.any(premiums <= 0):
        raise ValueError("all policies must have positive net premiums")
    rho = annual.sum(axis=1) / premiums
    if np.any(rho >= 1):
        raise ValueError("all policies must satisfy the net profit condition")
    scales = np.ones((3, 2))
    if mechanism == "severity":
        scales[1, 0], scales[2, 1] = 1 - d / (lh * mu), 1 - d / (ll * mu)
    seeds = np.random.SeedSequence(seed).spawn(3)
    probabilities = []
    for row in range(3):
        rng = np.random.default_rng(seeds[row])
        counts = rng.geometric(1 - rho[row], n) - 1
        ids = np.repeat(np.arange(n), counts)
        heavy = rng.random(ids.size) < annual[row, 0] / annual[row].sum()
        claims = np.empty(ids.size)
        claims[heavy] = rng.pareto(beta - 1, int(heavy.sum())) * (beta - 1) * mu * scales[row, 0]
        claims[~heavy] = rng.exponential(mu * scales[row, 1], int((~heavy).sum()))
        supremum = np.bincount(ids, weights=claims, minlength=n)
        probabilities.append(np.array([(supremum > x).mean() for x in u]))
    p = np.asarray(probabilities)
    return IsoMeanRuinComparison(
        u.copy(), ("baseline", "heavy_targeted", "light_targeted"), p, np.sqrt(p * (1 - p) / n), rho
    )
