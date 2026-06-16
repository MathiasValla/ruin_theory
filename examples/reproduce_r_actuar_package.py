"""Reproduce selected examples from ``R_actuar_package.pdf``.

The examples below mirror the visible numerical outputs in the actuar note:

- ``adjCoef`` for exponential claim sizes and exponential waiting times;
- proportional reinsurance adjustment coefficients;
- ``ruin`` for exponential/exponential and hyperexponential/exponential models;
- Beekman's convolution/Panjer bounds for the Pareto example.

The small Panjer/discretization helpers are kept local to this example because
the package does not yet expose an aggregate-claim distribution API.
"""

from __future__ import annotations

import numpy as np

from ruin_theory import (
    ClaimDistribution,
    CramerLundbergProcess,
    adjustment_coefficient,
    exponential,
    heavy_tail_integrated_tail_asymptotic,
    mixture_exponential,
    ultimate_ruin_exponential,
    ultimate_ruin_hyperexponential,
)


def _print_vector(name: str, values: np.ndarray, digits: int = 4) -> None:
    print(f"{name}:")
    print(np.array2string(np.asarray(values), precision=digits, suppress_small=False))
    print()


def _lomax_claim(shape: float, scale: float) -> ClaimDistribution:
    """Lomax/Pareto-II law used by actuar's ``ppareto(x, shape, scale)``."""

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return scale * rng.pareto(shape, size=n)

    def cdf(x):
        values = np.asarray(x, dtype=float)
        return np.where(values < 0.0, 0.0, 1.0 - (scale / (scale + values)) ** shape)

    def survival(x):
        values = np.asarray(x, dtype=float)
        return np.where(values < 0.0, 1.0, (scale / (scale + values)) ** shape)

    return ClaimDistribution(
        name="lomax",
        mean_value=scale / (shape - 1.0),
        variance_value=scale**2 * shape / ((shape - 1.0) ** 2 * (shape - 2.0)),
        sampler=sampler,
        cdf_function=cdf,
        survival_function=survival,
        metadata={"shape": shape, "scale": scale},
    )


def _lomax_cdf(x: np.ndarray | float, *, shape: float, scale: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.where(values < 0.0, 0.0, 1.0 - (scale / (scale + values)) ** shape)


def _discretize_lomax_equilibrium(method: str, *, to: int = 200, step: int = 1) -> np.ndarray:
    """actuar-style lower/upper discretization for H(x)=Pareto(4,4)."""

    if method == "lower":
        grid = np.arange(0, to + step, step)
        masses = np.empty_like(grid, dtype=float)
        masses[0] = _lomax_cdf(0.0, shape=4.0, scale=4.0)
        masses[1:] = _lomax_cdf(grid[1:], shape=4.0, scale=4.0) - _lomax_cdf(
            grid[:-1],
            shape=4.0,
            scale=4.0,
        )
        return masses
    if method == "upper":
        grid = np.arange(0, to, step)
        return _lomax_cdf(grid + step, shape=4.0, scale=4.0) - _lomax_cdf(
            grid,
            shape=4.0,
            scale=4.0,
        )
    raise ValueError("method must be 'lower' or 'upper'")


def _compound_geometric_cdf(severity_pmf: np.ndarray, *, probability: float) -> np.ndarray:
    """Compound geometric CDF for N on {0, 1, ...} with P(N=n)=p(1-p)^n."""

    masses = np.asarray(severity_pmf, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("severity_pmf must be a non-empty vector")
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    q = 1.0 - p
    aggregate = np.zeros_like(masses)
    aggregate[0] = p / (1.0 - q * masses[0])
    scale = q / (1.0 - q * masses[0])
    for k in range(1, masses.size):
        aggregate[k] = scale * sum(masses[j] * aggregate[k - j] for j in range(1, k + 1))
    return np.cumsum(aggregate)


def main() -> None:
    # actuar: adjCoef(mgfexp(x), mgfexp(x, 2), premium.rate=2.4, upper=1)
    model = CramerLundbergProcess(
        premium_rate=2.4,
        claim_arrival_rate=2.0,
        claim_distribution=exponential(rate=1.0),
    )
    print(f"Adjustment coefficient: {adjustment_coefficient(model):.4f}")
    print()

    # actuar proportional reinsurance example.
    alphas = np.array([0.75, 0.80, 0.90, 1.00])
    coefficients = []
    for alpha in alphas:
        retained = CramerLundbergProcess(
            premium_rate=2.6 * alpha - 0.2,
            claim_arrival_rate=2.0,
            claim_distribution=exponential(rate=1.0 / alpha),
        )
        coefficients.append(adjustment_coefficient(retained))
    _print_vector("Proportional reinsurance coefficients", np.asarray(coefficients))

    # actuar: ruin(claims="e", rate=5, wait="e", rate=3); psi(0:10)
    exponential_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    u = np.arange(11)
    _print_vector("Exponential/exponential ruin psi(0:10)", ultimate_ruin_exponential(exponential_model, u), 3)

    # actuar: mixture Exp(3), Exp(7), weights 0.5.
    hyper_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=mixture_exponential(rates=[3.0, 7.0], weights=[0.5, 0.5]),
    )
    gerber_formula = (24.0 * np.exp(-u) + np.exp(-6.0 * u)) / 35.0
    hyper_values = ultimate_ruin_hyperexponential(hyper_model, u)
    _print_vector("Hyperexponential ruin psi(0:10)", hyper_values, 7)
    print(f"Matches (24 exp(-u) + exp(-6u))/35: {np.allclose(hyper_values, gerber_formula)}")
    print()

    # actuar Beekman/Panjer Pareto example. Claims are Lomax(shape=5, scale=4),
    # mean 1, premium c=1.2 lambda mu, and the equilibrium tail H is Lomax(4,4).
    f_lower = _discretize_lomax_equilibrium("lower")
    f_upper = _discretize_lomax_equilibrium("upper")
    cdf_lower = _compound_geometric_cdf(f_lower, probability=1.0 / 6.0)
    cdf_upper = _compound_geometric_cdf(f_upper, probability=1.0 / 6.0)
    grid = np.arange(0, 55, 5)
    bounds = np.column_stack([1.0 - cdf_upper[grid], 1.0 - cdf_lower[grid]])
    print("Beekman/Panjer bounds from actuar Table:")
    print("u    lower       upper")
    for surplus, (lower, upper) in zip(grid, bounds, strict=True):
        print(f"{surplus:2d}  {lower:0.7f}  {upper:0.5f}")
    print()

    lomax_model = CramerLundbergProcess(
        premium_rate=1.2,
        claim_arrival_rate=1.0,
        claim_distribution=_lomax_claim(shape=5.0, scale=4.0),
    )
    asymptotic_u = np.array([20.0, 50.0, 100.0])
    equilibrium_tail = lambda x: (4.0 / (4.0 + np.asarray(x, dtype=float))) ** 4
    _print_vector(
        "Built-in heavy-tail asymptotic with custom equilibrium tail",
        heavy_tail_integrated_tail_asymptotic(
            lomax_model,
            asymptotic_u,
            integrated_tail_survival=equilibrium_tail,
        ),
        7,
    )


if __name__ == "__main__":
    main()
