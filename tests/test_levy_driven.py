import numpy as np

from ruin_theory import (
    CompoundPoissonSubordinator,
    GammaSubordinator,
    InverseGaussianSubordinator,
    LevelDependentLevyRiskProcess,
    exponential_convergence_rate,
    finite_to_ultimate_ruin_gap_bound,
    gamma,
)


def test_compound_poisson_diffusion_rate_reproduces_published_table():
    liabilities = CompoundPoissonSubordinator(rate=1.0, jump_distribution=gamma(shape=2, rate=1))
    # Safety loading eta=0.1 gives p=(1+eta) E[L_1] = 2.2.
    process = LevelDependentLevyRiskProcess(
        premium_rate=2.2,
        liability=liabilities,
        diffusion=0.0,
    )

    result = exponential_convergence_rate(process)

    assert abs(result.rate - 0.00319) < 5e-6
    assert abs(result.lambda_star - 0.03127086) < 5e-6


def test_gamma_and_inverse_gaussian_subordinator_rates_are_usable():
    gamma_process = LevelDependentLevyRiskProcess(
        premium_rate=1.2,
        liability=GammaSubordinator(alpha=0.5, beta=0.5),
        diffusion=0.0,
    )
    inverse_gaussian = LevelDependentLevyRiskProcess(
        premium_rate=1.2,
        liability=InverseGaussianSubordinator(gamma=1.0),
        diffusion=0.0,
    )

    gamma_rate = exponential_convergence_rate(gamma_process)
    ig_rate = exponential_convergence_rate(inverse_gaussian)

    assert gamma_rate.rate > 0.0
    assert ig_rate.rate > gamma_rate.rate


def test_finite_to_ultimate_gap_bound_uses_stationary_exponential_moment():
    process = LevelDependentLevyRiskProcess(
        premium_rate=2.2,
        liability=CompoundPoissonSubordinator(rate=1.0, jump_distribution=gamma(shape=2, rate=1)),
    )
    result = exponential_convergence_rate(process)

    np.testing.assert_allclose(
        finite_to_ultimate_ruin_gap_bound(result, [0.0, 1.0], stationary_exponential_moment=3.0),
        4.0 * np.exp(-result.rate * np.array([0.0, 1.0])),
    )
