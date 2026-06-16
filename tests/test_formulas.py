import numpy as np
import pytest

from ruin_theory import (
    CramerLundbergProcess,
    PreventionProgram,
    adjustment_coefficient,
    cramer_lundberg_asymptotic,
    deterministic,
    exponential,
    finite_time_ruin_exponential,
    lundberg_bound,
    mixture_exponential,
    pollaczek_khinchine_monte_carlo,
    ultimate_ruin_exponential,
    ultimate_ruin_hyperexponential,
)


def test_exponential_ultimate_ruin_matches_closed_form():
    model = CramerLundbergProcess(
        initial_capital=0,
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
    )
    u = np.array([0.0, 1.0, 2.0])
    expected = 0.6 * np.exp(-2.0 * u)
    np.testing.assert_allclose(ultimate_ruin_exponential(model, u), expected)


def test_adjustment_coefficient_for_exponential_claims():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
    )
    assert abs(adjustment_coefficient(model) - 2.0) < 1e-9


def test_exponential_formula_accounts_for_severity_prevention():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
        prevention=PreventionProgram(severity_multiplier=0.5),
    )
    u = np.array([0.0, 1.0])
    expected = 0.3 * np.exp(-7.0 * u)
    np.testing.assert_allclose(ultimate_ruin_exponential(model, u), expected)


def test_exponential_ultimate_ruin_is_one_without_net_profit_condition():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=2,
        claim_distribution=exponential(rate=1),
    )
    np.testing.assert_allclose(ultimate_ruin_exponential(model, [0.0, 2.0]), [1.0, 1.0])


def test_lundberg_bound_dominates_exact_exponential_ruin():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
    )
    u = np.linspace(0, 5, 11)
    assert np.all(ultimate_ruin_exponential(model, u) <= lundberg_bound(model, u))


def test_cramer_lundberg_asymptotic_is_exact_for_exponential_claims():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
    )
    u = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        cramer_lundberg_asymptotic(model, u),
        ultimate_ruin_exponential(model, u),
        rtol=1e-7,
    )


def test_hyperexponential_gerber_actuar_example():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=mixture_exponential(rates=[3, 7], weights=[0.5, 0.5]),
    )
    u = np.array([0.0, 1.0, 2.0])
    expected = (24 * np.exp(-u) + np.exp(-6 * u)) / 35
    np.testing.assert_allclose(ultimate_ruin_hyperexponential(model, u), expected, rtol=2e-4)


def test_hyperexponential_asymptotic_uses_leading_gerber_root():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=mixture_exponential(rates=[3, 7], weights=[0.5, 0.5]),
    )
    u = np.array([4.0, 6.0])
    expected = (24 / 35) * np.exp(-u)
    np.testing.assert_allclose(cramer_lundberg_asymptotic(model, u), expected, rtol=1e-6)


def test_finite_time_exponential_is_between_zero_and_ultimate():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1),
    )
    finite = finite_time_ruin_exponential(model, u=1.0, horizon=2.0)
    ultimate = ultimate_ruin_exponential(model, np.array([1.0]))[0]
    assert 0.0 <= finite <= ultimate


def test_finite_time_exponential_respects_horizon_limits():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1),
    )
    short = finite_time_ruin_exponential(model, u=1.0, horizon=1.0)
    long = finite_time_ruin_exponential(model, u=1.0, horizon=50.0)
    ultimate = ultimate_ruin_exponential(model, np.array([1.0]))[0]
    assert finite_time_ruin_exponential(model, u=1.0, horizon=0.0) == 0.0
    assert short < long
    assert abs(long - ultimate) < 2e-4


def test_pollaczek_khinchine_mc_estimator_matches_exponential_formula():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1),
    )
    u = np.array([0.0, 1.0, 2.0])
    estimates = pollaczek_khinchine_monte_carlo(model, u, n_simulations=50_000, seed=42)
    np.testing.assert_allclose(estimates, ultimate_ruin_exponential(model, u), atol=0.015)


def test_pollaczek_khinchine_mc_supports_deterministic_integrated_tail():
    model = CramerLundbergProcess(
        premium_rate=3,
        claim_arrival_rate=1,
        claim_distribution=deterministic(1),
    )
    estimates = pollaczek_khinchine_monte_carlo(model, [0.0, 1.0], n_simulations=2_000, seed=42)
    assert estimates.shape == (2,)
    assert np.all((0.0 <= estimates) & (estimates <= 1.0))


def test_adjustment_coefficient_requires_net_profit_condition():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=2,
        claim_distribution=exponential(rate=1),
    )
    with pytest.raises(ValueError, match="net profit"):
        adjustment_coefficient(model)
