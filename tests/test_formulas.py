import numpy as np
import pytest

from ruin_theory import (
    ByClaimModel,
    CapitalInjectionModel,
    CramerLundbergProcess,
    PreventionProgram,
    SparreAndersenProcess,
    adjustment_coefficient,
    cramer_lundberg_asymptotic,
    de_vylder_approximation,
    deterministic,
    exponential,
    finite_time_ruin_exponential,
    heavy_tail_integrated_tail_asymptotic,
    integrated_tail_survival,
    lognormal,
    lundberg_bound,
    mixture_exponential,
    pareto,
    pollaczek_khinchine_monte_carlo,
    ultimate_ruin_exponential,
    ultimate_ruin_hyperexponential,
    weibull,
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


def test_exponential_ultimate_ruin_is_zero_without_claim_arrivals():
    model = CramerLundbergProcess(
        premium_rate=0,
        claim_arrival_rate=2,
        claim_distribution=exponential(rate=1),
        prevention=PreventionProgram(frequency_multiplier=0),
    )

    np.testing.assert_allclose(ultimate_ruin_exponential(model, [0.0, 2.0]), [0.0, 0.0])


def test_closed_form_formulas_reject_non_classical_or_windowed_frequency():
    renewal = SparreAndersenProcess(
        premium_rate=1,
        interarrival_distribution=exponential(rate=3),
        claim_distribution=exponential(rate=5),
    )
    with pytest.raises(ValueError, match="CramerLundbergProcess"):
        ultimate_ruin_exponential(renewal, [0.0])

    windowed = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
        prevention=PreventionProgram(frequency_windows=((1.0, 2.0, 0.5),)),
    )
    with pytest.raises(ValueError, match="stationary frequency"):
        adjustment_coefficient(windowed)


def test_formula_surplus_inputs_reject_nan():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
    )

    with pytest.raises(ValueError, match="NaN"):
        ultimate_ruin_exponential(model, [0.0, np.nan])


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


def test_integrated_tail_survival_matches_pareto_closed_form():
    claims = pareto(shape=3.0, scale=2.0)
    u = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    expected = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 12.0, 1.0 / 48.0])
    np.testing.assert_allclose(integrated_tail_survival(claims, u), expected)


def test_integrated_tail_survival_supports_lognormal_and_weibull():
    lognormal_tail = integrated_tail_survival(lognormal(meanlog=0.0, sdlog=0.5), [0.0, 1.0, 3.0])
    assert lognormal_tail[0] == pytest.approx(1.0)
    assert np.all(np.diff(lognormal_tail) < 0.0)

    u = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        integrated_tail_survival(weibull(shape=1.0, scale=2.0), u),
        np.exp(-u / 2.0),
    )


def test_integrated_tail_survival_scales_severity():
    claims = pareto(shape=3.0, scale=2.0)
    u = np.array([1.0, 2.0, 4.0])
    expected = np.array([1.0 / 3.0, 1.0 / 12.0, 1.0 / 48.0])
    np.testing.assert_allclose(integrated_tail_survival(claims, u, scale=0.5), expected)


def test_integrated_tail_survival_requires_finite_pareto_mean():
    with pytest.raises(ValueError, match="finite mean"):
        integrated_tail_survival(pareto(shape=1.0, scale=1.0), [0.0])


def test_heavy_tail_asymptotic_can_use_builtin_integrated_tail():
    model = CramerLundbergProcess(
        premium_rate=4.0,
        claim_arrival_rate=1.0,
        claim_distribution=pareto(shape=3.0, scale=1.0),
        prevention=PreventionProgram(severity_multiplier=2.0),
    )
    u = np.array([1.0, 2.0, 4.0])
    rho = model.claim_intensity / model.premium_rate
    expected = rho / (1.0 - rho) * integrated_tail_survival(
        model.claim_distribution,
        u,
        scale=2.0,
    )
    np.testing.assert_allclose(heavy_tail_integrated_tail_asymptotic(model, u), expected)


def test_adjustment_coefficient_requires_net_profit_condition():
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=2,
        claim_distribution=exponential(rate=1),
    )
    with pytest.raises(ValueError, match="net profit"):
        adjustment_coefficient(model)


def test_de_vylder_rejects_non_classical_extensions():
    renewal = SparreAndersenProcess(
        premium_rate=1,
        interarrival_distribution=exponential(rate=3),
        claim_distribution=exponential(rate=5),
    )
    with pytest.raises(ValueError, match="CramerLundbergProcess"):
        de_vylder_approximation(renewal, [0.0])

    windowed = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=3,
        claim_distribution=exponential(rate=5),
        prevention=PreventionProgram(frequency_windows=((1.0, 2.0, 0.5),)),
    )
    with pytest.raises(ValueError, match="stationary frequency"):
        de_vylder_approximation(windowed, [0.0])

    by_claim = CramerLundbergProcess(
        premium_rate=10,
        claim_arrival_rate=1,
        claim_distribution=exponential(rate=5),
        by_claims=(ByClaimModel(probability=0.5, distribution=deterministic(1), count_mean=1),),
    )
    with pytest.raises(ValueError, match="by-claims"):
        de_vylder_approximation(by_claim, [0.0])

    injected = CramerLundbergProcess(
        premium_rate=10,
        claim_arrival_rate=1,
        claim_distribution=exponential(rate=5),
        capital_injections=(CapitalInjectionModel(rate=1, distribution=deterministic(1)),),
    )
    with pytest.raises(ValueError, match="capital injections"):
        de_vylder_approximation(injected, [0.0])


def test_heavy_tail_custom_tail_keeps_model_support_checks():
    def custom_tail(x):
        return np.ones_like(x, dtype=float)

    renewal = SparreAndersenProcess(
        premium_rate=4,
        interarrival_distribution=exponential(rate=1),
        claim_distribution=pareto(shape=3.0, scale=1.0),
    )
    with pytest.raises(ValueError, match="CramerLundbergProcess"):
        heavy_tail_integrated_tail_asymptotic(renewal, [1.0], custom_tail)

    by_claim = CramerLundbergProcess(
        premium_rate=10,
        claim_arrival_rate=1,
        claim_distribution=pareto(shape=3.0, scale=1.0),
        by_claims=(ByClaimModel(probability=0.5, distribution=deterministic(1), count_mean=1),),
    )
    with pytest.raises(ValueError, match="by-claims"):
        heavy_tail_integrated_tail_asymptotic(by_claim, [1.0], custom_tail)
