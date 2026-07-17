import math

import numpy as np
from scipy import stats

from ruin_theory import (
    DualRiskProcess,
    PremiumBoundary,
    deterministic,
    dual_poisson_exponential_ruin_time_atom,
    dual_poisson_exponential_ruin_time_cdf,
    dual_poisson_exponential_ruin_time_density,
    exponential,
    linear_death_ospp,
    negative_binomial_ospp,
    ordered_two_sided_survival,
    ordered_win_first_time_density,
    poisson_ospp,
    uniform_order_stat_rect_probability,
)


def test_uniform_order_stat_rect_probability_matches_simple_case():
    assert uniform_order_stat_rect_probability([0.0, 0.5], [0.5, 1.0]) == 0.5
    assert uniform_order_stat_rect_probability([], []) == 1.0


def test_ospp_constructors_match_known_count_laws():
    poisson = poisson_ospp(rate=2.0)
    assert abs(poisson.count_pmf(2, 1.0) - stats.poisson.pmf(2, 2.0)) < 1e-14
    assert poisson.conditional_cdf(0.25, 1.0) == 0.25

    negative_binomial = negative_binomial_ospp(gamma=2.0, beta=3.0)
    assert abs(negative_binomial.mean(6.0) - 4.0) < 1e-14

    death = linear_death_ospp(initial_size=4, death_rate=0.5)
    assert death.count_pmf(5, 1.0) == 0.0


def test_ordered_two_sided_survival_handles_deterministic_claims():
    value = ordered_two_sided_survival(
        poisson_ospp(rate=2.0),
        deterministic(0.1),
        PremiumBoundary.linear(1.0),
        initial_capital=0.5,
        upper_barrier=2.0,
        horizon=1.0,
        max_claims=5,
    )

    assert 0.0 <= value <= 1.0
    assert value > 0.9


def test_ordered_win_first_density_is_finite_for_exponential_claims():
    density = ordered_win_first_time_density(
        poisson_ospp(rate=2.0),
        exponential(rate=1.0),
        PremiumBoundary.linear(1.0),
        initial_capital=0.5,
        upper_barrier=2.0,
        t=2.0,
        max_claims=8,
        n_conditional_samples=128,
        seed=123,
    )

    assert math.isfinite(density)
    assert density > 0.0


def test_dual_poisson_exponential_density_and_atom():
    process = DualRiskProcess(
        initial_capital=1.0,
        cost_rate=1.0,
        profit_arrival_rate=1.0,
        profit_distribution=exponential(rate=1.0),
    )

    assert abs(dual_poisson_exponential_ruin_time_atom(process) - math.exp(-1.0)) < 1e-14
    np.testing.assert_allclose(
        dual_poisson_exponential_ruin_time_density(process, [0.5, 1.0]),
        [0.0, 0.0],
    )
    assert dual_poisson_exponential_ruin_time_density(process, [1.5])[0] > 0.0
    assert dual_poisson_exponential_ruin_time_cdf(process, [0.5, 1.0, 1.5])[1] == math.exp(-1.0)
