import math

import numpy as np
import pytest
from scipy import stats

from ruin_theory import (
    DualRiskProcess,
    PremiumBoundary,
    deterministic,
    dual_poisson_exponential_ruin_time_atom,
    dual_poisson_exponential_ruin_time_cdf,
    dual_poisson_exponential_ruin_time_density,
    dual_mixed_poisson_ruin_time_density,
    exponential,
    linear_death_ospp,
    linear_birth_immigration_ospp,
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


def test_review_dual_density_matches_poisson_gamma_mixture():
    process = DualRiskProcess(1.0, 1.0, 1.0, exponential(rate=1.0))
    for t in [1.01, 1.5, 5.0, 20.0]:
        expected = dual_mixed_poisson_ruin_time_density(
            lambda n, t: stats.poisson.pmf(n, t),
            lambda n, x: stats.gamma.pdf(x, a=n),
            1.0, 1.0, t, max_count=150,
        )
        assert float(dual_poisson_exponential_ruin_time_density(process, t)) == pytest.approx(expected)


def test_review_dual_density_normalizes_and_is_finite_at_long_times():
    from scipy.integrate import quad

    process = DualRiskProcess(1.0, 1.0, 1.0, exponential(rate=1.0))
    integral = quad(lambda t: float(dual_poisson_exponential_ruin_time_density(process, t)), 1, np.inf)[0]
    assert integral + dual_poisson_exponential_ruin_time_atom(process) == pytest.approx(1.0, abs=1e-8)
    density = dual_poisson_exponential_ruin_time_density(process, [1000.0, np.inf])
    assert np.isfinite(density).all()
    assert density[0] > 0.0
    assert density[1] == 0.0


@pytest.mark.parametrize("process", [linear_death_ospp(4, 0.5), linear_birth_immigration_ospp(1.0, 0.5)])
def test_review_ospp_cdf_handles_small_positive_times(process):
    assert process.conditional_cdf(0.5e-18, 1e-18) == pytest.approx(0.5)


def test_review_order_statistic_rectangle_is_stable_for_many_points():
    assert uniform_order_stat_rect_probability(np.zeros(60), np.ones(60)) == pytest.approx(1.0)
    assert uniform_order_stat_rect_probability(np.full(60, 0.1), np.full(60, 0.9)) == pytest.approx(0.8**60)
    lower = np.zeros(60)
    upper = np.ones(60)
    upper[:30] = 0.5
    assert uniform_order_stat_rect_probability(lower, upper) == pytest.approx(stats.binom.sf(29, 60, 0.5))


def test_review_negative_binomial_auto_truncation_meets_tail_tolerance():
    process = negative_binomial_ospp(gamma=1.0, beta=1.0)
    probabilities = process.count_probabilities(10.0, tail_tol=1e-10)
    assert 0.0 <= 1.0 - probabilities.sum() <= 1e-10


def test_review_dual_cdf_has_no_artificial_flat_segment_after_atom():
    process = DualRiskProcess(1.0, 1.0, 1.0, exponential(rate=1.0))
    cdf = dual_poisson_exponential_ruin_time_cdf(process, [1.0, 1.0 + 1e-6])
    assert cdf[1] > cdf[0]
