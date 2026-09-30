import math

import numpy as np
import pytest

from ruin_theory import (
    FiniteTimeDependentRuinResult,
    FiniteTimeDiscreteTimeBoundsResult,
    FiniteTimeDiscreteTimeRuinResult,
    FiniteTimeLundbergBoundResult,
    castaner_exponential_principle_roots,
    claim_size_intensities_from_functions,
    discounted_premiums,
    discount_factors_from_interest,
    exchangeable_bernoulli_claim_scenarios,
    exponential_lundberg_roots,
    finite_time_dependent_discrete_time_ruin,
    finite_time_discrete_time_bounds,
    finite_time_discrete_time_ruin,
    finite_time_lundberg_bounds,
    normal_lundberg_roots,
    period_lundberg_roots_from_pmf,
    ruin_deficit_cdf,
    ruin_deficit_quantile,
    surplus_cdf_given_survival,
)


def test_discounted_premiums_follow_castaner_timing_conventions():
    factors = discount_factors_from_interest([0.1, 0.2])

    np.testing.assert_allclose(factors, [1.0, 1.1, 1.32])
    np.testing.assert_allclose(
        discounted_premiums([110.0, 120.0], [0.1, 0.2]),
        [110.0, 120.0 / 1.1],
    )
    np.testing.assert_allclose(
        discounted_premiums([110.0, 120.0], [0.1, 0.2], timing="end"),
        [110.0 / 1.1, 120.0 / 1.32],
    )
    np.testing.assert_allclose(
        discounted_premiums([110.0, 120.0], [0.1, 0.2], timing="middle"),
        [110.0 / math.sqrt(1.1), 120.0 / (1.1 * math.sqrt(1.2))],
    )


def test_claim_size_intensity_quadrature_matches_polynomial_integrals():
    matrix = claim_size_intensities_from_functions(
        lambda time: 2.0 * time,
        lambda time: [0.0, time, 1.0 - time],
        [0.5, 1.0],
        max_claim_size=2,
    )

    expected_size_one = [(2.0 / 3.0) * 0.5**3, (2.0 / 3.0) * (1.0 - 0.5**3)]
    expected_size_two = [
        0.5**2 - (2.0 / 3.0) * 0.5**3,
        (1.0 - 0.5**2) - (2.0 / 3.0) * (1.0 - 0.5**3),
    ]
    np.testing.assert_allclose(matrix[:, 1], expected_size_one)
    np.testing.assert_allclose(matrix[:, 2], expected_size_two)


def test_discrete_time_ruin_recursion_matches_hand_check():
    result = finite_time_discrete_time_ruin(
        [[0.5, 0.5], [0.5, 0.5]],
        premiums=[0.0, 0.0],
        initial_capital=0.0,
        return_result=True,
    )

    assert isinstance(result, FiniteTimeDiscreteTimeRuinResult)
    np.testing.assert_allclose(result.survival_probabilities, [0.5, 0.25])
    np.testing.assert_allclose(result.ruin_time_probabilities, [0.5, 0.25])
    assert result.ruin_probability == pytest.approx(0.75)
    np.testing.assert_allclose(
        ruin_deficit_cdf(result, period=0, thresholds=[0.5, 1.0]),
        [0.0, 1.0],
    )
    assert ruin_deficit_quantile(result, period=0, probability=0.95) == pytest.approx(1.0)
    np.testing.assert_allclose(
        surplus_cdf_given_survival(result, period=1, thresholds=[0.0]),
        [1.0],
    )


def test_discrete_time_bounds_are_ordered_for_stochastic_bounds():
    result = finite_time_discrete_time_bounds(
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        premiums=[0.0],
        initial_capital=0.0,
    )

    assert isinstance(result, FiniteTimeDiscreteTimeBoundsResult)
    assert result.ruin_probability_interval == pytest.approx((0.0, 1.0))


def test_dependent_scenario_ruin_handles_exchangeable_bernoulli_law():
    scenarios, probabilities = exchangeable_bernoulli_claim_scenarios([0.25, 0.5, 0.25])
    result = finite_time_dependent_discrete_time_ruin(
        scenarios,
        probabilities,
        premiums=[0.0, 0.0],
        initial_capital=0.0,
        return_result=True,
    )

    assert isinstance(result, FiniteTimeDependentRuinResult)
    assert result.ruin_probability == pytest.approx(0.75)
    assert result.survival_probability == pytest.approx(0.25)
    np.testing.assert_allclose(result.ruin_time_probabilities, [0.5, 0.25])


def test_lundberg_roots_and_bounds_match_closed_forms():
    root = period_lundberg_roots_from_pmf([[0.75, 0.25]], premiums=[0.5])[0]
    bounds = finite_time_lundberg_bounds([root, 1.0], initial_capital=2.0)

    assert isinstance(bounds, FiniteTimeLundbergBoundResult)
    assert root == pytest.approx(2.0 * math.log(3.0))
    np.testing.assert_allclose(bounds.bounds, [math.exp(-2.0 * root), math.exp(-2.0)])
    np.testing.assert_allclose(exponential_lundberg_roots([1.0], [2.0], [0.5]), [1.5])
    np.testing.assert_allclose(normal_lundberg_roots([1.0], [2.0], [3.0]), [1.0])
    np.testing.assert_allclose(
        castaner_exponential_principle_roots([0.1], [1.0], [0.2]),
        [1.1 * 0.2 / 1.2],
    )
    np.testing.assert_allclose(
        castaner_exponential_principle_roots(
            [0.1],
            [1.0],
            [0.2],
            claim_arrival_rates=[2.0],
            principle="standard-deviation",
        ),
        [1.1 * 0.4 / 2.4],
    )
    np.testing.assert_allclose(
        castaner_exponential_principle_roots([0.1], [1.0], [0.2], principle="variance"),
        [1.1 * 0.4 / 1.5],
    )


def test_discrete_time_helpers_validate_inputs():
    with pytest.raises(ValueError, match="timing"):
        discounted_premiums([1.0], [0.1], timing="late")
    with pytest.raises(ValueError, match="sum"):
        exchangeable_bernoulli_claim_scenarios([0.5, 0.6])
    with pytest.raises(ValueError, match="one value per period"):
        finite_time_discrete_time_ruin([[1.0]], premiums=[1.0, 2.0])
    with pytest.raises(ValueError, match="sum to at most one"):
        finite_time_discrete_time_ruin([[0.8, 0.8]], premiums=[1.0])


def test_review_dependent_ruin_is_absorbing_after_premium_recovery():
    result = finite_time_dependent_discrete_time_ruin(
        [[2.0, 0.0], [0.0, 0.0]], [0.4, 0.6], premiums=[1.0, 2.0],
        return_result=True,
    )
    np.testing.assert_allclose(result.survival_probabilities, [0.6, 0.6])
    np.testing.assert_allclose(result.ruin_time_probabilities, [0.4, 0.0])
    assert result.ruin_probability == pytest.approx(0.4)
    np.testing.assert_allclose(result.surplus_distributions[1][0], [3.0])


def test_review_dependent_matches_independent_enumeration_with_recovery():
    scenarios, probabilities = exchangeable_bernoulli_claim_scenarios(
        [0.25, 0.5, 0.25], claim_amount=2.0,
    )
    dependent = finite_time_dependent_discrete_time_ruin(
        scenarios, probabilities, premiums=[1.0, 3.0], return_result=True,
    )
    independent = finite_time_discrete_time_ruin(
        [[0.5, 0.0, 0.5]] * 2, premiums=[1.0, 3.0], return_result=True,
    )
    np.testing.assert_allclose(dependent.survival_probabilities, independent.survival_probabilities)


@pytest.mark.parametrize("pmf", [lambda t: 1.0, lambda t: [], lambda t: [[1.0]]])
def test_review_intensity_rejects_invalid_callback_shape(pmf):
    with pytest.raises(ValueError, match="severity_pmf"):
        claim_size_intensities_from_functions(1.0, pmf, [1.0], max_claim_size=1)


def test_review_intensity_requires_integer_size():
    with pytest.raises(TypeError, match="max_claim_size"):
        claim_size_intensities_from_functions(1.0, [1.0], [1.0], max_claim_size=1.5)


def test_review_lundberg_roots_ignore_zero_mass_padding():
    with np.errstate(over="raise", invalid="raise"):
        roots = period_lundberg_roots_from_pmf([[1.0, 0.0, 0.0]], premiums=[0.5])
    assert np.isnan(roots[0])


def test_review_lundberg_bounds_missing_roots_are_uninformative():
    bounds = finite_time_lundberg_bounds([2.0, np.nan, 1.0], initial_capital=1.0)
    np.testing.assert_allclose(bounds.bounds, [np.exp(-2.0), 1.0, 1.0])
    np.testing.assert_allclose(
        finite_time_lundberg_bounds([2.0, 0.0, 1.0], initial_capital=1.0).bounds,
        [np.exp(-2.0), 1.0, 1.0],
    )


def test_review_deficit_quantile_uses_positive_mass_support():
    result = finite_time_discrete_time_ruin(
        [[0.0, 0.0, 1.0, 0.0]], premiums=[0.0], return_result=True,
    )
    assert ruin_deficit_quantile(result, period=0, probability=0.0) == 2.0
    assert ruin_deficit_quantile(result, period=0, probability=1.0) == 2.0


def test_review_lundberg_roots_require_complete_claim_law():
    with pytest.raises(ValueError, match="sum to one"):
        period_lundberg_roots_from_pmf([[0.5, 0.2]], premiums=[0.5])


def test_review_intensity_retains_short_interval_at_high_rate():
    actual = claim_size_intensities_from_functions(1e16, [0.0, 1.0], [1e-16], max_claim_size=1)
    np.testing.assert_allclose(actual, [[0.0, 1.0]])
