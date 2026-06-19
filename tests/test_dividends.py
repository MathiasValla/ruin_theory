import numpy as np
import pytest

from ruin_theory import (
    BarrierDividendAnalyticResult,
    BarrierDividendEstimate,
    BarrierDividendPath,
    barrier_continuation_probability_exponential_interest_force,
    barrier_dividend_analytic_exponential_interest_force,
    barrier_dividend_compound_geometric_cdf,
    barrier_dividend_payment_cdf,
    barrier_dividend_payment_mean,
    barrier_dividend_period_count_pmf,
    barrier_hit_probability_exponential_interest_force,
    deterministic,
    estimate_barrier_dividends,
    expected_cumulative_barrier_dividends,
    simulate_barrier_dividend_path,
    win_first_probability_exponential_interest_force,
)


def test_barrier_payment_period_distribution_matches_loisel_formula():
    x = np.array([0.0, 1.0, 2.0])

    no_interest = barrier_dividend_payment_cdf(
        x,
        claim_arrival_rate=2.0,
        dividend_rate=3.0,
    )
    np.testing.assert_allclose(no_interest, 1.0 - np.exp(-2.0 * x / 3.0))
    assert barrier_dividend_payment_mean(
        claim_arrival_rate=2.0,
        dividend_rate=3.0,
    ) == pytest.approx(1.5)

    interest = barrier_dividend_payment_cdf(
        x,
        claim_arrival_rate=2.0,
        interest_force=0.5,
        dividend_rate=3.0,
    )
    np.testing.assert_allclose(interest, 1.0 - (1.0 + 0.5 * x / 3.0) ** -4.0)
    assert barrier_dividend_payment_mean(
        claim_arrival_rate=2.0,
        interest_force=0.5,
        dividend_rate=3.0,
    ) == pytest.approx(2.0)


def test_barrier_period_count_and_expected_dividends_match_geometric_structure():
    pmf = barrier_dividend_period_count_pmf(
        4,
        hit_probability=0.6,
        continuation_probability=0.25,
    )

    np.testing.assert_allclose(
        pmf,
        [0.4, 0.45, 0.1125, 0.028125, 0.00703125],
    )
    assert expected_cumulative_barrier_dividends(
        hit_probability=0.6,
        continuation_probability=0.25,
        claim_arrival_rate=2.0,
    ) == pytest.approx(0.6 / (1.0 - 0.25) / 2.0)
    assert expected_cumulative_barrier_dividends(
        hit_probability=0.6,
        continuation_probability=0.25,
        claim_arrival_rate=2.0,
        convention="loisel",
    ) == pytest.approx(0.6 * 0.25 / (1.0 - 0.25) / 2.0)


def test_barrier_compound_geometric_cdf_reduces_to_single_exponential_payment():
    x = np.array([0.0, 1.0, 2.0])
    cdf = barrier_dividend_compound_geometric_cdf(
        x,
        hit_probability=0.75,
        continuation_probability=0.0,
        claim_arrival_rate=2.0,
    )

    np.testing.assert_allclose(cdf, 0.25 + 0.75 * (1.0 - np.exp(-2.0 * x)))

    continued = barrier_dividend_compound_geometric_cdf(
        x,
        hit_probability=0.75,
        continuation_probability=0.25,
        claim_arrival_rate=2.0,
    )
    np.testing.assert_allclose(continued, 1.0 - 0.75 * np.exp(-1.5 * x))


def test_barrier_hit_probability_is_win_first_probability():
    kwargs = dict(
        premium_rate=1.2,
        claim_arrival_rate=0.7,
        claim_rate=1.4,
        interest_force=0.08,
    )

    hit = barrier_hit_probability_exponential_interest_force(1.0, barrier=3.0, **kwargs)
    win_first = win_first_probability_exponential_interest_force(1.0, 2.0, **kwargs)

    assert hit == pytest.approx(win_first)
    assert barrier_hit_probability_exponential_interest_force(3.0, barrier=3.0, **kwargs) == 1.0


def test_barrier_continuation_and_analytic_summary_are_available():
    result = barrier_dividend_analytic_exponential_interest_force(
        initial_capital=1.0,
        barrier=3.0,
        premium_rate=1.2,
        claim_arrival_rate=0.7,
        claim_rate=1.4,
        interest_force=0.08,
    )

    assert isinstance(result, BarrierDividendAnalyticResult)
    assert 0.0 < result.hit_probability <= 1.0
    assert 0.0 <= result.continuation_probability < 1.0
    assert result.expected_dividends >= result.loisel_expected_dividends >= 0.0
    assert barrier_continuation_probability_exponential_interest_force(
        barrier=0.0,
        premium_rate=1.2,
        claim_arrival_rate=0.7,
        claim_rate=1.4,
    ) == 0.0


def test_simulate_barrier_dividend_path_pays_premium_at_the_barrier_until_ruin():
    path = simulate_barrier_dividend_path(
        deterministic(2.0),
        initial_capital=1.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        barrier=1.0,
        seed=123,
    )

    assert isinstance(path, BarrierDividendPath)
    assert path.ruined
    assert path.ruin_time == pytest.approx(path.total_dividends)
    assert path.reserves[-1] == pytest.approx(-1.0)
    assert path.claim_sizes.tolist() == [2.0]


def test_estimate_barrier_dividends_collects_totals_and_ruin_times():
    estimate = estimate_barrier_dividends(
        deterministic(2.0),
        initial_capital=1.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        barrier=1.0,
        n_simulations=8,
        seed=123,
    )

    assert isinstance(estimate, BarrierDividendEstimate)
    assert estimate.n_simulations == 8
    assert estimate.probability_ruin == 1.0
    np.testing.assert_allclose(estimate.total_dividends, estimate.ruin_times)
    assert estimate.mean_dividends == pytest.approx(float(np.mean(estimate.total_dividends)))


def test_barrier_dividend_validates_arguments():
    with pytest.raises(ValueError, match="claim_arrival_rate"):
        barrier_dividend_payment_mean(claim_arrival_rate=0.1, interest_force=0.2)
    with pytest.raises(ValueError, match="convention"):
        expected_cumulative_barrier_dividends(
            hit_probability=0.5,
            continuation_probability=0.2,
            claim_arrival_rate=1.0,
            convention="bad",
        )
    with pytest.raises(ValueError, match="initial_capital"):
        simulate_barrier_dividend_path(
            deterministic(1.0),
            initial_capital=2.0,
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            barrier=1.0,
        )
    with pytest.raises(ValueError, match="finite mean"):
        barrier_dividend_analytic_exponential_interest_force(
            initial_capital=0.5,
            barrier=1.0,
            premium_rate=1.0,
            claim_arrival_rate=0.1,
            claim_rate=2.0,
            interest_force=0.2,
        )
