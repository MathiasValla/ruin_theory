"""Multirisk dividend and insolvency-penalty CTMC tests."""

import numpy as np
import pytest

from ruin_theory import (
    CommonShock,
    estimate_multirisk_dividend_penalties_ctmc,
    linear_status_premium_function,
    multirisk_dividend_convergence,
)


def test_single_line_barrier_dividends_match_exponential_clock():
    result = estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[1.0],
        barriers=[1.0],
        lower_bounds=[0.0],
        grid_step=1.0,
        environment_generator=[[0.0]],
        environment_initial=[1.0],
        shocks=[CommonShock(intensities=[2.0], claim_pmfs={(2,): 1.0})],
        base_premium_rates=[3.0],
        ruin_lines=[0],
    )

    assert result.state_count == 2
    assert result.expected_time_to_ruin == pytest.approx(0.5)
    assert result.ruin_probability == pytest.approx(1.0)
    np.testing.assert_allclose(result.expected_dividends, [1.5])
    np.testing.assert_allclose(result.expected_penalties, [0.0])
    assert result.ruin_state_probabilities[(-1.0,)] == pytest.approx(1.0)
    np.testing.assert_allclose(result.expected_deficit_at_ruin, [1.0])


def test_secondary_insolvency_turns_barrier_excess_into_penalty():
    result = estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[1.0, -1.0],
        barriers=[1.0, 1.0],
        lower_bounds=[0.0, -1.0],
        grid_step=1.0,
        environment_generator=[[0.0]],
        environment_initial=[1.0],
        shocks=[CommonShock(intensities=[1.0], claim_pmfs={(2, 0): 1.0})],
        base_premium_rates=[2.0, 0.0],
        ruin_lines=[0],
    )

    assert result.expected_time_to_ruin == pytest.approx(1.0)
    np.testing.assert_allclose(result.expected_dividends, [0.0, 0.0])
    np.testing.assert_allclose(result.expected_penalties, [2.0, 0.0])
    assert result.ruin_state_probabilities[(-1.0, -1.0)] == pytest.approx(1.0)


def test_linear_status_interaction_can_increase_main_premium_at_secondary_barrier():
    premium = linear_status_premium_function(
        [1.0, 0.0],
        interaction_matrix=[[0.0, 0.5], [0.0, 0.0]],
    )
    result = estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[1.0, 1.0],
        barriers=[1.0, 1.0],
        lower_bounds=[0.0, 0.0],
        grid_step=1.0,
        environment_generator=[[0.0]],
        environment_initial=[1.0],
        shocks=[CommonShock(intensities=[1.0], claim_pmfs={(2, 0): 1.0})],
        premium_rate_function=premium,
        ruin_lines=[0],
    )

    np.testing.assert_allclose(result.expected_dividends, [1.5, 0.0])


def test_transition_claims_and_convergence_diagnostics():
    coarse = estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[1.0],
        barriers=[1.0],
        lower_bounds=[0.0],
        grid_step=1.0,
        environment_generator=[[-1.0, 1.0], [0.0, 0.0]],
        environment_initial=[1.0, 0.0],
        shocks=[CommonShock(intensities=[0.0, 0.0], claim_pmfs={(0,): 1.0})],
        base_premium_rates=[0.0],
        transition_claim_pmfs={(0, 1): {(2,): 1.0}},
        ruin_lines=[0],
    )
    fine = estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[1.0],
        barriers=[1.0],
        lower_bounds=[0.0],
        grid_step=0.5,
        environment_generator=[[-1.0, 1.0], [0.0, 0.0]],
        environment_initial=[1.0, 0.0],
        shocks=[CommonShock(intensities=[0.0, 0.0], claim_pmfs={(0,): 1.0})],
        base_premium_rates=[0.0],
        transition_claim_pmfs={(0, 1): {(4,): 1.0}},
        ruin_lines=[0],
    )
    convergence = multirisk_dividend_convergence([fine, coarse])

    assert coarse.expected_time_to_ruin == pytest.approx(1.0)
    assert coarse.ruin_state_probabilities[(-1.0,)] == pytest.approx(1.0)
    np.testing.assert_allclose(convergence.grid_steps, [1.0, 0.5])
    np.testing.assert_allclose(convergence.expected_time_to_ruin, [1.0, 1.0])
    assert convergence.last_time_change == pytest.approx(0.0)


def test_multirisk_dividend_ctmc_argument_validation():
    shock = CommonShock(intensities=[1.0], claim_pmfs={(1,): 1.0})

    with pytest.raises(ValueError, match="grid_step"):
        estimate_multirisk_dividend_penalties_ctmc(
            initial_reserves=[0.5],
            barriers=[1.0],
            lower_bounds=[0.0],
            grid_step=1.0,
            environment_generator=[[0.0]],
            environment_initial=[1.0],
            shocks=[shock],
            base_premium_rates=[0.0],
        )
    with pytest.raises(ValueError, match="base_premium_rates"):
        estimate_multirisk_dividend_penalties_ctmc(
            initial_reserves=[1.0],
            barriers=[1.0],
            lower_bounds=[0.0],
            grid_step=1.0,
            environment_generator=[[0.0]],
            environment_initial=[1.0],
            shocks=[shock],
        )
    with pytest.raises(ValueError, match="max_states"):
        estimate_multirisk_dividend_penalties_ctmc(
            initial_reserves=[1.0],
            barriers=[3.0],
            lower_bounds=[0.0],
            grid_step=1.0,
            environment_generator=[[0.0]],
            environment_initial=[1.0],
            shocks=[shock],
            base_premium_rates=[0.0],
            max_states=2,
        )
