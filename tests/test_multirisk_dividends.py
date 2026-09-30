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


def _one_line_ctmc(**overrides):
    arguments = dict(
        initial_reserves=[0.0], barriers=[1.0], lower_bounds=[0.0], grid_step=0.5,
        environment_generator=[[0.0]], environment_initial=[1.0],
        shocks=[CommonShock([2.0], {(3,): 1.0})], base_premium_rates=[3.0],
    )
    arguments.update(overrides)
    return estimate_multirisk_dividend_penalties_ctmc(**arguments)


@pytest.mark.parametrize("step", [0.25, 0.5, 1.0])
def test_premium_jump_rate_preserves_monetary_drift(step):
    jumps = round(1.0 / step)
    result = _one_line_ctmc(
        grid_step=step, shocks=[CommonShock([2.0], {(jumps + 1,): 1.0})],
    )
    reach_barrier = (3.0 / step / (3.0 / step + 2.0)) ** jumps
    assert result.expected_time_to_ruin == pytest.approx(0.5)
    assert result.expected_dividends[0] == pytest.approx(1.5 * reach_barrier)
    assert sum(result.ruin_state_probabilities.values()) == pytest.approx(1.0)


def test_state_cap_is_checked_before_cartesian_allocation(monkeypatch):
    import ruin_theory.multirisk_dividends as module

    def must_not_allocate(ranges):
        raise AssertionError("state grid was allocated before checking max_states")

    monkeypatch.setattr(module, "_cartesian_product", must_not_allocate)
    with pytest.raises(ValueError, match="max_states"):
        _one_line_ctmc(max_states=2)


def test_ctmc_rejects_reachable_nonabsorbing_class():
    with pytest.raises(ValueError, match="transient|absorbing"):
        _one_line_ctmc(shocks=[CommonShock([0.0], {(3,): 1.0})])


def test_ctmc_rejects_fractional_transition_state_index():
    with pytest.raises(ValueError, match="integer"):
        _one_line_ctmc(
            environment_generator=[[-1.0, 1.0], [1.0, -1.0]],
            environment_initial=[1.0, 0.0],
            shocks=[CommonShock([2.0, 2.0], {(3,): 1.0})],
            transition_claim_pmfs={(0.5, 1): {(0,): 1.0}},
        )


def test_large_reserve_does_not_relax_grid_alignment_tolerance():
    with pytest.raises(ValueError, match="grid_step"):
        _one_line_ctmc(initial_reserves=[100_000.25], barriers=[100_001.0],
                       lower_bounds=[100_000.0], grid_step=1.0)


def test_ctmc_normalizes_initial_probability_roundoff():
    result = _one_line_ctmc(environment_initial=[1.0 - 1e-6])
    assert result.ruin_probability == pytest.approx(1.0, abs=1e-14)
