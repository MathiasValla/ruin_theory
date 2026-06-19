"""Top-level public API smoke tests."""

import numpy as np

from ruin_theory import (
    CramerLundbergProcess,
    ConstantPreventionResult,
    ExpectedSurplusPreventionResult,
    HeavyTailPreventionResult,
    PeriodicPreventionResult,
    PreventionProgram,
    RiskProcess,
    deterministic,
    estimate_ruin_probability,
    plot_path,
    plot_paths,
    plot_prevention_calendar,
    plot_ruin_curve,
    plot_ruin_time_histogram,
    plot_terminal_reserve_distribution,
    optimize_constant_prevention,
    optimize_expected_surplus_prevention,
    optimize_heavy_tail_prevention_calendar,
    optimize_periodic_prevention_calendar,
    simulate_path,
    simulate_terminal_reserves,
)
import ruin_theory as rt


def test_simulation_and_diagnostics_are_top_level_exports():
    exported = {
        "RiskProcess",
        "ConstantPreventionResult",
        "ExpectedSurplusPreventionResult",
        "HeavyTailPreventionResult",
        "PeriodicPreventionResult",
        "estimate_ruin_probability",
        "optimize_constant_prevention",
        "optimize_expected_surplus_prevention",
        "optimize_heavy_tail_prevention_calendar",
        "optimize_periodic_prevention_calendar",
        "plot_path",
        "plot_paths",
        "plot_prevention_calendar",
        "plot_ruin_curve",
        "plot_ruin_time_histogram",
        "plot_terminal_reserve_distribution",
        "simulate_path",
        "simulate_terminal_reserves",
    }

    assert exported <= set(rt.__all__)
    for name in exported:
        assert getattr(rt, name) is globals()[name]


def test_top_level_simulation_helpers_remain_usable():
    model = CramerLundbergProcess(
        initial_capital=2.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        claim_distribution=deterministic(1.0),
        prevention=PreventionProgram(frequency_multiplier=0.0),
    )

    assert isinstance(model, RiskProcess)
    path = simulate_path(model, horizon=3.0, seed=123)
    estimate = estimate_ruin_probability(model, horizon=3.0, n_simulations=5, seed=123)
    terminal_reserves = simulate_terminal_reserves(
        model,
        horizon=3.0,
        n_simulations=5,
        seed=123,
    )

    assert not path.ruined
    assert estimate.probability == 0.0
    np.testing.assert_allclose(terminal_reserves, np.full(5, 5.0))
