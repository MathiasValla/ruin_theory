"""Top-level public API smoke tests."""

import numpy as np

from ruin_theory import (
    BINARByClaimModel,
    CramerLundbergProcess,
    ConstantPreventionResult,
    ExpectedSurplusPreventionResult,
    GerberShiuResult,
    HeavyTailPreventionResult,
    INARByClaimModel,
    IntegerByClaimPath,
    PeriodicPreventionResult,
    PreventionProgram,
    RiskProcess,
    deterministic,
    estimate_binar_byclaim_ruin_probability,
    estimate_inar_byclaim_ruin_probability,
    estimate_integer_byclaim_ruin_probability,
    estimate_gerber_shiu,
    estimate_ruin_probability,
    gerber_shiu_from_paths,
    lomax,
    plot_integer_byclaim_counts,
    plot_integer_byclaim_path,
    plot_deficit_at_ruin,
    plot_gerber_shiu_scatter,
    plot_path,
    plot_paths,
    plot_periodic_pressure,
    plot_prevention_calendar,
    plot_ruin_curve,
    plot_ruin_time_histogram,
    plot_surplus_before_ruin,
    plot_terminal_reserve_distribution,
    optimize_constant_prevention,
    optimize_expected_surplus_prevention,
    optimize_heavy_tail_prevention_calendar,
    optimize_periodic_prevention_calendar,
    periodic_controlled_pressure,
    periodic_lundberg_coefficient,
    periodic_net_profit,
    periodic_pressure_weights,
    simulate_path,
    simulate_binar_byclaim_path,
    simulate_binar_byclaim_terminal_reserves,
    simulate_inar_byclaim_path,
    simulate_inar_byclaim_terminal_reserves,
    simulate_integer_byclaim_path,
    simulate_integer_byclaim_terminal_reserves,
    simulate_terminal_reserves,
)
import ruin_theory as rt


def test_simulation_and_diagnostics_are_top_level_exports():
    exported = {
        "RiskProcess",
        "BINARByClaimModel",
        "ConstantPreventionResult",
        "ExpectedSurplusPreventionResult",
        "GerberShiuResult",
        "HeavyTailPreventionResult",
        "PeriodicPreventionResult",
        "INARByClaimModel",
        "IntegerByClaimPath",
        "estimate_binar_byclaim_ruin_probability",
        "estimate_inar_byclaim_ruin_probability",
        "estimate_integer_byclaim_ruin_probability",
        "estimate_gerber_shiu",
        "estimate_ruin_probability",
        "gerber_shiu_from_paths",
        "lomax",
        "optimize_constant_prevention",
        "optimize_expected_surplus_prevention",
        "optimize_heavy_tail_prevention_calendar",
        "optimize_periodic_prevention_calendar",
        "periodic_controlled_pressure",
        "periodic_lundberg_coefficient",
        "periodic_net_profit",
        "periodic_pressure_weights",
        "plot_integer_byclaim_counts",
        "plot_integer_byclaim_path",
        "plot_deficit_at_ruin",
        "plot_gerber_shiu_scatter",
        "plot_path",
        "plot_paths",
        "plot_periodic_pressure",
        "plot_prevention_calendar",
        "plot_ruin_curve",
        "plot_ruin_time_histogram",
        "plot_surplus_before_ruin",
        "plot_terminal_reserve_distribution",
        "simulate_path",
        "simulate_binar_byclaim_path",
        "simulate_binar_byclaim_terminal_reserves",
        "simulate_inar_byclaim_path",
        "simulate_inar_byclaim_terminal_reserves",
        "simulate_integer_byclaim_path",
        "simulate_integer_byclaim_terminal_reserves",
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
