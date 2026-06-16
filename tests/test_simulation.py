import numpy as np
import pytest

import ruin_theory.simulation as simulation_module
from ruin_theory import (
    ByClaimModel,
    CapitalInjectionModel,
    CramerLundbergProcess,
    PreventionProgram,
    SparreAndersenProcess,
    deterministic,
    exponential,
)
from ruin_theory.simulation import estimate_ruin_probability, simulate_path


class FixedByClaim(ByClaimModel):
    def __init__(self, amount):
        super().__init__(probability=1.0, distribution=deterministic(0), count_mean=0)
        object.__setattr__(self, "amount", float(amount))

    def sample_total(self, n_primary, rng):
        return np.full(n_primary, self.amount, dtype=float)


def test_simulate_path_returns_consistent_arrays():
    model = CramerLundbergProcess(
        initial_capital=2,
        premium_rate=1,
        claim_arrival_rate=1,
        claim_distribution=exponential(rate=2),
    )
    path = simulate_path(model, horizon=5, seed=123)
    assert path.times[0] == 0
    assert path.times[-1] <= 5
    assert path.times.shape == path.reserves.shape


def test_zero_frequency_prevention_suppresses_claims():
    model = CramerLundbergProcess(
        initial_capital=2,
        premium_rate=1,
        claim_arrival_rate=100,
        claim_distribution=deterministic(100),
        prevention=PreventionProgram(frequency_multiplier=0),
    )
    path = simulate_path(model, horizon=3, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.times, [0, 3])
    np.testing.assert_allclose(path.reserves, [2, 5])
    assert path.claim_times.size == 0
    assert not path.ruined


def test_frequency_window_prevention_delays_claim_clock():
    model = SparreAndersenProcess(
        initial_capital=0,
        premium_rate=0,
        interarrival_distribution=deterministic(1),
        claim_distribution=deterministic(1),
        prevention=PreventionProgram(
            frequency_multiplier=0,
            frequency_windows=((2.0, 4.0, 1.0),),
        ),
    )

    path = simulate_path(model, horizon=5, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.claim_times, [3, 4])
    np.testing.assert_allclose(path.claim_sizes, [1, 1])


def test_sparre_andersen_keeps_claim_schedule_across_injections(monkeypatch):
    model = SparreAndersenProcess(
        initial_capital=0,
        premium_rate=0,
        interarrival_distribution=deterministic(1),
        claim_distribution=deterministic(2),
        capital_injections=(CapitalInjectionModel(rate=1, distribution=deterministic(5)),),
    )

    monkeypatch.setattr(
        simulation_module,
        "_sample_injections",
        lambda injections, horizon, rng: (np.array([0.5]), np.array([5.0])),
    )

    path = simulate_path(model, horizon=1.5, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.injection_times, [0.5])
    np.testing.assert_allclose(path.injection_sizes, [5])
    np.testing.assert_allclose(path.claim_times, [1])
    np.testing.assert_allclose(path.claim_sizes, [2])
    assert path.terminal_reserve == 3


def test_same_time_injection_is_applied_before_claim(monkeypatch):
    model = SparreAndersenProcess(
        initial_capital=0,
        premium_rate=0,
        interarrival_distribution=deterministic(1),
        claim_distribution=deterministic(3),
        capital_injections=(CapitalInjectionModel(rate=1, distribution=deterministic(5)),),
    )

    monkeypatch.setattr(
        simulation_module,
        "_sample_injections",
        lambda injections, horizon, rng: (np.array([1.0]), np.array([5.0])),
    )

    path = simulate_path(model, horizon=1.5, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.injection_times, [1])
    np.testing.assert_allclose(path.injection_sizes, [5])
    np.testing.assert_allclose(path.claim_times, [1])
    assert path.terminal_reserve == 2
    assert not path.ruined


def test_sparre_andersen_prevention_scales_frequency_and_severity():
    model = SparreAndersenProcess(
        initial_capital=10,
        premium_rate=0,
        interarrival_distribution=deterministic(1),
        claim_distribution=deterministic(8),
        prevention=PreventionProgram(frequency_multiplier=0.5, severity_multiplier=0.25),
    )
    path = simulate_path(model, horizon=4.1, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.claim_times, [2, 4])
    np.testing.assert_allclose(path.claim_sizes, [2, 2])
    assert path.terminal_reserve == 6


def test_by_claim_totals_are_included_in_path_claim_sizes():
    model = SparreAndersenProcess(
        initial_capital=10,
        premium_rate=0,
        interarrival_distribution=deterministic(1),
        claim_distribution=deterministic(2),
        by_claims=(FixedByClaim(3),),
    )
    path = simulate_path(model, horizon=1.5, seed=123, stop_at_ruin=False)
    np.testing.assert_allclose(path.claim_times, [1])
    np.testing.assert_allclose(path.claim_sizes, [5])
    assert path.terminal_reserve == 5


def test_prevention_reduces_effective_frequency_and_severity():
    prevention = PreventionProgram(frequency_multiplier=0.5, severity_multiplier=0.5)
    model = CramerLundbergProcess(
        premium_rate=1,
        claim_arrival_rate=2,
        claim_distribution=deterministic(2),
        prevention=prevention,
    )
    assert model.claim_arrival_rate == 1
    assert model.expected_claim_amount == 1


def test_by_claims_are_included_in_expected_amount():
    by_claim = ByClaimModel(probability=0.5, distribution=deterministic(2), count_mean=1)
    model = CramerLundbergProcess(
        premium_rate=10,
        claim_arrival_rate=1,
        claim_distribution=deterministic(3),
        by_claims=(by_claim,),
    )
    assert model.expected_claim_amount == 4


def test_monte_carlo_estimate_shape():
    model = CramerLundbergProcess(
        initial_capital=5,
        premium_rate=2,
        claim_arrival_rate=1,
        claim_distribution=exponential(rate=1),
    )
    estimate = estimate_ruin_probability(model, horizon=2, n_simulations=200, seed=7)
    assert 0.0 <= estimate.probability <= 1.0
    assert estimate.ruin_times.shape == (200,)


def test_monte_carlo_wilson_interval_is_informative_for_zero_ruin_samples():
    model = CramerLundbergProcess(
        initial_capital=5,
        premium_rate=1,
        claim_arrival_rate=10,
        claim_distribution=deterministic(1),
        prevention=PreventionProgram(frequency_multiplier=0),
    )

    estimate = estimate_ruin_probability(model, horizon=2, n_simulations=20, seed=7)
    normal = estimate_ruin_probability(
        model,
        horizon=2,
        n_simulations=20,
        ci_method="normal",
        seed=7,
    )

    assert estimate.ci_method == "wilson"
    assert estimate.probability == 0.0
    assert estimate.ci_low == 0.0
    assert estimate.ci_high > 0.0
    assert normal.ci_high == 0.0


def test_monte_carlo_rejects_unknown_ci_method():
    model = CramerLundbergProcess(claim_distribution=deterministic(1))

    with pytest.raises(ValueError, match="ci_method"):
        estimate_ruin_probability(model, horizon=1, n_simulations=5, ci_method="exact")
