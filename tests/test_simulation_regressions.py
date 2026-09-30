"""Boundary cases found during the package-wide numerical review."""

import numpy as np
import pytest

from ruin_theory import (
    ByClaimModel,
    CramerLundbergProcess,
    PreventionProgram,
    deterministic,
    estimate_finite_time_ruin_sensitivity_ibp,
    exponential,
    finite_time_ruin_exponential,
    pareto,
)
from ruin_theory.results import SimulationPath


@pytest.mark.parametrize("probability,count_mean", [(0.0, 1.0), (1.0, 0.0)])
def test_disabled_infinite_mean_byclaims_have_zero_expectation(probability, count_mean):
    model = ByClaimModel(probability, pareto(0.5, 1.0), count_mean=count_mean)
    assert model.expected_amount_per_primary() == 0.0


def test_zero_severity_and_frequency_suppress_infinite_mean_risk():
    claims = pareto(0.5, 1.0)
    no_severity = CramerLundbergProcess(
        claim_distribution=claims, prevention=PreventionProgram(severity_multiplier=0.0),
    )
    no_frequency = CramerLundbergProcess(
        claim_distribution=claims, prevention=PreventionProgram(frequency_multiplier=0.0),
    )
    assert no_severity.expected_claim_amount == 0.0
    assert no_severity.claim_intensity == 0.0
    assert no_frequency.claim_intensity == 0.0
    assert no_frequency.safety_loading == np.inf


def test_ruin_claim_is_not_replaced_by_a_nearby_later_claim():
    path = SimulationPath(
        times=np.array([0.0, 1.0, 1.0, 1.000001, 1.000001]),
        reserves=np.array([1.0, 1.0, -1.0, -1.0, -8.0]),
        claim_times=np.array([1.0, 1.000001]),
        claim_sizes=np.array([2.0, 7.0]),
        ruin_time=1.0, horizon=1.000001, initial_capital=1.0, premium_rate=0.0,
    )
    assert path.claim_causing_ruin == 2.0
    assert path.surplus_before_ruin == 1.0
    assert path.deficit_at_ruin == 1.0


def test_ibp_conditional_density_is_stable_for_rare_claims():
    model = CramerLundbergProcess(
        claim_arrival_rate=1e-20, premium_rate=1.0, claim_distribution=deterministic(1.0),
    )
    result = estimate_finite_time_ruin_sensitivity_ibp(
        model, [0.5], 1.0, n_simulations=4, conditional_on_claim=True, seed=3,
    )
    np.testing.assert_allclose(result.density, [1.0])


def test_ibp_zero_frequency_honors_pathwise_output_shape():
    model = CramerLundbergProcess(prevention=PreventionProgram(frequency_multiplier=0.0))
    result = estimate_finite_time_ruin_sensitivity_ibp(
        model, [[0.1, 0.2], [0.3, 0.4]], 1.0, n_simulations=3,
        return_pathwise_density=True,
    )
    np.testing.assert_array_equal(result.pathwise_density, np.zeros((3, 4)))


@pytest.mark.parametrize("conditional", [False, True])
def test_ibp_streamed_statistics_match_retained_samples(conditional):
    model = CramerLundbergProcess(
        premium_rate=1.0, claim_arrival_rate=2.0, claim_distribution=deterministic(1.0),
    )
    # Unsorted and repeated reserves exercise the interval counting order.
    u = [[2.0, 0.0, 1.0], [0.5, 1.0, 10.0]]
    kwargs = dict(n_simulations=400, seed=41, conditional_on_claim=conditional)
    retained = estimate_finite_time_ruin_sensitivity_ibp(
        model, u, 2.0, return_pathwise_density=True, **kwargs,
    )
    streamed = estimate_finite_time_ruin_sensitivity_ibp(model, u, 2.0, **kwargs)
    assert streamed.pathwise_density.shape == (0, 0)
    np.testing.assert_allclose(streamed.density.ravel(), retained.pathwise_density.mean(axis=0))
    np.testing.assert_allclose(
        streamed.standard_error.ravel(), retained.pathwise_density.std(axis=0, ddof=1) / 20.0,
    )


@pytest.mark.parametrize("custom_income", [False, True])
def test_ibp_has_correct_premium_jacobian(custom_income):
    model = CramerLundbergProcess(
        premium_rate=2.0, claim_arrival_rate=0.5, claim_distribution=exponential(1.0),
    )
    u = np.array([0.5, 1.0])
    result = estimate_finite_time_ruin_sensitivity_ibp(
        model, u, 2.0, n_simulations=30000, seed=123,
        premium_income=(lambda t: 2.0 * t) if custom_income else None,
    )
    step = 1e-4
    expected = np.array([
        -(finite_time_ruin_exponential(model, u=x + step, horizon=2.0)
          - finite_time_ruin_exponential(model, u=x - step, horizon=2.0)) / (2 * step)
        for x in u
    ])
    np.testing.assert_allclose(result.density, expected, atol=4 * max(result.standard_error))


def test_ibp_rejects_nonpositive_drift_and_unvalidated_nonlinear_income():
    with pytest.raises(ValueError, match="positive"):
        estimate_finite_time_ruin_sensitivity_ibp(
            CramerLundbergProcess(premium_rate=0.0), [0.5], 1.0, n_simulations=5,
        )
    with pytest.raises(NotImplementedError, match="linear"):
        estimate_finite_time_ruin_sensitivity_ibp(
            CramerLundbergProcess(), [0.5], 1.0, n_simulations=5,
            premium_income=lambda t: t + 0.1 * t**2,
        )
