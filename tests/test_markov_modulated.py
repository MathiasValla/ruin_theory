import numpy as np
import pytest

from ruin_theory import (
    CommonShock,
    DependenceImpactResult,
    MarkovEnvironment,
    MarkovModulatedRuinResult,
    common_shock_increment_pmfs,
    compound_poisson_vector_pmf,
    convolve_vector_pmfs,
    dependence_impact,
    finite_time_markov_modulated_ruin,
    independent_common_shock_pmf,
    solvency_region,
    transition_matrix_from_generator,
)


def test_markov_environment_and_generator_validation():
    matrix = transition_matrix_from_generator([[-0.5, 0.5], [0.25, -0.25]], step=2.0)
    environment = MarkovEnvironment([0.75, 0.25], matrix)

    assert environment.n_states == 2
    np.testing.assert_allclose(np.sum(environment.transition_matrix, axis=1), [1.0, 1.0])

    with pytest.raises(ValueError, match="rows"):
        MarkovEnvironment([1.0, 0.0], [[0.5, 0.6], [0.2, 0.8]])


def test_independent_common_shock_pmf_and_convolution():
    pmf = independent_common_shock_pmf(
        [0.5, 1.0],
        [{1: 1.0}, {2: 1.0}],
    )

    assert pmf[(0, 2)] == pytest.approx(0.5)
    assert pmf[(1, 2)] == pytest.approx(0.5)
    assert sum(pmf.values()) == pytest.approx(1.0)

    convolved = convolve_vector_pmfs({(0, 0): 0.5, (1, 0): 0.5}, {(0, 1): 1.0})
    assert convolved[(0, 1)] == pytest.approx(0.5)
    assert convolved[(1, 1)] == pytest.approx(0.5)


def test_compound_poisson_and_common_shock_increment_pmfs():
    compound, tail = compound_poisson_vector_pmf({(1, 0): 1.0}, mean=0.2, max_count=6)
    shock = CommonShock(
        intensities=[0.2, 0.4],
        claim_pmfs=({(1, 0): 1.0}, {(0, 1): 1.0}),
        name="state-dependent",
    )
    replicated = CommonShock(intensities=[0.2, 0.4], claim_pmfs={(1, 0): 1.0})
    increments = common_shock_increment_pmfs([shock], max_count=6)

    assert compound[(0, 0)] == pytest.approx(np.exp(-0.2), rel=1e-5)
    assert tail < 1e-6
    assert replicated.n_states == 2
    assert replicated.claim_pmfs[0] == replicated.claim_pmfs[1]
    assert len(increments.increment_pmfs) == 2
    assert increments.increment_pmfs[0][(0, 0)] == pytest.approx(np.exp(-0.2), rel=1e-5)
    assert increments.increment_pmfs[1][(0, 0)] == pytest.approx(np.exp(-0.4), rel=1e-4)


def test_finite_time_markov_modulated_ruin_matches_one_period_manual_values():
    environment = MarkovEnvironment([1.0], [[1.0]])
    increment = [{(0, 0): 0.25, (2, 0): 0.25, (0, 2): 0.25, (2, 2): 0.25}]

    any_line = finite_time_markov_modulated_ruin(
        increment,
        environment,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="any_line",
    )
    total = finite_time_markov_modulated_ruin(
        increment,
        environment,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="total",
    )
    hybrid = finite_time_markov_modulated_ruin(
        increment,
        environment,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="hybrid",
        severity_limit=1.0,
    )

    assert isinstance(any_line, MarkovModulatedRuinResult)
    assert any_line.ruin_probabilities[1] == pytest.approx(0.75)
    assert total.ruin_probabilities[1] == pytest.approx(0.25)
    assert hybrid.ruin_probabilities[1] == pytest.approx(0.25)
    assert any_line.survival_by_state[1, 0] == pytest.approx(0.25)


def test_custom_region_and_markov_transition_keep_state_distribution():
    environment = MarkovEnvironment([1.0, 0.0], [[0.5, 0.5], [0.0, 1.0]])
    increments = [{(0,): 1.0}, {(1,): 1.0}]
    region = lambda claims, boundary, period: bool(claims[0] <= boundary[0] + period)

    result = finite_time_markov_modulated_ruin(
        increments,
        environment,
        initial_capitals=[0.0],
        premiums=[0.0],
        horizon=2,
        region=region,
    )

    assert result.region == "custom"
    assert result.ruin_probabilities[-1] == 0.0
    np.testing.assert_allclose(result.survival_by_state[1], [0.5, 0.5])


def test_positive_dependence_can_reduce_any_line_ruin_probability():
    environment = MarkovEnvironment([1.0], [[1.0]])
    independent = [{(2, 0): 0.25, (0, 2): 0.25, (2, 2): 0.25, (0, 0): 0.25}]
    comonotonic = [{(2, 2): 0.5, (0, 0): 0.5}]

    independent_result = finite_time_markov_modulated_ruin(
        independent,
        environment,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="any_line",
    )
    dependent_result = finite_time_markov_modulated_ruin(
        comonotonic,
        environment,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="any_line",
    )
    impact = dependence_impact(
        independent_result,
        dependent_result,
        reference_label="independent",
        comparison_label="positive dependence",
    )

    assert isinstance(impact, DependenceImpactResult)
    assert independent_result.ruin_probabilities[1] == pytest.approx(0.75)
    assert dependent_result.ruin_probabilities[1] == pytest.approx(0.5)
    assert impact.final_difference == pytest.approx(-0.25)


def test_solvency_region_validation_and_predicates():
    claims = np.array([2.0, 0.0])
    boundary = np.array([1.0, 1.0])

    assert not solvency_region("any_line")(claims, boundary, 1)
    assert solvency_region("total")(claims, boundary, 1)
    assert solvency_region("hybrid", severity_limit=1.0)(claims, boundary, 1)

    with pytest.raises(ValueError, match="kind"):
        solvency_region("bad")  # type: ignore[arg-type]


def test_markov_modulated_public_argument_validation():
    environment = MarkovEnvironment([1.0], [[1.0]])

    with pytest.raises(ValueError, match="off-diagonal"):
        transition_matrix_from_generator([[-1.0, -0.1], [0.0, 0.0]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        independent_common_shock_pmf([1.2], [{1: 1.0}])
    with pytest.raises(ValueError, match="environment state"):
        finite_time_markov_modulated_ruin(
            [{(0,): 1.0}, {(1,): 1.0}],
            environment,
            initial_capitals=[1.0],
            premiums=[0.0],
            horizon=1,
        )
    with pytest.raises(ValueError, match="dimension"):
        finite_time_markov_modulated_ruin(
            [{(0,): 1.0}],
            environment,
            initial_capitals=[1.0, 1.0],
            premiums=[0.0, 0.0],
            horizon=1,
        )
    with pytest.raises(TypeError, match="callable"):
        finite_time_markov_modulated_ruin(
            [{(0,): 1.0}],
            environment,
            initial_capitals=[1.0],
            premiums=[0.0],
            horizon=1,
            region=object(),  # type: ignore[arg-type]
        )
