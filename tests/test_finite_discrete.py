import math

import numpy as np
import pytest

from ruin_theory import (
    FiniteTimeDiscreteAppellResult,
    FiniteTimeDiscreteBoundaryGrid,
    FiniteTimeDiscreteBoundaryResult,
    FiniteTimeDiscreteRuinResult,
    compound_poisson_appell_base,
    compound_poisson_lattice_pmf,
    finite_time_discrete_appell_coefficients,
    finite_time_discrete_boundary_crossings,
    finite_time_ruin_discrete_appell,
    finite_time_ruin_discrete_boundary_function,
    finite_time_ruin_discrete_boundary,
    finite_time_ruin_discrete_inventory,
    finite_time_ruin_discrete,
)


def test_compound_poisson_lattice_pmf_matches_poisson_for_unit_claims():
    pmf = compound_poisson_lattice_pmf([0.0, 1.0], mean=2.0, max_aggregate=4)
    expected = np.exp(-2.0) * np.array([1.0, 2.0, 2.0, 4.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(pmf, expected)


def test_compound_poisson_appell_base_matches_unit_claim_polynomials():
    values = compound_poisson_appell_base(
        [0.0, 1.0],
        claim_arrival_rate=2.0,
        time=1.5,
        max_degree=4,
    )
    expected = np.array([1.0, 3.0, 4.5, 4.5, 3.375])

    np.testing.assert_allclose(values, expected)


def test_seal_formula_reproduces_de_vylder_deterministic_claim_table():
    claim_pmf = [0.0, 1.0]
    expected_ruin = {
        0: 0.765864440648,
        5: 0.039901595038,
        10: 0.000692886838,
        15: 0.000004740559,
        20: 0.000000014338,
    }

    for initial_capital, expected in expected_ruin.items():
        ruin = finite_time_ruin_discrete(
            claim_pmf,
            initial_capital=initial_capital,
            premium_rate=1.25,
            claim_arrival_rate=1.0,
            horizon=10.0,
            method="seal",
        )
        assert ruin == pytest.approx(expected, rel=5e-6, abs=1e-12)


def test_picard_lefevre_seal_and_inventory_methods_agree():
    claim_pmf = [0.0, 0.25, 0.5, 0.25]
    kwargs = dict(
        claim_pmf=claim_pmf,
        initial_capital=4,
        premium_rate=1.3,
        claim_arrival_rate=0.8,
        horizon=4.2,
    )

    seal = finite_time_ruin_discrete(**kwargs, method="seal")
    picard_lefevre = finite_time_ruin_discrete(**kwargs, method="picard-lefevre")
    inventory = finite_time_ruin_discrete(**kwargs, method="inventory", return_result=True)

    assert isinstance(inventory, FiniteTimeDiscreteRuinResult)
    assert inventory.inventory_times[-1] == pytest.approx(kwargs["horizon"])
    assert np.all(np.diff(inventory.survival_probabilities) <= 1e-14)
    assert seal == pytest.approx(inventory.ruin_probability, abs=2e-13)
    assert picard_lefevre == pytest.approx(inventory.ruin_probability, abs=2e-13)


def test_formulas_support_non_integer_initial_capital():
    # With u=0.5, c=1, lambda=1 and unit claims, survival to t=1 requires
    # N(0.5)=0 and at most one claim by t=1: exp(-1) * (1 + 0.5).
    expected_survival = 1.5 * math.exp(-1.0)
    kwargs = dict(
        claim_pmf=[0.0, 1.0],
        initial_capital=0.5,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=1.0,
    )

    seal = finite_time_ruin_discrete(**kwargs, method="seal")
    picard_lefevre = finite_time_ruin_discrete(**kwargs, method="picard-lefevre")
    inventory = finite_time_ruin_discrete(**kwargs, method="inventory", return_result=True)

    assert inventory.initial_capital == pytest.approx(0.5)
    assert inventory.survival_probability == pytest.approx(expected_survival)
    np.testing.assert_allclose(
        inventory.ruin_probabilities_by_time,
        1.0 - inventory.survival_probabilities,
    )
    assert seal == pytest.approx(1.0 - expected_survival)
    assert picard_lefevre == pytest.approx(1.0 - expected_survival)


def test_inventory_formula_matches_hand_check_for_size_two_claims():
    # With u=1, c=1 and deterministic claims of size 2, survival to t=2
    # requires no claim before time 1 and no second claim before time 2.
    expected_survival = 2.0 * math.exp(-2.0)
    result = finite_time_ruin_discrete(
        [0.0, 0.0, 1.0],
        initial_capital=1,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=2.0,
        method="inventory",
        return_result=True,
    )

    assert result.survival_probability == pytest.approx(expected_survival)
    assert result.ruin_probability == pytest.approx(1.0 - expected_survival)


def test_boundary_inventory_reproduces_linear_inventory_formula():
    claim_pmf = [0.0, 0.25, 0.5, 0.25]
    linear = finite_time_ruin_discrete(
        claim_pmf,
        initial_capital=4,
        premium_rate=1.3,
        claim_arrival_rate=0.8,
        horizon=4.2,
        method="inventory",
        return_result=True,
    )
    boundary = finite_time_ruin_discrete_boundary(
        claim_pmf,
        inventory_times=linear.inventory_times,
        boundary_values=4.0 + 1.3 * linear.inventory_times,
        claim_arrival_rate=0.8,
        convention="negative",
        boundary_kind="crossing",
        return_result=True,
    )

    assert isinstance(boundary, FiniteTimeDiscreteBoundaryResult)
    assert boundary.ruin_probability == pytest.approx(linear.ruin_probability)
    np.testing.assert_allclose(boundary.survival_probabilities, linear.survival_probabilities)


def test_boundary_crossing_grid_matches_linear_inverse_dates():
    grid = finite_time_discrete_boundary_crossings(lambda time: 4.0 + 1.3 * time, horizon=4.2)

    assert isinstance(grid, FiniteTimeDiscreteBoundaryGrid)
    np.testing.assert_allclose(
        grid.inventory_times,
        [1 / 1.3, 2 / 1.3, 3 / 1.3, 4 / 1.3, 5 / 1.3, 4.2],
    )
    np.testing.assert_allclose(grid.boundary_values, [5.0, 6.0, 7.0, 8.0, 9.0, 9.46])


def test_boundary_function_reproduces_linear_inventory_formula():
    claim_pmf = [0.0, 0.25, 0.5, 0.25]
    linear = finite_time_ruin_discrete(
        claim_pmf,
        initial_capital=4,
        premium_rate=1.3,
        claim_arrival_rate=0.8,
        horizon=4.2,
        method="inventory",
        return_result=True,
    )
    boundary = finite_time_ruin_discrete_boundary_function(
        claim_pmf,
        boundary=lambda time: 4.0 + 1.3 * time,
        horizon=4.2,
        claim_arrival_rate=0.8,
        return_result=True,
    )

    assert boundary.ruin_probability == pytest.approx(linear.ruin_probability)
    np.testing.assert_allclose(boundary.survival_probabilities, linear.survival_probabilities)


def test_generalized_appell_reproduces_boundary_inventory_formula():
    claim_pmf = [0.0, 0.25, 0.5, 0.25]
    boundary = lambda time: 4.0 + 1.3 * time
    inventory = finite_time_ruin_discrete_boundary_function(
        claim_pmf,
        boundary=boundary,
        horizon=4.2,
        claim_arrival_rate=0.8,
        return_result=True,
    )
    appell = finite_time_ruin_discrete_appell(
        claim_pmf,
        boundary=boundary,
        horizon=4.2,
        claim_arrival_rate=0.8,
        return_result=True,
    )

    assert isinstance(appell, FiniteTimeDiscreteAppellResult)
    assert appell.ruin_probability == pytest.approx(inventory.ruin_probability, abs=2e-9)
    assert appell.appell_coefficients[0] == pytest.approx(1.0)
    assert appell.state_probabilities.shape == (appell.appell_coefficients.size,)


def test_generalized_appell_coefficients_are_available_separately():
    coefficients = finite_time_discrete_appell_coefficients(
        [0.0, 1.0],
        claim_arrival_rate=1.0,
        boundary=lambda time: 1.0 + time,
        horizon=2.5,
    )

    assert coefficients[0] == pytest.approx(1.0)
    assert coefficients[1] == pytest.approx(0.0)
    assert coefficients.size == 4


def test_generalized_appell_handles_zero_claim_mass_by_thinning():
    with_zero_claims = finite_time_ruin_discrete_appell(
        [0.5, 0.5],
        boundary=lambda time: 0.5 + time,
        horizon=1.0,
        claim_arrival_rate=2.0,
    )
    thinned = finite_time_ruin_discrete_appell(
        [0.0, 1.0],
        boundary=lambda time: 0.5 + time,
        horizon=1.0,
        claim_arrival_rate=1.0,
    )

    assert with_zero_claims == pytest.approx(thinned)


def test_boundary_conventions_handle_ruin_at_zero():
    # Constant boundary h(t)=1 over [0,1] with unit claims. For negative ruin,
    # zero reserve is allowed, so at most one claim can occur. For non-positive
    # ruin, zero reserve is ruin, so no claim can occur.
    negative = finite_time_ruin_discrete_boundary(
        [0.0, 1.0],
        inventory_times=[1.0],
        boundary_values=[1.0],
        claim_arrival_rate=1.0,
        convention="negative",
    )
    nonpositive = finite_time_ruin_discrete_boundary(
        [0.0, 1.0],
        inventory_times=[1.0],
        boundary_values=[1.0],
        claim_arrival_rate=1.0,
        convention="nonpositive",
    )

    assert negative == pytest.approx(1.0 - 2.0 * math.exp(-1.0))
    assert nonpositive == pytest.approx(1.0 - math.exp(-1.0))


def test_boundary_function_accepts_nonhomogeneous_poisson_mean():
    boundary = lambda time: 0.6 + time
    result = finite_time_ruin_discrete_boundary_function(
        [0.0, 1.0],
        boundary=boundary,
        horizon=1.0,
        cumulative_arrival_mean=lambda time: time * time,
        return_result=True,
    )
    manual = finite_time_ruin_discrete_inventory(
        [0.0, 1.0],
        inventory_times=[0.4, 1.0],
        retained_counts=[1, 2],
        arrival_means=[0.16, 0.84],
        return_result=True,
    )

    assert result.claim_arrival_rate is None
    assert result.ruin_probability == pytest.approx(manual.ruin_probability)
    np.testing.assert_allclose(result.arrival_means, [0.16, 0.84], atol=1e-10)


def test_boundary_function_handles_nonpositive_initial_ruin():
    result = finite_time_ruin_discrete_boundary_function(
        [0.0, 1.0],
        boundary=lambda time: time,
        horizon=1.0,
        claim_arrival_rate=1.0,
        convention="nonpositive",
        return_result=True,
    )

    assert result.ruin_probability == 1.0
    assert result.survival_probability == 0.0
    assert result.inventory_times.size == 0


def test_inventory_recursion_accepts_interval_arrival_means():
    result = finite_time_ruin_discrete_inventory(
        [0.0, 1.0],
        inventory_times=[0.4, 1.0],
        retained_counts=[1, 2],
        arrival_means=[0.4, 0.6],
        return_result=True,
    )
    linear = finite_time_ruin_discrete(
        [0.0, 1.0],
        initial_capital=0.6,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=1.0,
        method="inventory",
        return_result=True,
    )

    assert result.claim_arrival_rate is None
    assert result.survival_probability == pytest.approx(linear.survival_probability)
    np.testing.assert_allclose(result.arrival_means, [0.4, 0.6])


def test_takacs_zero_initial_capital_formula_is_available():
    ruin = finite_time_ruin_discrete(
        [0.0, 0.0, 1.0],
        initial_capital=0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=2.0,
        method="takacs",
    )

    assert ruin == pytest.approx(1.0 - math.exp(-2.0))


def test_finite_discrete_formulas_validate_inputs():
    with pytest.raises(ValueError, match="sum to one"):
        compound_poisson_lattice_pmf([0.2, 0.2], mean=1.0, max_aggregate=3)
    with pytest.raises(TypeError, match="initial_capital"):
        finite_time_ruin_discrete(
            [0.0, 1.0],
            initial_capital="bad",
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            horizon=1.0,
        )
    with pytest.raises(ValueError, match="premium_rate"):
        finite_time_ruin_discrete(
            [0.0, 1.0],
            initial_capital=1,
            premium_rate=0.0,
            claim_arrival_rate=1.0,
            horizon=1.0,
        )
    with pytest.raises(ValueError, match="method"):
        finite_time_ruin_discrete(
            [0.0, 1.0],
            initial_capital=1,
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            horizon=1.0,
            method="appell",
        )
    with pytest.raises(ValueError, match="zero-initial-capital"):
        finite_time_ruin_discrete(
            [0.0, 1.0],
            initial_capital=1,
            premium_rate=1.0,
            claim_arrival_rate=1.0,
            horizon=1.0,
            method="takacs",
        )
    with pytest.raises(ValueError, match="exactly one"):
        finite_time_ruin_discrete_inventory(
            [0.0, 1.0],
            inventory_times=[1.0],
            retained_counts=[1],
            claim_arrival_rate=1.0,
            arrival_means=[1.0],
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        finite_time_ruin_discrete_boundary(
            [0.0, 1.0],
            inventory_times=[1.0, 2.0],
            boundary_values=[2.0, 1.0],
            claim_arrival_rate=1.0,
        )
    with pytest.raises(ValueError, match="match inventory_times"):
        finite_time_ruin_discrete_inventory(
            [0.0, 1.0],
            inventory_times=[1.0],
            retained_counts=[1, 2],
            claim_arrival_rate=1.0,
        )
    with pytest.raises(TypeError, match="boundary"):
        finite_time_discrete_boundary_crossings(3.0, horizon=1.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        finite_time_discrete_boundary_crossings(lambda time: 2.0 - time, horizon=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        finite_time_ruin_discrete_boundary_function(
            [0.0, 1.0],
            boundary=lambda time: time + 1.0,
            horizon=1.0,
            claim_arrival_rate=1.0,
            cumulative_arrival_mean=lambda time: time,
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        finite_time_ruin_discrete_boundary_function(
            [0.0, 1.0],
            boundary=lambda time: time + 1.0,
            horizon=1.0,
            cumulative_arrival_mean=lambda time: 1.0 - time,
        )
    with pytest.raises(ValueError, match="non-negative"):
        compound_poisson_appell_base(
            [0.0, 1.0],
            claim_arrival_rate=1.0,
            time=-1.0,
            max_degree=2,
        )
