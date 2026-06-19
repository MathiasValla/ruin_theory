import math

import numpy as np
import pytest

from ruin_theory import (
    FiniteTimeDiscreteRuinResult,
    compound_poisson_lattice_pmf,
    finite_time_ruin_discrete,
)


def test_compound_poisson_lattice_pmf_matches_poisson_for_unit_claims():
    pmf = compound_poisson_lattice_pmf([0.0, 1.0], mean=2.0, max_aggregate=4)
    expected = np.exp(-2.0) * np.array([1.0, 2.0, 2.0, 4.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(pmf, expected)


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
