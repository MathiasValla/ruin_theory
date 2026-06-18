"""Numerical reference checks from the local actuar and ruin references."""

import numpy as np
import pytest

from examples.r_actuar_package_python import compute_actuar_tables
from ruin_theory import (
    CramerLundbergProcess,
    adjustment_coefficient,
    de_vylder_approximation,
    exponential,
    ultimate_ruin_exponential,
)


def test_actuar_adjustment_coefficient_and_reinsurance_values():
    tables = compute_actuar_tables()

    assert tables.adjustment == pytest.approx(0.1667, abs=5e-5)
    np.testing.assert_allclose(
        tables.retention_coefficients,
        [0.1905, 0.1862, 0.1765, 0.1667],
        atol=5e-5,
    )


def test_actuar_exponential_and_hyperexponential_ruin_values():
    tables = compute_actuar_tables()

    np.testing.assert_allclose(
        tables.exponential_ruin,
        [
            6.000e-01,
            8.120e-02,
            1.099e-02,
            1.487e-03,
            2.013e-04,
            2.724e-05,
            3.687e-06,
            4.989e-07,
            6.752e-08,
            9.138e-09,
            1.237e-09,
        ],
        rtol=5e-4,
        atol=5e-13,
    )
    expected_hyper = (
        24.0 * np.exp(-tables.surplus_grid) + np.exp(-6.0 * tables.surplus_grid)
    ) / 35.0
    np.testing.assert_allclose(tables.hyperexponential_ruin, expected_hyper, rtol=2e-8)


def test_actuar_beekman_panjer_pareto_bounds():
    tables = compute_actuar_tables()

    np.testing.assert_array_equal(tables.beekman_grid, np.arange(0, 55, 5))
    np.testing.assert_allclose(
        tables.beekman_lower,
        [
            0.6719160,
            0.2892792,
            0.1361541,
            0.0662486,
            0.0329848,
            0.0167551,
            0.0086802,
            0.0045911,
            0.0024843,
            0.0013790,
            0.0007877,
        ],
        atol=5e-8,
    )
    np.testing.assert_allclose(
        tables.beekman_upper,
        [
            0.83333,
            0.51572,
            0.32938,
            0.21200,
            0.13700,
            0.08877,
            0.05764,
            0.03749,
            0.02443,
            0.01595,
            0.01043,
        ],
        atol=5e-6,
    )


def test_de_vylder_reduces_to_exact_exponential_case():
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    u = np.array([0.0, 1.0, 2.0])

    assert adjustment_coefficient(model) == pytest.approx(2.0)
    np.testing.assert_allclose(
        de_vylder_approximation(model, u),
        ultimate_ruin_exponential(model, u),
        rtol=1e-12,
        atol=1e-15,
    )
