import math

import numpy as np
import pytest

from ruin_theory import (
    PhaseTypeRenewalCountResult,
    phase_type,
    phase_type_convolution,
    phase_type_renewal_count_pmf,
    sparre_andersen_phase_type_ruin_probability_by_count,
)


def test_phase_type_convolution_matches_erlang_two_waits():
    wait = phase_type([1.0], [[-2.0]])
    two_waits = phase_type_convolution(wait, 2)
    x = np.array([0.0, 0.5, 1.0, 2.0])

    np.testing.assert_allclose(two_waits.survival(x), np.exp(-2.0 * x) * (1.0 + 2.0 * x))
    assert two_waits.mean() == pytest.approx(1.0)
    assert two_waits.variance() == pytest.approx(0.5)


def test_phase_type_renewal_count_pmf_reproduces_poisson_counts():
    rate = 2.0
    horizon = 1.5
    max_count = 8
    wait = phase_type([1.0], [[-rate]])

    result = phase_type_renewal_count_pmf(wait, horizon, max_count=max_count)

    assert isinstance(result, PhaseTypeRenewalCountResult)
    mean = rate * horizon
    expected = np.array(
        [math.exp(-mean) * mean**count / math.factorial(count) for count in range(max_count + 1)]
    )
    np.testing.assert_allclose(result.probabilities, expected, rtol=1e-11, atol=1e-14)
    assert result.tail_probability == pytest.approx(1.0 - expected.sum())
    assert result.total_mass == pytest.approx(1.0)


def test_sparre_andersen_phase_type_ruin_probability_mixes_count_law():
    wait = phase_type([1.0], [[-1.0]])
    ruin_by_count = np.array([0.0, 0.1, 0.4])
    count_law = phase_type_renewal_count_pmf(wait, 1.0, max_count=2)

    value = sparre_andersen_phase_type_ruin_probability_by_count(ruin_by_count, wait, 1.0)
    expected = float(np.dot(count_law.probabilities, ruin_by_count))
    expected += count_law.tail_probability * ruin_by_count[-1]

    assert value == pytest.approx(expected)


def test_phase_type_renewal_helpers_validate_arguments():
    wait = phase_type([1.0], [[-1.0]])

    with pytest.raises(ValueError, match="positive integer"):
        phase_type_convolution(wait, 0)
    with pytest.raises(ValueError, match="horizon"):
        phase_type_renewal_count_pmf(wait, 0.0, max_count=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sparre_andersen_phase_type_ruin_probability_by_count([0.0, 1.2], wait, 1.0)
