import numpy as np

from nfl_td_model.usage_prop_calibration import (
    distribution_probabilities,
    negative_binomial_alpha,
)


def test_negative_binomial_detects_overdispersion():
    alpha = negative_binomial_alpha(np.array([0, 1, 8, 2, 10]), np.full(5, 3.0))
    assert alpha > 0


def test_negative_binomial_probabilities_sum_to_one():
    over, under, push = distribution_probabilities("negative_binomial", 5.0, 5.0, alpha=0.5)
    assert abs(over + under + push - 1.0) < 1e-9
    assert 0 <= over <= 1 and 0 <= under <= 1 and 0 <= push <= 1


def test_empirical_residual_probabilities_sum_to_one():
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    probabilities = distribution_probabilities("empirical_residual", 4.0, 3.5, residuals=residuals)
    assert abs(sum(probabilities) - 1.0) < 1e-9

