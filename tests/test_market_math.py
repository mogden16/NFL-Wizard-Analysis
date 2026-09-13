import math

import pytest

from nfl_td_model.market_math import (
    american_to_decimal,
    closing_line_value_probability,
    decimal_implied_probability,
    expected_value,
    implied_team_points,
    poisson_at_least_one,
    poisson_at_least_two,
    raw_implied_probability,
)


@pytest.mark.parametrize("american,decimal", [(140, 2.4), (-150, 5 / 3), (100, 2.0)])
def test_american_to_decimal(american: int, decimal: float) -> None:
    assert american_to_decimal(american) == pytest.approx(decimal)
    assert decimal_implied_probability(decimal) == pytest.approx(1 / decimal)
    assert raw_implied_probability(american) == pytest.approx(1 / decimal)


@pytest.mark.parametrize("invalid", [0, 50, -50])
def test_invalid_american_price(invalid: int) -> None:
    with pytest.raises(ValueError):
        american_to_decimal(invalid)


def test_implied_team_points_home_favorite_and_underdog() -> None:
    # home_spread is the handicap applied to home score; negative means favorite.
    assert implied_team_points(total=47, home_spread=-3) == pytest.approx((25, 22))
    assert implied_team_points(total=47, home_spread=3) == pytest.approx((22, 25))
    assert implied_team_points(total=44, home_spread=0) == pytest.approx((22, 22))


def test_expected_value_and_clv() -> None:
    assert expected_value(0.5, 2.4) == pytest.approx(0.2)
    # Positive CLV means the quoted position gained implied probability by close.
    assert closing_line_value_probability(140, 120) == pytest.approx(1 / 2.2 - 1 / 2.4)
    assert closing_line_value_probability(-150, -130) < 0


def test_poisson_probabilities() -> None:
    assert poisson_at_least_one(0) == 0
    assert poisson_at_least_two(0) == 0
    assert poisson_at_least_one(1) == pytest.approx(1 - math.exp(-1))
    assert poisson_at_least_two(1) == pytest.approx(1 - 2 * math.exp(-1))
    with pytest.raises(ValueError):
        poisson_at_least_one(-0.1)
