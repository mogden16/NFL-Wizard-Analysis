"""Market calculations with a single home-spread convention."""

import math


def american_to_decimal(odds: int) -> float:
    """Convert a conventional American quote (absolute value at least 100)."""
    if abs(odds) < 100:
        raise ValueError("American odds must be at least +100 or at most -100")
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def decimal_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must exceed 1")
    return 1 / decimal_odds


def raw_implied_probability(american_odds: int) -> float:
    return decimal_implied_probability(american_to_decimal(american_odds))


def implied_team_points(total: float, home_spread: float) -> tuple[float, float]:
    """Return (home, away) points; negative home_spread means home favored."""
    if total < 0 or abs(home_spread) > total:
        raise ValueError("Total and spread imply negative team points")
    return ((total - home_spread) / 2, (total + home_spread) / 2)


def expected_value(probability: float, decimal_odds: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be within [0, 1]")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must exceed 1")
    return probability * decimal_odds - 1


def closing_line_value_probability(entry_american: int, close_american: int) -> float:
    """Positive when the selection's raw implied probability rises by close."""
    return raw_implied_probability(close_american) - raw_implied_probability(entry_american)


def poisson_at_least_one(expected_tds: float) -> float:
    if expected_tds < 0:
        raise ValueError("Expected touchdowns cannot be negative")
    return -math.expm1(-expected_tds)


def poisson_at_least_two(expected_tds: float) -> float:
    if expected_tds < 0:
        raise ValueError("Expected touchdowns cannot be negative")
    return 1 - math.exp(-expected_tds) * (1 + expected_tds)
