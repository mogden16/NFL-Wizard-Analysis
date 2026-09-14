"""Focused price-only scanner arithmetic and quote-quality tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nfl_td_model.atd_price_scan import decimal_to_american, summarize_player
from nfl_td_model.market_math import american_to_decimal, raw_implied_probability

NOW = datetime(2026, 9, 14, 17, tzinfo=UTC)


def quote(book: str, price: int, minute: int = 59) -> dict[str, object]:
    return {"sportsbook": book, "price": price,
            "quote_time": f"2026-09-14T16:{minute:02d}:00+00:00"}


def test_american_conversion_both_signs() -> None:
    assert american_to_decimal(200) == 3
    assert american_to_decimal(-200) == 1.5
    assert raw_implied_probability(200) == pytest.approx(1 / 3)
    assert raw_implied_probability(-200) == pytest.approx(2 / 3)
    assert decimal_to_american(3) == 200
    assert decimal_to_american(1.5) == -200


def test_median_uses_decimal_payoff_across_american_signs() -> None:
    rows = [quote("a", -110), quote("b", 100), quote("c", 120), quote("d", 130)]
    result = summarize_player("Runner", rows, NOW, "KC", "DEN")
    assert result["Median Market Odds"] == pytest.approx(110)
    assert result["median_decimal"] == pytest.approx(2.1)
    assert result["Median Market Implied Probability"] == pytest.approx(1 / 2.1)
    assert result["Best Odds"] == 130
    assert result["Probability-Price Difference"] == pytest.approx(1 / 2.1 - 1 / 2.3)


def test_fewer_than_three_books_has_no_consensus_comparison() -> None:
    result = summarize_player("Runner", [quote("a", 100), quote("b", 120)], NOW, "KC", "DEN")
    assert result["Books Quoting"] == 2
    assert result["Median Market Odds"] is None
    assert result["Median Market Implied Probability"] is None
    assert result["Probability-Price Difference"] is None
    assert result["Status"] == "INSUFFICIENT_MARKET"


def test_duplicate_book_does_not_satisfy_minimum() -> None:
    result = summarize_player("Runner", [quote("a", 100), quote("a", 120), quote("b", 110)],
                              NOW, "KC", "DEN")
    assert result["Books Quoting"] == 2
    assert result["Best Odds"] == 120
    assert result["Status"] == "INSUFFICIENT_MARKET"


def test_stale_uses_oldest_contributing_market_update() -> None:
    rows = [quote("a", 100, 44), quote("b", 110, 59), quote("c", 120, 59)]
    result = summarize_player("Runner", rows, NOW, "KC", "DEN")
    assert result["Quote Age"] == 16
    assert "STALE" in result["Status"]
    fresh = summarize_player("Runner", [quote("a", 100, 45), quote("b", 110, 59),
                                          quote("c", 120, 59)], NOW, "KC", "DEN")
    assert "STALE" not in fresh["Status"]


def test_large_best_price_deviation_flagged_not_removed() -> None:
    result = summarize_player("Runner", [quote("a", 100), quote("b", 110),
                                         quote("c", 300)], NOW, "KC", "DEN")
    assert result["Best Odds"] == 300
    assert result["Books Quoting"] == 3
    assert result["Probability-Price Difference"] > 0
    assert "PRICE_OUTLIER" in result["Status"]


def test_future_quote_is_rejected() -> None:
    with pytest.raises(ValueError, match="Future market update"):
        summarize_player("Runner", [{"sportsbook": "a", "price": 100,
                                     "quote_time": "2026-09-14T17:01:00+00:00"}], NOW)
