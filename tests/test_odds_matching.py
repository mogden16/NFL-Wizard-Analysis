import pytest

from nfl_td_model.phase1 import normalize_name, player_quote_candidates


def quote(description: str, sportsbook: str = "book") -> dict[str, str]:
    return {
        "market": "player_anytime_td",
        "name": "Yes",
        "description": description,
        "sportsbook": sportsbook,
    }


@pytest.mark.parametrize(
    "source,market",
    [
        ("Travis Etienne", "Travis Etienne Jr."),
        ("Michael Pittman", "Michael Pittman Jr."),
        ("Deebo Samuel Sr.", "Deebo Samuel"),
    ],
)
def test_terminal_suffix_alias(source: str, market: str) -> None:
    assert normalize_name(source) == normalize_name(market)
    assert player_quote_candidates([quote(market)], source) == [quote(market)]


def test_ambiguous_same_book_alias_rejected() -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        player_quote_candidates([quote("John Smith Jr."), quote("John Smith Sr.")], "John Smith")
