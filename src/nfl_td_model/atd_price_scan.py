"""Current-slate cross-book ATD Yes-price scanner; no football model or results."""

from __future__ import annotations

import csv
import logging
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import nflreadpy  # type: ignore[import-untyped]

from nfl_td_model.config import Settings
from nfl_td_model.market_math import american_to_decimal, decimal_implied_probability
from nfl_td_model.odds import parse_time
from nfl_td_model.phase46_predict import normalize_name

LIVE_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
EASTERN = ZoneInfo("America/New_York")
MIN_BOOKS = 3
STALE_MINUTES = 15
# Reuse the Phase 4.6 best-decimal / median-decimal outlier convention.
OUTLIER_DECIMAL_DEVIATION = 0.25
COLUMNS = (
    "Player", "Team", "Opponent", "Best Book", "Best Odds", "Median Market Odds",
    "Books Quoting", "Best Implied Probability", "Median Market Implied Probability",
    "Probability-Price Difference", "Quote Age", "Status",
)


def decimal_to_american(decimal_odds: float) -> float:
    """Equivalent American quote; may be fractional for an even-book median."""
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must exceed 1")
    return 100 * (decimal_odds - 1) if decimal_odds >= 2 else -100 / (decimal_odds - 1)


def summarize_player(
    player: str, quotes: list[dict[str, Any]], captured_at: datetime,
    team: str | None = None, opponent: str | None = None,
) -> dict[str, Any]:
    """Compare distinct books from one event response, preserving quote ages."""
    if not quotes:
        raise ValueError("A quoted player needs at least one Yes price")
    # A provider may repeat a book; its newest market update is the one quote.
    by_book: dict[str, dict[str, Any]] = {}
    for quote in quotes:
        book = str(quote["sportsbook"])
        updated = parse_time(str(quote["quote_time"]))
        if updated > captured_at:
            raise ValueError("Future market update entered the live scanner")
        if (book not in by_book or updated > parse_time(str(by_book[book]["quote_time"]))
                or (updated == parse_time(str(by_book[book]["quote_time"])) and
                    american_to_decimal(int(quote["price"])) >
                    american_to_decimal(int(by_book[book]["price"])))):
            by_book[book] = quote
    offers = list(by_book.values())
    best = min(offers, key=lambda q: (-american_to_decimal(int(q["price"])),
                                      str(q["sportsbook"])))
    best_decimal = american_to_decimal(int(best["price"]))
    # Median the payoffs, then convert; averaging American numbers can create
    # invalid odds when an even-sized market straddles -100 and +100.
    median_decimal = statistics.median(american_to_decimal(int(q["price"])) for q in offers)
    best_probability = decimal_implied_probability(best_decimal)
    enough_books = len(offers) >= MIN_BOOKS
    median_probability = decimal_implied_probability(median_decimal) if enough_books else None
    quote_age = max((captured_at - parse_time(str(q["quote_time"]))).total_seconds() / 60
                    for q in offers)
    flags = []
    if not enough_books:
        flags.append("INSUFFICIENT_MARKET")
    if quote_age > STALE_MINUTES:
        flags.append("STALE")
    if enough_books and best_decimal / median_decimal - 1 >= OUTLIER_DECIMAL_DEVIATION:
        flags.append("PRICE_OUTLIER")
    if team is None or opponent is None:
        flags.append("TEAM_UNKNOWN")
    return {
        "Player": player, "Team": team, "Opponent": opponent,
        "Best Book": best["sportsbook"], "Best Odds": int(best["price"]),
        "Median Market Odds": decimal_to_american(median_decimal) if enough_books else None,
        "Books Quoting": len(offers), "Best Implied Probability": best_probability,
        "Median Market Implied Probability": median_probability,
        "Probability-Price Difference": median_probability - best_probability
        if median_probability is not None else None,
        "Quote Age": quote_age, "Status": "|".join(flags) if flags else "OK",
        "median_decimal": median_decimal if enough_books else None,
        "best_quote_time": best["quote_time"],
        "best_vs_median_decimal_deviation": best_decimal / median_decimal - 1 if enough_books else None,
    }


def _roster_lookup(season: int) -> dict[str, set[str]]:
    """Current-season names are team labels only, never eligibility evidence."""
    if season == 2025:
        raise ValueError("The 2025 holdout is sealed")
    rosters = nflreadpy.load_rosters([season])
    latest_week = rosters["week"].max()
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in rosters.filter(rosters["week"] == latest_week).select("full_name", "team").iter_rows(named=True):
        if row["full_name"] and row["team"]:
            lookup[normalize_name(row["full_name"])].add(row["team"])
    return lookup


def scan_current_slate(settings: Settings, now: datetime | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch each today's NFL event exactly once with only the ATD market."""
    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required for the live ATD scanner")
    scan_time = now or datetime.now(UTC)
    local = scan_time.astimezone(EASTERN)
    season = local.year - 1 if local.month <= 3 else local.year
    if season == 2025:
        raise ValueError("The 2025 holdout is sealed")
    team_names = {r["team_name"]: r["team_abbr"] for r in nflreadpy.load_teams().to_dicts()}
    roster = _roster_lookup(season)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{LIVE_BASE}/events", params={"apiKey": settings.odds_api_key})
        response.raise_for_status()
        events = [e for e in response.json() if parse_time(e["commence_time"]).astimezone(
            EASTERN).date() == local.date()]
        summaries: list[dict[str, Any]] = []
        all_quotes: list[dict[str, Any]] = []
        for event in events:
            response = client.get(
                f"{LIVE_BASE}/events/{event['id']}/odds",
                params={"apiKey": settings.odds_api_key, "regions": settings.odds_regions,
                        "markets": "player_anytime_td", "oddsFormat": "american"},
            )
            response.raise_for_status()
            captured = datetime.now(UTC)
            payload = response.json()
            if payload["id"] != event["id"]:
                raise ValueError("ATD odds response event ID changed")
            home = team_names.get(event["home_team"])
            away = team_names.get(event["away_team"])
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            display: dict[str, str] = {}
            for book in payload.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "player_anytime_td":
                        continue
                    updated = parse_time(market["last_update"])
                    if updated > captured:
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") != "Yes" or not outcome.get("description"):
                            continue
                        name = str(outcome["description"])
                        key = normalize_name(name)
                        quote = {"event_id": event["id"], "player": name,
                                 "sportsbook": book["key"], "price": int(outcome["price"]),
                                 "quote_time": updated.isoformat(), "captured_at": captured.isoformat()}
                        grouped[key].append(quote)
                        display[key] = name
                        all_quotes.append(quote)
            for key, quotes in grouped.items():
                candidate_teams = roster.get(key, set()).intersection({home, away})
                team = next(iter(candidate_teams)) if len(candidate_teams) == 1 else None
                opponent = away if team == home else home if team == away else None
                summaries.append(summarize_player(display[key], quotes, captured, team, opponent))
    summaries.sort(key=lambda r: (r["Probability-Price Difference"] is None,
                                  -(r["Probability-Price Difference"] or 0), r["Player"]))
    return summaries, all_quotes


def write_report(rows: list[dict[str, Any]], quotes: list[dict[str, Any]],
                 output_dir: Path, day: str) -> tuple[Path, Path, Path]:
    """Write a sortable CSV and concise labeled Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f"atd_price_opportunities_{day}.md"
    summary_path = output_dir / f"atd_price_opportunities_{day}.csv"
    quotes_path = output_dir / f"atd_price_quotes_{day}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row[column] for column in COLUMNS} for row in rows)
    with quotes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "event_id", "player", "sportsbook", "price", "quote_time", "captured_at"))
        writer.writeheader()
        writer.writerows(quotes)
    lines = ["# CROSS-BOOK ATD PRICE OPPORTUNITIES", "",
             "Raw one-sided price comparison only. These are not bets or positive-EV claims.", "",
             f"Slate date (America/New_York): {day}. Players: {len(rows)}.", "",
             "| " + " | ".join(COLUMNS) + " |",
             "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        values: list[str] = []
        for column in COLUMNS:
            value = row[column]
            if value is None:
                rendered = "—"
            elif column in {"Best Implied Probability", "Median Market Implied Probability",
                            "Probability-Price Difference"}:
                rendered = f"{float(value) * 100:.2f}%"
            elif column == "Quote Age":
                rendered = f"{float(value):.1f} min"
            elif column in {"Best Odds", "Median Market Odds"}:
                rendered = f"{float(value):+.0f}"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Consensus requires at least three distinct sportsbooks. Quote age is the",
              "oldest contributing market update; over 15 minutes is STALE. PRICE_OUTLIER",
              "uses the existing 25% best-decimal versus median-decimal deviation flag.",
              "Team labels come from the current-season nflverse roster and do not confirm",
              "active status. Median odds are derived from median decimal payoff.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report, summary_path, quotes_path
