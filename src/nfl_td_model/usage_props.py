"""Small receptions/rushing-attempt usage-prop MVP built on frozen features."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import nflreadpy  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import PoissonRegressor  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from nfl_td_model.atd_price_scan import decimal_to_american
from nfl_td_model.config import Settings
from nfl_td_model.market_math import american_to_decimal, decimal_implied_probability
from nfl_td_model.odds import parse_time
from nfl_td_model.phase46_predict import normalize_name

LIVE_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
PROP_MARKETS = {"receptions": "player_receptions", "rushing_attempts": "player_rush_attempts"}
POSITIONS = {"RB", "WR", "TE", "QB"}
FEATURES = {
    "receptions": ("ewma_receptions_per_game", "last5_receptions_per_game",
                   "ewma_target_share", "last5_target_share", "ewma_snap_share",
                   "last5_team_targets_per_game", "smoothed_catch_rate"),
    "rushing_attempts": ("ewma_carries_per_game", "last5_carries_per_game",
                         "ewma_carry_share", "last5_carry_share", "ewma_snap_share",
                         "last5_team_carries_per_game"),
}


def target_column(prop: str) -> str:
    """Return the frozen player-stat target for one supported prop."""
    if prop == "receptions":
        return "receptions"
    if prop == "rushing_attempts":
        return "carries"
    raise ValueError(f"Unsupported usage prop: {prop}")


def count_distribution(mean: float, max_count: int = 40) -> dict[int | str, float]:
    """Poisson count probabilities with a 4+ tail bucket."""
    if mean < 0:
        raise ValueError("Count mean must be nonnegative")
    probabilities: dict[int | str, float] = {
        k: math.exp(-mean) * mean**k / math.factorial(k) for k in range(4)
    }
    probabilities["4+"] = max(0.0, 1 - sum(probabilities.values()))
    return probabilities


def over_under_push(mean: float, line: float) -> tuple[float, float, float]:
    """Return (over, under, push) from a Poisson count distribution."""
    if line < 0:
        raise ValueError("Prop line cannot be negative")
    max_count = max(80, math.ceil(line) + 10)
    probs = {k: math.exp(-mean) * mean**k / math.factorial(k) for k in range(max_count)}
    tail = max(0.0, 1 - sum(probs.values()))
    if line.is_integer():
        push = probs.get(int(line), 0.0)
        over = sum(value for k, value in probs.items() if k > line) + tail
        under = sum(value for k, value in probs.items() if k < line)
    else:
        push = 0.0
        over = sum(value for k, value in probs.items() if k > line) + tail
        under = 1 - over
    total = over + under + push
    return over / total, under / total, push / total


def two_sided_de_vig(over_american: int, under_american: int) -> tuple[float, float, float]:
    """Return raw over, raw under, and proportional no-vig over probability."""
    over_raw = decimal_implied_probability(american_to_decimal(over_american))
    under_raw = decimal_implied_probability(american_to_decimal(under_american))
    total = over_raw + under_raw
    return over_raw, under_raw, over_raw / total


def _season_stats(season: int) -> pl.DataFrame:
    if not 2017 <= season <= 2024:
        raise ValueError("Usage-prop history is restricted to 2017-2024; 2025 is sealed")
    hashes = json_load(Path("reports/phase2_historical_source_hashes.json"))[str(season)]
    path = Path("data/raw/nflverse") / f"player-stats-{season}-{hashes['player_stats']}.parquet"
    return pl.read_parquet(path, columns=["game_id", "player_id", "player_display_name", "position",
                                          "team", "carries", "targets", "receptions"]).rename(
                                              {"game_id": "game"}
                                          )


def json_load(path: Path) -> dict[str, Any]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def load_historical_prop_frame() -> pl.DataFrame:
    """Join frozen lagged features to season-specific player-stat labels."""
    frames = []
    for season in range(2017, 2025):
        features = pl.read_parquet(f"data/derived/phase2_{season}_player_features.parquet")
        stats = _season_stats(season).rename({"player_display_name": "label_player"})
        joined = features.join(stats, on=["game", "player_id"], how="left")
        joined = joined.filter(pl.col("position").is_in(list(POSITIONS)))
        frames.append(joined.with_columns(pl.lit(season).alias("season")))
    result = pl.concat(frames, how="diagonal")
    result = result.with_columns(
        ((pl.col("ewma_receptions_per_game") + 6.5)
         / (pl.col("ewma_targets_per_game") + 10.0)).alias("smoothed_catch_rate")
    )
    if result["season"].max() != 2024:
        raise ValueError("Usage-prop frame crossed the 2025 holdout boundary")
    return result


def _fit_predict(train: pl.DataFrame, validation: pl.DataFrame, prop: str) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    target = target_column(prop)
    fields = FEATURES[prop]
    model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                                 keep_empty_features=True)),
                      ("scaler", StandardScaler()),
                      ("model", PoissonRegressor(alpha=0.2, max_iter=500))])
    x_train = np.asarray(train.select(fields).to_numpy(), dtype=float)
    x_val = np.asarray(validation.select(fields).to_numpy(), dtype=float)
    y_train = np.asarray(train[target].fill_null(0).to_numpy(), dtype=float)
    y_val = np.asarray(validation[target].fill_null(0).to_numpy(), dtype=float)
    model.fit(x_train, y_train)
    predicted = np.clip(np.asarray(model.predict(x_val), dtype=float), 1e-6, None)
    baseline = np.clip(np.asarray(validation["ewma_receptions_per_game" if prop == "receptions" else "ewma_carries_per_game"].fill_null(0).to_numpy(), dtype=float), 1e-6, None)
    metrics = lambda values: {"mae": float(mean_absolute_error(y_val, values)),
                              "rmse": float(math.sqrt(mean_squared_error(y_val, values))),
                              "bias": float(np.mean(values - y_val)),
                              "count_deviance": float(mean_poisson_deviance(y_val, values))}
    return predicted, baseline, metrics(predicted), metrics(baseline)


def validation_report() -> Path:
    frame = load_historical_prop_frame()
    report: dict[str, Any] = {"training_seasons": list(range(2017, 2024)), "validation_season": 2024}
    for prop in PROP_MARKETS:
        train = frame.filter(pl.col("season") < 2024)
        validation = frame.filter(pl.col("season") == 2024)
        predicted, baseline, model_metrics, baseline_metrics = _fit_predict(train, validation, prop)
        target = target_column(prop)
        actual = np.asarray(validation[target].fill_null(0).to_numpy(), dtype=float)
        buckets = []
        for label, low, high in (("<40%", 0, .4), ("40-45%", .4, .45), ("45-50%", .45, .5),
                                 ("50-55%", .5, .55), ("55-60%", .55, .6), ("60-65%", .6, .65), ("65%+", .65, 1.01)):
            # Probability of going over the median integer line is a calibration proxy.
            line = np.floor(predicted).astype(int)
            probability = np.array([over_under_push(float(m), float(l))[0] for m, l in zip(predicted, line, strict=True)])
            mask = (probability >= low) & (probability < high)
            if mask.any():
                outcomes = actual[mask] > line[mask]
                buckets.append({"bucket": label, "player_games": int(mask.sum()),
                                "average_model_probability": float(probability[mask].mean()),
                                "actual_hit_rate": float(outcomes.mean())})
        report[prop] = {"training_rows": train.height, "validation_rows": validation.height,
                        "model": model_metrics, "ewma_baseline": baseline_metrics,
                        "model_minus_baseline_mae": model_metrics["mae"] - baseline_metrics["mae"],
                        "model_minus_baseline_rmse": model_metrics["rmse"] - baseline_metrics["rmse"],
                        "calibration": buckets}
        rows = validation.select(["game", "player", "position", "team", "opponent", "prediction_time", target]).with_columns(
            pl.Series("model_mean", predicted), pl.Series("ewma_mean", baseline),
        )
        rows.write_csv(f"reports/usage_{prop}_validation.csv")
    output = Path("reports/usage_props_validation.json")
    output.write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def _live_prop_rows(payload: dict[str, Any], captured: datetime) -> list[dict[str, Any]]:
    rows = []
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") not in PROP_MARKETS.values():
                continue
            updated = parse_time(market["last_update"])
            if updated > captured:
                continue
            prop = next(name for name, key in PROP_MARKETS.items() if key == market["key"])
            for outcome in market.get("outcomes", []):
                if outcome.get("name") not in {"Over", "Under"} or outcome.get("description") is None:
                    continue
                rows.append({"sportsbook": book["key"], "prop": prop, "player": outcome["description"],
                             "line": float(outcome["point"]), "side": outcome["name"],
                             "price": int(outcome["price"]), "quote_time": updated.isoformat()})
    return rows


def group_prop_quotes(quotes: list[dict[str, Any]]) -> dict[tuple[str, str, float], list[dict[str, Any]]]:
    """Group identical player/prop/line markets; different lines never mix."""
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for quote in quotes:
        grouped[(normalize_name(str(quote["player"])), str(quote["prop"]), float(quote["line"]))].append(quote)
    return grouped


def _live_means() -> dict[str, dict[str, Any]]:
    frame = pl.read_parquet("data/derived/phase2_2024_player_features.parquet")
    latest = frame.sort("kickoff_time").group_by("player_id").agg(pl.all().last())
    return {normalize_name(str(row["player"])): row for row in latest.to_dicts()}


def _current_roster() -> dict[str, set[str]]:
    """Use the current 2026 roster only for descriptive team labels."""
    rosters = nflreadpy.load_rosters([2026])
    result: dict[str, set[str]] = defaultdict(set)
    for row in rosters.select("full_name", "team").iter_rows(named=True):
        if row["full_name"] and row["team"]:
            result[normalize_name(str(row["full_name"]))].add(str(row["team"]))
    return result


def scan_current_slate(settings: Settings, now: datetime | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required for usage-prop scan")
    captured = now or datetime.now(UTC)
    eastern = ZoneInfo("America/New_York")
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{LIVE_BASE}/events", params={"apiKey": settings.odds_api_key})
        response.raise_for_status()
        events = [event for event in response.json() if parse_time(event["commence_time"]).astimezone(eastern).date() == captured.astimezone(eastern).date()]
        means = _live_means()
        roster = _current_roster()
        team_names = {row["team_name"]: row["team_abbr"] for row in nflreadpy.load_teams().to_dicts()}
        rows: list[dict[str, Any]] = []
        quotes: list[dict[str, Any]] = []
        for event in events:
            response = client.get(f"{LIVE_BASE}/events/{event['id']}/odds",
                                  params={"apiKey": settings.odds_api_key, "regions": settings.odds_regions,
                                          "markets": ",".join(PROP_MARKETS.values()), "oddsFormat": "american"})
            response.raise_for_status()
            event_quotes = _live_prop_rows(response.json(), captured)
            for quote in event_quotes:
                quote.update({"event_id": event["id"], "team": event["home_team"], "opponent": event["away_team"]})
            quotes.extend(event_quotes)
            grouped = group_prop_quotes(event_quotes)
            for (name, prop, line), offers in grouped.items():
                by_book_side = {(o["sportsbook"], o["side"]): o for o in offers}
                books = sorted({o["sportsbook"] for o in offers if (o["sportsbook"], "Over") in by_book_side and (o["sportsbook"], "Under") in by_book_side})
                if not books:
                    continue
                paired = [(by_book_side[(book, "Over")], by_book_side[(book, "Under")]) for book in books]
                over_prices = [int(pair[0]["price"]) for pair in paired]
                under_prices = [int(pair[1]["price"]) for pair in paired]
                best_over = max(paired, key=lambda pair: american_to_decimal(pair[0]["price"]))[0]
                best_under = max(paired, key=lambda pair: american_to_decimal(pair[1]["price"]))[1]
                med_over_decimal = statistics.median(american_to_decimal(price) for price in over_prices)
                med_under_decimal = statistics.median(american_to_decimal(price) for price in under_prices)
                market_over = (1 / med_over_decimal) / (1 / med_over_decimal + 1 / med_under_decimal) if len(books) >= 3 else None
                source = means.get(name)
                mean = float(source["ewma_receptions_per_game" if prop == "receptions" else "ewma_carries_per_game"]) if source and source["ewma_receptions_per_game" if prop == "receptions" else "ewma_carries_per_game"] is not None else None
                over, under, push = over_under_push(mean, line) if mean is not None else (None, None, None)
                difference = over - market_over if over is not None and market_over is not None else None
                age = max((captured - parse_time(str(o["quote_time"]))).total_seconds() / 60 for o in offers)
                status = []
                if len(books) < 3:
                    status.append("INSUFFICIENT_MARKET")
                if age > 15:
                    status.append("STALE")
                if difference is not None and abs(difference) < .05:
                    status.append("MARKET_ALIGNED")
                elif difference is not None:
                    status.append("MODEL LEANS OVER" if difference > 0 else "MODEL LEANS UNDER")
                player_name = next(iter({o["player"] for o in offers}))
                teams = roster.get(name, set())
                team = next(iter(teams)) if len(teams) == 1 else None
                home_abbr = team_names.get(event.get("home_team"), event.get("home_team"))
                away_abbr = team_names.get(event.get("away_team"), event.get("away_team"))
                opponent = away_abbr if team == home_abbr else home_abbr if team == away_abbr else None
                rows.append({"Player": player_name, "Team": team,
                             "Opponent": opponent, "Prop": prop, "Sportsbook Line": line,
                             "Model Mean": mean, "P(Over)": over, "P(Under)": under, "P(Push)": push,
                             "Best Over Odds": best_over["price"], "Best Under Odds": best_under["price"],
                             "Median Over Odds": decimal_to_american(med_over_decimal),
                             "Median Under Odds": decimal_to_american(med_under_decimal),
                             "Books Quoting": len(books),
                             "Market No-Vig P(Over)": market_over, "Model Minus Market": difference,
                             "Quote Age": age, "Status": "|".join(status) if status else "OK",
                             "EV Over": over * american_to_decimal(best_over["price"]) - 1 if over is not None else None,
                             "EV Under": under * american_to_decimal(best_under["price"]) - 1 if under is not None else None,
                             "event_id": event["id"], "books": len(books)})
    rows.sort(key=lambda row: -(abs(row["Model Minus Market"]) if row["Model Minus Market"] is not None else -1))
    return rows, quotes


def write_live_report(rows: list[dict[str, Any]], quotes: list[dict[str, Any]], output_dir: Path, day: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f"usage_prop_opportunities_{day}.md"
    csv_path = output_dir / f"usage_prop_opportunities_{day}.csv"
    quote_path = output_dir / f"usage_prop_quotes_{day}.csv"
    fields = ["Player", "Team", "Opponent", "Prop", "Sportsbook Line", "Model Mean", "P(Over)",
              "P(Under)", "P(Push)", "Best Over Odds", "Best Under Odds", "Market No-Vig P(Over)",
              "Median Over Odds", "Median Under Odds", "Books Quoting", "Model Minus Market", "Quote Age", "Status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({f: row[f] for f in fields} for row in rows)
    with quote_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "player", "prop", "line", "side", "price", "sportsbook", "quote_time", "team", "opponent"]); writer.writeheader(); writer.writerows(quotes)
    lines = ["# NFL USAGE PROP RESEARCH MVP", "", "Receptions and rushing-attempt market comparison. These are research leans, not guaranteed bets.", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if value is None: text = "—"
            elif field in {"P(Over)", "P(Under)", "P(Push)", "Market No-Vig P(Over)", "Model Minus Market"}: text = f"{float(value) * 100:.2f}%"
            elif field == "Quote Age": text = f"{float(value):.1f} min"
            else: text = str(value)
            values.append(text.replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Consensus is grouped by identical player/prop/line and requires both Over and Under from at least one book; the report comparison requires three paired books.", "Market probabilities are proportional no-vig estimates; raw probabilities and every quote are retained in the quote CSV.", "The live mean is an EWMA prior through the frozen 2024 feature history and does not read the sealed 2025 holdout.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report, csv_path, quote_path
