"""Settlement-only side of the September 13, 2026 exhibition replay."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from nfl_td_model.config import Settings
from nfl_td_model.market_math import (
    american_to_decimal,
    closing_line_value_probability,
)
from nfl_td_model.odds import HistoricalOddsClient, extract_market_rows, parse_time
from nfl_td_model.phase1 import normalize_name
from nfl_td_model.phase45 import MANIFEST, PREDICTIONS, QUOTES, SLATE_DATE

RESULTS = Path("reports/phase45_settled_predictions.csv")
HTML_REPORT = Path("reports/phase45_full_slate.html")
REPORT = Path("reports/phase45_report.json")
OUTCOMES = Path("data/raw/phase45_espn_final_2026-09-13.json")
THRESHOLDS = (("EV > 0%", 0.0), ("EV >= 5%", 0.05), ("EV >= 10%", 0.10), ("EV >= 15%", 0.15))
TEAM_ALIASES = {"WSH": "WAS"}


def verify_frozen() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError("Stage A manifest must exist before settlement")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, field in ((PREDICTIONS, "prediction_sha256"), (QUOTES, "quotes_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f"Stage A artifact was modified: {path}")
    if manifest["slate_date"] != SLATE_DATE or not manifest["diagnostic_exhibition_slate"]:
        raise ValueError("Incorrect or non-exhibition Stage A artifact")
    return manifest


def _fetch_final_games() -> list[dict[str, Any]]:
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    with httpx.Client(timeout=25) as http:
        scoreboard = http.get(scoreboard_url, params={"dates": "20260913"})
        scoreboard.raise_for_status()
        events = scoreboard.json().get("events", [])
        if len(events) != 13 or any(e["status"]["type"]["name"] != "STATUS_FINAL" for e in events):
            raise RuntimeError("All 13 games must be final before Stage B loads any game outcome")
        games = []
        for event in events:
            response = http.get(
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
                params={"event": event["id"]},
            )
            response.raise_for_status()
            games.append({"event": event, "summary": response.json()})
    return games


def _outcome_map(games: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], int], dict[str, Any]]:
    scores: dict[tuple[str, str, str], int] = defaultdict(int)
    crosscheck: dict[str, Any] = {}
    for game in games:
        competitors = game["event"]["competitions"][0]["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")["team"]["abbreviation"]
        away = next(c for c in competitors if c["homeAway"] == "away")["team"]["abbreviation"]
        home = TEAM_ALIASES.get(home, home)
        away = TEAM_ALIASES.get(away, away)
        code = f"{away}@{home}"
        summary = game["summary"]
        for team in summary["boxscore"]["players"]:
            abbreviation = TEAM_ALIASES.get(team["team"]["abbreviation"], team["team"]["abbreviation"])
            for group in team["statistics"]:
                if group["name"] not in {"rushing", "receiving"}:
                    continue
                td_index = group["labels"].index("TD")
                for athlete in group["athletes"]:
                    td = int(athlete["stats"][td_index])
                    if td:
                        name = normalize_name(athlete["athlete"]["displayName"])
                        scores[(code, abbreviation, name)] += td
        play_count = sum(
            p["type"]["text"] in {"Passing Touchdown", "Rushing Touchdown"}
            for p in summary.get("scoringPlays", [])
        )
        box_count = sum(count for (g, _, _), count in scores.items() if g == code)
        crosscheck[code] = {"rushing_receiving_box_tds": box_count, "listed_rush_pass_scoring_plays": play_count}
        if box_count != play_count:
            raise ValueError(f"Scoring-play and boxscore touchdown totals disagree: {code}")
    return dict(scores), crosscheck


def _injury_map(games: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Retrospective availability audit, never an input to frozen predictions."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for game in games:
        competitors = game["event"]["competitions"][0]["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")["team"]["abbreviation"]
        away = next(c for c in competitors if c["homeAway"] == "away")["team"]["abbreviation"]
        code = f"{TEAM_ALIASES.get(away, away)}@{TEAM_ALIASES.get(home, home)}"
        for team in game["summary"].get("injuries", []):
            for injury in team.get("injuries", []):
                name = normalize_name(injury["athlete"]["displayName"])
                result[(code, name)] = {
                    "status": injury.get("status"),
                    "timestamp": injury.get("date"),
                    "fantasy_status": injury.get("details", {}).get("fantasyStatus", {}).get("description"),
                }
    return result


def _read_predictions() -> list[dict[str, Any]]:
    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "expected_team_td", "expected_player_td", "model_atd_probability", "rolling_xtd_share",
            "raw_implied_best", "raw_implied_median", "edge_best", "edge_median", "ev_best", "ev_median",
        ):
            row[key] = float(row[key]) if row[key] else None
        for key in ("best_odds", "books_quoting"):
            row[key] = int(row[key])
        for key in ("diagnostic_bet_best", "diagnostic_bet_median"):
            row[key] = row[key] == "True"
    return rows


def _portfolio(rows: list[dict[str, Any]], kind: str, threshold: float, strict: bool) -> dict[str, Any]:
    ev_key = f"ev_{kind}"
    edge_key = f"edge_{kind}"
    odds_key = f"{kind}_odds"
    selected = [
        row for row in rows
        if row[ev_key] is not None and row[edge_key] is not None
        and row[edge_key] >= 0.025
        and (row[ev_key] > threshold if strict else row[ev_key] >= threshold)
    ]
    wins = sum(row["actual_td"] > 0 for row in selected)
    gross = sum(
        (1 / row["raw_implied_median"] if kind == "median" else american_to_decimal(int(row[odds_key]))) - 1
        for row in selected if row["actual_td"] > 0
    )
    net = gross - (len(selected) - wins)
    return {
        "bets": len(selected), "wins": wins, "losses": len(selected) - wins,
        "hit_rate": wins / len(selected) if selected else None,
        "risked_units": float(len(selected)), "gross_winnings_units": gross,
        "gross_returns_including_stakes_units": gross + wins,
        "net_units": net, "roi": net / len(selected) if selected else None,
        "average_american_odds": statistics.mean(float(row[odds_key]) for row in selected) if selected else None,
        "average_decimal_odds": statistics.mean(
            (1 / row["raw_implied_median"] if kind == "median" else american_to_decimal(int(row[odds_key])))
            for row in selected
        ) if selected else None,
        "average_model_probability": statistics.mean(row["model_atd_probability"] for row in selected) if selected else None,
        "average_probability_edge": statistics.mean(row[edge_key] for row in selected) if selected else None,
        "average_ev": statistics.mean(row[ev_key] for row in selected) if selected else None,
    }


def _closing_quotes(settings: Settings, rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    if not settings.odds_api_key:
        return {}
    client = HistoricalOddsClient(settings.odds_api_key, settings.data_dir, settings.odds_regions)
    by_event = {row["event_id"]: parse_time(row["kickoff"]) for row in rows}
    closing: dict[tuple[str, str, str], int] = {}
    for event_id, kickoff in by_event.items():
        payload = client.event_odds(event_id, kickoff - timedelta(seconds=1))
        for quote in extract_market_rows(payload, kickoff - timedelta(seconds=1)):
            if quote["market"] != "player_anytime_td" or quote["name"] != "Yes":
                continue
            key = (event_id, normalize_name(str(quote["description"])), quote["sportsbook"])
            closing[key] = int(quote["price"])
    return closing


def _write_sortable_html(rows: list[dict[str, Any]]) -> None:
    """Render every quoted player in a self-contained, sortable audit table."""
    if HTML_REPORT.exists():
        raise FileExistsError("Sortable report already exists")
    fields = (
        ("game", "Game"), ("player", "Player"), ("position", "Pos"),
        ("team", "Team"), ("opponent", "Opp"), ("kickoff", "Kickoff UTC"),
        ("prediction_time", "T−60 UTC"), ("expected_team_td", "Team xTD"),
        ("expected_player_td", "Player xTD"), ("model_atd_probability", "P(ATD)"),
        ("rolling_xtd_share", "Rolling xTD share"), ("goal_line_share", "Goal-line share"),
        ("best_sportsbook", "Best book"), ("best_odds", "Best odds"),
        ("median_odds", "Median odds"), ("books_quoting", "Books"),
        ("raw_implied_best", "Implied P"), ("edge_best", "Edge"),
        ("ev_best", "EV"), ("diagnostic_bet_best", "Bet"),
        ("actual_td", "Actual TD"), ("pl_best_units", "P/L units"),
        ("ineligibility_reason", "Missing reason"),
    )
    percentage = {"model_atd_probability", "rolling_xtd_share", "goal_line_share", "raw_implied_best", "edge_best", "ev_best"}
    def display(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        if value is None or value == "":
            return ""
        if key in percentage:
            return f"{float(value):.1%}"
        if key in {"expected_team_td", "expected_player_td", "pl_best_units"}:
            return f"{float(value):.3f}"
        return str(value)
    head = "".join(f"<th onclick='sortTable({index})'>{html.escape(label)} ↕</th>" for index, (_, label) in enumerate(fields))
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(display(row, key))}</td>" for key, _ in fields) + "</tr>"
        for row in rows
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Phase 4.5 full-slate ATD replay</title>
<style>body{{font:14px system-ui;margin:24px;color:#20242a}}p{{max-width:1000px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}}
th,td{{border-bottom:1px solid #ddd;padding:6px 10px;text-align:right}}
th{{position:sticky;top:0;background:#f0f3f6;cursor:pointer}}th:nth-child(-n+5),td:nth-child(-n+5){{text-align:left}}
tr:hover{{background:#f7faff}}.scroll{{overflow:auto;max-height:80vh}}</style>
<h1>September 13, 2026 — Phase 4.5 hierarchical baseline</h1>
<p>Diagnostic exhibition only. Predictions were frozen before settlement. Click a column to sort.
Blank model fields indicate insufficient pregame history or ambiguous team assignment.
The provisional betting rule was EV ≥ 5% and probability edge ≥ 2.5 percentage points.</p>
<div class="scroll"><table id="players"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<script>let direction={{}};function sortTable(n){{let table=document.getElementById('players');
let body=table.tBodies[0],rows=[...body.rows],up=!(direction[n]??false);direction[n]=up;
rows.sort((a,b)=>{{let x=a.cells[n].innerText.trim(),y=b.cells[n].innerText.trim();
let numeric=/^[+-]?\\d+(\\.\\d+)?%?$/,nx=parseFloat(x),ny=parseFloat(y);
let result=(numeric.test(x)&&numeric.test(y))?nx-ny:x.localeCompare(y);
return up?result:-result}});rows.forEach(r=>body.appendChild(r))}}</script></html>"""
    HTML_REPORT.write_text(document, encoding="utf-8")


def settle_stage_b(settings: Settings) -> dict[str, Any]:
    """Verify frozen predictions before loading any 2026 outcomes or close prices."""
    manifest = verify_frozen()
    if RESULTS.exists() or REPORT.exists():
        raise FileExistsError("Stage B report already exists")
    games = _fetch_final_games()
    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    if OUTCOMES.exists():
        raise FileExistsError("Raw settlement source already exists")
    OUTCOMES.write_text(json.dumps(games) + "\n", encoding="utf-8")
    outcome_hash = hashlib.sha256(OUTCOMES.read_bytes()).hexdigest()
    scores, crosscheck = _outcome_map(games)
    injuries = _injury_map(games)
    rows = _read_predictions()
    closing = _closing_quotes(settings, rows)
    team_actual: dict[tuple[str, str], int] = defaultdict(int)
    scorer_by_name: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (game, team, _), count in scores.items():
        team_actual[(game, team)] += count
    for (game, team, name), count in scores.items():
        scorer_by_name[(game, name)].append((team, count))
    for row in rows:
        injury = injuries.get((row["game"], normalize_name(row["player"])), {})
        injury_time = parse_time(injury["timestamp"]) if injury.get("timestamp") else None
        cutoff = parse_time(row["prediction_time"])
        kickoff = parse_time(row["kickoff"])
        row["retrospective_injury_status"] = injury.get("status")
        row["retrospective_injury_time"] = injury["timestamp"] if injury_time else None
        is_inactive = injury.get("status") == "Out" and injury.get("fantasy_status") == "INACTIVE"
        row["known_inactive_at_t60_audit"] = bool(is_inactive and injury_time and injury_time <= cutoff)
        row["inactive_by_kickoff_audit"] = bool(is_inactive and injury_time and injury_time <= kickoff)
        candidates = scorer_by_name.get((row["game"], normalize_name(row["player"])), [])
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous touchdown scorer name: {row['game']} {row['player']}")
        row["actual_td"] = candidates[0][1] if candidates else 0
        row["actual_scoring_team"] = candidates[0][0] if candidates else None
        row["team_assignment_mismatch"] = bool(candidates and row["team"] != candidates[0][0])
        row["won"] = row["actual_td"] > 0
        row["team_actual_offensive_td"] = team_actual.get((row["game"], row["team"])) if row["team"] else None
        for kind in ("best", "median"):
            placed = row[f"diagnostic_bet_{kind}"]
            if placed:
                row[f"pl_{kind}_units"] = (
                    (1 / row["raw_implied_median"] if kind == "median" else american_to_decimal(row["best_odds"])) - 1
                    if row["won"] else -1.0
                )
            else:
                row[f"pl_{kind}_units"] = 0.0
        close = closing.get((row["event_id"], normalize_name(row["player"]), row["best_sportsbook"]))
        row["closing_same_book_odds"] = close
        row["clv_raw_probability"] = closing_line_value_probability(row["best_odds"], close) if close else None
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _write_sortable_html(rows)
    calibration = []
    bins = (("<10%", 0, .1), ("10–20%", .1, .2), ("20–30%", .2, .3),
            ("30–40%", .3, .4), ("40–50%", .4, .5), ("50–60%", .5, .6), ("60%+", .6, 1.01))
    for label, low, high in bins:
        sample = [r for r in rows if r["model_atd_probability"] is not None and low <= r["model_atd_probability"] < high]
        calibration.append({"bin": label, "players": len(sample),
                            "expected_scorers": sum(r["model_atd_probability"] for r in sample),
                            "actual_scorers": sum(r["actual_td"] > 0 for r in sample)})
    portfolios = {
        kind: {label: _portfolio(rows, kind, threshold, label == "EV > 0%") for label, threshold in THRESHOLDS}
        for kind in ("best", "median")
    }
    eligible = [r for r in rows if r["model_atd_probability"] is not None]
    misses = sorted(
        eligible, key=lambda r: abs(float(r["model_atd_probability"]) - float(r["won"])), reverse=True
    )[:12]
    bets = [r for r in rows if r["diagnostic_bet_best"]]
    inactive_bets = [r for r in bets if r["inactive_by_kickoff_audit"]]
    void_adjusted = _portfolio(
        [r for r in rows if not r["inactive_by_kickoff_audit"]], "best", 0.05, False
    )
    matched_median_gross = sum(1 / r["raw_implied_median"] - 1 for r in bets if r["actual_td"] > 0)
    matched_median_net = matched_median_gross - sum(r["actual_td"] == 0 for r in bets)
    edge_ranges = (("below_market", -10.0, 0.0), ("0–2.5pp", 0.0, .025),
                   ("2.5–10pp", .025, .10), ("10–20pp", .10, .20), ("20pp+", .20, 10.0))
    edge_diagnostic = []
    for label, low, high in edge_ranges:
        sample = [r for r in eligible if r["edge_best"] is not None and low <= r["edge_best"] < high]
        edge_diagnostic.append({
            "edge_bin": label, "players": len(sample),
            "mean_model_probability": statistics.mean(r["model_atd_probability"] for r in sample) if sample else None,
            "actual_score_rate": sum(r["actual_td"] > 0 for r in sample) / len(sample) if sample else None,
        })
    report = {
        "settled_at_utc": datetime.now(UTC).isoformat(), "slate_date": SLATE_DATE,
        "diagnostic_exhibition_slate": True, "research_firewall": "Excluded from all later Phase 5 model/feature/calibration/threshold decisions",
        "stage_a_prediction_sha256": manifest["prediction_sha256"],
        "stage_a_quote_sha256": manifest["quotes_sha256"], "settlement_source": "ESPN final scoreboard and game summary",
        "settlement_source_sha256": outcome_hash, "games": len(games), "quoted_players": len(rows),
        "eligible_players": len(eligible), "scoring_crosscheck": crosscheck,
        "portfolios": portfolios, "calibration_descriptive_only": calibration,
        "same_best_selected_bets_at_median_price": {
            "bets": len(bets), "wins": sum(r["actual_td"] > 0 for r in bets),
            "net_units": matched_median_net,
            "roi": matched_median_net / len(bets) if bets else None,
        },
        "retrospective_eligibility_audit": {
            "known_inactive_at_t60_bets": [
                {"game": r["game"], "player": r["player"], "injury_time": r["retrospective_injury_time"]}
                for r in bets if r["known_inactive_at_t60_audit"]
            ],
            "inactive_by_kickoff_bets": [
                {"game": r["game"], "player": r["player"], "injury_time": r["retrospective_injury_time"]}
                for r in inactive_bets
            ],
            "best_portfolio_excluding_inactive_by_kickoff": void_adjusted,
            "interpretation": "This is a retrospective audit. It does not repair the frozen Stage A eligibility decision.",
        },
        "edge_performance_descriptive_only": edge_diagnostic,
        "clv_best_bets": {"available": sum(r["clv_raw_probability"] is not None for r in bets),
                          "positive": sum((r["clv_raw_probability"] or 0) > 0 for r in bets),
                          "mean_probability_clv": statistics.mean(r["clv_raw_probability"] for r in bets if r["clv_raw_probability"] is not None)
                          if any(r["clv_raw_probability"] is not None for r in bets) else None},
        "largest_misses": [{"game": r["game"], "player": r["player"], "team": r["team"],
                            "model_probability": r["model_atd_probability"], "actual_td": r["actual_td"],
                            "expected_team_td": r["expected_team_td"],
                            "actual_team_offensive_td": r["team_actual_offensive_td"]} for r in misses],
        "result_file": str(RESULTS), "result_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "sortable_report_file": str(HTML_REPORT),
        "sortable_report_sha256": hashlib.sha256(HTML_REPORT.read_bytes()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_settlement() -> dict[str, Any]:
    """Check that settlement still points to the exact frozen prediction bytes."""
    frozen = verify_frozen()
    if not REPORT.exists() or not RESULTS.exists() or not OUTCOMES.exists() or not HTML_REPORT.exists():
        raise FileNotFoundError("Phase 4.5 settlement artifacts are incomplete")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["stage_a_prediction_sha256"] != frozen["prediction_sha256"]:
        raise ValueError("Settlement does not reference frozen Stage A predictions")
    if report["stage_a_quote_sha256"] != frozen["quotes_sha256"]:
        raise ValueError("Settlement does not reference frozen Stage A quotes")
    if hashlib.sha256(RESULTS.read_bytes()).hexdigest() != report["result_sha256"]:
        raise ValueError("Settlement result was modified")
    if hashlib.sha256(OUTCOMES.read_bytes()).hexdigest() != report["settlement_source_sha256"]:
        raise ValueError("Settlement source was modified")
    if hashlib.sha256(HTML_REPORT.read_bytes()).hexdigest() != report["sortable_report_sha256"]:
        raise ValueError("Sortable report was modified")
    if parse_time(frozen["created_at_utc"]) >= parse_time(report["settled_at_utc"]):
        raise ValueError("Settlement precedes frozen predictions")
    if report["games"] != 13 or not report["diagnostic_exhibition_slate"]:
        raise ValueError("Incomplete or non-exhibition settlement")
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != frozen["quoted_players"] or len({row["game"] for row in rows}) != 13:
        raise ValueError("Settled player table does not cover the slate")
    return {"games": 13, "players": len(rows), "prediction_sha256": frozen["prediction_sha256"]}
