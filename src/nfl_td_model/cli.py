"""Command-line entry points for environment and Phase 1 audit."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import typer

from nfl_td_model.atd_price_scan import scan_current_slate, write_report
from nfl_td_model.config import Settings
from nfl_td_model.odds import OddsAPIError
from nfl_td_model.phase1 import build_phase1_audit
from nfl_td_model.phase2 import build_phase2_features
from nfl_td_model.phase2_verify import verify_historical_phase2, verify_phase2_features
from nfl_td_model.phase3 import build_phase3
from nfl_td_model.phase3_verify import verify_phase3
from nfl_td_model.phase4 import build_phase4
from nfl_td_model.phase4_market import collect_market_snapshots
from nfl_td_model.phase4_verify import verify_phase4
from nfl_td_model.phase5 import build_phase5
from nfl_td_model.phase5_verify import verify_phase5
from nfl_td_model.phase6 import build_phase6
from nfl_td_model.phase6_verify import verify_phase6
from nfl_td_model.phase7_verify import verify_phase7
from nfl_td_model.phase45 import build_stage_a
from nfl_td_model.phase45_settle import settle_stage_b, verify_frozen, verify_settlement
from nfl_td_model.receptions_backfill import build_plan, download_raw_backfill, normalize_backfill
from nfl_td_model.storage import connect_catalog
from nfl_td_model.usage_prop_audit import audit_large_disagreements
from nfl_td_model.usage_prop_calibration import calibration_report as usage_calibration_report
from nfl_td_model.usage_props import scan_current_slate as scan_usage_props
from nfl_td_model.usage_props import validation_report as usage_validation_report
from nfl_td_model.usage_props import write_live_report
from nfl_td_model.verify import verify_phase1

app = typer.Typer(help="NFL touchdown point-in-time research")


@app.command("receptions-backfill-plan")
def receptions_backfill_plan() -> None:
    """Create a no-network, cache-aware receptions backfill plan."""
    plan = build_plan(Settings().data_dir, Path("reports"))
    typer.echo(
        f"Receptions backfill: {plan.games} games, {plan.remaining_market_calls} calls, "
        f"{plan.estimated_credits} estimated credits; manifest: {plan.manifest_path}"
    )


@app.command("receptions-backfill")
def receptions_backfill(
    execute: bool = typer.Option(False, help="Allow network requests; defaults to dry-run."),
    available_credits: int = typer.Option(0, help="Credits available for the complete run."),
    force: bool = typer.Option(False, help="Redownload cached requests intentionally."),
) -> None:
    """Plan or run the immutable 2023–2024 receptions market backfill."""
    settings = Settings()
    plan = build_plan(settings.data_dir, Path("reports"))
    if not execute:
        typer.echo(f"DRY RUN: {plan.remaining_market_calls} calls / {plan.estimated_credits} credits")
        return
    if not settings.odds_api_key:
        raise typer.BadParameter("ODDS_API_KEY is required with --execute")
    try:
        output = download_raw_backfill(settings.data_dir, settings.odds_api_key, available_credits, force)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"BACKFILL NOT STARTED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Backfill complete: {output}")


@app.command("receptions-backfill-normalize")
def receptions_backfill_normalize() -> None:
    """Normalize cached receptions snapshots without network access."""
    outputs = normalize_backfill(Settings().data_dir, Path("reports"))
    typer.echo("Normalized: " + ", ".join(str(path) for path in outputs.values()))


@app.command()
def init() -> None:
    """Initialize the local DuckDB artifact catalog."""
    settings = Settings()
    with connect_catalog(settings.data_dir):
        pass
    typer.echo(f"Initialized {settings.data_dir / 'catalog.duckdb'}")


@app.command("phase1-audit")
def phase1_audit(games: int = typer.Option(10, min=1, max=16)) -> None:
    """Build a ten-game 2024 point-in-time audit and representative histories."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        audit, histories = build_phase1_audit(Settings(), games)
    except OddsAPIError as exc:
        typer.echo(f"HISTORICAL ODDS UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Audit: {audit}\nRepresentative histories: {histories}")


@app.command("phase1-verify")
def phase1_verify() -> None:
    """Enforce strict Phase 1 acceptance; exit nonzero on missing evidence."""
    settings = Settings()
    reports = Path("reports")
    try:
        verify_phase1(
            reports / "phase1_2024_week4_audit.csv",
            reports / "phase1_source_hashes.json",
            settings.data_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"PHASE 1 NOT ACCEPTED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("PHASE 1 GATE PASSED: 10 games; football publication times use a documented proxy")


@app.command("phase2-features")
def phase2_features() -> None:
    """Build strictly lagged 2017–2025 player features, without fitting a model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    output, coverage = build_phase2_features(Settings())
    typer.echo(f"Phase 2 features: {output}\nCoverage: {coverage}")


@app.command("phase2-verify")
def phase2_verify() -> None:
    """Check the 2024 foundation and the full 2017–2025 history."""
    try:
        verify_phase2_features(Settings().data_dir, Path("reports"))
        summary = verify_historical_phase2(Settings().data_dir, Path("reports"))
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"PHASE 2 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"PHASE 2 VERIFIED: {summary['rows']} rows across {summary['games']} games "
        f"in {summary['seasons']} seasons"
    )


@app.command("phase3-build")
def phase3_build() -> None:
    """Build chronological play-level xTD and strictly lagged player-game candidates."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    output, report = build_phase3(Settings())
    typer.echo(f"Phase 3 lagged features: {output}\nReport: {report}")


@app.command("phase3-verify")
def phase3_verify() -> None:
    """Enforce Phase 3 artifact, aggregation and temporal acceptance."""
    try:
        summary = verify_phase3(Settings().data_dir, Path("reports"))
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"PHASE 3 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 3 VERIFIED: {summary}")


@app.command("phase4-market-collect")
def phase4_market_collect() -> None:
    """Cache 2021–2024 historical T-60 featured-market snapshots."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    output = collect_market_snapshots(Settings())
    typer.echo(f"Phase 4 market snapshot manifest: {output}")


@app.command("phase4-build")
def phase4_build() -> None:
    """Build and compare chronological team offensive TD count models."""
    output, report = build_phase4(Settings())
    typer.echo(f"Phase 4 predictions: {output}\nReport: {report}")


@app.command("phase4-verify")
def phase4_verify() -> None:
    """Check strict odds cutoff, PBP target, temporal features and output distribution."""
    try:
        summary = verify_phase4(Settings(), Path("reports"))
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"PHASE 4 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 4 VERIFIED: {summary}")


@app.command("phase45-predict")
def phase45_predict() -> None:
    """Freeze September 13 T-60 predictions; rejects an existing artifact."""
    typer.echo(f"PHASE 4.5 STAGE A: {build_stage_a(Settings())}")


@app.command("phase45-settle")
def phase45_settle() -> None:
    """Settle only after all games are final and Stage A hashes verify."""
    try:
        result = settle_stage_b(Settings())
    except RuntimeError as exc:
        typer.echo(f"PHASE 4.5 SETTLEMENT WAITING: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 4.5 STAGE B: {result}")


@app.command("phase45-verify")
def phase45_verify() -> None:
    """Verify the immutable pregame artifact and its exhibition designation."""
    manifest = verify_frozen()
    typer.echo(f"PHASE 4.5 STAGE A VERIFIED: {manifest['prediction_sha256']}")
    if Path("reports/phase45_report.json").exists():
        typer.echo(f"PHASE 4.5 SETTLEMENT VERIFIED: {verify_settlement()}")


@app.command("phase5-build")
def phase5_build() -> None:
    """Build 2017-2024 player baselines and 2024 validation predictions."""
    predictions, report = build_phase5(Settings())
    typer.echo(f"Phase 5 predictions: {predictions}\nReport: {report}")


@app.command("phase5-verify")
def phase5_verify() -> None:
    """Enforce Phase 5 artifact, label, temporal, and coherence acceptance."""
    try:
        summary = verify_phase5()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        typer.echo(f"PHASE 5 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 5 VERIFIED: {summary}")


@app.command("phase6-build")
def phase6_build() -> None:
    """Evaluate preregistered position, LightGBM, and OOF-blend challengers."""
    predictions, report = build_phase6()
    typer.echo(f"Phase 6 predictions: {predictions}\nReport: {report}")


@app.command("phase6-verify")
def phase6_verify() -> None:
    """Verify frozen inputs, true 2023 OOF scores, and champion promotion."""
    try:
        summary = verify_phase6()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        typer.echo(f"PHASE 6 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 6 VERIFIED: {summary}")


@app.command("phase7-verify")
def phase7_verify() -> None:
    """Verify frozen T-60 quotes, chronological settlements, and 2023 rule."""
    try:
        summary = verify_phase7()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        typer.echo(f"PHASE 7 NOT VERIFIED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"PHASE 7 VERIFIED: {summary}")


@app.command("atd-price-scan")
def atd_price_scan() -> None:
    """Rank today's cross-sportsbook ATD Yes-price differences only."""
    captured = datetime.now(UTC)
    day = captured.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    try:
        rows, quotes = scan_current_slate(Settings(), now=captured)
        report, sortable_csv, quote_csv = write_report(rows, quotes, Path("reports"), day)
    except httpx.HTTPError as exc:
        typer.echo(f"ATD PRICE SCAN FAILED: live odds request failed ({exc.__class__.__name__})", err=True)
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(f"ATD PRICE SCAN FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"CROSS-BOOK ATD PRICE OPPORTUNITIES: {len(rows)} players")
    typer.echo(f"Report: {report}\nSortable CSV: {sortable_csv}\nAll quotes: {quote_csv}")


@app.command("usage-prop-validate")
def usage_prop_validate() -> None:
    """Evaluate the frozen EWMA and Poisson usage-prop models on 2024."""
    typer.echo(f"Usage prop mean validation: {usage_validation_report()}")
    typer.echo(f"Usage prop distribution calibration: {usage_calibration_report()}")


@app.command("usage-prop-audit")
def usage_prop_audit() -> None:
    """Audit the largest live usage-prop model/market discrepancies."""
    typer.echo(f"Usage prop disagreement audit: {audit_large_disagreements()}")


@app.command("usage-prop-scan")
def usage_prop_scan() -> None:
    """Rank current receptions and rushing-attempt market disagreements."""
    captured = datetime.now(UTC)
    day = captured.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    try:
        rows, quotes = scan_usage_props(Settings(), now=captured)
        report, sortable_csv, quote_csv = write_live_report(rows, quotes, Path("reports"), day)
    except httpx.HTTPError as exc:
        typer.echo(f"USAGE PROP SCAN FAILED: live odds request failed ({exc.__class__.__name__})", err=True)
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(f"USAGE PROP SCAN FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"NFL USAGE PROP RESEARCH MVP: {len(rows)} player-lines")
    typer.echo(f"Report: {report}\nSortable CSV: {sortable_csv}\nAll quotes: {quote_csv}")
