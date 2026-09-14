"""Command-line entry points for environment and Phase 1 audit."""

import logging
from pathlib import Path

import typer

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
from nfl_td_model.storage import connect_catalog
from nfl_td_model.verify import verify_phase1

app = typer.Typer(help="NFL touchdown point-in-time research")


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
