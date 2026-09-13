"""Command-line entry points for environment and Phase 1 audit."""

import logging
from pathlib import Path

import typer

from nfl_td_model.config import Settings
from nfl_td_model.odds import OddsAPIError
from nfl_td_model.phase1 import build_phase1_audit
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
