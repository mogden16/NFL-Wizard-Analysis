"""Small DuckDB catalog plus immutable raw artifact persistence."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


def connect_catalog(data_dir: Path) -> duckdb.DuckDBPyConnection:
    data_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(data_dir / "catalog.duckdb"))
    connection.execute(
        """CREATE TABLE IF NOT EXISTS raw_artifacts (
            sha256 VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, path VARCHAR NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL, source_url VARCHAR
        )"""
    )
    return connection


def persist_raw_json(data_dir: Path, source: str, payload: Any, source_url: str) -> Path:
    """Write content-addressed raw JSON and never mutate prior snapshots."""
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path = data_dir / "raw" / source / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    with connect_catalog(data_dir) as catalog:
        catalog.execute(
            "INSERT OR IGNORE INTO raw_artifacts VALUES (?, ?, ?, ?, ?)",
            [digest, source, str(path), datetime.now(UTC), source_url],
        )
    return path
