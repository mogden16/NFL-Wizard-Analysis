"""One-minute prospective T-60 dispatch with a per-event claim directory."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nfl_td_model.config import Settings
from nfl_td_model.odds import parse_time
from nfl_td_model.phase46_collect import capture_event, discover_events


def run_due(
    settings: Settings, now: datetime | None = None, event_id: str | None = None,
) -> list[dict[str, Any]]:
    """A scheduled call discovers events free and runs each due Stage A once."""
    current = now or datetime.now(UTC)
    results: list[dict[str, Any]] = []
    root = Path("reports/phase46")
    for event in discover_events(settings):
        if event_id is not None and event["event_id"] != event_id:
            continue
        kickoff = parse_time(event["kickoff"])
        cutoff = kickoff - timedelta(minutes=60)
        # A task launched in the minute before cutoff waits for the exact time.
        # A delayed task can still accept only market updates timestamped <= T-60.
        if not cutoff - timedelta(seconds=60) <= current <= cutoff + timedelta(seconds=30):
            continue
        event_dir = root / cutoff.date().isoformat() / event["event_id"]
        if (event_dir / "manifest.json").exists():
            continue
        lock = settings.data_dir / "phase46_claims" / event["event_id"]
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        if now is None and datetime.now(UTC) < cutoff:
            time.sleep((cutoff - datetime.now(UTC)).total_seconds())
        availability = settings.data_dir / "pregame_availability" / f"{event['event_id']}.json"
        input_path = capture_event(
            settings, event, root, availability if availability.exists() else None,
            now=now,
        )
        # A fresh worker process receives only the closed-schema pregame file.
        subprocess.run(
            [sys.executable, "-m", "nfl_td_model.phase46_predict", str(input_path), str(event_dir)],
            check=True,
        )
        import json

        results.append(json.loads((event_dir / "manifest.json").read_text(encoding="utf-8")))
    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Prospective Phase 4.6 T-60 dispatcher")
    parser.add_argument("--list-events", action="store_true")
    parser.add_argument("--event-id")
    args = parser.parse_args()
    if args.list_events:
        print(json.dumps(discover_events(Settings()), sort_keys=True))
    else:
        print(json.dumps(run_due(Settings(), event_id=args.event_id), sort_keys=True))
