"""Copy existing scan reports to the read-only Pages dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_DATA = ROOT / "cloudflare-app" / "data"


def latest_common_slate() -> str:
    usage = {p.stem.removeprefix("usage_prop_opportunities_") for p in REPORTS.glob("usage_prop_opportunities_????-??-??.csv")}
    atd = {p.stem.removeprefix("atd_price_opportunities_") for p in REPORTS.glob("atd_price_opportunities_????-??-??.csv")}
    common = sorted(usage & atd)
    if not common:
        raise FileNotFoundError("No matching usage-prop and ATD report dates")
    return common[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def publish(day: str | None = None, end_day: str | None = None) -> dict[str, str]:
    day = day or latest_common_slate()
    start_date = date.fromisoformat(day)
    end_date = date.fromisoformat(end_day) if end_day else start_date
    if end_date < start_date:
        raise ValueError("Dashboard end date cannot precede its start date")
    usage_path = REPORTS / f"usage_prop_opportunities_{day}.csv"
    atd_path = REPORTS / f"atd_price_opportunities_{day}.csv"
    quotes_path = REPORTS / f"atd_price_quotes_{day}.csv"
    usage = read_rows(usage_path)
    atd = read_rows(atd_path)
    quotes = read_rows(quotes_path)
    if not usage or not atd or not quotes:
        raise ValueError("Both scanners need nonempty reports and ATD quotes")
    usage_times = [datetime.fromisoformat(row["Quote Timestamp"]) for row in usage if row["Quote Timestamp"]]
    atd_times = [datetime.fromisoformat(row["captured_at"]) for row in quotes if row["captured_at"]]
    if not usage_times or not atd_times:
        raise ValueError("Scanner timestamps are required for the dashboard")
    metadata = {
        "slate_date": day,
        "slate_end_date": end_date.isoformat(),
        "usage_latest_quote_at": max(usage_times).isoformat(),
        "atd_captured_at": max(atd_times).isoformat(),
    }
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(usage_path, PUBLIC_DATA / "usage_prop_opportunities.csv")
    shutil.copyfile(atd_path, PUBLIC_DATA / "atd_price_opportunities.csv")
    (PUBLIC_DATA / "scan_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", help="Report date to publish (YYYY-MM-DD)")
    parser.add_argument("--through", help="Inclusive slate end date (YYYY-MM-DD)")
    args = parser.parse_args()
    print(json.dumps(publish(args.day, args.through), indent=2))
