"""The Odds API historical snapshot adapter, with immutable response caching."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from nfl_td_model.storage import persist_raw_json

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"


class OddsAPIError(RuntimeError):
    """An API failure whose message never includes credential-bearing URLs."""


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Source timestamp has no timezone")
    return result


class HistoricalOddsClient:
    def __init__(self, key: str, data_dir: Path, regions: str = "us") -> None:
        # httpx's INFO request log includes query parameters, including apiKey.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.key = key
        self.data_dir = data_dir
        self.regions = regions
        self.http = httpx.Client(timeout=30)

    def _get(self, endpoint: str, prediction_time: datetime, **parameters: str) -> dict[str, Any]:
        date = prediction_time.isoformat().replace("+00:00", "Z")
        # Cache by path and parameters without ever writing the API key.
        cache_key = json.dumps([endpoint, date, parameters], sort_keys=True)
        import hashlib

        request_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        index = self.data_dir / "raw" / "odds_request_index" / f"{request_hash}.txt"
        if index.exists():
            return json.loads(Path(index.read_text(encoding="utf-8")).read_text(encoding="utf-8"))
        safe_url = f"{BASE_URL}{endpoint}?date={date}"
        for attempt in range(4):
            try:
                response = self.http.get(
                    f"{BASE_URL}{endpoint}",
                    params={"apiKey": self.key, "date": date, **parameters},
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2**attempt, 8))
                    continue
                if response.status_code >= 400:
                    error_code = None
                    if response.headers.get("content-type", "").startswith("application/json"):
                        body = response.json()
                        if isinstance(body, dict):
                            error_code = body.get("error_code")
                    if error_code == "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN":
                        raise OddsAPIError("Historical odds require a paid The Odds API usage plan")
                    raise OddsAPIError(
                        f"The Odds API returned HTTP {response.status_code}; "
                        "check the key and historical-event access"
                    )
                payload = response.json()
                if parse_time(payload["timestamp"]) > prediction_time:
                    raise ValueError("Historical odds API returned a future snapshot")
                raw_path = persist_raw_json(self.data_dir, "the_odds_api", payload, safe_url)
                index.parent.mkdir(parents=True, exist_ok=True)
                index.write_text(str(raw_path), encoding="utf-8")
                return payload
            except httpx.TransportError:
                if attempt == 3:
                    raise OddsAPIError("Historical odds transport failed") from None
                time.sleep(min(2**attempt, 8))
        raise OddsAPIError("Historical odds request exhausted retry budget")

    def events(self, prediction_time: datetime) -> dict[str, Any]:
        return self._get("/events", prediction_time)

    def featured_odds(self, prediction_time: datetime) -> dict[str, Any]:
        """Return historical timestamped spread and total snapshots for all listed games."""
        return self._get(
            "/odds",
            prediction_time,
            regions=self.regions,
            markets="spreads,totals",
            oddsFormat="american",
        )

    def event_odds(self, event_id: str, prediction_time: datetime) -> dict[str, Any]:
        return self._get(
            f"/events/{event_id}/odds",
            prediction_time,
            regions=self.regions,
            markets="player_anytime_td,spreads,totals",
            oddsFormat="american",
        )


def extract_market_rows(payload: dict[str, Any], prediction_time: datetime) -> list[dict[str, Any]]:
    """Flatten only market quotes actually timestamped before prediction time."""
    snapshot_time = parse_time(payload["timestamp"])
    if snapshot_time > prediction_time:
        raise ValueError("Historical odds snapshot is later than prediction time")
    rows: list[dict[str, Any]] = []
    for book in (payload.get("data") or {}).get("bookmakers", []):
        for market in book.get("markets", []):
            market_time = parse_time(market["last_update"])
            if market_time > prediction_time:
                # A wrapper snapshot can contain an asynchronously updated market.
                # Its own update time is the availability boundary.
                continue
            for outcome in market.get("outcomes", []):
                rows.append(
                    {
                        "sportsbook": book["key"],
                        "market": market["key"],
                        "quote_time": market_time,
                        "snapshot_time": snapshot_time,
                        "name": outcome.get("name"),
                        "description": outcome.get("description"),
                        "point": outcome.get("point"),
                        "price": outcome.get("price"),
                    }
                )
    return rows
