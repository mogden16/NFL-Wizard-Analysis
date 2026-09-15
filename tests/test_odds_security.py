import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from nfl_td_model.odds import HistoricalOddsClient, OddsAPIError


def test_unauthorized_request_does_not_log_or_raise_credential(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "FAKE_SECRET_NEVER_LOG"
    requests = 0

    def unauthorized(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            401,
            request=request,
            json={"error_code": "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN"},
        )

    client = HistoricalOddsClient(secret, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(unauthorized))
    with caplog.at_level(logging.INFO), pytest.raises(OddsAPIError, match="paid") as raised:
        client.events(datetime(2024, 9, 26, 23, 15, tzinfo=UTC))
    assert requests == 1
    assert secret not in str(raised.value)
    assert secret not in caplog.text


def test_target_market_request_does_not_bundle_other_markets(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({key: value for key, value in request.url.params.items()})
        return httpx.Response(200, request=request, json={"timestamp": "2024-09-08T15:55:00Z", "data": {}})

    client = HistoricalOddsClient("secret", tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    client.event_market_odds("event", datetime(2024, 9, 8, 16, tzinfo=UTC), "player_receptions")
    assert seen["markets"] == "player_receptions"
    assert seen["regions"] == "us"
