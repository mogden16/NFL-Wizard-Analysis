import json
from datetime import UTC, datetime

import pytest

from nfl_td_model.receptions_backfill import (
    BACKFILL_MARKET,
    build_plan,
    request_cache_key,
    require_budget,
)


def test_budget_guard_fails_closed():
    plan = build_plan()
    with pytest.raises(RuntimeError, match="no requests made"):
        require_budget(plan, plan.estimated_credits - 1)


def test_request_cache_key_is_stable_and_market_specific():
    a = request_cache_key("event", "2024-09-08T16:00:00+00:00")
    b = request_cache_key("event", "2024-09-08T16:00:00+00:00")
    assert a == b and len(a) == 64
    assert BACKFILL_MARKET == "player_receptions"


def test_plan_has_target_seasons_and_no_2025_access():
    plan = build_plan()
    assert plan.seasons == (2023, 2024)
    with open(plan.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert {row["season"] for row in manifest["games"]} == {2023, 2024}
    assert datetime.now(UTC).year >= 2026
