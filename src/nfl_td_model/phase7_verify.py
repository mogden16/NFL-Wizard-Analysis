"""Verify the frozen Phase 7 chronology, market cutoffs, and settlement."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_td_model.phase7_stage_a import FORBIDDEN, PREGAME_FIELDS
from nfl_td_model.phase7_stage_b import verify_frozen_rule, verify_stage_a


def verify_phase7() -> dict[str, Any]:
    root = Path("reports/phase7")
    stage_a = verify_stage_a()
    rule = verify_frozen_rule()
    t60_manifest_path = Path("reports/phase7_t60_quote_manifest.json")
    if hashlib.sha256(t60_manifest_path.read_bytes()).hexdigest() != stage_a["quote_manifest_sha256"]:
        raise ValueError("Stage A source quote manifest changed")
    t60_records = json.loads(t60_manifest_path.read_text(encoding="utf-8"))
    if len(t60_records) != 512:
        raise ValueError("Historical T-60 event coverage changed")
    for record in t60_records:
        if record["season"] not in (2023, 2024) or datetime.fromisoformat(
                record["snapshot_time"]) > datetime.fromisoformat(record["prediction_time"]):
            raise ValueError("T-60 source snapshot crossed the prediction cutoff")
        payload = Path("data/raw/the_odds_api") / f"{record['payload_sha256']}.json"
        if hashlib.sha256(payload.read_bytes()).hexdigest() != record["payload_sha256"]:
            raise ValueError("Historical T-60 odds payload changed")
    pregame = pl.scan_parquet("data/phase7/pregame_player_view.parquet")
    if set(pregame.collect_schema().names()) != PREGAME_FIELDS:
        raise ValueError("Phase 7 pregame interface leaked a result field")
    predictions = pl.read_csv(root / "stage_a/predictions.csv")
    quotes = pl.read_csv(root / "stage_a/all_quotes.csv")
    if FORBIDDEN.intersection(predictions.columns) or FORBIDDEN.intersection(quotes.columns):
        raise ValueError("Stage A artifact contains forbidden result or settlement data")
    if predictions.height != stage_a["quoted_players"] or predictions["season"].unique().sort().to_list() != [2023, 2024]:
        raise ValueError("Stage A player universe changed")
    if quotes.filter(pl.col("quote_time") > pl.col("prediction_time")).height:
        raise ValueError("Post-T-60 sportsbook quote entered Stage A")
    if predictions.filter(pl.col("best_quote_time") > pl.col("prediction_time")).height:
        raise ValueError("Post-T-60 best quote entered predictions")
    if predictions.filter(pl.col("eligible_at_prediction_time") == True).height:
        raise ValueError("Unverified historical active status became strict eligibility")
    modeled = predictions.filter(pl.col("p_football").is_not_null())
    for row in modeled.iter_rows(named=True):
        if not math.isclose(row["edge_best"], row["p_football"] - row["p_market_raw_best"], abs_tol=1e-10):
            raise ValueError("Best-price probability edge arithmetic changed")
        if not math.isclose(row["ev_best"], row["p_football"] * row["best_decimal"] - 1, abs_tol=1e-10):
            raise ValueError("Best-price expected value arithmetic changed")
        if (datetime.fromisoformat(row["kickoff"]) - datetime.fromisoformat(
                row["prediction_time"])).total_seconds() != 3600:
            raise ValueError("Phase 7 prediction was not at T-60")
    summaries = {}
    for season in (2023, 2024):
        closing_manifest = json.loads(Path(
            f"reports/phase7_closing_{season}_manifest.json"
        ).read_text(encoding="utf-8"))
        if len(closing_manifest) != 256:
            raise ValueError("Closing snapshot season coverage changed")
        for record in closing_manifest:
            if record["season"] != season or datetime.fromisoformat(
                    record["snapshot_time"]) > datetime.fromisoformat(record["kickoff"]):
                raise ValueError("Closing quote occurred after kickoff")
            payload = Path("data/raw/the_odds_api") / f"{record['payload_sha256']}.json"
            if hashlib.sha256(payload.read_bytes()).hexdigest() != record["payload_sha256"]:
                raise ValueError("Historical closing odds payload changed")
        path = root / f"settlement_{season}.csv"
        manifest = json.loads((root / f"settlement_{season}_manifest.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["settlement_sha256"]:
            raise ValueError("Frozen settlement changed")
        if manifest["stage_a_sha256"] != stage_a["prediction_sha256"]:
            raise ValueError("Settlement used another Stage A prediction file")
        if season == 2024 and manifest["frozen_rule_sha256"] != hashlib.sha256(
                (root / "development/frozen_rule.json").read_bytes()).hexdigest():
            raise ValueError("2024 settlement did not preserve the 2023 rule hash")
        settled = pl.read_csv(path)
        if settled.height != predictions.filter(pl.col("season") == season).height:
            raise ValueError("Settlement removed pregame player rows")
        unknown = settled.filter(pl.col("settlement_status") == "policy_unknown")
        if unknown["profit_best_units"].null_count() != unknown.height:
            raise ValueError("Unknown DNP policy was graded")
        if settled.filter(pl.col("season") != season).height:
            raise ValueError("Cross-season outcome leakage in settlement")
        summaries[str(season)] = {"quoted_players": settled.height,
                                  "graded_played": manifest["graded_played"],
                                  "policy_unknown": manifest["policy_unknown"]}
    validation = json.loads((root / "validation_report.json").read_text())
    if not rule["shortlist"] and validation["PHASE7_RESULT"] != "NO_ROBUST_EDGE":
        raise ValueError("A 2024 rule appeared despite the empty frozen 2023 shortlist")
    return {"stage_a_sha256": stage_a["prediction_sha256"],
            "frozen_rule_sha256": hashlib.sha256(
                (root / "development/frozen_rule.json").read_bytes()).hexdigest(),
            "seasons": summaries, "PHASE7_RESULT": validation["PHASE7_RESULT"]}


if __name__ == "__main__":
    print(json.dumps(verify_phase7(), indent=2))
