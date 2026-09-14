"""Phase 5 player-probability and holdout leakage regression tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_td_model import phase5
from nfl_td_model.phase5 import (
    MODEL_FAMILIES,
    _matrix,
    _safe_season_file,
    _sigmoid_fit,
    fit_hierarchy,
    fit_player_logistic,
    guard_model_input,
    hierarchy_from_arrays,
    historical_td_features,
)
from nfl_td_model.phase5_verify import verify_phase5


def test_current_game_outcome_cannot_enter_historical_td_rate() -> None:
    row = {"game": "2024_04_A_B", "player_id": "player",
           "player_source_game_ids": "2024_01_A_B;2024_02_A_C;2024_03_A_D"}
    labels = {("2024_01_A_B", "player"): 0, ("2024_02_A_C", "player"): 1,
              ("2024_03_A_D", "player"): 0, ("2024_04_A_B", "player"): 0}
    before = historical_td_features(row, labels)
    labels[("2024_04_A_B", "player")] = 3
    assert historical_td_features(row, labels) == before
    row["player_source_game_ids"] += ";2024_04_A_B"
    with pytest.raises(ValueError, match="Current game"):
        historical_td_features(row, labels)


@pytest.mark.parametrize("unsafe", [
    "anytime_td", "actual_td_count", "p_market", "market_median_decimal",
    "current_game_xtd", "current_game_carries", "current_game_targets",
    "current_game_snaps", "closing_atd_odds", "diagnostic_exhibition_slate",
    "carries", "targets", "snaps", "rush_touchdown",
])
def test_outcome_market_and_current_game_fields_are_not_model_inputs(unsafe: str) -> None:
    frame = pl.DataFrame({"season": [2022], unsafe: [1], "last5_carry_share": [.3]})
    with pytest.raises(ValueError, match="declared lagged"):
        guard_model_input(frame, (unsafe,))


@pytest.mark.parametrize("season", [2025, 2026])
def test_holdout_and_exhibition_cannot_enter_feature_matrix(season: int) -> None:
    frame = pl.DataFrame({"season": [season], "last5_carry_share": [.3]})
    with pytest.raises(ValueError, match="holdout firewall"):
        _matrix(frame, ("last5_carry_share",))
    with pytest.raises(ValueError, match="only 2017-2024"):
        _safe_season_file(season)


def test_fit_and_calibration_periods_are_separate() -> None:
    fitting = pl.DataFrame({"season": [2023], "last5_carry_share": [.3],
                            "anytime_td": [0]})
    with pytest.raises(ValueError, match="2022 only"):
        fit_player_logistic(fitting, ("last5_carry_share",))
    with pytest.raises(ValueError, match="2022 only"):
        fit_hierarchy(fitting)
    with pytest.raises(ValueError, match="2023"):
        _sigmoid_fit(fitting.with_columns(pl.lit(2024).alias("season")), np.array([.2]))


def test_team_environment_is_fit_only_on_prior_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted_years: list[tuple[int, ...]] = []

    class FakeModel:
        def predict(self, current: pl.DataFrame) -> np.ndarray:
            assert max(fitted_years[-1]) < current["season"].min()
            return np.full(current.height, 2.5)

    def fake_fit(prior: pl.DataFrame, _fields: object, _kind: str) -> FakeModel:
        fitted_years.append(tuple(sorted(prior["season"].unique().to_list())))
        return FakeModel()

    monkeypatch.setattr(phase5, "fit_count_model", fake_fit)
    strict = pl.DataFrame({"season": [2021, 2022, 2023, 2024],
                           "game_id": ["a", "b", "c", "d"]})
    scored = phase5._out_of_sample_team_expectations(strict)
    assert fitted_years == [(2021,), (2021, 2022), (2021, 2022, 2023)]
    assert scored["season"].to_list() == [2022, 2023, 2024]


def test_hierarchical_expectations_sum_to_team_expectation() -> None:
    values = np.array([[.4, .2], [.1, .3], [.2, .2], [.0, .0]])
    available = np.ones_like(values)
    groups = np.array([0, 0, 1, 1])
    team_expected = np.array([3., 3., 1.5, 1.5])
    probabilities, expected = hierarchy_from_arrays(
        values, available, groups, team_expected, np.array([.5, .5])
    )
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert expected[:2].sum() == pytest.approx(3.)
    assert expected[2:].sum() == pytest.approx(1.5)


def test_training_and_inference_share_exact_feature_projection() -> None:
    fields = MODEL_FAMILIES["E_team_usage_xtd"]
    values: dict[str, list[float | int]] = {field: [.25] for field in fields}
    values["season"] = [2024]
    values["anytime_td"] = [1]
    values["p_market"] = [.45]
    frame = pl.DataFrame(values)
    matrix = _matrix(frame, fields)
    assert matrix.shape == (1, len(fields))
    assert np.array_equal(matrix, _matrix(frame.drop("anytime_td", "p_market"), fields))


def test_phase5_artifacts_verify_without_2025_data() -> None:
    summary = verify_phase5()
    assert summary["validation_rows"] > 6000
    assert summary["market_matches"] > 0
