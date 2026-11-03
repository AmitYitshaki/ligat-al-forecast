"""Numerical illustrations, saved-run integrity and diagnostic aggregation."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.analysis.prematch_b_review import (
    WORDS,
    alignment_summary,
    assert_same_table_state,
    calibration_bins,
    configuration_words,
    control_numbers,
    elo_example,
    load_elo_review,
    load_table_state_snapshot,
    ordinal_probabilities,
    prediction_diagnostics,
    score_example,
)
from ligat.viz.prematch_b import (
    calibration_chart,
    confusion_chart,
    review_table,
    time_weight_chart,
)


def test_update_and_goal_rule_match_hand_calculation():
    example, multipliers = elo_example()
    expected = 1 / (1 + 10 ** (-30 / 400))
    assert example.E.iloc[0] == pytest.approx(expected)
    assert example["Δ"].iloc[0] == pytest.approx(15 * (1 - expected))
    assert example["בית אחרי"].iloc[0] + example["חוץ אחרי"].iloc[0] == 3000
    assert example["בקיץ: דירוג 1600"].iloc[0] == 1567
    assert multipliers["מכפיל G"].tolist() == [1, 1, 1.5, 1.75, 1.875, 2]


def test_ordinal_forward_has_correct_orientation_and_no_second_home_offset():
    equal = ordinal_probabilities(0, 3, -0.8, 0.45)
    stronger = ordinal_probabilities(100, 3, -0.8, 0.45)
    assert equal.sum() == pytest.approx(1)
    assert equal[0] == pytest.approx(1 / (1 + math.exp(0.45)))
    assert stronger[0] > equal[0]
    assert stronger[2] < equal[2]
    assert np.all(stronger > 0)
    with pytest.raises(ValueError):
        ordinal_probabilities(0, 1, 1, 0)


def test_scoring_example_uses_two_cdf_boundaries():
    frame = score_example()
    assert WORDS in frame
    assert frame.log_loss.iloc[0] == pytest.approx(-math.log(0.58))
    assert frame.rps.iloc[0] == pytest.approx(((0.58 - 1) ** 2 + (0.83 - 1) ** 2) / 2)
    assert frame.rps.iloc[2] == pytest.approx((0.58**2 + 0.83**2) / 2)


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "p_home": [1.0, 0.6, 0.2],
            "p_draw": [0.0, 0.3, 0.1],
            "p_away": [0.0, 0.1, 0.7],
            "outcome": [0, 1, 2],
        }
    )


def test_calibration_keeps_boundary_values_and_counts(predictions):
    bins = calibration_bins(predictions)
    assert bins.groupby("outcome").games.sum().tolist() == [3, 3, 3]
    certain = bins.loc[(bins.outcome == "בית") & (bins.bin == 9)].iloc[0]
    assert certain.predicted == 1 and certain.observed == 1 and certain.games == 1
    assert bins.loc[(bins.outcome == "תיקו") & (bins.bin == 0), "observed"].iloc[0] == 0
    assert len(calibration_chart(bins).data) == 6


def test_argmax_confusion_has_actual_rows_and_predicted_columns(predictions):
    summary, matrix = prediction_diagnostics(predictions)
    assert summary.accuracy.iloc[0] == pytest.approx(2 / 3)
    assert summary.argmax_draws.iloc[0] == 0
    assert matrix.loc["תיקו", "בית"] == 1
    assert matrix.to_numpy().sum() == 3
    assert np.array_equal(confusion_chart(matrix).data[0].z, matrix.to_numpy())


def test_time_curve_connects_half_lives_in_order_even_when_saved_rows_are_ranked():
    stage2 = pd.DataFrame(
        {
            "half_life": [4, np.nan, 2, 1, 4, np.nan, 2, 1],
            "log_loss": [1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 1.4, 1.5],
            "config_id": list("abcdefgh"),
        }
    )
    curve = time_weight_chart(stage2).data[0]
    assert list(curve.x) == ["ללא", "עונה אחת", "2 עונות", "4 עונות"]
    assert list(curve.y) == pytest.approx([1.2, 1.4, 1.3, 1.1])


def test_alignment_denominator_is_all_team_seasons():
    frame = pd.DataFrame({"abs_difference": [0, 1, 2, 4]})
    result = alignment_summary(frame).iloc[0]
    assert result["מקום זהה"] == 0.25
    assert result["עד מקום אחד"] == 0.5
    assert result["עד שני מקומות"] == 0.75


def test_configuration_explains_absence_and_presence_of_extras():
    row = pd.Series(
        dict(
            k=10,
            home_advantage=30,
            season_regression=0.33,
            use_goal_multiplier=True,
            promoted_initialization="fixed_1450",
            dead_game_discount=1,
            half_life=np.nan,
        )
    )
    words = configuration_words(row)
    assert "1450 קבוע" in words and "33%" in words and "במשקל שווה" in words
    row["half_life"] = 4
    row["promoted_initialization"] = "promoted_mean"
    row["use_goal_multiplier"] = False
    words = configuration_words(row)
    assert "4 עונות" in words and "ממוצע עולות" in words and "בלי מכפיל" in words
    html = review_table(
        pd.DataFrame({"label": ["E0050"], WORDS: [words], "log_loss": [1.02]})
    )._repr_html_()
    assert WORDS in html and "E0050" in html


def test_corruption_fails_before_any_data_read(tmp_path, monkeypatch):
    directory = tmp_path / "reports/experiments/EXP-001"
    directory.mkdir(parents=True)
    artifact = directory / "top20.csv"
    artifact.write_text("corrupt")
    manifest = {
        "status": "completed_not_selected",
        "files": {"top20.csv": {"sha256": hashlib.sha256(b"original").hexdigest()}},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("read before verification")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_elo_review(tmp_path)


def test_table_state_verifies_metadata_then_payload_before_loading(tmp_path):
    artifact = tmp_path / "table_state_v2.parquet"
    frame = pd.DataFrame({"season_id": [8], "points": [0]})
    frame.to_parquet(artifact)
    metadata = artifact.with_suffix(".metadata.json")
    values = {
        "version": "2",
        "seasons": list(range(8, 26)),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    metadata.write_text(json.dumps(values))
    expected = hashlib.sha256(metadata.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(
        load_table_state_snapshot(artifact, metadata_sha256=expected), frame
    )
    metadata.write_text(json.dumps({**values, "version": "3"}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_table_state_snapshot(artifact, metadata_sha256=expected)
    metadata.write_text(json.dumps(values))
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_table_state_snapshot(artifact, metadata_sha256=expected)


def test_snapshot_comparison_normalizes_only_index_integer_representation():
    index = pd.MultiIndex.from_arrays(
        [[1, 2], pd.array([7, 8], dtype="Int64")], names=["game_id", "team_id"]
    )
    computed = pd.DataFrame({"points": pd.array([0, 3], dtype="Int64")}, index=index)
    saved = computed.copy()
    saved.index = pd.MultiIndex.from_tuples([(1, 7), (2, 8)], names=index.names)
    assert_same_table_state(computed, saved)
    saved.iloc[0, 0] = 1
    with pytest.raises(AssertionError):
        assert_same_table_state(computed, saved)


def test_saved_controls_when_local_artifacts_exist():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/derived/elo/exp001_predictions_stage1.parquet").exists():
        pytest.skip("Local saved experiment predictions are not distributed in Git")
    review = load_elo_review(root)
    values = control_numbers(review).iloc[0]
    assert values["E0050 log loss"] == pytest.approx(1.020423847325146, abs=1e-12)
    assert values["E0050 RPS"] == pytest.approx(0.2084126718207988, abs=1e-12)
    assert values["B0 log loss"] == pytest.approx(1.0849744118664024, abs=1e-12)
    assert values["מקום זהה (%)"] == pytest.approx(100 * 97 / 252)
    assert values["עד ±1 (%)"] == pytest.approx(100 * 187 / 252)
    assert values["עד ±2 (%)"] == pytest.approx(100 * 226 / 252)
    assert len(review.predictions) == 3721
    probabilities = review.predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    actual = review.predictions.outcome.to_numpy(dtype=int)
    observed = probabilities[np.arange(len(actual)), actual]
    assert -np.log(observed).mean() == pytest.approx(values["E0050 log loss"], abs=1e-12)
    assert review.predictions.rps.mean() == pytest.approx(values["E0050 RPS"], abs=1e-12)
    assert observed.mean() == pytest.approx(0.3877837647658396, abs=1e-12)
