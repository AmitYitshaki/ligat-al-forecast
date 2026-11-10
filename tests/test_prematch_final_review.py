"""C9–C14 controls, mirrored examples, descriptive ranges and honest plot scales."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis import prematch_final_review as final
from ligat.analysis.prematch_final_review import (
    WORDS,
    analyze_final_prematch,
    check_c5_reference,
    column_groups,
    cross_family_pairs,
    dead_game_summary,
    final_controls,
    load_final_prematch,
    m0_deliverables,
    match_example,
    missingness,
    outcome_statistics,
    overlap_checks,
    planning_decisions,
    target_leaders,
    target_trends,
)
from ligat.analysis.prepare import load_analysis_tables
from ligat.viz.prematch_final import (
    contrast_heatmap,
    cramers_chart,
    example_table,
    final_table,
    outcome_chart,
    save_final_figures,
    target_comparison_chart,
    trend_description,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checksum_failure_precedes_any_parquet_read(tmp_path, monkeypatch):
    path = tmp_path / "data/derived/prematch/prematch_v1.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "component": "prematch",
                "version": "1",
                "artifact_sha256": "wrong",
                "inputs": {},
                "config_sha256": "config",
            }
        )
    )
    monkeypatch.setattr(
        final,
        "load_prematch_config",
        lambda p: SimpleNamespace(inputs={}, sha256="config", input_records=lambda: {}),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Must verify checksum before reading data")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="checksum/provenance"):
        load_final_prematch(tmp_path)


@pytest.mark.parametrize("season", [26, 27, 28])
def test_new_statistics_cannot_use_non_development_tables(season):
    tables = SimpleNamespace(match_table=pd.DataFrame({"season_id": [season]}))
    with pytest.raises(ValueError, match="8–25"):
        analyze_final_prematch(pd.DataFrame(), tables, "unused")


def test_column_groups_and_decision_summary_match_the_document_contract():
    groups = column_groups()
    assert groups["columns"].tolist() == [30, 4, 3, 2, 8, 4]
    assert groups["columns"].sum() == 51 and WORDS in groups
    decisions = planning_decisions()
    assert decisions.decision.tolist() == [f"ה־ת{i}" for i in range(1, 15)]
    assert decisions.source.eq("docs/VISION.md §2.4").all()
    assert "בחריגה" in decisions.iloc[-1][WORDS]
    assert WORDS in m0_deliverables()


def test_signed_heatmaps_share_diverging_zero_centered_scale():
    columns = ["prev_gd_diff", "elo_diff"]
    matrix = pd.DataFrame([[1, -0.8], [-0.8, 1]], columns=columns, index=columns)
    figure = contrast_heatmap({"pearson_all": matrix, "pearson_last5": matrix / 2})
    assert all(trace.type == "heatmap" and trace.coloraxis == "coloraxis" for trace in figure.data)
    assert (
        figure.layout.coloraxis.cmin,
        figure.layout.coloraxis.cmid,
        figure.layout.coloraxis.cmax,
    ) == (-1, 0, 1)
    assert figure.layout.coloraxis.colorscale[5][1] == "rgb(247,247,247)"
    assert np.array(figure.data[1].z) == pytest.approx(matrix.to_numpy() / 2)


def test_comparison_uses_points_and_preserves_both_windows():
    frame = pd.DataFrame(
        {
            "feature": ["elo_diff", "value_avg_log_ratio"],
            "pearson": [0.38, 0.31],
            "pearson_last5": [0.42, 0.39],
            "change": [0.04, 0.08],
        }
    )
    figure = target_comparison_chart(frame)
    assert all(trace.type == "scatter" for trace in figure.data)
    assert list(figure.data[0].x) == [0.38, 0.31]
    assert list(figure.data[1].x) == [0.42, 0.39]
    assert figure.data[0].marker.color != figure.data[1].marker.color


def test_explanation_computes_both_elo_changes_and_ratios_without_rounded_endpoints():
    frame = pd.DataFrame(
        {
            "feature": ["value_avg_log_ratio", "elo_expected", "elo_diff"],
            "change": [0.071275, 0.025248, 0.037278],
        }
    )
    html = trend_description(frame)
    for text in ("+0.071", "+0.025", "+0.037", "1.9–2.8", "לא הוכחה"):
        assert text in html
    changed = frame.copy()
    changed.loc[0, "change"] = 0.04
    assert "+0.040" in trend_description(changed)


def test_cramers_bars_start_at_zero_and_warning_keeps_series_color():
    statistics = pd.DataFrame(
        {
            "feature": ["elo_diff", "stage"],
            "cramers_v": [0.22, 0.04],
            "excluded_reason": ["", ""],
            "expected_below_5": [False, True],
            "n": [4315, 4315],
            "min_expected": [200, 2],
        }
    )
    figure = cramers_chart(statistics)
    assert figure.layout.xaxis.range[0] == 0
    assert list(figure.data[0].x) == [0.04, 0.22]
    assert list(figure.data[0].marker.line.width) == [2, 0]
    assert isinstance(figure.data[0].marker.color, str)
    assert figure.data[0].text[0].endswith(" *")


def test_presentation_rejects_opaque_tables_and_old_export_filenames(tmp_path):
    with pytest.raises(ValueError, match="explanatory words"):
        final_table(pd.DataFrame({"feature": ["elo_diff"]}))
    with pytest.raises(ValueError, match="c6b_ prefix"):
        save_final_figures({"context_correlations": object()}, tmp_path)
    html = final_table(pd.DataFrame({"value": [-0.05], WORDS: ["פער תיאורי"]}))._repr_html_()
    assert "−0.05000" in html and 'dir="rtl"' in html and 'dir="ltr"' in html


@pytest.fixture(scope="module")
def frozen():
    if not (ROOT / "data/derived/prematch/prematch_v1.parquet").exists():
        pytest.skip("Local M0 artifacts unavailable")
    return load_final_prematch(ROOT)


@pytest.fixture(scope="module")
def saved(frozen):
    frame, digest = frozen
    tables = load_analysis_tables(seasons=list(range(8, 26)))
    return analyze_final_prematch(frame, tables, digest)


def test_example_uses_yaml_names_and_mirrors_contrasts_and_opponent_attack(frozen):
    frame, _ = frozen
    before = frame.copy(deep=True)
    example = match_example(frame, ROOT / "config/teams.yaml").set_index("feature")
    assert example.home_name.nunique() == example.away_name.nunique() == 1
    for name in ("elo_diff", *final.PREMATCH_CONTRAST_COLUMNS):
        assert example.at[name, "home"] == pytest.approx(-example.at[name, "away"])
    assert example.at["gf_venue_ewma5", "home"] == example.at["opp_gf_venue_ewma5", "away"]
    assert "בית: " in example_table(example.reset_index())._repr_html_()
    assert_frame_equal(frame, before)


def test_prior_missingness_and_verified_overlap_exceptions_are_not_imputed(frozen):
    frame, _ = frozen
    missing = missingness(frame)
    prior = missing.loc[~missing.feature.str.startswith("gap_to_")]
    assert len(prior) == 8
    assert prior.missing.eq(396).all() and prior.missing_later.eq(0).all()
    overlaps = overlap_checks(frame)
    assert overlaps.value.iloc[:3].tolist() == [1, 0, 35]
    assert pd.isna(overlaps.value.iloc[3])
    with pytest.raises(ValueError, match="overlap reference"):
        overlap_checks(frame.assign(rounds_to_stage_end=frame.rounds_to_stage_end + 1))


def test_recomputed_statistics_match_c5_and_all_control_anchors(saved):
    check_c5_reference(saved, ROOT / "data/reports/nb03_C5_correlations.json")
    assert final_controls(saved).passed.all()
    assert len(cross_family_pairs(saved)) == 9
    dead = dead_game_summary(saved)
    assert dead.games.tolist() == [174, 160] and dead.union_games.eq(245).all()
    assert dead.mean_elo_diff.tolist() == pytest.approx([5.917978, -13.762122], abs=1e-6)
    assert dead.draw_percent.tolist() == pytest.approx([32 / 174 * 100, 29 / 160 * 100])


def test_seasonal_range_and_target_ranking_are_descriptive_only(saved):
    leaders = target_leaders(saved, "goal_difference_game", 10)
    assert len(leaders) == 10 and leaders.pearson.abs().is_monotonic_decreasing
    assert "p_value" not in leaders
    feature = leaders.feature.iloc[0]
    # The displayed endpoints must bracket C5's independently retained median.
    frame = saved.frame
    seasonal = saved.associations.loc[saved.associations.feature.eq(feature)]
    assert seasonal.pearson_min_season.le(seasonal.median_season_pearson).all()
    assert seasonal.pearson_max_season.ge(seasonal.median_season_pearson).all()
    recent = frame.loc[frame.season_id.ge(21)]
    assert len(recent) == 2396
    trends = target_trends(saved)
    assert (trends.pearson_last5 - trends.pearson).to_numpy() == pytest.approx(trends.change)
    bad = replace(saved, associations=saved.associations.assign(pearson=0.7))
    with pytest.raises(ValueError, match="control discrepancy"):
        final_controls(bad)


def test_outcome_stacks_preserve_sums_reference_lines_and_excluded_rows(saved):
    excluded = outcome_statistics(saved, excluded=True)
    assert excluded.feature.tolist() == ["is_home"]
    assert excluded.cramers_v.isna().all()
    figure = outcome_chart(saved.outcomes, ("nothing_to_play", "away_nothing_to_play"), "בדיקה")
    assert len(figure.data) == 6 and len(figure.layout.shapes) == 4
    for start in (0, 3):
        heights = np.array([trace.y for trace in figure.data[start : start + 3]])
        assert heights.sum(axis=0) == pytest.approx([100, 100])
    assert len(set(trace.marker.color for trace in figure.data)) == 3
    assert figure.layout.yaxis.range == (0, 100)


def test_approved_b1_through_c8_sources_remain_unchanged():
    notebook = json.loads((ROOT / "notebooks/explore/03_prematch.ipynb").read_text("utf8"))
    cells = notebook["cells"]
    start = next(
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and "<h2>B1 — Elo ו־EXP-001" in "".join(cell["source"])
    )
    guarded = cells[start : start + 145]
    assert len(guarded) == 145
    prefix = [{"type": c["cell_type"], "source": "".join(c["source"])} for c in guarded]
    digest = hashlib.sha256(
        json.dumps(prefix, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    # HEAD 0036c7e cells 30–174, with only the approved 0–119 → 5–124 reference in C8.
    assert digest == "44a30fb47bdc7ad9fac6aac77aef290ecb81e0068d0185a54029b849a76c9b06"
