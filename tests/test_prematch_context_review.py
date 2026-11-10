"""Saved-only C review: independent pairing, depth, season guards and displayed units."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prematch_context_review import (
    CANDIDATES,
    METRICS,
    WORDS,
    ContextReview,
    added_results,
    alone_results,
    context_associations,
    context_correlations,
    context_schema,
    control_numbers,
    direct_decisions,
    experiment_design,
    gd_translation,
    load_context_review,
    probe_reference,
    promotion_check,
    reference_ruler,
    selected_coefficients,
    sensitivity_overview,
    team_season_context,
    training_depth_table,
    validate_context_controls,
    value_fold_gaps,
    value_methodology_tables,
)
from ligat.analysis.prepare import load_analysis_tables
from ligat.features.registry import SEASON_CONTEXT_COLUMNS
from ligat.viz.league_stats import DRAW_COLOR
from ligat.viz.prematch_context import (
    added_chart,
    alone_chart,
    context_heatmap,
    context_matrix_table,
    context_table,
    depth_display,
    distribution_chart,
    save_context_figures,
    value_comparison_chart,
    value_fold_chart,
)
from ligat.viz.seasons import season_label

ROOT = Path(__file__).resolve().parents[1]


def synthetic_review():
    seasons = np.arange(11, 26)
    weights = np.arange(1, 16) * 10
    differences = np.array([0.02, -0.01, 0.03] * 5)
    frames = []
    fits = []
    for route in ("A", "O", "P"):
        reference = route + {"A": "_b0", "O": "_elo", "P": "_poisson"}[route]
        for config in [reference, *[route + "_" + c for c in CANDIDATES]]:
            delta = 0 if config == reference else differences
            frames.append(
                pd.DataFrame(
                    dict(
                        config_id=config,
                        season_id=seasons,
                        games=weights,
                        rows=weights * 2,
                        log_loss=1.1 - delta,
                        poisson_nll=1.5 - delta,
                    )
                )
            )
            fits.extend(
                dict(config_id=config, season_id=s, training_seasons=list(range(9, s)))
                for s in seasons
            )
    return (
        ContextReview(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.concat(frames, ignore_index=True),
            pd.DataFrame(fits),
            pd.DataFrame(),
        ),
        differences,
        weights,
    )


def test_saved_pairing_uses_weighted_mean_and_unweighted_paired_se():
    review, differences, weights = synthetic_review()
    before = review.folds.copy(deep=True)
    result = added_results(review)
    assert len(result) == 14 and WORDS in result
    assert result.improvement.to_numpy() == pytest.approx(np.average(differences, weights=weights))
    assert result.se.to_numpy() == pytest.approx(differences.std(ddof=1) / np.sqrt(15))
    assert result.better.eq(10).all() and result.folds.eq(15).all()
    assert result.recent_improvement.to_numpy() == pytest.approx(
        np.average(differences[-5:], weights=weights[-5:])
    )
    assert_frame_equal(review.folds, before)


@pytest.mark.parametrize("column", ["games", "rows", "season_id"])
def test_pairing_rejects_different_validation_populations(column):
    review, _, _ = synthetic_review()
    changed = review.folds.copy()
    index = changed.index[changed.config_id.eq("P_value_avg")][0]
    changed.loc[index, column] += 1
    with pytest.raises(ValueError, match="identical validation|nonunique saved folds"):
        added_results(replace(review, folds=changed))


def test_training_depth_filters_actual_fit_history_and_keeps_recent_descriptive():
    review, differences, weights = synthetic_review()
    result = training_depth_table(review)
    assert len(result) == 6
    for minimum, count in [(2, 15), (4, 13), (6, 11), (8, 9)]:
        offset = 15 - count
        assert result[f"folds_{minimum}"].eq(count).all()
        assert result[f"mean_{minimum}"].to_numpy() == pytest.approx(
            np.average(differences[offset:], weights=weights[offset:])
        )
        assert result[f"se_{minimum}"].to_numpy() == pytest.approx(
            differences[offset:].std(ddof=1) / np.sqrt(count)
        )
    assert "adds_last5" not in result


def test_rule6_compares_last_five_score_without_a_last_five_se_threshold():
    review, _, _ = synthetic_review()
    folds = review.folds.copy()
    mask = folds.config_id.eq("O_value_total")
    # Tiny, consistent recent advantage; large early losses make 15-fold rule fail.
    folds.loc[mask, "log_loss"] = (
        folds.loc[folds.config_id.eq("O_value_avg"), "log_loss"].to_numpy()
        + np.r_[np.full(10, 0.05), np.full(5, -0.00001)]
    )
    result = direct_decisions(replace(review, folds=folds))
    row = result.loc[result.route.eq("O") & result.method.str.contains("שווי")].iloc[0]
    assert not row.condition and row.recent_improvement > 0 and row.disagreement


@pytest.mark.parametrize("season", [26, 27, 28])
def test_new_context_diagnostics_reject_non_development_seasons(season):
    frame = pd.DataFrame({"season_id": [season]})
    for function in (context_associations, context_correlations, promotion_check):
        with pytest.raises(ValueError, match="8–25"):
            function(frame)


def test_schema_and_probe_are_static_and_need_no_source_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Source reads are forbidden for quoted probe results")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    schema, probe = context_schema(), probe_reference()
    assert tuple(schema.column) == SEASON_CONTEXT_COLUMNS and len(schema) == 9
    assert WORDS in schema and WORDS in probe
    assert probe.clubs.tolist() == [11, 11]
    assert probe.spearman.tolist() == [0.945455, 0.772727]
    assert probe.source.str.contains("C1_value_check").all()


def test_added_chart_scales_both_mean_and_se_and_does_not_call_se_a_ci():
    review, _, _ = synthetic_review()
    frame = added_results(review)
    chart = added_chart(frame, "P")
    rows = frame.loc[frame.route.eq("P")]
    assert np.array(chart.data[0].x) == pytest.approx(rows.improvement * 1000)
    assert np.array(chart.data[0].error_x.array) == pytest.approx(rows.se * 1000)
    title = chart.layout.title.text.replace("\u202a", "").replace("\u202c", "")
    assert "±SE" in title and "Poisson NLL" in chart.layout.xaxis.title.text
    assert chart.layout.shapes[0].x0 == chart.layout.shapes[0].x1 == 0


def test_alone_chart_uses_each_windows_own_reference_lines():
    review, _, _ = synthetic_review()
    references = reference_ruler(review)
    references.loc[references.method.eq("B0"), "recent_score"] = 1.08
    references.loc[references.method.eq("Elo"), "recent_score"] = 1.01
    chart = alone_chart(alone_results(review), references)
    assert [shape.x0 for shape in chart.layout.shapes] == pytest.approx([1.1, 1.1, 1.08, 1.01])


def test_fold_plot_highlights_only_2009_10_and_sorts_season_categories():
    review, _, _ = synthetic_review()
    chart = value_fold_chart(value_fold_gaps(review))
    assert list(chart.layout.xaxis.categoryarray) == [season_label(s) for s in range(11, 26)]
    assert len(chart.data) == 4
    for full, recent in ((chart.data[0], chart.data[1]), (chart.data[2], chart.data[3])):
        assert full.marker.color == recent.marker.color != DRAW_COLOR
        assert list(full.marker.line.width) == [3] + [0] * 14
        assert all(width == 0 for width in recent.marker.line.width)
        assert season_label(11) not in recent.x


def test_tables_require_words_and_isolate_signed_numbers_and_full_season_labels():
    with pytest.raises(ValueError, match="words"):
        context_table(pd.DataFrame({"value": [1]}))
    html = context_table(
        pd.DataFrame({"season_id": [11], "improvement": [-0.00124], WORDS: ["בדיקה"]})
    )._repr_html_()
    assert 'dir="rtl"' in html and 'dir="ltr"' in html
    assert "−0.001240" in html and "2009/2010" in html


def test_exports_cannot_write_into_data_directories(tmp_path):
    with pytest.raises(ValueError, match="reports/figures"):
        save_context_figures({}, tmp_path / "data")


def test_notebook_keeps_approved_b1_b2_sources_and_explains_every_new_code_cell():
    notebook = json.loads((ROOT / "notebooks/explore/03_prematch.ipynb").read_text("utf-8"))
    cells = notebook["cells"]
    start = next(
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and "<h2>B1 — Elo ו־EXP-001" in "".join(cell["source"])
    )
    guarded = cells[start : start + 96]
    assert len(guarded) == 96
    prefix = [{"type": c["cell_type"], "source": "".join(c["source"])} for c in guarded]
    digest = hashlib.sha256(
        json.dumps(prefix, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    # R4: approved C edits; source guard passed against baseline b4fbf1c.
    assert digest == "204627f3d80019df8f3ebb2b1f85bd3084606b89b5ad476499fbe0015d96315a"
    explanation_start = next(
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "markdown"
        and "<h2>חלק C — מהעונה הקודמת אל המידע שלפני השריקה" in "".join(cell["source"])
    )
    for i, cell in enumerate(cells[explanation_start:], explanation_start):
        if cell["cell_type"] == "code":
            assert cells[i - 1]["cell_type"] == "markdown"
            explanation = "".join(cells[i - 1]["source"])
            for label in ("מה התא עושה", "מה רואים", "הסבר נתונים", "הסבר כדורגלני"):
                assert label in explanation


@pytest.fixture(scope="module")
def saved():
    if not (ROOT / "data/derived/prematch/season_context_v1.parquet").exists():
        pytest.skip("Local frozen C artifacts are unavailable")
    review = load_context_review(ROOT)
    tables = load_analysis_tables(seasons=list(range(8, 26)))
    return review, tables


def test_actual_c1_c2_controls_anomaly_rank_and_promotions(saved):
    review, tables = saved
    team_seasons = team_season_context(review, tables)
    assert len(team_seasons) == 252 and team_seasons.prev_season_known.sum() == 240
    assert validate_context_controls(team_seasons).passed.all()
    results = value_methodology_tables(tables)
    assert results["paired_summary"].iloc[0]["mean"] == pytest.approx(0.05794247458872742)
    anomaly = results["anomaly"]
    hapoel = anomaly.loc[anomaly.season_id.eq(10) & anomaly.team_name.str.contains("הפועל")]
    assert hapoel.value_rank.tolist() == [7]
    assert context_matrix_table(context_correlations(team_seasons))._repr_html_()
    assert len(distribution_chart(team_seasons).data) == 10
    heatmap = context_heatmap(context_correlations(team_seasons))
    assert heatmap.data[0].zmin == -1 and heatmap.data[0].zmax == 1
    assert len(value_comparison_chart(results["paired"]).data[0].x) == 17


def test_actual_exp003_controls_sensitivity_and_saved_coefficients(saved):
    review, _ = saved
    assert control_numbers(review).passed.all()
    sensitivity = sensitivity_overview(review)
    assert sensitivity.first_season.eq(12).all() and sensitivity.first_training.eq(2).all()
    assert sensitivity.first_improvement.lt(0).all()
    coefficients = selected_coefficients(review)
    assert len(coefficients) == 16
    assert set(coefficients.season_id) == {11, 12, 18, 25}
    assert not direct_decisions(review).disagreement.any()
    depth = training_depth_table(review)
    row = depth.loc[depth.route.eq("P") & depth.candidate.eq("value_avg")].iloc[0]
    # EXPERIMENTS reports +0.61 thousandths; the saved fits retain full precision.
    assert row.mean_4 == pytest.approx(0.00061, abs=5e-6)
    assert row.mean_8 == pytest.approx(0.000186, abs=5e-7)
    assert WORDS in depth_display(depth)._repr_html_()
    translation = gd_translation(review)
    assert translation.home_win_low.tolist() == pytest.approx([33.7339265, 39.7351777])
    assert translation.home_win_high.tolist() == pytest.approx([48.4657894, 41.1240023])


def test_design_describes_all_three_routes_and_all_seven_candidates():
    routes, candidates = experiment_design()
    assert routes.route.tolist() == ["A", "O", "P"]
    assert candidates.candidate.tolist() == list(CANDIDATES)
    assert WORDS in routes and WORDS in candidates
    assert set(METRICS) <= set(SEASON_CONTEXT_COLUMNS)
