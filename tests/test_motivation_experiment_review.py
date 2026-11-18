"""B2 paired decomposition, saved-artifact views, strict live gates and Hebrew plots."""

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.analysis import motivation_experiment_review as er
from ligat.analysis import motivation_walkthrough as mw
from ligat.evaluate.motivation_experiment import motivation_configurations
from ligat.viz import motivation_experiment as ev
from ligat.viz.motivation_style import verify_rtl_labels
from ligat.viz.seasons import season_label


@pytest.fixture(scope="module")
def experiment_review(real_tables):
    return er.load_experiment_review(Path.cwd(), real_tables)


def test_configuration_labels_are_unique_hebrew_and_grid_parameters_are_total():
    grid = er.configuration_map()
    labels = [er.configuration_label(c.config_id) for c in motivation_configurations()]
    assert len(set(labels)) == len(grid) == 20
    assert all(re.search(r"[\u0590-\u05ff]", label) for label in labels)
    assert grid.set_index("config_id").parameters.to_dict() == {
        c.config_id: 3 + len(c.columns) for c in motivation_configurations()
    }
    assert grid.assumption.str.len().gt(10).all()
    with pytest.raises(ValueError, match="Unknown"):
        er.configuration_label("new")


def test_full_season_labels_and_rtl_table_never_expose_only_internal_season_ids():
    assert season_label(11) == "2009/2010"
    assert season_label(25) == "2023/2024"
    html = ev.table(pd.DataFrame(dict(season_id=[11, 25], config_id=["C0", "Comp-hl4"]))).data
    assert all(word in html for word in ["2009/2010", "2023/2024", "Comp-hl4", "רכיב לכל יעד"])
    assert "season_id" not in html
    assert "&lt;" in ev.prose("<test>").data


def test_groups_are_disjoint_exhaustive_symmetric_and_reject_unknown_levels():
    home = pd.Series([0, 0, 0, 0.5, 1, 0.5, 1, 1, 0.5])
    away = pd.Series([1, 0.5, 0, 1, 1, 0.5, 0, 0.5, 0])
    expected = [
        "dead_alive",
        "dead_almost",
        "both_dead",
        "almost",
        "alive",
        "almost",
        "dead_alive",
        "almost",
        "dead_almost",
    ]
    assert er.l3_groups(home, away).tolist() == expected
    assert er.l3_groups(away, home).tolist() == expected
    for invalid in (np.nan, 0.25, -1):
        with pytest.raises(ValueError, match="shared L3"):
            er.l3_groups(pd.Series([invalid]), pd.Series([1]))
    with pytest.raises(ValueError):
        er.l3_groups(pd.Series([0], index=[1]), pd.Series([1], index=[2]))


def test_heatmap_is_exact_matching_decay_control_minus_candidate():
    folds = pd.DataFrame(
        [
            dict(config_id=c.config_id, season_id=s, log_loss=i + s / 100)
            for i, c in enumerate(motivation_configurations())
            for s in (11, 25)
        ]
    )
    matrix = er.fold_improvements(folds)
    for c in motivation_configurations():
        a = folds.loc[folds.config_id.eq(c.config_id)].set_index("season_id").log_loss
        b = folds.loc[folds.config_id.eq(c.reference_id)].set_index("season_id").log_loss
        np.testing.assert_array_equal(matrix.loc[c.config_id], b - a)


def test_live_validation_gate_checks_values_keys_and_finiteness():
    saved = pd.DataFrame(dict(log_loss=[1.0, 2.0], rps=[0.2, 0.3]), index=[11, 25])
    assert er.validate_live_values(saved.iloc[::-1], saved, ["log_loss", "rps"]) == 0
    actual = saved.copy()
    actual.loc[25, "log_loss"] += 1e-6
    with pytest.raises(AssertionError, match="differs"):
        er.validate_live_values(actual, saved, ["log_loss", "rps"])
    with pytest.raises(AssertionError, match="keys"):
        er.validate_live_values(saved.iloc[:1], saved, ["log_loss"])
    actual.loc[25, "log_loss"] = np.nan
    with pytest.raises(AssertionError, match="differs"):
        er.validate_live_values(actual, saved, ["log_loss"])


def test_nothing_to_play_is_forbidden_in_visible_prose_but_code_is_not_prose():
    notebook = dict(
        cells=[
            dict(cell_type="markdown", source="nothing_to_play"),
            dict(cell_type="code", source="nothing_to_play", outputs=[]),
        ]
    )
    assert [v["cell"] for v in mw.visible_text_violations(notebook)] == [0]


def test_saved_decomposition_all_configurations_sums_and_known_group_values(experiment_review):
    result = er.group_decomposition(experiment_review)
    assert len(result) == 18 * 3 * 2 * 5
    for _, part in result.groupby(["config_id", "period", "metric"]):
        assert set(part.group) == set(er.GROUPS)
        assert part.games.sum() == part.all_games.iloc[0]
    deep = result.loc[result.period.eq("deep")].set_index(["config_id", "metric", "group"])
    assert deep.loc[("L2-diff", "log_loss", "dead_almost"), "games"] == 73
    assert deep.loc[("L2-diff", "log_loss", "dead_almost"), "mean_improvement"] == pytest.approx(
        -0.031, abs=0.001
    )
    assert deep.loc[("L3-diff", "log_loss", "dead_almost"), "mean_improvement"] == pytest.approx(
        0.021, abs=0.001
    )
    totals = (
        result.loc[result.period.eq("deep")].groupby(["config_id", "metric"]).contribution.sum()
    )
    comparisons = (
        experiment_review.frames["comparison"]
        .loc[lambda x: x.scope.eq("deep")]
        .set_index(["config_id", "metric"])
    )
    np.testing.assert_allclose(
        totals, comparisons.mean_improvement.reindex(totals.index), rtol=0, atol=1e-12
    )
    broken = dict(experiment_review.frames)
    broken["comparison"] = broken["comparison"].assign(
        mean_improvement=lambda x: x.mean_improvement + 0.01
    )
    with pytest.raises(AssertionError, match="contributions"):
        er.group_decomposition(replace(experiment_review, frames=broken))


def test_synthetic_known_losses_assign_exact_means_and_contributions(experiment_review):
    context = experiment_review.context.copy()
    ids = context.loc[context.season_id.eq(12), "game_id"].head(5).tolist()
    context["group"] = "alive"
    for game_id, group in zip(ids, er.GROUPS, strict=True):
        context.loc[context.game_id.eq(game_id), "group"] = group
    frames = dict(experiment_review.frames)
    predictions = frames["predictions"].copy()
    for config_id in predictions.config_id.unique():
        predictions.loc[predictions.config_id.eq(config_id), "log_loss"] = 1.0
    for i, game_id in enumerate(ids, 1):
        predictions.loc[
            predictions.config_id.eq("L2-diff") & predictions.game_id.eq(game_id), "log_loss"
        ] = 1.0 - i / 100
    frames["predictions"] = predictions
    result = er.paired_games(replace(experiment_review, context=context, frames=frames), "L2-diff")
    for i, group in enumerate(er.GROUPS, 1):
        values = result.loc[
            result.game_id.isin(ids) & result.group.eq(group), "log_loss_improvement"
        ]
        assert len(values) == 1
        assert values.iloc[0] == pytest.approx(i / 100)


def test_saved_rankings_results_points_extremes_and_definition_audit(experiment_review):
    assert er.ranking_check(experiment_review).groupby("scope").size().eq(5).all()
    altered = dict(experiment_review.decision)
    altered["rule6"] = False
    with pytest.raises(AssertionError):
        er.ranking_check(replace(experiment_review, decision=altered))
    for metric in ("log_loss", "rps"):
        result = er.result_table(experiment_review, metric)
        assert len(result) == 20
        assert not result[f"{metric}_passes_rule8"].any()
        np.testing.assert_allclose(
            result.geometric_all15_pct,
            100 * np.exp(-experiment_review.frames["summary"].all15_log_loss),
        )
    involved = er.involved_results(experiment_review)
    assert len(involved) == 20 and involved.games.eq(845).all()
    extremes = er.extreme_games(experiment_review)
    assert len(extremes) == 20 and extremes.groupby("extreme").size().eq(10).all()
    assert extremes.home_team_name.notna().all()
    assert extremes.actual_probability.gt(0).all()
    points = er.point_effects(experiment_review)
    assert points.elo_diff.eq(0).all()
    assert points.config_id.nunique() == 20
    assert points.loc[points.config_id.isin(["C0", "C0-hl4"])].expected_points_change.eq(0).all()
    diff = er.definition_comparison(experiment_review)
    assert diff.season_id.value_counts().to_dict() == {12: 9, 11: 2}
    assert er.numerical_checks(experiment_review).matches.all()
    assert er.entry_counts(experiment_review.frames["summary"]) == {
        "log_loss": 4,
        "rps": 7,
        "either": 7,
    }
    story = er.decomposition_findings(er.group_decomposition(experiment_review))
    assert all(value in story for value in ("211", "-0.008", "+0.005", "2019/2020"))


def test_actual_live_three_configurations_match_saved_coefficients_and_scores(experiment_review):
    result = er.live_validation(experiment_review)
    assert set(result.config_id) == {"C0", "L2-diff", "L3-diff-hl4"}
    assert result.folds.eq(15).all()
    assert result.loss_error.le(1e-10).all() and result.coefficient_error.le(1e-10).all()


def test_all_new_plots_have_rtl_hebrew_legends_annotations_and_season_axes(experiment_review):
    decomposition = er.group_decomposition(experiment_review)
    figures = [
        ev.forest(experiment_review),
        ev.heatmap(experiment_review),
        ev.recent_scatter(experiment_review),
        ev.group_contributions(decomposition),
        ev.coefficient_paths(experiment_review),
        ev.placebo(experiment_review),
    ]
    for figure in figures:
        verify_rtl_labels(figure.to_plotly_json())
        text = json.dumps(figure.to_plotly_json(), ensure_ascii=False, default=str)
        assert "\u200e" not in text and "\u200f" not in text
    forest = figures[0]
    assert "הפס: פעמיים שגיאת התקן" in forest.layout.legend.title.text
    for trace, metric in zip(forest.data, ("log_loss", "rps"), strict=True):
        np.testing.assert_allclose(
            trace.error_x.array, 2000 * experiment_review.frames["summary"].iloc[1:][f"{metric}_se"]
        )
        expected = experiment_review.frames["summary"].iloc[1:]
        assert (
            sum(s == "circle" for s in trace.marker.symbol)
            == (expected[f"{metric}_mean_improvement"] > expected[f"{metric}_se"]).sum()
        )
    matrix = er.fold_improvements(experiment_review.frames["fold_scores"])
    np.testing.assert_array_equal(figures[1].data[0].z, 1000 * matrix.to_numpy())
    assert "2009/2010" in " ".join(figures[1].layout.xaxis.ticktext)
    scatter = figures[2]
    coordinates = experiment_review.frames["summary"].set_index("config_id")
    for annotation in scatter.layout.annotations:
        assert annotation.x == 1000 * coordinates.loc[annotation.text, "log_loss_mean_improvement"]
        assert annotation.y == 1000 * coordinates.loc[annotation.text, "log_loss_last5_improvement"]


@pytest.mark.real_data
def test_notebook_fixed_quote_preserved_references_remapped_and_no_cells_deleted():
    import subprocess

    path = "notebooks/explore/04_motivation.ipynb"
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    original = json.loads(subprocess.check_output(["git", "show", "ff58cef:" + path]))
    before = original["cells"]
    cells = {c["id"]: c for c in notebook["cells"]}
    assert set(c["id"] for c in before) <= set(cells)
    assert "".join(cells[before[81]["id"]]["source"]) == "".join(before[81]["source"])
    for c in notebook["cells"]:
        if c["cell_type"] != "markdown" or c["id"] == before[81]["id"]:
            continue
        text = "".join(c["source"])
        assert not re.search(r"6ב(?:\.|\s*[—–])|6 — נספח|7 — סיכום חלק A", text)
    assert not mw.visible_text_violations(notebook)
