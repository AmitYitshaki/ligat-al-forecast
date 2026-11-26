"""Walkthrough graphs read signed evidence and never train or retune models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_runner import load_goal_inputs
from ligat.viz import goal_training_curves as curves
from ligat.viz import goals_configuration_names as naming
from ligat.viz import goals_qa as qa
from ligat.viz import goals_qa_graphs as graphs

pytestmark = [pytest.mark.slow, pytest.mark.real_data]

NAMES = [
    "training_selector",
    "train_vs_validation_heatmap",
    "hyperparameter_heatmap",
    "learning_curves_train_vs_validation",
    "paired_folds",
    "ladder_progress",
    "base_family_forest",
    "paired_forest",
    "metric_by_season",
    "win_rate_by_season",
    "models_vs_baselines",
    "elo_by_scope",
    "open_reviews",
    "post_hoc_forest",
    "correlation_scatter",
    "hda_histogram",
    "worker_bars",
    "hyperparameter_effects",
    "season_decay",
    "daily_decay",
]


def test_selectors_update_explanations_and_put_proposal_first(signed_graphs):
    import re

    import plotly.graph_objects as go

    b, originals = signed_graphs
    keys = ["training_selector", "train_vs_validation_heatmap", "hyperparameter_heatmap"]
    figures = {key: go.Figure(originals[key]) for key in keys}
    naming.selectors(figures, b)
    forbidden = re.compile(r"d\d+ ?/|\bT\d+\b|lr0\.|mcw\d|cw\d+\.|L2=|proposal|\bn\d+ ·")
    for figure in figures.values():
        menu = figure.layout.updatemenus[0]
        assert menu.active == 0 and menu.buttons[0].label.startswith("★ ההצעה")
        for button in menu.buttons:
            assert button.method == "update"
            assert "annotations" in button.args[1]
            assert "התצורה שנבחרה" in button.args[1]["annotations"][-1]["text"]
            assert not forbidden.search(button.label)
    for figure in originals.values():
        translated = naming.figure_labels(figure)
        visible = str(translated.layout.to_plotly_json())
        assert not forbidden.search(visible)


def test_right_y_titles_are_empty_while_left_titles_survive_finalization():
    rows = [
        dict(
            unit_id=f"d{depth}-c{caution}-f{fold}",
            stage_role="S2-A",
            preset="grid",
            depth=depth,
            min_child_weight=caution,
            n_estimators=150,
            learning_rate=0.05,
            games=100,
            poisson_nll=1.4 + fold * 0.002 + depth * 0.005,
            iteration=150,
            train_loss=1.25 + depth * 0.01 + caution * 0.0001,
        )
        for depth in [2, 3]
        for caution in [1, 30]
        for fold in range(1, 16)
    ]
    frame = pd.DataFrame(rows)
    b = dict(
        training=frame,
        validation=frame.drop(columns=["train_loss", "iteration"]),
        report=dict(
            winners={"S2": {"parameters": {"n_estimators": 150, "learning_rate": 0.05}}},
            readable_metrics={
                "R0": {"summary": [{"scope": "all", "poisson_nll": 1.5}]},
                "R1": {"summary": [{"scope": "all", "poisson_nll": 1.4}]},
            },
        ),
    )
    heatmap = graphs.slice_maps(b)["train_vs_validation_heatmap"]
    assert heatmap.layout.yaxis.title.text
    assert not heatmap.layout.yaxis2.title.text
    assert heatmap.layout.yaxis2.anchor == "x2"
    assert heatmap.layout.xaxis2.domain[0] > heatmap.layout.xaxis.domain[1]

    outputs = [
        dict(
            identity=dict(config=dict(max_depth=depth, learning_rate=rate, n_estimators=3)),
            train_nll=[1.45, 1.42, 1.4 + fold * 0.0001],
            validation_nll=[1.46, 1.44, 1.43 + fold * 0.0001],
        )
        for depth in [2, 3]
        for rate in [0.05, 0.1]
        for fold in range(15)
    ]
    learning = graphs.learning_curves(b, dict(outputs=outputs, selected_config={"n_estimators": 3}))
    for left, right in [("yaxis", "yaxis2"), ("yaxis3", "yaxis4")]:
        assert learning.layout[left].title.text
        assert not learning.layout[right].title.text
    assert learning.layout.yaxis2.anchor == "x2"
    assert learning.layout.yaxis4.anchor == "x4"


@pytest.fixture(scope="module")
def signed_graphs(real_tables):
    if not Path("data/derived/exp005/diagnostics/post_hoc/report.json").exists():
        pytest.skip("Signed local diagnostics are not committed")
    spec = load_goal_experiment()

    def forbidden(*args, **kwargs):
        pytest.fail("Notebook graph called fit")

    with pytest.MonkeyPatch.context() as patch:
        for model in [
            "m2_xgb.XGBGoalModel",
            "m2_catboost.CatBoostGoalModel",
            "m2_glm.GLMGoalModel",
            "m2_maher.MaherGoalModel",
        ]:
            # The native classes have intentionally remained frozen.
            patch.setattr("ligat.models." + model + ".fit", forbidden)
        data = load_goal_inputs(spec, tables=real_tables)
        patch.setattr("ligat.viz.goals_notebook.development_data", lambda: data)
        b = graphs.bundle(spec)
        learning = curves.load_learning_diagnostics(spec)
        yield b, graphs.all_figures(b, learning)


@pytest.mark.parametrize("name", NAMES)
def test_each_walk_graph_uses_verified_evidence_without_fit(signed_graphs, name):
    b, figures = signed_graphs
    figure = figures[name]
    assert len(figure.data) > 0
    assert figure.layout.title.text
    assert graphs.graph_explanation(name, figure, after=False).startswith('<div dir="rtl">')
    assert graphs.graph_explanation(name, figure, after=True).startswith('<div dir="rtl">')
    for after in [False, True]:
        paragraph = graphs.explain_graph(name, b, figure, after=after)
        assert len(graphs.plain_explanation(paragraph)) >= 250
    assert all(chr(c) not in figure.to_json() for c in (0x200E, 0x200F))
    assert b["post"]["units"] == 210
    if name.endswith("decay"):
        labels = " ".join(trace.name for trace in figure.data)
        labels = "".join(c for c in labels if not 0x202A <= ord(c) <= 0x202E)
        assert "עונה אחת" in labels
        assert "מחצית חיים" not in labels
    if name == "hyperparameter_effects":
        assert figure.data[7].visible is False  # colsample is constant in the grid.


def test_learning_curve_axes_share_one_complete_range(signed_graphs):
    _, figures = signed_graphs
    figure = figures["learning_curves_train_vs_validation"]
    axes = [figure.layout[name] for name in figure.layout if name.startswith("yaxis")]
    assert len(axes) == 4
    assert all(axis.range == axes[0].range for axis in axes)
    assert all(figure.layout[name].matches == "y" for name in ["yaxis2", "yaxis3", "yaxis4"])
    values = np.concatenate([graphs._array(trace.y) for trace in figure.data])
    assert axes[0].range[0] <= values.min() <= values.max() <= axes[0].range[1]


def test_ladder_uses_family_entry_not_cumulative_retuning(signed_graphs):
    b, figures = signed_graphs
    entries, _ = qa.entry_tables(b["report"])
    scale = 100 / graphs._elo_denominator(b["report"])
    for trace in figures["ladder_progress"].data:
        name = trace.y[0]
        expected = entries.loc[entries.name.eq(name)].iloc[0]
        np.testing.assert_allclose(graphs._array(trace.x), [expected.d * scale], atol=1e-12)
        np.testing.assert_allclose(
            graphs._array(trace.error_x.array), [(expected.upper - expected.d) * scale], atol=1e-12
        )
    assert set(t.y[0] for t in figures["ladder_progress"].data) == {
        "F1-R",
        "F1-A",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
    }


def test_selectors_cover_full_grid_and_default_to_frozen_proposal(signed_graphs):
    b, figures = signed_graphs
    selector = figures["training_selector"]
    assert len(selector.layout.updatemenus[0].buttons) == 90
    assert sum(t.visible is True for t in selector.data) == 4
    active = selector.layout.updatemenus[0].active
    config = b["report"]["winners"]["S2"]["parameters"]
    label = selector.layout.updatemenus[0].buttons[active].label
    assert f"d{config['max_depth']}" in label
    assert f"T{config['n_estimators']}" in label
    for name in ["train_vs_validation_heatmap", "hyperparameter_heatmap"]:
        menu = figures[name].layout.updatemenus[0]
        assert len(menu.buttons) == 6
        assert "proposal" in menu.buttons[menu.active].label


@pytest.mark.parametrize("check,configurations", [("P1", 1), ("P2", 1), ("P3", 8), ("P4", 4)])
def test_post_hoc_tables_keep_every_configuration_scope_and_reference(
    signed_graphs, check, configurations
):
    b, _ = signed_graphs
    table, introduction, explanation = graphs.post_hoc_sections(b, check)
    assert len(table) == configurations * 3 * 3
    assert set(table.reference) == {"S2", "EXP-002", "F5"}
    assert table.groupby("scope").n.unique().map(list).to_dict() == {
        "all": [15],
        "deep": [14],
        "last5": [5],
    }
    assert len(graphs.plain_explanation(introduction)) >= 250
    assert len(graphs.plain_explanation(explanation)) >= 250
    assert "S2" in explanation
    assert "נהנית גם ממזל" not in explanation


def test_scoped_diagnostics_include_shallow_fold_only_in_all_without_mutating_depth(monkeypatch):
    baseline = pd.DataFrame(
        [
            dict(
                fold=s - 10,
                season_id=s,
                games=100,
                deep_fold=s > 11,
                poisson_nll=2.0 if s == 11 else 1.6,
            )
            for s in range(11, 26)
        ]
    )
    candidate = baseline.assign(poisson_nll=1.5)
    original = baseline.copy(deep=True)
    record = dict(
        check="P1",
        features=["is_home"],
        parameters=dict(max_depth=2, n_estimators=150, learning_rate=0.05, min_child_weight=1),
        folds=candidate.to_dict("records"),
        summary=[
            dict(
                scope=scope,
                poisson_nll=1.5,
                observed_probability_percent=20.0,
                elo_share_percent=90.0,
                log_loss=1.1,
            )
            for scope in ["all", "deep", "last5"]
        ],
    )
    monkeypatch.setattr(graphs, "_selected_folds", lambda b, name: baseline)
    table = graphs.scoped_post_hoc_table(dict(post=dict(records=[record])), "P1")
    rows = table.loc[table.reference.eq("S2")].set_index("scope")
    assert rows.loc["all", "n"] == 15
    assert rows.loc["deep", "n"] == 14
    assert rows.loc["last5", "n"] == 5
    np.testing.assert_allclose(rows.loc["all", "d"], (0.5 + 14 * 0.1) / 15)
    np.testing.assert_allclose(rows.loc[["deep", "last5"], "d"], 0.1)
    assert rows.loc["all", "se"] > rows.loc["deep", "se"]
    pd.testing.assert_frame_equal(baseline, original)
    assert record["folds"][0]["deep_fold"] is False
    _, introduction, conclusion = graphs.post_hoc_sections(dict(post=dict(records=[record])), "P1")
    assert "1 תצורות" not in introduction
    assert len(graphs.plain_explanation(introduction)) >= 250
    assert len(graphs.plain_explanation(conclusion)) >= 250


def test_proposed_f5_is_visible_and_highlighted_in_model_comparison(signed_graphs):
    _, figures = signed_graphs
    traces = {trace.name: trace for trace in figures["models_vs_baselines"].data}
    assert "F5" in traces
    assert traces["F5"].marker.symbol == "diamond"
    assert traces["F5"].marker.size > traces["S2"].marker.size
    assert traces["glm"].y == ("GLM בבסיס (D3)",)


def test_post_hoc_forest_compares_s2_and_f5_on_shared_axes(signed_graphs):
    b, figures = signed_graphs
    fig = figures["post_hoc_forest"]
    assert len(fig.data) == 28
    assert {trace.customdata[0][1] for trace in fig.data} == {"S2", "F5"}
    assert fig.layout.xaxis.range == fig.layout.xaxis2.range
    p1_f5 = next(trace for trace in fig.data if tuple(trace.customdata[0][1:]) == ("F5", "P1"))
    p1 = next(r for r in b["post"]["records"] if r["check"] == "P1")
    candidate = pd.DataFrame(p1["folds"]).set_index("season_id")
    baseline = graphs._selected_folds(b, "F5").set_index("season_id")
    seasons = candidate.index[candidate.deep_fold & baseline.deep_fold]
    differences = baseline.loc[seasons, "poisson_nll"] - candidate.loc[seasons, "poisson_nll"]
    expected = np.average(differences, weights=candidate.loc[seasons, "games"])
    np.testing.assert_allclose(graphs._array(p1_f5.x), [expected], atol=1e-12)


def test_training_validation_is_a_single_endpoint_not_an_iteration_curve(signed_graphs):
    _, figures = signed_graphs
    fig = figures["training_selector"]
    for trace in fig.data[3::4]:
        assert trace.mode == "markers"
        assert len(trace.x) == len(trace.y) == len(trace.error_y.array) == 1
    for button in fig.layout.updatemenus[0].buttons:
        visible = button.args[0]["visible"]
        traces = [trace for trace, flag in zip(fig.data, visible, strict=True) if flag]
        assert int(graphs._array(traces[-1].x)[0]) == int(graphs._array(traces[2].x)[-1])


def test_proposal_is_the_first_learning_panel_from_the_right(signed_graphs):
    b, figures = signed_graphs
    cfg = b["report"]["winners"]["S2"]["parameters"]
    annotations = sorted(
        figures["learning_curves_train_vs_validation"].layout.annotations[:4],
        key=lambda a: (-a.y, -a.x),
    )
    title = "".join(c for c in annotations[0].text if not 0x202A <= ord(c) <= 0x202E)
    assert f"עומק {cfg['max_depth']}" in title
    assert f"קצב {cfg['learning_rate']}" in title
    assert f"עצים {cfg['n_estimators']}" in title


def test_elo_color_scale_has_contrast_inside_every_slice(signed_graphs):
    _, figures = signed_graphs
    for trace in figures["hyperparameter_heatmap"].data:
        values = graphs._array(trace.z)
        np.testing.assert_allclose([trace.zmin, trace.zmax], [values.min(), values.max()])
        assert trace.zmid is None
