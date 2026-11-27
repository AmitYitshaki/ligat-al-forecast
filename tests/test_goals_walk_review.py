"""Second-walk displays preserve signed evidence and declare comparison populations."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_runner import load_goal_inputs
from ligat.viz import goals_walk_evidence as evidence_views
from ligat.viz import goals_walk_review as review


def test_model_translation_preserves_coordinates():
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[1.4], y=["glm"], name="EXP-002"))
    result = review.model_names_figure(figure)
    assert result.data[0].y == figure.data[0].y
    assert result.data[0].x == figure.data[0].x
    assert result.data[0].name == "GLM סרגל (EXP-002)"
    assert result.layout.yaxis.ticktext == ("GLM בבסיס (D3)",)


def test_signed_intervals_keep_six_decimal_places():
    frame = pd.DataFrame([dict(d=0.000822, lower=0.000048, upper=0.001597)])
    result = review.render(frame, dict(d="שיפור", lower="תחתון", upper="עליון"))
    assert "0.000048" in result and "0.001597" in result
    assert "0.00–0.00" not in result


def test_inline_prior_expressions_are_isolated_from_hebrew():
    book = json.loads(Path("notebooks/explore/05_goals.ipynb").read_bytes())
    section = next(
        "".join(c["source"])
        for c in book["cells"]
        if c["cell_type"] == "markdown" and "3.3 — GLM" in "".join(c["source"])
    )
    assert section.count('<bdi dir="ltr"') >= 3
    assert "<bdi" in section.split("Σ log α = 0")[0]


def test_default_table_has_no_filler_sentence():
    book = json.loads(Path("notebooks/explore/05_goals.ipynb").read_bytes())
    setup = "".join(book["cells"][2]["source"])
    assert "אלה השדות והשורות המוצגים" not in setup
    assert "if description:" in setup


def test_e3g_counts_and_parameter_names_are_readable():
    from ligat.viz.goals_configuration_names import configuration_name

    assert "15.000000" not in review.render(pd.DataFrame({"n": [15]}), {"n": "עונות"})
    value = configuration_name(
        "depth: 2; iterations: 150; half_life: None; "
        "reg_lambda: 1; subsample: 0.7; preset: library_default; 2 עונה"
    )
    for term in ["depth", "iterations", "half_life", "reg_lambda", "subsample", "preset", "None"]:
        assert term not in value
    assert "2 עונות" in value
    assert "13" not in review.scope_description("all")
    assert "13" not in review.scope_description("last5")


def test_configuration_translation_preserves_plotly_styles_and_readable_catboost():
    import plotly.graph_objects as go

    from ligat.viz import goals_configuration_names as naming

    original = go.Figure(go.Scatter(x=[1], y=[2], name="d2 / T150 / lr0.05 / mcw1"))
    translated = naming.figure_labels(original)
    assert translated.layout.template == original.layout.template
    assert translated.data[0].x == original.data[0].x
    assert "150 עצים" in translated.data[0].name
    assert "150 איטרציות" in naming.configuration_name("CatBoost d2 T150 L2=10")


@pytest.fixture(scope="module")
def evidence(real_tables):
    path = Path("data/derived/exp005/full/ladder_report.json")
    if not path.exists():
        pytest.skip("Signed local D3 artifacts are not committed")
    data = load_goal_inputs(load_goal_experiment(), tables=real_tables)
    from ligat.evaluate.goals_post_hoc import load_post_hoc

    return dict(
        report=json.loads(path.read_bytes())["payload"],
        validation=pd.read_parquet(path.parent / "validation_views.parquet").assign(
            season_id=lambda p: p.fold + 10
        ),
        data=data,
        post=load_post_hoc(load_goal_experiment()),
    )


@pytest.mark.slow
@pytest.mark.real_data
def test_all_deep_metrics_use_the_same_thirteen_seasons(evidence):
    table = review.metrics_table(evidence, "deep")
    assert set(table.seasons) == {13}
    assert len(table) == 10
    assert "F6" not in set(table.model)
    assert table.mae.notna().all() and table.rmse.notna().all()
    assert set(review.paired_models_table(evidence).n) == {13}


@pytest.mark.real_data
def test_decay_includes_glm_and_all_three_populations(evidence):
    table = review.decay_table(evidence)
    assert len(table) == 36
    assert set(table.scope) == set(review.SCOPES.values())
    assert "GLM בבסיס (D3)" in set(table.model)


@pytest.mark.real_data
def test_closed_ruler_reference_uses_exact_exponential(evidence):
    result = review.ruler_html(evidence)
    assert "23.292539" in result and "24.352630" in result


@pytest.mark.real_data
def test_candidates_preserve_signed_paired_populations_and_direction(evidence):
    table = evidence_views.candidate_table(evidence)
    assert len(table) == 54
    deep = table.loc[table.scope.eq("deep") & table.metric.eq("poisson_nll")]
    glm = deep.loc[deep.candidate.eq(review.LABELS["glm"])].iloc[0]
    assert glm.n == 13
    assert glm.d == pytest.approx(-0.000471824944, abs=1e-10)
    assert glm.se == pytest.approx(0.001526782897, abs=1e-10)
    assert set(deep.loc[~deep.candidate.eq(review.LABELS["glm"]), "n"]) == {14}
    assert set(table.loc[table.scope.eq("last5"), "n"]) == {5}
    figure = evidence_views.candidate_figure(evidence)
    assert len(figure.data) == 9
    assert len(figure.layout.xaxis.tickvals) == 3
    assert figure.layout.xaxis.tickvals[0] < 0 < figure.layout.xaxis.tickvals[-1]


@pytest.mark.real_data
def test_flagged_seasons_are_distinct_from_configuration_averages(evidence):
    table = evidence_views.flagged_seasons(evidence)
    assert len(table) == 10
    assert set(table.season_id) == {19}
    assert table.log_loss.min() == pytest.approx(0.972233, abs=5e-7)
    assert "1.020161" in evidence_views.leakage_html(evidence)


@pytest.mark.real_data
def test_family_order_and_multiple_testing_claim_are_supported(evidence):
    assert list(evidence_views.family_ranking(evidence).name) == [
        "F5",
        "F3",
        "F1-R",
        "F1-A",
        "F4",
        "F2",
        "F6",
    ]
    text = evidence_views.family_explanation(evidence)
    assert "11 עונות חיוביות מתוך 14" in text
    assert "1 ל-4" not in text and "סיכוי מזל" in text


@pytest.mark.real_data
def test_final_forests_share_thirteen_seasons_and_show_f5(evidence):
    from ligat.viz import goals_walk_plots as fixes

    figure = fixes.paired_forest(evidence)
    assert all(trace.customdata == (13,) for trace in figure.data)
    assert any("F5" in trace.y[0] for trace in figure.data)
    assert figure.layout.xaxis2.domain
    season = fixes.metric_by_season(evidence)
    assert any("F5" in trace.name for trace in season.data)
    assert not any("F6" in trace.name for trace in season.data)
    assert all(len(trace.x) == 1 for trace in fixes.models_vs_baselines(evidence).data)
    assert len(fixes.population(evidence, "deep")) == 13
    elo = fixes.elo_by_scope(evidence)
    assert any("F5" in name for name in elo.data[0].y)
    assert not any("F6" in name for name in elo.data[0].y)


@pytest.mark.real_data
def test_budget_uses_paired_intervals_and_includes_caution_one_proposal(evidence):
    from ligat.viz import goals_walk_plots as fixes

    table = fixes.budget_table(evidence)
    proposal = table.loc[table.proposal].iloc[0]
    assert proposal.caution == 1 and set(table.loc[~table.proposal, "caution"]) == {30}
    assert proposal.lower == pytest.approx(74.21356527114)
    assert proposal.upper == pytest.approx(113.50134181188)
    assert proposal.versus_proposal == 0
    assert table.includes_zero.any()


@pytest.mark.real_data
def test_learning_change_uses_paired_long_diagnostic_folds(evidence):
    from ligat.viz import goal_training_curves as curves
    from ligat.viz import goals_walk_plots as fixes

    result = fixes.learning_changes(curves.load_learning_diagnostics(load_goal_experiment()))
    assert list(result.trees) == [400, 1000]
    assert set(result.n) == {15}
    assert result.better.between(0, 15).all()


@pytest.mark.real_data
def test_review_table_compares_its_actual_alternative_and_keeps_deep_plot_reference(evidence):
    from ligat.viz import goals_walk_plots as fixes

    table = review.review_table(evidence)
    assert len(table) == 7
    assert (table.proposal != table.alternative).all()
    maher = table.loc[table.question.str.contains("Maher")].iloc[0]
    assert maher.gap == pytest.approx(-0.001885, abs=5e-7)
    deep = next(r for r in table.attrs["deep_rows"] if "Maher" in r["question"])
    assert deep["deep_gap"] == 0
    assert len(fixes.review_plot(evidence).data) == len(table)


@pytest.mark.real_data
def test_posthoc_rendered_tables_and_prose_use_readable_names(evidence):
    import re

    forbidden = re.compile(r"d\d+ ?/|\bT\d+\b|lr0\.|mcw\d|cw\d+\.|L2=|proposal|\bn\d+ ·")
    for check in ["P1", "P2", "P3", "P4"]:
        rendered = review.post_summary(evidence, check)
        assert not forbidden.search(rendered)
        assert "כולל 0?" in rendered


@pytest.mark.real_data
def test_executed_notebook_contains_no_numpy_float_representation():
    book = json.loads(Path("notebooks/explore/05_goals.ipynb").read_bytes())
    visible = []
    for cell in book["cells"]:
        for output in cell.get("outputs", []):
            for mime, value in output.get("data", {}).items():
                if mime in {"text/html", "text/plain", "application/vnd.plotly.v1+json"}:
                    visible.append(json.dumps(value, ensure_ascii=False))
    assert visible, "Execute the notebook before checking its displayed text"
    assert all("np.float64" not in text for text in visible)
