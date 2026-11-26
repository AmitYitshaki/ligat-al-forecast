"""Second-walk displays preserve signed evidence and declare comparison populations."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_runner import load_goal_inputs
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


@pytest.fixture(scope="module")
def evidence(real_tables):
    path = Path("data/derived/exp005/full/ladder_report.json")
    if not path.exists():
        pytest.skip("Signed local D3 artifacts are not committed")
    data = load_goal_inputs(load_goal_experiment(), tables=real_tables)
    return dict(
        report=json.loads(path.read_bytes())["payload"],
        validation=pd.read_parquet(path.parent / "validation_views.parquet").assign(
            season_id=lambda p: p.fold + 10
        ),
        data=data,
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
