"""Notebook figures preserve RTL while never embedding forbidden marks."""

import json
from pathlib import Path

import numpy as np
import pytest

from ligat.viz import goals_notebook as views
from ligat.viz.motivation_style import verify_rtl_labels


def test_notebook_math_is_outside_html_blocks():
    import re

    book = json.loads(Path("notebooks/explore/05_goals.ipynb").read_text(encoding="utf-8"))
    for cell in book["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            for block in re.findall(r"<div\b[^>]*>.*?</div>", source, flags=re.S):
                assert "$" not in block, block
            for formula in re.findall(r"\$\$(.*?)\$\$", source, flags=re.S):
                assert len(formula.strip()) >= 12, formula


def test_ladder_table_applies_approved_correlation_drops():
    from ligat.evaluate.goals_experiment import load_goal_experiment

    table = views.ladder_plan_table()
    for dropped in load_goal_experiment().raw["correlation_drops"]:
        assert all(dropped not in row.split(", ") for row in table.features)


def test_short_display_math_does_not_interrupt_a_sentence():
    import html
    import re

    book = json.loads(Path("notebooks/explore/05_goals.ipynb").read_text(encoding="utf-8"))
    for number, cell in enumerate(book["cells"], 1):
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        for match in re.finditer(r"\$\$(.*?)\$\$", source, flags=re.S):
            if len(match[1].strip()) < 40:
                prefix = html.unescape(re.sub(r"<[^>]*>", "", source[: match.start()])).rstrip()
                assert not prefix or prefix.endswith((".", ":")), (number, match[1], prefix[-80:])


def test_review_gap_is_positive_when_candidate_has_higher_loss():
    import pandas as pd

    from ligat.evaluate.goals_ladder import LadderConfiguration
    from ligat.viz.goals_qa import review_evidence

    configs = {
        "deep": dict(backend="xgb", parameters={"max_depth": 2}, half_life=None),
        "recent": dict(backend="xgb", parameters={"max_depth": 3}, half_life=None),
    }
    records = [
        dict(
            configuration_key=LadderConfiguration(**cfg).key,
            feature_list=json.dumps(["is_home"]),
            fold=season - 10,
            games=100,
            deep_fold=True,
            poisson_nll=1.4 + (0.01 if name == "recent" else 0),
        )
        for name, cfg in configs.items()
        for season in range(21, 26)
    ]
    report = dict(
        decisions=[
            dict(
                name="test",
                configurations=configs,
                features=["is_home"],
                labels={"deep": "עמוקה", "recent": "אחרונות"},
            )
        ],
        open_reviews=[
            dict(
                name="test",
                deep_leaders=[{"key": "deep"}],
                recent_leaders=[{"key": "recent"}],
                proposal="recent",
                dependents=[],
            )
        ],
    )
    result = review_evidence(report, pd.DataFrame(records))
    np.testing.assert_allclose(result.gap, 0.01, atol=1e-12)


def test_math_and_plan_figures_use_rtl_without_direction_marks():
    figures = [
        views.chain_figure(),
        views.walk_forward_figure(),
        views.decay_figure(),
        views.decay_figure(daily=True),
    ]
    for figure in figures:
        serialized = json.loads(figure.to_json())
        verify_rtl_labels(serialized)
        assert all(
            chr(c) not in json.dumps(serialized, ensure_ascii=False) for c in (0x200E, 0x200F)
        )
    assert len(figures[1].data[0].z) == 15
    assert len(figures[1].data[0].z[0]) == 18
    assert views.hyperparameter_table().parameter.notna().all()


@pytest.mark.real_data
def test_fixed_poisson_example_uses_a_real_fixture_without_fitting(real_tables, monkeypatch):
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from ligat.evaluate.goals_runner import load_goal_inputs

    if not Path("data/derived/prematch/prematch_v2.parquet").exists():
        pytest.skip("Frozen M0 local only")
    data = load_goal_inputs(load_goal_experiment(), tables=real_tables)
    monkeypatch.setattr(
        "ligat.models.m2_xgb.XGBGoalModel.fit", lambda *a, **k: pytest.fail("Notebook fitted")
    )
    figure, example = views.poisson_example(data)
    assert example.lambda_example.iloc[0] == 1.6
    assert example.game_id.iloc[0] in data.context.game_id.to_numpy()
    np.testing.assert_allclose(np.asarray(figure.data[0].y, dtype=float).sum(), 1, atol=1e-14)
    assert set(data.context.season_id) == set(range(8, 26))
