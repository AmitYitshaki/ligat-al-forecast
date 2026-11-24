"""Notebook figures preserve RTL while never embedding forbidden marks."""

import json
from pathlib import Path

import numpy as np
import pytest

from ligat.viz import goals_notebook as views
from ligat.viz.motivation_style import verify_rtl_labels


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
