"""Checks against the original signed F5 prediction keys and recorded losses."""

import pandas as pd
import pytest

from ligat.evaluate.goals_runner import load_goal_unit
from ligat.viz import goals_f5_diagnostics as diag
from ligat.viz import goals_f5_refit as refit


def test_home_alternatives_only_replace_the_home_flag():
    x = pd.DataFrame({"is_home": [0, 1], "elo_pre": [1200, 1250], "stage": ["upper", "lower"]})
    for value, rewritten in zip([1, 0], diag.home_alternatives(x), strict=True):
        pd.testing.assert_frame_equal(rewritten.drop(columns="is_home"), x.drop(columns="is_home"))
        assert rewritten.is_home.eq(value).all()
        assert rewritten.is_home.dtype == x.is_home.dtype


@pytest.mark.real_data
def test_f5_signed_checkpoint_and_shap(real_tables):
    spec, data, provenance, units = refit.inputs(real_tables)
    unit = units[17]
    model, _ = refit.reconstruct_fold(spec.output, unit, data, provenance)
    signed = load_goal_unit(
        spec.output / "full" / (unit.unit_id + ".json"), unit, provenance, data
    ).evaluation.per_team
    features = data.X.loc[signed.index, list(unit.features)]
    _, gain, absolute, relative = diag.contributions(model, features, signed["lambda"])
    assert absolute > 0 and relative < 1e-6
    assert sum(gain.values()) == pytest.approx(100)


@pytest.mark.real_data
def test_home_comparison_uses_development_rows_and_actual_totals(real_tables):
    spec, data, _, _ = refit.inputs(real_tables)
    result = diag.diagnostics(data, spec.output)
    assert set(result["home"].season_id) == set(range(11, 26))
    assert result["home"].rows.sum() == 7442
    mask = data.context.season_id.eq(25)
    targets = data.y[mask]
    home = data.X.loc[targets.index, "is_home"].eq(1)
    ratio = targets[home].sum() / targets[~home].sum()
    assert result["home"].set_index("season_id").loc[25, "observed"] == pytest.approx(ratio)
