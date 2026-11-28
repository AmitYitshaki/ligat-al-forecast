import numpy as np
import pandas as pd
import pytest

from ligat.viz import goals_f5_refit as refit
from ligat.viz import goals_f5_review as review
from ligat.viz.goals_m3_dossier import source_bundle


def test_defensive_residual_is_the_opponents_scoring_error():
    rows = pd.DataFrame(
        dict(
            game_id=[10, 10, 20, 20],
            team_id=[1, 2, 2, 1],
            is_home=[True, False, True, False],
            residual=[1.0, -0.5, 0.5, 3.0],
        )
    ).set_index(["game_id", "team_id"])
    result = review.team_residuals(rows).set_index(["team_id", "side"])
    assert result.loc[(1, "הבקעה"), "mean"] == 2.0
    assert result.loc[(1, "ספיגה"), "mean"] == 0.0
    assert result.loc[(1, "ספיגה"), "sem"] == 0.5
    assert result.loc[(2, "ספיגה"), "size"] == 2


def test_rate_calibration_keeps_ties_and_matches_manual_means():
    rows = pd.DataFrame({"lambda": [1.2] * 8 + [2.4] * 8, "goals": [0, 2] * 8})
    result = review.rate_calibration(rows)
    assert sorted(result.rows) == [8, 8]
    np.testing.assert_allclose(result.predicted, [1.2, 2.4])
    np.testing.assert_allclose(result.actual, [1.0, 1.0])
    np.testing.assert_allclose(result.se, [1 / np.sqrt(7)] * 2)


@pytest.mark.real_data
def test_verified_validation_rows_and_training_gap(real_tables):
    spec, data, _, _ = refit.inputs(real_tables)
    f5 = review.signed_rows(data, source_bundle(), spec.output, "F5")
    deep = f5.loc[f5.deep_fold]
    summary = review.team_residuals(deep)
    assert summary.groupby("side")["size"].sum().tolist() == [len(deep)] * 2
    scores = review.training_gap(data, spec.output, f5)
    assert scores.validation_rows.sum() == 7442
    assert scores.training_rows.is_monotonic_increasing
    assert review.rate_calibration(deep).rows.sum() == len(deep)
