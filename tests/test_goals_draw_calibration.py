"""Season-cluster uncertainty and tie-safe calibration preserve native evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.viz import goals_draw_calibration as diagnostic
from ligat.viz import goals_m3_dossier_views as views
from tests.test_goals_m3_dossier import native as native


def test_bootstrap_draws_whole_unequal_season_blocks_with_fixed_seed():
    seasons = np.array([13, 14, 15])
    counts = np.array([[1, 2, 1], [7, 2, 4], [2, 4, 9]])
    draws = np.array([[1, 0, 1], [0, 2, 1], [1, 3, 0]])
    rates, samples = diagnostic.season_bootstrap(seasons, counts, draws, B=30)
    repeated, again = diagnostic.season_bootstrap(seasons, counts, draws, B=30)
    np.testing.assert_array_equal(samples, again)
    np.testing.assert_array_equal(rates, repeated)
    for row, sample in zip(rates, samples, strict=True):
        total = np.zeros(3)
        observed = np.zeros(3)
        for season in sample:
            index = np.flatnonzero(seasons == season)[0]
            total += counts[index]
            observed += draws[index]
        np.testing.assert_array_equal(row, observed / total)
    assert samples.shape == (30, 3)


def test_equal_count_bins_preserve_every_tie_including_constant_probabilities():
    p = np.repeat([0.17, 0.20, 0.21, 0.25, 0.31], [8, 31, 7, 10, 44])
    bins = diagnostic.equal_count_bins(p)
    for value in np.unique(p):
        assert len(set(bins[p == value])) == 1
    assert sum(np.bincount(bins)) == len(p)
    np.testing.assert_array_equal(diagnostic.equal_count_bins([0.2] * 30), np.zeros(30))


@pytest.mark.real_data
def test_native_diagnostics_preserve_e3g_ece_nll_counts_and_original_tables(native):
    data, original = native
    before = original["calibration"].copy(deep=True)
    result = diagnostic.augment(original, data)
    pd.testing.assert_frame_equal(before, result["calibration"])
    pd.testing.assert_frame_equal(
        original["summary"], result["summary"][original["summary"].columns]
    )
    saved = pd.read_json(Path("data/reports/T-014/E3g_summary.json"))
    np.testing.assert_allclose(result["summary"].ece_draw, saved.ece_draw, atol=1e-12, rtol=0)
    assert len(result["draw_bootstrap"]) == 21 and result["bootstrap_samples"].shape == (2000, 13)
    assert set(result["bootstrap_samples"].ravel()) == set(range(13, 26))
    for (scope, name), p in result["draw_deciles"].groupby(["scope", "candidate"]):
        expected = (
            original["summary"]
            .loc[lambda x, s=scope, n=name: x.scope.eq(s) & x.candidate.eq(n)]
            .iloc[0]
        )
        assert p.games.sum() == expected.games and p.bin.nunique() <= 10
    figure = views.calibration_figure(result)
    for button in figure.layout.updatemenus[0].buttons:
        text = button.args[1]["annotations"][0]["text"]
        assert "עשירונים" in text and "ECE" in text
    assert "ece_draw_deciles" in views.summary(result)
    assert "ECE לפי עשירונים" in views.table(result["draw_deciles"])
    assert "מקורי, E3g" not in views.table(result["draw_deciles"])
    assert "np.float64" not in figure.to_json()
    assert len(json.loads(figure.to_json())["data"]) == 22
