"""Era boundaries, retrospective fold isolation and chapter-A diagnostic contracts."""

import numpy as np
import pandas as pd
import pytest

from ligat.analysis.league_stats.chapter_a import (
    ERA14_SEASONS,
    adjacent_pairs,
    correlation_evidence,
    era_persistence,
    home_advantage_diagnostics,
    loso_predictions,
    predictor_comparison,
    promoted_performance,
    promoted_rule_comparison,
    style_diagnostics,
)


def test_adjacent_pairs_respects_era_and_missing_seasons():
    frame = pd.DataFrame(
        {
            "season_id": [14, 15, 17, 14, 15],
            "team_id": [1, 1, 1, 2, 2],
            "points_per_match": [1.0, 2.0, 3.0, 2.0, 1.0],
        }
    ).set_index(["season_id", "team_id"])
    pairs = adjacent_pairs(frame, ["points_per_match"])
    assert pairs.index.tolist() == [(15, 1), (15, 2)]
    assert pairs.points_per_match_prev.tolist() == [1.0, 2.0]


def test_loso_held_out_targets_cannot_change_own_predictions():
    index = pd.MultiIndex.from_product([[14, 15, 16], [1, 2, 3]], names=["season_id", "team_id"])
    data = pd.DataFrame(
        {"prior": np.arange(9), "points_per_match": np.arange(9) * 0.2}, index=index
    )
    expected = loso_predictions(data, ["prior"])
    altered = data.copy()
    altered.loc[15, "points_per_match"] = [100.0, 200.0, 300.0]
    actual = loso_predictions(altered, ["prior"])
    np.testing.assert_allclose(actual.loc[15, "predicted"], expected.loc[15, "predicted"])
    assert actual.n_train.eq(6).all()
    assert not np.allclose(actual.loc[14, "predicted"], expected.loc[14, "predicted"])


def test_correlation_family_and_degenerate_values():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 2.0, 5.0], "constant": [1.0, 1.0, 1.0, 1.0]}
    )
    result = correlation_evidence(
        frame, [("test", "x", "y"), ("undefined", "x", "constant")], comparisons=4
    )
    assert result.loc["test", "p_bonferroni"] == pytest.approx(
        min(1.0, 4 * result.loc["test", "p_value"])
    )
    assert np.isnan(result.loc["undefined", "p_value"])
    with pytest.raises(ValueError):
        correlation_evidence(frame, [("test", "x", "y")], comparisons=0)


def test_diagnostic_table_preserves_small_meaningful_differences():
    from ligat.viz.notebook02 import COLUMN_NAMES
    from ligat.viz.presentation import display_table

    frame = pd.DataFrame({"loso_rmse": [0.268, 0.270], "p_value": [0.007434, 1e-9]})
    html = display_table(
        frame, COLUMN_NAMES, precision={"loso_rmse": 3, "p_value": 4}
    )._repr_html_()
    assert "0.268" in html and "0.270" in html and "0.0074" in html and "1.00e-09" in html


def test_chart_palette_and_display_only_jitter(real_tables):
    from ligat.analysis.league_stats import team_season_stats
    from ligat.viz.league_stats import chapter_a_chart

    source = team_season_stats(real_tables, ERA14_SEASONS)
    chart = chapter_a_chart("win_style", source)
    assert chart.layout.coloraxis.cmin == 1 and chart.layout.coloraxis.cmax == 14
    np.testing.assert_allclose(chart.data[0].customdata[:, 3].astype(float), source.big_win_rate)
    np.testing.assert_allclose(chart.data[0].customdata[:, 4].astype(float), source.narrow_win_rate)
    assert np.max(np.abs(np.asarray(chart.data[0].x) - source.big_win_rate)) <= 0.004
    assert "customdata[3]" in chart.data[0].hovertemplate
    np.testing.assert_allclose(chapter_a_chart("win_style", source).data[0].x, chart.data[0].x)
    detail, _ = promoted_rule_comparison(real_tables)
    promoted = detail.xs("q30", level="rule")
    promoted.attrs["rule"] = "q30"
    assert chapter_a_chart("promoted", promoted).data


@pytest.mark.parametrize(
    "function",
    [
        era_persistence,
        predictor_comparison,
        home_advantage_diagnostics,
        style_diagnostics,
        promoted_rule_comparison,
        promoted_performance,
    ],
)
def test_new_diagnostics_reject_non_development(function):
    with pytest.raises(ValueError, match="development"):
        function(None, seasons=[25, 26])


def test_era_and_predictor_samples(real_tables):
    pairs, stats = era_persistence(real_tables, n_bootstrap=30)
    assert len(pairs) == 132
    assert pairs.index.get_level_values("season_id").min() == 15
    assert 0.8 < stats.loc["ppm", "slope"] < 0.9
    compared, frame = predictor_comparison(real_tables)
    assert compared.xs("adjacent").n.eq(132).all()
    assert compared.xs("history_common").n.eq(125).all()
    assert frame.index.equals(pairs.index)
    assert compared.loso_rmse.between(0.2, 0.4).all()


def test_home_and_styles_are_descriptive(real_tables):
    home = home_advantage_diagnostics(real_tables)
    style, rates = style_diagnostics(real_tables)
    assert home.n == 168 and home.n_pairs == 132
    assert abs(home.persistence) < 0.1
    assert abs(home.observed_sd - home.luck_sd) < 0.02
    assert -0.7 < style.attack_defence_corr < -0.6
    assert rates.loc["btts_rate", "r_squared"] >= 0
    assert rates.loc["over_2_5_rate", "r_squared"] > 0.7


def test_promoted_rules_and_performance(real_tables):
    detail, summary = promoted_rule_comparison(real_tables)
    assert summary.loc[("all", "q30"), "n"] == 36
    assert summary.loc[("all", "median"), "fraction_below"] == pytest.approx(26 / 36)
    assert summary.loc[("all", "q30"), "fraction_below"] == pytest.approx(16 / 36)
    assert summary.loc[("all", "q30"), "mae_pp"] < summary.loc[("all", "median"), "mae_pp"]
    q30 = detail.xs("q30", level="rule")
    old = detail.xs("median", level="rule")
    assert q30.imputed_share.le(old.imputed_share).all()
    performance = promoted_performance(real_tables)
    assert performance.loc[True, "n"] == 23
    assert performance.loc[True, "relegations"] == 7
    assert performance.loc[True, "best_final_position"] == 5
    without_transition = promoted_performance(real_tables, range(15, 26))
    assert without_transition.loc[True, "n"] == 22
    assert without_transition.loc[True, "relegations"] == 6
