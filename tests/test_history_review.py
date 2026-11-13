"""Regression tests for the review estimands, chronology and presentation boundaries."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
import yaml

from ligat.analysis.league_stats import calendar_effects
from ligat.analysis.league_stats.conditions import (
    classify_conditions,
    condition_comparison,
    elo_residuals,
    home_period_comparison,
    load_match_conditions,
    prematch_elo,
    stratified_effect,
)
from ligat.analysis.league_stats.history_review import (
    balance_decomposition,
    champion_previous_rank,
    era_uncertainty,
    home_season_uncertainty,
    seasonal_trend_tests,
    threshold_diagnostics,
)
from ligat.analysis.prepare import load_playoff_upper_sizes
from ligat.viz.presentation import display_table, rtl_figure
from ligat.viz.seasons import expand_season_text, season_label


def matches():
    return pd.DataFrame(
        [
            (25, "2023-08-01 12:00Z", 1, 2, "Netanya", True, 2, 0),
            (25, "2023-08-02 12:00Z", 2, 3, "Netanya", True, 0, 1),
            (25, "2023-08-03 12:00Z", 3, 1, "Other", True, 1, 0),
            (25, "2023-11-25 22:00Z", 1, 2, "Netanya", True, 1, 1),
        ],
        columns=[
            "season_id",
            "date",
            "home_team_id",
            "away_team_id",
            "venue",
            "is_home_venue",
            "home_goals",
            "away_goals",
        ],
    )


def test_configuration_and_hierarchy():
    config = load_match_conditions(Path("config/match_conditions.yaml"))
    games = matches()
    result = classify_conditions(games, config)
    assert result.condition.tolist() == ["SHARED", "NORMAL", "NORMAL", "GHOST"]
    # Full local endpoint is included, even when UTC is on the following/previous day.
    games.loc[3, "date"] = "2023-12-16 21:59Z"
    assert classify_conditions(games, config).condition.iloc[3] == "GHOST"
    games.loc[3, "date"] = "2023-12-16 22:01Z"
    assert classify_conditions(games, config).condition.iloc[3] == "LIMITED"
    assert "condition" not in games


@pytest.mark.parametrize("case", ["reversed", "overlap", "holdout", "source"])
def test_invalid_configuration(tmp_path, case):
    config = yaml.safe_load(Path("config/match_conditions.yaml").read_text(encoding="utf-8"))
    if case == "reversed":
        config["crowd_periods"][0]["start"] = "2020-03-14"
    if case == "overlap":
        config["crowd_periods"][0]["end"] = "2021-03-14"
    if case == "holdout":
        config["relocations"][0]["seasons"] = [26]
    if case == "source":
        config["relocations"][0]["source"] = ""
    path = tmp_path / "conditions.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_match_conditions(path)


def test_elo_is_prematch_and_simultaneous_order_independent():
    original = matches()
    before = prematch_elo(original)
    changed = original.copy()
    changed.loc[1, ["home_goals", "away_goals"]] = [6, 0]
    after = prematch_elo(changed)
    pd.testing.assert_frame_equal(
        before.loc[:1, ["elo_home", "elo_away"]], after.loc[:1, ["elo_home", "elo_away"]]
    )
    assert after.elo_gap.iloc[2] != before.elo_gap.iloc[2]
    original.loc[1, "date"] = original.loc[0, "date"]
    simultaneous = prematch_elo(original)
    assert simultaneous.elo_home.iloc[1] == 1450
    pd.testing.assert_frame_equal(
        simultaneous, prematch_elo(original.iloc[::-1]).reindex(original.index)
    )


def test_elo_regression_and_residual_definition():
    result = elo_residuals(matches())
    assert result.points_gap.tolist() == [3, -3, 3, 0]
    assert abs(result.residual.mean()) < 1e-12


def test_weighted_estimand_and_permutations_are_season_stratified():
    groups = [
        (np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0, 2.0])),
        (np.array([20.0, 22.0]), np.array([17.0, 20.0, 21.0])),
    ]
    first = stratified_effect(groups, n_permutations=199, n_bootstrap=299)
    weights = np.array([3, 2]) / 5
    expected = sum(w * (a.mean() - b.mean()) for w, (a, b) in zip(weights, groups, strict=True))
    assert first["effect"] == pytest.approx(expected)
    # Huge season-specific offsets have NO effect on the estimate, null distribution or BCa.
    shifted = [
        (a + offset, b + offset) for (a, b), offset in zip(groups, [1000.0, -4000.0], strict=True)
    ]
    second = stratified_effect(shifted, n_permutations=199, n_bootstrap=299)
    assert first == pytest.approx(second, abs=1e-9)
    assert first["ci_lower"] < first["effect"] < first["ci_upper"]


@pytest.mark.parametrize(
    "function",
    [
        prematch_elo,
        elo_residuals,
        home_period_comparison,
        home_season_uncertainty,
        era_uncertainty,
        seasonal_trend_tests,
        balance_decomposition,
        champion_previous_rank,
    ],
)
def test_rejects_non_development_before_access(function):
    with pytest.raises(ValueError):
        function(None, seasons=[26])


def test_other_development_guards():
    with pytest.raises(ValueError):
        classify_conditions(None, {}, seasons=[28])
    with pytest.raises(ValueError):
        condition_comparison(None, {}, seasons=[7])
    with pytest.raises(ValueError):
        threshold_diagnostics(None, seasons=[26], upper_sizes={})


def test_full_season_labels_preserve_data():
    assert season_label(12) == "2010/2011"
    assert expand_season_text("2010/11–2012/13") == "2010/2011–2012/2013"
    data = pd.DataFrame({"season_id": [12, 25], "value": [1.0, 2.0]})
    original = deepcopy(data)
    figure = rtl_figure(px.scatter(data, x="value", y="value", hover_data=["season_id"]))
    assert np.array_equal(figure.data[0].x, [1.0, 2.0])
    assert figure.data[0].customdata[:, 0].tolist() == ["2010/2011", "2023/2024"]
    table = display_table(data.set_index("season_id"), {"season_id": "עונה", "value": "ערך"})
    assert "2010/2011" in table._repr_html_() and "2023/2024" in table._repr_html_()
    pd.testing.assert_frame_equal(data, original)


def test_new_diagnostics_real_data(real_tables):
    thresholds = threshold_diagnostics(real_tables, upper_sizes=load_playoff_upper_sizes())
    assert thresholds.loc["champion", "mean"] == pytest.approx(2.283, abs=0.001)
    trends = seasonal_trend_tests(real_tables)
    assert len(trends) == 8
    assert trends.loc[("all", "zero_zero_rate"), "p_bonferroni"] < 0.05
    assert trends.loc[("14_teams", "red_cards_per_match"), "p_bonferroni"] > 0.05
    eras = era_uncertainty(real_tables)
    assert (
        eras.loc["16_teams", "goals_per_match_ci_upper"]
        - eras.loc["16_teams", "goals_per_match_ci_lower"]
        > 0.2
    )
    assert eras.std_ppm_ci_upper.gt(eras.std_ppm).all()
    home = home_season_uncertainty(real_tables)
    assert home.ppm_gap_se.mean() == pytest.approx(0.165, abs=0.02)
    balance = balance_decomposition(real_tables).loc[14:25]
    assert balance.std_ppm.mean() == pytest.approx(0.419, abs=0.002)
    assert balance.luck_std.mean() == pytest.approx(0.222, abs=0.003)
    ranks = champion_previous_rank(real_tables)
    assert (len(ranks), ranks.best_previous.sum(), ranks.top3_previous.sum()) == (17, 9, 14)
    recent = ranks.loc[ranks.n_teams.eq(14)]
    assert (len(recent), recent.best_previous.sum(), recent.top3_previous.sum()) == (12, 7, 11)


@pytest.mark.slow
def test_conditions_real_counts_and_inference_family(real_tables):
    original = real_tables.match_table.copy(deep=True)
    config = load_match_conditions(Path("config/match_conditions.yaml"))
    result, games = condition_comparison(
        real_tables.match_table, config, n_permutations=199, n_bootstrap=299
    )
    assert result.n.to_dict() == dict(
        NORMAL=3409, GHOST=252, LIMITED=125, SHARED=206, RELOCATED=172, OTHER=151
    )
    tested = result.loc[["GHOST", "LIMITED", "SHARED", "RELOCATED"]]
    np.testing.assert_allclose(tested.p_bonferroni, np.minimum(1, 4 * tested.p))
    assert tested.normal_n.tolist() == [303, 144, 3409, 1225]
    assert tested.effect.tolist() == pytest.approx([0.144, -0.009, -0.277, -0.047], abs=0.002)
    assert result.loc[["NORMAL", "OTHER"], "p"].isna().all()
    periods = home_period_comparison(games, normal_only=True)
    assert periods.mean_gap.tolist() == pytest.approx([0.433529, 0.105673], abs=1e-6)
    pd.testing.assert_frame_equal(real_tables.match_table, original)


def test_calendar_small_season_intervals_are_unavailable(real_tables):
    frame = calendar_effects(real_tables, min_bootstrap_seasons=5, n_bootstrap=30)
    assert frame.loc[("month", "6"), "n_seasons"] == 4
    assert frame.loc[("month", "7"), "n_seasons"] == 1
    assert frame.loc[frame.n_seasons.lt(5), "goals_per_match_ci_lower"].isna().all()
    assert frame.loc[frame.n_seasons.ge(5), "goals_per_match_ci_lower"].notna().all()
