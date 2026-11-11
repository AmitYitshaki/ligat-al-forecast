"""Descriptive statistics: units, independent references, support and mirrored games."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prematch_final_review import TREND_FEATURES, WORDS
from ligat.analysis.prematch_match_review import (
    MatchReview,
    elo_bins,
    grouped_linearity,
    match_view,
    matched_dead_games,
    period_checks,
    poisson_noise_limits,
    promotion_chi_example,
    venue_checks,
)
from ligat.viz.prematch_match import match_description, match_table, period_chart


@pytest.mark.parametrize("season", [26, 27, 28])
def test_match_checks_reject_non_development_seasons(season):
    with pytest.raises(ValueError, match="8–25"):
        elo_bins(pd.DataFrame({"season_id": [season]}))


def test_match_view_uses_one_fixture_and_joins_the_away_flag():
    frame = pd.DataFrame(
        {
            "season_id": [8, 8],
            "is_home": [True, False],
            "opponent_id": [2, 1],
            "nothing_to_play": [False, True],
        },
        index=pd.MultiIndex.from_tuples([(10, 1), (10, 2)], names=["game_id", "team_id"]),
    )
    rows = frame.reset_index().assign(goals_scored=[2, 1], goals_conceded=[1, 2])
    games, targets = match_view(frame, rows)
    assert len(games) == 1 and len(targets) == 2
    assert games.dead_union.tolist() == [True]
    assert games.total_goals.tolist() == [3] and games.outcome.tolist() == [0]
    assert targets.goal_difference_game.tolist() == [1, -1]
    assert "goals_scored" not in frame


def test_poisson_decomposition_uses_population_variance_and_exposes_negative_signal():
    games = pd.DataFrame(
        {
            "season_id": [8] * 4,
            "goal_difference_game": [-2, -1, 1, 2],
            "total_goals": [2, 1, 1, 2],
            "goals_scored": [0, 0, 1, 2],
            "goals_conceded": [2, 1, 0, 0],
        }
    )
    row = poisson_noise_limits(games).iloc[0]
    assert row.variance == 2.5 and row.noise == 1.5
    assert row.signal_fraction == pytest.approx(0.4)
    assert row.heuristic_correlation == pytest.approx(np.sqrt(0.4))
    high_noise = games.assign(total_goals=10)
    negative = poisson_noise_limits(high_noise).iloc[0]
    assert negative.raw_signal_fraction == -3
    assert negative.heuristic_correlation == 0


def test_bins_count_games_once_and_eta_does_not_correlate_only_averages():
    x = np.repeat(np.arange(20), 10)
    games = pd.DataFrame(
        {
            "season_id": 8,
            "elo_diff": x,
            "goal_difference_game": x + np.tile([-20, 20], 100),
            "outcome": np.tile([0, 1], 100),
        }
    )
    bins = elo_bins(games)
    assert bins.games.sum() == len(games)
    assert bins.elo_low.is_monotonic_increasing
    row = grouped_linearity(games).iloc[0]
    assert row.eta == pytest.approx(row.pearson)
    assert row.eta < 0.3  # Correlation between the twenty means alone would be one.
    assert bins.home_win_percent.add(bins.draw_percent).eq(100).all()


def test_disjoint_periods_share_cases_and_bootstrap_draws_and_report_partial_correlation():
    rng = np.random.default_rng(7)
    n = 400
    elo = rng.normal(size=n)
    value = elo + rng.normal(size=n)
    target = elo + 0.2 * value + rng.normal(size=n)
    games = pd.DataFrame(
        {
            "season_id": np.repeat([20, 21], n // 2),
            "elo_diff": elo,
            "value_avg_log_ratio": value,
            "goal_difference_game": target,
        }
    )
    for feature in set(TREND_FEATURES) - set(games):
        games[feature] = elo + rng.normal(size=n)
    games.loc[0, "prev_gd_diff"] = np.nan
    periods, overlap, bootstrap = period_checks(games, repeats=40, seed=8)
    assert periods.games_early.eq(199).all() and periods.games_recent.eq(200).all()
    early = games.loc[games.season_id.eq(20)].dropna()
    design = np.column_stack([np.ones(len(early)), early.elo_diff])
    residuals = [
        early[c] - design @ np.linalg.lstsq(design, early[c], rcond=None)[0]
        for c in ("value_avg_log_ratio", "goal_difference_game")
    ]
    assert overlap.partial_value.iloc[0] == pytest.approx(np.corrcoef(residuals)[0, 1])
    again = period_checks(games, repeats=40, seed=8)[2]
    assert_frame_equal(bootstrap, again)
    assert bootstrap.se.iloc[0] > 0


def test_venue_correlations_preserve_mirror_control_and_distinct_targets():
    index = pd.MultiIndex.from_product([[1, 2, 3], [10, 20]], names=["game_id", "team_id"])
    frame = pd.DataFrame(
        {
            "season_id": 8,
            "is_home": [True, False] * 3,
            "elo_diff": [-1, 1, 0, 0, 2, -2],
            "gf_venue_ewma5": [1, 2, 2, 3, 3, 1],
            "ga_venue_ewma5": [2, 1, 3, 1, 2, 3],
            "opp_ga_venue_ewma5": [1, 2, 1, 3, 3, 2],
        },
        index=index,
    )
    targets = pd.DataFrame(
        {
            "goals_scored": [0, 1, 2, 2, 3, 0],
            "goals_conceded": [1, 0, 2, 2, 0, 3],
            "goal_difference_game": [-1, 1, 0, 0, 3, -3],
        },
        index=index,
    )
    result = venue_checks(frame, targets)
    assert result.iloc[-1].pearson_home == pytest.approx(result.iloc[-1].pearson_away)
    with pytest.raises(ValueError, match="align"):
        venue_checks(frame, targets.iloc[::-1])


def test_chi_example_retains_all_six_contributions_and_expected_margins():
    games = pd.DataFrame(
        {
            "season_id": 8,
            "promoted_in": [False] * 6 + [True] * 6,
            "outcome": [0, 0, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2],
        }
    )
    cells, summary = promotion_chi_example(games)
    assert len(cells) == 6 and cells.expected.tolist() == [2, 1.5, 2.5] * 2
    assert cells.contribution.sum() == pytest.approx(1 + 1 / 3 + 0.2)
    assert summary.cramers_v.iloc[0] == pytest.approx(np.sqrt((1 + 1 / 3 + 0.2) / 12))


def test_dead_standardization_excludes_unsupported_games_from_both_sides_of_contrast():
    rows = []
    for stage, dead, outcomes in [
        ("regular", True, [1, 0]),
        ("regular", False, [1, 1, 0, 0]),
        ("lower", True, [0, 0]),
        ("lower", False, [1, 0]),
        ("middle", True, [1]),
    ]:
        rows.extend(
            dict(season_id=8, stage=stage, dead_union=dead, rounds_to_stage_end=0, outcome=y)
            for y in outcomes
        )
    summary, difference, strata = matched_dead_games(pd.DataFrame(rows))
    row = difference.iloc[0]
    assert (row.matched_dead, row.excluded_dead) == (4, 1)
    assert summary.iloc[0].draw_percent == 40
    assert summary.iloc[1].draw_percent == 25
    assert summary.iloc[3].draw_percent == 50
    assert row.difference_pp == 25
    expected_se = np.sqrt(0.25 * 0.75 / 4 + 0.5**2 * 0.25 / 4 + 0.5**2 * 0.25 / 2)
    assert row.se_pp == pytest.approx(100 * expected_se)
    assert strata.weight.sum() == 1 and "middle" not in set(strata.stage)


def test_match_tables_require_distinct_words_and_isolate_signs_and_integer_counts():
    table = pd.DataFrame({"games": [245], "difference_pp": [-0.25], WORDS: ["regular, דוגמה"]})
    table["stage"] = pd.Categorical(["regular"])
    html = match_table(table, ["games", "difference_pp", "stage", WORDS])._repr_html_()
    assert "סדיר" in html
    assert "regular" not in html and table[WORDS].iloc[0] == "regular, דוגמה"
    assert "−0.2500" in html and 'dir="ltr"' in html and 'dir="rtl"' in html
    assert "245.0000" not in html and "מה זה, במילים" in html
    with pytest.raises(ValueError, match="different words"):
        match_table(pd.concat([table, table]), list(table))


def test_period_chart_has_disjoint_windows_and_prose_is_computed():
    periods = pd.DataFrame(
        {
            "feature": ["elo_diff", "value_avg_log_ratio"],
            "pearson_early": [0.37, 0.30],
            "pearson_recent": [0.42, 0.39],
        }
    )
    chart = period_chart(periods)
    assert chart.layout.xaxis.range == (0, 0.5)
    assert "2007/2008" in chart.data[0].name and "2019/2020" in chart.data[1].name
    assert list(chart.data[1].x) == [0.42, 0.39]
    table = periods.assign(**{WORDS: ["Elo", "שווי"]})
    html = match_table(table, ["pearson_early", "pearson_recent", WORDS])._repr_html_()
    assert "\u20662007/2008–2018/2019\u2069" in html
    tables = {
        "periods": periods.assign(change=[0.05, 0.09]),
        "bootstrap": pd.DataFrame(
            [dict(change_difference=0.04, se=0.02, descriptive_z=2, repeats=40, seed=7)]
        ),
    }
    assert "0.0400" in match_description(MatchReview(tables), "periods")


def test_saved_development_examples_match_source_counts_without_retraining():
    root = Path(__file__).resolve().parents[1]
    artifact = root / "data/derived/prematch/prematch_v1.parquet"
    if not artifact.exists():
        pytest.skip("Local saved prematch artifact unavailable")
    from ligat.analysis.prepare import load_analysis_tables
    from ligat.features.table_state import game_rows

    frame = pd.read_parquet(artifact)
    games, _ = match_view(frame, game_rows(load_analysis_tables(seasons=list(range(8, 26)))))
    cells, chi = promotion_chi_example(games)
    assert cells.observed.tolist() == [1560, 997, 1138, 187, 173, 260]
    assert chi.chi2.iloc[0] == pytest.approx(39.557589, abs=1e-6)
    assert poisson_noise_limits(games).heuristic_correlation.iloc[0] == pytest.approx(
        0.384187, abs=1e-6
    )
    summary, comparison, _ = matched_dead_games(games)
    assert comparison.matched_dead.iloc[0] == 233
    assert summary.iloc[3].draw_percent == pytest.approx(29.181191, abs=1e-6)
    assert comparison.se_pp.iloc[0] > 5  # Sparse controls must not support the claimed ~4 SE.
    sensitive, difference, strata = matched_dead_games(games, minimum_controls=20)
    assert difference.matched_dead.iloc[0] == 151
    assert sensitive.iloc[1].draw_percent == pytest.approx(100 * 32 / 151)
    assert strata.alive_n.ge(20).all()


def test_twenty_control_filter_removes_whole_strata_and_recomputes_both_sides():
    rows = []
    for stage, dead, outcomes in [
        ("regular", True, [1, 1]),
        ("regular", False, [1] * 19),
        ("upper", True, [1, 0, 0]),
        ("upper", False, [1] * 10 + [0] * 10),
        ("lower", True, [0]),
        ("lower", False, [1] * 7 + [0] * 14),
        ("middle", True, [1, 0]),
    ]:
        rows.extend(
            dict(season_id=8, stage=stage, dead_union=dead, rounds_to_stage_end=0, outcome=y)
            for y in outcomes
        )
    games = pd.DataFrame(rows)
    summary, comparison, strata = matched_dead_games(games, minimum_controls=20)
    assert set(strata.stage) == {"upper", "lower"}  # 20 is included; 19 is excluded.
    assert comparison.matched_dead.iloc[0] == 4
    assert comparison.excluded_dead.iloc[0] == 4  # Two unsupported plus two filtered.
    assert strata.set_index("stage").weight.to_dict() == {"lower": 0.25, "upper": 0.75}
    assert summary.iloc[1].draw_percent == 25
    expected_alive = 0.75 * 0.5 + 0.25 / 3
    assert summary.iloc[3].draw_percent == pytest.approx(100 * expected_alive)
    assert comparison.difference_pp.iloc[0] == pytest.approx(100 * (expected_alive - 0.25))
    expected_se = np.sqrt(0.25 * 0.75 / 4 + 0.75**2 * 0.25 / 20 + 0.25**2 * (2 / 9) / 21)
    assert comparison.se_pp.iloc[0] == pytest.approx(100 * expected_se)
    for minimum in (0, True):
        with pytest.raises(ValueError, match="positive integer"):
            matched_dead_games(games, minimum_controls=minimum)
