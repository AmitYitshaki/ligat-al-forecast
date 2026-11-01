"""Chapter C: fractional ties, protected regular-only expectation and stratified inference."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from ligat.analysis.league_stats import course_review as review
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, load_playoff_upper_sizes


@pytest.mark.parametrize(
    "function",
    [
        review.rank_snapshots,
        review.rank_heatmap,
        review.region_persistence,
        review.leader_benchmarks,
        review.regular_strength_expectation,
        review.stage_strength_comparison,
        review.draw_strata,
        review.combined_draw_test,
        review.playoff_noise_diagnostics,
    ],
)
def test_rejects_holdout_before_access(function):
    with pytest.raises(ValueError):
        function(None, seasons=[26])


def test_other_season_guards():
    for function in (review.settling_review, review.split_diagnostics):
        with pytest.raises(ValueError):
            function(None, seasons=[27], upper_sizes={})
    with pytest.raises(ValueError):
        review.position_credits(pd.DataFrame({"season_id": [26]}))
    with pytest.raises(ValueError):
        review.mantel_haenszel(
            pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(26, "regular")], names=["season_id", "stage"])
            )
        )


def test_fractional_ties_cross_all_region_boundaries():
    ranks = [1, 3, 3, 3, 5, 6.5, 6.5, 8, 9, 10.5, 10.5, 12, 13, 14]
    frame = pd.DataFrame(dict(season_id=14, k=1, team_id=range(1, 15), rank=ranks))
    expanded = review.position_credits(frame)
    np.testing.assert_allclose(expanded.groupby("team_id").weight.sum(), 1)
    np.testing.assert_allclose(expanded.groupby("position").weight.sum(), 1)
    assert expanded.loc[expanded.team_id.eq(2), "position"].tolist() == [2, 3, 4]
    for team, expected in [
        (2, {"top": 2 / 3, "upper_line": 1 / 3}),
        (6, {"upper_line": 0.5, "middle_table": 0.5}),
        (10, {"middle_table": 0.5, "bottom": 0.5}),
    ]:
        group = expanded.loc[expanded.team_id.eq(team)]
        weights = {
            name: group.loc[group.position.between(low, high), "weight"].sum()
            for low, high, name in review.REGIONS
        }
        for key, value in expected.items():
            assert weights[key] == pytest.approx(value)
        assert sum(weights.values()) == pytest.approx(1)


def test_random_displacement_matches_all_permutations():
    for n in range(1, 7):
        observed = np.mean(
            [np.mean(np.abs(np.arange(n) - np.array(order))) for order in permutations(range(n))]
        )
        assert review.random_mean_movement(n) == pytest.approx(observed)
    with pytest.raises(ValueError):
        review.random_mean_movement(0)


def test_mantel_haenszel_matches_manual_stratification():
    cells = pd.DataFrame(
        [[3, 7, 8, 12], [4, 6, 12, 8], [1, 9, 5, 15]],
        columns=["a", "b", "c", "d"],
        index=pd.MultiIndex.from_tuples(
            [(14, "regular"), (14, "upper"), (15, "regular")], names=["season_id", "stage"]
        ),
    )
    a, b, c, d = cells.to_numpy(dtype=float).T
    n = a + b + c + d
    expected = ((a * d / n).sum()) / ((b * c / n).sum())
    residual = (a - (a + b) * (a + c) / n).sum()
    variance = ((a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))).sum()
    result = review.mantel_haenszel(cells)
    assert result["odds_ratio"] == pytest.approx(expected)
    assert result["p"] == pytest.approx(chi2.sf(residual**2 / variance, 1))
    assert result["n_strata"] == 3  # Same season, different stages remain separate.


def test_transition_counts_and_regions_real(real_tables):
    for split in (False, True):
        matrix, expanded = review.rank_heatmap(real_tables, at_split=split)
        np.testing.assert_allclose(matrix.sum(axis=0), 12)
        np.testing.assert_allclose(matrix.sum(axis=1), 12)
        np.testing.assert_allclose(expanded.groupby(["season_id", "team_id"]).weight.sum(), 1)
        assert matrix.to_numpy().sum() == pytest.approx(168)
    matrix, _ = review.rank_heatmap(real_tables)
    assert matrix.loc[1:3, 1:3].sum().sum() == 26
    regions = review.region_persistence(real_tables)
    assert regions.loc[("top", 10), "same_region_rate"] == pytest.approx(26 / 36)
    first = regions.loc[regions.same_region_rate.ge(0.8)].reset_index().groupby("region").k.min()
    assert first.to_dict() == dict(top=14, upper_line=23, middle_table=26, bottom=23)


def test_playoff_result_cannot_change_expected_goals(real_tables):
    matches = real_tables.match_table.copy(deep=True)
    team_games = real_tables.team_matches.copy(deep=True)
    matches.loc[matches.stage.ne("regular"), ["home_goals", "away_goals", "total_goals"]] = 9
    team_games.loc[team_games.stage.ne("regular"), ["goals_scored", "goals_conceded"]] = 9
    changed = replace(real_tables, match_table=matches, team_matches=team_games)
    original = review.regular_strength_expectation(real_tables)
    after = review.regular_strength_expectation(changed)
    pd.testing.assert_series_equal(original.expected_goals, after.expected_goals)
    regular = original.loc[original.stage.eq("regular")]
    assert regular.expected_goals.mean() == pytest.approx(regular.total_goals.mean(), abs=0.02)
    summary, effects = review.stage_strength_comparison(real_tables)
    assert summary.loc["upper", "expected_goals"] == pytest.approx(2.463242, abs=1e-6)
    assert effects.loc[("lower", "excess_strength"), "mean"] == pytest.approx(0.337972, abs=1e-6)


def test_strata_and_benchmark_results_real(real_tables):
    cells = review.draw_strata(real_tables)
    assert cells.index.is_unique and len(cells) == 36
    assert "middle" not in cells.index.get_level_values("stage")
    result = review.combined_draw_test(real_tables)
    assert result.loc["all", "odds_ratio"] == pytest.approx(0.643953, abs=1e-6)
    assert result.attrs["same_direction_seasons"] == 11
    groups, facts = review.split_diagnostics(real_tables, upper_sizes=load_playoff_upper_sizes())
    assert groups.loc["upper", "mean_movement"] == pytest.approx(26 / 72)
    assert facts.loc["last_two_relegated", "success"] == 17
    settling = review.settling_review(
        real_tables, DEVELOPMENT_SEASONS, upper_sizes=load_playoff_upper_sizes()
    )
    assert settling.loc[("14_teams", 10), "spearman_mean"] == pytest.approx(0.68315, abs=1e-5)
    for era in ("12_teams", "16_teams"):
        assert settling.xs(era).spearman_ci_lower.isna().all()


def test_review_plot_contracts(real_tables):
    import re

    from ligat.analysis.league_stats import final_rounds_check, season_points_race
    from ligat.viz.course_review import region_chart, settling_chart, transition_chart
    from ligat.viz.league_stats import AWAY_COLOR, HOME_COLOR
    from ligat.viz.season_course import chapter_c_chart

    matrix, _ = review.rank_heatmap(real_tables)
    chart = transition_chart(matrix)
    assert chart.data[0].zmax == 12 and chart.layout.yaxis.range == (14.5, 0.5)
    assert len(region_chart(review.region_persistence(real_tables)).data) == 4
    settling = review.settling_review(
        real_tables, DEVELOPMENT_SEASONS, upper_sizes=load_playoff_upper_sizes()
    )
    assert settling_chart(settling).layout.yaxis.range == (0, 1)
    last = final_rounds_check(real_tables, stages=("regular", "upper", "lower"))
    chart = chapter_c_chart("last_rounds", last)
    for trace in chart.data:
        name = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", trace.name)
        assert trace.marker.color == (AWAY_COLOR if name == "3 המחזורים האחרונים" else HOME_COLOR)
        assert "אמצעי" not in "".join(trace.x)
    race = season_points_race(real_tables, [25]).join(
        real_tables.team_seasons[["final_position"]], on=["season_id", "team_id"]
    )
    figure = chapter_c_chart("race", race)
    assert sum(bool(t.showlegend) for t in figure.data) == 4
    noise = review.playoff_noise_diagnostics(real_tables)
    assert noise.loc["upper", "regular_sd"] > noise.loc["lower", "regular_sd"]
