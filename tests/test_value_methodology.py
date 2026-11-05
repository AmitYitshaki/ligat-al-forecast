"""Hand-calculated value methodology checks; held-out seasons never enter B–E."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.value_methodology import (
    correlation_summary,
    direction_correlations,
    match_ppm,
    missing_values,
    paired_difference_summary,
    paired_value_correlations,
    parse_probe_table,
    probe_comparison,
    rank_disagreements,
    share_stability,
    squad_correlations,
    value_coverage,
    value_shares,
)


@pytest.fixture
def values():
    return pd.DataFrame(
        {
            "season_id": [8] * 3 + [9] * 3,
            "team_id": [1, 2, 3] * 2,
            "team_name": ["A", "B", "C"] * 2,
            "squad_size": [10, 20, 30] * 2,
            "total_value_eur": [10, 20, 30, 10, 30, 20],
            "avg_value_eur": [30, 20, 10, 20, 10, 30],
        }
    )


@pytest.fixture
def ppm():
    return pd.DataFrame(
        {"season_id": [8] * 3 + [9] * 3, "team_id": [1, 2, 3] * 2, "ppm": [3, 2, 1, 1, 2, 3]}
    )


def test_probe_parser_reads_first_table_and_requested_date_not_current_value():
    html = b"""<table class="items"><thead><tr><th>Club</th><th>Value 15/05/2025</th>
    <th>Current value</th></tr></thead><tbody><tr><td>
    <a href="/a/startseite/verein/1">A</a></td><td>\xe2\x82\xac2m</td><td>\xe2\x82\xac9m</td>
    </tr><tr><td><a href="/b/startseite/verein/2">B</a></td><td>-</td><td>\xe2\x82\xac7m</td>
    </tr></tbody></table><table class="items"><tr><td>ignored</td></tr></table>"""
    result = parse_probe_table(html, date="15/05/2025")
    assert result.club_name.tolist() == ["A", "B"]
    assert result.value_eur.iloc[0] == 2_000_000
    assert pd.isna(result.value_eur.iloc[1])
    season_html = html.replace(b"Value 15/05/2025", b"Total market value")
    assert_frame_equal(parse_probe_table(season_html), result)
    with pytest.raises(ValueError, match="Missing value column"):
        parse_probe_table(html, date="01/09/2025")


def test_probe_uses_intersection_of_all_pages_and_recomputes_shares():
    season = pd.DataFrame({"club_name": ["A", "B", "X"], "value_eur": [1, 3, 100]})
    may = pd.DataFrame({"club_name": ["A", "B", "Y"], "value_eur": [2, 2, 200]})
    september = pd.DataFrame({"club_name": ["A", "B", "X"], "value_eur": [2, 6, 1000]})
    result = probe_comparison(season, {"may": may, "sep": september}).set_index("date")
    assert result.clubs.tolist() == [2, 2]
    assert result.loc["may", "share_mae_pp"] == pytest.approx(25)
    assert result.loc["may", "median_ratio"] == pytest.approx(1)
    assert pd.isna(result.loc["may", "spearman"])  # tied dated values: undefined, not zero
    assert result.loc["sep", "spearman"] == pytest.approx(1)
    assert result.loc["sep", "share_mae_pp"] == pytest.approx(0)
    assert result.loc["sep", "median_ratio"] == pytest.approx(0.5)


def test_shares_preserve_missing_values_and_do_not_mutate_input(values):
    values.loc[0, ["total_value_eur", "avg_value_eur"]] = np.nan
    original = values.copy(deep=True)
    result = value_shares(values)
    assert result.share_total.isna().tolist() == [True, False, False, False, False, False]
    assert result.share_total.iloc[1] == pytest.approx(0.4)
    assert result.share_avg.iloc[1] == pytest.approx(2 / 3)
    assert_frame_equal(values, original)
    blank = value_shares(values.assign(total_value_eur=np.nan))
    assert blank.share_total.isna().all()


def test_ppm_uses_scores_not_official_points_deductions_or_halving():
    games = pd.DataFrame(
        {
            "season_id": [11, 11, 11],
            "home_team_id": [1, 2, 3],
            "away_team_id": [2, 3, 1],
            "home_goals": [1, 0, 2],
            "away_goals": [0, 0, 0],
            "points_official": [-5, 20, 30],
            "deduction": [-9, 0, -3],
            "points_halved": [True] * 3,
        }
    )
    result = match_ppm(games).set_index("team_id")
    assert result.ppm.tolist() == [1.5, 0.5, 2.0]
    assert result.points.tolist() == [3, 1, 4]
    assert result.matches.eq(2).all()
    assert_frame_equal(
        match_ppm(games),
        match_ppm(games.assign(points_official=0, deduction=100, points_halved=False)),
    )


def test_direction_is_within_season_and_keeps_all_four_comparisons(values, ppm):
    result = direction_correlations(values, ppm).set_index(["comparison", "season_id"])
    assert len(result) == 5
    assert result.loc[("share_total_s vs ppm_s+0", 8), "spearman"] == pytest.approx(-1)
    assert result.loc[("share_total_s vs ppm_s+0", 9), "spearman"] == pytest.approx(0.5)
    assert result.loc[("share_total_s vs ppm_s+1", 8), "spearman"] == pytest.approx(1)
    assert result.loc[("share_total_s vs ppm_s-1", 9), "spearman"] == pytest.approx(-0.5)
    assert result.loc[("ppm_s vs ppm_s+1", 8), "spearman"] == pytest.approx(-1)


def test_summary_weights_seasons_equally_and_does_not_replace_undefined_rho():
    correlations = pd.DataFrame(
        {
            "season_id": [8, 9, 10],
            "comparison": ["x"] * 3,
            "teams": [2, 100, 20],
            "spearman": [1, -0.5, np.nan],
        }
    )
    summary = correlation_summary(correlations).iloc[0]
    assert summary.mean_spearman == pytest.approx(0.25)
    assert summary.min_spearman == -0.5 and summary.max_spearman == 1
    assert summary.seasons == 2


def test_squad_size_correlations_and_median_are_within_each_season(values):
    result = squad_correlations(values).set_index("season_id")
    assert result.loc[8, "median_squad_size"] == 20
    assert result.loc[8, "squad_vs_total"] == pytest.approx(1)
    assert result.loc[8, "squad_vs_avg"] == pytest.approx(-1)
    assert result.loc[8, "total_vs_avg"] == pytest.approx(-1)


def test_paired_sum_minus_average_and_dispersion_are_season_level(values, ppm):
    paired = paired_value_correlations(values, ppm)
    assert len(paired) == 1
    assert paired.rho_total.iloc[0] == pytest.approx(1)
    assert paired.rho_avg.iloc[0] == pytest.approx(-1)
    assert paired.difference.iloc[0] == pytest.approx(2)
    summary = paired_difference_summary(
        pd.DataFrame({"season_id": [8, 9], "difference": [0, 2]})
    ).iloc[0]
    assert summary["mean"] == 1
    assert summary["std"] == pytest.approx(np.sqrt(2))
    assert summary.se == pytest.approx(1)
    assert summary.seasons == 2


def test_paired_correlations_use_identical_nonmissing_clubs(values, ppm):
    values.loc[0, "avg_value_eur"] = np.nan
    result = paired_value_correlations(values, ppm).iloc[0]
    assert result.teams == 2
    assert result.difference == pytest.approx(2)


def test_rank_gaps_are_absolute_and_ties_have_average_ranks(values):
    result = rank_disagreements(values)
    assert len(result) == 5
    assert result.iloc[0].season_id == 8 and result.iloc[0].team_id == 1
    assert result.iloc[0].rank_total == 3 and result.iloc[0].rank_avg == 1
    assert result.iloc[0].rank_gap == 2 and result.iloc[0].squad_size == 10
    tied = values.copy()
    tied.loc[tied.season_id.eq(8), "total_value_eur"] = 10
    result = rank_disagreements(tied, limit=6)
    assert result.loc[result.season_id.eq(8), "rank_total"].eq(2).all()


def test_coverage_uses_played_rosters_and_missing_source_stays_missing(values, ppm):
    values.loc[2, ["total_value_eur", "avg_value_eur"]] = np.nan
    values.loc[4, "team_id"] = 4
    ppm.loc[4, "team_id"] = 4
    coverage = value_coverage(values, ppm).set_index("season_id")
    assert pd.isna(coverage.loc[8, "promoted"])
    assert pd.isna(coverage.loc[8, "missing_other"])
    assert coverage.loc[9, "teams"] == 3
    assert coverage.loc[9, "with_previous_value"] == 1
    assert coverage.loc[9, "promoted"] == 1
    assert coverage.loc[9, "missing_other"] == 1
    missing = missing_values(values, ppm).iloc[0]
    assert missing.team_id == 3 and missing.team_name == "C"
    assert missing.played_next_season
    assert "avg_value_eur" in missing.missing_fields
    assert pd.isna(missing.total_value_eur)


def test_stability_joins_continuing_teams_and_reports_both_league_sizes(values):
    result = share_stability(values).iloc[0]
    assert result.spearman == pytest.approx(0.5)
    assert result.teams == result.source_teams == result.target_teams == 3


def test_complete_values_produce_no_missing_rows(values, ppm):
    result = missing_values(values, ppm)
    assert result.empty
    assert "played_next_season" in result


@pytest.mark.parametrize(
    "function",
    [value_shares, squad_correlations, rank_disagreements, share_stability],
)
def test_value_analyses_reject_held_out_seasons(function, values):
    with pytest.raises(ValueError, match="8–25"):
        function(values.assign(season_id=26))


@pytest.mark.parametrize(
    "function", [direction_correlations, paired_value_correlations, missing_values, value_coverage]
)
@pytest.mark.parametrize("side", ["values", "ppm"])
def test_two_input_analyses_reject_either_held_out_input(function, side, values, ppm):
    with pytest.raises(ValueError, match="8–25"):
        function(
            values.assign(season_id=26) if side == "values" else values,
            ppm.assign(season_id=26) if side == "ppm" else ppm,
        )


@pytest.mark.parametrize("function", [match_ppm, correlation_summary, paired_difference_summary])
def test_remaining_development_functions_reject_held_out_seasons(function):
    with pytest.raises(ValueError, match="8–25"):
        function(pd.DataFrame({"season_id": [26]}))
