"""Descriptive invariants on synthetic seasons and optional local clean data."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ligat.analysis import league_stats as stats
from ligat.analysis.prepare import (
    DEVELOPMENT_SEASONS,
    AnalysisTables,
    build_team_seasons,
    load_analysis_tables,
)
from ligat.data.repository import Repository
from ligat.viz.league_stats import chapter_a_chart

FUNCTIONS = [
    stats.team_season_stats,
    stats.position_profile,
    stats.season_persistence,
    stats.persistence_summary,
    stats.value_vs_performance,
    stats.value_summary,
    stats.promoted_value_check,
    stats.promoted_value_summary,
    stats.team_consistency,
    stats.team_style_profile,
]

BOOTSTRAP_FUNCTIONS = {
    stats.position_profile,
    stats.persistence_summary,
    stats.value_summary,
    stats.team_consistency,
    stats.team_style_profile,
}


def _run(function: Any, tables: AnalysisTables) -> pd.DataFrame:
    options = {"n_bootstrap": 50} if function in BOOTSTRAP_FUNCTIONS else {}
    return function(tables, **options)


@pytest.fixture
def synthetic_tables() -> AnalysisTables:
    games = []
    finals = []
    values = []
    for season in range(8, 13):
        for game, (home, away) in enumerate(((1, 2), (2, 1), (1, 2), (2, 1))):
            home_goals, away_goals = ((4, 0), (1, 2), (1, 1), (0, 0))[game]
            for team, opponent, scored, conceded in (
                (home, away, home_goals, away_goals),
                (away, home, away_goals, home_goals),
            ):
                diff = scored - conceded
                result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
                games.append(
                    dict(
                        season_id=season,
                        team_id=team,
                        game_id=season * 10 + game,
                        opponent_id=opponent,
                        date=pd.Timestamp(f"{1998 + season}-09-01") + pd.Timedelta(days=game),
                        team_name=f"Team {team}",
                        is_home=team == home,
                        goals_scored=scored,
                        goals_conceded=conceded,
                        goal_difference=diff,
                        result=result,
                        points_earned=3 if diff > 0 else 0 if diff < 0 else 1,
                        is_clean_sheet=conceded == 0,
                        failed_to_score=scored == 0,
                        ht_state="level",
                        has_timeline=True,
                        red_cards=0,
                        coach_id=pd.NA,
                        coach_name=pd.NA,
                        era="12_teams",
                    )
                )
        for team in (1, 2):
            finals.append(
                dict(
                    season_id=season,
                    team_id=team,
                    position=team,
                    points_model=0,
                    points_official=0,
                    deduction=99,
                    final_stage="regular",
                    relegated=team == 2,
                    promoted_in=pd.NA if season == 8 else team == 2 and season in (9, 11),
                )
            )
            values.append(
                dict(
                    season_id=season,
                    team_id=team,
                    squad_size=20 + team,
                    avg_value_eur=10,
                    total_value_eur=(30 + season) if team == 1 else 10,
                )
            )
    teams = pd.DataFrame(games).set_index(["team_id", "game_id"], drop=True)
    seasons = build_team_seasons(teams, pd.DataFrame(finals), pd.DataFrame(values))
    summary = pd.DataFrame({"season_id": range(8, 13), "n_teams": 2}).set_index("season_id")
    empty = pd.DataFrame()
    return AnalysisTables(empty, teams, empty, seasons, summary, empty)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_development_scope(
    function: Any, seasons: list[int], synthetic_tables: AnalysisTables
) -> None:
    with pytest.raises(ValueError, match="development"):
        function(synthetic_tables, seasons)


def test_known_score_rates_and_input_immutability(synthetic_tables: AnalysisTables) -> None:
    before = synthetic_tables.team_matches.copy(deep=True)
    output = stats.team_season_stats(synthetic_tables)
    one, two = output.loc[(8, 1)], output.loc[(8, 2)]
    assert one.win_rate == 0.5 and one.draw_rate == 0.5 and one.loss_rate == 0
    assert one.big_win_rate == 0.25 and one.narrow_win_rate == 0.25
    assert one.home_win_rate == 0.5 and one.away_win_rate == 0.5
    assert one.btts_rate == 0.5 and one.over_2_5_rate == 0.5
    assert two.heavy_loss_rate == 0.25
    assert one.points_per_match == 2  # No official deduction or halving enters rates.
    assert output[["win_rate", "draw_rate", "loss_rate"]].sum(axis=1).eq(1).all()
    assert (output.wins + output.draws + output.losses).eq(output.matches_played).all()
    assert (output.big_win_rate + output.narrow_win_rate).le(output.win_rate).all()
    for rate in stats.RATE_FLAGS:
        assert output[f"{rate}_ci_lower"].le(output[rate]).all()
        assert output[f"{rate}_ci_upper"].ge(output[rate]).all()
    for function in FUNCTIONS:
        frame = _run(function, synthetic_tables)
        assert frame.index.is_unique
        assert not set(frame.index.names) & set(frame.columns)
    pd.testing.assert_frame_equal(before, synthetic_tables.team_matches)


def test_previous_season_pairing_does_not_bridge_gaps(synthetic_tables: AnalysisTables) -> None:
    seasons = synthetic_tables.team_seasons.drop(index=(10, 2))
    tables = replace(synthetic_tables, team_seasons=seasons)
    pairs = stats.season_persistence(tables)
    assert 8 not in pairs.index.get_level_values("season_id")
    assert (11, 2) not in pairs.index
    assert (12, 2) in pairs.index
    # Prior development seasons provide context even when only t is requested.
    subset = stats.season_persistence(tables, [8, 10])
    assert subset.index.tolist() == [(10, 1)]


def test_value_regression_missing_and_constant_inputs(synthetic_tables: AnalysisTables) -> None:
    seasons = synthetic_tables.team_seasons.copy()
    seasons["prev_season_value_share"] = 0.5
    result = stats.value_vs_performance(replace(synthetic_tables, team_seasons=seasons))
    assert result.expected_ppm.isna().all()
    assert result.over_under.isna().all()
    seasons["prev_season_value_share"] = pd.NA
    summary = stats.value_summary(replace(synthetic_tables, team_seasons=seasons), n_bootstrap=50)
    assert summary.loc["prev_season_value_share", "n"] == 0


def test_promoted_benchmark_and_consistency(synthetic_tables: AnalysisTables) -> None:
    promoted = stats.promoted_value_check(synthetic_tables)
    assert promoted.index.tolist() == [(9, 2), (11, 2)]
    assert promoted["diff"].lt(0).all() and promoted.ratio.lt(1).all()
    assert stats.promoted_value_summary(synthetic_tables).loc["promoted", "fraction_below"] == 1
    consistent = stats.team_consistency(synthetic_tables, n_bootstrap=50)
    assert consistent.loc[2, "promotions"] == 2 and consistent.loc[2, "is_yoyo"]
    assert stats.team_consistency(synthetic_tables, [8, 9], n_bootstrap=50).std_ppm.isna().all()
    assert len(stats.team_style_profile(synthetic_tables, n_bootstrap=50)) == 2
    assert stats.team_style_profile(synthetic_tables, [8, 9], n_bootstrap=50).empty


def test_wilson_edge_cases() -> None:
    assert stats.rate_interval(0, 10)[0] == 0
    assert stats.rate_interval(10, 10)[0] == 1
    assert pd.isna(stats.rate_interval(0, 0)[0])
    with pytest.raises(ValueError):
        stats.rate_interval(11, 10)


def test_promoted_reference_is_previous_season_and_excludes_current_entrants(
    synthetic_tables: AnalysisTables,
) -> None:
    rows = [
        (8, 1, 0.1, pd.NA),
        (8, 2, 0.2, pd.NA),
        (8, 3, 0.3, pd.NA),
        (8, 4, 0.4, pd.NA),
        (9, 1, 0.2, False),
        (9, 2, 0.3, False),
        (9, 5, 0.5, True),
    ]
    seasons = pd.DataFrame(
        rows, columns=["season_id", "team_id", "value_share_of_league", "promoted_in"]
    )
    seasons["team_name"] = "Fixture"
    seasons["promoted_in"] = seasons.promoted_in.astype("boolean")
    seasons = seasons.set_index(["season_id", "team_id"], drop=True)
    tables = replace(synthetic_tables, team_seasons=seasons)
    check = stats.promoted_value_check(tables, [9]).loc[(9, 5)]
    assert check.imputed_share == pytest.approx(0.15)
    assert check.same_season_share_excl_promoted == pytest.approx(0.2)
    assert check["diff"] == pytest.approx(0.35)
    assert (8, 5) not in seasons.index  # Entrant is absent from the previous league reference.
    seasons.loc[(9, 5), "value_share_of_league"] = 0.9
    changed = stats.promoted_value_check(tables, [9]).loc[(9, 5)]
    assert changed.imputed_share == check.imputed_share
    assert changed.same_season_share_excl_promoted == check.same_season_share_excl_promoted
    seasons.loc[(9, 1), "value_share_of_league"] = 0.4
    assert (
        stats.promoted_value_check(tables, [9]).loc[(9, 5), "imputed_share"] == check.imputed_share
    )


@pytest.mark.parametrize("function", list(BOOTSTRAP_FUNCTIONS))
def test_bootstrap_count_is_configurable(function: Any, synthetic_tables: AnalysisTables) -> None:
    small = function(synthetic_tables, n_bootstrap=50)
    repeated = function(synthetic_tables, n_bootstrap=50)
    pd.testing.assert_frame_equal(small, repeated)
    with pytest.raises(ValueError, match="n_bootstrap"):
        function(synthetic_tables, n_bootstrap=0)


@pytest.fixture(scope="module")
def real_tables() -> AnalysisTables:
    if not Path("data/clean/squad_values.parquet").exists():
        pytest.skip("Local clean data are deliberately not committed")
    repo = Repository()
    original = repo.read_filtered
    calls = []

    def filtered(name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        calls.append((name, filters))
        for column, _, values in filters:
            if column == "season_id":
                assert all(7 <= season <= 25 for season in values)
        return original(name, filters)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(repo, "read_filtered", filtered)
        patch.setattr(repo, "read", lambda _: pytest.fail("Unfiltered read"))
        tables = load_analysis_tables(repo, seasons=DEVELOPMENT_SEASONS)
    assert calls
    return tables


def test_real_counts_and_positions(real_tables: AnalysisTables) -> None:
    frame = stats.team_season_stats(real_tables)
    assert len(real_tables.match_table) == 4315
    assert len(real_tables.team_matches) == 8630
    assert len(frame) == 252
    assert frame.final_position.eq(1).sum() == 18
    assert frame.relegated.sum() == 37
    assert (
        frame.groupby(level="season_id")
        .apply(lambda g: sorted(g.final_position) == list(range(1, int(g.n_teams.iloc[0]) + 1)))
        .all()
    )
    assert (frame.wins + frame.draws + frame.losses).eq(frame.matches_played).all()
    assert (frame.win_rate + frame.draw_rate + frame.loss_rate - 1).abs().lt(1e-12).all()
    assert (frame.big_win_rate + frame.narrow_win_rate).le(frame.win_rate + 1e-12).all()
    assert (
        (frame.points_from_matches / frame.matches_played - frame.points_per_match)
        .abs()
        .lt(1e-12)
        .all()
    )
    assert set(
        stats.position_profile(real_tables, n_bootstrap=50).index.get_level_values("era")
    ) == {
        "12_teams",
        "16_teams",
        "14_teams",
    }
    pairs = stats.season_persistence(real_tables)
    assert 8 not in pairs.index.get_level_values("season_id")
    for season, team in pairs.index:
        assert (season - 1, team) in frame.index
    promoted = stats.promoted_value_check(real_tables)
    assert frame.loc[promoted.index].promoted_in.eq(True).all()
    promoted_summary = stats.promoted_value_summary(real_tables).loc["promoted"]
    assert promoted_summary["n"] == 36
    assert promoted_summary.fraction_below == pytest.approx(26 / 36)
    assert promoted_summary.median_diff == pytest.approx(-0.0039, abs=0.0001)
    for function in BOOTSTRAP_FUNCTIONS:
        assert not function(real_tables, n_bootstrap=50).empty


def test_heldout_rows_cannot_change_results(synthetic_tables: AnalysisTables) -> None:
    heldout = synthetic_tables.team_seasons.reset_index().copy()
    heldout["season_id"] = 26
    heldout["points_per_match"] = 999
    heldout = heldout.drop_duplicates(["season_id", "team_id"]).set_index(
        ["season_id", "team_id"], drop=True
    )
    contaminated = replace(
        synthetic_tables, team_seasons=pd.concat([synthetic_tables.team_seasons, heldout])
    )
    for function in FUNCTIONS:
        pd.testing.assert_frame_equal(
            _run(function, synthetic_tables), _run(function, contaminated)
        )


def test_chart_contracts(synthetic_tables: AnalysisTables) -> None:
    source = stats.team_season_stats(synthetic_tables)
    source["era"] = "14_teams"  # Presentation fixture, not an actual league-format assertion.
    pairs = stats.season_persistence(synthetic_tables)
    values = stats.value_vs_performance(synthetic_tables)
    cases = [
        ("position", source, None),
        ("attack_defense", source, None),
        ("home_away", source, None),
        ("win_style", source, None),
        ("open_style", stats.team_style_profile(synthetic_tables, n_bootstrap=50), None),
        ("persistence", pairs, stats.persistence_summary(synthetic_tables, n_bootstrap=50)),
        ("value", values, stats.value_summary(synthetic_tables, n_bootstrap=50)),
        ("promoted", stats.promoted_value_check(synthetic_tables), None),
        ("consistency", stats.team_consistency(synthetic_tables, n_bootstrap=50), None),
    ]
    for kind, frame, summary in cases:
        figure = chapter_a_chart(kind, frame, summary)
        assert figure.data
        assert figure.layout.title.text
        assert figure.to_json()


def test_previous_value_residuals_and_era_separation(synthetic_tables: AnalysisTables) -> None:
    result = stats.value_vs_performance(synthetic_tables)
    complete = result.dropna(subset=["over_under"])
    assert len(complete) < len(result)  # First season and promoted-club values are unknown.
    assert complete.over_under.sum() == pytest.approx(0, abs=1e-10)
    seasons = synthetic_tables.team_seasons.copy()
    seasons.loc[(10, 2), "value_share_of_league"] = pd.NA
    summary = stats.promoted_value_summary(replace(synthetic_tables, team_seasons=seasons))
    assert summary.loc["promoted", "n"] == 1  # Missing t-1 reference leaves its benchmark unknown.
    assert summary.loc["promoted", "n_missing"] == 1


def test_loading_filters_never_include_holdouts(real_tables: AnalysisTables) -> None:
    # The module fixture validates every read; reuse its guarded load rather than loading twice.
    assert real_tables.match_table.season_id.max() == 25
