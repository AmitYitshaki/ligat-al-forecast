"""Source artifacts and intermediate display ordering never weaken sporting rules."""

from dataclasses import replace

import pandas as pd
import pytest

from ligat.analysis.prematch_validation import (
    UnexplainedStandingsError,
    _check_display_order_limit,
    _counter_discrepancies,
    _position_comparison,
    load_technical_ledger,
    validate_prematch_tables,
)
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.data.repository import Repository
from ligat.features.table_state import historical_groups
from ligat.simulate.rules import load_season_rules, sporting_table


def counters() -> tuple[pd.Series, pd.Series]:
    calculated = pd.Series(
        dict(
            games_played=1,
            wins=0,
            draws=1,
            losses=0,
            goals_for=1,
            goals_against=1,
            points=1,
            goal_difference=0,
        )
    )
    official = pd.Series(
        dict(
            played=1,
            won=0,
            drawn=1,
            lost=0,
            goals_for=0,
            goals_against=0,
            points_model=1,
            goal_difference=0,
        )
    )
    return calculated, official


def test_season8_equal_goal_offsets_are_a_source_artifact() -> None:
    calculated, official = counters()
    assert _counter_discrepancies(calculated, official, 8) == ([], True)


@pytest.mark.parametrize("season", [9, 25])
def test_equal_goal_offsets_in_other_seasons_are_not_exempt(season: int) -> None:
    calculated, official = counters()
    errors, artifact = _counter_discrepancies(calculated, official, season)
    assert set(errors) == {"goals_for", "goals_against"}
    assert not artifact


def test_unequal_goal_offsets_or_other_counters_still_fail() -> None:
    calculated, official = counters()
    official["goals_against"] = 1
    official["goal_difference"] = -1
    errors, artifact = _counter_discrepancies(calculated, official, 8)
    assert set(errors) == {"goals_for", "goal_difference"}
    assert not artifact
    calculated, official = counters()
    official["points_model"] = 0
    errors, artifact = _counter_discrepancies(calculated, official, 8)
    assert errors == ["points"] and artifact


def display_table() -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            points=[7, 3, 3, 3],
            goal_difference=[0, 1, 1, 0],
            position=[1.0, 2.0, 3.0, 4.0],
            table_group="regular",
        ),
        index=[1, 2, 3, 4],
    )


def test_intermediate_display_reordering_only_inside_equal_points_and_gd() -> None:
    table = display_table()
    assert _position_comparison(table, 2, 3, True) == (True, True)
    assert _position_comparison(table, 3, 2, True) == (True, True)
    assert _position_comparison(table, 2, 3, False) == (False, False)
    assert _position_comparison(table, 2, 1, True) == (False, False)
    assert _position_comparison(table, 2, 4, True) == (False, False)


def test_playoff_offsets_and_average_ties_remain_inside_their_block() -> None:
    table = display_table()
    table["position"] += 6
    table["table_group"] = "lower"
    table.loc[[2, 3], "position"] = 8.5
    assert _position_comparison(table, 2, 8, False) == (True, False)
    assert _position_comparison(table, 2, 9, False) == (True, False)
    assert _position_comparison(table, 2, 10, True) == (False, False)


def test_one_percent_gate_counts_only_intermediate_rows() -> None:
    checks = pd.DataFrame(
        dict(scope="round", intermediate=True, display_order=[True] + [False] * 99)
    )
    _check_display_order_limit(checks)
    checks.loc[1, "display_order"] = True
    finals = pd.DataFrame(dict(scope=["final"] * 1000, intermediate=False, display_order=False))
    with pytest.raises(UnexplainedStandingsError, match="exceeds 1%"):
        _check_display_order_limit(pd.concat([checks, finals], ignore_index=True))


def test_all_development_tables_and_verified_artifact_pattern(real_tables: AnalysisTables) -> None:
    repo = Repository()
    seasons = list(DEVELOPMENT_SEASONS)
    rounds = repo.read_filtered("round_tables", [("season_id", "in", seasons)])
    finals = repo.read_filtered("final_tables", [("season_id", "in", seasons)])
    rules = load_season_rules()
    validation = validate_prematch_tables(
        real_tables, rules, rounds, finals, load_technical_ledger(seasons)
    )
    assert len(validation.checks.loc[validation.checks.scope.eq("round")]) == len(rounds)
    artifacts = validation.differences.loc[validation.differences.category.eq("source_artifact")]
    assert len(artifacts) == 157
    assert artifacts.season_id.eq(8).all()
    assert artifacts.delta_goals_for.eq(artifacts.delta_goals_against).all()
    assert artifacts.delta_goals_for.ne(0).all()
    display = validation.differences.loc[validation.differences.category.eq("display_order")]
    pair = display.loc[display.season_id.eq(15) & display["round"].eq(2)]
    assert set(pair.team_id) == {1005, 2182}
    assert set(validation.split_differences.team_id) == {2171, 3595}
    # All 198 source scores in season 8 also agree with the independent goal events.
    matches = real_tables.match_table.loc[real_tables.match_table.season_id.eq(8)]
    event_goals = (
        real_tables.goal_events.groupby(level="game_id").size().reindex(matches.index, fill_value=0)
    )
    assert len(matches) == 198
    assert matches.total_goals.eq(event_goals).all()


@pytest.mark.parametrize("season,teams", [(9, (1061, 2176)), (14, (2184, 4823))])
def test_real_end_of_season_ties_match_official_order(
    real_tables: AnalysisTables, season: int, teams: tuple[int, int]
) -> None:
    games = real_tables.team_matches.reset_index()
    games = games.loc[games.season_id.eq(season)]
    rule = load_season_rules()[season]
    groups = historical_groups(games) if rule.groups else None
    table = sporting_table(games, sorted(games.team_id.unique()), rule, groups=groups)
    official = (
        Repository()
        .read_filtered("final_tables", [("season_id", "==", season)])
        .set_index("team_id")
    )
    first, second = teams
    assert table.loc[first, "points"] == table.loc[second, "points"]
    assert table.loc[first, "goal_difference"] > table.loc[second, "goal_difference"]
    assert table.loc[first, "position"] == official.loc[first, "position"]
    assert table.loc[second, "position"] == official.loc[second, "position"]


def test_full_gate_rejects_an_unexplained_counter_change() -> None:
    rows = []
    for team, opponent, gf, ga in [(1, 2, 1, 0), (2, 1, 0, 1)]:
        rows.append(
            dict(
                team_id=team,
                game_id=1,
                season_id=8,
                opponent_id=opponent,
                goals_scored=gf,
                goals_conceded=ga,
                is_home=team == 1,
                date=pd.Timestamp("2006-08-01", tz="Asia/Jerusalem"),
                round=1,
                stage="regular",
            )
        )
    tables = AnalysisTables(
        pd.DataFrame(),
        pd.DataFrame(rows).set_index(["team_id", "game_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    official = pd.DataFrame(
        dict(
            team_id=[1, 2],
            season_id=8,
            round=1,
            played=1,
            won=[1, 0],
            drawn=0,
            lost=[0, 1],
            goals_for=[1, 0],
            goals_against=[0, 1],
            points_model=[3, 0],
            points_official=[3, 0],
            deduction=0,
            position=[1, 2],
        )
    )
    finals = official.assign(final_stage="regular", relegated=[False, True])
    rules = {8: replace(load_season_rules()[8], n_teams=2, regular_rounds=1, relegated=1)}
    validate_prematch_tables(tables, rules, official, finals, pd.DataFrame())
    official.loc[0, "won"] = 2
    with pytest.raises(UnexplainedStandingsError, match="wins"):
        validate_prematch_tables(tables, rules, official, finals, pd.DataFrame())
