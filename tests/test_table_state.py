"""Prematch causality: paired rows, calendar-day isolation and registry enforcement."""

from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prepare import AnalysisTables
from ligat.features.registry import FEATURES, validate_feature_columns
from ligat.features.table_state import TableStateBlock, _snapshot_features, historical_groups
from ligat.simulate.rules import load_season_rules, split_playoffs, sporting_table


def synthetic_tables() -> AnalysisTables:
    rows = []
    for game, home, away, day, home_goals, away_goals in [
        (1, 1, 2, "01 14:00", 1, 0),
        (2, 3, 4, "01 23:00", 0, 2),
        (3, 1, 3, "08 14:00", 1, 1),
        (4, 2, 4, "08 20:00", 0, 1),
        (5, 1, 4, "15 14:00", 0, 0),
        (6, 2, 3, "15 20:00", 2, 0),
    ]:
        for team, opponent, gf, ga, is_home in [
            (home, away, home_goals, away_goals, True),
            (away, home, away_goals, home_goals, False),
        ]:
            rows.append(
                dict(
                    game_id=game,
                    team_id=team,
                    opponent_id=opponent,
                    season_id=8,
                    date=pd.Timestamp("2006-08-" + day, tz="Asia/Jerusalem"),
                    round=(game + 1) // 2,
                    stage="regular",
                    is_home=is_home,
                    goals_scored=gf,
                    goals_conceded=ga,
                )
            )
    return AnalysisTables(
        pd.DataFrame(),
        pd.DataFrame(rows).set_index(["team_id", "game_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def block() -> TableStateBlock:
    return TableStateBlock(
        {8: replace(load_season_rules()[8], n_teams=4, regular_rounds=3, relegated=1)}
    )


def test_result_change_leaves_present_and_past_but_changes_both_next_games() -> None:
    tables = synthetic_tables()
    original = tables.team_matches.copy(deep=True)
    before = block().compute(tables)
    changed = tables.team_matches.copy(deep=True)
    changed.loc[(1, 1), ["goals_scored", "goals_conceded"]] = [0, 2]
    changed.loc[(2, 1), ["goals_scored", "goals_conceded"]] = [2, 0]
    after = block().compute(replace(tables, team_matches=changed))
    assert_frame_equal(before.loc[[1, 2]], after.loc[[1, 2]])
    assert before.loc[(3, 1), "points"] == 3
    assert after.loc[(3, 1), "points"] == 0
    assert before.loc[(4, 2), "points"] == 0
    assert after.loc[(4, 2), "points"] == 3
    assert_frame_equal(tables.team_matches, original)
    assert before.index.names == ["game_id", "team_id"]


def test_later_game_change_leaves_every_earlier_row_unchanged() -> None:
    tables = synthetic_tables()
    before = block().compute(tables)
    changed = tables.team_matches.copy(deep=True)
    changed.loc[(1, 3), "goals_scored"] = 9
    changed.loc[(3, 3), "goals_conceded"] = 9
    after = block().compute(replace(tables, team_matches=changed))
    assert_frame_equal(before.loc[[1, 2, 3, 4]], after.loc[[1, 2, 3, 4]])


def test_same_israeli_calendar_day_even_when_utc_date_differs() -> None:
    tables = synthetic_tables()
    tables.team_matches.loc[(1, 1), "date"] = pd.Timestamp("2006-08-01 00:05", tz="Asia/Jerusalem")
    tables.team_matches.loc[(2, 1), "date"] = pd.Timestamp("2006-08-01 00:05", tz="Asia/Jerusalem")
    frame = block().compute(tables)
    assert frame.loc[[1, 2], "games_played"].eq(0).all()
    assert frame.loc[[1, 2], "as_of"].nunique() == 1
    assert tables.team_matches.loc[(1, 1), "date"].tz_convert("UTC").day == 31


def test_no_official_summary_or_eda_feature_input_and_remaining_from_format() -> None:
    tables = synthetic_tables()
    before = block().compute(tables)
    contaminated = tables.team_matches.assign(
        points_before=9999, referee="blocked", coach_id=42, venue="blocked"
    )
    after = block().compute(
        replace(
            tables,
            team_matches=contaminated,
            team_seasons=pd.DataFrame({"points_model": [9999], "final_stage": ["wrong"]}),
        )
    )
    assert_frame_equal(before, after)
    assert before.loc[(1, 1), "games_remaining_stage"] == 3
    assert before.loc[(1, 1), "rounds_to_stage_end"] == 2
    assert before.loc[(5, 1), "rounds_to_stage_end"] == 0
    # Removing a future game cannot change counters of earlier games.
    truncated = tables.team_matches.drop(index=6, level="game_id")
    early = block().compute(replace(tables, team_matches=truncated))
    assert_frame_equal(before.loc[[1, 2, 3, 4]], early.loc[[1, 2, 3, 4]])


@pytest.mark.parametrize(
    "column", ["referee", "coach_id", "venue", "goals_scored", "red_cards", "unknown_feature"]
)
def test_unregistered_and_forbidden_columns_fail(column: str) -> None:
    output = block().compute(synthetic_tables())
    output[column] = 1
    with pytest.raises(ValueError, match="Unregistered"):
        validate_feature_columns(output, "table_state")


def test_registry_metadata_and_known_output() -> None:
    frame = block().compute(synthetic_tables())
    validate_feature_columns(frame, "table_state")
    for spec in FEATURES.values():
        assert spec.status in {"✅", "🔬"}
        assert spec.availability in {"D", "S", "K"}
        assert spec.models and spec.decision
    assert "places_decided" in frame
    assert FEATURES["places_decided"] == replace(FEATURES["title_decided"], decision="ה-ת10")


def test_2010_11_schedule_is_beer_sheva_middle_beitar_lower(real_tables: AnalysisTables) -> None:
    games = real_tables.team_matches.reset_index()
    games = games.loc[games.season_id.eq(12)]
    groups = historical_groups(games)
    assert groups[2171] == "middle"
    assert groups[3595] == "lower"
    rule = load_season_rules()[12]
    regular = games.loc[games.stage.eq("regular")]
    roster = sorted(games.team_id.unique())
    sporting = split_playoffs(sporting_table(regular, roster, rule), rule, regular)
    assert sporting[2171] == "lower"
    assert sporting[3595] == "middle"
    table = sporting_table(regular, roster, rule, groups=groups)
    assert 7 <= table.loc[2171, "position"] <= 10
    assert 11 <= table.loc[3595, "position"] <= 16


@pytest.mark.parametrize(
    "season,stage,defined_gaps",
    [
        pytest.param(8, "regular", {"gap_to_leader", "gap_to_safety"}, id="regular-no-playoffs"),
        pytest.param(
            14,
            "regular",
            {"gap_to_leader", "gap_to_cut", "gap_to_safety"},
            id="regular-before-split",
        ),
        pytest.param(14, "upper", {"gap_to_leader"}, id="upper-14-teams"),
        pytest.param(14, "lower", {"gap_to_safety"}, id="lower-14-teams"),
        pytest.param(11, "upper", {"gap_to_leader"}, id="upper-16-teams"),
        pytest.param(11, "middle", set(), id="middle-16-teams"),
        pytest.param(11, "lower", {"gap_to_safety"}, id="lower-16-teams"),
    ],
)
def test_gaps_only_defined_for_reachable_objectives(
    season: int, stage: str, defined_gaps: set[str]
) -> None:
    rule = load_season_rules()[season]
    roster = list(range(1, rule.n_teams + 1))
    rows = []
    for home in roster[::2]:
        away = home + 1
        rows.extend(
            [
                dict(
                    team_id=home,
                    opponent_id=away,
                    goals_scored=2,
                    goals_conceded=1,
                    stage="regular",
                ),
                dict(
                    team_id=away,
                    opponent_id=home,
                    goals_scored=1,
                    goals_conceded=2,
                    stage="regular",
                ),
            ]
        )
    groups = None
    if stage != "regular":
        groups = {
            team: group.name
            for group in rule.groups
            for team in roster[rule.offset(group.name) : rule.offset(group.name) + group.teams]
        }
    snapshot = _snapshot_features(pd.DataFrame(rows), roster, rule, groups)
    targets = roster if groups is None else [team for team in roster if groups[team] == stage]
    selected = snapshot.loc[targets]
    assert not selected.empty
    for column in ("gap_to_leader", "gap_to_cut", "gap_to_safety"):
        if column in defined_gaps:
            assert selected[column].notna().all(), (season, stage, column)
        else:
            assert selected[column].isna().all(), (season, stage, column)
