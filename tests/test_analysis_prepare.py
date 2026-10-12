"""Synthetic four-team/two-season analysis contracts, nulls and temporal isolation."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ligat.analysis.prepare import (
    ANALYSIS_SEASONS,
    COLUMN_GROUPS,
    build_match_table,
    team_history,
    to_team_perspective,
)
from ligat.data.repository import Repository
from ligat.data.schemas import make_frame


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Repository, Path, Path]:
    repo = Repository(tmp_path / "clean")
    definitions = [
        (101, 8, 1, 10, 1, 2, 2, 1, 1, 0),
        (102, 8, 1, 10, 3, 4, 0, 0, 0, 0),
        (103, 8, 2, 17, 1, 3, 0, 1, 0, 1),
        (104, 8, 2, 17, 2, 4, 1, 1, 0, 0),
        (105, 8, 1, 24, 1, 4, 1, 0, 1, 0),
        (201, 9, 1, 10, 1, 2, 0, 0, 0, 0),
        (202, 9, 2, 17, 3, 1, 1, 2, 0, 0),
    ]
    matches, details, coaches = [], [], []
    for game, season, round_id, day, home, away, hg, ag, hh, ah in definitions:
        matches.append(
            {
                "game_id": game,
                "season_id": season,
                "season": f"{season + 1998}/{season + 1999}",
                "round": round_id,
                "stage": "regular",
                "date": pd.Timestamp(f"{season + 1998}-08-{day:02d} 20:00", tz="Asia/Jerusalem"),
                "home_team_id": home,
                "away_team_id": away,
                "home_goals": hg,
                "away_goals": ag,
                "venue": "Arena",
                "venue_raw": "(סגור)Arena",
                "is_home_venue": game != 105,
                "status": "played",
                "source": "synthetic",
            }
        )
        details.append(
            {
                "game_id": game,
                "ht_home_goals": hh,
                "ht_away_goals": ah,
                "referee": None if game == 102 else "Referee",
                "has_timeline": game != 102,
                "has_lineups": True,
                "has_coaches": game != 102,
                "ht_conflict": game == 104,
                "goals_available": game != 105,
            }
        )
        for team in (home, away):
            coaches.append(
                {
                    "game_id": game,
                    "team_id": team,
                    "coach_id": None if game == 102 else team + 10,
                    "coach_name": None if game == 102 else f"Coach {team}",
                }
            )
    goals = []
    for game, team, minute, half in [
        (101, 1, 46, 1),
        (101, 1, 60, 2),
        (101, 2, 70, 2),
        (103, 3, 5, 1),
        (104, 2, 4, None),
        (104, 4, 17, None),
        (202, 3, 47, 2),
        (202, 1, 55, 2),
        (202, 1, 75, 2),
    ]:
        goals.append(
            {
                "game_id": game,
                "team_id": team,
                "minute": minute,
                "half": half,
                "added_time": None,
                "is_first_half_stoppage": half == 1 and minute > 45,
                "is_penalty": False,
                "is_own_goal": game == 103,
            }
        )
    reds = [{"game_id": 104, "team_id": 2, "minute": None, "is_second_yellow": True}]
    values = [
        {
            "season_id": 8,
            "tm_club_id": team + 100,
            "team_id": team,
            "squad_size": 30,
            "total_value_eur": team * 1000000,
            "avg_value_eur": None if team == 2 else team * 30000,
        }
        for team in (1, 2, 4)
    ]
    for name, rows in {
        "matches": matches,
        "game_details": details,
        "goals": goals,
        "red_cards": reds,
        "coaches": coaches,
        "squad_values": values,
    }.items():
        repo.write(name, make_frame(name, rows))
    teams = tmp_path / "teams.yaml"
    teams.write_text(
        yaml.safe_dump(
            {
                "teams": [
                    {
                        "team_id": team,
                        "name_he": f"קבוצה {team}",
                        "aliases": [f"כינוי {team}", "משותף"],
                    }
                    for team in range(1, 5)
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    seasons = tmp_path / "seasons.yaml"
    seasons.write_text(
        yaml.safe_dump(
            {
                "seasons": [
                    {"season_id": 8, "n_teams": 12},
                    {"season_id": 9, "n_teams": 16},
                ]
            }
        )
    )
    return repo, teams, seasons


def build(inputs: tuple[Repository, Path, Path]) -> pd.DataFrame:
    repo, teams, seasons = inputs
    return build_match_table(repo, seasons=[8, 9], teams_path=teams, seasons_path=seasons)


def test_team_translation(inputs: tuple[Repository, Path, Path]) -> None:
    matches = build(inputs)
    teams = to_team_perspective(matches)
    home, away = teams.loc[(1, 101)], teams.loc[(2, 101)]
    assert (home.goals_scored, home.goals_conceded, home.points_earned) == (2, 1, 3)
    assert (away.goals_scored, away.goals_conceded, away.points_earned) == (1, 2, 0)
    assert home.ht_state == "leading" and away.ht_state == "trailing"
    assert home.scored_first and not away.scored_first
    assert teams.loc[(2, 104)].result == "draw" and teams.loc[(2, 104)].points_earned == 1
    assert len(teams) == 2 * len(matches)
    assert teams.loc[(1, 105)].match_number == 3  # postponed round 1 follows round 2
    assert teams.loc[(1, 201)].match_number == 1
    assert matches.loc[101].era == "12_teams" and matches.loc[201].era == "16_teams"


def test_strictly_previous_results(inputs: tuple[Repository, Path, Path]) -> None:
    repo = inputs[0]
    before = build(inputs)
    source = repo.read("matches")
    source.loc[source.game_id.eq(101), ["home_goals", "away_goals"]] = [0, 3]
    repo.write("matches", source)
    after = build(inputs)
    pd.testing.assert_series_equal(
        before.loc[101, COLUMN_GROUPS["pre_match"]], after.loc[101, COLUMN_GROUPS["pre_match"]]
    )
    assert after.loc[103, "home_points_before"] == 0
    assert before.loc[103, "home_points_before"] == 3
    assert before.loc[103, "home_goals_scored_before"] == 2
    assert after.loc[103, "home_goals_scored_before"] == 0
    assert after.loc[201, "home_points_before"] == 0  # reset every season


def test_simultaneous_games_do_not_see_each_other(inputs: tuple[Repository, Path, Path]) -> None:
    repo = inputs[0]
    source = repo.read("matches")
    source.loc[source.game_id.eq(102), "home_team_id"] = 1
    repo.write("matches", source)
    frame = build(inputs)
    assert frame.loc[101, "home_matches_before"] == frame.loc[102, "home_matches_before"] == 0
    assert frame.loc[103, "home_matches_before"] == 2
    assert frame.loc[103, "home_points_before"] == 4
    to_team_perspective(frame)


def test_nulls_and_goal_credit(inputs: tuple[Repository, Path, Path]) -> None:
    frame = build(inputs)
    teams = to_team_perspective(frame)
    assert pd.isna(frame.loc[102, "home_red_cards"])
    assert frame.loc[103, "home_red_cards"] == 0
    assert frame.loc[104, "home_red_cards"] == 1  # unknown minute still counts
    assert pd.isna(frame.loc[104, "home_goals_first_half"])
    assert pd.isna(teams.loc[(2, 104)].ht_state)
    assert frame.loc[101, "home_goals_first_half"] == 1  # minute 46, half 1
    assert frame.loc[103, "first_goal_side"] == "away"  # own goal already credited
    assert frame.loc[102, "first_goal_side"] == "none"
    assert pd.isna(frame.loc[102, "first_goal_minute"])
    assert pd.isna(teams.loc[(3, 102)].scored_first)
    assert pd.isna(frame.loc[105, "first_goal_side"])
    assert pd.isna(frame.loc[105, "first_goal_minute"])
    assert pd.isna(teams.loc[(1, 105)].scored_first)
    assert pd.isna(frame.loc[101, "home_prev_season_total_value_eur"])
    assert frame.loc[201, "home_prev_season_total_value_eur"] == 1000000
    assert pd.isna(frame.loc[201, "away_prev_season_avg_value_eur"])
    assert pd.isna(frame.loc[202, "home_prev_season_total_value_eur"])
    assert pd.isna(frame.loc[102, "home_coach_id"])
    assert pd.isna(frame.loc[102, "referee"])


def test_postponement_and_column_groups(inputs: tuple[Repository, Path, Path]) -> None:
    frame = build(inputs)
    assert frame.loc[105, "is_postponed"]
    assert not frame.loc[101, "is_postponed"]
    assert not frame.loc[104, "is_postponed"]  # last round
    columns = [column for group in COLUMN_GROUPS.values() for column in group]
    assert len(columns) == len(set(columns)) == len(frame.columns)
    assert set(columns) == set(frame.columns)
    assert {"home_coach_id", "away_coach_name", "referee"} <= set(COLUMN_GROUPS["pre_match"])
    assert {"date", "venue", "home_team_id", "away_team_name"} <= set(COLUMN_GROUPS["identity"])


def test_first_goal_uses_half_before_running_minute(inputs: tuple[Repository, Path, Path]) -> None:
    repo = inputs[0]
    goals = repo.read("goals")
    goals.loc[goals.game_id.eq(101) & goals.team_id.eq(2), "minute"] = 45
    repo.write("goals", goals)
    frame = build(inputs)
    assert frame.loc[101, "first_goal_side"] == "home"
    assert frame.loc[101, "first_goal_minute"] == 46


def test_simultaneous_first_goals_preserve_uncertainty(
    inputs: tuple[Repository, Path, Path],
) -> None:
    repo = inputs[0]
    goals = repo.read("goals")
    away = goals.game_id.eq(101) & goals.team_id.eq(2)
    goals.loc[away, ["minute", "half", "is_first_half_stoppage"]] = [46, 1, True]
    repo.write("goals", goals)
    frame = build(inputs)
    assert pd.isna(frame.loc[101, "first_goal_side"])
    assert frame.loc[101, "first_goal_minute"] == 46
    teams = to_team_perspective(frame)
    assert pd.isna(teams.loc[(1, 101)].scored_first)
    assert pd.isna(teams.loc[(2, 101)].scored_first)


def test_indexes_and_history(inputs: tuple[Repository, Path, Path]) -> None:
    frame = build(inputs)
    teams = to_team_perspective(frame)
    assert "game_id" not in frame.columns and frame.index.is_unique
    assert "team_id" not in teams.columns and "game_id" not in teams.columns
    teams.groupby(level="team_id").points_earned.sum()
    teams.sort_values(["team_id", "date"])
    frame.sort_values(["date", "game_id"])
    assert len(teams.loc[1]) == 5
    by_id = team_history(teams, 1, inputs[1])
    pd.testing.assert_frame_equal(by_id, team_history(teams, "קבוצה 1", inputs[1]))
    pd.testing.assert_frame_equal(by_id, team_history(teams, "כינוי 1", inputs[1]))
    assert by_id.date.is_monotonic_increasing
    by_id.iloc[0, by_id.columns.get_loc("venue")] = "Changed"
    assert teams.loc[(1, 101)].venue == "Arena"
    for team in ("unknown", "משותף", 999):
        with pytest.raises(ValueError):
            team_history(teams, team, inputs[1])


def test_filtered_reads_and_season_bounds(
    inputs: tuple[Repository, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = inputs[0]
    calls = []
    original = repo.read_filtered

    def filtered(name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        calls.append((name, filters))
        return original(name, filters)

    monkeypatch.setattr(repo, "read_filtered", filtered)
    monkeypatch.setattr(repo, "read", lambda _: pytest.fail("Unfiltered analysis read"))
    frame = build(inputs)
    assert set(frame.season_id) == {8, 9}
    assert calls[0] == ("matches", [("season_id", "in", [8, 9])])
    assert calls[-1] == ("squad_values", [("season_id", "in", [7, 8])])
    for _, filters in calls[1:-1]:
        assert filters == [("game_id", "in", frame.index.tolist())]
    assert list(ANALYSIS_SEASONS) == list(range(8, 26))
    for selection in ([], [26], [8, 8]):
        with pytest.raises(ValueError, match="8–25"):
            build_match_table(repo, seasons=selection, teams_path=inputs[1], seasons_path=inputs[2])
