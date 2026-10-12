"""Synthetic four-team/two-season analysis contracts, nulls and temporal isolation."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ligat.analysis.prepare import (
    ANALYSIS_SEASONS,
    COLUMN_GROUPS,
    AnalysisTables,
    build_coach_spells,
    build_goal_events,
    build_match_table,
    build_season_summary,
    build_team_seasons,
    check_analysis_tables,
    describe_missing_values,
    flag_unusual_matches,
    load_analysis_tables,
    outcome_base_rates,
    preview_pre_match_row,
    team_history,
    to_team_perspective,
)
from ligat.data.repository import Repository
from ligat.data.schemas import make_frame


@pytest.fixture
def healthy_analysis(
    inputs: tuple[Repository, Path, Path], tmp_path: Path
) -> tuple[AnalysisTables, Path, Path, Path]:
    """Complete 12-team triple round robin: no real or held-out data required."""
    repo, teams_path, seasons_path = inputs
    template = (
        build_match_table(repo, teams_path=teams_path, seasons_path=seasons_path).iloc[0].to_dict()
    )
    rows = []
    ring = list(range(1, 13))
    for round_id in range(1, 34):
        for pair in range(6):
            home, away = ring[pair], ring[-pair - 1]
            if round_id % 2 == 0:
                home, away = away, home
            row = dict(template)
            row.update(
                game_id=len(rows) + 1000,
                season_id=8,
                season="2006/2007",
                round=round_id,
                stage="regular",
                date=pd.Timestamp("2006-08-01", tz="Asia/Jerusalem")
                + pd.Timedelta(days=round_id * 7),
                home_team_id=home,
                away_team_id=away,
                home_team_name=f"Team {home}",
                away_team_name=f"Team {away}",
                home_goals=0,
                away_goals=0,
                total_goals=0,
                goal_difference=0,
                result="draw",
                ht_home_goals=0,
                ht_away_goals=0,
                ht_result="draw",
                home_goals_first_half=0,
                away_goals_first_half=0,
                home_red_cards=0,
                away_red_cards=0,
                first_goal_side="none",
                first_goal_minute=pd.NA,
                is_postponed=False,
                is_home_venue=True,
                ht_conflict=False,
                goals_available=True,
            )
            for column in row:
                if column.endswith("_before"):
                    row[column] = 0
            rows.append(row)
        ring = [ring[0], ring[-1], *ring[1:-1]]
    matches = pd.DataFrame(rows).set_index("game_id", drop=True)
    team_matches = to_team_perspective(matches)
    goals = build_goal_events(matches, make_frame("goals", []))
    finals = pd.DataFrame(
        [
            dict(
                season_id=8,
                team_id=team,
                position=team,
                points_model=33,
                points_official=33,
                deduction=0,
                final_stage="regular",
                relegated=team >= 11,
                promoted_in=pd.NA,
            )
            for team in range(1, 13)
        ]
    )
    values = pd.DataFrame(
        [
            dict(season_id=8, team_id=team, squad_size=25, avg_value_eur=400, total_value_eur=10000)
            for team in range(1, 13)
        ]
    )
    seasons = build_team_seasons(team_matches, finals, values)
    summary = build_season_summary(matches, goals, seasons, {8: {"n_teams": 12}})
    tables = AnalysisTables(
        matches, team_matches, goals, seasons, summary, build_coach_spells(team_matches)
    )
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("matches: {}\n", encoding="utf-8")
    teams_path.write_text(
        yaml.safe_dump(
            {
                "teams": [
                    dict(team_id=team, name_he=f"Team {team}", aliases=[f"Team {team}"])
                    for team in range(1, 13)
                ]
            }
        ),
        encoding="utf-8",
    )
    return tables, seasons_path, overrides, teams_path


def test_healthy_analysis(healthy_analysis: tuple[AnalysisTables, Path, Path, Path]) -> None:
    tables, seasons, overrides, teams = healthy_analysis
    warnings = check_analysis_tables(tables, seasons, overrides, teams)
    assert all("missing" in warning for warning in warnings)
    assert len(tables.team_seasons) == 12
    assert tables.team_seasons.matches_played.eq(33).all()
    assert tables.team_seasons.longest_unbeaten_run.eq(33).all()
    assert tables.team_seasons.longest_winless_run.eq(33).all()
    assert tables.season_summary.loc[8, "champion_points_per_match"] == 1


def test_goal_order_states_and_buckets(inputs: tuple[Repository, Path, Path]) -> None:
    repo, teams, seasons = inputs
    matches = build_match_table(repo, teams_path=teams, seasons_path=seasons)
    goals = repo.read("goals")
    events = build_goal_events(matches, goals)
    assert events.loc[(101, 1), "minute_bucket"] == "45+ (first-half stoppage)"
    assert events.loc[101].score_before.tolist() == ["level", "leading", "trailing"]
    assert events.loc[104].score_before.isna().all()
    assert events.loc[(103, 1), "team_id"] == 3  # Own goal already credited to beneficiary.
    goals.loc[goals.game_id.eq(104), "minute"] = 47
    assert build_goal_events(matches, goals).loc[104].minute_bucket.isna().all()
    goals.loc[goals.game_id.eq(101), "minute"] = 46
    goals.loc[goals.game_id.eq(101), "half"] = 1
    assert build_goal_events(matches, goals).loc[101].score_before.isna().all()


@pytest.mark.parametrize(
    "half,minute,bucket",
    [
        (1, 15, "0-15"),
        (1, 16, "16-30"),
        (1, 31, "31-45"),
        (1, 45, "31-45"),
        (1, 46, "45+ (first-half stoppage)"),
        (2, 45, "46-60"),
        (2, 60, "46-60"),
        (2, 61, "61-75"),
        (2, 76, "76-90"),
        (2, 91, "90+"),
    ],
)
def test_minute_bucket_boundaries(
    inputs: tuple[Repository, Path, Path], half: int, minute: int, bucket: str
) -> None:
    repo, teams, seasons = inputs
    matches = build_match_table(repo, teams_path=teams, seasons_path=seasons)
    goals = repo.read("goals").iloc[:1].copy()
    goals["half"], goals["minute"] = half, minute
    assert build_goal_events(matches, goals).iloc[0].minute_bucket == bucket


def test_coach_spells_bridge_nulls_without_imputing(inputs: tuple[Repository, Path, Path]) -> None:
    repo, teams, seasons = inputs
    games = (
        to_team_perspective(build_match_table(repo, teams_path=teams, seasons_path=seasons))
        .loc[[1]]
        .copy()
    )
    games["season_id"] = 11
    games["coach_id"] = pd.array([10, None, 10, 10, 20], dtype="Int64")
    games["coach_name"] = pd.array(["A", None, "A", "A", "B"], dtype="string")
    spells = build_coach_spells(games)
    assert spells.matches.tolist() == [3, 1]
    assert spells.coach_id.tolist() == [10, 20]
    assert spells.is_mid_season_start.tolist() == [False, True]
    assert spells.iloc[0].last_date == games.iloc[3].date


def test_current_and_prior_league_denominators(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path],
) -> None:
    tables, *_ = healthy_analysis
    finals = tables.team_seasons.reset_index().rename(columns={"final_position": "position"})
    finals["season_id"] = 9
    finals["promoted_in"] = False
    team_matches = tables.team_matches.copy()
    team_matches["season_id"] = 9
    values = finals[["season_id", "team_id"]].copy()
    values["squad_size"], values["avg_value_eur"], values["total_value_eur"] = 10, 10, 100
    previous = values.copy()
    previous["season_id"] = 8
    previous.loc[previous.team_id.eq(12), "team_id"] = 13
    previous.loc[previous.team_id.eq(13), "total_value_eur"] = 900
    seasons = build_team_seasons(team_matches, finals, pd.concat([values, previous]))
    assert seasons.loc[(9, 1), "value_share_of_league"] == pytest.approx(1 / 12)
    assert seasons.loc[(9, 1), "prev_season_value_share"] == pytest.approx(100 / 2000)
    assert pd.isna(seasons.loc[(9, 12), "prev_season_value_share"])
    values.loc[values.team_id.eq(12), "total_value_eur"] = pd.NA
    seasons = build_team_seasons(team_matches, finals, pd.concat([values, previous]))
    assert seasons.loc[(9, 1), "value_share_of_league"] == pytest.approx(1 / 11)
    assert pd.isna(seasons.loc[(9, 12), "total_value_eur"])


def test_sporting_benchmarks_ignore_halving_and_deductions(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path],
) -> None:
    tables, *_ = healthy_analysis
    seasons = tables.team_seasons.copy()
    seasons["points_model"], seasons["points_official"] = 17, 8
    seasons.loc[(8, 1), "points_per_match"] = 2.5
    seasons.loc[(8, 11), "points_per_match"] = 0.25
    seasons.loc[(8, 12), "points_per_match"] = 0.1
    summary = build_season_summary(
        tables.match_table, tables.goal_events, seasons, {8: {"n_teams": 12}}
    )
    assert summary.loc[8, "champion_points_per_match"] == 2.5
    assert summary.loc[8, "relegation_line_points_per_match"] == 0.25
    assert summary.loc[8, "top3_value_share"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "defect,message",
    [
        ("two_rows", "two team rows"),
        ("points", "points_earned"),
        ("round", "twice in a round"),
        ("games", "matches_played"),
        ("pair", "regular pair count"),
        ("goals", "Goal count mismatch"),
        ("halftime", "Halftime goal count"),
        ("team_count", "team_seasons.matches_played"),
        ("relegation", "relegation count"),
        ("squad", "squad_values row"),
        ("coach_overlap", "Overlapping coach"),
        ("missing", "Undocumented missing"),
        ("benchmark", "sporting points benchmark"),
    ],
)
def test_health_checks_reject_corruption(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path], defect: str, message: str
) -> None:
    original, formats, overrides, teams = healthy_analysis
    tables = replace(
        original, **{name: frame.copy(deep=True) for name, frame in vars(original).items()}
    )
    first_game = tables.match_table.index[0]
    first_team = tables.team_matches.index[0]
    if defect == "two_rows":
        tables = replace(tables, team_matches=tables.team_matches.iloc[1:])
    elif defect == "points":
        tables.team_matches.loc[first_team, "points_earned"] = 3
    elif defect == "round":
        gid = tables.match_table.index[6]
        tables.match_table.loc[gid, "round"] = 1
        tables = replace(tables, team_matches=to_team_perspective(tables.match_table))
    elif defect == "games":
        tables.team_seasons.iloc[0, tables.team_seasons.columns.get_loc("matches_played")] = 32
    elif defect == "pair":
        tables.match_table["stage"] = "upper"
        tables = replace(tables, team_matches=to_team_perspective(tables.match_table))
    elif defect in ("goals", "halftime"):
        if defect == "halftime":
            tables.match_table.loc[first_game, "ht_home_goals"] = 1
        else:
            row = dict(
                game_id=first_game,
                team_id=tables.match_table.loc[first_game, "home_team_id"],
                minute=30,
                half=1,
                added_time=None,
                is_penalty=False,
                is_own_goal=False,
                is_first_half_stoppage=False,
            )
            tables = replace(
                tables,
                goal_events=build_goal_events(tables.match_table, make_frame("goals", [row])),
            )
    elif defect == "team_count":
        tables = replace(tables, team_seasons=tables.team_seasons.iloc[1:])
    elif defect == "relegation":
        tables.team_seasons["relegated"] = False
    elif defect == "squad":
        tables.team_seasons["squad_row_available"] = False
    elif defect == "coach_overlap":
        rows = [
            dict(
                team_id=1,
                spell_number=n,
                coach_id=n,
                coach_name="Coach",
                season_id_start=8,
                season_id_end=8,
                first_date=pd.Timestamp("2009-08-01", tz="Asia/Jerusalem"),
                last_date=pd.Timestamp("2009-08-02", tz="Asia/Jerusalem"),
                matches=1,
                points_per_match=1,
                is_mid_season_start=False,
            )
            for n in (1, 2)
        ]
        tables = replace(
            tables, coach_spells=pd.DataFrame(rows).set_index(["team_id", "spell_number"])
        )
    elif defect == "missing":
        tables.match_table["unexpected"] = pd.NA
    elif defect == "benchmark":
        tables.season_summary.loc[8, "champion_points_per_match"] = 0
    with pytest.raises(AssertionError, match=message):
        check_analysis_tables(tables, formats, overrides, teams)


def test_alias_warning_and_flags(healthy_analysis: tuple[AnalysisTables, Path, Path, Path]) -> None:
    tables, formats, overrides, teams = healthy_analysis
    config = yaml.safe_load(teams.read_text(encoding="utf-8"))
    config["teams"][0]["aliases"] = ["Completely unrelated"]
    teams.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert any(
        "aliases" in warning for warning in check_analysis_tables(tables, formats, overrides, teams)
    )
    tables.match_table.iloc[0, tables.match_table.columns.get_loc("is_postponed")] = True
    assert len(flag_unusual_matches(tables)) == 1
    missing = describe_missing_values(tables)
    assert missing.loc[missing.column.eq("first_goal_minute"), "reason"].notna().all()


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


@pytest.mark.parametrize("parallel_stage", ["lower", "middle"])
def test_parallel_playoff_rounds_are_not_postponed(
    inputs: tuple[Repository, Path, Path], parallel_stage: str
) -> None:
    repo = inputs[0]
    definitions = [
        (101, 301, "upper", 34, 10),
        (103, 302, "upper", 35, 17),
        (102, 303, parallel_stage, 34, 20),
        (104, 304, parallel_stage, 35, 24),
    ]
    for name in ("matches", "game_details", "goals", "red_cards", "coaches"):
        source = repo.read(name)
        copies = []
        for original_id, game_id, stage, round_id, day in definitions:
            extra = source.loc[source.game_id.eq(original_id)].copy()
            extra["game_id"] = game_id
            if name == "matches":
                extra.loc[:, "stage"] = stage
                extra["round"] = round_id
                extra["date"] = pd.Timestamp(f"2006-08-{day} 20:00", tz="Asia/Jerusalem")
            copies.append(extra)
        repo.write(name, pd.concat([source, *copies], ignore_index=True))
    frame = build(inputs)
    assert not frame.loc[[301, 302, 303, 304], "is_postponed"].any()
    assert frame.loc[105, "is_postponed"]  # regular round 1 after regular round 2
    assert not frame.loc[[302, 304], "is_postponed"].any()  # last round of each stage


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


def test_dropped_match_schedule_metadata(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path],
) -> None:
    tables, formats, overrides, teams = healthy_analysis
    game_id = tables.match_table.index[0]
    game = tables.match_table.loc[game_id]
    rule = dict(
        action="drop",
        verified=True,
        season_id=8,
        round=1,
        stage="regular",
        home_team_id=int(game.home_team_id),
        away_team_id=int(game.away_team_id),
    )
    overrides.write_text(yaml.safe_dump({"matches": {int(game_id): rule}}), encoding="utf-8")
    matches = tables.match_table.drop(game_id)
    team_matches = to_team_perspective(matches)
    finals = tables.team_seasons.reset_index().rename(columns={"final_position": "position"})
    values = finals[["season_id", "team_id", "squad_size", "avg_value_eur", "total_value_eur"]]
    seasons = build_team_seasons(team_matches, finals, values)
    summary = build_season_summary(matches, tables.goal_events, seasons, {8: {"n_teams": 12}})
    reduced = replace(
        tables,
        match_table=matches,
        team_matches=team_matches,
        team_seasons=seasons,
        season_summary=summary,
    )
    check_analysis_tables(reduced, formats, overrides, teams)
    overrides.write_text("matches: {}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Wrong schedule length"):
        check_analysis_tables(reduced, formats, overrides, teams)


def test_load_all_tables_uses_scoped_reads(
    inputs: tuple[Repository, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, teams, seasons = inputs
    original = repo.read_filtered
    calls = []

    def filtered(name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        calls.append((name, filters))
        assert all(
            not ({26, 27, 28} & set(value)) for column, _, value in filters if column == "season_id"
        )
        if name == "final_tables":
            return make_frame(
                "final_tables",
                [
                    dict(
                        season_id=season,
                        round=2,
                        team_id=team,
                        position=team,
                        played=0,
                        won=0,
                        drawn=0,
                        lost=0,
                        goals_for=0,
                        goals_against=0,
                        points_model=0,
                        points_official=0,
                        deduction=0,
                        final_stage="regular",
                        relegated=team >= 3,
                        promoted_in=None if season == 8 else False,
                        regular_season_points=0,
                    )
                    for season in (8, 9)
                    for team in range(1, 5)
                ],
            )
        return original(name, filters)

    monkeypatch.setattr(repo, "read_filtered", filtered)
    monkeypatch.setattr(repo, "read", lambda _: pytest.fail("Unfiltered read"))
    tables = load_analysis_tables(repo, seasons=[8, 9], teams_path=teams, seasons_path=seasons)
    assert set(tables.team_seasons.index.get_level_values("season_id")) == {8, 9}
    assert ("final_tables", [("season_id", "in", [8, 9])]) in calls
    assert calls[-1] == ("squad_values", [("season_id", "in", [7, 8, 9])])
    assert ("goals", [("game_id", "in", tables.match_table.index.tolist())]) in calls


def test_preview_features_use_strict_past_and_do_not_leak(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path],
) -> None:
    tables, *_ = healthy_analysis
    game_id = tables.match_table.index[-1]
    before = preview_pre_match_row(tables, game_id, window=2)
    matches = tables.match_table.copy()
    matches.loc[game_id, ["home_goals", "away_goals", "total_goals", "goal_difference"]] = [
        5,
        0,
        5,
        5,
    ]
    matches.loc[game_id, "result"] = "home_win"
    changed = replace(tables, match_table=matches, team_matches=to_team_perspective(matches))
    after = preview_pre_match_row(changed, game_id, window=2)
    columns = before.filter(like="feature_").columns
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert after.iloc[0].target_home_goals == 5
    assert before.iloc[0].feature_home_points_earned_mean == 1
    team = tables.match_table.loc[game_id, "home_team_id"]
    history = tables.team_matches.loc[team].sort_values(["date", "game_id"])
    previous = history.index[-2]
    history_view = tables.team_matches.copy()
    history_view.loc[(team, previous), "date"] = tables.match_table.loc[game_id, "date"]
    history_view.loc[(team, previous), "goals_scored"] = 100
    simultaneous = preview_pre_match_row(
        replace(tables, team_matches=history_view), game_id, window=2
    )
    assert simultaneous.iloc[0].feature_home_goals_scored_mean == 0
    first = preview_pre_match_row(tables, tables.match_table.index[0])
    assert first.filter(like="feature_").isna().all().all()
    with pytest.raises(ValueError, match="positive"):
        preview_pre_match_row(tables, game_id, window=0)


def test_b0_rates_and_wilson_intervals(
    healthy_analysis: tuple[AnalysisTables, Path, Path, Path],
) -> None:
    tables, *_ = healthy_analysis
    rates = outcome_base_rates(tables.match_table)
    assert len(rates) == 6
    assert rates.groupby("scope").rate.sum().eq(1).all()
    overall = rates.loc[rates.scope.eq("overall")].set_index("outcome")
    assert overall.loc["draw", "rate"] == 1
    assert overall.loc["home_win", "ci_upper"] == pytest.approx(0.01903206032664819)
    assert overall.ci_lower.le(overall.rate).all()
    assert overall.ci_upper.ge(overall.rate).all()
    with pytest.raises(ValueError):
        outcome_base_rates(tables.match_table.iloc[:0])


def test_coach_left_censoring(inputs: tuple[Repository, Path, Path]) -> None:
    repo, teams, seasons = inputs
    games = (
        to_team_perspective(build_match_table(repo, teams_path=teams, seasons_path=seasons))
        .loc[[1]]
        .copy()
    )
    games["season_id"] = 11
    games["match_number"] = pd.array([1, 2, 3, 4, 5], dtype="Int64")
    games["coach_id"] = pd.array([None, None, 10, 20, 10], dtype="Int64")
    games["coach_name"] = pd.array([None, None, "A", "B", "A"], dtype="string")
    spells = build_coach_spells(games)
    assert spells.is_left_censored.tolist() == [True, False, False]
    assert spells.is_flicker.tolist() == [False, True, False]
    assert spells.iloc[0].is_mid_season_start


@pytest.mark.parametrize(
    "coaches,flicker", [([10, 20, 20, 10, 10], True), ([10, 20, 20, 20, 10], False)]
)
def test_flicker_length_threshold(
    inputs: tuple[Repository, Path, Path], coaches: list[int], flicker: bool
) -> None:
    repo, teams, seasons = inputs
    games = (
        to_team_perspective(build_match_table(repo, teams_path=teams, seasons_path=seasons))
        .loc[[1]]
        .copy()
    )
    games["season_id"] = 11
    games["coach_id"] = pd.array(coaches, dtype="Int64")
    games["coach_name"] = pd.array([str(coach) for coach in coaches], dtype="string")
    assert bool(build_coach_spells(games).iloc[1].is_flicker) == flicker
