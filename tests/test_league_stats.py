"""Descriptive invariants on synthetic seasons and optional local clean data."""

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from ligat.analysis import league_stats as stats
from ligat.analysis.league_stats.season_course import _pregame_rates
from ligat.analysis.prepare import (
    DEVELOPMENT_SEASONS,
    AnalysisTables,
    build_team_seasons,
    load_playoff_upper_sizes,
)
from ligat.data.repository import Repository
from ligat.viz.league_history import chapter_b_chart
from ligat.viz.league_stats import chapter_a_chart
from ligat.viz.season_course import chapter_c_chart

C_FUNCTIONS = [
    stats.standings_by_match_number,
    stats.table_settling,
    stats.settling_sensitivity,
    stats.split_movement,
    stats.split_summary,
    stats.stage_comparison,
    stats.playoff_form,
    stats.playoff_form_summary,
    stats.final_rounds_check,
    stats.season_points_race,
]


def _run_c(
    function: Any, tables: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    options: dict[str, Any] = {}
    if function in (
        stats.table_settling,
        stats.settling_sensitivity,
        stats.split_movement,
        stats.split_summary,
    ):
        options["upper_sizes"] = {8: None, 11: 2}
    if function in (
        stats.table_settling,
        stats.settling_sensitivity,
        stats.split_summary,
        stats.stage_comparison,
        stats.playoff_form_summary,
        stats.final_rounds_check,
    ):
        options["n_bootstrap"] = 50
    return function(tables, seasons, **options)


@pytest.fixture
def course_tables() -> AnalysisTables:
    matches, teams, finals = [], [], []
    for season in (8, 11):
        schedule = [
            (1, 2, 2, 0, 1, "regular"),
            (3, 4, 1, 1, 1, "regular"),
            (1, 3, 0, 0, 2, "regular"),
            (2, 4, 1, 0, 2, "regular"),
            (1, 4, 0, 0, 3, "regular"),
            (2, 3, 0, 0, 3, "regular"),
        ]
        if season == 11:
            schedule += [(1, 2, 0, 1, 4, "upper"), (3, 4, 0, 1, 4, "lower")]
        for i, (home, away, hg, ag, round_id, stage) in enumerate(schedule):
            game_id = season * 100 + i
            date = pd.Timestamp(f"2020-09-{round_id:02d} 20:00", tz="Asia/Jerusalem")
            matches.append(
                dict(
                    game_id=game_id,
                    season_id=season,
                    stage=stage,
                    round=round_id,
                    date=date,
                    home_team_id=home,
                    away_team_id=away,
                    home_goals=hg,
                    away_goals=ag,
                    total_goals=hg + ag,
                    era="12_teams" if season == 8 else "16_teams",
                    result="home_win" if hg > ag else "away_win" if hg < ag else "draw",
                )
            )
            for team, scored, conceded in ((home, hg, ag), (away, ag, hg)):
                teams.append(
                    dict(
                        team_id=team,
                        game_id=game_id,
                        season_id=season,
                        stage=stage,
                        date=date,
                        era=matches[-1]["era"],
                        goals_scored=scored,
                        goals_conceded=conceded,
                        goal_difference=scored - conceded,
                        points_earned=3 if scored > conceded else 0 if scored < conceded else 1,
                    )
                )
        positions = {1: 1, 2: 2, 3: 3, 4: 4} if season == 8 else {1: 2, 2: 1, 3: 4, 4: 3}
        for team, position in positions.items():
            finals.append(
                dict(
                    season_id=season,
                    team_id=team,
                    team_name=f"Club {team}",
                    final_position=position,
                    relegated=position == 4,
                    final_stage="regular" if season == 8 else "upper" if team <= 2 else "lower",
                )
            )
    empty = pd.DataFrame()
    return AnalysisTables(
        pd.DataFrame(matches).set_index("game_id", drop=True),
        pd.DataFrame(teams).set_index(["team_id", "game_id"], drop=True),
        empty,
        pd.DataFrame(finals).set_index(["season_id", "team_id"], drop=True),
        empty,
        empty,
    )


@pytest.mark.parametrize("function", C_FUNCTIONS)
@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_course_development_scope(
    function: Any, seasons: list[int], course_tables: AnalysisTables
) -> None:
    with pytest.raises(ValueError, match="development"):
        _run_c(function, course_tables, seasons)


def test_course_known_standings_and_short_games(course_tables: AnalysisTables) -> None:
    frame = stats.standings_by_match_number(course_tables)
    assert frame.loc[(8, 3, 1), "points"] == 5
    assert frame.loc[(8, 3, 1), "goal_difference"] == 2
    assert frame.loc[(8, 3, 1), "goals_scored"] == 2
    assert frame.loc[(8, 1, 3), "rank"] == 2.5  # Full tie with club 4.
    assert frame.loc[(8, 1, 4), "rank"] == 2.5
    assert frame["rank"].between(1, 4).all()
    assert frame.groupby(level=["season_id", "team_id"]).points.diff().dropna().ge(0).all()
    shortened = course_tables.team_matches.drop(index=[(1, 804), (4, 804)])
    short = stats.standings_by_match_number(replace(course_tables, team_matches=shortened))
    assert short.loc[(8, 3, 1), "games_short"] and short.loc[(8, 3, 4), "games_short"]
    assert short.loc[(8, 3, 1), "points"] == short.loc[(8, 2, 1), "points"] == 4
    assert short.loc[(8, 3, 1), "games_played"] == 2
    with pytest.raises(ValueError, match="Unknown stage"):
        stats.standings_by_match_number(course_tables, stage="bad")


def test_course_ties_and_split_labels(course_tables: AnalysisTables) -> None:
    settling = _run_c(stats.table_settling, course_tables)
    assert settling.loc[("12_teams", 3), "spearman_mean"] == pytest.approx(1)
    assert settling.xs("12_teams", level="era").in_upper_line_stayed_rate.isna().all()
    assert settling.loc[("16_teams", 3), "leader_won_title_rate"] == 0
    assert settling.loc[("16_teams", 3), "in_upper_line_stayed_rate"] == 1
    movement = _run_c(stats.split_movement, course_tables)
    assert set(movement.index.get_level_values("season_id")) == {11}
    assert movement.loc[(11, 1), "move"] == -1 and movement.loc[(11, 2), "move"] == 1
    assert movement.points_halved.all()
    summary = _run_c(stats.split_summary, course_tables)
    assert summary.leader_n.eq(1).all() and summary.leader_won_count.eq(0).all()
    ties = course_tables.team_matches.copy()
    ties["points_earned"] = 1
    ties["goal_difference"] = 0
    ties["goals_scored"] = 0
    result = _run_c(stats.table_settling, replace(course_tables, team_matches=ties))
    assert result.leader_won_title_rate.eq(0.25).all()
    assert result.leader_won_title_rate_tied.eq(1).all()
    assert result.xs("16_teams", level="era").in_upper_line_stayed_rate.eq(0.5).all()
    assert result.leader_won_title_excl_ties.isna().all()


def test_course_stage_form_and_upsets(course_tables: AnalysisTables) -> None:
    stages = _run_c(stats.stage_comparison, course_tables)
    assert stages.n_matches.to_dict() == {"lower": 1, "regular": 12, "upper": 1}
    assert stages.loc["regular", "one_one_rate"] == pytest.approx(1 / 6)
    form = stats.playoff_form(course_tables)
    assert form.loc[(11, 1), "ppm_regular"] == pytest.approx(5 / 3)
    assert form.loc[(11, 1), "ppm_playoff"] == 0
    assert form.loc[(11, 2), "ppm_playoff"] == 3
    final = stats.final_rounds_check(course_tables, last_n=1, n_bootstrap=50)
    assert final.n.sum() == 14
    assert final.loc[("regular", "last"), "n"] == 4
    assert final.loc[("upper", "last"), "upset_rate"] == 1
    assert final.loc[("lower", "last"), "upset_rate"] == 1
    assert final.loc[("regular", "rest"), "upset_n"] == 4
    assert final.loc[("regular", "rest"), "upset_count"] == 2
    with pytest.raises(ValueError, match="last_n"):
        stats.final_rounds_check(course_tables, last_n=0)


def test_course_upsets_use_strictly_prior_results(course_tables: AnalysisTables) -> None:
    original = _pregame_rates(course_tables, [8, 11])
    changed = course_tables.team_matches.copy()
    changed.loc[(1, 1102), "points_earned"] = 0
    changed.loc[(3, 1102), "points_earned"] = 3
    new = _pregame_rates(replace(course_tables, team_matches=changed), [8, 11])
    pd.testing.assert_frame_equal(original.loc[[1102, 1103]], new.loc[[1102, 1103]])
    assert original.loc[(1104, 1), "ppm_before"] != new.loc[(1104, 1), "ppm_before"]


@pytest.mark.parametrize("function", C_FUNCTIONS)
def test_course_indices_input_immutability_and_holdouts(
    function: Any, course_tables: AnalysisTables
) -> None:
    original = [
        f.copy(deep=True)
        for f in (course_tables.match_table, course_tables.team_matches, course_tables.team_seasons)
    ]

    def heldout(frame: pd.DataFrame) -> pd.DataFrame:
        extra = frame.reset_index()
        extra["season_id"] = 26
        extra = extra.drop_duplicates(list(frame.index.names)).set_index(
            list(frame.index.names), drop=True
        )
        return pd.concat([frame, extra])

    contaminated = replace(
        course_tables,
        match_table=heldout(course_tables.match_table),
        team_matches=heldout(course_tables.team_matches),
        team_seasons=heldout(course_tables.team_seasons),
    )
    result = _run_c(function, course_tables)
    pd.testing.assert_frame_equal(result, _run_c(function, contaminated))
    assert result.index.is_unique and not set(result.index.names) & set(result.columns)
    for old, new in zip(
        original,
        (course_tables.match_table, course_tables.team_matches, course_tables.team_seasons),
        strict=True,
    ):
        pd.testing.assert_frame_equal(old, new)


def test_course_real_invariants(real_tables: AnalysisTables, real_standings: pd.DataFrame) -> None:
    sizes = load_playoff_upper_sizes()
    standings = real_standings.copy(deep=True)
    last = standings.reset_index().groupby("season_id").k.transform("max")
    totals = standings.reset_index().loc[lambda g: g.k.eq(last)].set_index(["season_id", "team_id"])
    expected = (
        real_tables.team_matches.loc[real_tables.team_matches.stage.eq("regular")]
        .reset_index()
        .groupby(["season_id", "team_id"])
        .points_earned.sum()
    )
    pd.testing.assert_series_equal(
        totals.points.astype("Int64"), expected.astype("Int64"), check_names=False
    )
    assert standings.groupby(level=["season_id", "team_id"]).points.diff().dropna().ge(0).all()
    settling = stats.table_settling(real_tables, upper_sizes=sizes, n_bootstrap=50)
    assert settling.spearman_mean.between(-1, 1).all()
    sensitivity = stats.settling_sensitivity(real_tables, upper_sizes=sizes, n_bootstrap=50)
    assert sensitivity.max_abs_diff.tolist() == pytest.approx([1 / 44, 1 / 44])
    assert settling.loc[("12_teams", 33), "spearman_mean"] >= 0.99
    assert settling.xs("12_teams", level="era").in_upper_line_stayed_rate.isna().all()
    movement = stats.split_movement(real_tables, upper_sizes=sizes)
    assert set(movement.index.get_level_values("season_id")) == set(range(11, 26))
    for season, group in movement.groupby(level="season_id", observed=True):
        cutoff = sizes[int(season)]
        assert cutoff is not None
        assert group.loc[group.group.eq("upper"), "final_position"].between(1, cutoff).all()
        middle = group.loc[group.group.eq("middle")]
        if not middle.empty:
            assert int(season) in (11, 12) and middle.final_position.between(7, 10).all()
        lower_start = 11 if int(season) in (11, 12) else cutoff + 1
        assert (
            group.loc[group.group.eq("lower"), "final_position"]
            .between(lower_start, len(group))
            .all()
        )
    form_summary = stats.playoff_form_summary(real_tables, n_bootstrap=50)
    assert form_summary.n.to_dict() == {"lower": 116, "middle": 8, "upper": 92}
    assert form_summary.n_seasons.to_dict() == {"lower": 15, "middle": 2, "upper": 15}
    for stage, expected_change in {"upper": -0.41, "lower": 0.35, "middle": 0.15}.items():
        assert form_summary.loc[stage, "mean_ppm_change"] == pytest.approx(
            expected_change, abs=0.01
        )
    stages = stats.stage_comparison(real_tables, n_bootstrap=50)
    assert stages.n_matches.sum() == 4315 and stages.loc["middle", "n_matches"] == 12
    assert set(
        real_tables.match_table.loc[real_tables.match_table.stage.eq("middle"), "season_id"]
    ) == {11, 12}
    rounds = stats.final_rounds_check(real_tables, n_bootstrap=50)
    pd.testing.assert_series_equal(
        rounds.groupby(level="stage").n.sum().sort_index(),
        stages.n_matches.sort_index(),
        check_names=False,
    )


@pytest.mark.parametrize(
    "function",
    [
        stats.table_settling,
        stats.settling_sensitivity,
        stats.split_summary,
        stats.stage_comparison,
        stats.playoff_form_summary,
        stats.final_rounds_check,
    ],
)
def test_course_bootstrap_count(function: Any, course_tables: AnalysisTables) -> None:
    small = _run_c(function, course_tables)
    pd.testing.assert_frame_equal(small, _run_c(function, course_tables))
    options = (
        {"upper_sizes": {8: None, 11: 2}}
        if function in (stats.table_settling, stats.settling_sensitivity, stats.split_summary)
        else {}
    )
    with pytest.raises(ValueError, match="n_bootstrap"):
        function(course_tables, n_bootstrap=0, **options)


def test_course_chart_contracts(course_tables: AnalysisTables) -> None:
    settling = _run_c(stats.table_settling, course_tables)
    movement = _run_c(stats.split_movement, course_tables)
    summary = _run_c(stats.split_summary, course_tables)
    stages = _run_c(stats.stage_comparison, course_tables)
    form = stats.playoff_form(course_tables)
    final = _run_c(stats.final_rounds_check, course_tables)
    race = stats.season_points_race(course_tables, [11])
    for kind, frame, extra in [
        ("settling", settling, None),
        ("leaders", settling, None),
        ("movement", movement, summary),
        ("stages", stages, None),
        ("form", form, stats.playoff_form_summary(course_tables, n_bootstrap=50)),
        ("last_rounds", final, None),
        ("race", race, None),
    ]:
        fig = chapter_c_chart(kind, frame, extra)
        assert fig.data and any(c.isdigit() for c in fig.layout.title.text)
        assert "?" not in fig.layout.title.text
        assert fig.to_json()


def test_fractional_leader_bottom_and_upper_boundary_credit(course_tables: AnalysisTables) -> None:
    rows = []
    for game_id, home, away in ((9001, 1, 3), (9002, 2, 4)):
        for team, won in ((home, True), (away, False)):
            rows.append(
                dict(
                    season_id=11,
                    team_id=team,
                    game_id=game_id,
                    stage="regular",
                    date=pd.Timestamp("2020-09-01", tz="Asia/Jerusalem"),
                    era="16_teams",
                    points_earned=3 if won else 0,
                    goal_difference=1 if won else -1,
                    goals_scored=1 if won else 0,
                )
            )
    teams = pd.DataFrame(rows).set_index(["team_id", "game_id"], drop=True)
    labels = course_tables.team_seasons.xs(11, level="season_id", drop_level=False).copy()
    labels["final_position"] = [1, 2, 3, 4]
    labels["relegated"] = [False, False, True, True]
    labels["final_stage"] = ["upper", "lower", "lower", "lower"]
    tables = replace(course_tables, team_matches=teams, team_seasons=labels)
    result = stats.table_settling(tables, [11], upper_sizes={11: 1}, n_bootstrap=50).loc[
        ("16_teams", 1)
    ]
    assert result.leader_won_title_rate == 0.5  # One champion out of two tied leaders.
    assert result.bottom_relegated_rate == 1  # Both tied bottom teams were relegated.
    assert result.in_upper_line_stayed_rate == 0.5  # One place shared by two tied clubs.
    assert result.in_upper_line_stayed_rate_team_weight == 1
    assert result.leader_won_title_rate_n == result.bottom_relegated_rate_n == 1
    assert result.n_tied_leader == result.n_tied_bottom == 1
    assert pd.isna(result.leader_won_title_excl_ties)
    assert pd.isna(result.bottom_relegated_excl_ties)
    assert result.leader_won_title_rate_ci_method == "bootstrap_seasons"
    split = stats.split_summary(tables, [11], upper_sizes={11: 1}, n_bootstrap=50)
    assert split.leader_at_split_won_title_rate.eq(0.5).all()
    assert split.leader_n.eq(1).all() and split.leader_tied_seasons.eq(1).all()
    extra = teams.reset_index()
    extra["season_id"] = 12
    extra["game_id"] += 100
    extra = extra.set_index(["team_id", "game_id"], drop=True)
    extra_labels = labels.reset_index()
    extra_labels["season_id"] = 12
    extra_labels.loc[extra_labels.team_id.eq(2), "final_stage"] = "upper"
    extra_labels = extra_labels.set_index(["season_id", "team_id"], drop=True)
    two_seasons = replace(
        tables,
        team_matches=pd.concat([teams, extra]),
        team_seasons=pd.concat([labels, extra_labels]),
    )
    result = stats.table_settling(two_seasons, [11, 12], upper_sizes={11: 1, 12: 2}, n_bootstrap=50)
    assert result.loc[("16_teams", 1), "in_upper_line_stayed_rate"] == 0.75
    assert result.loc[("16_teams", 1), "in_upper_line_stayed_rate_n"] == 2
    assert result.loc[("16_teams", 1), "in_upper_line_stayed_rate_team_weight"] == 3


B_FUNCTIONS = [
    stats.season_thresholds,
    stats.threshold_ranges,
    stats.critical_ties,
    stats.home_advantage_by_season,
    stats.home_advantage_periods,
    stats.era_comparison,
    stats.competitive_balance,
    stats.champions_summary,
    stats.calendar_effects,
    stats.postponed_summary,
]


def _run_b(
    function: Any, tables: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    options: dict[str, Any] = {}
    if function in (stats.season_thresholds, stats.threshold_ranges, stats.critical_ties):
        options["upper_sizes"] = {8: None, 11: 2}
    if function in (stats.era_comparison, stats.calendar_effects, stats.postponed_summary):
        options["n_bootstrap"] = 50
    return function(tables, seasons, **options)


@pytest.fixture
def history_tables() -> AnalysisTables:
    matches = []
    team_games = []
    seasons = []
    # Deliberately sporting points different from official/halved values.
    for season in (8, 11):
        for game, (home, away, hg, ag, date, stage) in enumerate(
            [
                (1, 4, 3, 0, "2020-09-05 14:59Z", "regular"),
                (2, 3, 2, 0, "2020-09-06 15:00Z", "regular"),
                (1, 2, 0, 0, "2020-09-07 16:59Z", "regular"),
                (3, 4, 1, 1, "2020-09-08 17:00Z", "regular"),
                (1, 3, 2, 1, "2020-09-09 19:00Z", "upper"),
            ]
        ):
            game_id = season * 10 + game
            matches.append(
                dict(
                    game_id=game_id,
                    season_id=season,
                    date=pd.Timestamp(date),
                    home_goals=hg,
                    away_goals=ag,
                    total_goals=hg + ag,
                    result="home_win" if hg > ag else "draw" if hg == ag else "away_win",
                    era="12_teams" if season == 8 else "16_teams",
                    is_postponed=game == 4,
                )
            )
            for team, scored, conceded in ((home, hg, ag), (away, ag, hg)):
                team_games.append(
                    dict(
                        team_id=team,
                        game_id=game_id,
                        season_id=season,
                        stage=stage,
                        points_earned=3 if scored > conceded else 1 if scored == conceded else 0,
                        era=matches[-1]["era"],
                    )
                )
        for team, points, n in ((1, 7, 3), (2, 4, 2), (3, 1, 3), (4, 1, 2)):
            seasons.append(
                dict(
                    season_id=season,
                    team_id=team,
                    team_name=f"Club {team}",
                    final_position=team,
                    points_from_matches=points,
                    matches_played=n,
                    points_per_match=points / n,
                    relegated=team == 4,
                    points_official=-100,
                    points_model=0,
                )
            )
    empty = pd.DataFrame()
    return AnalysisTables(
        pd.DataFrame(matches).set_index("game_id", drop=True),
        pd.DataFrame(team_games).set_index(["team_id", "game_id"], drop=True),
        empty,
        pd.DataFrame(seasons).set_index(["season_id", "team_id"], drop=True),
        empty,
        empty,
    )


@pytest.mark.parametrize("function", B_FUNCTIONS)
@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_history_development_scope(
    function: Any, seasons: list[int], history_tables: AnalysisTables
) -> None:
    with pytest.raises(ValueError, match="development"):
        _run_b(function, history_tables, seasons)


def test_history_thresholds_regular_stage_and_ties(history_tables: AnalysisTables) -> None:
    frame = _run_b(stats.season_thresholds, history_tables)
    assert pd.isna(frame.loc[8, "upper_line_points"])
    assert frame.loc[11, "upper_line_points"] == 4  # Both leaders tie at 4 before playoff.
    assert frame.loc[11, "champion_points"] == 7  # No halving or deduction.
    assert frame.champion_points.ge(frame.upper_line_points.fillna(0)).all()
    assert frame.loc[11, "upper_line_points"] >= frame.loc[11, "safety_points"]
    ties = _run_b(stats.critical_ties, history_tables)
    assert not ties.loc[11, "upper_line_tie"] and ties.relegation_tie.all()
    assert pd.isna(ties.loc[8, "upper_line_tie"])
    summary = stats.critical_ties_summary(ties)
    assert summary.loc["upper_line_tie", "n_applicable"] == 1
    tied = history_tables.team_matches.copy()
    tied.loc[
        (tied.index.get_level_values("team_id") == 3) & tied.stage.eq("regular"), "points_earned"
    ] = 2
    assert _run_b(stats.critical_ties, replace(history_tables, team_matches=tied)).loc[
        11, "upper_line_tie"
    ]
    altered = history_tables.team_matches.copy()
    altered.loc[altered.stage.eq("upper"), "points_earned"] = 1000
    pd.testing.assert_frame_equal(
        frame, _run_b(stats.season_thresholds, replace(history_tables, team_matches=altered))
    )
    with pytest.raises(ValueError, match="Missing playoff"):
        stats.season_thresholds(history_tables, upper_sizes={8: None})


def test_history_inverted_relegation_is_flagged(history_tables: AnalysisTables) -> None:
    teams = history_tables.team_seasons.copy()
    teams.loc[(11, 4), "points_from_matches"] = 2
    frame = _run_b(stats.season_thresholds, replace(history_tables, team_seasons=teams))
    assert frame.loc[11, "flag_inverted"]
    assert not frame.loc[8, "flag_inverted"]


def test_history_local_time_and_missing_dates(history_tables: AnalysisTables) -> None:
    output = _run_b(stats.calendar_effects, history_tables)
    assert output.loc[("kickoff_hour", "before_18"), "n"] == 2
    assert output.loc[("kickoff_hour", "18_to_20"), "n"] == 4
    assert output.loc[("kickoff_hour", "20_plus"), "n"] == 4
    assert output.loc[("weekday", "Saturday"), "n"] == 2
    assert output.groupby(level="dimension").n.sum().eq(10).all()
    matches = history_tables.match_table.copy()
    matches.loc[80, "date"] = pd.NaT
    missing = _run_b(stats.calendar_effects, replace(history_tables, match_table=matches))
    assert missing.xs("unknown", level="level").n.eq(1).all()
    assert missing.groupby(level="dimension").n.sum().eq(10).all()
    # UTC Sunday evening is already Monday in Israel: group by local date too.
    matches.loc[80, "date"] = pd.Timestamp("2020-09-06 22:00Z")
    local = _run_b(stats.calendar_effects, replace(history_tables, match_table=matches))
    assert local.loc[("weekday", "Monday"), "n"] == 3


@pytest.mark.parametrize("function", B_FUNCTIONS)
def test_history_immutability_indices_and_holdout_immunity(
    function: Any, history_tables: AnalysisTables
) -> None:
    before = [
        frame.copy(deep=True)
        for frame in (
            history_tables.match_table,
            history_tables.team_matches,
            history_tables.team_seasons,
        )
    ]

    def contaminate(frame: pd.DataFrame) -> pd.DataFrame:
        heldout = frame.copy()
        if "season_id" in heldout.columns:
            heldout["season_id"] = 26
        else:
            heldout = heldout.reset_index()
            heldout["season_id"] = 26
            heldout = heldout.drop_duplicates(list(frame.index.names)).set_index(
                list(frame.index.names), drop=True
            )
        return pd.concat([frame, heldout])

    contaminated = replace(
        history_tables,
        match_table=contaminate(history_tables.match_table),
        team_matches=contaminate(history_tables.team_matches),
        team_seasons=contaminate(history_tables.team_seasons),
    )
    result = _run_b(function, history_tables)
    pd.testing.assert_frame_equal(result, _run_b(function, contaminated))
    assert result.index.is_unique
    assert not set(result.index.names) & set(result.columns)
    for old, new in zip(
        before,
        (history_tables.match_table, history_tables.team_matches, history_tables.team_seasons),
        strict=True,
    ):
        pd.testing.assert_frame_equal(old, new)


@pytest.mark.parametrize(
    "function", [stats.era_comparison, stats.calendar_effects, stats.postponed_summary]
)
def test_history_bootstrap_control(function: Any, history_tables: AnalysisTables) -> None:
    pd.testing.assert_frame_equal(
        _run_b(function, history_tables), _run_b(function, history_tables)
    )
    with pytest.raises(ValueError, match="n_bootstrap"):
        function(history_tables, n_bootstrap=0)


def test_history_real_counts_and_boundaries(real_tables: AnalysisTables) -> None:
    sizes = load_playoff_upper_sizes()
    assert [sizes[s] for s in (8, 9, 10, 11, 12, 13, 14)] == [None, None, None, 6, 6, 8, 6]
    thresholds = stats.season_thresholds(real_tables, upper_sizes=sizes)
    assert thresholds.upper_line_points[thresholds.upper_line_points.isna()].index.tolist() == [
        8,
        9,
        10,
    ]
    assert thresholds.loc[13, "n_relegated"] == 3
    assert thresholds.drop(index=13).n_relegated.eq(2).all()
    assert thresholds.champion_points.ge(thresholds.upper_line_points.fillna(0)).all()
    # Different observation times: final survival totals can exceed regular-stage cutoff.
    assert thresholds.index[thresholds.upper_line_points.lt(thresholds.safety_points)].tolist() == [
        20,
        25,
    ]
    assert thresholds.index[thresholds.flag_inverted].tolist() == [18]
    assert thresholds.loc[11, "title_margin_points"] == -3  # Official title after halving.
    home = stats.home_advantage_by_season(real_tables)
    periods = stats.home_advantage_periods(real_tables)
    assert periods.n_seasons.tolist() == [6, 9]
    assert periods.mean_ppm_gap.tolist() == pytest.approx([0.377695, 0.135130], abs=0.000001)
    assert periods.mean_home_win_rate.tolist() == pytest.approx([0.428897, 0.388087], abs=0.000001)
    assert (home.home_win_rate + home.draw_rate + home.away_win_rate - 1).abs().lt(1e-12).all()
    for metric in ("home_win_rate", "draw_rate", "away_win_rate"):
        assert home[f"{metric}_ci_lower"].le(home[metric]).all()
        assert home[f"{metric}_ci_upper"].ge(home[metric]).all()
    rates = (
        home[["home_win_rate", "draw_rate", "away_win_rate"]].mul(home.n, axis=0).sum()
        / home.n.sum()
    )
    assert (rates * 100).round(1).tolist() == [40.5, 27.1, 32.4]
    assert stats.era_comparison(real_tables, n_bootstrap=50).n_matches.sum() == 4315
    calendar = stats.calendar_effects(real_tables, n_bootstrap=50)
    assert calendar.groupby(level="dimension").n.sum().eq(4315).all()
    postponed = stats.postponed_summary(real_tables, n_bootstrap=50)
    assert postponed.loc[True, "n"] == 49 and postponed.n.sum() == 4315
    assert stats.champions_summary(real_tables).titles.sum() == 18
    ties = stats.critical_ties_summary(stats.critical_ties(real_tables, upper_sizes=sizes))
    assert ties.n_ties.tolist() == [0, 1, 5]


def test_history_chart_contracts(history_tables: AnalysisTables) -> None:
    thresholds = _run_b(stats.season_thresholds, history_tables)
    thresholds["era"] = "14_teams"
    trends = _run_b(stats.home_advantage_by_season, history_tables).rename(
        columns={"n": "n_matches"}
    )
    for kind, frame, summary in [
        ("thresholds", thresholds, None),
        ("threshold_boxes", thresholds, None),
        (
            "home",
            _run_b(stats.home_advantage_by_season, history_tables),
            stats.home_advantage_periods(history_tables),
        ),
        ("eras", _run_b(stats.era_comparison, history_tables), None),
        ("season_trends", trends, None),
        (
            "balance",
            _run_b(stats.competitive_balance, history_tables),
            stats.champions_summary(history_tables),
        ),
        ("calendar", _run_b(stats.calendar_effects, history_tables), None),
    ]:
        figure = chapter_b_chart(kind, frame, summary)
        assert figure.data and figure.layout.title.text and figure.to_json()


def test_home_period_means_weight_seasons_not_matches(history_tables: AnalysisTables) -> None:
    base = history_tables.match_table.loc[history_tables.match_table.season_id.eq(8)]
    groups = []
    for season, positions in ((11, list(range(5))), (12, [0]), (17, [2])):
        group = base.iloc[positions].copy()
        group["season_id"] = season
        group.index = pd.Index([season * 100 + i for i in range(len(group))], name="game_id")
        groups.append(group)
    tables = replace(history_tables, match_table=pd.concat(groups))
    periods = stats.home_advantage_periods(tables)
    early, late = periods.iloc[0], periods.iloc[1]
    assert early.n_seasons == 2 and early.n_matches == 6
    assert early.mean_home_win_rate == pytest.approx(0.8)  # Mean of 3/5 and 1/1, not 4/6.
    assert early.mean_ppm_gap == pytest.approx(2.4)  # Mean of +1.8 and +3.0.
    assert late.n_seasons == 1 and late.mean_ppm_gap == 0
    subset = stats.home_advantage_periods(tables, [17])
    assert subset.iloc[0].n_seasons == 0 and pd.isna(subset.iloc[0].mean_ppm_gap)
    assert subset.iloc[1].n_seasons == 1


def test_chapter_b_titles_are_computed_from_changed_data(history_tables: AnalysisTables) -> None:
    thresholds = _run_b(stats.season_thresholds, history_tables)
    thresholds["champion_ppm"] = 2.1
    thresholds["safety_ppm"] = 0.9
    title = chapter_b_chart("thresholds", thresholds).layout.title.text.replace("<br>", " ")
    assert "2.1–2.1" in title and "0.90" in title
    home = stats.home_advantage_by_season(history_tables)
    periods = stats.home_advantage_periods(history_tables)
    periods["n_seasons"] = [2, 2]
    periods["mean_ppm_gap"] = [0.5, 0.1]
    periods["mean_home_win_rate"] = [0.45, 0.35]
    title = chapter_b_chart("home", home, periods).layout.title.text.replace("<br>", " ")
    assert "+0.50" in title and "+0.10" in title and "45%" in title and "35%" in title
    champions = pd.DataFrame(
        {"team_name": ["TA", "Haifa", "Other"], "titles": [2, 3, 1]},
        index=pd.Index([1061, 1005, 99], name="team_id"),
    )
    title = chapter_b_chart(
        "balance", stats.competitive_balance(history_tables), champions
    ).layout.title.text.replace("<br>", " ")
    assert "3 מועדונים" in title and "6 אליפויות" in title and "ב־5" in title
    calendar = stats.calendar_effects(history_tables, n_bootstrap=50)
    calendar.loc[("kickoff_hour", "20_plus"), "goals_per_match"] = 2.7
    calendar.loc[("kickoff_hour", "18_to_20"), "goals_per_match"] = 2.2
    calendar["goals_per_match_ci_lower"] = 2.0
    calendar["goals_per_match_ci_upper"] = 3.0
    title = chapter_b_chart("calendar", calendar).layout.title.text.replace("<br>", " ")
    assert "2.70" in title and "2.20" in title and "בתוך טווחי אי־הוודאות" in title
    calendar.loc[("kickoff_hour", "20_plus"), "goals_per_match_ci_lower"] = 2.6
    calendar.loc[("kickoff_hour", "18_to_20"), "goals_per_match_ci_upper"] = 2.3
    title = chapter_b_chart("calendar", calendar).layout.title.text.replace("<br>", " ")
    assert "בתוך טווחי אי־הוודאות" not in title


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


def test_playoff_form_summary(course_tables: AnalysisTables) -> None:
    summary = stats.playoff_form_summary(course_tables, iter([8, 11]), n_bootstrap=50)
    assert summary.n.to_dict() == {"lower": 2, "upper": 2}
    assert summary.n_seasons.eq(1).all()
    assert summary.loc["upper", "mean_ppm_change"] == pytest.approx(0)
    assert summary.loc["lower", "mean_ppm_change"] == pytest.approx(2 / 3)
    assert summary.loc["lower", "median_ppm_change"] == pytest.approx(2 / 3)
    assert summary.mean_ppm_change_ci_lower.isna().all()
    # Duplicate the entire season: identical blocks give an exact CI, despite team variation.
    teams = course_tables.team_matches.reset_index()
    teams = teams.loc[teams.season_id.eq(11)].copy()
    teams["season_id"] = 12
    teams["game_id"] += 1000
    labels = course_tables.team_seasons.xs(11, level="season_id").reset_index()
    labels["season_id"] = 12
    doubled = replace(
        course_tables,
        team_matches=pd.concat(
            [
                course_tables.team_matches,
                teams.set_index(["team_id", "game_id"], drop=True),
            ]
        ),
        team_seasons=pd.concat(
            [
                course_tables.team_seasons,
                labels.set_index(["season_id", "team_id"], drop=True),
            ]
        ),
    )
    result = stats.playoff_form_summary(doubled, [11, 12], n_bootstrap=50)
    assert result.n.eq(4).all() and result.n_seasons.eq(2).all()
    for stage in ("upper", "lower"):
        assert result.loc[stage, "mean_ppm_change_ci_lower"] == pytest.approx(
            result.loc[stage, "mean_ppm_change"]
        )
        assert result.loc[stage, "mean_ppm_change_ci_upper"] == pytest.approx(
            result.loc[stage, "mean_ppm_change"]
        )


# Chapter D: independent synthetic match flow and optional real-data controls.
D_FUNCTIONS = [
    stats.scoreline_distribution,
    stats.scoring_rate_by_state,
    stats.top_scorelines,
    stats.poisson_reference,
    stats.observed_vs_poisson,
    stats.goal_count_dispersion,
    stats.draw_types,
    stats.timeline_exclusions,
    stats.game_timelines,
    stats.timeline_team_summary,
    stats.timeline_resilience,
    stats.halftime_transitions,
    stats.zero_zero_at_60,
    stats.first_goal_effect,
    stats.first_goal_minute_distribution,
    stats.goal_timing,
    stats.team_late_goals,
    stats.penalty_owngoal_stats,
    stats.penalty_owngoal_by_season,
    stats.red_card_stats,
]
D_BOOTSTRAP = {
    stats.goal_count_dispersion,
    stats.scoring_rate_by_state,
    stats.timeline_resilience,
    stats.halftime_transitions,
    stats.first_goal_effect,
    stats.red_card_stats,
}


def _flow_cards() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(game_id=801, team_id=1, minute=pd.NA, is_second_yellow=True),
            dict(game_id=801, team_id=2, minute=65, is_second_yellow=False),
            dict(game_id=803, team_id=1, minute=60, is_second_yellow=False),
        ]
    ).astype({"minute": "Int64"})


def _run_d(
    function: Any, tables: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    options: dict[str, Any] = {"n_bootstrap": 50} if function in D_BOOTSTRAP else {}
    if function == stats.red_card_stats:
        options["red_cards"] = _flow_cards()
    return function(tables, seasons, **options)


@pytest.fixture
def flow_tables() -> AnalysisTables:
    from ligat.analysis.prepare import _minute_bucket

    matches, teams, goals, labels = [], [], [], []
    schedules = [
        [(1, 1, 46), (2, 2, 93)],
        [],
        [(1, 1, 10), (2, 1, 20), (2, 1, 30)],
        [(2, 1, 10), (1, 1, 20)],
        [(1, 2, 60)],
        [(2, 2, 61)],
        [(1, 1, 4), (2, 1, 17)],
    ]
    for season in range(8, 13):
        for number, events in enumerate(schedules, start=1):
            game_id = season * 100 + number
            conflict = number == 7
            hg = sum(t == 1 for t, _, _ in events)
            ag = len(events) - hg
            ht = {t: sum(team == t and half == 1 for team, half, _ in events) for t in (1, 2)}
            matches.append(
                dict(
                    game_id=game_id,
                    season_id=season,
                    home_team_id=1,
                    away_team_id=2,
                    home_goals=hg,
                    away_goals=ag,
                    total_goals=hg + ag,
                    ht_conflict=conflict,
                    goals_available=True,
                )
            )
            for team, other in ((1, 2), (2, 1)):
                diff = hg - ag if team == 1 else ag - hg
                hdiff = ht[team] - ht[other]
                teams.append(
                    dict(
                        game_id=game_id,
                        season_id=season,
                        team_id=team,
                        opponent_id=other,
                        is_home=team == 1,
                        result="win" if diff > 0 else "loss" if diff < 0 else "draw",
                        points_earned=3 if diff > 0 else 0 if diff < 0 else 1,
                        ht_state=pd.NA
                        if conflict
                        else "leading"
                        if hdiff > 0
                        else "trailing"
                        if hdiff < 0
                        else "level",
                        scored_first=team == events[0][0] if events else pd.NA,
                    )
                )
            for k, (team, half, minute) in enumerate(events, start=1):
                goals.append(
                    dict(
                        game_id=game_id,
                        goal_number=k,
                        team_id=team,
                        season_id=season,
                        half=half,
                        minute=minute,
                        added_time=pd.NA,
                        score_before=(
                            "leading"
                            if sum(t == team for t, _, _ in events[: k - 1])
                            > sum(t != team for t, _, _ in events[: k - 1])
                            else "trailing"
                            if sum(t == team for t, _, _ in events[: k - 1])
                            < sum(t != team for t, _, _ in events[: k - 1])
                            else "level"
                        ),
                        is_first_half_stoppage=half == 1 and minute > 45,
                        minute_bucket=_minute_bucket(half, minute),
                        is_penalty=number == 3 and k == 3,
                        is_own_goal=number == 3 and k == 1,
                    )
                )
        labels += [dict(season_id=season, team_id=t, team_name=f"Club {t}") for t in (1, 2)]
    matches.append(
        dict(
            game_id=864982,
            season_id=8,
            home_team_id=1,
            away_team_id=2,
            home_goals=1,
            away_goals=2,
            total_goals=3,
            ht_conflict=False,
            goals_available=False,
        )
    )
    for team in (1, 2):
        teams.append(
            dict(
                game_id=864982,
                season_id=8,
                team_id=team,
                opponent_id=3 - team,
                is_home=team == 1,
                result="loss" if team == 1 else "win",
                points_earned=0 if team == 1 else 3,
                ht_state=pd.NA,
                scored_first=pd.NA,
            )
        )
    empty = pd.DataFrame()
    return AnalysisTables(
        pd.DataFrame(matches).set_index("game_id", drop=True),
        pd.DataFrame(teams).set_index(["team_id", "game_id"], drop=True),
        pd.DataFrame(goals).set_index(["game_id", "goal_number"], drop=True),
        pd.DataFrame(labels).set_index(["season_id", "team_id"], drop=True),
        empty,
        empty,
    )


@pytest.mark.parametrize("function", D_FUNCTIONS)
@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_flow_development_scope(
    function: Any, seasons: list[int], flow_tables: AnalysisTables
) -> None:
    with pytest.raises(ValueError, match="development"):
        _run_d(function, flow_tables, seasons)


def test_flow_timeline_invariants(flow_tables: AnalysisTables) -> None:
    timeline = stats.game_timelines(flow_tables)
    assert (
        timeline[["minutes_leading", "minutes_level", "minutes_trailing"]].sum(axis=1).eq(90).all()
    )
    assert len(timeline) + 2 * timeline.attrs["n_excluded_matches"] == len(flow_tables.team_matches)
    for _, group in timeline.groupby(level="game_id"):
        assert group.minutes_leading.iloc[0] == group.minutes_trailing.iloc[1]
    assert timeline.points_dropped_from_lead.dropna().isin([0, 2, 3]).all()
    assert timeline.points_won_from_behind.dropna().isin([0, 1, 3]).all()
    assert timeline.loc[(802, 1), "minutes_level"] == 90
    assert timeline.loc[(801, 1), "minutes_leading"] == 45  # first-half 46 -> clock 45
    assert timeline.loc[(801, 2), "late_goal_for"] == 1
    assert timeline.loc[(801, 1), "first_half_stoppage_goals"] == 1
    assert timeline.loc[(801, 1), "points_dropped_from_lead"] == 2
    assert pd.isna(timeline.loc[(802, 1), "points_dropped_from_lead"])
    assert pd.isna(timeline.loc[(802, 1), "points_won_from_behind"])
    assert (
        timeline.loc[(803, 1), "led_at_any_point"]
        and timeline.loc[(803, 1), "trailed_at_any_point"]
    )
    assert timeline.loc[(803, 1), "points_dropped_from_lead"] == 3
    assert timeline.loc[(803, 1), "points_won_from_behind"] == 0
    assert timeline.loc[(804, 1), "equalized_within_10"]
    assert not timeline.loc[(801, 2), "equalized_within_10"]
    assert timeline.attrs["exclusions"] == {"ht_conflict": 5, "unreliable_scorers_864982": 1}


def test_flow_distributions_and_denominators(flow_tables: AnalysisTables) -> None:
    observed = stats.scoreline_distribution(flow_tables, max_goals=0)
    assert observed.to_numpy().sum() == pytest.approx(1 - observed.attrs["n_beyond"] / 36)
    poisson = stats.poisson_reference(flow_tables)
    assert poisson.to_numpy().sum() + poisson.attrs["tail_mass"] == pytest.approx(1)
    transitions = stats.halftime_transitions(flow_tables, n_bootstrap=50)
    assert transitions[["win_rate", "draw_rate", "loss_rate"]].sum(axis=1).eq(1).all()
    assert transitions.n.sum() + transitions.attrs["n_excluded_rows"] == 72
    effect = stats.first_goal_effect(flow_tables, n_bootstrap=50)
    assert (
        effect.xs(True, level="scored_first").n.sum()
        == effect.xs(False, level="scored_first").n.sum()
    )
    at60 = stats.zero_zero_at_60(flow_tables)
    assert at60.n.eq(10).all()  # 0:0 and minute-61 opener; minute-60 opener excluded
    assert at60["count"].sum() == 10
    timing = stats.goal_timing(flow_tables)
    assert timing["count"].sum() == 45  # 9 eligible goals per synthetic season
    penalties = stats.penalty_owngoal_stats(flow_tables)
    assert penalties.penalty_for_n.sum() == penalties.penalty_against_n.sum() == 5
    assert penalties.own_goal_for_n.sum() == penalties.own_goal_against_n.sum() == 5
    assert penalties.loc[(8, 1), "own_goal_for_n"] == 1  # own goal credited to beneficiary
    cards = stats.red_card_stats(flow_tables, red_cards=_flow_cards(), n_bootstrap=50)
    assert cards.xs("cards", level="part").n.sum() == 72
    assert cards.attrs["n_missing_minute"] == 1
    assert cards.xs("first_card_minute", level="part").n.sum() == 2
    assert cards.xs("second_yellow", level="part").second_yellow_n.sum() == 1


@pytest.mark.parametrize("function", list(D_BOOTSTRAP))
def test_flow_bootstrap_deterministic(function: Any, flow_tables: AnalysisTables) -> None:
    pd.testing.assert_frame_equal(_run_d(function, flow_tables), _run_d(function, flow_tables))
    options = {"red_cards": _flow_cards()} if function == stats.red_card_stats else {}
    with pytest.raises(ValueError, match="n_bootstrap"):
        function(flow_tables, n_bootstrap=0, **options)


def test_flow_pure_inputs_and_indexes(flow_tables: AnalysisTables) -> None:
    from dataclasses import fields

    originals = {f.name: getattr(flow_tables, f.name).copy(deep=True) for f in fields(flow_tables)}
    for function in D_FUNCTIONS:
        frame = _run_d(function, flow_tables)
        assert frame.index.is_unique
        assert not set(frame.index.names) & set(frame.columns)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(getattr(flow_tables, name), original)


def test_flow_real_controls(real_tables: AnalysisTables, real_timelines: pd.DataFrame) -> None:
    rate = stats.scoring_rate_by_state(real_tables, n_bootstrap=50)
    assert rate.minutes.sum() == 4301 * 180
    for home, expected in [(True, [1.62, 1.19, 1.29]), (False, [1.35, 1.03, 1.17])]:
        assert rate.xs(home, level="is_home").loc[
            ["leading", "level", "trailing"], "goals_per_90"
        ].tolist() == pytest.approx(expected, abs=0.01)
    resilience = stats.timeline_resilience(real_tables, n_bootstrap=50)
    for team, expected in {
        2171: (0.80, 0.44),
        1061: (0.79, 0.40),
        1005: (0.77, 0.41),
        2176: (0.53, 0.72),
    }.items():
        assert resilience.loc[
            team, ["points_won_per_trailed_game", "points_dropped_per_led_game"]
        ].tolist() == pytest.approx(expected, abs=0.01)
    observed = stats.scoreline_distribution(real_tables)
    assert observed.attrs["n_matches"] == 4315
    assert observed.to_numpy().sum() == pytest.approx(1 - observed.attrs["n_beyond"] / 4315)
    timeline = real_timelines.copy(deep=True)
    assert len(timeline) + 2 * timeline.attrs["n_excluded_matches"] == 8630
    assert (
        timeline[["minutes_leading", "minutes_level", "minutes_trailing"]].sum(axis=1).eq(90).all()
    )
    transitions = stats.halftime_transitions(real_tables, n_bootstrap=50)
    assert transitions.n.sum() + transitions.attrs["n_excluded_rows"] == 8630
    effect = stats.first_goal_effect(real_tables, n_bootstrap=50)
    assert (
        effect.xs(True, level="scored_first").n.sum()
        == effect.xs(False, level="scored_first").n.sum()
    )
    cards = Repository().read_filtered(
        "red_cards", [("game_id", "in", real_tables.match_table.index.tolist())]
    )
    summary = stats.red_card_stats(real_tables, red_cards=cards, n_bootstrap=50)
    assert summary.xs("cards", level="part").n.sum() == 8630


def test_flow_charts_and_dispersion(flow_tables: AnalysisTables) -> None:
    from ligat.viz.match_flow import chapter_d_chart

    cases = {
        "scorelines": stats.scoreline_distribution,
        "poisson": stats.observed_vs_poisson,
        "dispersion": stats.goal_count_dispersion,
        "halftime": stats.halftime_transitions,
        "first_goal": stats.first_goal_effect,
        "timing": stats.scoring_rate_by_state,
        "time_in_state": stats.timeline_team_summary,
        "resilience": stats.timeline_resilience,
        "late_goals": stats.team_late_goals,
        "penalties": stats.penalty_owngoal_by_season,
        "red_cards": stats.red_card_stats,
    }
    for kind, fn in cases.items():
        fig = chapter_d_chart(kind, _run_d(fn, flow_tables))
        assert fig.data and any(c.isdigit() for c in fig.layout.title.text)
        assert "?" not in fig.layout.title.text
        assert fig.to_json()
    dispersion = stats.goal_count_dispersion(flow_tables, n_bootstrap=50)
    source = flow_tables.match_table
    assert dispersion.loc["home_goals", "variance"] == pytest.approx(source.home_goals.var())
    assert dispersion.loc["away_goals", "mean"] == pytest.approx(source.away_goals.mean())
    assert dispersion.loc["total_goals", "dispersion"] == pytest.approx(
        source.total_goals.var() / source.total_goals.mean()
    )
    assert dispersion.iloc[0].home_away_correlation == pytest.approx(
        source.home_goals.corr(source.away_goals)
    )


def test_flow_same_minute_zero_duration(flow_tables: AnalysisTables) -> None:
    events = flow_tables.goal_events.copy()
    events.loc[(804, 2), "minute"] = 10
    result = stats.game_timelines(replace(flow_tables, goal_events=events))
    assert result.loc[(804, 1), "minutes_trailing"] == 0
    assert result.loc[(804, 1), "trailed_at_any_point"]
    assert result.loc[(804, 1), "equalized_within_10"]
    assert result.loc[(804, 1), "points_won_from_behind"] == 1


def test_scoring_rate_exposure_and_unknown_states(flow_tables: AnalysisTables) -> None:
    matches = flow_tables.match_table.loc[[803]]
    teams = flow_tables.team_matches.loc[
        flow_tables.team_matches.index.get_level_values("game_id") == 803
    ]
    tables = replace(flow_tables, match_table=matches, team_matches=teams)
    frame = stats.scoring_rate_by_state(tables, [8], n_bootstrap=50)
    assert frame.loc[(True, "leading"), "minutes"] == 10
    assert frame.loc[(True, "level"), "minutes"] == 20
    assert frame.loc[(True, "trailing"), "minutes"] == 60
    assert frame.loc[(True, "level"), "goals_per_90"] == 4.5
    assert frame.loc[(False, "trailing"), "goals_per_90"] == 9
    assert frame.loc[(False, "level"), "goals_per_90"] == 4.5
    assert frame.n_goals.sum() == 3
    assert frame.minutes.sum() == 180
    assert frame.goals_per_90_ci_lower.isna().all()  # Only one season.
    events = tables.goal_events.copy()
    events.loc[(803, 2), "score_before"] = pd.NA
    unknown = stats.scoring_rate_by_state(replace(tables, goal_events=events), [8], n_bootstrap=50)
    assert unknown.attrs["n_missing_score_before"] == 1
    assert unknown.attrs["missing_score_before_by_venue"] == {"home": 0, "away": 1}
    assert unknown.n_goals.sum() == 2 and unknown.minutes.sum() == 180
    zero = replace(
        tables,
        match_table=flow_tables.match_table.loc[[802]],
        team_matches=flow_tables.team_matches.loc[
            flow_tables.team_matches.index.get_level_values("game_id") == 802
        ],
    )
    result = stats.scoring_rate_by_state(zero, [8], n_bootstrap=50)
    assert pd.isna(result.loc[(True, "leading"), "goals_per_90"])
    assert result.loc[(True, "level"), "goals_per_90"] == 0


def test_resilience_rates_use_exposed_games(flow_tables: AnalysisTables) -> None:
    frame = stats.timeline_resilience(flow_tables, n_bootstrap=50)
    for team in (1, 2):
        row = frame.loc[team]
        assert row.points_won_per_trailed_game == pytest.approx(
            row.points_won_from_behind / row.n_trailed
        )
        assert row.points_dropped_per_led_game == pytest.approx(
            row.points_dropped_from_lead / row.n_led
        )
        # Identical season blocks: within-season team/game variation must not widen the CI.
        for metric in ("points_won_per_trailed_game", "points_dropped_per_led_game"):
            assert row[f"{metric}_ci_lower"] == pytest.approx(row[metric])
            assert row[f"{metric}_ci_upper"] == pytest.approx(row[metric])


# Chapter E: context, exact funnels and matched coach controls.
E_FUNCTIONS = [
    stats.referee_stats,
    stats.referee_home_bias,
    stats.venue_effects,
    stats.venue_summary,
    stats.coach_change_effect,
    stats.coach_summary,
    stats.derby_and_big_games,
    stats.upset_rates,
]


def _context_config() -> list[dict[str, Any]]:
    from ligat.analysis.league_stats.context import CITY_CLUBS

    names = [name for group in CITY_CLUBS.values() for name in group]
    # IDs are fixture-only; production resolves config records.
    return [dict(team_id=i + 1, name_he=name, aliases=[]) for i, name in enumerate(names)]


def _context_cards() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(game_id=1201, team_id=1, minute=55, is_second_yellow=False),
            dict(game_id=1201, team_id=1, minute=75, is_second_yellow=True),
            dict(game_id=1201, team_id=2, minute=pd.NA, is_second_yellow=False),
        ]
    )


@pytest.fixture
def context_tables() -> AnalysisTables:
    from ligat.analysis.league_stats.context import CITY_CLUBS

    names = [name for group in CITY_CLUBS.values() for name in group]
    matches, teams, goals, labels, spells = [], [], [], [], []
    for season in (12, 13):
        for number in range(1, 13):
            for pair, (a, b) in enumerate(((1, 2), (3, 4), (5, 6))):
                game_id = season * 100 + pair * 20 + number
                # Team 1: +1.2 ppm after change; team 3: +0.6, the only nearby control.
                a_wins = (
                    number in (5, 6, 7, 8) if a == 1 else number in (5, 6, 7) if a == 3 else True
                )
                home, away = (a, b) if number % 2 else (b, a)
                winner = a if a_wins else b
                hg, ag = int(winner == home), int(winner == away)
                date = pd.Timestamp(f"{season + 1998}-08-01", tz="Asia/Jerusalem") + pd.Timedelta(
                    days=number
                )
                venue = "Shared" if home in (1, 3) else f"Ground {home}"
                matches.append(
                    dict(
                        game_id=game_id,
                        season_id=season,
                        date=date,
                        home_team_id=home,
                        away_team_id=away,
                        home_goals=hg,
                        away_goals=ag,
                        total_goals=hg + ag,
                        referee=f"Ref {pair}",
                        venue=venue,
                        is_home_venue=not (home == 1 and number == 3),
                        goals_available=True,
                    )
                )
                goals.append(
                    dict(
                        game_id=game_id,
                        goal_number=1,
                        season_id=season,
                        team_id=winner,
                        is_penalty=number % 5 == 0,
                    )
                )
                for team, _other in ((a, b), (b, a)):
                    won = team == winner
                    coach = 200 + team if team == 1 and number >= 6 else 100 + team
                    teams.append(
                        dict(
                            team_id=team,
                            game_id=game_id,
                            season_id=season,
                            date=date,
                            match_number=number,
                            points_earned=3 if won else 0,
                            is_home=team == home,
                            goals_scored=int(won),
                            goals_conceded=int(not won),
                            coach_id=coach,
                            coach_name=f"Coach {coach}",
                        )
                    )
        for team in range(1, 7):
            labels.append(
                dict(
                    season_id=season,
                    team_id=team,
                    team_name=names[team - 1],
                    final_position=team,
                    relegated=team == 6,
                    value_share_of_league=team / 21,
                    prev_season_value_share=pd.NA if season == 12 or team == 6 else team / 21,
                    promoted_in=season == 13 and team == 6,
                )
            )
            starts = (1, 6) if team == 1 else (1,)
            for offset, start in enumerate(starts):
                first = pd.Timestamp(f"{season + 1998}-08-01", tz="Asia/Jerusalem") + pd.Timedelta(
                    days=start
                )
                spells.append(
                    dict(
                        team_id=team,
                        spell_number=(season - 12) * 2 + offset + 1,
                        season_id_start=season,
                        season_id_end=season,
                        first_date=first,
                        last_date=first + pd.Timedelta(days=4 if start == 1 and team == 1 else 6),
                        coach_name=f"Coach {200 + team if start == 6 else 100 + team}",
                        matches=5 if start == 1 and team == 1 else 7 if team == 1 else 12,
                        is_mid_season_start=start == 6,
                        is_left_censored=False,
                        is_flicker=False,
                    )
                )
    return AnalysisTables(
        pd.DataFrame(matches).set_index("game_id", drop=True),
        pd.DataFrame(teams).set_index(["team_id", "game_id"], drop=True),
        pd.DataFrame(goals).set_index(["game_id", "goal_number"], drop=True),
        pd.DataFrame(labels).set_index(["season_id", "team_id"], drop=True),
        pd.DataFrame(),
        pd.DataFrame(spells).set_index(["team_id", "spell_number"], drop=True),
    )


def _run_e(fn: Any, tables: AnalysisTables, seasons: Any = DEVELOPMENT_SEASONS) -> pd.DataFrame:
    options: dict[str, Any] = {"n_bootstrap": 50}
    if fn in (stats.referee_stats, stats.referee_home_bias):
        options["red_cards"] = _context_cards()
    if fn in (stats.referee_stats, stats.venue_summary):
        options["min_matches"] = 3
    if fn == stats.derby_and_big_games:
        options["teams_config"] = _context_config()
    if fn in (stats.coach_change_effect, stats.coach_summary) and seasons is DEVELOPMENT_SEASONS:
        seasons = range(12, 26)
    return fn(tables, seasons, **options)


@pytest.mark.parametrize("function", E_FUNCTIONS)
@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
def test_context_development_scope(
    function: Any, seasons: Any, context_tables: AnalysisTables
) -> None:
    with pytest.raises(ValueError, match="development"):
        _run_e(function, context_tables, seasons)


def test_exact_funnels_and_overdispersion() -> None:
    import numpy as np

    from ligat.analysis.league_stats.context import _add_funnel

    rng = np.random.default_rng(20261015)
    samples = rng.poisson(0.2 * 100, 10000)
    frame = stats.funnel_limits_poisson(0.2, [100, 10000])
    inside = (
        (samples / 100 >= frame.loc[100, "lower95"]) & (samples / 100 <= frame.loc[100, "upper95"])
    ).mean()
    assert 0.94 < inside < 0.98
    for fn, mean in [(stats.funnel_limits_poisson, 0.2), (stats.funnel_limits_binomial, 0.4)]:
        limits = fn(mean, [30, 10000])
        assert (
            limits.loc[10000, "upper95"] - limits.loc[10000, "lower95"]
            < limits.loc[30, "upper95"] - limits.loc[30, "lower95"]
        )
        assert limits.loc[10000, "lower95"] <= mean <= limits.loc[10000, "upper95"]
        with pytest.raises(ValueError):
            fn(mean, [0])
    assert stats.funnel_limits is stats.funnel_limits_binomial
    dispersed = pd.DataFrame(
        {"n_matches": [100] * 10, "n_events": [0, 0, 0, 0, 0, 100, 100, 100, 100, 100]}
    )
    dispersed["per_match"] = dispersed.n_events / 100
    _add_funnel(dispersed, "per_match", "n_events", "n_matches", 0.5, "poisson", 30)
    model = dispersed.attrs["funnel_models"]["per_match"]
    assert model["phi"] > 1 and model["widened"] and model["overdispersion_clear"]
    assert (dispersed.per_match_upper95_expanded > dispersed.per_match_upper95).all()


def test_coach_matched_synthetic_control(context_tables: AnalysisTables) -> None:
    effect = stats.coach_change_effect(context_tables, n_bootstrap=50)
    assert len(effect) == 2
    assert effect.n_before.eq(5).all() and effect.n_after.eq(5).all()
    assert effect.ppm_before.tolist() == pytest.approx([0.6, 0.6])
    assert effect.ppm_after.tolist() == pytest.approx([1.8, 1.8])
    assert effect.control_diff.tolist() == pytest.approx([0.6, 0.6])
    assert effect.effect.tolist() == pytest.approx([0.6, 0.6])
    assert effect.control_team_ids.tolist() == [(3,), (3,)]
    assert not effect.is_left_censored.any() and not effect.is_flicker.any()
    with pytest.raises(ValueError, match="12–25"):
        stats.coach_change_effect(context_tables, [11])
    spells = context_tables.coach_spells.copy()
    spells.loc[(1, 2), "is_left_censored"] = True
    spells.loc[(1, 4), "is_flicker"] = True
    assert stats.coach_change_effect(
        replace(context_tables, coach_spells=spells), n_bootstrap=50
    ).empty
    spells = context_tables.coach_spells.reset_index()
    # A control coach change in the ten-game interval excludes that club.
    extra = spells.loc[spells.team_id.eq(3) & spells.season_id_start.eq(12)].copy()
    extra["spell_number"] = 99
    extra["is_mid_season_start"] = True
    extra["first_date"] += pd.Timedelta(days=5)
    spells = pd.concat([spells, extra]).set_index(["team_id", "spell_number"], drop=True)
    row = stats.coach_change_effect(
        replace(context_tables, coach_spells=spells), n_bootstrap=50
    ).loc[(1, 2)]
    assert row.n_controls == 0 and pd.isna(row.effect)


def test_context_disjoint_groups_and_unknowns(context_tables: AnalysisTables) -> None:
    kinds = _run_e(stats.derby_and_big_games, context_tables)
    assert kinds.n.sum() == 72
    allowed = {tuple(x["team_ids"]) for x in kinds.attrs["city_pairs"]}
    assert set(kinds.loc["city_derby", "pairs"]) <= allowed
    venue = _run_e(stats.venue_effects, context_tables)
    assert venue.loc[False, "n"] == 2
    refs = _run_e(stats.referee_stats, context_tables)
    assert refs.n_matches.sum() + refs.attrs["n_missing_referee"] == 72
    assert refs.red_cards_count.sum() == 3  # Multiple cards in one game retained.
    upset = _run_e(stats.upset_rates, context_tables)
    assert upset.groupby(level="metric").n.sum().eq(72).all()
    assert upset[["home_win_rate", "draw_rate", "away_win_rate"]].sum(axis=1).eq(1).all()
    unknown = upset.index.get_level_values("weaker_side").isin(
        ["unknown", "home_promoted", "away_promoted"]
    )
    assert upset.loc[unknown, ["weak_win_rate", "strong_win_rate"]].isna().all().all()
    promoted = upset.xs("veteran_vs_promoted", level="gap_bin")
    assert promoted[["veteran_win_rate", "draw_rate", "promoted_win_rate"]].sum(axis=1).eq(1).all()
    equal_seasons = context_tables.team_seasons.copy()
    equal_seasons.loc[(13, 2), "prev_season_value_share"] = equal_seasons.loc[
        (13, 1), "prev_season_value_share"
    ]
    equal = stats.upset_rates(
        replace(context_tables, team_seasons=equal_seasons), [13], n_bootstrap=50
    ).xs("equal_strength", level="gap_bin")
    assert equal.n.sum() == 12
    assert equal[["weak_win_rate", "strong_win_rate"]].isna().all().all()
    assert equal[["home_win_rate", "draw_rate", "away_win_rate"]].sum(axis=1).eq(1).all()


def test_context_purity_and_bootstrap(context_tables: AnalysisTables) -> None:
    from dataclasses import fields

    originals = {
        f.name: getattr(context_tables, f.name).copy(deep=True) for f in fields(context_tables)
    }
    for fn in E_FUNCTIONS:
        output = _run_e(fn, context_tables)
        assert output.index.is_unique
        assert not set(output.index.names) & set(output.columns)
        pd.testing.assert_frame_equal(output, _run_e(fn, context_tables))
        for name, original in originals.items():
            pd.testing.assert_frame_equal(original, getattr(context_tables, name))


def test_context_real_controls(real_tables: AnalysisTables) -> None:
    from pathlib import Path

    import yaml

    reds = Repository().read_filtered(
        "red_cards", [("game_id", "in", real_tables.match_table.index.tolist())]
    )
    refs = stats.referee_stats(real_tables, red_cards=reds, n_bootstrap=50)
    assert refs.n_matches.sum() + refs.attrs["n_missing_referee"] == 4315
    venue = stats.venue_effects(real_tables, n_bootstrap=50)
    assert venue.loc[False, "n"] == venue.attrs["n_borrowed_all"] == 194
    coaches = stats.coach_change_effect(real_tables, n_bootstrap=50)
    assert coaches.attrs["n_clean_mid_season"] == 246
    assert coaches.n_before.eq(5).all() and coaches.n_after.eq(5).all()
    for (team, _), row in coaches.iterrows():
        assert team not in row.control_team_ids
    config = yaml.safe_load(Path("config/teams.yaml").read_text(encoding="utf-8"))["teams"]
    kinds = stats.derby_and_big_games(real_tables, teams_config=config, n_bootstrap=50)
    assert kinds.n.sum() == 4315
    assert {row["team_id"] for row in kinds.attrs["big_teams"]} == {1061, 1005, 2171, 3595}
    upset = stats.upset_rates(real_tables, n_bootstrap=50)
    assert upset.groupby(level="metric").n.sum().eq(4315).all()
    unknown = upset.index.get_level_values("weaker_side").isin(
        ["unknown", "home_promoted", "away_promoted"]
    )
    assert upset.loc[unknown, ["weak_win_rate", "strong_win_rate"]].isna().all().all()


def test_context_chart_contracts(context_tables: AnalysisTables) -> None:
    from ligat.viz.context import chapter_e_chart

    for kind, fn in [
        ("referee_home", stats.referee_stats),
        ("referee_events", stats.referee_stats),
        ("venues", stats.venue_summary),
        ("coaches", stats.coach_summary),
        ("borrowed", stats.venue_effects),
        ("game_types", stats.derby_and_big_games),
        ("upsets", stats.upset_rates),
    ]:
        figure = chapter_e_chart(kind, _run_e(fn, context_tables))
        assert figure.data and any(c.isdigit() for c in figure.layout.title.text)
        assert "?" not in figure.layout.title.text
        assert figure.to_json()
