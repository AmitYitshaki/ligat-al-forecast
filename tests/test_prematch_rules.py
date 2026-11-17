"""Season rules, mathematical proofs and independently checkable sporting ties."""

from dataclasses import replace
from itertools import product

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.simulate.rules import (
    PlayoffGroup,
    halve_points,
    load_season_rules,
    objective_flags,
    rank_table,
    split_playoffs,
    sporting_table,
)


@pytest.mark.parametrize("points,expected", [(77, 39), (71, 36), (65, 33), (0, 0), (10, 5)])
def test_carry_rounds_up(points: int, expected: int) -> None:
    assert halve_points(points) == expected


def test_era_and_format_configuration() -> None:
    rules = load_season_rules()
    assert [sid for sid, rule in rules.items() if rule.points_halved] == [11, 12]
    assert rules[13].relegated == 3
    assert rules[11].last_safe_position == rules[12].last_safe_position == 13
    assert rules[8].groups == ()
    assert [(g.teams, g.rounds) for g in rules[11].groups] == [(6, 5), (4, 3), (6, 5)]
    assert [(g.teams, g.rounds) for g in rules[13].groups] == [(8, 7), (8, 7)]
    assert [(g.teams, g.rounds) for g in rules[14].groups] == [(6, 10), (8, 7)]


@pytest.mark.parametrize("season,winner", [(8, 1), (9, 1), (10, 2), (25, 2)])
def test_wins_not_used_in_early_era(season: int, winner: int) -> None:
    table = pd.DataFrame(
        {"points": [30, 30], "goal_difference": [5, 5], "wins": [5, 9], "goals_for": [30, 20]},
        index=[1, 2],
    )
    assert rank_table(table, load_season_rules()[season]).index[0] == winner


def test_head_to_head_and_average_ties() -> None:
    rules = load_season_rules()[25]
    table = pd.DataFrame(
        {"points": [30, 30], "goal_difference": [5, 5], "wins": [8, 8], "goals_for": [30, 30]},
        index=[1, 2],
    )
    assert rank_table(table, rules).position.tolist() == [1.5, 1.5]
    games = pd.DataFrame(
        {"team_id": [1, 2], "opponent_id": [2, 1], "goals_scored": [0, 1], "goals_conceded": [1, 0]}
    )
    assert rank_table(table, rules, games).index.tolist() == [2, 1]
    # Earlier era stops at goals_for; no head-to-head criterion is added silently.
    assert rank_table(table, load_season_rules()[9], games).position.tolist() == [1.5, 1.5]


@pytest.mark.parametrize("season", [8, 11, 13, 14])
def test_split_sizes_and_offsets(season: int) -> None:
    rules = load_season_rules()[season]
    table = pd.DataFrame(
        {
            "points": list(range(rules.n_teams, 0, -1)),
            "goal_difference": 0,
            "wins": 0,
            "goals_for": 0,
        },
        index=range(1, rules.n_teams + 1),
    )
    assignments = split_playoffs(table, rules)
    if not rules.groups:
        assert set(assignments.values()) == {"regular"}
    for group in rules.groups:
        assert list(assignments.values()).count(group.name) == group.teams
        assert assignments[rules.offset(group.name) + 1] == group.name


def test_split_requires_previous_rank_for_unresolved_boundary() -> None:
    rules = load_season_rules()[14]
    table = pd.DataFrame(
        {"points": 0, "goal_difference": 0, "wins": 0, "goals_for": 0}, index=range(1, 15)
    )
    with pytest.raises(ValueError, match="Unresolved tie"):
        split_playoffs(table, rules)
    groups = split_playoffs(
        table, rules, previous_positions={team: float(15 - team) for team in range(1, 15)}
    )
    assert {team for team, stage in groups.items() if stage == "upper"} == set(range(9, 15))


def test_champion_doomed_safe_and_point_tie_open() -> None:
    rules = replace(load_season_rules()[8], n_teams=4, relegated=1)
    table = pd.DataFrame({"points": [20, 10, 8, 0]}, index=[1, 2, 3, 4])
    flags = objective_flags(table, rules, "regular", pd.Series(1, index=table.index))
    assert flags.title_decided.all()
    assert flags.relegation_decided.all()
    assert flags.loc[[1, 4], "nothing_to_play"].all()
    # The title and safety are settled, but teams 2/3 can still cross the top-2 line.
    assert not flags.loc[[2, 3], "places_decided"].any()
    assert not flags.loc[[2, 3], "nothing_to_play"].any()
    table.loc[1, "points"] = 13
    flags = objective_flags(table, rules, "regular", pd.Series(1, index=table.index))
    assert not flags.loc[1, "title_decided"]
    assert not flags.loc[2, "title_decided"]


def test_playoff_structural_targets_and_halved_bounds() -> None:
    rule = load_season_rules()[11]
    table = pd.DataFrame({"points": [30, 30, 30, 30]}, index=[1, 2, 3, 4])
    flags = objective_flags(table, rule, "middle", pd.Series(3, index=table.index))
    assert flags.nothing_to_play.all()
    assert flags.places_decided.all()
    upper = objective_flags(table, rule, "upper", pd.Series(5, index=table.index))
    assert upper.relegation_decided.all() and upper.cut_decided.all()
    lower = objective_flags(table, rule, "lower", pd.Series(5, index=table.index))
    assert lower.title_decided.all() and lower.cut_decided.all()
    assert lower.places_decided.all()
    small = replace(
        rule, n_teams=4, groups=(PlayoffGroup("upper", 2, 1), PlayoffGroup("lower", 2, 1))
    )
    table = pd.DataFrame({"points": [12, 5, 4, 0]}, index=[1, 2, 3, 4])
    # ceil(5/2) == ceil(6/2): a half-point effect must not close the cut.
    flags = objective_flags(table, small, "regular", pd.Series([0, 0, 0, 2], index=table.index))
    assert not flags.loc[2, "cut_decided"]


def test_no_false_positive_against_exhaustive_independent_point_envelopes() -> None:
    rule = replace(load_season_rules()[8], n_teams=3, relegated=1)
    for points in product([0, 2, 6], repeat=3):
        table = pd.DataFrame({"points": points}, index=[1, 2, 3])
        flags = objective_flags(table, rule, "regular", pd.Series(1, index=table.index))
        for team in table.index:
            for column, places in [
                ("title_decided", 1),
                ("relegation_decided", 2),
                *(("places_decided", k) for k in (2, 3, 4)),
            ]:
                if not flags.loc[team, column]:
                    continue
                states = set()
                for extra in product([0, 1, 3], repeat=3):
                    end = [points[i] + extra[i] for i in range(3)]
                    mine = end[team - 1]
                    others = [value for i, value in enumerate(end) if i != team - 1]
                    if sum(value > mine for value in others) >= places:
                        states.add(False)
                    elif sum(value >= mine for value in others) < places:
                        states.add(True)
                    else:
                        states.update([True, False])
                assert len(states) == 1, (points, team, column)


def test_upper_title_settled_but_second_and_third_places_open() -> None:
    table = pd.DataFrame({"points": [80, 45, 44, 20, 10, 0]}, index=range(1, 7))
    flags = objective_flags(
        table, load_season_rules()[14], "upper", pd.Series(1, index=table.index)
    )
    assert flags.title_decided.all()
    assert not flags.loc[[2, 3], "places_decided"].any()
    assert not flags.loc[[2, 3], "nothing_to_play"].any()


def test_upper_all_place_lines_locked() -> None:
    table = pd.DataFrame({"points": [90, 70, 50, 30, 10, 0]}, index=range(1, 7))
    flags = objective_flags(
        table, load_season_rules()[14], "upper", pd.Series(1, index=table.index)
    )
    assert flags.places_decided.all()
    assert flags.nothing_to_play.all()


def test_possible_point_tie_at_third_place_stays_open() -> None:
    table = pd.DataFrame({"points": [90, 70, 40, 37, 10, 0]}, index=range(1, 7))
    flags = objective_flags(
        table, load_season_rules()[14], "upper", pd.Series(1, index=table.index)
    )
    # Fourth can reach 40, exactly third's minimum: no goal-difference shortcut.
    assert not flags.loc[[3, 4], "places_decided"].any()
    assert not flags.loc[[3, 4], "nothing_to_play"].any()


@pytest.mark.parametrize("season", [8, 9, 10])
def test_place_lines_apply_to_regular_seasons_without_playoffs(season: int) -> None:
    points = [120, 100, 80, 77, 60, 50, 40, 30, 20, 10, 5, 0]
    table = pd.DataFrame({"points": points}, index=range(1, 13))
    flags = objective_flags(
        table, load_season_rules()[season], "regular", pd.Series(1, index=table.index)
    )
    assert flags.title_decided.all()
    assert not flags.loc[[3, 4], "places_decided"].any()
    assert not flags.loc[[3, 4], "nothing_to_play"].any()


@pytest.mark.parametrize("season", [11, 13, 14])
def test_before_split_only_exclusion_from_upper_settles_place_lines(season: int) -> None:
    rule = load_season_rules()[season]
    upper = int(rule.upper_size or 0)
    table = pd.DataFrame(
        {"points": [100] * upper + [10] * (rule.n_teams - upper)}, index=range(1, rule.n_teams + 1)
    )
    flags = objective_flags(table, rule, "regular", pd.Series(1, index=table.index))
    assert flags.places_decided.equals(flags.title_decided)
    assert not flags.iloc[:upper].places_decided.any()
    assert flags.iloc[upper:].places_decided.all()


@pytest.fixture
def a1_boundary_table() -> pd.DataFrame:
    """Teams 90 and 20 share sixth place after every sporting criterion."""
    return pd.DataFrame(
        {
            "points": [60, 50, 40, 30, 20, 10, 10, 0],
            "goal_difference": 0,
            "wins": 0,
            "goals_for": 0,
        },
        index=[1, 2, 3, 4, 5, 90, 20, 8],
    )


@pytest.mark.parametrize("promoted_previous", [None, float("nan")], ids=["missing", "nan"])
def test_a1_veteran_precedes_promoted_at_split(
    a1_boundary_table: pd.DataFrame, promoted_previous: float | None
) -> None:
    rules = replace(
        load_season_rules()[14],
        n_teams=8,
        groups=(PlayoffGroup("upper", 6, 10), PlayoffGroup("lower", 2, 1)),
    )
    previous = {90: 8.0}
    if promoted_previous is not None:
        previous[20] = promoted_previous
    ranked = rank_table(a1_boundary_table, rules, previous_positions=previous)
    assert ranked.loc[[90, 20], "position"].tolist() == [6.0, 7.0]
    groups = split_playoffs(a1_boundary_table, rules, previous_positions=previous)
    assert groups[90] == "upper" and groups[20] == "lower"


def test_a1_two_promoted_need_lottery_at_split(a1_boundary_table: pd.DataFrame) -> None:
    rules = replace(
        load_season_rules()[14],
        n_teams=8,
        groups=(PlayoffGroup("upper", 6, 10), PlayoffGroup("lower", 2, 1)),
    )
    previous = {90: float("nan")}
    with pytest.raises(ValueError, match="Unresolved tie"):
        split_playoffs(a1_boundary_table, rules, previous_positions=previous)
    keys = {int(team): float(i) for i, team in enumerate(a1_boundary_table.index)}
    groups = split_playoffs(
        a1_boundary_table, rules, previous_positions=previous, lottery_keys=keys
    )
    assert groups[90] == "upper" and groups[20] == "lower"
    keys[90], keys[20] = keys[20], keys[90]
    swapped = split_playoffs(
        a1_boundary_table, rules, previous_positions=previous, lottery_keys=keys
    )
    assert swapped[20] == "upper" and swapped[90] == "lower"


def test_a1_veteran_then_two_promoted_form_separate_tie_groups() -> None:
    table = pd.DataFrame(
        {"points": 0, "goal_difference": 0, "wins": 0, "goals_for": 0}, index=[90, 30, 20]
    )
    rules = load_season_rules()[14]
    previous = {90: 8.0, 30: float("nan")}
    partial = rank_table(table, rules, previous_positions=previous)
    assert partial.loc[[90, 30, 20], "position"].tolist() == [1.0, 2.5, 2.5]
    ranked = rank_table(
        table, rules, previous_positions=previous, lottery_keys={90: 0.9, 30: 0.1, 20: 0.2}
    )
    assert ranked.index.tolist() == [90, 30, 20]
    assert ranked.position.tolist() == [1.0, 2.0, 3.0]


def test_a1_head_to_head_precedes_previous_position_and_lottery() -> None:
    table = pd.DataFrame(
        {"points": 30, "goal_difference": 5, "wins": 8, "goals_for": 30}, index=[1, 2]
    )
    games = pd.DataFrame(
        {"team_id": [1, 2], "opponent_id": [2, 1], "goals_scored": [0, 1], "goals_conceded": [1, 0]}
    )
    ranked = rank_table(
        table,
        load_season_rules()[25],
        games,
        previous_positions={1: 1.0, 2: 9.0},
        lottery_keys={1: 0.1, 2: 0.9},
    )
    assert ranked.index.tolist() == [2, 1]


def test_a1_lottery_is_deterministic_and_does_not_mutate_inputs() -> None:
    table = pd.DataFrame(
        {"points": 0, "goal_difference": 0, "wins": 0, "goals_for": 0}, index=[1, 2, 3]
    )
    before = table.copy(deep=True)
    keys = {1: 0.8, 2: 0.5, 3: 0.1}
    first = rank_table(table, load_season_rules()[8], lottery_keys=keys)
    second = rank_table(table, load_season_rules()[8], lottery_keys=keys)
    assert first.index.tolist() == [3, 2, 1]
    assert_frame_equal(first, second, check_exact=True)
    assert_frame_equal(table, before, check_exact=True)
    assert keys == {1: 0.8, 2: 0.5, 3: 0.1}


@pytest.mark.parametrize("grouped", [False, True], ids=["overall", "playoff_groups"])
def test_a1_sporting_table_resolves_title_and_safety_ties(grouped: bool) -> None:
    rules = replace(
        load_season_rules()[14],
        n_teams=4,
        relegated=1,
        groups=(PlayoffGroup("upper", 2, 2), PlayoffGroup("lower", 2, 1)),
    )
    # Each pair's drawn match leaves a full tie in both playoff groups.
    games = pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4],
            "opponent_id": [2, 1, 4, 3],
            "goals_scored": 0,
            "goals_conceded": 0,
            "stage": ["upper", "upper", "lower", "lower"],
        }
    )
    groups = {1: "upper", 2: "upper", 3: "lower", 4: "lower"} if grouped else None
    baseline = sporting_table(games, [1, 2, 3, 4], rules, groups=groups)
    assert baseline.position.tolist() == ([1.5, 1.5, 3.5, 3.5] if grouped else [2.5] * 4)
    result = sporting_table(
        games, [1, 2, 3, 4], rules, groups=groups, lottery_keys={1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    )
    assert result.position.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result.position.eq(1).sum() == 1
    assert result.position.le(rules.last_safe_position).sum() == rules.last_safe_position
    assert result.index.tolist() == ([2, 1, 4, 3] if grouped else [4, 3, 2, 1])
    assert list(result.columns) == list(baseline.columns)


@pytest.mark.parametrize(
    "keys",
    [{1: 0.1}, {1: 0.1, 2: float("nan")}, {1: float("inf"), 2: 0.2}, {1: 0.3, 2: 0.3}],
    ids=["missing", "nan", "infinite", "duplicate_tied"],
)
def test_a1_invalid_lottery_keys_fail(keys: dict[int, float]) -> None:
    table = pd.DataFrame(
        {"points": 0, "goal_difference": 0, "wins": 0, "goals_for": 0}, index=[1, 2]
    )
    with pytest.raises(ValueError, match="lottery"):
        rank_table(table, load_season_rules()[25], lottery_keys=keys)


def test_a1_missing_key_fails_even_for_an_untied_team() -> None:
    table = pd.DataFrame(
        {"points": [3, 0], "goal_difference": 0, "wins": 0, "goals_for": 0}, index=[1, 2]
    )
    with pytest.raises(ValueError, match="lottery"):
        rank_table(table, load_season_rules()[25], lottery_keys={1: 0.1})


def test_a1_equal_keys_do_not_override_resolved_sporting_order() -> None:
    table = pd.DataFrame(
        {"points": [3, 0], "goal_difference": 0, "wins": 0, "goals_for": 0}, index=[1, 2]
    )
    result = rank_table(table, load_season_rules()[25], lottery_keys={1: 0.1, 2: 0.1})
    assert result.index.tolist() == [1, 2]
