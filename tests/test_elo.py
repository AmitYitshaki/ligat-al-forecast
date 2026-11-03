"""Elo arithmetic, calendar causality, season initialization and exact replay."""

from dataclasses import replace
from math import isclose

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.analysis.prepare import AnalysisTables
from ligat.features.base import FeatureBlock
from ligat.features.elo import (
    EloBlock,
    EloParams,
    compute_elo,
    expected_home_score,
    goal_difference_multiplier,
    replay_elo_season,
)
from ligat.features.registry import FEATURES, validate_feature_columns


def tables_from(
    fixtures: list[tuple[int, int, int, int, str, int, int]],
) -> AnalysisTables:
    rows = []
    for season, game, home, away, date, gf, ga in fixtures:
        for team, opponent, scored, conceded, is_home in (
            (home, away, gf, ga, True),
            (away, home, ga, gf, False),
        ):
            rows.append(
                dict(
                    game_id=game,
                    team_id=team,
                    opponent_id=opponent,
                    season_id=season,
                    date=pd.Timestamp(date, tz="Asia/Jerusalem"),
                    round=1,
                    stage="regular",
                    is_home=is_home,
                    goals_scored=scored,
                    goals_conceded=conceded,
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


def flags_for(tables: AnalysisTables) -> pd.Series:
    index = tables.team_matches.reset_index().set_index(["game_id", "team_id"]).index
    return pd.Series(False, index=index, name="nothing_to_play")


def fixtures() -> list[tuple[int, int, int, int, str, int, int]]:
    return [
        (8, 1, 1, 2, "2006-08-01 00:05", 1, 0),
        (8, 2, 3, 4, "2006-08-01 23:00", 0, 2),
        (8, 3, 1, 3, "2006-08-08 14:00", 1, 1),
        (8, 4, 2, 4, "2006-08-08 20:00", 0, 1),
        (8, 5, 1, 4, "2006-08-15 14:00", 0, 0),
        (8, 6, 2, 3, "2006-08-15 20:00", 2, 0),
    ]


def seasons() -> AnalysisTables:
    return tables_from(
        [
            (8, 1, 1, 2, "2006-08-01", 2, 0),
            (8, 2, 3, 4, "2006-08-02", 1, 0),
            (9, 3, 1, 5, "2007-08-01", 3, 0),
            (9, 4, 2, 6, "2007-08-02", 2, 0),
            (10, 5, 1, 7, "2008-08-01", 2, 0),
            (10, 6, 2, 8, "2008-08-02", 3, 0),
            (11, 7, 9, 1, "2009-08-01", 1, 0),
            (11, 8, 2, 10, "2009-08-02", 1, 1),
            (12, 9, 1, 3, "2010-08-01", 1, 0),
            (12, 10, 2, 4, "2010-08-02", 0, 1),
        ]
    )


def test_manual_home_win_and_perspective() -> None:
    tables = tables_from(fixtures()[:1])
    result = compute_elo(tables, flags_for(tables))
    expected = 1 / (1 + 10 ** (-50 / 400))
    delta = 20 * (1 - expected)
    assert expected == pytest.approx(0.5714631174)
    assert delta == pytest.approx(8.57073765)
    assert result.prematch.loc[(1, 1), "elo_expected"] == pytest.approx(expected)
    assert result.prematch.loc[(1, 2), "elo_expected"] == pytest.approx(1 - expected)
    assert result.prematch.elo_pre.eq(1500).all()
    assert result.prematch.elo_diff.eq(0).all()
    assert result.season_snapshots.loc[(8, 1), "elo_end"] == pytest.approx(1500 + delta)
    assert result.season_snapshots.loc[(8, 2), "elo_end"] == pytest.approx(1500 - delta)
    assert result.season_snapshots.elo_start.eq(1500).all()


@pytest.mark.parametrize("difference,multiplier", [(0, 1), (1, 1), (2, 1.5), (3, 1.75), (5, 2)])
def test_goal_margin_formula_and_switch(difference: int, multiplier: float) -> None:
    assert goal_difference_multiplier(difference) == multiplier
    assert goal_difference_multiplier(-difference) == multiplier
    assert goal_difference_multiplier(difference, False) == 1
    tables = tables_from([(8, 1, 1, 2, "2006-08-01", difference, 0)])
    flags = flags_for(tables)
    enabled = compute_elo(tables, flags)
    disabled = compute_elo(tables, flags, EloParams(use_goal_multiplier=False))
    assert enabled.season_snapshots.loc[(8, 1), "elo_end"] - 1500 == pytest.approx(
        multiplier * (disabled.season_snapshots.loc[(8, 1), "elo_end"] - 1500)
    )


def test_result_change_preserves_present_and_past_but_changes_both_next_fixtures() -> None:
    original = fixtures()
    before_tables = tables_from(original)
    original[0] = (*original[0][:5], 0, 2)
    after_tables = tables_from(original)
    before = compute_elo(before_tables, flags_for(before_tables)).prematch
    after = compute_elo(after_tables, flags_for(after_tables)).prematch
    assert_frame_equal(before.loc[[1, 2]], after.loc[[1, 2]], check_exact=True)
    assert before.loc[(3, 1), "elo_pre"] != after.loc[(3, 1), "elo_pre"]
    assert before.loc[(4, 2), "elo_pre"] != after.loc[(4, 2), "elo_pre"]


def test_same_calendar_day_with_shared_team_is_batched_even_across_utc_dates() -> None:
    games = [
        (8, 1, 1, 2, "2006-08-01 00:05", 1, 0),
        (8, 2, 1, 3, "2006-08-01 23:00", 2, 0),
        (8, 3, 1, 2, "2006-08-02 20:00", 0, 0),
    ]
    tables = tables_from(games)
    before = compute_elo(tables, flags_for(tables))
    assert tables.team_matches.loc[(1, 1), "date"].tz_convert("UTC").day == 31
    assert before.prematch.loc[[1, 2], "elo_pre"].eq(1500).all()
    games[0] = (*games[0][:5], 0, 5)
    changed = tables_from(games)
    after = compute_elo(changed, flags_for(changed))
    assert_frame_equal(before.prematch.loc[[1, 2]], after.prematch.loc[[1, 2]], check_exact=True)
    assert before.prematch.loc[(3, 1), "elo_pre"] != after.prematch.loc[(3, 1), "elo_pre"]
    expected = 1500 + 20 * (1 + 1.5) * (1 - 1 / (1 + 10 ** (-50 / 400)))
    assert before.prematch.loc[(3, 1), "elo_pre"] == pytest.approx(expected)


@pytest.mark.parametrize("discount", [0, 0.25, 1])
def test_rating_sum_is_preserved_after_every_game(discount: float) -> None:
    for n in range(1, len(fixtures()) + 1):
        tables = tables_from(fixtures()[:n])
        flags = flags_for(tables)
        flags.iloc[::3] = True
        result = compute_elo(tables, flags, EloParams(dead_game_discount=discount))
        assert result.season_snapshots.elo_end.sum() == pytest.approx(
            result.season_snapshots.elo_start.sum(), abs=1e-9
        )


@pytest.mark.parametrize("team", [1, 2])
def test_one_dead_team_is_enough_for_symmetric_discount(team: int) -> None:
    tables = tables_from(fixtures()[:1])
    flags = flags_for(tables)
    baseline = compute_elo(tables, flags).season_snapshots.elo_end
    flags.loc[(1, team)] = True
    stopped = compute_elo(tables, flags, EloParams(dead_game_discount=0))
    assert stopped.season_snapshots.elo_end.eq(1500).all()
    discounted = compute_elo(tables, flags, EloParams(dead_game_discount=0.25))
    assert_series_equal(discounted.season_snapshots.elo_end, 1500 + (baseline - 1500) * 0.25)
    unchanged = compute_elo(tables, flags, EloParams(dead_game_discount=1))
    assert_series_equal(unchanged.season_snapshots.elo_end, baseline)


def test_regression_newcomer_threshold_and_returning_clubs() -> None:
    tables = seasons()
    snapshots = compute_elo(tables, flags_for(tables)).season_snapshots
    assert snapshots.loc[8, "elo_start"].eq(1500).all()
    assert snapshots.loc[(9, 1), "elo_start"] == pytest.approx(
        snapshots.loc[(8, 1), "elo_end"] + 0.33 * (1500 - snapshots.loc[(8, 1), "elo_end"])
    )
    for key in [(9, 5), (9, 6), (10, 7), (10, 8)]:
        assert snapshots.loc[key, "elo_start"] == 1450
    earlier = snapshots.loc[[(9, 5), (9, 6), (10, 7), (10, 8)], "elo_end"].mean()
    assert earlier != pytest.approx(1450)
    assert snapshots.loc[(11, 9), "elo_start"] == pytest.approx(earlier)
    assert snapshots.loc[(11, 10), "elo_start"] == pytest.approx(earlier)
    pool = snapshots.loc[[(9, 5), (9, 6), (10, 7), (10, 8), (11, 9), (11, 10)], "elo_end"].mean()
    for team in [3, 4]:
        assert snapshots.loc[(12, team), "elo_start"] == pytest.approx(pool)
        assert snapshots.loc[(12, team), "elo_start"] != pytest.approx(
            snapshots.loc[(8, team), "elo_end"]
        )
    fixed = compute_elo(tables, flags_for(tables), EloParams(promoted_initialization="fixed_1450"))
    for key in [(9, 5), (9, 6), (10, 7), (10, 8), (11, 9), (11, 10), (12, 3), (12, 4)]:
        assert fixed.season_snapshots.loc[key, "elo_start"] == 1450


@pytest.mark.parametrize("discount", [0.5, 1])
def test_replay_is_exact_for_every_season_and_promoted_history(discount: float) -> None:
    tables = seasons()
    flags = flags_for(tables)
    flags.iloc[::3] = True
    params = EloParams(dead_game_discount=discount)
    result = compute_elo(tables, flags, params)
    for season in range(8, 13):
        start = result.season_snapshots.loc[season, "elo_start"]
        one_season = replace(
            tables, team_matches=tables.team_matches.loc[tables.team_matches.season_id.eq(season)]
        )
        replay = replay_elo_season(
            one_season, flags, season_id=season, elo_start=start, params=params
        )
        assert_frame_equal(
            replay.prematch,
            result.prematch.loc[result.prematch.season_id.eq(season)],
            check_exact=True,
        )
        assert_frame_equal(
            replay.season_snapshots, result.season_snapshots.loc[[season]], check_exact=True
        )
        assert replay.config_id == result.config_id


def test_future_seasons_and_foreign_features_cannot_change_the_past() -> None:
    tables = seasons()
    before_rows = tables.team_matches.copy(deep=True)
    flags = flags_for(tables)
    before_flags = flags.copy(deep=True)
    full = compute_elo(tables, flags)
    truncated = replace(
        tables, team_matches=tables.team_matches.loc[tables.team_matches.season_id.le(10)]
    )
    early = compute_elo(truncated, flags)
    assert_frame_equal(
        early.prematch, full.prematch.loc[full.prematch.season_id.le(10)], check_exact=True
    )
    poisoned = replace(
        tables,
        team_matches=tables.team_matches.assign(points=9999, elo=0, referee="ignored"),
        team_seasons=pd.DataFrame({"value": [99999]}),
        match_table=pd.DataFrame({"table": [99999]}),
    )
    assert_frame_equal(compute_elo(poisoned, flags).prematch, full.prematch, check_exact=True)
    shuffled = replace(tables, team_matches=tables.team_matches.sample(frac=1, random_state=7))
    assert_frame_equal(
        compute_elo(shuffled, flags.sample(frac=1, random_state=8)).prematch,
        full.prematch,
        check_exact=True,
    )
    assert_frame_equal(tables.team_matches, before_rows)
    assert_series_equal(flags, before_flags)


def test_registry_and_feature_block_contract() -> None:
    tables = tables_from(fixtures())
    flags = flags_for(tables)
    block: FeatureBlock = EloBlock(flags)
    frame = block.compute(tables)
    validate_feature_columns(frame, "elo")
    for column in ["elo_pre", "elo_opponent_pre", "elo_diff", "elo_expected"]:
        spec = FEATURES[column]
        assert (spec.block, spec.status, spec.availability, spec.models) == (
            "elo",
            "🔬",
            "D",
            ("M1", "M2", "M4"),
        )
        assert "ה-ת2" in spec.decision and "ה-ת11" in spec.decision
    for column in ["unknown_feature", "referee", "elo_end"]:
        with pytest.raises(ValueError, match="Unregistered"):
            validate_feature_columns(frame.assign(**{column: 1}), "elo")


def test_config_identity_covers_all_six_parameters() -> None:
    base = EloParams()
    assert EloParams(k=20).config_id == EloParams(k=20.0).config_id
    alternatives = [
        replace(base, k=30),
        replace(base, home_advantage=70),
        replace(base, season_regression=0.16),
        replace(base, use_goal_multiplier=False),
        replace(base, promoted_initialization="fixed_1450"),
        replace(base, dead_game_discount=0.5),
    ]
    assert len({p.config_id for p in [base, *alternatives]}) == 7


@pytest.mark.parametrize(
    "values",
    [
        {"k": -1},
        {"home_advantage": float("nan")},
        {"season_regression": 1.1},
        {"dead_game_discount": -0.1},
        {"promoted_initialization": "unknown"},
        {"use_goal_multiplier": 1},
    ],
)
def test_invalid_params_fail(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EloParams(**values)


@pytest.mark.parametrize("season", [26, 27, 28])
def test_heldout_input_is_rejected(season: int) -> None:
    tables = tables_from([(season, 1, 1, 2, "2024-08-01", 1, 0)])
    with pytest.raises(ValueError, match="development"):
        compute_elo(tables, flags_for(tables))


def test_invalid_flags_fail_instead_of_silently_dropping_rows() -> None:
    tables = tables_from(fixtures())
    flags = flags_for(tables)
    for bad in [
        flags.iloc[1:],
        flags.astype("boolean").mask(flags.index.get_level_values("game_id") == 1),
        flags.astype(int),
        pd.concat([flags, flags.iloc[:1]]),
    ]:
        with pytest.raises(ValueError, match="flag|Flags"):
            compute_elo(tables, bad)


def test_missing_history_and_bad_replay_snapshot_fail() -> None:
    tables = seasons()
    incomplete = replace(
        tables, team_matches=tables.team_matches.loc[tables.team_matches.season_id.ne(9)]
    )
    with pytest.raises(ValueError, match="without gaps"):
        compute_elo(incomplete, flags_for(tables))
    for start in [pd.Series({1: 1500}), pd.Series({1: 1500, 2: float("nan"), 3: 1500, 4: 1500})]:
        with pytest.raises(ValueError, match="snapshot|finite"):
            replay_elo_season(tables, flags_for(tables), season_id=8, elo_start=start)


def test_expected_score_is_numerically_stable_at_extreme_gaps() -> None:
    assert expected_home_score(1e6, 0, 50) == 1
    assert expected_home_score(0, 1e6, 50) == 0
    assert isclose(expected_home_score(1500, 1500, 0), 0.5)


@pytest.mark.parametrize("problem", ["mismatch", "negative", "missing", "no_away", "date"])
def test_bad_results_and_dates_fail(problem: str) -> None:
    tables = tables_from(fixtures())
    rows = tables.team_matches.copy()
    if problem == "mismatch":
        rows.loc[(1, 1), "goals_scored"] = 9
    elif problem == "negative":
        rows.loc[(1, 1), "goals_scored"] = -1
        rows.loc[(2, 1), "goals_conceded"] = -1
    elif problem == "missing":
        rows["goals_scored"] = rows.goals_scored.astype("Int64")
        rows.loc[(2, 1), "goals_scored"] = pd.NA
    elif problem == "no_away":
        rows["is_home"] = True
    else:
        rows.loc[rows.index.get_level_values("game_id") == 1, "date"] = pd.NaT
    with pytest.raises(ValueError):
        compute_elo(replace(tables, team_matches=rows), flags_for(tables))
