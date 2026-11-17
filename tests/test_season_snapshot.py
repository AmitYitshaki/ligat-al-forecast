"""Morning history, hidden playoff fixtures, canonical keys and shared M0 truth."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.analysis.prepare import AnalysisTables
from ligat.features.elo import EloParams, replay_elo_season
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import PlayoffGroup, load_season_rules, sporting_table
from ligat.simulate.snapshot import season_snapshots, snapshot_as_of


def snapshot_inputs():
    records = []
    fixtures = [
        (1, 1, 2, 1, "regular", 1, 0),
        (2, 3, 4, 1, "regular", 0, 0),
        (3, 1, 3, 5, "regular", 2, 0),
        (4, 2, 4, 10, "regular", 1, 1),  # Postponed past the split.
        (5, 1, 2, 8, "upper", 0, 0),
        (6, 3, 4, 8, "lower", 1, 0),
    ]
    for game, home, away, day, stage, hg, ag in fixtures:
        for team, opponent, gf, ga, is_home in [
            (home, away, hg, ag, True),
            (away, home, ag, hg, False),
        ]:
            date = pd.Timestamp(f"2006-08-{day:02d} 19:00", tz="Asia/Jerusalem")
            records.append(
                dict(
                    game_id=game,
                    team_id=team,
                    opponent_id=opponent,
                    season_id=8,
                    date=date,
                    as_of=date.normalize(),
                    round=game,
                    stage=stage,
                    is_home=is_home,
                    goals_scored=gf,
                    goals_conceded=ga,
                )
            )
    rows = pd.DataFrame(records)
    elo = rows[["game_id", "team_id", "season_id", "as_of"]].copy()
    elo["elo_pre"] = 1500.0 + 10 * elo.game_id + elo.team_id
    elo = elo.set_index(["game_id", "team_id"])
    rule = replace(
        load_season_rules()[8],
        n_teams=4,
        regular_rounds=3,
        groups=(PlayoffGroup("upper", 2, 1), PlayoffGroup("lower", 2, 1)),
        points_halved=True,
        relegated=1,
    )
    return rows, elo, {8: rule}


def morning(day):
    return pd.Timestamp(f"2006-08-{day:02d}", tz="Asia/Jerusalem")


def test_same_day_is_unplayed_and_future_results_do_not_change_snapshot():
    rows, elo, rules = snapshot_inputs()
    original = rows.copy(deep=True)
    snap = snapshot_as_of(rows, elo, rules, 8, morning(1))
    assert snap.played.empty
    assert snap.table.games_played.eq(0).all()
    assert {1, 2} <= set(snap.remaining.game_id)
    assert "goals_scored" not in snap.remaining
    changed = rows.copy()
    changed.loc[changed.as_of.ge(morning(5)), ["goals_scored", "goals_conceded"]] = 99
    other = snapshot_as_of(changed, elo, rules, 8, morning(5))
    before = snapshot_as_of(rows, elo, rules, 8, morning(5))
    assert_frame_equal(before.table, other.table)
    assert_frame_equal(before.remaining, other.remaining)
    assert_frame_equal(rows, original)


def test_postponed_regular_game_survives_split_and_playoffs_stay_hidden_before_it():
    rows, elo, rules = snapshot_inputs()
    early = snapshot_as_of(rows, elo, rules, 8, morning(5))
    split = snapshot_as_of(rows, elo, rules, 8, morning(8))
    assert early.phase == "regular" and early.groups is None
    assert set(early.remaining.game_id) == {3, 4}
    assert early.remaining.stage.eq("regular").all()
    assert split.phase == "split"
    assert set(split.remaining.game_id) == {4, 5, 6}
    assert split.groups == historical_groups(rows)
    expected = sporting_table(split.played, [1, 2, 3, 4], rules[8], groups=split.groups)
    assert_frame_equal(split.table, expected)
    assert split.table.loc[1, "points"] == 3  # Six regular points, halved.
    assert split.table.loc[3, "position"] > 2


@pytest.mark.parametrize("day", [1, 5, 8, 10, 11])
def test_visible_schedule_partition_and_finished_rating(day):
    rows, elo, rules = snapshot_inputs()
    snap = snapshot_as_of(rows, elo, rules, 8, morning(day))
    played, remaining = set(snap.played.game_id), set(snap.remaining.game_id)
    assert played.isdisjoint(remaining)
    hidden = set(rows.loc[rows.stage.ne("regular"), "game_id"]) if day < 8 else set()
    assert played | remaining == set(rows.game_id) - hidden
    if day == 11:
        assert snap.elo.isna().all()


def test_next_rating_includes_club_waiting_for_playoffs():
    rows, elo, rules = snapshot_inputs()
    snap = snapshot_as_of(rows, elo, rules, 8, morning(6))
    # Club 1 has no regular fixture left, but its playoff opening Elo is known.
    assert 1 not in set(snap.remaining.home_team_id) | set(snap.remaining.away_team_id)
    assert snap.elo.loc[1] == elo.loc[(5, 1), "elo_pre"]
    assert snap.elo.loc[4] == elo.loc[(6, 4), "elo_pre"]


@pytest.mark.parametrize(
    "day",
    [
        pd.Timestamp("2006-08-01"),
        pd.Timestamp("2006-08-01 01:00", tz="Asia/Jerusalem"),
        pd.Timestamp("2006-08-01", tz="UTC"),
    ],
)
def test_non_morning_or_wrong_timezone_is_rejected(day):
    rows, elo, rules = snapshot_inputs()
    with pytest.raises(ValueError, match="normalized midnight"):
        snapshot_as_of(rows, elo, rules, 8, day)


def test_nullable_identifiers_and_generator_are_canonical_and_independent():
    rows, elo, rules = snapshot_inputs()
    for column in ("game_id", "team_id", "opponent_id"):
        rows[column] = rows[column].astype("Int64")
    frames = list(season_snapshots(rows, elo, rules, 8))
    assert len(frames) == 4
    for snap in frames:
        assert snap.previous_positions == {}
        for col in ("game_id", "team_id", "opponent_id"):
            assert snap.played[col].dtype == np.dtype("int64")
        for col in ("game_id", "home_team_id", "away_team_id"):
            assert snap.remaining[col].dtype == np.dtype("int64")
        assert snap.table.index.dtype == snap.elo.index.dtype == np.dtype("int64")
    frames[0].table.loc[:, "points"] = 999
    assert frames[1].table.points.max() < 999


def test_previous_positions_only_contain_current_incumbents():
    rows, elo, rules = snapshot_inputs()
    following = rows.copy()
    following["season_id"] = 9
    following["game_id"] += 100
    following["date"] += pd.DateOffset(years=1)
    following["as_of"] = following.date.dt.normalize()
    following["team_id"] = following.team_id.replace({4: 5})
    following["opponent_id"] = following.opponent_id.replace({4: 5})
    next_elo = (
        following[["game_id", "team_id", "season_id", "as_of"]]
        .assign(elo_pre=1500.0)
        .set_index(["game_id", "team_id"])
    )
    all_rows = pd.concat([rows, following], ignore_index=True)
    rules[9] = replace(rules[8], season_id=9)
    snap = next(season_snapshots(all_rows, pd.concat([elo, next_elo]), rules, 9))
    expected = sporting_table(rows, [1, 2, 3, 4], rules[8], groups=historical_groups(rows))
    assert snap.previous_positions == expected.loc[[1, 2, 3], "position"].to_dict()
    assert 5 not in snap.previous_positions


@pytest.fixture(scope="module")
def development_snapshots(real_tables: AnalysisTables):
    rows = game_rows(real_tables)
    elo = pd.read_parquet(
        "data/derived/elo/elo_E0050_prematch.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    rules = load_season_rules()
    return {sid: list(season_snapshots(rows, elo, rules, sid)) for sid in range(8, 26)}


def test_every_development_morning_matches_frozen_table_exactly(development_snapshots):
    expected = pd.read_parquet(
        "data/derived/prematch/table_state_v2.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    columns = ["points", "position", "goal_difference", "games_played"]
    count = 0
    for sid, snapshots in development_snapshots.items():
        season = expected.loc[expected.season_id.eq(sid)]
        for snap in snapshots:
            targets = season.loc[season.as_of.eq(snap.as_of)].reset_index().set_index("team_id")
            actual = snap.table.reindex(targets.index).loc[:, columns]
            assert_frame_equal(actual, targets.loc[:, columns], check_exact=True)
            count += len(targets)
    assert count == 8630


def test_replayed_elo_matches_all_clubs_in_two_formats(real_tables, development_snapshots):
    params = EloParams(k=10, home_advantage=30, season_regression=0.33)
    # Parameters affecting season openings are immaterial to replay from known openings.
    config = "elo-b24989debf11c0ba4af09aa247bd264ad754349b8d0db4ba88e9c440fd00e950"
    openings = pd.read_parquet(
        "data/derived/elo/exp001_combined_snapshots.parquet",
        filters=[("season_id", "in", [8, 25]), ("config_id", "==", config)],
    )
    frozen = pd.read_parquet(
        "data/derived/prematch/table_state_v2.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    for sid in (8, 25):
        initial = openings.loc[openings.season_id.eq(sid)].set_index("team_id").elo_start
        replayed = replay_elo_season(
            real_tables, frozen.nothing_to_play, season_id=sid, elo_start=initial, params=params
        ).prematch.reset_index()
        for snap in development_snapshots[sid]:
            future = replayed.loc[replayed.as_of.ge(snap.as_of)].sort_values(["as_of", "game_id"])
            independent = (
                future.drop_duplicates("team_id")
                .set_index("team_id")
                .elo_pre.reindex(snap.elo.index)
            )
            assert_series_equal(snap.elo, independent, check_exact=False, atol=1e-9, rtol=0)


def test_real_playoff_schedule_only_appears_on_the_split_day(real_tables, development_snapshots):
    rows = game_rows(real_tables)
    elo = pd.read_parquet(
        "data/derived/elo/elo_E0050_prematch.parquet",
        filters=[("season_id", "in", list(range(8, 26)))],
    )
    rules = load_season_rules()
    for sid, snapshots in development_snapshots.items():
        split = next((snap for snap in snapshots if snap.phase == "split"), None)
        if split is None:
            continue
        playoff_ids = set(rows.loc[rows.season_id.eq(sid) & rows.stage.ne("regular"), "game_id"])
        assert playoff_ids <= set(split.remaining.game_id)
        before = snapshot_as_of(rows, elo, rules, sid, split.as_of - pd.DateOffset(days=1))
        assert playoff_ids.isdisjoint(before.remaining.game_id)
        assert before.groups is None
