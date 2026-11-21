"""M2 goal attribution, explicit feature projection and frozen-source alignment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.evaluate.goal_inputs import GOAL_CONTEXT_COLUMNS, GoalData, prepare_goal_data
from ligat.features.pipeline import DEVELOPMENT_SEASONS
from ligat.features.prematch_v2 import validate_prematch_v2_frame
from ligat.features.registry import M2_BASE_COLUMNS


@pytest.fixture
def goal_sources():
    """Unequal season sizes, asymmetric goals, and two fixtures on the same local day."""
    games, rows = [], []
    for season in DEVELOPMENT_SEASONS:
        for number in range(2 + season % 3):
            game_id = season * 100 + number
            home, away = ((1, 2), (3, 4), (1, 3), (4, 2))[number]
            date = pd.Timestamp(1998 + season, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(
                days=max(0, number - 1), hours=10 + number
            )
            hg, ag = (3, 0, 1, 4)[number] + season % 2, (1, 2, 4, 0)[number]
            games.append(
                {
                    "game_id": game_id,
                    "season_id": season,
                    "date": date,
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_goals": hg,
                    "away_goals": ag,
                }
            )
            for team, opponent, is_home in ((home, away, True), (away, home, False)):
                rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team,
                        "season_id": season,
                        "date": date,
                        "as_of": date.normalize(),
                        "is_home": is_home,
                        "opponent_id": opponent,
                        "elo_diff": 100.0 if is_home else -100.0,
                        "goal_difference": 500 + number,
                        "goals_for": 999,
                        "points": 700,
                    }
                )
    return pd.DataFrame(rows).set_index(["game_id", "team_id"]), pd.DataFrame(games)


def prepare(sources, columns=("elo_diff",)):
    return prepare_goal_data(*sources, feature_columns=columns, seasons=DEVELOPMENT_SEASONS)


def test_goals_come_from_matches_and_only_requested_columns_enter_X(goal_sources):
    """A 3:1 fixture is attributed correctly despite unrelated cumulative goal/point values."""
    original = [f.copy(deep=True) for f in goal_sources]
    columns = ("points", "goal_difference", "goals_for", "elo_diff")
    data = prepare(goal_sources, columns)
    assert data.y.loc[(800, 1)] == 3 and data.y.loc[(800, 2)] == 1
    assert tuple(data.X.columns) == columns
    assert tuple(data.context.columns) == GOAL_CONTEXT_COLUMNS
    assert not {"game_id", "team_id", "goals_scored", "opponent_id"} & set(data.X)
    assert data.X.index.equals(data.y.index) and data.y.index.equals(data.context.index)
    assert data.X.points.eq(700).all() and data.X.goals_for.eq(999).all()
    for actual, before in zip(goal_sources, original, strict=True):
        assert_frame_equal(actual, before)


def test_explicit_F6_keys_and_empty_feature_list_are_supported(goal_sources):
    """team_id is available from the index only when requested; intercept-only models need no X."""
    data = prepare(goal_sources, ("team_id", "opponent_id"))
    assert tuple(data.X.columns) == ("team_id", "opponent_id")
    np.testing.assert_array_equal(data.X.team_id, data.X.index.get_level_values("team_id"))
    assert prepare(goal_sources, ()).X.shape == (len(data.y), 0)


def test_input_permutations_and_nullable_keys_preserve_the_exact_result(goal_sources):
    """Projection is key-based across row permutations and clean nullable integer storage."""
    expected = prepare(goal_sources)
    prematch, games = goal_sources
    games = (
        games.sample(frac=1, random_state=17)
        .astype({c: "Int64" for c in ("game_id", "season_id", "home_team_id", "away_team_id")})
        .set_index("game_id")
    )
    actual = prepare((prematch.sample(frac=1, random_state=19), games))
    assert_frame_equal(actual.X, expected.X)
    assert_frame_equal(actual.context, expected.context)
    assert_series_equal(actual.y, expected.y)


@pytest.mark.parametrize("columns", [("unknown",), ("elo_diff", "elo_diff"), ("goals_scored",)])
def test_bad_or_target_feature_requests_fail(goal_sources, columns):
    """Missing, duplicate and injected postmatch columns cannot become model inputs."""
    prematch, games = goal_sources
    if columns == ("goals_scored",):
        prematch = prematch.assign(goals_scored=999)
    with pytest.raises(ValueError, match="missing|distinct|never targets"):
        prepare((prematch, games), columns)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_prematch",
        "duplicate_match",
        "missing_team",
        "missing_match",
        "missing_game",
        "team",
        "opponent",
        "season",
        "venue",
        "date",
        "as_of",
        "home_team_equals_away",
        "missing_identity",
        "null_key",
        "null_role",
        "integer_role",
        "naive_date",
    ],
)
def test_bad_fixture_keys_pairing_and_timing_fail_closed(goal_sources, damage):
    """No silent inner join or side swapping can hide a fixture identity discrepancy."""
    prematch, games = [f.copy(deep=True) for f in goal_sources]
    if damage == "duplicate_prematch":
        prematch = pd.concat([prematch, prematch.iloc[[0]]])
    elif damage == "duplicate_match":
        games = pd.concat([games, games.iloc[[0]]])
    elif damage == "missing_team":
        prematch = prematch.iloc[1:]
    elif damage == "missing_match":
        games = games.iloc[1:]
    elif damage == "missing_game":
        prematch = prematch.drop(index=800, level="game_id")
    elif damage == "team":
        prematch = prematch.rename(index={1: 99}, level="team_id")
    elif damage in ("opponent", "season", "venue", "date", "as_of"):
        column, value = {
            "opponent": ("opponent_id", 99),
            "season": ("season_id", 9),
            "venue": ("is_home", False),
            "date": ("date", prematch.date.iloc[0] + pd.Timedelta(hours=1)),
            "as_of": ("as_of", prematch.as_of.iloc[0] + pd.Timedelta(hours=1)),
        }[damage]
        prematch.iloc[0, prematch.columns.get_loc(column)] = value
    elif damage == "home_team_equals_away":
        games.loc[0, "away_team_id"] = games.loc[0, "home_team_id"]
    elif damage == "missing_identity":
        prematch = prematch.drop(columns="opponent_id")
    elif damage == "null_key":
        games.loc[0, "game_id"] = np.nan
    elif damage == "null_role":
        prematch["is_home"] = prematch.is_home.astype("boolean")
        prematch.iloc[0, prematch.columns.get_loc("is_home")] = pd.NA
    elif damage == "integer_role":
        prematch["is_home"] = prematch.is_home.astype(int)
    else:
        prematch["date"] = prematch.date.dt.tz_localize(None)
    with pytest.raises(ValueError):
        prepare((prematch, games))


@pytest.mark.parametrize("value", [-1, 0.5, np.nan, np.inf])
def test_invalid_observed_goals_are_rejected(goal_sources, value):
    prematch, games = goal_sources
    games = games.astype({"home_goals": float})
    games.loc[0, "home_goals"] = value
    with pytest.raises(ValueError, match="home_goals"):
        prepare((prematch, games))


@pytest.mark.parametrize(
    "seasons", [(), (8, 8), (9, 8), (True,), (8.0,), (8, 26), (8, 27), (8, 28)]
)
def test_invalid_or_heldout_season_requests_are_rejected(goal_sources, seasons):
    with pytest.raises(ValueError, match="development seasons"):
        prepare_goal_data(*goal_sources, feature_columns=("elo_diff",), seasons=seasons)


@pytest.mark.parametrize("season", [26, 27, 28])
def test_supplied_holdout_rows_are_rejected_even_when_not_requested(goal_sources, season):
    prematch, games = [f.copy(deep=True) for f in goal_sources]
    games.loc[0, "season_id"] = season
    prematch.loc[800, "season_id"] = season
    with pytest.raises(ValueError, match="requested development seasons"):
        prepare((prematch, games))


def test_goal_data_rejects_misaligned_target_and_context(goal_sources):
    data = prepare(goal_sources)
    with pytest.raises(ValueError, match="aligned"):
        GoalData(data.X, data.y.iloc[::-1], data.context)
    changed = data.context.copy()
    changed.iloc[0, changed.columns.get_loc("team_id")] = 999
    with pytest.raises(ValueError, match="context keys"):
        GoalData(data.X, data.y, changed)


@pytest.mark.real_data
def test_real_adapter_attributes_five_seeded_games_from_matches(real_tables):
    """Use the season-guarded real_tables fixture; never reload clean analysis inputs."""
    seasons = tuple(range(8, 26))
    assert set(real_tables.match_table.season_id) == set(seasons)
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Published prematch v2 is local only")
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["artifact_sha256"]
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    validate_prematch_v2_frame(prematch)
    data = prepare_goal_data(
        prematch, real_tables.match_table, feature_columns=M2_BASE_COLUMNS, seasons=seasons
    )
    assert len(data.y) == 8630 and tuple(data.X.columns) == M2_BASE_COLUMNS
    assert data.y.ge(0).all() and data.y.mod(1).eq(0).all()
    sample = real_tables.match_table.sort_index().sample(n=5, random_state=14051)
    report = []
    for game_id, row in sample.iterrows():
        home = int(data.y.loc[(game_id, row.home_team_id)])
        away = int(data.y.loc[(game_id, row.away_team_id)])
        assert (home, away) == (row.home_goals, row.away_goals)
        report.append(
            {
                "game_id": int(game_id),
                "home_team": row.home_team_name,
                "away_team": row.away_team_name,
                "matches": [int(row.home_goals), int(row.away_goals)],
                "y": [home, away],
            }
        )
    print(json.dumps(report, ensure_ascii=True))
