"""Pure M2 input adapter: frozen prematch features and separately observed match goals."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.features.pipeline import DEVELOPMENT_SEASONS
from ligat.features.prematch_v2 import PREMATCH_V2_COLUMNS

GOAL_CONTEXT_COLUMNS = ("game_id", "team_id", "season_id", "date", "as_of", "is_home")
_PAIR_COLUMNS = ("season_id", "date", "as_of", "is_home", "opponent_id")


def _integer_values(values: pd.Series, name: str, *, positive: bool = False) -> None:
    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
        or values.isna().any()
    ):
        raise ValueError(f"{name} must contain complete integer values")
    numbers = values.to_numpy(dtype=float)
    if (
        not np.isfinite(numbers).all()
        or (numbers % 1 != 0).any()
        or (numbers < (1 if positive else 0)).any()
    ):
        raise ValueError(
            f"{name} must contain finite {'positive' if positive else 'nonnegative'} integers"
        )


def _dates(values: pd.Series, name: str) -> None:
    if (
        not isinstance(values.dtype, pd.DatetimeTZDtype)
        or str(values.dt.tz) != "Asia/Jerusalem"
        or values.isna().any()
    ):
        raise ValueError(f"{name} requires complete Asia/Jerusalem timestamps")


def _feature_columns(columns: Sequence[str], available: pd.Index) -> list[str]:
    selected = list(columns)
    if len(set(selected)) != len(selected):
        raise ValueError("Goal features must be distinct")
    if not set(selected) <= set(available):
        raise ValueError("Requested goal features are missing from prematch")
    if not set(selected) <= {*PREMATCH_V2_COLUMNS, "team_id"}:
        raise ValueError("Goal features must come from the prematch v2 contract, never targets")
    return selected


@dataclass(frozen=True)
class GoalData:
    """Aligned X, integer goals y and target-free context, indexed by (game_id, team_id).

    Frames are copied by the adapter and again at the judge boundary. Context
    columns identify fixtures and their timing; they are not implicit features.
    """

    X: pd.DataFrame
    y: pd.Series
    context: pd.DataFrame

    def __post_init__(self) -> None:
        index = self.X.index
        if (
            not isinstance(index, pd.MultiIndex)
            or index.names != ["game_id", "team_id"]
            or not index.is_unique
            or not len(index)
            or not index.equals(self.y.index)
            or not index.equals(self.context.index)
        ):
            raise ValueError("Goal data requires aligned, unique (game_id, team_id) keys")
        if not self.X.columns.is_unique or tuple(self.context.columns) != GOAL_CONTEXT_COLUMNS:
            raise ValueError("Goal data requires distinct features and the exact identity context")
        _feature_columns(list(self.X.columns), self.X.columns)
        keys = index.to_frame(index=False)
        context = self.context.reset_index(drop=True)
        for column in ("game_id", "team_id", "season_id"):
            _integer_values(context[column], column, positive=True)
        for column in ("game_id", "team_id"):
            if not np.array_equal(keys[column], context[column]):
                raise ValueError("Goal context keys differ from the feature index")
        if not context.season_id.isin(DEVELOPMENT_SEASONS).all():
            raise ValueError("Goal data only accepts development seasons 8-25")
        _dates(context.date, "date")
        _dates(context.as_of, "as_of")
        if not context.as_of.eq(context.date.dt.normalize()).all():
            raise ValueError("Goal as_of must be the Asia/Jerusalem calendar-day start")
        if not pd.api.types.is_bool_dtype(context.is_home.dtype) or context.is_home.isna().any():
            raise ValueError("Goal is_home must be complete boolean values")
        pairs = context.groupby("game_id")
        if not pairs.size().eq(2).all() or not pairs.is_home.sum().eq(1).all():
            raise ValueError("Goal data requires one home and one away row per game")
        if not pairs[["season_id", "date", "as_of"]].nunique().eq(1).all().all():
            raise ValueError("Paired goal rows disagree on season or timing")
        _integer_values(self.y, "goals_scored")
        if self.y.name != "goals_scored":
            raise ValueError("Goal target must be explicitly named goals_scored")


def prepare_goal_data(
    prematch: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    seasons: Sequence[int],
) -> GoalData:
    """Attribute actual home/away goals by keys, never from cumulative prematch columns.

    Both supplied frames must contain exactly the explicit development seasons
    and fixture population. Small synthetic projections are supported; frozen
    artifact/schema validation remains the loader's responsibility. team_id may
    be selected explicitly from the prematch index for the later F6 experiment.
    No I/O, feature recomputation, filling or numeric feature casting occurs.
    """
    chosen = tuple(seasons)
    if (
        not chosen
        or chosen != tuple(sorted(set(chosen)))
        or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in chosen)
        or not set(chosen) <= set(DEVELOPMENT_SEASONS)
    ):
        raise ValueError("Explicit distinct development seasons 8-25 are required")
    if (
        not isinstance(prematch.index, pd.MultiIndex)
        or prematch.index.names != ["game_id", "team_id"]
        or not prematch.index.is_unique
        or prematch.empty
        or not prematch.columns.is_unique
        or not set(_PAIR_COLUMNS) <= set(prematch)
    ):
        raise ValueError("Prematch requires unique (game_id, team_id) keys and fixture identity")
    if set(prematch.columns) & {"game_id", "team_id"}:
        raise ValueError("Prematch identity keys must appear only in the index")
    rows = prematch.reset_index().sort_values(["game_id", "team_id"]).reset_index(drop=True)
    games = matches.copy(deep=True)
    if "game_id" not in games and games.index.name == "game_id":
        games = games.reset_index()
    required = (
        "game_id",
        "season_id",
        "date",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
    )
    if (
        not games.columns.is_unique
        or not set(required) <= set(games)
        or games.empty
        or games.game_id.duplicated().any()
    ):
        raise ValueError("Matches requires unique game_id keys, identities and both goal targets")
    games = games.loc[:, required].copy()
    for column in ("game_id", "season_id", "home_team_id", "away_team_id"):
        _integer_values(games[column], f"matches.{column}", positive=True)
    for column in ("home_goals", "away_goals"):
        _integer_values(games[column], column)
    _dates(games.date, "matches.date")
    if games.home_team_id.eq(games.away_team_id).any():
        raise ValueError("Matches home and away teams must differ")
    if set(rows.season_id) != set(chosen) or set(games.season_id) != set(chosen):
        raise ValueError("Goal inputs must contain exactly the requested development seasons")
    if not pd.api.types.is_bool_dtype(rows.is_home.dtype) or rows.is_home.isna().any():
        raise ValueError("Prematch is_home must be complete boolean values")
    _integer_values(rows.opponent_id, "opponent_id", positive=True)

    sides = []
    for side, opponent in (("home", "away"), ("away", "home")):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": games.game_id,
                    "team_id": games[f"{side}_team_id"],
                    "season_id": games.season_id,
                    "date": games.date,
                    "is_home": side == "home",
                    "opponent_id": games[f"{opponent}_team_id"],
                    "goals_scored": games[f"{side}_goals"],
                }
            )
        )
    expected = pd.concat(sides, ignore_index=True).set_index(["game_id", "team_id"]).sort_index()
    actual = rows.set_index(["game_id", "team_id"])
    # Nullable clean IDs and NumPy artifact IDs represent the same numeric keys.
    if len(actual) != len(expected) or not actual.index.isin(expected.index).all():
        raise ValueError("Prematch and matches must contain exactly the same team-game keys")
    expected = expected.reindex(actual.index)
    for column in ("season_id", "date", "is_home", "opponent_id"):
        if actual[column].isna().any() or not actual[column].eq(expected[column]).all():
            raise ValueError(f"Prematch/matches fixture identity mismatch: {column}")
    selected = _feature_columns(feature_columns, rows.columns)
    X = rows.loc[:, selected].copy()
    X.index = actual.index
    context = rows.loc[:, GOAL_CONTEXT_COLUMNS].copy()
    context.index = actual.index
    y = expected.goals_scored.astype("int64").copy()
    return GoalData(X, y, context)
