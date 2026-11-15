"""Pure historical morning states; future playoff fixtures remain hidden until split."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.features.table_state import INPUT_COLUMNS, historical_groups
from ligat.simulate.rules import SeasonRules, sporting_table

DEVELOPMENT_SEASONS = frozenset(range(8, 26))
FIXTURE_COLUMNS = ("game_id", "date", "as_of", "stage", "home_team_id", "away_team_id")


@dataclass(frozen=True)
class SeasonSnapshot:
    """What was known before any fixture on one normalized Jerusalem calendar day.

    Frames are independent copies. Only played rows contain scores. Before the
    split, remaining contains regular fixtures alone; ratings can still come from
    a team's next playoff appearance, because it has not played in the meantime.
    """

    season_id: int
    as_of: pd.Timestamp
    rules: SeasonRules
    phase: Literal["regular", "split"]
    groups: Mapping[int, str] | None
    played: pd.DataFrame
    table: pd.DataFrame
    remaining: pd.DataFrame
    elo: pd.Series
    previous_positions: Mapping[int, float]


def _morning(as_of: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(as_of)
    if pd.isna(day) or str(day.tz) != "Asia/Jerusalem" or day != day.normalize():
        raise ValueError("as_of must be normalized midnight in Asia/Jerusalem")
    return day


def _canonical_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = [*INPUT_COLUMNS, "as_of"]
    if not set(required) <= set(rows.columns):
        raise ValueError("Expected the game_rows participant projection")
    games = rows.loc[:, required].copy()
    if games.empty or not set(games.season_id) <= DEVELOPMENT_SEASONS:
        raise ValueError("Snapshots require nonempty development seasons 8-25")
    for column in ("game_id", "team_id", "opponent_id"):
        values = games[column]
        if values.isna().any() or values.mod(1).ne(0).any():
            raise ValueError("Game and team identifiers must be nonmissing integers")
        games[column] = values.astype("int64")
    if (
        str(games.date.dt.tz) != "Asia/Jerusalem"
        or games.date.isna().any()
        or not games.as_of.equals(games.date.dt.normalize())
        or games.duplicated(["game_id", "team_id"]).any()
        or not games.groupby("game_id").size().eq(2).all()
    ):
        raise ValueError("Invalid calendar dates or paired game identities")
    return games.sort_values(["as_of", "game_id", "team_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class _SeasonHistory:
    rows: pd.DataFrame
    fixtures: pd.DataFrame
    ratings: pd.DataFrame
    roster: list[int]
    rules: SeasonRules
    groups: Mapping[int, str]
    split_day: pd.Timestamp | None
    previous: Mapping[int, float]

    def morning(self, as_of: pd.Timestamp) -> SeasonSnapshot:
        day = _morning(as_of)
        split = self.split_day is not None and day >= self.split_day
        groups = dict(self.groups) if split else None
        played = self.rows.loc[self.rows.as_of.lt(day)].copy()
        remaining = self.fixtures.loc[self.fixtures.as_of.ge(day)]
        if not split:
            remaining = remaining.loc[remaining.stage.eq("regular")]
        # Look across all stages: a club may have finished its regular schedule.
        next_games = self.ratings.loc[self.ratings.as_of.ge(day)].drop_duplicates("team_id")
        ratings = next_games.set_index("team_id").elo_pre.reindex(self.roster)
        ratings.index = pd.Index(self.roster, dtype="int64", name="team_id")
        ratings.name = "elo_pre"
        if not np.isfinite(next_games.elo_pre.to_numpy(float)).all():
            raise ValueError("Every next fixture requires a finite prematch Elo")
        table = sporting_table(played, self.roster, self.rules, groups=groups)
        table.index = table.index.astype("int64")
        return SeasonSnapshot(
            self.rules.season_id,
            day,
            self.rules,
            "split" if split else "regular",
            groups,
            played,
            table,
            remaining.copy(),
            ratings.copy(),
            dict(self.previous),
        )


def _prepare(
    rows: pd.DataFrame,
    elo: pd.DataFrame,
    rules: Mapping[int, SeasonRules],
    season_id: int,
) -> _SeasonHistory:
    games = _canonical_rows(rows)
    if season_id not in DEVELOPMENT_SEASONS or season_id not in rules:
        raise ValueError("Requested season must be in development seasons 8-25")
    if "season_id" not in elo or not set(elo.season_id) <= DEVELOPMENT_SEASONS:
        raise ValueError("Elo inputs must contain only development seasons 8-25")
    if elo.index.names != ["game_id", "team_id"] or not elo.index.is_unique:
        raise ValueError("Elo requires unique (game_id, team_id) keys")
    current = games.loc[games.season_id.eq(season_id)].copy()
    roster = sorted(int(team) for team in current.team_id.unique())
    rule = rules[season_id]
    if len(roster) != rule.n_teams:
        raise ValueError("Season roster does not match SeasonRules")
    home = current.loc[current.is_home.eq(True)]
    fixtures = home.rename(columns={"team_id": "home_team_id", "opponent_id": "away_team_id"})
    fixtures = fixtures.loc[:, FIXTURE_COLUMNS].sort_values(["as_of", "date", "game_id"])
    if len(fixtures) * 2 != len(current) or not fixtures.game_id.is_unique:
        raise ValueError("Expected one home fixture per pair")
    rating_rows = elo.loc[elo.season_id.eq(season_id), ["elo_pre"]].reset_index()
    for column in ("game_id", "team_id"):
        if rating_rows[column].isna().any():
            raise ValueError("Elo identities cannot be missing")
        rating_rows[column] = rating_rows[column].astype("int64")
    ratings = (
        current[["game_id", "team_id", "as_of"]]
        .merge(rating_rows, on=["game_id", "team_id"], how="left", validate="one_to_one")
        .sort_values(["as_of", "game_id", "team_id"])
    )
    if not np.isfinite(ratings.elo_pre.to_numpy(float)).all():
        raise ValueError("Missing or nonfinite prematch Elo for a participant")
    playoff = current.loc[current.stage.ne("regular")]
    split_day = None if playoff.empty else playoff.as_of.min()
    previous: dict[int, float] = {}
    if season_id != 8:
        prior = games.loc[games.season_id.eq(season_id - 1)]
        if prior.empty or season_id - 1 not in rules:
            raise ValueError("Previous development season is required")
        prior_rule = rules[season_id - 1]
        positions = sporting_table(
            prior,
            sorted(int(t) for t in prior.team_id.unique()),
            prior_rule,
            groups=historical_groups(prior) if prior_rule.groups else None,
        ).position
        previous = {int(t): float(p) for t, p in positions.items() if t in roster}
    return _SeasonHistory(
        current, fixtures, ratings, roster, rule, historical_groups(current), split_day, previous
    )


def snapshot_as_of(
    rows: pd.DataFrame,
    elo: pd.DataFrame,
    rules: Mapping[int, SeasonRules],
    season_id: int,
    as_of: pd.Timestamp,
) -> SeasonSnapshot:
    """Build a pure sporting state strictly before as_of; read no files or forecasts."""
    _morning(as_of)
    return _prepare(rows, elo, rules, season_id).morning(as_of)


def season_snapshots(
    rows: pd.DataFrame,
    elo: pd.DataFrame,
    rules: Mapping[int, SeasonRules],
    season_id: int,
) -> Iterator[SeasonSnapshot]:
    """Project history once, then yield independent states for every fixture morning."""
    history = _prepare(rows, elo, rules, season_id)
    for day in history.rows.as_of.drop_duplicates().sort_values():
        yield history.morning(day)
