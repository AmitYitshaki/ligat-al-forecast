"""Calendar-date prematch tables, rebuilt only from earlier sporting results."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables
from ligat.features.registry import validate_feature_columns
from ligat.simulate.rules import SeasonRules, objective_flags, sporting_table

INPUT_COLUMNS = (
    "game_id",
    "team_id",
    "season_id",
    "date",
    "round",
    "stage",
    "is_home",
    "opponent_id",
    "goals_scored",
    "goals_conceded",
)


def game_rows(tables: AnalysisTables) -> pd.DataFrame:
    """Project the approved loader onto identity, schedule and observed goal columns.

    Official summaries, existing EDA cumulative columns and all rejected features
    are excluded from the computational input, not merely dropped from its output.
    """
    games = tables.team_matches.reset_index().loc[:, INPUT_COLUMNS].copy()
    if str(games.date.dt.tz) != "Asia/Jerusalem":
        raise ValueError("Dates must have the Asia/Jerusalem timezone")
    if (
        games.duplicated(["game_id", "team_id"]).any()
        or not games.groupby("game_id").size().eq(2).all()
    ):
        raise ValueError("Expected exactly two distinct team rows per game")
    games["as_of"] = games.date.dt.normalize()
    return games.sort_values(["as_of", "game_id", "team_id"]).reset_index(drop=True)


def historical_groups(season_games: pd.DataFrame) -> dict[int, str]:
    """Read fixed playoff assignment from schedule metadata, never from final_tables.

    This mapping must only be used at/after the first scheduled playoff date.
    Future playoff results do not enter it. Before the split no feature uses it.
    """
    playoff = season_games.loc[
        season_games.stage.ne("regular"), ["team_id", "stage"]
    ].drop_duplicates()
    if playoff.team_id.duplicated().any():
        raise ValueError("A team appears in multiple historical playoff groups")
    return {int(row.team_id): str(row.stage) for row in playoff.itertuples()}


def _stage_played(past: pd.DataFrame, roster: list[int], stage: str) -> pd.Series:
    return past.loc[past.stage.eq(stage)].groupby("team_id").size().reindex(roster, fill_value=0)


def _snapshot_features(
    past: pd.DataFrame,
    roster: list[int],
    rules: SeasonRules,
    groups: Mapping[int, str] | None,
) -> pd.DataFrame:
    table = sporting_table(past, roster, rules, groups=groups)
    table["position_pct"] = (table.position - 1) / (rules.n_teams - 1)
    table["gap_to_leader"] = table.sort_values("position").points.iloc[0] - table.points
    safety_points = table.sort_values("position").points.iloc[rules.last_safe_position - 1]
    table["gap_to_safety"] = safety_points - table.points
    table["gap_to_cut"] = np.nan
    if groups is None and rules.upper_size is not None:
        table["gap_to_cut"] = (
            table.sort_values("position").points.iloc[rules.upper_size - 1] - table.points
        )
    parts = []
    for stage, part in table.groupby("table_group", sort=False):
        part = part.copy()
        # Fixed playoff groups cannot cross these objective lines after splitting.
        if stage in ("upper", "middle"):
            part["gap_to_safety"] = np.nan
        if stage in ("middle", "lower"):
            part["gap_to_leader"] = np.nan
        played = _stage_played(past, list(part.index), str(stage))
        remaining = rules.stage_rounds(str(stage)) - played
        if remaining.lt(0).any():
            raise ValueError(f"More games than format permits in season {rules.season_id}/{stage}")
        part["games_remaining_stage"] = remaining
        part["games_behind"] = part.games_played.max() - part.games_played
        flags = objective_flags(part, rules, str(stage), remaining)
        parts.append(part.join(flags))
    return pd.concat(parts).drop(columns="table_group")


@dataclass(frozen=True)
class TableStateBlock:
    """League state on the morning of each fixture; no cross-day or cross-season leak."""

    rules: Mapping[int, SeasonRules]
    name: str = "table_state"
    football_rationale: str = (
        "The sporting table and mathematically open objectives before kickoff."
    )

    def compute(self, tables: AnalysisTables) -> pd.DataFrame:
        """Compute independent frames; the input AnalysisTables are never modified."""
        source = game_rows(tables)
        rows = []
        for season, games in source.groupby("season_id", sort=True):
            rule = self.rules[int(season)]
            roster = sorted(int(team) for team in games.team_id.unique())
            if len(roster) != rule.n_teams:
                raise ValueError(f"Roster size mismatch for season {season}")
            assignments = historical_groups(games)
            playoff = games.loc[games.stage.ne("regular")]
            split_day = playoff.as_of.min() if not playoff.empty else None
            for day, targets in games.groupby("as_of", sort=True):
                past = games.loc[games.as_of.lt(day)]
                groups = assignments if split_day is not None and day >= split_day else None
                snapshot = _snapshot_features(past, roster, rule, groups)
                identity = targets[
                    [
                        "game_id",
                        "team_id",
                        "season_id",
                        "date",
                        "as_of",
                        "round",
                        "stage",
                        "is_home",
                        "opponent_id",
                    ]
                ]
                output = identity.join(snapshot, on="team_id", validate="many_to_one")
                # Normally the target stage equals the table stage. Recompute the
                # schedule counters explicitly to also cover a late regular fixture.
                for stage, indices in output.groupby("stage", observed=True).groups.items():
                    played = _stage_played(past, roster, str(stage))
                    output.loc[indices, "games_remaining_stage"] = rule.stage_rounds(
                        str(stage)
                    ) - output.loc[indices, "team_id"].map(played)
                output["rounds_to_stage_end"] = output.games_remaining_stage - 1
                rows.append(output)
        result = pd.concat(rows, ignore_index=True).set_index(["game_id", "team_id"]).sort_index()
        validate_feature_columns(result, self.name)
        return result
