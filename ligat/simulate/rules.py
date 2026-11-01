"""Sporting league rules, shared by historical prematch and future simulated paths.

Only ``load_season_rules`` performs I/O. Administrative deductions/results never
enter these calculations. A historical schedule may supply fixed playoff groups;
``split_playoffs`` instead splits a simulated sporting table.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml


@dataclass(frozen=True)
class PlayoffGroup:
    """A fixed group size and its scheduled games per team."""

    name: str
    teams: int
    rounds: int


@dataclass(frozen=True)
class SeasonRules:
    """Season-specific sporting rules; ranks in the relegation test are also at risk."""

    season_id: int
    n_teams: int
    regular_rounds: int
    groups: tuple[PlayoffGroup, ...]
    points_halved: bool
    relegated: int
    relegation_playoff_position: int | None
    tiebreak_rules: tuple[str, ...]

    @property
    def upper_size(self) -> int | None:
        return self.groups[0].teams if self.groups else None

    @property
    def last_safe_position(self) -> int:
        direct = self.n_teams - self.relegated
        return (
            min(direct, self.relegation_playoff_position - 1)
            if (self.relegation_playoff_position is not None)
            else direct
        )

    def stage_rounds(self, stage: str) -> int:
        """Return games per team from the format, never from future result rows."""
        if stage == "regular":
            return self.regular_rounds
        return next(group.rounds for group in self.groups if group.name == stage)

    def offset(self, stage: str) -> int:
        """Return the fixed overall-rank offset of a playoff group."""
        if stage == "regular":
            return 0
        offset = 0
        for group in self.groups:
            if group.name == stage:
                return offset
            offset += group.teams
        raise ValueError(f"Unknown stage: {stage}")


def load_season_rules(path: Path = Path("config/seasons.yaml")) -> dict[int, SeasonRules]:
    """Load the single configuration boundary; all later rule operations are pure."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["playoff_split_tiebreak"] != "previous_season_rank":
        raise ValueError("Unsupported playoff split tiebreak")
    rules = {}
    for item in config["seasons"]:
        row = config["defaults"] | item
        sid = int(row["season_id"])
        groups = tuple(
            PlayoffGroup(
                name,
                int(row["playoff_groups"][name]),
                (int(row["playoff_groups"][name]) - 1)
                * int(row.get("playoff_legs", config["playoff_formats"][name]["legs"])),
            )
            for name in row["playoffs"]
        )
        if groups and sum(group.teams for group in groups) != row["n_teams"]:
            raise ValueError(f"Playoff group sizes do not cover season {sid}")
        if (row["points_win"], row["points_draw"]) != (3, 1):
            raise ValueError("Only sporting 3/1/0 is supported")
        era = "until_2007_08" if sid <= 9 else "from_2008_09"
        rules[sid] = SeasonRules(
            sid,
            int(row["n_teams"]),
            int(row["regular_rounds"]),
            groups,
            bool(row["points_halved"]),
            int(row["relegated"]),
            row["relegation_playoff_position"],
            tuple(config["tiebreak_rules"][era]),
        )
    return rules


def halve_points(points: int) -> int:
    """Carry regular-season points into a halved playoff, rounding upward."""
    if points < 0:
        raise ValueError("Sporting points cannot be negative")
    return (points + 1) // 2


def _totals(games: pd.DataFrame, roster: Sequence[int]) -> pd.DataFrame:
    frame = games.assign(
        games_played=1,
        wins=games.goals_scored.gt(games.goals_conceded).astype(int),
        draws=games.goals_scored.eq(games.goals_conceded).astype(int),
        losses=games.goals_scored.lt(games.goals_conceded).astype(int),
    ).rename(columns={"goals_scored": "goals_for", "goals_conceded": "goals_against"})
    columns = ["games_played", "wins", "draws", "losses", "goals_for", "goals_against"]
    table = frame.groupby("team_id", observed=True)[columns].sum().reindex(roster, fill_value=0)
    table["goal_difference"] = table.goals_for - table.goals_against
    table["points"] = 3 * table.wins + table.draws
    return table


def rank_table(
    table: pd.DataFrame,
    rules: SeasonRules,
    games: pd.DataFrame | None = None,
    *,
    previous_positions: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Rank by era, then head-to-head; complete ties share their average rank.

    Previous-season positions are supplied only at the playoff split. Team IDs
    never resolve a sporting tie. ``games`` contains one row per participant.
    """
    keys = [key for key in rules.tiebreak_rules if key != "head_to_head"]
    ranked = table.sort_values(keys, ascending=False).copy()
    ranked["position"] = np.nan
    start = 1
    for _, tied in ranked.groupby(keys, sort=False, observed=True, dropna=False):
        if len(tied) > 1 and "head_to_head" in rules.tiebreak_rules and games is not None:
            mutual = games.loc[games.team_id.isin(tied.index) & games.opponent_id.isin(tied.index)]
            mini = _totals(mutual, list(tied.index)).sort_values(keys, ascending=False)
            subgroups = [part.index for _, part in mini.groupby(keys, sort=False)]
        else:
            subgroups = [tied.index]
        for indices in subgroups:
            if len(indices) > 1 and previous_positions is not None:
                previous = pd.Series(
                    {team: previous_positions.get(team, np.nan) for team in indices}
                )
                if previous.notna().all() and previous.is_unique:
                    indices = previous.sort_values().index
                    for team in indices:
                        ranked.at[team, "position"] = float(start)
                        start += 1
                    continue
            ranked.loc[indices, "position"] = start + (len(indices) - 1) / 2
            start += len(indices)
    return ranked.sort_values("position", kind="stable")


def split_playoffs(
    table: pd.DataFrame,
    rules: SeasonRules,
    games: pd.DataFrame | None = None,
    *,
    previous_positions: Mapping[int, float] | None = None,
) -> dict[int, str]:
    """Split a sporting path; refuse to invent an assignment across an unresolved tie."""
    if not rules.groups:
        return {int(team): "regular" for team in table.index}
    ranked = rank_table(table, rules, games, previous_positions=previous_positions)
    assignments: dict[int, str] = {}
    offset = 0
    for group in rules.groups:
        stop = offset + group.teams
        if stop < len(ranked) and ranked.position.iloc[stop - 1] == ranked.position.iloc[stop]:
            raise ValueError("Unresolved tie across playoff boundary; previous rank required")
        assignments.update({int(team): group.name for team in ranked.index[offset:stop]})
        offset = stop
    return assignments


def sporting_table(
    games: pd.DataFrame,
    roster: Sequence[int],
    rules: SeasonRules,
    *,
    groups: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Aggregate supplied past results; fixed groups imply the split has occurred."""
    table = _totals(games, roster)
    if groups is not None and rules.points_halved:
        regular = _totals(games.loc[games.stage.eq("regular")], roster).points
        table["points"] += (regular + 1) // 2 - regular
    if groups is None:
        table = rank_table(table, rules, games)
        table["group_position"] = table.position
        table["table_group"] = "regular"
    else:
        if set(groups) != set(roster):
            raise ValueError("Every team needs a known playoff group")
        parts = []
        for group in rules.groups:
            ids = [team for team in roster if groups[team] == group.name]
            if len(ids) != group.teams:
                raise ValueError(f"Invalid size for playoff group {group.name}")
            part = rank_table(table.loc[ids], rules, games)
            part["group_position"] = part.position
            part["position"] += rules.offset(group.name)
            part["table_group"] = group.name
            parts.append(part)
        table = pd.concat(parts)
    table["points_per_game"] = table.points.div(table.games_played.replace(0, np.nan)).fillna(0)
    return table


def _side_decided(low: pd.Series, high: pd.Series, places: int) -> pd.Series:
    """Prove membership/nonmembership in the top k; equal points remain open."""
    decided = pd.Series(False, index=low.index)
    for team in low.index:
        others = low.index[low.index != team]
        secured = int(high.loc[others].ge(low.loc[team]).sum()) < places
        excluded = int(low.loc[others].gt(high.loc[team]).sum()) >= places
        decided.loc[team] = secured or excluded
    return decided


def objective_flags(
    table: pd.DataFrame,
    rules: SeasonRules,
    stage: str,
    remaining: pd.Series,
) -> pd.DataFrame:
    """Conservative mathematical target decisions, without a model or fixture outcomes.

    Before splitting, title exclusion is proved only by exclusion from the upper
    group; safety only by securing a non-lower group. This intentionally misses
    some clinches rather than comparing future points across fixed rank offsets.
    After splitting (or without playoffs), independent point envelopes suffice.
    Decision ה-ת10 also requires a proven side of the top-2, top-3 and top-4
    boundaries before calling a team's remaining games irrelevant.
    """
    low = table.points.astype(float)
    high = low + 3 * remaining
    result = pd.DataFrame(
        False,
        index=table.index,
        columns=[
            "title_decided",
            "cut_decided",
            "relegation_decided",
        ],
    )
    if stage == "regular" and rules.groups:
        if rules.points_halved:
            low, high = np.ceil(low / 2), np.ceil(high / 2)
        upper = int(rules.upper_size or 0)
        result["cut_decided"] = _side_decided(low, high, upper)
        non_lower = sum(group.teams for group in rules.groups if group.name != "lower")
        for team in table.index:
            others = table.index[table.index != team]
            result.at[team, "title_decided"] = (
                int(low.loc[others].gt(high.loc[team]).sum()) >= upper
            )
            result.at[team, "relegation_decided"] = (
                int(high.loc[others].ge(low.loc[team]).sum()) < non_lower
            )
    elif stage == "middle":
        result.loc[:, :] = True
    else:
        result["cut_decided"] = True
        result["title_decided"] = _side_decided(low, high, 1) if stage != "lower" else True
        if stage == "upper":
            result["relegation_decided"] = True
        else:
            safe = rules.last_safe_position - rules.offset(stage)
            result["relegation_decided"] = _side_decided(low, high, safe)
    if stage == "regular" and rules.groups:
        result["places_decided"] = result.title_decided
    elif stage in ("middle", "lower"):
        result["places_decided"] = True
    else:
        result["places_decided"] = pd.concat(
            [_side_decided(low, high, places) for places in (2, 3, 4)], axis=1
        ).all(axis=1)
    result["nothing_to_play"] = result.all(axis=1)
    return result.astype(bool)
