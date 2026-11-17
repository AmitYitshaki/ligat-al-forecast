"""Vectorized sporting totals and ranks, checked against the canonical rules module."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ligat.simulate.rules import SeasonRules

Array = NDArray[Any]
State = dict[str, Array]


class LeagueEngine:
    """Accumulate shared paths; rule configuration determines every ranking criterion."""

    def __init__(
        self,
        rules: SeasonRules,
        roster: Sequence[int],
        previous_positions: Mapping[int, float] | None = None,
    ) -> None:
        self.rules = rules
        self.teams = np.asarray(sorted(roster), dtype=np.int64)
        self.n = len(self.teams)
        if self.n != rules.n_teams or len(np.unique(self.teams)) != self.n:
            raise ValueError("Engine roster must match SeasonRules")
        self.lookup = {int(t): i for i, t in enumerate(self.teams)}
        previous = previous_positions or {}
        self.previous = np.array([previous.get(int(t), np.nan) for t in self.teams])
        self.tie_counts: Counter[str] = Counter()

    def empty(self, paths: int) -> State:
        """Allocate integer totals and directed head-to-head counters for each path."""
        if paths < 1:
            raise ValueError("At least one path is required")
        state = {
            name: np.zeros((paths, self.n), dtype=np.int16)
            for name in ("points", "gf", "ga", "wins", "gp", "regular_points")
        }
        state["h2h"] = np.zeros((paths, self.n, self.n, 4), dtype=np.int16)
        return state

    @staticmethod
    def clone(state: State) -> State:
        """Isolate a focal counterfactual without drawing a second future."""
        return {name: values.copy() for name, values in state.items()}

    def add(
        self,
        state: State,
        home: int | Array,
        away: int | Array,
        home_goals: int | Array,
        away_goals: int | Array,
        *,
        regular: bool = False,
        sign: int = 1,
    ) -> None:
        """Add or remove one score per path, including directed mutual-game totals."""
        paths = len(state["points"])
        index = np.arange(paths)
        h, a, hg, ag = (np.broadcast_to(x, (paths,)) for x in (home, away, home_goals, away_goals))
        hw, aw, draw = (
            (hg > ag).astype(np.int16),
            (ag > hg).astype(np.int16),
            (hg == ag).astype(np.int16),
        )
        hp, ap = 3 * hw + draw, 3 * aw + draw
        for column, hv, av in (
            ("points", hp, ap),
            ("gf", hg, ag),
            ("ga", ag, hg),
            ("wins", hw, aw),
            ("gp", np.ones(paths, dtype=np.int16), np.ones(paths, dtype=np.int16)),
        ):
            state[column][index, h] += sign * hv
            state[column][index, a] += sign * av
        if regular:
            state["regular_points"][index, h] += sign * hp
            state["regular_points"][index, a] += sign * ap
        for k, hv, av in ((0, hp, ap), (1, hg, ag), (2, ag, hg), (3, hw, aw)):
            state["h2h"][index, h, a, k] += sign * hv
            state["h2h"][index, a, h, k] += sign * av

    def from_played(self, rows: pd.DataFrame, paths: int) -> State:
        """Use each completed fixture once; caller excludes the entire focal day."""
        state = self.empty(paths)
        for row in rows.loc[rows.is_home.eq(True)].itertuples():
            self.add(
                state,
                self.lookup[int(row.team_id)],
                self.lookup[int(row.opponent_id)],
                int(row.goals_scored),
                int(row.goals_conceded),
                regular=row.stage == "regular",
            )
        return state

    def points(self, state: State, halved: bool) -> Array:
        """Apply the same upward regular-points halving as sporting_table."""
        if halved and self.rules.points_halved:
            return np.asarray(
                state["points"] + (state["regular_points"] + 1) // 2 - state["regular_points"]
            )
        return state["points"]

    def rank(
        self,
        state: State,
        *,
        halved: bool = False,
        group_ids: Array | None = None,
        at_split: bool = False,
        lottery: Array | None = None,
    ) -> tuple[Array, Array]:
        """Return team-index order and ranks using sporting, split and lottery rules.

        Head-to-head is evaluated once within full-table ties, exactly as in
        rank_table. Previous positions apply at split only. IDs do not decide ties.
        """
        paths = len(state["points"])
        groups = np.zeros((paths, self.n), dtype=np.int16) if group_ids is None else group_ids
        criteria = [name for name in self.rules.tiebreak_rules if name != "head_to_head"]
        totals = {
            "points": self.points(state, halved),
            "goal_difference": state["gf"] - state["ga"],
            "wins": state["wins"],
            "goals_for": state["gf"],
        }
        keys = [totals[name] for name in criteria]
        same = groups[:, :, None] == groups[:, None, :]
        full = same.copy()
        for key in keys:
            full &= key[:, :, None] == key[:, None, :]
        if "head_to_head" in self.rules.tiebreak_rules:
            mutual = (state["h2h"] * full[:, :, :, None]).sum(axis=2)
            mini = {
                "points": mutual[:, :, 0],
                "goal_difference": mutual[:, :, 1] - mutual[:, :, 2],
                "wins": mutual[:, :, 3],
                "goals_for": mutual[:, :, 1],
            }
            for name in criteria:
                key = mini[name]
                keys.append(key)
                full &= key[:, :, None] == key[:, None, :]
        off_diagonal = ~np.eye(self.n, dtype=bool)[None, :, :]
        full_pairs = full & off_diagonal
        phase = "split" if at_split else "final"
        if at_split:
            previous = np.where(np.isfinite(self.previous), self.previous, np.inf)
            keys.append(np.broadcast_to(-previous, (paths, self.n)))
            if lottery is not None:
                mixed = np.isfinite(self.previous)[:, None] != np.isfinite(self.previous)[None, :]
                self.tie_counts["split_promoted_rule_paths"] += int(
                    (full_pairs & mixed[None, :, :]).any(axis=(1, 2)).sum()
                )
                different = previous[:, None] != previous[None, :]
                self.tie_counts["split_previous_rule_paths"] += int(
                    (full_pairs & different[None, :, :]).any(axis=(1, 2)).sum()
                )
            full &= previous[None, :, None] == previous[None, None, :]
        if lottery is not None:
            if lottery.shape != (paths, self.n) or not np.isfinite(lottery).all():
                raise ValueError("Every path/team requires a finite lottery key")
            draw_pairs = full & off_diagonal
            if (draw_pairs & (lottery[:, :, None] == lottery[:, None, :])).any():
                raise ValueError("Tied teams need distinct lottery keys")
            self.tie_counts[f"{phase}_rank_paths"] += paths
            self.tie_counts[f"{phase}_full_tie_paths"] += int(full_pairs.any(axis=(1, 2)).sum())
            self.tie_counts[f"{phase}_lottery_paths"] += int(draw_pairs.any(axis=(1, 2)).sum())
            self.tie_counts[f"{phase}_lottery_pairs"] += int(draw_pairs.sum() // 2)
            keys.append(-lottery)
        sort_keys = [-groups, *keys]
        order = np.lexsort(tuple(-k for k in reversed(sort_keys)), axis=1)
        equal = same.copy()
        for key in keys:
            equal &= key[:, :, None] == key[:, None, :]
        inverse = np.empty_like(order)
        np.put_along_axis(
            inverse, order, np.broadcast_to(np.arange(1, self.n + 1), order.shape), axis=1
        )
        rank = (equal * inverse[:, None, :]).sum(axis=2) / equal.sum(axis=2)
        if lottery is not None and not np.array_equal(
            np.sort(rank, axis=1), np.broadcast_to(np.arange(1, self.n + 1), rank.shape)
        ):
            raise AssertionError("Lottery ranks are not a complete permutation")
        if at_split and lottery is None:
            sorted_ranks = np.take_along_axis(rank, order, axis=1)
            stop = 0
            for group in self.rules.groups[:-1]:
                stop += group.teams
                if np.any(sorted_ranks[:, stop - 1] == sorted_ranks[:, stop]):
                    raise ValueError("Unresolved tie across playoff boundary")
        return order, rank

    def membership(self, order: Array) -> Array:
        """Assign every path's ranked participants to configured playoff groups."""
        groups = np.zeros(order.shape, dtype=np.int16)
        offset = 0
        for i, group in enumerate(self.rules.groups):
            np.put_along_axis(groups, order[:, offset : offset + group.teams], i, axis=1)
            offset += group.teams
        return groups

    def terminal(self, state: State, groups: Array, lottery: Array) -> dict[str, Array]:
        """Resolve terminal targets and assert conserved champion and survivor counts."""
        _, ranks = self.rank(
            state, halved=bool(self.rules.groups), group_ids=groups, lottery=lottery
        )
        targets = {
            "title": ranks <= 1,
            "cut": groups == 0,
            "survival": ranks <= self.rules.last_safe_position,
            "survival_direct": ranks <= self.n - self.rules.relegated,
        }
        for name, count in (
            ("title", 1),
            ("survival", self.rules.last_safe_position),
            ("survival_direct", self.n - self.rules.relegated),
        ):
            if not (targets[name].sum(axis=1) == count).all():
                raise AssertionError(f"Nonconserved {name} membership")
        if self.rules.groups:
            for index, group in enumerate(self.rules.groups):
                if not ((groups == index).sum(axis=1) == group.teams).all():
                    raise AssertionError("Invalid playoff group size")
        self.tie_counts["conservation_checked_paths"] += len(ranks)
        return targets

    def totals_frame(self, state: State, groups: Array | None = None) -> pd.DataFrame:
        """Expose a single engineering replay path, without stochastic deciders."""
        _, rank = self.rank(state, halved=groups is not None, group_ids=groups)
        return pd.DataFrame(
            {
                "points": self.points(state, groups is not None)[0],
                "goals_for": state["gf"][0],
                "goals_against": state["ga"][0],
                "wins": state["wins"][0],
                "games_played": state["gp"][0],
                "position": rank[0],
            },
            index=pd.Index(self.teams, name="team_id"),
        )
