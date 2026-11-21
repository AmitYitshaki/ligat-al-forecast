"""Shared, target-free team identity encoding for the M2 tree adapters (Q29)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.goal_inputs import _integer_values

IDENTITY_COLUMNS = ("team_id", "opponent_id")


def _context(context: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(context, pd.DataFrame)
        or context.empty
        or not context.columns.is_unique
        or not {"team_id", "season_id"} <= set(context)
        or not isinstance(context.index, pd.MultiIndex)
        or context.index.names != ["game_id", "team_id"]
        or not context.index.is_unique
        or context.index.to_frame(index=False).isna().any().any()
    ):
        raise ValueError("Identity encoding requires unique team-game context and seasons")
    for column in ("team_id", "season_id"):
        _integer_values(context[column], column, positive=True)
    if context.season_id.lt(8).any() or not np.array_equal(
        context.team_id, context.index.get_level_values("team_id")
    ):
        raise ValueError("Identity context keys disagree or precede season 8")
    return context.loc[:, ["team_id", "season_id"]].reset_index(drop=True)


@dataclass(frozen=True)
class TeamIdentity:
    """Fixed vocabulary plus earliest original appearances, never encoded appearances.

    The vocabulary is supplied explicitly (28 development IDs in EXP-005). It
    defines the schema only; membership never means that a team is familiar.
    State contains no targets and is immutable, so prediction cannot update it.
    """

    team_ids: tuple[int, ...]
    first_seen: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.team_ids
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
                for v in self.team_ids
            )
            or len(set(self.team_ids)) != len(self.team_ids)
        ):
            raise ValueError("Identity vocabulary must contain distinct positive integer IDs")
        ids = tuple(sorted(int(v) for v in self.team_ids))
        if any(
            len(row) != 2
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in row
            )
            or row[0] not in ids
            or row[1] < 8
            for row in self.first_seen
        ) or len({row[0] for row in self.first_seen}) != len(self.first_seen):
            raise ValueError(
                "Identity history requires distinct vocabulary IDs and valid first seasons"
            )
        object.__setattr__(self, "team_ids", ids)
        object.__setattr__(
            self, "first_seen", tuple(sorted((int(i), int(s)) for i, s in self.first_seen))
        )

    @property
    def dtype(self) -> pd.CategoricalDtype:
        """All category names are strings, including unused IDs and unknown."""
        return pd.CategoricalDtype([*(f"team:{i}" for i in self.team_ids), "unknown"])

    @classmethod
    def from_context(cls, context: pd.DataFrame, *, team_ids: Sequence[int]) -> Self:
        """Record original IDs' first seasons using only the supplied training context."""
        rows = _context(context)
        first = rows.groupby("team_id", sort=True).season_id.min()
        return cls(tuple(team_ids), tuple((int(i), int(s)) for i, s in first.items()))

    def transform(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.DataFrame:
        """Encode explicit own/opponent columns by strictly earlier-season appearances.

        Every season-8 team is known. Later first-season rows remain unknown for
        the entire season. A team outside the fixed vocabulary can be predicted
        as unknown, but cannot silently become a new fitted category.
        """
        rows = _context(context)
        if (
            not isinstance(X, pd.DataFrame)
            or not X.columns.is_unique
            or not X.index.equals(context.index)
            or not set(IDENTITY_COLUMNS) <= set(X)
        ):
            raise ValueError(
                "Identity encoding requires both explicitly selected IDs and aligned keys"
            )
        first = dict(self.first_seen)
        result = X.copy(deep=True)
        for column in IDENTITY_COLUMNS:
            _integer_values(X[column], column, positive=True)
            ids = [int(v) for v in X[column]]
            if column == "team_id" and ids != rows.team_id.tolist():
                raise ValueError("Selected team_id differs from its context")
            labels = [
                f"team:{i}" if s == 8 or first.get(i, s) < s else "unknown"
                for i, s in zip(ids, rows.season_id, strict=True)
            ]
            if not set(labels) <= set(self.dtype.categories):
                raise ValueError("A known team is absent from the fixed identity vocabulary")
            result[column] = pd.Categorical(labels, dtype=self.dtype)
        return result

    def to_metadata(self) -> dict[str, object]:
        """JSON-compatible immutable history and the complete fixed category definition."""
        return {
            "team_ids": list(self.team_ids),
            "first_seen": [list(r) for r in self.first_seen],
            "categories": list(self.dtype.categories),
        }

    @classmethod
    def from_metadata(cls, metadata: object) -> Self:
        """Restore and validate the fixed category definition and original team history."""
        if not isinstance(metadata, dict) or set(metadata) != {
            "team_ids",
            "first_seen",
            "categories",
        }:
            raise ValueError("Invalid identity metadata")
        if (
            not isinstance(metadata["team_ids"], list)
            or not isinstance(metadata["first_seen"], list)
            or any(not isinstance(r, list) or len(r) != 2 for r in metadata["first_seen"])
        ):
            raise ValueError("Invalid identity vocabulary or history")
        result = cls(
            tuple(metadata["team_ids"]), tuple((r[0], r[1]) for r in metadata["first_seen"])
        )
        if metadata["categories"] != list(result.dtype.categories):
            raise ValueError("Identity categories disagree with the fixed vocabulary")
        return result
