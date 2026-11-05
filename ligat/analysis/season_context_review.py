"""Audit-only sporting/official season-end rank comparison, outside the feature block."""

from collections.abc import Mapping

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.features.season_context import completed_sporting_positions
from ligat.simulate.rules import SeasonRules


def season_position_comparison(
    tables: AnalysisTables, rules: Mapping[int, SeasonRules], final_tables: pd.DataFrame
) -> pd.DataFrame:
    """Count different ranks for every development season; official tables are audit-only."""
    if not set(final_tables.season_id) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Position review requires development seasons 8–25")
    sporting = completed_sporting_positions(tables, rules).reset_index()
    comparison = sporting.merge(
        final_tables[["season_id", "team_id", "position"]],
        on=["season_id", "team_id"],
        how="left",
        suffixes=("_sporting", "_official"),
        validate="one_to_one",
    )
    if comparison.position_official.isna().any():
        raise ValueError("Missing official rank for position audit")
    comparison["different"] = comparison.position_sporting.ne(comparison.position_official)
    return comparison.groupby("season_id", as_index=False).agg(
        teams=("team_id", "size"), different_positions=("different", "sum")
    )
