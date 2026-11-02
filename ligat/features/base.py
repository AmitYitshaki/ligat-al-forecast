"""Small interface for independently computed prematch feature blocks."""

from typing import Protocol

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables


class FeatureBlock(Protocol):
    """Produce two prematch rows per game, indexed by (game_id, team_id)."""

    name: str
    football_rationale: str

    def compute(self, tables: AnalysisTables) -> pd.DataFrame:
        """Compute each target using results from strictly earlier calendar dates."""
        ...
