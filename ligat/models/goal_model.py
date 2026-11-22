"""Row-wise M2 contract; the judge controls refitting and target visibility."""

from typing import Literal, Protocol, Self

import pandas as pd  # type: ignore[import-untyped]

RefitPolicy = Literal["per_fold", "per_matchday"]


class TeamGoalModel(Protocol):
    """One positive finite lambda per indexed team-game row, preserving its order."""

    @property
    def refit_policy(self) -> RefitPolicy:
        """Declare whether the judge fits once per fold or before every matchday."""
        ...

    @property
    def supports_sample_weight(self) -> bool:
        """Explicitly opt in to consuming every supplied observation weight."""
        ...

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
    ) -> Self:
        """Fit aligned past rows; reject non-None weights if unsupported, never ignore them."""
        ...

    def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
        """Predict without receiving observed goals for any requested fixture."""
        ...


class MatchdayGoalModel(Protocol):
    """Opt-in daily fit receives explicit prediction time and target-free fixture features."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        prediction_features: pd.DataFrame | None = None,
    ) -> Self: ...
