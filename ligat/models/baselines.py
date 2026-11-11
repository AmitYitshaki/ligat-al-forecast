"""B0 uses H/D/A frequencies from the current training fold, without smoothing."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ligat.evaluate.metrics import validate_outcomes


class B0:
    """Predict the empirical outcome distribution of training games only."""

    def __init__(self) -> None:
        self._probabilities: NDArray[np.float64] | None = None

    def fit(self, train: pd.DataFrame) -> None:
        """Estimate three frequencies from this fit call's outcome codes."""
        y = validate_outcomes(train.outcome.to_numpy(), len(train))
        self._probabilities = np.bincount(y, minlength=3) / len(y)

    def predict(self, games: pd.DataFrame) -> NDArray[np.float64]:
        """Repeat the fitted distribution without reading any validation target."""
        if self._probabilities is None:
            raise ValueError("B0 must be fitted before predict")
        return np.tile(self._probabilities, (len(games), 1))
