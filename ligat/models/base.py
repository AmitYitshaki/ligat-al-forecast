"""Minimal probability-predictor contract shared by the evaluation judge."""

from typing import Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray


class Predictor(Protocol):
    """Fit on past labeled games; predict H/D/A in the given games' row order."""

    def fit(self, train: pd.DataFrame) -> None:
        """Fit using the supplied training games, including their observed outcomes."""
        ...

    def predict(self, games: pd.DataFrame) -> NDArray[np.float64]:
        """Return n×3 H/D/A probabilities; games contain no match outcomes."""
        ...
