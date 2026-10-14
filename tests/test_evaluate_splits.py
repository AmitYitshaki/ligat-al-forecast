"""Temporal fold boundaries and the two single final evaluations."""

import pandas as pd
import pytest

from ligat.data.repository import Repository
from ligat.evaluate.splits import expanding_window_folds


class CountsRepository(Repository):
    def read_filtered(self, name: str, filters: list) -> pd.DataFrame:
        assert name == "matches"
        assert 28 not in filters[0][2]
        return pd.DataFrame({"season_id": [s for s in range(8, 28) for _ in range(s)]})


def test_expanding_folds_and_final_boundaries() -> None:
    folds = expanding_window_folds(repository=CountsRepository())
    assert len(folds) == 17
    assert folds.iloc[0].train_seasons == (8, 9, 10)
    assert folds.iloc[0].validation_season == 11
    assert folds.iloc[14].train_seasons == tuple(range(8, 25))
    assert folds.iloc[14].validation_season == 25
    assert folds.iloc[15].fold == "final_validation"
    assert folds.iloc[15].train_seasons == tuple(range(8, 26))
    assert folds.iloc[16].fold == "test"
    assert folds.iloc[16].train_seasons == tuple(range(8, 27))
    for row in folds.itertuples():
        assert set(row.train_seasons).isdisjoint([row.validation_season])
        assert max(row.train_seasons) < row.validation_season
        assert row.n_train_matches == sum(row.train_seasons)
        assert row.n_validation_matches == row.validation_season


@pytest.mark.parametrize(
    "seasons,minimum", [([8, 8, 9], 1), ([8, 26], 1), ([8, 9], 0), ([8, 9], 2)]
)
def test_invalid_fold_inputs(seasons: list[int], minimum: int) -> None:
    with pytest.raises(ValueError):
        expanding_window_folds(seasons, minimum, CountsRepository())
