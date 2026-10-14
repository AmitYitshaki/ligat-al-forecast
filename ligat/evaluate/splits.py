"""Approved expanding-window plan; generates a schedule, never fits a model."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, FINAL_VALIDATION_SEASON, TEST_SEASON
from ligat.data.repository import Repository


def expanding_window_folds(
    seasons: range | list[int] = DEVELOPMENT_SEASONS,
    min_train_seasons: int = 3,
    repository: Repository | None = None,
) -> pd.DataFrame:
    """15 default development folds, then single final-validation and test evaluations.

    Hyperparameters are selected once using validation losses weighted by game count
    across development folds. The final two rows are not tuning folds.
    """
    selected = sorted(seasons)
    if len(set(selected)) != len(selected) or not set(selected) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Fold seasons must be distinct development seasons 8–25")
    if min_train_seasons < 1 or min_train_seasons >= len(selected):
        raise ValueError("Minimum training seasons must leave at least one development fold")
    requested = [*selected, FINAL_VALIDATION_SEASON, TEST_SEASON]
    matches = (repository or Repository()).read_filtered(
        "matches", [("season_id", "in", requested)]
    )
    counts = matches.groupby("season_id").size().to_dict()
    rows: list[dict[str, Any]] = []
    plans: list[tuple[int | str, list[int], int]] = [
        (n - min_train_seasons + 1, selected[:n], selected[n])
        for n in range(min_train_seasons, len(selected))
    ]
    plans += [
        ("final_validation", selected, FINAL_VALIDATION_SEASON),
        ("test", [*selected, FINAL_VALIDATION_SEASON], TEST_SEASON),
    ]
    for fold, training, validation in plans:
        assert max(training) < validation and validation not in training
        rows.append(
            dict(
                fold=fold,
                train_seasons=tuple(training),
                validation_season=validation,
                n_train_matches=sum(counts.get(season, 0) for season in training),
                n_validation_matches=counts.get(validation, 0),
            )
        )
    return pd.DataFrame(rows)
