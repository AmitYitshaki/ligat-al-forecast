"""Metric-independent paired fold evidence and explicitly versioned rule 8."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import t  # type: ignore[import-untyped]

_FOLD_COLUMNS = ("fold", "season_id", "games", "deep_fold")


@dataclass(frozen=True)
class PairedStatistics:
    """Reference minus candidate: positive means improvement in a lower-is-better score.

    The mean uses game counts; SE uses unweighted fold dispersion (ddof=1).
    Undefined quantities are None: no mean with zero deep folds, and no SE or
    interval with fewer than two. Recent means use deep seasons 21-25 only.
    """

    score: str
    differences: pd.DataFrame
    n: int
    mean_improvement: float | None
    se: float | None
    ci_lower: float | None
    ci_upper: float | None
    last5_folds: int
    last5_folds_better: int
    last5_improvement: float | None


@dataclass(frozen=True)
class Rule8Decision:
    """All conditions plus their evidence; passes=None means insufficient evidence.

    recent_mean_positive is not applicable in v1 and therefore None. A 95%
    interval is descriptive: passing rule 8 does not require excluding zero.
    """

    version: Literal[1, 2]
    stats: PairedStatistics
    mean_exceeds_se: bool | None
    recent_count_passes: bool | None
    recent_mean_positive: bool | None
    passes: bool | None
    status: Literal["passes", "fails", "insufficient_evidence"]


def _numeric(values: pd.Series, name: str, *, count: bool = False) -> None:
    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
        or values.isna().any()
    ):
        raise ValueError(f"{name} must contain finite real numbers")
    numbers = values.to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if count and ((numbers <= 0).any() or (numbers % 1 != 0).any()):
        raise ValueError(f"{name} must contain positive integer counts")


def _prepare_folds(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    if (
        not frame.columns.is_unique
        or not {*_FOLD_COLUMNS, score} <= set(frame)
        or score in _FOLD_COLUMNS
        or frame.empty
    ):
        raise ValueError("Paired evidence requires fold, season_id, games, deep_fold and a score")
    result = frame.loc[:, [*_FOLD_COLUMNS, score]].copy(deep=True)
    if (
        result.fold.isna().any()
        or result.fold.duplicated().any()
        or result.season_id.duplicated().any()
    ):
        raise ValueError("Each validation season must have exactly one distinct nonmissing fold")
    for column in ("season_id", "games"):
        _numeric(result[column], column, count=True)
    _numeric(result[score], score)
    if not pd.api.types.is_bool_dtype(result.deep_fold.dtype) or result.deep_fold.isna().any():
        raise ValueError("deep_fold must contain complete boolean flags")
    return result.sort_values("season_id").reset_index(drop=True)


def paired_statistics(
    candidate: pd.DataFrame, reference: pd.DataFrame, *, score: str
) -> PairedStatistics:
    """Pair identical fold populations and compute evidence only on their deep folds.

    Callers declare fold depth; it is never inferred from a fixed fold count.
    Row order is irrelevant. Fold IDs, seasons, game counts and depth flags must
    agree, including shallow folds. This pure core has no model or data I/O.
    """
    a, b = (_prepare_folds(frame, score) for frame in (candidate, reference))
    if len(a) != len(b) or any(
        not np.array_equal(a[column].to_numpy(), b[column].to_numpy()) for column in _FOLD_COLUMNS
    ):
        raise ValueError("Paired configurations require identical folds, seasons, games and depth")
    deep = a.deep_fold
    differences = a.loc[deep, list(_FOLD_COLUMNS)].reset_index(drop=True)
    differences["candidate_score"] = a.loc[deep, score].to_numpy(float)
    differences["reference_score"] = b.loc[deep, score].to_numpy(float)
    differences["difference"] = differences.reference_score - differences.candidate_score
    _numeric(differences.difference, "Paired score differences")
    d = differences.difference
    n = len(d)
    mean = float(np.average(d, weights=differences.games)) if n else None
    se = float(d.std(ddof=1) / np.sqrt(n)) if n >= 2 else None
    lower = upper = None
    if se is not None:
        assert mean is not None
        margin = float(t.ppf(0.975, df=n - 1)) * se
        lower, upper = mean - margin, mean + margin
    recent = differences.season_id.between(21, 25)
    return PairedStatistics(
        score,
        differences,
        n,
        mean,
        se,
        lower,
        upper,
        int(recent.sum()),
        int(d.loc[recent].gt(0).sum()),
        float(np.average(d.loc[recent], weights=differences.loc[recent, "games"]))
        if recent.any()
        else None,
    )


def rule8(stats: PairedStatistics, *, version: Literal[1, 2]) -> Rule8Decision:
    """Apply v1 (4/5) or v2 (3/5 and positive recent mean), with strict inequalities.

    Fewer than two deep folds or fewer than five deep recent seasons yield
    insufficient_evidence, never an automatic failure or a fabricated SE.
    The nested stats retain n, both means, SE, interval, count and fold details.
    """
    if (
        isinstance(version, bool)
        or not isinstance(version, (int, np.integer))
        or version not in (1, 2)
    ):
        raise ValueError("Rule 8 requires an explicit integer version 1 or 2")
    mean_ok = (
        stats.mean_improvement > stats.se
        if stats.mean_improvement is not None and stats.se is not None
        else None
    )
    count_ok = (
        stats.last5_folds_better >= (4 if version == 1 else 3) if stats.last5_folds == 5 else None
    )
    recent_ok = (
        stats.last5_improvement > 0
        if version == 2 and stats.last5_folds == 5 and stats.last5_improvement is not None
        else None
    )
    if stats.n < 2 or stats.last5_folds < 5:
        return Rule8Decision(
            version, stats, mean_ok, count_ok, recent_ok, None, "insufficient_evidence"
        )
    passed = bool(mean_ok and count_ok and (version == 1 or recent_ok))
    return Rule8Decision(
        version, stats, mean_ok, count_ok, recent_ok, passed, "passes" if passed else "fails"
    )
