"""Readable M2 row metrics and calibration, without fitting reference models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goal_selection import _check_fold_scores, _team_rows
from ligat.evaluate.selection import _numeric
from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation


@dataclass(frozen=True)
class ReadableGoalMetrics:
    """Three scopes of pooled row metrics and exact better/worse/tie rates by season."""

    summary: pd.DataFrame
    by_season: pd.DataFrame


def _goal_rows(evaluation: TeamGoalEvaluation) -> pd.DataFrame:
    rows = _team_rows(evaluation, "poisson_nll")
    if not {"lambda", "is_home"} <= set(rows):
        raise ValueError("Goal reporting requires lambda and is_home for every team row")
    if not pd.api.types.is_bool_dtype(rows.is_home.dtype) or rows.is_home.isna().any():
        raise ValueError("is_home must contain complete boolean values")
    if not rows.groupby(level="game_id").is_home.sum().eq(1).all():
        raise ValueError("Goal reporting requires one home and one away row per game")
    if not rows.season_id.isin(range(8, 26)).all():
        raise ValueError("Goal reporting only accepts development seasons 8-25")
    _numeric(rows["lambda"], "lambda")
    _numeric(rows.poisson_nll, "poisson_nll")
    expected = poisson_nll(rows.goals_scored.to_numpy(float), rows["lambda"].to_numpy(float))
    if rows.poisson_nll.lt(0).any() or not np.allclose(
        rows.poisson_nll, expected, rtol=0, atol=1e-12
    ):
        raise ValueError("Stored Poisson NLL differs from the supplied goals and lambdas")
    _check_fold_scores(evaluation, rows, "poisson_nll")
    return rows


def _scopes(rows: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=rows.index),
        "deep": rows.deep_fold,
        "last5": rows.season_id.between(21, 25),
    }


def _row_rates(a: pd.Series, b: pd.Series) -> dict[str, int | float | None]:
    better, worse, tied = int(a.lt(b).sum()), int(a.gt(b).sum()), int(a.eq(b).sum())
    n = len(a)
    return {
        "rows": n,
        "better_rows": better,
        "worse_rows": worse,
        "tie_rows": tied,
        "better_percent": 100 * better / n if n else None,
        "worse_percent": 100 * worse / n if n else None,
        "tie_percent": 100 * tied / n if n else None,
        "better_non_tie_percent": 100 * better / (better + worse) if better + worse else None,
    }


def readable_goal_metrics(
    candidate: TeamGoalEvaluation,
    reference: TeamGoalEvaluation,
    *,
    r0: TeamGoalEvaluation,
    r1: TeamGoalEvaluation,
) -> ReadableGoalMetrics:
    """Report exp(-mean NLL), Elo contribution, row win rates, MAE and pooled RMSE.

    All percentages are on a 0-100 scale. R0/R1 are supplied evaluations, never
    fitted here. Keys, observed goals, home side and fold/season must agree.
    Candidate scope masks apply to all four models on exactly the same rows;
    baselines may have different training depths. Empty scopes have no metric
    values. An Elo denominator <= 0 yields None and elo_share_defined=False.
    Only paired fold differences receive confidence intervals, via selection.py.
    """
    a, b, zero, one = (_goal_rows(e) for e in (candidate, reference, r0, r1))
    for other in (b, zero, one):
        if not a.index.equals(other.index) or any(
            not np.array_equal(a[c].to_numpy(), other[c].to_numpy())
            for c in ("goals_scored", "is_home", "fold", "season_id")
        ):
            raise ValueError("Readable metrics require identical keys, goals, sides and folds")
    records = []
    for scope, mask in _scopes(a).items():
        part = a.loc[mask]
        loss = float(part.poisson_nll.mean()) if len(part) else None
        l0, l1 = (
            float(f.loc[mask, "poisson_nll"].mean()) if len(part) else None for f in (zero, one)
        )
        defined = l0 is not None and l1 is not None and l0 > l1
        errors = part.goals_scored.to_numpy(float) - part["lambda"].to_numpy(float)
        records.append(
            {
                "scope": scope,
                "poisson_nll": loss,
                "observed_probability_percent": 100 * float(np.exp(-loss))
                if loss is not None
                else None,
                "r0_nll": l0,
                "r1_nll": l1,
                "elo_share_percent": 100 * (l0 - loss) / (l0 - l1)
                if defined and l0 is not None and l1 is not None and loss is not None
                else None,
                "elo_share_defined": defined,
                "mae": float(np.abs(errors).mean()) if len(part) else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(part) else None,
                **_row_rates(part.poisson_nll, b.loc[mask, "poisson_nll"]),
            }
        )
    by_season = []
    for (fold, season), part in a.groupby(["fold", "season_id"], sort=True):
        by_season.append(
            {
                "fold": fold,
                "season_id": season,
                "deep_fold": bool(part.deep_fold.iloc[0]),
                **_row_rates(part.poisson_nll, b.loc[part.index, "poisson_nll"]),
            }
        )
    return ReadableGoalMetrics(pd.DataFrame(records).set_index("scope"), pd.DataFrame(by_season))


def calibration_deciles(evaluation: TeamGoalEvaluation) -> pd.DataFrame:
    """Up to ten lambda-quantile groups per scope (all and last5), never split ties.

    Constant lambdas form one group. Duplicate quantile edges are dropped, and
    empty bins omitted. Bin IDs are consecutive within each scope. Calibration
    compares lambda means with actual goal means, counting every team row once.
    """
    rows = _goal_rows(evaluation)
    records = []
    for scope, mask in _scopes(rows).items():
        if scope == "deep":
            continue
        part = rows.loc[mask].copy()
        if part.empty:
            continue
        if part["lambda"].nunique() == 1:
            part["bin"] = 1
        else:
            bins = pd.qcut(part["lambda"], q=10, labels=False, duplicates="drop")
            part["bin"] = pd.factorize(bins, sort=True)[0] + 1
        for number, group in part.groupby("bin", sort=True):
            records.append(
                {
                    "scope": scope,
                    "bin": number,
                    "n": len(group),
                    "lambda_mean": float(group["lambda"].mean()),
                    "goals_mean": float(group.goals_scored.mean()),
                    "lambda_min": float(group["lambda"].min()),
                    "lambda_max": float(group["lambda"].max()),
                }
            )
    return pd.DataFrame(records)
