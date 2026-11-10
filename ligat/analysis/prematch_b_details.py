"""Small R2 diagnostics from frozen EXP-001 outputs, without fitting or replay."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from ligat.analysis.prematch_b_review import (
    OUTCOMES,
    WORDS,
    EloReview,
    calibration_bins,
    ordinal_probabilities,
)
from ligat.models.ordered_logit import time_weights
from ligat.viz.seasons import season_label

BASE_PARAMETERS = [
    "k",
    "home_advantage",
    "season_regression",
    "use_goal_multiplier",
    "promoted_initialization",
    "dead_game_discount",
]


def configuration_contrast(row: pd.Series) -> str:
    """Describe only deviations from E0050, without redefining the saved additions count."""
    changes = []
    if row.k != 10:
        changes.append(f"תגובה K={row.k:g} במקום 10")
    if row.home_advantage != 30:
        changes.append(f"ביתיות בעדכון {row.home_advantage:g} במקום 30")
    if row.season_regression != 0.33:
        changes.append(f"מחיקה בקיץ {100 * row.season_regression:g}% במקום 33%")
    if not row.use_goal_multiplier:
        changes.append("ללא מכפיל הפרש שערים")
    if row.promoted_initialization != "fixed_1450":
        changes.append("אתחול לפי ממוצע עולות קודמות")
    if row.dead_game_discount != 1:
        changes.append(f"משקל {row.dead_game_discount:g} במשחק מת")
    if not pd.isna(row.half_life):
        changes.append(f"דעיכה: המשקל נחצה כל {row.half_life:g} עונות")
    return "; ".join(changes) or "E0050: בסיס ההשוואה, ללא שינוי נוסף"


def interaction_table(rankings: pd.DataFrame) -> pd.DataFrame:
    """Average coarse-grid loss within each K/regression cell, not its best member."""
    coarse = rankings.loc[rankings.half_life.isna()].copy()
    if len(coarse) != 1280 or coarse.config_id.duplicated().any():
        raise ValueError("Expected the 1280 unique coarse configurations")
    result = (
        coarse.groupby(["k", "season_regression"])
        .agg(log_loss=("log_loss", "mean"), configs=("config_id", "size"))
        .reset_index()
    )
    result["best_in_row"] = result.log_loss.eq(result.groupby("k").log_loss.transform("min"))
    result[WORDS] = "ממוצע על כל יתר פרמטרי הרשת; לא הציון הטוב ביותר במשבצת"
    return result


def decay_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold 15 season weights and their sum; effective seasons, not a game ESS."""
    seasons = np.arange(8, 25)
    rows: list[dict[str, object]] = []
    totals: list[dict[str, object]] = []
    for half_life in (None, 1.0, 2.0, 4.0):
        label = (
            "בלי דעיכה"
            if half_life is None
            else "המשקל נחצה כל עונה"
            if half_life == 1
            else f"המשקל נחצה כל {half_life:g} עונות"
        )
        weights = time_weights(seasons, 25, half_life)
        totals.append(
            {
                "decay": label,
                "effective_seasons": float(weights.sum()),
                WORDS: "סכום משקל אחד לכל עונת אימון; העונה האחרונה שווה 1, לא גודל מדגם משחקים",
            }
        )
        rows.extend(
            {
                "season_id": int(season),
                "season": season_label(int(season)),
                "decay": label,
                "weight": float(weight),
            }
            for season, weight in zip(seasons, weights, strict=True)
        )
    return pd.DataFrame(rows), pd.DataFrame(totals)


def decay_differences(configurations: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Mean the ten paired base differences within each fold, then use 15-fold SE.

    Validation games weight the mean. SE uses the unweighted sample SD of the
    fifteen fold differences / sqrt(15); configurations are not extra replicates.
    """
    configs = configurations.copy()
    configs["decay_key"] = configs.half_life.fillna(0)
    if len(configs) != 40 or configs.duplicated(BASE_PARAMETERS + ["decay_key"]).any():
        raise ValueError("Expected four decay options for each of ten unique Elo bases")
    merged = scores.merge(
        configs[["config_id", *BASE_PARAMETERS, "decay_key"]],
        on="config_id",
        validate="many_to_one",
    )
    if len(merged) != 600 or merged.duplicated(BASE_PARAMETERS + ["decay_key", "fold"]).any():
        raise ValueError("Expected 40 configurations on the same 15 folds")
    keys = [*BASE_PARAMETERS, "fold", "season_id", "games"]
    baseline = merged.loc[merged.decay_key.eq(0), keys + ["log_loss"]].rename(
        columns={"log_loss": "reference_loss"}
    )
    pairs = merged.merge(baseline, on=keys, validate="many_to_one")
    if len(pairs) != len(merged):
        raise ValueError("Decay comparisons do not share validation populations")
    pairs["difference"] = pairs.log_loss - pairs.reference_loss
    rows = []
    for key, group in pairs.groupby("decay_key", sort=True):
        if group.groupby("fold").size().ne(10).any() or group.fold.nunique() != 15:
            raise ValueError("Decay comparison is missing a base/fold pair")
        folds = group.groupby("fold").agg(
            difference=("difference", "mean"), games=("games", "first")
        )
        rows.append(
            {
                "decay_key": float(key),
                "decay": (
                    "בלי דעיכה"
                    if key == 0
                    else "המשקל נחצה כל עונה"
                    if key == 1
                    else f"המשקל נחצה כל {key:g} עונות"
                ),
                "difference": float(np.average(folds.difference, weights=folds.games)),
                "se": float(folds.difference.std(ddof=1) / np.sqrt(15)),
                "log_loss": float(group.log_loss.mul(group.games).sum() / group.games.sum()),
                WORDS: "הפסד עם דעיכה פחות בלי דעיכה; שלילי טוב יותר; SE בין קיפולים מזווגים",
            }
        )
    return pd.DataFrame(rows)


def probability_curves(review: EloReview) -> pd.DataFrame:
    """Use the saved last-fold thresholds for raw home-minus-away rating gaps."""
    fit = review.tables["proposal_converter_effect"].sort_values("season_id").iloc[-1]
    return pd.DataFrame(
        [
            {
                "gap": gap,
                **dict(
                    zip(
                        OUTCOMES,
                        ordinal_probabilities(gap, fit.beta, fit.theta1, fit.theta2),
                        strict=True,
                    )
                ),
            }
            for gap in range(-300, 301, 5)
        ]
    )


def calibration_error(predictions: pd.DataFrame) -> pd.DataFrame:
    """ECE with exactly the graph's ten equal-width bins, weighted by bin count."""
    bins = calibration_bins(predictions, bins=10)
    bins["error_mass"] = bins.games * (bins.predicted - bins.observed).abs()
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "ece": float(group.error_mass.sum() / group.games.sum()),
                "games": int(group.games.sum()),
                WORDS: "המרחק הממוצע בין ההסתברות שנחזתה לשכיחות בפועל, במשקל מספר המשחקים בבין",
            }
            for outcome, group in bins.groupby("outcome", sort=False)
        ]
    )


def draw_peaks(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Analytic maximum of the logistic mass within each fold's fixed draw interval."""
    result = coefficients[["fold", "season_id", "beta", "theta1", "theta2"]].copy()
    if (result.beta <= 0).any() or (result.theta2 <= result.theta1).any():
        raise ValueError("Draw peaks require positive beta and ordered finite thresholds")
    result["width"] = result.theta2 - result.theta1
    result["peak_probability"] = 2 * expit(result.width / 2) - 1
    result["peak_gap"] = 400 * (result.theta1 + result.theta2) / (2 * result.beta)
    result[WORDS] = [
        f"תקרה {100 * row.peak_probability:.1f}%; "
        f"התיקו מרבי כשהבית חלש ב־{abs(row.peak_gap):.1f} נקודות Elo"
        for row in result.itertuples()
    ]
    return result


def alignment_trends(alignment: pd.DataFrame) -> pd.DataFrame:
    """Descriptive Spearman trends after excluding the common-1500 first season."""
    subset = alignment.loc[alignment.season_id.ne(8)]
    rows = []
    for metric in ("spearman", "exact_percent"):
        correlation, pvalue = spearmanr(subset.season_id, subset[metric])
        rows.append(
            {
                "metric": metric,
                "correlation": float(correlation),
                "p_value": float(pvalue),
                "seasons": len(subset),
                WORDS: "קשר בין מספר העונה למדד ההתאמה, בלי העונה הראשונה; לא ראיה לסיבתיות",
            }
        )
    return pd.DataFrame(rows)


def gap_case_explanations(gaps: pd.DataFrame, prematch: pd.DataFrame) -> pd.DataFrame:
    """Distinguish actual promoted teams from returning teams with a low opening rating."""
    context = prematch.reset_index()[["season_id", "team_id", "promoted_in"]].drop_duplicates()
    if context.duplicated(["season_id", "team_id"]).any():
        raise ValueError("Promoted status changes within a team-season")
    frame = gaps.merge(context, on=["season_id", "team_id"], validate="one_to_one")
    if len(frame) != len(gaps):
        raise ValueError("Missing promoted status for a gap case")
    frame["crosses_cut"] = frame.position.le(6).ne(frame.elo_rank.le(6))
    frame["הסבר כדורגלני"] = [
        "; ".join(
            [
                *(["המיקום הספורטיבי ודירוג Elo משני צדי קו 6/7"] if row.crosses_cut else []),
                "עולה באותה עונה שהתחילה ב-1450; דירוג איטי מגיב בהדרגה"
                if row.promoted_in
                else "דירוג הפתיחה נושא זיכרון מהעונה הקודמת, אחרי נסיגה בקיץ",
            ]
        )
        for row in frame.itertuples()
    ]
    return frame


def choice_precision(matrix: pd.DataFrame) -> pd.DataFrame:
    """Accuracy among games chosen as each category; distinct from row-normalized recall."""
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "chosen": int(matrix[outcome].sum()),
                "correct": int(matrix.at[outcome, outcome]),
                "precision": float(matrix.at[outcome, outcome] / matrix[outcome].sum())
                if matrix[outcome].sum()
                else np.nan,
                WORDS: "מתוך המשחקים שבהם argmax בחר בקטגוריה הזו, איזה חלק אכן הסתיים בה",
            }
            for outcome in OUTCOMES
        ]
    )
