"""Read-only notebook 03 C diagnostics from frozen context and EXP-003 artifacts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import AnalysisTables
from ligat.analysis.value_methodology import (
    correlation_summary,
    direction_correlations,
    match_ppm,
    paired_difference_summary,
    paired_value_correlations,
    value_shares,
)
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.features.pipeline import load_prematch_config
from ligat.features.registry import SEASON_CONTEXT_COLUMNS
from ligat.models.ordered_logit import OrderedFit, ordered_probabilities

WORDS = "מה השיטה עושה, במילים"
METRICS = {
    "prev_value_share_avg": "נתח שווי ממוצע אשתקד",
    "prev_value_share_total": "נתח שווי כולל אשתקד",
    "prev_position_pct": "מיקום יחסי אשתקד",
    "prev_ppm": "נקודות למשחק אשתקד",
    "prev_gd_per_game": "הפרש שערים למשחק אשתקד",
}
CANDIDATES = {
    "value_avg": "יחס שווי ממוצע",
    "value_total": "יחס שווי כולל",
    "position": "יחס מיקום: עולה 0.75",
    "position_mean": "יחס מיקום: ממוצע עולות",
    "ppm": "פער נקודות למשחק",
    "gd": "פער הפרש שערים למשחק",
    "value_avg+position": "שווי ממוצע ומיקום יחד",
}
EXPLANATIONS = {
    "value_avg": "log היחס בין נתחי השווי הממוצע של הבית והחוץ בעונה הקודמת",
    "value_total": "log היחס בין נתחי השווי הכולל של הבית והחוץ בעונה הקודמת",
    "position": "log יחס המיקום הספורטיבי היחסי; לעולה ערך קבוע 0.75",
    "position_mean": "log יחס המיקום; לעולה ממוצע של עולות מעונות קודמות בלבד",
    "ppm": "נקודות למשחק אשתקד של הבית פחות החוץ, ללא חצייה",
    "gd": "הפרש שערים למשחק אשתקד של הבית פחות החוץ",
    "value_avg+position": "שני משתנים נפרדים: log יחס השווי הממוצע ו-log יחס המיקום",
}
DEVELOPMENT = set(range(8, 26))


@dataclass(frozen=True)
class ContextReview:
    """Verified frozen data and fit parameters; no model fitting or block rebuilding."""

    context: pd.DataFrame
    identity: pd.DataFrame
    audit: pd.DataFrame
    folds: pd.DataFrame
    fits: pd.DataFrame
    coefficients: pd.DataFrame


def _check(path: Path, digest: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"C review checksum mismatch: {path}")


def _dev(frame: pd.DataFrame) -> None:
    if frame.season_id.isna().any() or not set(frame.season_id) <= DEVELOPMENT:
        raise ValueError("Context review requires development seasons 8–25")


def load_context_review(root: Path) -> ContextReview:
    """Verify EXP-003 and all pinned block identities, then read saved data only."""
    directory = root / "reports/experiments/EXP-003/full"
    manifest = verify_calibration(directory / "metadata.json")
    if manifest["training_start"] != 9 or manifest["fold_rows"] != 487:
        raise ValueError("Unexpected EXP-003 declaration")
    for item in manifest["inputs"].values():
        path = root / item["path"]
        if not path.resolve().is_relative_to((root / "data/derived").resolve()):
            raise ValueError("C review inputs must be frozen derived artifacts")
        _check(path, item["sha256"])
        _check(path.with_suffix(".metadata.json"), item["metadata_sha256"])
    context_path = root / manifest["inputs"]["context"]["path"]
    metadata = json.loads(context_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    state = load_prematch_config(root / "config/prematch.yaml").inputs["table_state"]
    _check(root / state.path, state.sha256)
    _check((root / state.path).with_suffix(".metadata.json"), state.metadata_sha256)
    review = ContextReview(
        pd.read_parquet(context_path),
        pd.read_parquet(root / state.path),
        pd.DataFrame(metadata["audit"]),
        pd.read_parquet(directory / "fold_scores.parquet"),
        pd.read_parquet(directory / "fits.parquet"),
        pd.read_parquet(directory / "coefficients.parquet"),
    )
    if (
        not review.context.index.equals(review.identity.index)
        or tuple(review.context.columns) != SEASON_CONTEXT_COLUMNS
    ):
        raise ValueError("C review context/identity keys or columns differ")
    for frame in (review.identity, review.audit, review.folds, review.fits, review.coefficients):
        _dev(frame)
    if len(review.fits) != 487 or not review.fits.converged.all():
        raise ValueError("C review requires all 487 converged saved fits")
    return review


def context_schema() -> pd.DataFrame:
    """Nine cold-block columns and the approved newcomer policies, without recomputation."""
    descriptions = {
        **{key: value for key, value in METRICS.items()},
        "promoted_in": "הקבוצה עלתה לליגה לקראת העונה הנוכחית",
        "squad_value_imputed": "ערך השווי הושלם לפי הכלל הקפוא",
        "prev_season_known": "קיימים נתוני עונה קודמת; כבוי בעונה 8",
        "league_size": "מספר הקבוצות בעונת המשחק",
    }
    policies = {
        "prev_value_share_avg": "אחוזון 30 בחצי התחתון של נתחי השווי הממוצע בליגת t−1",
        "prev_value_share_total": "אחוזון 30 בחצי התחתון של נתחי השווי הכולל בליגת t−1",
        "prev_position_pct": "0.75 קבוע",
        "prev_ppm": "ממוצע עולות מעונות קודמות בלבד",
        "prev_gd_per_game": "ממוצע עולות קודמות: שערי זכות פחות שערי חובה למשחק",
        "promoted_in": "דולק לעולה, כולל קבוצה שחזרה לליגה",
        "squad_value_imputed": "דולק גם לוותיקה ששווי המקור שלה חסר",
        "prev_season_known": "דולק מעונה 9; חמשת הערכים המספריים חסרים בעונה 8",
        "league_size": "זהה לכל הקבוצות באותה עונה",
    }
    return pd.DataFrame(
        [
            {"column": c, WORDS: descriptions[c], "newcomer_policy": policies[c]}
            for c in SEASON_CONTEXT_COLUMNS
        ]
    )


def probe_reference() -> pd.DataFrame:
    """Quoted C1-A findings only: never load the season-26 probe files again."""
    return pd.DataFrame(
        [
            {
                "date": "15.5.2025",
                "clubs": 11,
                "spearman": 0.945455,
                "source": "nb03_C1_value_check.md §A",
                WORDS: "דף העונה מול הדף המתוארך לסוף העונה; נתון מצוטט",
            },
            {
                "date": "1.9.2025",
                "clubs": 11,
                "spearman": 0.772727,
                "source": "nb03_C1_value_check.md §A",
                WORDS: "דף העונה מול הדף אחרי הקיץ; נתון מצוטט",
            },
        ]
    )


def team_season_context(review: ContextReview, tables: AnalysisTables) -> pd.DataFrame:
    """One row per club-season, paired to current unhalved 3/1/0 points per game."""
    _dev(tables.match_table)
    frame = review.context.join(review.identity[["season_id"]]).reset_index()
    grouped = frame.groupby(["season_id", "team_id"])
    if not grouped[list(SEASON_CONTEXT_COLUMNS)].nunique(dropna=False).le(1).all().all():
        raise ValueError("Season context must be constant within each club-season")
    result = grouped.first().reset_index()
    result = result.merge(
        match_ppm(tables.match_table),
        on=["season_id", "team_id"],
        validate="one_to_one",
        how="left",
    )
    if result.ppm.isna().any():
        raise ValueError("Missing current-season sporting ppm")
    names = tables.team_seasons.reset_index()[["season_id", "team_id", "team_name"]]
    return result.merge(names, on=["season_id", "team_id"], validate="one_to_one", how="left")


def context_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pooled club-season Spearman, explicitly not weighted by matches."""
    _dev(frame)
    return (
        pd.DataFrame(
            [
                {
                    "metric": column,
                    "spearman": float(frame[column].corr(frame.ppm, method="spearman")),
                    "teams": int(frame[[column, "ppm"]].dropna().shape[0]),
                    WORDS: f"{label} מול נקודות למשחק בעונה הנוכחית; שורה לכל קבוצה־עונה",
                }
                for column, label in METRICS.items()
            ]
        )
        .sort_values("spearman", key=lambda values: values.abs(), ascending=False)
        .reset_index(drop=True)
    )


def context_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Signed Spearman between the five prior-season measurements on club-season rows."""
    _dev(frame)
    return frame.loc[:, list(METRICS)].corr(method="spearman")


def promotion_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Equal weight to every promotion event, not to every game played by a newcomer."""
    _dev(frame)
    promoted = frame.loc[frame.promoted_in]
    return pd.DataFrame(
        [
            {
                "method": "ערך ההשלמה לעולה",
                "ppm": float(promoted.prev_ppm.mean()),
                "teams": len(promoted),
                WORDS: "ממוצע ערכי הפתיחה ההיסטוריים של העולות, שנקבעו לפני כל עונה",
            },
            {
                "method": "העולות בפועל",
                "ppm": float(promoted.ppm.mean()),
                "teams": len(promoted),
                WORDS: "ממוצע נקודות 3/1/0 למשחק שהעולות השיגו באותה עונה",
            },
        ]
    )


def validate_context_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail at the declared 0.001 discrepancy before presenting C2 reference findings."""
    expected = {
        "prev_gd_per_game": 0.589,
        "prev_position_pct": -0.587,
        "prev_value_share_avg": 0.581,
        "prev_value_share_total": 0.559,
        "prev_ppm": 0.549,
    }
    result = context_associations(frame)
    result["expected"] = result.metric.map(expected)
    result["difference"] = result.spearman - result.expected
    result["passed"] = result.difference.abs().le(0.001)
    if not result.passed.all():
        raise ValueError("C2 Spearman reference discrepancy exceeds 0.001; stop and report")
    promoted = promotion_check(frame)
    if promoted.teams.tolist() != [37, 37] or not np.allclose(
        promoted.ppm, [1.12, 1.09], atol=0.005, rtol=0
    ):
        raise ValueError("C2 promoted ppm/count reference discrepancy; stop and report")
    return result


def value_methodology_tables(tables: AnalysisTables) -> dict[str, pd.DataFrame]:
    """Recompute development-only C1 summaries; stop if they differ from the C1 report."""
    _dev(tables.match_table)
    values = tables.team_seasons.reset_index()
    _dev(values)
    performance = match_ppm(tables.match_table)
    summary = correlation_summary(direction_correlations(values, performance))
    reference = {
        "share_total_s vs ppm_s-1": 0.656283,
        "share_total_s vs ppm_s+0": 0.627631,
        "share_total_s vs ppm_s+1": 0.500400,
        "ppm_s vs ppm_s+1": 0.579626,
    }
    if set(summary.comparison) != set(reference) or not np.allclose(
        summary.mean_spearman, summary.comparison.map(reference), atol=5e-7, rtol=0
    ):
        raise ValueError("C1 direction summary differs from the approved report; stop and report")
    labels = {
        "share_total_s vs ppm_s-1": "שווי s מול נקודות בעונה הקודמת",
        "share_total_s vs ppm_s+0": "שווי s מול נקודות באותה עונה",
        "share_total_s vs ppm_s+1": "שווי s מול נקודות בעונה הבאה",
        "ppm_s vs ppm_s+1": "נקודות s מול נקודות בעונה הבאה",
    }
    summary[WORDS] = summary.comparison.map(labels)
    paired = paired_value_correlations(values, performance)
    difference = paired_difference_summary(paired)
    if (
        not np.allclose(difference[["mean", "se"]], [[-0.057942, 0.017294]], atol=5e-7, rtol=0)
        or int(paired.difference.lt(0).sum()) != 15
    ):
        raise ValueError("C1 average/total contrast differs from report; stop and report")
    difference["mean"] *= -1  # Present average minus total; original C1 uses the reverse sign.
    difference["better_seasons"] = int(paired.difference.lt(0).sum())
    difference[WORDS] = "תוספת ספירמן לממוצע על פני הסכום, מול נקודות למשחק בעונה הבאה"
    paired["improvement"] = -paired.difference
    paired[WORDS] = "השוואה מזווגת על אותם מועדונים בשתי העונות; חיובי לטובת ממוצע השווי"
    # The named source anomaly is descriptive only; no values are changed.
    names = values.team_name.astype(str)
    clubs = names.str.contains("מכבי חיפה", regex=False) | (
        names.str.contains("הפועל", regex=False) & names.str.contains('ת"א', regex=False)
    )
    shares = value_shares(values)
    shares["value_rank"] = shares.groupby("season_id").share_avg.rank(ascending=False)
    anomaly = shares.loc[values.season_id.isin([9, 10, 11]) & clubs].copy()
    if len(anomaly) != 6:
        raise ValueError("Expected two source-anomaly clubs in three development seasons")
    anomaly["total_million"] = anomaly.total_value_eur.astype(float) / 1e6
    anomaly["average_million"] = anomaly.avg_value_eur.astype(float) / 1e6
    anomaly[WORDS] = "ערך המקור ללא תיקון; הצגה מול העונות הסמוכות בלבד"
    return {
        "direction": summary,
        "paired": paired,
        "paired_summary": difference,
        "anomaly": anomaly[
            ["season_id", "team_name", "total_million", "average_million", "value_rank", WORDS]
        ],
    }


def experiment_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predeclared routes and candidates, without an experimental run or a new choice."""
    routes = pd.DataFrame(
        [
            {
                "route": "A",
                "method": "לבד",
                "metric": "log loss",
                WORDS: "המדד לבדו ברגרסיה סדורה; ייחוסים B0 ו-Elo",
            },
            {
                "route": "O",
                "method": "מעל Elo",
                "metric": "log loss",
                WORDS: "Elo/400 והמדד ברגרסיה סדורה",
            },
            {
                "route": "P",
                "method": "מעל מודל השערים",
                "metric": "Poisson NLL",
                WORDS: "בית, Elo, התקפה והגנה EWMA5/venue והמדד במודל Poisson",
            },
        ]
    )
    formulas = [
        "log(value_avg_home / value_avg_away)",
        "log(value_total_home / value_total_away)",
        "log(position_home / position_away)",
        "log(position_mean_home / position_mean_away)",
        "ppm_home − ppm_away",
        "gd_home − gd_away",
        "[log(value_avg_home/value_avg_away), log(position_home/position_away)]",
    ]
    candidates = pd.DataFrame(
        [
            {"candidate": c, "method": CANDIDATES[c], "formula": formula, WORDS: EXPLANATIONS[c]}
            for c, formula in zip(CANDIDATES, formulas, strict=True)
        ]
    )
    return routes, candidates


def _folds(review: ContextReview, config: str) -> pd.DataFrame:
    result = review.folds.loc[review.folds.config_id.eq(config)].set_index("season_id").sort_index()
    if result.empty or not result.index.is_unique:
        raise ValueError(f"Missing/nonunique saved folds: {config}")
    return result


def _score(review: ContextReview, config: str, recent: bool = False) -> float:
    frame = _folds(review, config)
    if recent:
        frame = frame.loc[frame.index >= 21]
    metric = "poisson_nll" if config.startswith("P_") else "log_loss"
    return float(
        np.average(frame[metric], weights=frame.rows if metric == "poisson_nll" else frame.games)
    )


def _paired(
    review: ContextReview, reference: str, candidate: str, seasons: list[int] | None = None
) -> dict[str, Any]:
    base, own = _folds(review, reference), _folds(review, candidate)
    if seasons is not None:
        base, own = base.loc[seasons], own.loc[seasons]
    if (
        not base.index.equals(own.index)
        or not base.games.equals(own.games)
        or not base.rows.equals(own.rows)
    ):
        raise ValueError("C review comparisons must use identical validation games")
    metric = "poisson_nll" if candidate.startswith("P_") else "log_loss"
    differences = base[metric] - own[metric]
    weights = base.rows if metric == "poisson_nll" else base.games
    return {
        "improvement": float(np.average(differences, weights=weights)),
        "se": float(differences.std(ddof=1) / np.sqrt(len(differences))),
        "better": int(differences.gt(0).sum()),
        "folds": len(differences),
    }


def reference_ruler(review: ContextReview) -> pd.DataFrame:
    """Saved train-from-9 references and sourced official train-from-8 anchors."""
    rows = [
        {
            "method": name,
            "training_start": 9,
            "metric": metric,
            "score": _score(review, config),
            "recent_score": _score(review, config, True),
            WORDS: description,
        }
        for name, config, metric, description in [
            ("B0", "A_b0", "log loss", "שכיחויות בית/תיקו/חוץ מעונות האימון"),
            ("Elo", "O_elo", "log loss", "Elo בלבד באותם קיפולים, אימון מעונה 9"),
            ("Poisson", "P_poisson", "Poisson NLL", "EWMA5/venue עם Elo, אימון מעונה 9"),
        ]
    ]
    rows += [
        {
            "method": "E0050 הרשמי",
            "training_start": 8,
            "metric": "log loss",
            "score": 1.020423847325146,
            WORDS: "ייחוס EXP-001 השמור, שמתחיל אימון בעונה 8",
        },
        {
            "method": "Poisson הרשמי",
            "training_start": 8,
            "metric": "Poisson NLL",
            "score": 1.4094552099479694,
            WORDS: "ייחוס EXP-002 השמור, שמתחיל אימון בעונה 8",
        },
    ]
    return pd.DataFrame(rows)


def alone_results(review: ContextReview) -> pd.DataFrame:
    """The seven A candidates, on the original 15 and latest five saved folds."""
    result = pd.DataFrame(
        [
            {
                "candidate": c,
                "method": CANDIDATES[c],
                "score": _score(review, "A_" + c),
                "recent_score": _score(review, "A_" + c, True),
                WORDS: EXPLANATIONS[c],
            }
            for c in CANDIDATES
        ]
    )
    result["rank_all"] = result.score.rank(method="min").astype(int)
    result["rank_last5"] = result.recent_score.rank(method="min").astype(int)
    return result.sort_values("score").reset_index(drop=True)


def added_results(review: ContextReview) -> pd.DataFrame:
    """Paired reference-minus-candidate means and SE; last five remain descriptive."""
    records = []
    for route in ("O", "P"):
        reference = route + ("_elo" if route == "O" else "_poisson")
        for candidate in CANDIDATES:
            config = route + "_" + candidate
            primary = _paired(review, reference, config)
            recent = _paired(review, reference, config, list(range(21, 26)))
            records.append(
                {
                    "route": route,
                    "candidate": candidate,
                    "method": CANDIDATES[candidate],
                    **primary,
                    "recent_improvement": recent["improvement"],
                    "recent_better": recent["better"],
                    "adds": primary["improvement"] > primary["se"],
                    WORDS: EXPLANATIONS[candidate],
                }
            )
    return pd.DataFrame(records)


def direct_decisions(review: ContextReview) -> pd.DataFrame:
    """Report the four approved comparisons; no independent feature selection."""
    records = []
    for route in ("O", "P"):
        for default, alternative in (("value_avg", "value_total"), ("position", "position_mean")):
            primary = _paired(review, route + "_" + default, route + "_" + alternative)
            recent = _paired(
                review, route + "_" + default, route + "_" + alternative, list(range(21, 26))
            )
            condition = primary["improvement"] > primary["se"]
            records.append(
                {
                    "route": route,
                    "method": CANDIDATES[default] + " מול " + CANDIDATES[alternative],
                    **primary,
                    "condition": condition,
                    "recent_improvement": recent["improvement"],
                    "disagreement": condition != (recent["improvement"] > 0),
                    "approved": "ממוצע נשאר" if default == "value_avg" else "0.75 נשאר",
                    WORDS: "החלופה נדרשת לשפר ביותר מ-SE ב-15; בחמש האחרונות משווים ציונים בלבד",
                }
            )
    return pd.DataFrame(records)


def value_fold_gaps(review: ContextReview, *, sensitivity: bool = False) -> pd.DataFrame:
    """Every saved value-average gap, plus the actual number of training seasons."""
    records = []
    suffix = "_no_s11" if sensitivity else ""
    for route in ("O", "P"):
        candidate = route + "_value_avg" + suffix
        reference = route + ("_elo" if route == "O" else "_poisson") + suffix
        a, b = _folds(review, reference), _folds(review, candidate)
        metric = "poisson_nll" if route == "P" else "log_loss"
        fits = review.fits.loc[review.fits.config_id.eq(candidate)].set_index("season_id")
        for season in a.index:
            records.append(
                {
                    "route": route,
                    "season_id": int(season),
                    "improvement": float(a.at[season, metric] - b.at[season, metric]),
                    "training_seasons": len(fits.at[season, "training_seasons"]),
                    WORDS: "ייחוס פחות שווי ממוצע, באותם משחקים ובעונות האימון השמורות",
                }
            )
    return pd.DataFrame(records)


def selected_coefficients(review: ContextReview) -> pd.DataFrame:
    """Selected folds with and without value: coefficients are read, never re-fitted."""
    records = []
    for route in ("O", "P"):
        for value in (False, True):
            config = route + ("_value_avg" if value else "_elo" if route == "O" else "_poisson")
            part = review.coefficients.loc[review.coefficients.config_id.eq(config)].pivot(
                index="season_id", columns="coefficient", values="value"
            )
            for season in (11, 12, 18, 25):
                records.append(
                    {
                        "route": route,
                        "season_id": season,
                        "method": "עם שווי" if value else "ללא שווי",
                        "elo_coefficient": float(part.at[season, "b_elo"]),
                        "value_coefficient": float(part.at[season, "z_value_avg"])
                        if value
                        else np.nan,
                        WORDS: "מקדמי הקיפול שנאמדו במקור; השווי ו-Elo חולקים מידע",
                    }
                )
    return pd.DataFrame(records)


def sensitivity_overview(review: ContextReview) -> pd.DataFrame:
    """The same missing-season policy for reference and candidate, including its first fold."""
    gaps = value_fold_gaps(review, sensitivity=True)
    rows = []
    for route in ("O", "P"):
        reference = route + ("_elo_no_s11" if route == "O" else "_poisson_no_s11")
        summary = _paired(review, reference, route + "_value_avg_no_s11")
        first = gaps.loc[gaps.route.eq(route)].sort_values("season_id").iloc[0]
        rows.append(
            {
                "route": route,
                **summary,
                "first_season": int(first.season_id),
                "first_training": int(first.training_seasons),
                "first_improvement": float(first.improvement),
                WORDS: "עונה 11 הוסרה גם מהאימון וגם מהולידציה; הייחוס מאומן באותו תנאי",
            }
        )
    return pd.DataFrame(rows)


def training_depth_table(review: ContextReview) -> pd.DataFrame:
    """Post-hoc thresholds from EXPERIMENTS; explicitly not a new predeclared experiment."""
    records = []
    for route, candidate in (
        ("P", "value_avg"),
        ("P", "value_total"),
        ("P", "value_avg+position"),
        ("P", "gd"),
        ("O", "value_avg"),
        ("O", "gd"),
    ):
        config = route + "_" + candidate
        reference = route + ("_elo" if route == "O" else "_poisson")
        fits = review.fits.loc[review.fits.config_id.eq(config)].set_index("season_id")
        row: dict[str, Any] = {
            "route": route,
            "candidate": candidate,
            "method": CANDIDATES[candidate],
            WORDS: EXPLANATIONS[candidate],
        }
        for minimum in (2, 4, 6, 8):
            seasons = [int(s) for s in fits.index if len(fits.at[s, "training_seasons"]) >= minimum]
            row.update(
                {
                    f"mean_{minimum}": _paired(review, reference, config, seasons)["improvement"],
                    f"se_{minimum}": _paired(review, reference, config, seasons)["se"],
                    f"folds_{minimum}": len(seasons),
                }
            )
        recent = _paired(review, reference, config, list(range(21, 26)))
        row.update(recent_improvement=recent["improvement"], recent_better=recent["better"])
        records.append(row)
    return pd.DataFrame(records)


def gd_translation(review: ContextReview) -> pd.DataFrame:
    """C3b home-quartile illustration using medians of saved ordinal parameters only."""
    home = review.identity.loc[review.identity.is_home & review.identity.season_id.ge(11)]
    keys = pd.MultiIndex.from_arrays([home.index.get_level_values("game_id"), home.opponent_id])
    gap = review.context.loc[home.index, "prev_gd_per_game"].to_numpy(
        float
    ) - review.context.reindex(keys).prev_gd_per_game.to_numpy(float)
    low, high = np.quantile(gap, [0.25, 0.75])
    records = []
    for route in ("A", "O"):
        fits = review.fits.loc[review.fits.config_id.eq(route + "_gd")]
        extra = (
            (float(np.median([values[0] for values in fits.additional_betas])),)
            if route == "O"
            else ()
        )
        model = OrderedFit(
            float(fits.beta.median()),
            float(fits.theta1.median()),
            float(fits.delta.median()),
            0,
            True,
            0,
            False,
            0,
            extra,
        )
        design = np.array([[low], [high]]) if route == "A" else np.array([[0, low], [0, high]])
        probabilities = ordered_probabilities(model, design)[:, 0] * 100
        records.append(
            {
                "route": route,
                "q25": low,
                "q75": high,
                "home_win_low": probabilities[0],
                "home_win_high": probabilities[1],
                "change_pp": probabilities[1] - probabilities[0],
                WORDS: "מעבר בין רבעוני פער הפרש השערים אשתקד; ב-O מחזיקים Elo שווה",
            }
        )
    return pd.DataFrame(records)


def control_numbers(review: ContextReview) -> pd.DataFrame:
    """Primary notebook anchors, checked against the rounded approved C3b report."""
    pairs = added_results(review)
    items = [
        ("B0", _score(review, "A_b0"), 1.08486, 5e-6),
        ("Elo", _score(review, "O_elo"), 1.02076, 5e-6),
        ("Poisson", _score(review, "P_poisson"), 1.41010, 5e-6),
        (
            "P value_avg dbar",
            float(
                pairs.loc[
                    pairs.route.eq("P") & pairs.candidate.eq("value_avg"), "improvement"
                ].iloc[0]
            ),
            -0.00124,
            5e-6,
        ),
        ("התאמות", float(len(review.fits)), 487.0, 0.0),
    ]
    result = pd.DataFrame(
        [
            {
                "quantity": name,
                "value": value,
                "expected": expected,
                "passed": abs(value - expected) <= tolerance,
                WORDS: "מספר בקרה מחושב מהתוצרים השמורים מול דוח C3b",
            }
            for name, value, expected, tolerance in items
        ]
    )
    if not result.passed.all():
        raise ValueError("C3b notebook control discrepancy; stop and report")
    return result
