"""Read-only, checksum-verified diagnostics for notebook 03, B1 (EXP-001)."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.features.elo import expected_home_score, goal_difference_multiplier
from ligat.viz.seasons import season_label

E0050 = "elo-b24989debf11c0ba4af09aa247bd264ad754349b8d0db4ba88e9c440fd00e950"
WORDS = "מה השיטה עושה, במילים"
OUTCOMES = ("בית", "תיקו", "חוץ")


@dataclass(frozen=True)
class EloReview:
    """Verified saved tables; no model fitting or Elo replay is performed."""

    tables: dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    prematch: pd.DataFrame
    execution: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _check(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Saved artifact checksum mismatch: {path}")


def load_table_state_snapshot(artifact: Path, *, metadata_sha256: str) -> pd.DataFrame:
    """Validate both immutable v2 files before reading the saved feature frame."""
    metadata_path = artifact.with_suffix(".metadata.json")
    _check(metadata_path, metadata_sha256)
    metadata = _json(metadata_path)
    if metadata["version"] != "2" or set(metadata["seasons"]) != set(range(8, 26)):
        raise ValueError("Expected the development table_state version 2 snapshot")
    _check(artifact, metadata["artifact_sha256"])
    return pd.read_parquet(artifact)


def assert_same_table_state(computed: pd.DataFrame, saved: pd.DataFrame) -> None:
    """Compare exact values/dtypes, normalizing only Parquet's integer index type."""
    frames = []
    for frame in (computed, saved):
        normalized = frame.copy()
        normalized.index = pd.MultiIndex.from_arrays(
            [frame.index.get_level_values(name).astype("int64") for name in frame.index.names],
            names=frame.index.names,
        )
        frames.append(normalized)
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def load_elo_review(root: Path) -> EloReview:
    """Verify manifests before opening any CSV/Parquet; reject held-out seasons."""
    directory = root / "reports/experiments/EXP-001"
    manifest = _json(directory / "manifest.json")
    if manifest.get("status") != "completed_not_selected":
        raise ValueError("Unexpected EXP-001 saved-run status")
    # The root run manifest has a status, whereas child manifests have completed.
    # Keep the frozen formats and verifier intact and check both layers.
    for name, info in manifest["files"].items():
        _check(directory / name, info["sha256"])
    for stage in ("stage1", "stage2", "combined", "loso"):
        verify_calibration(directory / stage / "metadata.json")
    elo_path = root / "data/derived/elo/elo_E0050_prematch.parquet"
    elo_meta = _json(elo_path.with_suffix(".metadata.json"))
    _check(elo_path, elo_meta["artifact_sha256"])
    if elo_meta["config_id"] != E0050 or set(elo_meta["seasons"]) != set(range(8, 26)):
        raise ValueError("E0050 artifact must cover development seasons 8–25 only")
    names = (
        "labels",
        "all_rankings",
        "top20",
        "close_configurations",
        "stage1_marginals",
        "stage2_table",
        "b0_folds",
        "loso_vs_walk_forward",
        "proposal_converter_effect",
        "proposal_sporting_alignment",
        "proposal_1_all_teams",
        "proposal_1_gaps_4plus",
    )
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        filename = f"{name}.csv"
        if filename not in manifest["files"]:
            raise ValueError(f"Unverified review input: {filename}")
        tables[name] = pd.read_csv(directory / filename)
    tables["folds"] = pd.read_csv(directory / "stage1/fold_scores.csv")
    tables["folds"] = tables["folds"].loc[lambda x: x.config_id == E0050].copy()
    tables["stage2_folds"] = pd.read_csv(directory / "stage2/fold_scores.csv")
    prediction_path = root / "data/derived/elo/exp001_predictions_stage1.parquet"
    if not any(
        (directory / name).resolve() == prediction_path.resolve() for name in manifest["files"]
    ):
        raise ValueError("Predictions missing from verified manifest")
    predictions = pd.read_parquet(prediction_path, filters=[("config_id", "==", E0050)])
    prematch = pd.read_parquet(elo_path)
    for frame in [*tables.values(), predictions, prematch]:
        if "season_id" in frame and not set(frame.season_id) <= set(range(8, 26)):
            raise ValueError("Review input contains a held-out season")
    if "execution.json" not in manifest["files"]:
        raise ValueError("Unverified execution summary")
    return EloReview(tables, predictions, prematch, _json(directory / "execution.json"))


def configuration_words(row: pd.Series) -> str:
    """Explain all calibrated choices, including the absence of extra weighting."""
    multiplier = "מכפיל לפי הפרש השערים" if row.use_goal_multiplier else "בלי מכפיל הפרש"
    newcomer = (
        "1450 קבוע לעולה" if row.promoted_initialization == "fixed_1450" else "ממוצע עולות קודמות"
    )
    half_life = row.half_life
    time = (
        "כל עונות האימון במשקל שווה"
        if pd.isna(half_life)
        else f"משקל עונה נחצה כל {half_life:g} עונות"
    )
    return (
        f"גודל התגובה למשחק K={row.k:g}; תוספת בית {row.home_advantage:g} בעדכון; "
        f"בקיץ נמחקים {100 * row.season_regression:g}% מהפער מ־1500; {multiplier}; "
        f"{newcomer}; במשחק מת משקל {row.dead_game_discount:g}; {time}"
    )


def configurations(review: EloReview, name: str, *, limit: int | None = None) -> pd.DataFrame:
    """Give saved configurations readable labels and a mandatory words column."""
    frame = review.tables[name].head(limit).copy()
    frame = frame.merge(review.tables["labels"], on="config_id", validate="one_to_one", sort=False)
    frame[WORDS] = frame.apply(configuration_words, axis=1)
    return frame


def top_configuration_contrasts(review: EloReview) -> pd.DataFrame:
    """Expose differences between the ten saved leaders; common settings go above."""
    frame = configurations(review, "top20", limit=10)
    common = (
        frame.k.eq(10)
        & frame.season_regression.eq(0.33)
        & frame.use_goal_multiplier.eq(True)
        & frame.promoted_initialization.eq("fixed_1450")
    )
    if not common.all():
        raise ValueError("Saved top-ten common settings have changed; review the explanation")
    differences, explanations = [], []
    for row in frame.itertuples():
        changes, words = [], []
        if not pd.isna(row.half_life):
            changes.append(f"דעיכה: המשקל נחצה כל {row.half_life:g} עונות")
            words.append("ההמרה מדגישה עונות עדכניות")
        if row.dead_game_discount != 1:
            changes.append(f"משקל {row.dead_game_discount:g} במשחק מת")
            words.append("משחק בלי יעד מזיז את הדירוג פחות")
        if row.home_advantage != 30:
            changes.append(f"יתרון בית בעדכון {row.home_advantage:g}")
            words.append("הבית מקבל תוספת דירוג גדולה יותר בחישוב הציפייה לעדכון")
        differences.append("; ".join(changes) or "ללא משקל זמן או הנחה; יתרון בית 30")
        explanations.append("; ".join(words) or "כל משחק במלוא משקלו; כל עונות האימון במשקל שווה")
    frame["מה מייחד אותה"] = differences
    frame[WORDS] = explanations
    return frame


def grid_overview(review: EloReview) -> pd.DataFrame:
    """Describe the grid that was actually run, with fixed meanings per axis."""
    descriptions = {
        "k": ("K", "גודל עדכון: ערך קטן מגיב פחות למשחק יחיד"),
        "home_advantage": ("יתרון בית", "תוספת לדירוג הבית בחישוב E בעדכון בלבד"),
        "season_regression": ("נסיגה בקיץ", "החלק מהפער מול 1500 שנמחק בתחילת עונה"),
        "use_goal_multiplier": ("מכפיל הפרש", "האם ניצחון בהפרש גדול מזיז יותר את הדירוג"),
        "promoted_initialization": ("אתחול עולה", "1450 קבוע או ממוצע Elo בסיום של עולות קודמות"),
        "dead_game_discount": ("משקל משחק מת", "מקטין עדכון כשאין יעד פתוח לפחות לאחת הקבוצות"),
    }
    translations = {
        "False": "בלי",
        "True": "עם",
        "fixed_1450": "1450 קבוע",
        "previous_promoted_mean": "ממוצע עולות קודמות",
    }
    rows = []
    for key, (name, description) in descriptions.items():
        marginal = review.tables["stage1_marginals"]
        values = marginal.loc[marginal.parameter.eq(key), "value"]
        rows.append(
            {
                "פרמטר": name,
                "ערכים שנבדקו": ", ".join(translations.get(str(x), str(x)) for x in values),
                WORDS: description,
            }
        )
    rows.append(
        {
            "פרמטר": "דעיכת משקל עונה",
            "ערכים שנבדקו": "ללא; המשקל נחצה כל 1, 2 או 4 עונות",
            WORDS: "שלב שני בלבד, על אותם 10 בסיסי Elo המובילים; בלי שינוי בסיס ה־Elo",
        }
    )
    return pd.DataFrame(rows)


def selection_comparison(review: EloReview) -> pd.DataFrame:
    """The all-fold leader, last-five leader and approved simplest proposal."""
    frame = configurations(review, "all_rankings")
    chosen_ids = {E0050, frame.iloc[0].config_id, frame.sort_values("rank_last5").iloc[0].config_id}
    return frame.loc[frame.config_id.isin(chosen_ids)].copy()


def elo_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A numerical update and the production goal-margin rule, without replay."""
    expected = expected_home_score(1500, 1500, 30)
    delta = 10 * goal_difference_multiplier(2) * (1 - expected)
    example = pd.DataFrame(
        [
            {
                "מפגש": "שתי קבוצות 1500; ניצחון בית 2:0",
                "E": expected,
                "Δ": delta,
                "בית אחרי": 1500 + delta,
                "חוץ אחרי": 1500 - delta,
                "בקיץ: דירוג 1600": 1500 + (1 - 0.33) * 100,
            }
        ]
    )
    multipliers = pd.DataFrame(
        {"הפרש שערים": range(6), "מכפיל G": [goal_difference_multiplier(n) for n in range(6)]}
    )
    return example, multipliers


def ordinal_probabilities(gap: float, beta: float, theta1: float, theta2: float) -> np.ndarray:
    """Evaluate saved ordered-logit parameters, in H/D/A order; never refit."""
    if not theta1 < theta2 or not np.isfinite([gap, beta, theta1, theta2]).all():
        raise ValueError("Invalid ordinal illustration parameters")
    away = float(expit(theta1 - beta * gap / 400))
    draw_or_away = float(expit(theta2 - beta * gap / 400))
    return np.array([1 - draw_or_away, draw_or_away - away, away])


def conversion_examples(review: EloReview) -> pd.DataFrame:
    last = review.tables["proposal_converter_effect"].sort_values("season_id").iloc[-1]
    rows = []
    for gap, title in (
        (0, "דירוג שווה"),
        (90, "חיפה 1540 מול עולה 1450 — דוגמה היפותטית"),
        (100, "תוספת 100 נקודות לבית"),
    ):
        probabilities = ordinal_probabilities(gap, last.beta, last.theta1, last.theta2)
        rows.append(
            {"דוגמה": title, "פער Elo": gap, **dict(zip(OUTCOMES, probabilities, strict=True))}
        )
    return pd.DataFrame(rows)


def fold_plan(review: EloReview) -> pd.DataFrame:
    frame = review.tables["folds"].sort_values("season_id").copy()
    frame["אימון"] = frame.training_seasons.map(lambda value: ast.literal_eval(value)).map(
        lambda seasons: f"{season_label(min(seasons))} עד {season_label(max(seasons))}"
    )
    frame["ולידציה"] = frame.season_id.map(lambda sid: season_label(int(sid)))
    return frame[["fold", "אימון", "ולידציה", "games"]]


def score_example() -> pd.DataFrame:
    """One game, fixed illustrative probabilities, and the normalized RPS."""
    probability = np.array([0.58, 0.25, 0.17])
    rows = []
    for outcome in range(3):
        actual = np.eye(3)[outcome]
        rps = float(np.mean((np.cumsum(probability)[:2] - np.cumsum(actual)[:2]) ** 2))
        rows.append(
            {
                "מה קרה": OUTCOMES[outcome],
                "log_loss": -math.log(probability[outcome]),
                "rps": rps,
                WORDS: "אותה תחזית: 58% בית, 25% תיקו, 17% חוץ; בודקים כל תוצאה אפשרית בנפרד",
            }
        )
    return pd.DataFrame(rows)


def score_ruler(review: EloReview) -> pd.DataFrame:
    chosen = review.tables["all_rankings"].set_index("config_id").loc[E0050]
    b0 = review.execution["b0_summary"]
    rows = [
        ("⅓ לכל תוצאה", math.log(3), "בית, תיקו וחוץ בהסתברות שווה"),
        ("B0", float(b0["log_loss"]), "שכיחויות תוצאות בעונות האימון בלבד; בלי זהות הקבוצות"),
        ("E0050", float(chosen.log_loss), configuration_words(chosen)),
        ("סף חשד לדליפה", 0.98, "מתחת לסף בממוצע הקיפולים עוצרים; Elo שאחרי המשחק קיבל 0.968"),
    ]
    return pd.DataFrame(
        [
            {
                "שיטה": name,
                "log_loss": loss,
                "ממוצע גאומטרי של ההסתברות לתוצאה שקרתה": math.exp(-loss),
                WORDS: words,
            }
            for name, loss, words in rows
        ]
    )


def season_scores(review: EloReview) -> pd.DataFrame:
    frame = review.tables["folds"][["season_id", "games", "log_loss", "rps"]].merge(
        review.tables["b0_folds"][["season_id", "log_loss"]],
        on="season_id",
        suffixes=("_elo", "_b0"),
        validate="one_to_one",
    )
    frame["season"] = frame.season_id.map(lambda sid: season_label(int(sid)))
    return frame.sort_values("season_id")


def calibration_bins(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Fixed equal-width bins, retaining only populated bins and their counts."""
    if bins < 1:
        raise ValueError("bins must be positive")
    rows = []
    for code, column in enumerate(("p_home", "p_draw", "p_away")):
        probabilities = predictions[column].to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Invalid saved probabilities")
        indices = np.minimum((probabilities * bins).astype(int), bins - 1)
        for index in np.unique(indices):
            mask = indices == index
            rows.append(
                {
                    "outcome": OUTCOMES[code],
                    "bin": int(index),
                    "games": int(mask.sum()),
                    "predicted": float(probabilities[mask].mean()),
                    "observed": float((predictions.outcome.to_numpy()[mask] == code).mean()),
                }
            )
    return pd.DataFrame(rows)


def prediction_diagnostics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Argmax is a diagnostic only; draws remain in the probability distribution."""
    probabilities = predictions[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Saved probabilities must sum to one")
    actual = predictions.outcome.to_numpy(dtype=int)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    summary = pd.DataFrame(
        [
            {
                "games": len(actual),
                "accuracy": float((predicted == actual).mean()),
                "max_draw": float(probabilities[:, 1].max()),
                "argmax_draws": int((predicted == 1).sum()),
                "observed_draw": float((actual == 1).mean()),
                "mean_draw": float(probabilities[:, 1].mean()),
            }
        ]
    )
    return summary, pd.DataFrame(matrix, index=OUTCOMES, columns=OUTCOMES)


def alignment_summary(all_teams: pd.DataFrame) -> pd.DataFrame:
    difference = all_teams.abs_difference
    return pd.DataFrame(
        [
            {
                "קבוצות־עונות": len(all_teams),
                "מקום זהה": float((difference == 0).mean()),
                "עד מקום אחד": float((difference <= 1).mean()),
                "עד שני מקומות": float((difference <= 2).mean()),
            }
        ]
    )


def control_numbers(review: EloReview) -> pd.DataFrame:
    chosen = review.tables["all_rankings"].set_index("config_id").loc[E0050]
    alignment = alignment_summary(review.tables["proposal_1_all_teams"]).iloc[0]
    return pd.DataFrame(
        [
            {
                "E0050 log loss": chosen.log_loss,
                "E0050 RPS": chosen.rps,
                "B0 log loss": review.execution["b0_summary"]["log_loss"],
                "מקום זהה (%)": alignment["מקום זהה"] * 100,
                "עד ±1 (%)": alignment["עד מקום אחד"] * 100,
                "עד ±2 (%)": alignment["עד שני מקומות"] * 100,
                WORDS: (
                    "תצורת E0050: Elo והמרה סדורה; B0: שכיחויות מעונות אימון; "
                    "אחוזי המקום: דירוג סוף עונה מול מיקום ספורטיבי, בדיעבד"
                ),
            }
        ]
    )
