"""Read-only notebook 03 B2 diagnostics from frozen form and EXP-002 artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.analysis.prematch_b_review import WORDS
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.goal_metrics import poisson_outcomes

# C0 relocates predictions while retaining the original frozen manifest and hashes.
EXP002_PREDICTIONS_DIRECTORY = Path("data/derived/exp002")
SELECTED_P = "P_ewma5_venue_elo"
POINTS_EXAMPLE = "O_ewma20_all_elo"
FORM_METADATA_SHA256 = "b3f7a38f40d5db31f86695e3f1ba00788741b07422d4b6d06f993b5085c7b02c"
METRICS = {"gf": "שערי זכות", "ga": "שערי חובה", "pts": "נקודות"}
STRATA = {"all": "כל המשחקים", "home": "בית", "away": "חוץ"}
DISTINCT = "מה מייחד אותה"


@dataclass(frozen=True)
class FormReview:
    """Verified saved feature rows, fit parameters and validation diagnostics."""

    form: pd.DataFrame
    elo: pd.DataFrame
    metadata: dict[str, Any]
    identities: pd.DataFrame
    p_rankings: pd.DataFrame
    o_rankings: pd.DataFrame
    folds: pd.DataFrame
    fits: pd.DataFrame
    predictions: pd.DataFrame
    team_predictions: pd.DataFrame


def _check(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Review input checksum mismatch: {path}")


def load_form_review(root: Path) -> FormReview:
    """Verify saved inputs before reading frames; never read clean or replay models."""
    directory = root / "reports/experiments/EXP-002/full"
    prediction_directory = root / EXP002_PREDICTIONS_DIRECTORY
    manifest = verify_calibration(
        directory / "metadata.json",
        file_locations={
            name: prediction_directory / name
            for name in ("predictions.parquet", "team_predictions.parquet")
        },
    )
    for key in ("form", "elo"):
        item = manifest["inputs"][key]
        _check(root / item["path"], item["sha256"])
    form_path = root / manifest["inputs"]["form"]["path"]
    metadata_path = form_path.with_suffix(".metadata.json")
    _check(metadata_path, FORM_METADATA_SHA256)
    metadata = dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    if set(metadata["seasons"]) != set(range(8, 26)):
        raise ValueError("Form review requires development seasons 8–25")
    _check(form_path, metadata["artifact_sha256"])
    identities = []
    for item in json.loads((directory / "configurations.json").read_text(encoding="utf-8")):
        identities.append({**item, **(item["candidate"] or {})})
    result = FormReview(
        form=pd.read_parquet(form_path),
        elo=pd.read_parquet(root / manifest["inputs"]["elo"]["path"]),
        metadata=metadata,
        identities=pd.DataFrame(identities),
        p_rankings=pd.read_csv(directory / "comparison_P.csv"),
        o_rankings=pd.read_csv(directory / "comparison_O.csv"),
        folds=pd.read_parquet(directory / "fold_scores.parquet"),
        fits=pd.read_parquet(directory / "fits.parquet"),
        predictions=pd.read_parquet(
            prediction_directory / "predictions.parquet", filters=[("config_id", "==", SELECTED_P)]
        ),
        team_predictions=pd.read_parquet(
            prediction_directory / "team_predictions.parquet",
            filters=[("config_id", "==", SELECTED_P)],
        ),
    )
    for frame in (
        result.elo,
        result.folds,
        result.fits,
        result.predictions,
        result.team_predictions,
    ):
        if not set(frame.season_id) <= set(range(8, 26)):
            raise ValueError("Held-out season in B2 input")
    if not result.form.index.equals(result.elo.index):
        raise ValueError("Frozen form/Elo team-game rows do not align")
    return result


def opening_coefficients(review: FormReview) -> pd.DataFrame:
    """All nine causal rho values per season, including explicitly marked fallbacks."""
    return pd.DataFrame(review.metadata["coefficients"]).sort_values(
        ["season_id", "metric", "stratum"]
    )


def ewma_illustrations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure weights: age 1 is the last completed match; prior mass is separate."""
    weights, priors = [], []
    for half_life in (5, 10, 20):
        alpha = 1 - 0.5 ** (1 / half_life)
        label = f"EWMA {half_life}"
        row: dict[str, Any] = {
            "שיטה": label,
            WORDS: f"משקל משחק נחצה כל {half_life} משחקים באותה גרסה; מתחיל מערך הפתיחה",
        }
        row.update({f"משחק {age}": alpha * (1 - alpha) ** (age - 1) for age in (1, 5, 10, 20)})
        row["חלק מחמשת האחרונים"] = 1 - (1 - alpha) ** 5
        weights.append(row)
        priors.append(
            {
                "שיטה": label,
                WORDS: row[WORDS],
                **{f"אחרי {k} משחקים": (1 - alpha) ** k for k in (5, 10, 20, 30)},
            }
        )
    weights.append(
        {
            "שיטה": "חלון 5",
            WORDS: "חמשת המשחקים האחרונים שווים במשקל; לאחר חמישה משחקים אין משקל לפתיחה",
            **{f"משחק {age}": 0.2 if age <= 5 else 0.0 for age in (1, 5, 10, 20)},
            "חלק מחמשת האחרונים": 1.0,
        }
    )
    priors.append(
        {
            "שיטה": "חלון 5",
            WORDS: weights[-1][WORDS],
            **{f"אחרי {k} משחקים": 0.0 for k in (5, 10, 20, 30)},
        }
    )
    # Put the aggregate mass on a separate row, as requested, beside the four ages.
    by_method = pd.DataFrame(weights).set_index("שיטה")
    displayed = []
    for quantity in [*(f"משחק {age}" for age in (1, 5, 10, 20)), "חלק מחמשת האחרונים"]:
        displayed.append(
            {
                "מה נמדד": quantity,
                **{name: by_method.loc[name, quantity] for name in by_method.index},
                WORDS: (
                    "סכום משקלי חמש ההופעות האחרונות; יתרת המשקל נשארת לעבר ולפתיחה"
                    if quantity == "חלק מחמשת האחרונים"
                    else (
                        f"משקל התוצאה שהתקבלה לפני {quantity.split()[-1]} הופעות; "
                        "חלון 5 חותך את העבר בבת אחת"
                    )
                ),
            }
        )
    return pd.DataFrame(displayed), pd.DataFrame(priors)


def candidate_description(candidate: str, variant: str) -> tuple[str, str]:
    """Plain-language memory rule and venue distinction for the 22 methods."""
    if candidate.startswith("ewma"):
        memory = f"המשקל נחצה כל {candidate[4:]} הופעות"
        words = f"עדכון מעריכי: משקל משחק נחצה כל {candidate[4:]} משחקים"
    elif candidate.startswith("std_m"):
        memory = f"מתחילת העונה; פתיחה במשקל {candidate[5:]}"
        words = f"ממוצע כל משחקי העונה, עם ערך פתיחה במשקל {candidate[5:]} משחקים"
    elif candidate.startswith("w"):
        memory = f"חלון {candidate[1:]} משחקים"
        words = f"ממוצע {candidate[1:]} המשחקים האחרונים; הפתיחה ממלאת משחקים חסרים"
    elif candidate == "prev":
        memory, words = "העונה הקודמת בלבד", "ממוצע העונה הקודמת; לא מתעדכן במהלך העונה"
    elif candidate == "hist":
        memory, words = "כל העונות הקודמות", "ממוצע כל עונות העבר; לא מתעדכן במהלך העונה"
    else:
        raise ValueError("Unknown saved form candidate")
    venue = "בית וחוץ בנפרד" if variant == "venue" else "כל המגרשים יחד"
    if variant not in ("all", "venue"):
        raise ValueError("Unknown saved form variant")
    return f"{words}; {venue}", f"{memory}; {venue}"


def candidate_table(review: FormReview, route: str) -> pd.DataFrame:
    """Join saved rankings to paired, chronological-fold improvements over Elo."""
    if route not in ("P", "O"):
        raise ValueError("Route must be P or O")
    frame = (review.p_rankings if route == "P" else review.o_rankings).merge(
        review.identities[["config_id", "candidate", "variant"]],
        on="config_id",
        validate="one_to_one",
    )
    metric, count = ("poisson_nll", "rows") if route == "P" else ("log_loss", "games")
    reference = (
        review.folds.loc[review.folds.config_id.eq(f"{route}_reference_elo")]
        .set_index("season_id")
        .sort_index()
    )
    if reference.index.tolist() != list(range(11, 26)):
        raise ValueError("B2 comparisons require all 15 chronological development folds")
    for index, row in frame.iterrows():
        own = (
            review.folds.loc[review.folds.config_id.eq(row.config_id)]
            .set_index("season_id")
            .sort_index()
        )
        if not own.index.equals(reference.index) or not np.array_equal(
            own[count], reference[count]
        ):
            raise ValueError("Paired comparisons require the same folds and row counts")
        difference = reference[metric] - own[metric]
        improvement = float(np.average(difference, weights=reference[count]))
        se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        frame.loc[index, "improvement"] = improvement
        frame.loc[index, "improvement_se"] = se
        frame.loc[index, "z"] = improvement / se if se else np.nan
        frame.loc[index, "seasons_better"] = int((difference > 0).sum())
        words, distinct = candidate_description(row.candidate, row.variant)
        frame.loc[index, WORDS], frame.loc[index, DISTINCT] = words, distinct
    frame["method"] = frame.candidate + "/" + frame.variant
    frame["family"] = np.where(
        frame.candidate.isin(["prev", "hist"]), "עונות קודמות בלבד", "מידע מתוך העונה"
    )
    return frame.sort_values("rank_all")


def route_references(review: FormReview) -> pd.DataFrame:
    """Read weighted reference scores; P NLL and O log loss have different units."""
    rows = []
    descriptions = {
        "P_reference_no_elo": "Poisson: קבוע וביתיות בלבד, בלי זהות הקבוצה",
        "P_reference_elo": "Poisson: קבוע, ביתיות ופער Elo, בלי חלונות שערים",
        "O_reference_elo": "המרה סדורה של פער Elo בלבד — E0050",
    }
    for identifier, words in descriptions.items():
        frame = review.folds.loc[review.folds.config_id.eq(identifier)]
        is_p = identifier.startswith("P_")
        metric, count = ("poisson_nll", "rows") if is_p else ("log_loss", "games")
        recent = frame.loc[frame.season_id.ge(21)]
        rows.append(
            {
                "method": identifier,
                "metric": "NLL לקבוצה־משחק" if is_p else "log loss למשחק",
                "score": float(np.average(frame[metric], weights=frame[count])),
                "recent_score": float(np.average(recent[metric], weights=recent[count])),
                WORDS: words,
            }
        )
    return pd.DataFrame(rows)


def haifa_season(review: FormReview, *, team_id: int = 1005, season_id: int = 25) -> pd.DataFrame:
    """Join saved prematch features to saved actual goals; never forward-fill venues."""
    rows = review.team_predictions.loc[
        (review.team_predictions.team_id == team_id)
        & (review.team_predictions.season_id == season_id)
    ].copy()
    rows = rows.set_index(["game_id", "team_id"])
    opponent = review.predictions.set_index("game_id")
    rows["goals_against"] = np.where(
        rows.is_home,
        rows.index.get_level_values("game_id").map(opponent.away_goals),
        rows.index.get_level_values("game_id").map(opponent.home_goals),
    )
    frame = rows.join(review.elo[["as_of"]], validate="one_to_one").join(
        review.form[["gf_venue_ewma5", "ga_venue_ewma5", "gf_all_ewma20", "ga_all_ewma20"]],
        validate="one_to_one",
    )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Missing Haifa saved example rows")
    return frame.sort_values(["as_of", "game_id"])


def poisson_coefficients(review: FormReview, *, identifier: str = SELECTED_P) -> pd.DataFrame:
    """Translate saved coefficients at the exact predeclared gf=1.5→1.65 point."""
    fits = review.fits.loc[review.fits.config_id.eq(identifier)].sort_values("season_id")
    coefficients = opening_coefficients(review)
    league = (
        coefficients.loc[(coefficients.metric == "gf") & (coefficients.stratum == "all")]
        .set_index("season_id")
        .league_mean
    )
    counts = review.elo.season_id.value_counts()
    rows = []
    for fit in fits.itertuples():
        c, gamma, a, d, b = (float(value) for value in fit.coefficients)
        mean = float(
            np.average(
                [league[s + 1] for s in fit.training_seasons],
                weights=[counts[s] for s in fit.training_seasons],
            )
        )
        home = np.exp(c + gamma + a * np.log(1.6) + d * np.log(mean + 0.1))
        away = np.exp(c + (a + d) * np.log(mean + 0.1))
        ratio = (1.75 / 1.6) ** a
        probabilities, _ = poisson_outcomes([[home, away], [home * ratio, away]])
        rows.append(
            {
                "season_id": fit.season_id,
                "gamma": gamma,
                "a": a,
                "d": d,
                "b": b,
                "training_mean": mean,
                "attack_change_percent": 100 * (ratio - 1),
                "defense_change_percent": 100 * ((1.75 / 1.6) ** d - 1),
                "home_win_change_pp": 100 * (probabilities[1, 0] - probabilities[0, 0]),
                WORDS: (
                    f"ביתיות {100 * np.expm1(gamma):.2f}% לשערים; "
                    f"רגישות התקפה {a:.3f}, הגנה {d:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def points_coefficients(review: FormReview) -> pd.DataFrame:
    """The saved simplicity proposal ewma20/all, not the different O leader."""
    frame = review.fits.loc[review.fits.config_id.eq(POINTS_EXAMPLE)].copy()
    frame["beta_form"] = frame.additional_betas.map(lambda value: float(value[0]))
    frame[WORDS] = [
        f"Elo עם מקדם {row.beta:.2f}, לצד כושר עם מקדם {row.beta_form:+.2f}; שני משתנים באותה המרה"
        for row in frame.itertuples()
    ]
    return frame[["season_id", "beta", "beta_form", WORDS]].sort_values("season_id")


def coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Median and observed range across the 15 saved training-fold fits."""
    columns = [
        "a",
        "d",
        "b",
        "gamma",
        "attack_change_percent",
        "defense_change_percent",
        "home_win_change_pp",
    ]
    return pd.DataFrame(
        [
            {
                "quantity": column,
                "median": coefficients[column].median(),
                "minimum": coefficients[column].min(),
                "maximum": coefficients[column].max(),
                WORDS: {
                    "a": "רגישות השערים לכושר ההבקעה של הקבוצה",
                    "d": "רגישות השערים לכושר הספיגה של היריבה",
                    "b": "משקל פער Elo במודל השערים",
                    "gamma": "תוספת הבית מעבר לשאר המשתנים",
                    "attack_change_percent": "תוספת שערים בדוגמת הבקעה מ־1.5 ל־1.65",
                    "defense_change_percent": "תוספת שערים כשהיריבה סופגת מ־1.5 ל־1.65",
                    "home_win_change_pp": "שינוי בניצחון בית באותה דוגמת הבקעה",
                }[column],
            }
            for column in columns
        ]
    )


def reported_summer_check() -> pd.DataFrame:
    """Amit/Claude's reported joint OLS, T-010 §8.4; not refitted in this notebook."""
    return pd.DataFrame(
        [
            {
                "שיטה": "נקודות למשחק בכל העונה הקודמת",
                "מקדם באותה רגרסיה": 0.72,
                "מתאם פשוט עם 10 הראשונים": 0.54,
                WORDS: "מתאר את הכוח לאורך עונה מלאה; עשרת האחרונים כלולים גם בממוצע זה",
            },
            {
                "שיטה": "נקודות למשחק בעשרת האחרונים",
                "מקדם באותה רגרסיה": 0.11,
                "מתאם פשוט עם 10 הראשונים": 0.37,
                WORDS: "בודק מה סוף העונה מוסיף מעבר לעונה המלאה; כולל פלייאוף, לפי תאריך",
            },
        ]
    )


def route_overview() -> pd.DataFrame:
    """The two frozen diagnostic tasks, not two interchangeable scoring scales."""
    return pd.DataFrame(
        [
            {
                "מסלול": "P — שערים",
                "יעד": "שערים לכל קבוצה במשחק",
                "מה נבדק": "22 חלונות gf/ga עם Elo; אותם 22 גם ללא Elo לדיווח",
                "מדד": "Poisson NLL לקבוצה־משחק",
                "ייחוס": "Poisson עם ביתיות ו־Elo",
                WORDS: "מצליב את התקפת הקבוצה עם הגנת היריבה; Elo וביתיות נשארים במודל",
            },
            {
                "מסלול": "O — נקודות",
                "יעד": "בית / תיקו / חוץ",
                "מה נבדק": "22 הפרשי קצב נקודות נוסף על Elo",
                "מדד": "log loss למשחק",
                "ייחוס": "E0050 עם המרה סדורה",
                WORDS: "מוסיף את הפרש הנקודות למשחק לפער Elo בהמרה סדורה; בוחן תוספת מידע",
            },
        ]
    )


def points_leaders(review: FormReview) -> pd.DataFrame:
    """A compact O table with its reference, leader and simplicity proposal explicit."""
    frame = candidate_table(review, "O")
    reference = route_references(review).set_index("method").loc["O_reference_elo"]
    shown = frame.head(3)
    proposal = frame.loc[frame.config_id.eq(POINTS_EXAMPLE)]
    shown = pd.concat([shown, proposal]).drop_duplicates("config_id")
    row = {
        "method": "E0050 — ייחוס",
        "log_loss": reference.score,
        "last5_log_loss": reference.recent_score,
        "improvement": 0.0,
        WORDS: reference[WORDS],
    }
    return pd.concat(
        [
            pd.DataFrame([row]),
            shown[["method", "log_loss", "last5_log_loss", "improvement", WORDS]],
        ],
        ignore_index=True,
    )


def decision_table() -> pd.DataFrame:
    """Record the approved B decisions; no selection is performed here."""
    return pd.DataFrame(
        [
            {
                "רכיב": "E0050",
                "מעמד": "נבחר",
                "סרגל": "log loss = 1.020424",
                WORDS: (
                    "דירוג מתוצאות עבר: K=10, ביתיות 30, נסיגה 33%, מכפיל הפרש, "
                    "עולה 1450; ללא הנחה או משקל זמן"
                ),
            },
            {
                "רכיב": "התקפה והגנה ewma5/venue",
                "מעמד": "נבחר, חריגה מודעת מכלל הפשטות",
                "סרגל": "NLL = 1.409455",
                WORDS: "המשקל נחצה כל חמש הופעות, בית וחוץ בנפרד, נוסף מעל Elo במודל השערים",
            },
            {
                "רכיב": "כושר בנקודות",
                "מעמד": "🔬 נמצא ב־M0 כמועמד; לא ברירת מחדל",
                "סרגל": "0 מ־22 מוסיפים בממוצע",
                WORDS: (
                    "קצב הנקודות לא הוסיף בהמרה הסדורה; "
                    "pts_venue_ewma5 ייבדק ב־ablation של M2; פער המעמד ל־M4 פתוח ב־S4 D02"
                ),
            },
        ]
    )


def displayed_controls(review: FormReview) -> pd.DataFrame:
    """Readable vertical controls, preserving numerical values in control_numbers."""
    values = control_numbers(review).iloc[0]
    explanations = {
        "P Elo בלבד": "Poisson עם ביתיות ו־Elo; ייחוס לתוספת של חלונות שערים",
        "P ללא Elo": "קבוע וביתיות בלבד; ממחיש את תרומת דירוג הכוח",
        "נבחר NLL": "ציון השערים של התקפה והגנה EWMA 5 בבית ובחוץ בנפרד",
        "נבחר z": "תוספת הנבחר מעל ייחוס Elo, ביחידות רעש של הפרש הקיפולים",
        "O Elo בלבד": "סרגל בית/תיקו/חוץ של E0050, ביחידות שונות מציון שערים",
        "מועמדי כושר משפרים": "כמה מהגרסאות במסלול הנקודות משפרות בממוצע מעל Elo",
        "תיקו חזוי (%)": "ממוצע ההסתברות לתיקו מתוך רשת התוצאות השמורה",
        "תיקו נצפה (%)": "שיעור המשחקים שהסתיימו בפועל בתיקו בוולידציה",
        "ρ התקפה": "שיעור הפער בהבקעה ששורד את הקיץ לקראת 2023/2024",
        "ρ הגנה": "שיעור הפער בספיגה ששורד את הקיץ בכל המגרשים",
        "ρ נקודות": "שיעור פער קצב הנקודות ששורד את הקיץ",
    }
    rows = []
    for name, value in values.items():
        if name == WORDS:
            continue
        digits = 2 if name in ("נבחר z", "תיקו חזוי (%)", "תיקו נצפה (%)") else 6
        if name.startswith("ρ"):
            digits = 3
        if name == "מועמדי כושר משפרים":
            digits = 0
        rows.append({"מדד": name, "ערך": f"{value:.{digits}f}", WORDS: explanations[name]})
    return pd.DataFrame(rows)


def draw_diagnostics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Saved H/D/A, lambda-gap tertiles, exact draws and total-goal means."""
    rates = predictions[["lambda_home", "lambda_away"]].to_numpy(float)
    if len(rates) < 3 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Need finite positive saved Poisson rates")
    outcome_rows = [
        {
            "תוצאה": name,
            "predicted": predictions[column].mean(),
            "observed": predictions.outcome.eq(code).mean(),
            "games": len(predictions),
        }
        for code, (name, column) in enumerate(
            zip(("בית", "תיקו", "חוץ"), ("p_home", "p_draw", "p_away"), strict=True)
        )
    ]
    spread = np.abs(rates[:, 0] - rates[:, 1])
    groups, boundaries = pd.qcut(spread, 3, labels=["שקול", "בינוני", "לא שקול"], retbins=True)
    frame = predictions.assign(group=groups, spread=spread)
    strata = []
    for index, label in enumerate(("שקול", "בינוני", "לא שקול")):
        part = frame.loc[frame.group.eq(label)]
        strata.append(
            {
                "סוג משחק": label,
                "games": len(part),
                "gap_low": boundaries[index],
                "gap_high": boundaries[index + 1],
                "predicted": part.p_draw.mean(),
                "observed": part.outcome.eq(1).mean(),
            }
        )
    # Match the frozen 0..10 normalized square used to produce saved H/D/A.
    retained = poisson.cdf(10, rates[:, 0]) * poisson.cdf(10, rates[:, 1])
    exact = []
    for goals in range(4):
        expected = (
            poisson.pmf(goals, rates[:, 0]) * poisson.pmf(goals, rates[:, 1]) / retained
        ).mean()
        observed = ((predictions.home_goals == goals) & (predictions.away_goals == goals)).mean()
        exact.append(
            {
                "תוצאה": f"{goals}:{goals}",
                "predicted": expected,
                "observed": observed,
                "ratio": observed / expected if expected else np.nan,
            }
        )
    totals = pd.DataFrame(
        [
            {
                "מדד": "סך שערים למשחק",
                "predicted": rates.sum(axis=1).mean(),
                "observed": predictions[["home_goals", "away_goals"]].sum(axis=1).mean(),
            }
        ]
    )
    return pd.DataFrame(outcome_rows), pd.DataFrame(strata), pd.DataFrame(exact), totals


def control_numbers(review: FormReview) -> pd.DataFrame:
    """The requested saved-run anchors, suitable for a displayed notebook audit."""
    refs = route_references(review).set_index("method")
    chosen = candidate_table(review, "P").set_index("config_id").loc[SELECTED_P]
    points = candidate_table(review, "O")
    coefficients = opening_coefficients(review)
    rho = (
        coefficients.loc[(coefficients.season_id == 25) & (coefficients.stratum == "all")]
        .set_index("metric")
        .rho
    )
    return pd.DataFrame(
        [
            {
                "P Elo בלבד": refs.loc["P_reference_elo", "score"],
                "P ללא Elo": refs.loc["P_reference_no_elo", "score"],
                "נבחר NLL": chosen.poisson_nll,
                "נבחר z": chosen.z,
                "O Elo בלבד": refs.loc["O_reference_elo", "score"],
                "מועמדי כושר משפרים": int((points.improvement > 0).sum()),
                "תיקו חזוי (%)": review.predictions.p_draw.mean() * 100,
                "תיקו נצפה (%)": review.predictions.outcome.eq(1).mean() * 100,
                "ρ התקפה": rho.gf,
                "ρ הגנה": rho.ga,
                "ρ נקודות": rho.pts,
                WORDS: (
                    "שערים: Poisson מעל Elo; כושר: המרה סדורה מעל Elo; "
                    "תיקו מהתפלגות שמורה ו־ρ ממטא־דאטה"
                ),
            }
        ]
    )
