"""Read-only EXP-004 review; the sole refit entry point is explicit live validation."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.evaluate.backtest import DevelopmentPolicy, evaluate
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.motivation_experiment import (
    DEEP_SEASONS,
    VALIDATION_SEASONS,
    MotivationConfiguration,
    MotivationExperimentInput,
    MotivationOrderedPredictor,
    load_motivation_experiment,
    load_motivation_inputs,
    motivation_configurations,
    prepare_motivation_experiment,
)
from ligat.motivation.representations import game_contrast

GROUPS = {
    "dead_alive": "צד מת מול חיה",
    "dead_almost": "צד מת מול כמעט מתה",
    "both_dead": "שתי מתות",
    "almost": "כמעט מתה, בלי מתה",
    "alive": "שתיהן חיות",
}
PERIODS = {
    "deep": "2010/2011–2023/2024",
    "first9": "2010/2011–2018/2019",
    "last5": "2019/2020–2023/2024",
}
SCALARS = ("L2-diff", "L3-diff", "L5-diff", "Lc-diff")


def configuration_label(config_id: str) -> str:
    """One unique Hebrew description beside each of the twenty technical identities."""
    configuration = next((c for c in motivation_configurations() if c.config_id == config_id), None)
    if configuration is None:
        raise ValueError(f"Unknown EXP-004 configuration: {config_id}")
    base = configuration.name
    representation, form = base.split("-") if "-" in base else (base, "")
    words = {
        "C0": "Elo לבדו",
        "L2": "הדגל",
        "L3": "שלוש מדרגות",
        "L5": "חמש מדרגות",
        "Lc": "חשיבות רציפה",
        "Comp": "רכיב לכל יעד",
    }
    description = words[representation]
    if form:
        description += ", " + {"diff": "הפרש", "sides": "שני צדדים"}[form]
    description += ", עם דעיכה" if configuration.half_life else ", בלי דעיכה"
    return f"{config_id} — {description}"


def configuration_map() -> pd.DataFrame:
    """The frozen grid, total free parameters and explicit football assumptions."""
    rows = []
    for c in motivation_configurations():
        representation = c.name.split("-")[0]
        form = c.name.split("-")[-1] if "-" in c.name else "diff" if c.name == "Comp" else "—"
        assumption = {
            "C0": "החוזק של Elo מספיק לחיזוי התוצאה.",
            "L2": "קבוצה ללא יעד שונה מקבוצה עם יעד.",
            "L3": "גם יריבה שכמעט ללא יעד משנה את פער החשיבות.",
            "L5": "עוצמת החשיבות פועלת במדרגות דקות יותר.",
            "Lc": "השפעת החשיבות רציפה לאורך הסולם.",
            "Comp": "לאליפות, לעליון ולהישארות משקלים נפרדים.",
        }[representation]
        if form == "sides":
            assumption += " ההשפעה בבית ובחוץ יכולה להיות שונה."
        if c.half_life:
            assumption += " העבר הקרוב מקבל משקל גדול יותר."
        rows.append(
            dict(
                config_id=c.config_id,
                representation=representation,
                form=form,
                decay="המשקל נחצה כל ארבע עונות" if c.half_life else "בלי דעיכה",
                parameters=3 + len(c.columns),
                assumption=assumption,
            )
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ExperimentReview:
    """Verified saved artifacts and prematch-only design, with observed match context separate."""

    frames: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, Any]
    decision: Mapping[str, Any]
    pilot: Mapping[str, Any]
    prepared: MotivationExperimentInput
    context: pd.DataFrame


def load_experiment_review(root: Path, tables: AnalysisTables) -> ExperimentReview:
    """Verify both manifests; filter every Parquet read before loading any season data."""
    directory = root / "reports/experiments/EXP-004"
    metadata = verify_calibration(directory / "full/metadata.json")
    verify_calibration(directory / "pilot/metadata.json")
    frames = {}
    for name in (
        "summary",
        "comparison",
        "involved_games",
        "expected_points",
        "convergence",
        "shallow_fold",
    ):
        frames[name] = pd.read_csv(directory / f"full/{name}.csv", float_precision="round_trip")
    for name in ("fold_scores", "predictions", "fits", "coefficients"):
        frames[name] = pd.read_parquet(
            directory / f"full/{name}.parquet",
            filters=[("season_id", "in", list(DEVELOPMENT_SEASONS))],
        )
    for name in ("audit_L2-diff_decomposition", "audit_L2-diff_placebos"):
        frames[name] = pd.read_csv(directory / f"pilot/{name}.csv", float_precision="round_trip")
    prepared = prepare_motivation_experiment(
        tables, load_motivation_inputs(load_motivation_experiment())
    )
    assert prepared.motivation is not None
    sides = game_contrast(prepared.motivation, "L3", "sides")
    context = prepared.games.merge(
        sides, left_on="game_id", right_index=True, validate="one_to_one"
    )
    names = tables.match_table.reset_index()[
        ["game_id", "home_team_name", "away_team_name", "stage"]
    ]
    context = context.merge(names, on="game_id", validate="one_to_one")
    context["group"] = l3_groups(context.m_home, context.m_away)
    pilot = {
        name: json.loads((directory / f"pilot/{name}.json").read_text(encoding="utf-8"))
        for name in ("audit_L2-diff", "timing_audit")
    }
    decision = json.loads((directory / "full/review.json").read_text(encoding="utf-8"))
    return ExperimentReview(frames, metadata, decision, pilot, prepared, context)


def l3_groups(home: pd.Series, away: pd.Series) -> pd.Series:
    """Partition already computed L3 levels; never reproduce importance thresholds here."""
    if (
        not home.index.equals(away.index)
        or not home.isin([0, 0.5, 1]).all()
        or not away.isin([0, 0.5, 1]).all()
    ):
        raise ValueError("Expected aligned, finite shared L3 levels")
    low, high = np.minimum(home, away), np.maximum(home, away)
    values = np.select(
        [low.eq(0) & high.eq(1), low.eq(0) & high.eq(0.5), high.eq(0), low.eq(0.5)],
        ["dead_alive", "dead_almost", "both_dead", "almost"],
        default="alive",
    )
    return pd.Series(values, index=home.index, name="group")


def paired_games(review: ExperimentReview, config_id: str) -> pd.DataFrame:
    """Exactly paired game losses and actual-outcome probabilities; positive means improvement."""
    c = next(c for c in motivation_configurations() if c.config_id == config_id)
    predictions = review.frames["predictions"]
    a = predictions.loc[predictions.config_id.eq(config_id)].set_index("game_id").sort_index()
    b = predictions.loc[predictions.config_id.eq(c.reference_id)].set_index("game_id").sort_index()
    if not a.index.is_unique or not a.index.equals(b.index) or not a.outcome.equals(b.outcome):
        raise ValueError("Unpaired saved predictions")
    result = a.copy()
    for metric in ("log_loss", "rps"):
        result[f"{metric}_improvement"] = b[metric] - a[metric]
    result["actual_probability"] = np.exp(-a.log_loss)
    result["reference_probability"] = np.exp(-b.log_loss)
    context = review.context.set_index("game_id").drop(columns=["season_id"])
    return result.join(context, validate="one_to_one").reset_index()


def group_decomposition(review: ExperimentReview) -> pd.DataFrame:
    """All eighteen representations, both losses, five disjoint groups and three scopes."""
    records = []
    for c in motivation_configurations():
        if c.name == "C0":
            continue
        games = paired_games(review, c.config_id)
        for period in PERIODS:
            seasons = (
                DEEP_SEASONS
                if period == "deep"
                else tuple(range(12, 21))
                if period == "first9"
                else tuple(range(21, 26))
            )
            part = games.loc[games.season_id.isin(seasons)]
            for metric in ("log_loss", "rps"):
                for group in GROUPS:
                    values = part.loc[part.group.eq(group), f"{metric}_improvement"]
                    records.append(
                        dict(
                            config_id=c.config_id,
                            period=period,
                            group=group,
                            metric=metric,
                            games=len(values),
                            all_games=len(part),
                            mean_improvement=float(values.mean()) if len(values) else 0.0,
                            contribution=float(values.sum() / len(part)),
                        )
                    )
    result = pd.DataFrame(records)
    totals = (
        result.loc[result.period.eq("deep")].groupby(["config_id", "metric"]).contribution.sum()
    )
    expected = (
        review.frames["comparison"]
        .loc[lambda x: x.scope.eq("deep")]
        .set_index(["config_id", "metric"])
        .mean_improvement
    )
    if not np.allclose(totals, expected.reindex(totals.index), atol=1e-12, rtol=0):
        raise AssertionError("Group contributions do not reproduce saved paired improvement")
    return result


def fold_improvements(folds: pd.DataFrame, metric: str = "log_loss") -> pd.DataFrame:
    """Configuration by season matrix of matching-decay control minus candidate losses."""
    matrix = folds.pivot(index="config_id", columns="season_id", values=metric)
    ids = [c.config_id for c in motivation_configurations()]
    result = matrix.reindex(ids).copy()
    for c in motivation_configurations():
        result.loc[c.config_id] = matrix.loc[c.reference_id] - matrix.loc[c.config_id]
    return result


def ranking_check(review: ExperimentReview) -> pd.DataFrame:
    """Reconstruct deep/recent weighted rankings and the rule-6 flag from saved fold scores."""
    rows: list[dict[str, Any]] = []
    for key, seasons in (("deep", DEEP_SEASONS), ("last5", tuple(range(21, 26)))):
        folds = review.frames["fold_scores"]
        part = folds.loc[folds.season_id.isin(seasons)]
        scores = {
            str(c): float(np.average(g.log_loss, weights=g.games))
            for c, g in part.groupby("config_id")
        }
        stored = (
            review.decision["rankings"]["log_loss"]
            if key == "deep"
            else review.decision["last5_ranking"]
        )
        if set(stored["scores"]) != set(scores) or any(
            abs(scores[c] - stored["scores"][c]) > 1e-12 for c in scores
        ):
            raise AssertionError("Saved rule-6 ranking differs from fold scores")
        ordered = sorted(scores, key=lambda c: scores[c])
        if set(stored["winners"]) != {c for c in scores if scores[c] == min(scores.values())}:
            raise AssertionError("Saved rule-6 winner differs")
        rows.extend(
            dict(scope=key, rank=i + 1, config_id=c, log_loss=scores[c])
            for i, c in enumerate(ordered[:5])
        )
    flag = (
        review.decision["rankings"]["log_loss"]["winners"]
        != review.decision["last5_ranking"]["winners"]
    )
    assert flag == review.decision["rule6"]
    return pd.DataFrame(rows)


def result_table(review: ExperimentReview, metric: str) -> pd.DataFrame:
    """All twenty rows, both scoring scopes, geometric probability and separate entry conditions."""
    s = review.frames["summary"].copy()
    s["geometric_all15_pct"] = 100 * np.exp(-s.all15_log_loss)
    s["geometric_deep_pct"] = 100 * np.exp(-s.deep_log_loss)
    s["mean_gt_se"] = s[f"{metric}_mean_improvement"] > s[f"{metric}_se"]
    s["recent_at_least_four"] = s[f"{metric}_last5_folds_better"] >= 4
    s["reference"] = [
        next(c.reference_id for c in motivation_configurations() if c.config_id == i)
        for i in s.config_id
    ]
    s["comparison_rule"] = np.where(
        s.config_id.eq("C0-hl4"), "כלל 7", np.where(s.config_id.eq("C0"), "בקרה", "כלל 8")
    )
    return s[
        [
            "config_id",
            "reference",
            "comparison_rule",
            f"all15_{metric}",
            f"deep_{metric}",
            "geometric_all15_pct",
            "geometric_deep_pct",
            f"{metric}_mean_improvement",
            f"{metric}_se",
            f"{metric}_folds_better",
            f"{metric}_last5_folds_better",
            "mean_gt_se",
            "recent_at_least_four",
            f"{metric}_passes_rule8",
            "audit_required",
        ]
    ]


def involved_results(review: ExperimentReview) -> pd.DataFrame:
    """Weight the saved involved-game losses by their actual game counts."""
    rows = []
    for c, part in review.frames["involved_games"].groupby("config_id", sort=False):
        candidate = float(np.average(part.log_loss, weights=part.games))
        reference = float(np.average(part.reference_log_loss, weights=part.games))
        rows.append(
            dict(
                config_id=c,
                reference=part.reference.iloc[0],
                games=int(part.games.sum()),
                log_loss=candidate,
                reference_log_loss=reference,
                mean_improvement=reference - candidate,
            )
        )
    return pd.DataFrame(rows)


def extreme_games(review: ExperimentReview, config_id: str = "L3-diff-hl4") -> pd.DataFrame:
    """Ten best and ten worst actual fixtures, with names/result/stage and shared L3 sides."""
    games = paired_games(review, config_id).sort_values("log_loss_improvement")
    result = pd.concat(
        [
            games.tail(10).iloc[::-1].assign(extreme="השיפור הגדול"),
            games.head(10).assign(extreme="ההרעה הגדולה"),
        ]
    )
    result["result"] = (
        result.home_goals.astype(int).astype(str) + ":" + result.away_goals.astype(int).astype(str)
    )
    return result[
        [
            "config_id",
            "extreme",
            "season_id",
            "date",
            "stage",
            "home_team_name",
            "away_team_name",
            "result",
            "m_home",
            "m_away",
            "reference_probability",
            "actual_probability",
            "log_loss_improvement",
        ]
    ]


def point_effects(review: ExperimentReview) -> pd.DataFrame:
    """Saved equal-Elo translations; controls have identically zero motivation effect."""
    saved = review.frames["expected_points"].loc[lambda x: x.elo_diff.eq(0)].copy()
    controls = pd.DataFrame(
        [
            dict(
                config_id=c.config_id,
                fit_scope=scope,
                side=side,
                component="none",
                elo_diff=0.0,
                expected_points_change=0.0,
            )
            for c in motivation_configurations()
            if not c.columns
            for scope in ("last_fold", "deep_mean")
            for side in ("home", "away")
        ]
    )
    return pd.concat([controls, saved], ignore_index=True)


def validate_live_values(
    actual: pd.DataFrame, saved: pd.DataFrame, columns: list[str], *, tolerance: float = 1e-10
) -> float:
    """A missing/misaligned key, nonfinite value or excessive error is a hard live gate failure."""
    a, b = actual.sort_index(), saved.sort_index()
    if not a.index.is_unique or not a.index.equals(b.index):
        raise AssertionError("Live EXP-004 validation: mismatched fold/parameter keys")
    delta = np.abs(a[columns].to_numpy(float) - b[columns].to_numpy(float))
    if not np.isfinite(delta).all() or (delta.size and float(delta.max()) > tolerance):
        raise AssertionError("Live EXP-004 validation differs from saved values by more than 1e-10")
    return float(delta.max()) if delta.size else 0.0


def live_validation(review: ExperimentReview) -> pd.DataFrame:
    """Only the three authorized fifteen-fold refits; compare every loss and fitted parameter."""
    rows = []
    for c in (
        MotivationConfiguration("C0"),
        MotivationConfiguration("L2-diff"),
        MotivationConfiguration("L3-diff", 4),
    ):
        models: list[MotivationOrderedPredictor] = []

        def factory(
            c: MotivationConfiguration = c,
            models: list[MotivationOrderedPredictor] = models,
        ) -> MotivationOrderedPredictor:
            model = MotivationOrderedPredictor(c)
            models.append(model)
            return model

        start = perf_counter()
        evaluated = evaluate(
            factory,
            review.prepared.games,
            feature_columns=("elo_diff", *c.columns),
            policy=DevelopmentPolicy(training_start=8, validation_seasons=VALIDATION_SEASONS),
        )
        scores = evaluated.per_fold.set_index("season_id")
        folds = review.frames["fold_scores"]
        saved = folds.loc[folds.config_id.eq(c.config_id)].set_index("season_id")
        loss_error = validate_live_values(scores, saved, ["log_loss", "rps"])
        coefficients: list[dict[str, Any]] = []
        for model, season in zip(models, VALIDATION_SEASONS, strict=True):
            fit = model.fit_result
            assert fit is not None
            coefficients.extend(
                dict(season_id=season, coefficient=k, value=v)
                for k, v in zip(
                    ["b_elo", *c.columns, "theta1", "delta", "theta2"],
                    [fit.beta, *fit.additional_betas, fit.theta1, fit.delta, fit.theta2],
                    strict=True,
                )
            )
        actual = pd.DataFrame(coefficients).set_index(["season_id", "coefficient"])
        coefs = review.frames["coefficients"]
        saved_coefs = coefs.loc[coefs.config_id.eq(c.config_id)].set_index(
            ["season_id", "coefficient"]
        )
        coef_error = validate_live_values(actual, saved_coefs, ["value"])
        rows.append(
            dict(
                config_id=c.config_id,
                folds=len(scores),
                loss_error=loss_error,
                coefficient_error=coef_error,
                seconds=perf_counter() - start,
            )
        )
    return pd.DataFrame(rows)


def definition_comparison(review: ExperimentReview) -> pd.DataFrame:
    """Audit Claude's alternative maximum without fitting or changing the experiment definition."""
    motivation = review.prepared.motivation
    assert motivation is not None
    alternate = motivation.copy()
    alternate["leverage_max"] = alternate.filter(regex=r"_lev$").max(axis=1)
    old, new = (game_contrast(f, "L3", "sides") for f in (motivation, alternate))
    changed = old.ne(new).any(axis=1)
    context = review.context.set_index("game_id")
    return (
        context.loc[
            changed.reindex(context.index), ["season_id", "home_team_name", "away_team_name"]
        ]
        .join(old.rename(columns=lambda c: "package_" + c))
        .join(new.rename(columns=lambda c: "claude_" + c))
        .reset_index()
    )


def numerical_checks(review: ExperimentReview) -> pd.DataFrame:
    """Check every new rounded finding quoted in the opening and Amit's unchanged decision."""
    summary = review.frames["summary"].set_index("config_id")
    coefs = review.frames["coefficients"].set_index(["config_id", "season_id", "coefficient"])
    games = paired_games(review, "L2-diff")
    recent = games.loc[games.season_id.ge(21) & games.group.isin(list(GROUPS)[:3])]
    deep = games.loc[games.season_id.ge(12) & games.group.isin(list(GROUPS)[:3])]
    findings = {
        "תצורות": (len(summary), "20", 0),
        "עברו כלל כניסה": (summary.log_loss_passes_rule8.sum(), "0", 0),
        "ציון המובילה בכל הקיפולים": (summary.loc["L3-diff-hl4", "all15_log_loss"], "1.0175", 4),
        "ציון Elo בכל הקיפולים": (summary.loc["C0", "all15_log_loss"], "1.0204", 4),
        "תנאי ממוצע גדול משגיאת תקן באחד המדדים": (
            (
                (summary.log_loss_mean_improvement > summary.log_loss_se)
                | (summary.rps_mean_improvement > summary.rps_se)
            ).sum(),
            "7",
            0,
        ),
        "דגל בתקופה האחרונה, משחקים עם צד מת": (recent.log_loss_improvement.mean(), "-0.014", 3),
        "מקדם הדגל ב-2010/2011": (coefs.loc[("L2-diff", 12, "L2_m_diff"), "value"], "1.42", 2),
        "מקדם הדגל ב-2023/2024": (coefs.loc[("L2-diff", 25, "L2_m_diff"), "value"], "0.70", 2),
        "משחקים עם צד מת לעונה עמוקה": (len(deep) / len(DEEP_SEASONS), "15", 0),
    }
    for metric, expected in (("log_loss", "4"), ("rps", "7")):
        findings[f"תנאי ממוצע גדול משגיאת תקן: {metric}"] = (
            (summary[f"{metric}_mean_improvement"] > summary[f"{metric}_se"]).sum(),
            expected,
            0,
        )
    almost = (
        group_decomposition(review)
        .loc[lambda x: x.config_id.eq("L3-diff") & x.group.eq("almost") & x.metric.eq("log_loss")]
        .set_index("period")
    )
    findings.update(
        {
            "כמעט מתה בלי מתה, משחקים בחמש האחרונות": (almost.loc["last5", "games"], "211", 0),
            "כמעט מתה בלי מתה, שיפור בחמש האחרונות": (
                almost.loc["last5", "mean_improvement"],
                "-0.008",
                3,
            ),
            "כמעט מתה בלי מתה, שיפור בתקופה הראשונה": (
                almost.loc["first9", "mean_improvement"],
                "0.005",
                3,
            ),
        }
    )
    rows = [
        dict(
            claim=k, calculated=float(v), displayed=expected, matches=f"{v:.{digits}f}" == expected
        )
        for k, (v, expected, digits) in findings.items()
    ]
    if not all(r["matches"] for r in rows):
        raise AssertionError(
            f"B2 rounded numerical claim failed: {[r for r in rows if not r['matches']]}"
        )
    return pd.DataFrame(rows)


def entry_counts(summary: pd.DataFrame) -> dict[str, int]:
    """Count the first rule-8 condition per metric and once in their union."""
    log_loss = summary.log_loss_mean_improvement > summary.log_loss_se
    rps = summary.rps_mean_improvement > summary.rps_se
    return {
        "log_loss": int(log_loss.sum()),
        "rps": int(rps.sum()),
        "either": int((log_loss | rps).sum()),
    }


def decomposition_findings(frame: pd.DataFrame) -> str:
    """Dynamic package-only findings; Claude's comparison belongs to the execution report."""
    part = frame.loc[
        frame.config_id.eq("L3-diff") & frame.group.eq("almost") & frame.metric.eq("log_loss")
    ].set_index("period")
    return (
        "בחמש העונות האחרונות, 2019/2020–2023/2024, "
        f"L3-diff מקלקל ב-{int(part.loc['last5', 'games'])} משחקים עם קבוצה כמעט מתה ובלי מתה: "
        f"{part.loc['last5', 'mean_improvement']:+.3f} לכל משחק. "
        "בתקופה הראשונה, 2010/2011–2018/2019, הוא שיפר באותם סוגי משחקים: "
        f"{part.loc['first9', 'mean_improvement']:+.3f} לכל משחק. "
        "הפירוש הכדורגלני האפשרי: פעם קבוצה שכמעט לא נשאר לה על מה לשחק שיחקה רגוע, "
        "והיום כבר לא. זהו הסבר אפשרי נוסף לכישלון L3 בעונות האחרונות, ולא הוכחת סיבתיות."
    )
