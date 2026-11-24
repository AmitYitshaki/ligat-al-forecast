"""Signed evidence tables for the notebook QA walk; never fit or select."""

import json
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.goals_experiment import GoalExperimentSpec
from ligat.evaluate.goals_ladder import LadderConfiguration
from ligat.evaluate.selection import paired_statistics
from ligat.viz.goal_training_curves import frozen_full_report, full_validation_table
from ligat.viz.seasons import season_label


def evidence(spec: GoalExperimentSpec) -> tuple[dict[str, Any], pd.DataFrame]:
    return frozen_full_report(spec), full_validation_table(spec)


def configuration_folds(v: pd.DataFrame, decision: dict[str, Any], key: str) -> pd.DataFrame:
    config = decision["configurations"][key]
    columns = json.dumps(decision["features"])
    part = v.loc[v.configuration_key.eq(LadderConfiguration(**config).key)].copy()
    part = part.loc[part.feature_list.map(json.loads).map(lambda x: x == json.loads(columns))]
    return part.assign(season_id=part.fold + 10)


def review_evidence(report: dict[str, Any], v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for review in report["open_reviews"]:
        d = next(d for d in report["decisions"] if d["name"] == review["name"])
        deep = review["deep_leaders"][0]["key"]
        reference = configuration_folds(v, d, deep)
        for kind, key in [
            ("חמש האחרונות", review["recent_leaders"][0]["key"]),
            ("הצעה זמנית", review["proposal"]),
        ]:
            candidate = configuration_folds(v, d, key)
            s = paired_statistics(candidate, reference, score="poisson_nll")
            a, b = d["configurations"][deep], d["configurations"][key]
            changes = [
                f"{p}: {a['parameters'].get(p)} → {value}"
                for p, value in b["parameters"].items()
                if a["parameters"].get(p) != value
            ]
            if a["half_life"] != b["half_life"]:
                changes.append(f"זמן מחצית: {a['half_life']} → {b['half_life']}")
            rows.append(
                dict(
                    stage=review["name"],
                    kind=kind,
                    label=d["labels"][key],
                    changes="; ".join(changes) or "אותה תצורה",
                    gap=-s.mean_improvement if s.mean_improvement is not None else None,
                    se=s.se,
                    n=s.n,
                    depends="כל הסולם שאחרי S2"
                    if review["name"] == "S2-AB"
                    else ", ".join(x for x in review["dependents"] if len(x) < 40),
                )
            )
    return pd.DataFrame(rows)


def entry_tables(report: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, differences = [], []
    for d in report["decisions"]:
        if d["kind"] != "entry":
            continue
        s = d["rule8"]["stats"]
        entries.append(
            dict(
                name=d["name"],
                d=s["mean_improvement"],
                se=s["se"],
                lower=s["ci_lower"],
                upper=s["ci_upper"],
                n=s["n"],
                last5=s["last5_improvement"],
                better=s["last5_folds_better"],
                passes=d["rule8"]["passes"],
                entered=d["entered"],
            )
        )
        if d["name"] == "F5":
            for row in s["differences"]:
                differences.append(
                    dict(
                        season=season_label(row["season_id"]),
                        d=row["difference"],
                        games=row["games"],
                    )
                )
    return pd.DataFrame(entries), pd.DataFrame(differences)


def outcome_diagnostics(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in v.groupby("group_key"):
        loss = float(np.average(part.log_loss, weights=part.games))
        rows.append(
            dict(
                group=key,
                label=part.iloc[0].label,
                half_life=part.iloc[0].half_life,
                loss=loss,
                flag=loss < 1,
            )
        )
    return pd.DataFrame(rows)


def fold_group(v: pd.DataFrame, cfg: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    key = LadderConfiguration(**cfg).key
    frame = v.loc[v.configuration_key.eq(key)].copy()
    return frame.loc[frame.feature_list.map(json.loads).map(lambda x: x == columns)].assign(
        season_id=lambda x: x.fold + 10
    )


def disagreements(report: dict[str, Any], v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in report["decisions"]:
        if d.get("rule10") is not True:
            continue
        if d["kind"] == "entry":
            reference = fold_group(v, d["config"], d["reference"])
            candidate = fold_group(v, d["config"], d["candidate"])
        else:
            ranking = d["rule4"]["ranking"]
            winner = ranking["winners"][0]
            reference = configuration_folds(v, d, winner)
            candidates = {key: configuration_folds(v, d, key) for key in d["configurations"]}
            rps_winner = min(
                candidates,
                key=lambda key: float(
                    np.average(
                        candidates[key].loc[candidates[key].deep_fold, "rps"],
                        weights=candidates[key].loc[candidates[key].deep_fold, "games"],
                    )
                ),
            )
            candidate = candidates[rps_winner]
        for score in ["poisson_nll", "rps"]:
            stats = paired_statistics(candidate, reference, score=score)
            rows.append(
                dict(stage=d["name"], score=score, d=stats.mean_improvement, se=stats.se, n=stats.n)
            )
    return pd.DataFrame(rows)


def decay_summary(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backend, role in [("XGBoost", "S2-AB"), ("Maher", "maher")]:
        part = v.loc[v.stage_role.eq(role)]
        for half, group in part.groupby("half_life", dropna=False):
            rows.append(
                dict(
                    model=backend,
                    half="בלי דעיכה" if pd.isna(half) else half,
                    nll=float(np.average(group.poisson_nll, weights=group.games)),
                    configurations=group.group_key.nunique(),
                )
            )
    return pd.DataFrame(rows)


def feature_guide() -> pd.DataFrame:
    from ligat.viz.goals_notebook import ladder_plan_table

    names = {
        "elo_pre": "כוח Elo של הקבוצה",
        "elo_diff": "פער הכוח מול היריבה",
        "is_home": "האם המשחק בבית",
        "value_avg_log_ratio": "יחס שווי הסגלים בלוג",
        "stage": "שלב הליגה",
        "rounds_to_stage_end": "מחזורים שנותרו לשלב",
        "promoted_in": "הקבוצה עולה חדשה",
        "opp_promoted_in": "היריבה עולה חדשה",
        "nothing_to_play": "לקבוצה אין יעד שנותר להשיג",
        "opp_nothing_to_play": "ליריבה אין יעד שנותר להשיג",
        "games_played": "משחקים ששוחקו",
        "points_per_game": "נקודות למשחק",
        "gd_per_game": "הפרש שערים למשחק",
        "position_pct": "מיקום יחסי בליגה",
        "gap_to_leader": "פער מהמובילה",
        "gap_to_cut": "פער מסף הפלייאוף",
        "gap_to_safety": "פער מקו ההישרדות",
        "games_behind": "משחקים חסרים",
        "opp_points_per_game": "נקודות היריבה למשחק",
        "opp_position_pct": "מיקום היריבה",
        "prev_gd_diff": "פער בהפרש השערים בעונה הקודמת",
        "position_log_ratio": "יחס המיקומים הקודמים בלוג",
        "team_id": "זהות הקבוצה",
        "opponent_id": "זהות היריבה",
    }

    def form_name(field: str) -> str:
        side = "היריבה" if field.startswith("opp_") else "הקבוצה"
        metric = "נקודות" if "pts" in field else "שערי חובה" if "ga" in field else "שערי זכות"
        period = (
            "פיזור שישה משחקים"
            if "std_m6" in field
            else "המשחק הקודם"
            if field.endswith("prev")
            else "כל ההיסטוריה"
            if field.endswith("hist")
            else "חמשת המשחקים האחרונים"
        )
        venue = "לפי מגרש" if "venue" in field else "בכל מגרש"
        return f"{metric} של {side}; {period}; {venue}"

    return pd.DataFrame(
        [
            dict(
                stage=row.stage,
                field=field,
                hebrew=names.get(field) or form_name(field),
                purpose=row.meaning,
            )
            for row in ladder_plan_table().itertuples()
            for field in row.features.split(", ")
        ]
    )


def grid_explanation(spec: GoalExperimentSpec) -> str:
    from ligat.models.m2_xgb import XGBGoalConfig

    a, b = spec.raw["xgboost"]["stage_a"], spec.raw["xgboost"]["stage_b"]
    count_a = int(
        np.prod(
            [len(a[k]) for k in ["max_depth", "min_child_weight", "n_estimators", "learning_rate"]]
        )
    )
    count_b = int(np.prod([len(b[k]) for k in ["reg_lambda", "subsample", "half_life"]]))
    default = XGBGoalConfig.library_default()
    return (
        f'<div dir="rtl"><p>שלב A: {count_a} צירופי רשת ועוד תצורת ברירת מחדל, '
        f"בסך הכול {count_a + 1}. שלב B: {count_b} צירופי ריסון ודעיכה. "
        f"ברירת המחדל היא עומק {default.max_depth}, {default.n_estimators} עצים, "
        f"קצב {default.learning_rate}, דגימת שורות {default.subsample} "
        f"ודגימת פיצ׳רים {default.colsample_bytree}. הנימוק הכללי של התקציב ב-§7; "
        "הטבלה מפרידה בין פעולת הפרמטר למשמעות בכדורגל.</p></div>"
    )


def correlation_evidence(data: Any) -> pd.DataFrame:
    fields = ["prev_ppm_diff", "prev_gd_diff", "position_log_ratio"]
    pair = data.X[fields[:2]].dropna()
    relative = data.X[[fields[0], fields[2]]].dropna()
    return pd.DataFrame(
        [
            dict(
                pair="פער נקודות / פער שערים בעונה קודמת",
                r=float(pair.corr().iloc[0, 1]),
                n=len(pair),
                missing=len(data.X) - len(pair),
                position_r=float(relative.corr().iloc[0, 1]),
                scope=f"חושב על {season_label(int(data.context.season_id.min()))}–"
                f"{season_label(int(data.context.season_id.max()))}",
                codex="לשמור נקודות; זו הצעה שלא התקבלה",
                claude="לשמור שערים; תיאור ישיר יותר של כוח לקראת חיזוי שערים",
                decision="הוצאת prev_ppm_diff; נקודות קודמות כבר קשורות למיקום",
            )
        ]
    )


def hyperparameter_guide() -> pd.DataFrame:
    from ligat.viz.goals_notebook import hyperparameter_table

    frame = hyperparameter_table().drop(columns=["rationale"])
    reasons = {
        "max_depth": ("מגביל אינטראקציות", "שילוב כוח, בית ושלב בלי לשנן משחק בודד"),
        "depth": ("השוואת רמות מורכבות באותו תקציב", "מורכבות הקשרים בין מצבי משחק"),
        "min_child_weight": ("נדרש מספיק מידע לפני פיצול", "לא ליצור חוק על משחק חריג"),
        "learning_rate": ("כל תיקון נשאר קטן", "ריסון קפיצה בתחזית בגלל מעט משחקים"),
        "n_estimators": ("השוואת תקציבי תיקונים", "כמה דפוסי משחק אפשר לצבור"),
        "iterations": ("אותו תקציב עצים ב-CatBoost", "מספר תיקונים מצטבר לתחזית"),
        "subsample": ("מפחית תלות במדגם יחיד", "לא לבסס דפוס על משחק אחד"),
        "colsample_bytree": ("מפחית תלות בפיצ׳ר יחיד", "לא להסתמך רק על Elo או על שווי סגל"),
        "reg_lambda": ("מקטין תיקונים קיצוניים", "ממתן שינוי גדול בציפיית השערים"),
        "l2_leaf_reg": ("קנס ריבועי ב-CatBoost", "ממתן קיצוניות בתחזית לקבוצה"),
        "half_life": ("השוואת זיכרון קצר וארוך", "כמה ממשחקי העבר עדיין מתארים את הקבוצה"),
        "one_hot_max_size": ("שלב מקודד כן/לא; מזהים נשארים ב-CTR", "מבחין בין שלבי הליגה"),
        "library_default": ("נקודת ייחוס של הספרייה בשלמותה", "אין משמעות כדורגלנית ישירה"),
    }
    frame["reason"] = frame.parameter.map(
        lambda p: reasons.get(p, ("הגדרה קבועה לצורך שחזור", "אין משמעות כדורגלנית ישירה"))[0]
    )
    frame["football"] = frame.parameter.map(
        lambda p: reasons.get(p, ("הגדרה קבועה לצורך שחזור", "אין משמעות כדורגלנית ישירה"))[1]
    )
    return frame
