"""Read-only explanations of F5 predictions from the verified reconstructions."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import xgboost as xgb

from ligat.evaluate.goals_runner import _payload_digest
from ligat.models.m2_xgb import XGBGoalModel, _features
from ligat.viz.goals_f5_refit import SEASONS, sha256
from ligat.viz.goals_notebook import COLORS, finish
from ligat.viz.seasons import season_label

FEATURE_NAMES = {
    "elo_pre": "כוח Elo לפני המשחק",
    "elo_diff": "פער Elo",
    "is_home": "בית",
    "gf_venue_ewma5": "כושר הבקעה במגרש דומה",
    "opp_ga_venue_ewma5": "כושר ספיגה של היריבה במגרש דומה",
    "value_avg_log_ratio": "יחס שווי הסגלים",
    "stage": "שלב הליגה",
    "rounds_to_stage_end": "מחזורים עד סוף השלב",
    "promoted_in": "קבוצה עולה",
    "opp_promoted_in": "יריבה עולה",
    "nothing_to_play": "לקבוצה אין על מה לשחק",
    "opp_nothing_to_play": "ליריבה אין על מה לשחק",
}
FLAGS = ("nothing_to_play", "opp_nothing_to_play")


def verified_models(root: Path) -> tuple[dict[str, Any], dict[int, XGBGoalModel]]:
    target = root / "diagnostics" / "f5_explain"
    saved = json.loads((target / "manifest.json").read_bytes())
    manifest = saved["payload"]
    records = manifest["folds"]
    if (
        saved["sha256"] != _payload_digest(manifest)
        or manifest["ladder_sha256"] != sha256(root / "full" / "ladder_report.json")
        or tuple(p["season_id"] for p in records) != SEASONS
    ):
        raise ValueError("Invalid F5 verification manifest")
    models = {}
    for p in records:
        path = target / f"season-{p['season_id']}.json"
        if (
            not p["bit_identical"]
            or p["model_sha256"] != sha256(path)
            or p["metadata_sha256"] != sha256(path.with_suffix(".metadata.json"))
        ):
            raise ValueError(f"Unverified F5 reconstruction: {path}")
        models[p["season_id"]] = XGBGoalModel.load(path)
    return manifest, models


def contributions(
    model: XGBGoalModel, X: Any, native: Any
) -> tuple[Any, dict[str, float], float, float]:
    prepared, _ = _features(X)
    estimator = model._estimator
    if estimator is None:
        raise ValueError("F5 reconstruction must be loaded")
    booster = estimator.get_booster()
    matrix = xgb.DMatrix(prepared, enable_categorical=True, nthread=1)
    values = booster.predict(matrix, pred_contribs=True).astype(np.float64)
    rates = np.asarray(native, dtype=np.float64)
    reconstructed = np.exp(values.sum(axis=1))
    np.testing.assert_allclose(reconstructed, rates, rtol=1e-6, atol=0)
    errors = np.abs(reconstructed - rates)
    raw = cast(dict[str, float], booster.get_score(importance_type="gain"))
    total = sum(raw.values())
    gain = {c: 100 * raw.get(c, 0) / total for c in X.columns}
    return (
        pd.DataFrame(values[:, :-1], index=X.index, columns=X.columns),
        gain,
        float(errors.max()),
        float(np.max(errors / rates)),
    )


def home_alternatives(X: Any) -> tuple[Any, Any]:
    home = X.is_home
    at_home, away = (
        X.assign(is_home=pd.Series(value, index=X.index, dtype=home.dtype)) for value in (1, 0)
    )
    return at_home, away


def _home_summary(p: Any, fields: list[str]) -> Any:
    grouped = p.groupby(fields, observed=True)
    by = grouped.log_factor.agg(["mean", "size"]).rename(columns={"size": "rows"})
    by["model"] = np.exp(by.pop("mean"))
    goals = p.groupby([*fields, "is_home"], observed=True).goals.mean().unstack("is_home")
    by["observed"] = goals[True] / goals[False]
    return by.reset_index()


def diagnostics(data: Any, root: Path) -> dict[str, Any]:
    manifest, models = verified_models(root)
    journal = pd.DataFrame(manifest["folds"])
    shap_rows, gains, flags, home, errors = [], [], [], [], []
    for entry in manifest["folds"]:
        season = entry["season_id"]
        model = models[season]
        signed = json.loads((root / "full" / (entry["unit_id"] + ".json")).read_bytes())
        unit = signed["payload"]
        if signed["sha256"] != _payload_digest(unit):
            raise ValueError("Invalid signed F5 unit")
        native = pd.DataFrame(unit["per_team"]).set_index(["game_id", "team_id"]).sort_index()
        X = data.X.loc[native.index, unit["unit"]["features"]]
        context = data.context.loc[X.index]
        shap, gain, absolute, relative = contributions(model, X, native["lambda"])
        errors.append((absolute, relative))
        shap_rows.append(shap.abs().mean())
        gains.append(dict(season_id=season, **gain))
        for field in FLAGS:
            active = X[field].eq(1)
            q = shap.loc[active, field]
            flags.append(
                dict(
                    season_id=season,
                    feature=field,
                    rows=len(q),
                    mean=q.mean(),
                    mean_abs=q.abs().mean(),
                )
            )
        at_home, away = home_alternatives(X)
        log_factor = np.log(model.predict(at_home, context=context)) - np.log(
            model.predict(away, context=context)
        )
        home.append(
            pd.DataFrame(
                dict(
                    season_id=season,
                    stage=X.stage,
                    is_home=X.is_home,
                    goals=data.y.loc[X.index],
                    log_factor=log_factor,
                )
            )
        )
    marks = pd.DataFrame(shap_rows).quantile([0.25, 0.5, 0.75]).T
    marks.columns = ["lower", "median", "upper"]
    home_rows = pd.concat(home)
    by_stage_season = _home_summary(home_rows, ["stage", "season_id"])
    stage = _home_summary(home_rows, ["stage"])
    stage["sd"] = stage.stage.map(by_stage_season.groupby("stage", observed=True).model.std())
    flag_folds = pd.DataFrame(flags)
    flag_totals = []
    for field, group in flag_folds.query("rows > 0").groupby("feature"):
        flag_totals.append(
            dict(
                feature=FEATURE_NAMES[field],
                rows=int(group.rows.sum()),
                mean=float(np.average(group["mean"], weights=group.rows)),
                mean_abs=float(np.average(group.mean_abs, weights=group.rows)),
            )
        )
    return dict(
        journal=journal,
        importance=marks.sort_values("median"),
        flags=pd.DataFrame(flag_totals),
        gain=pd.DataFrame(gains),
        home=_home_summary(home_rows, ["season_id"]),
        stage=stage,
        stage_seasons=by_stage_season,
        overall=float(np.exp(home_rows.log_factor.mean())),
        elapsed_seconds=manifest["elapsed_seconds"],
        max_absolute_error=max(a for a, _ in errors),
        max_relative_error=max(r for _, r in errors),
    )


def importance_figure(result: dict[str, Any]) -> Any:
    p = result["importance"]
    fig = go.Figure(
        go.Scatter(
            x=p["median"],
            y=p.index.map(FEATURE_NAMES),
            mode="markers",
            marker_color=COLORS[0],
            error_x=dict(
                array=p.upper - p["median"], arrayminus=p["median"] - p.lower, symmetric=False
            ),
        )
    )
    fig.update_xaxes(title="ממוצע תרומה מוחלטת ל-log λ, חציון בין קיפולים")
    fig = finish(fig, "על איזה מידע F5 נשען בתחזית מחוץ לאימון", height=740)
    fig.update_layout(margin_l=270)
    return fig


def home_figure(result: dict[str, Any]) -> Any:
    p = result["home"]
    fig = go.Figure()
    for i, (field, label) in enumerate(
        (
            ("model", "תגובת F5 להחלפת דגל הבית"),
            ("observed", "בנתונים, כולל כל השפעות המגרש"),
        )
    ):
        fig.add_scatter(
            x=p.season_id.map(season_label),
            y=p[field],
            name=label,
            mode="lines+markers",
            line_color=COLORS[i],
        )
    fig.add_hline(y=1, line_dash="dash", line_color="#64748B")
    fig.update_yaxes(title="מכפיל שערים בבית מול חוץ")
    return finish(fig, "יתרון הבית בכל עונת ולידציה")


def explanation(name: str, result: dict[str, Any], *, after: bool) -> str:
    if name == "f5_shap":
        if not after:
            return (
                "כל נקודה היא חציון של ממוצע התרומה המוחלטת על שורות ולידציה בכל קיפול. "
                "הפס מציג את 50% האמצעיים בין 15 הקיפולים. SHAP מחלק את הלוגריתם של λ "
                "לתרומות ולערך בסיס. תרומה חתומה יכולה להיות חיובית או שלילית. "
                'תרומה של <bdi dir="ltr">+0.1</bdi> מכפילה את קצב השערים בערך ב-1.105. '
                "בכדורגל זה מסביר מה משך תחזית לשערים כלפי מעלה או מטה."
            )
        top = result["importance"].iloc[-1]
        flags = result["flags"].set_index("feature")
        own = flags.loc[FEATURE_NAMES[FLAGS[0]], "mean_abs"]
        opp = flags.loc[FEATURE_NAMES[FLAGS[1]], "mean_abs"]
        return (
            f"התרומה המוחלטת הגדולה ביותר היא של {FEATURE_NAMES[top.name]}, "
            f"עם חציון {top['median']:.4f}. פער Elo ו-Elo לפני המשחק מתואמים ומתחלקים "
            f"בקרדיט. בשורות שבהן כל דגל פעיל, ממוצע התרומה המוחלטת הוא {own:.4f} "
            f"לדגל הקבוצה ו-{opp:.4f} לדגל היריבה. הטבלה מציגה גם את הכיוון. "
            "אלה ערכים בסולם log λ. הם מתארים את התחזית ומבחן ההוצאה מתאר את השינוי בביצועיה."
        )
    if not after:
        return (
            "בכל שורת ולידציה חוזים פעמיים. מחליפים רק את דגל הבית ומחלקים את תחזית "
            "הבית בתחזית החוץ. מחשבים ממוצע גאומטרי, כלומר ממצעים את לוגריתם היחס "
            "ואז חוזרים לסולם המכפילים. המכפיל חיובי. למשל, 1.10 פירושו 10% יותר "
            "שערים צפויים לאותה קבוצה מול אותו יריב ובאותו כושר. בגרף כל נקודה היא "
            "עונה אחת. הטבלה מסכמת את סטיית התקן בין עונות לפי שלב."
        )
    p = result["home"].set_index("season_id")
    return (
        f"המכפיל על פני כל שורות הוולידציה הוא {result['overall']:.3f}. "
        f"בעונה הראשונה {p.iloc[0].model:.3f}, ובעונה האחרונה {p.iloc[-1].model:.3f}. "
        f"בעונות 2019/20 ו-2020/21 התקבלו {p.loc[21].model:.3f} ו-{p.loc[22].model:.3f}. "
        "בפלייאוף 2019/20 וברוב 2020/21 שיחקו בלי קהל. זו השוואה תיאורית. "
        "ההחלפה מחזיקה קבוע את מדדי ההבקעה והספיגה לפי מגרש, שבהם יש ביתיות נוספת. "
        "היחס הנצפה כולל את כל השפעות המגרש, וההחלפה אינה אומדן סיבתי."
    )
