"""Compare signed F5 validation rates and illustrate the training gap."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import yaml

from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goals_runner import _payload_digest
from ligat.models.ordered_logit import time_weights
from ligat.viz.goals_draw_calibration import equal_count_bins
from ligat.viz.goals_f5_diagnostics import verified_models
from ligat.viz.goals_notebook import COLORS, finish
from ligat.viz.goals_walk_review import folds
from ligat.viz.seasons import season_label

LABELS = {"F5": "F5", "glm": "GLM בבסיס", "maher": "Maher", "R1": "R1: Elo ובית"}


def signed_rows(data: Any, bundle: dict[str, Any], root: Path, name: str) -> Any:
    frames = []
    for fold in folds(bundle, name).itertuples():
        source = json.loads((root / "full" / (fold.unit_id + ".json")).read_bytes())
        payload = source["payload"]
        if source["sha256"] != _payload_digest(payload):
            raise ValueError("Invalid signed validation unit")
        frames.append(pd.DataFrame(payload["per_team"]))
    frame = pd.concat(frames, ignore_index=True).set_index(["game_id", "team_id"]).sort_index()
    if set(frame.season_id) != set(range(11, 26)) or not frame.index.is_unique:
        raise ValueError("Invalid validation coverage")
    frame["goals"] = data.y.reindex(frame.index)
    frame["residual"] = frame.goals - frame["lambda"]
    frame["nll"] = poisson_nll(frame.goals, frame["lambda"])
    return frame


def team_residuals(frame: Any) -> Any:
    """Attribute conceded residual to the defending team via the opponent key."""
    offense = frame.reset_index()
    opponent = offense.loc[:, ["game_id", "is_home", "team_id"]].assign(
        is_home=lambda p: ~p.is_home.astype(bool)
    )
    defense = offense.drop(columns="team_id").merge(
        opponent, on=["game_id", "is_home"], validate="one_to_one", how="left"
    )
    if defense.team_id.isna().any():
        raise ValueError("Missing opponent's defensive row")
    both = pd.concat([offense.assign(side="הבקעה"), defense.assign(side="ספיגה")])
    table = both.groupby(["team_id", "side"]).residual.agg(["mean", "sem", "size"]).reset_index()
    table["flag"] = table["mean"].abs().gt(2 * table["sem"])
    return table


def rate_calibration(frame: Any) -> Any:
    """Tie-safe quantile groups, with mean goals and sampling SE by group."""
    rates = frame["lambda"]
    groups = equal_count_bins(rates / rates.max()) + 1
    return (
        frame.assign(bin=groups)
        .groupby("bin", as_index=False)
        .agg(
            predicted=("lambda", "mean"),
            actual=("goals", "mean"),
            se=("goals", "sem"),
            rows=("goals", "size"),
        )
    )


def training_gap(data: Any, root: Path, f5: Any) -> Any:
    manifest, models = verified_models(root)
    rows = []
    for item in manifest["folds"]:
        season = item["season_id"]
        valid = f5.loc[f5.season_id.eq(season)]
        (history,) = {tuple(seq) for seq in valid.train_seasons}
        train = data.context.season_id.isin(history)
        context = data.context.loc[train]
        if context.date.max() >= pd.to_datetime(valid.as_of, utc=True).min():
            raise ValueError("Training comparison crosses the held-out as_of")
        model = models[season]
        columns = list(model._schema or ())
        x = data.X.loc[train, [column[0] for column in columns]]
        score = poisson_nll(data.y[train], model.predict(x, context=context))
        weights = time_weights(context.season_id, season, 4)
        rows.append(
            dict(
                season_id=season,
                training=float(np.mean(score)),
                weighted=float(np.average(score, weights=weights)),
                validation=float(valid.nll.mean()),
                training_rows=len(score),
                validation_rows=len(valid),
            )
        )
    return pd.DataFrame(rows)


def review(data: Any, bundle: dict[str, Any], root: Path, prematch: Path) -> dict[str, Any]:
    sources = {name: signed_rows(data, bundle, root, name) for name in LABELS}
    f5 = sources["F5"]
    for name in LABELS:
        scores = folds(bundle, name)
        if float(np.average(scores.log_loss, weights=scores.games)) < 0.98:
            raise ValueError(f"H/D/A leakage threshold crossed: {name}")
    names = {
        int(r["team_id"]): r["name_he"]
        for r in yaml.safe_load(Path("config/teams.yaml").read_text(encoding="utf-8"))["teams"]
    }
    deep = f5.loc[f5.deep_fold]
    residuals = team_residuals(deep)
    residuals["team"] = residuals.team_id.map(names)
    if residuals.team.isna().any():
        raise ValueError("Missing canonical Hebrew team name")
    residuals = residuals.drop(columns="team_id")
    calib = rate_calibration(deep)
    calib_sides = pd.concat(
        [
            rate_calibration(deep.loc[deep.is_home.eq(home)]).assign(side=label)
            for home, label in ((True, "בית"), (False, "חוץ"))
        ],
        ignore_index=True,
    )
    common = deep.index[deep.season_id.ge(13)]
    bins = np.floor(data.X.loc[common, "elo_diff"].astype(float) / 100).astype(int) * 100
    elo = pd.DataFrame({"bin": bins, "actual": f5.loc[common, "goals"]})
    for name in ("F5", "glm", "R1"):
        elo[name] = sources[name].loc[common, "lambda"]
    elo_curve = (
        elo.groupby("bin").mean().join(elo.groupby("bin").size().rename("rows")).reset_index()
    )
    features = pd.read_parquet(
        prematch,
        columns=["round", "promoted_in"],
        filters=[("season_id", ">=", 11), ("season_id", "<=", 25)],
    ).reindex(f5.index)
    if features.isna().any().any():
        raise ValueError("Missing prematch round or promotion flag")
    bands = pd.cut(features["round"], [0, 5, 15, np.inf], labels=["1–5", "6–15", "16 ומעלה"])
    parts = []
    for name in ("F5", "glm", "maher"):
        rows = sources[name].assign(band=bands)
        for population, selected in (
            ("כל הקבוצות", rows),
            ("עולות", rows.loc[features.promoted_in.eq(1)]),
        ):
            parts.append(
                selected.groupby("band", observed=True)
                .agg(nll=("nll", "mean"), residual=("residual", "mean"), rows=("nll", "size"))
                .reset_index()
                .assign(model=LABELS[name], population=population)
            )
    return dict(
        residuals=residuals,
        calibration=calib,
        calibration_sides=calib_sides,
        elo=elo_curve,
        stages=pd.concat(parts, ignore_index=True),
        training=training_gap(data, root, f5),
    )


def figure(name: str, result: dict[str, Any]) -> Any:
    fig = go.Figure()
    if name == "f5_residuals":
        p = result["residuals"]
        order = p.loc[p.side.eq("הבקעה")].sort_values("mean").team.tolist()
        for i, (side, group) in enumerate(p.groupby("side")):
            fig.add_scatter(
                x=group["mean"],
                y=group.team,
                name=side,
                mode="markers",
                marker=dict(color=COLORS[i], symbol=np.where(group.flag, "diamond", "circle")),
                error_x=dict(array=2 * group["sem"]),
                customdata=group["size"],
                hovertemplate="%{y}<br>%{x:.3f}<br>n=%{customdata}<extra>%{fullData.name}</extra>",
            )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_yaxes(categoryorder="array", categoryarray=order)
        fig.update_xaxes(title="שערים בפועל פחות λ")
        fig = finish(fig, "שאריות F5 לפי קבוצה בעונות העמוקות", height=1150)
        fig.update_layout(margin_l=260)
        return fig
    if name == "f5_rate_calibration":
        p = result["calibration"]
        hi = max(float(p.predicted.max()), float((p.actual + p.se).max()))
        fig.add_scatter(
            x=[0, hi],
            y=[0, hi],
            mode="lines",
            name="כיול מושלם",
            line=dict(color="#64748B", dash="dash"),
        )
        fig.add_scatter(
            x=p.predicted,
            y=p.actual,
            mode="markers",
            name="עשירוני λ",
            error_y=dict(array=p.se),
            customdata=p.rows,
        )
        fig.update_xaxes(title="שערים צפויים בממוצע")
        fig.update_yaxes(title="שערים בפועל בממוצע")
        return finish(fig, "כיול קצב השערים של F5")
    if name == "f5_elo_curve":
        p = result["elo"]
        for i, field in enumerate(("F5", "glm", "R1", "actual")):
            fig.add_scatter(
                x=p.bin + 50,
                y=p[field],
                mode="markers+lines",
                customdata=p.rows,
                name=LABELS.get(field, "שערים בפועל"),
                line_color=COLORS[i],
                hovertemplate=(
                    "מרכז מקטע %{x}<br>%{y:.3f}<br>n=%{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        fig.update_xaxes(title="פער Elo, מרכז מקטע של 100 נקודות")
        fig.update_yaxes(title="שערים לקבוצה בממוצע")
        return finish(fig, "קצב שערים לפי פער Elo בעונות המשותפות")
    p = result["training"]
    for i, (field, label) in enumerate(
        (
            ("training", "אימון בלי משקלים"),
            ("weighted", "אימון עם דעיכה"),
            ("validation", "ולידציה"),
        )
    ):
        fig.add_scatter(
            x=p.season_id.map(season_label),
            y=p[field],
            mode="markers+lines",
            name=label,
            line_color=COLORS[i],
        )
    fig.update_yaxes(title="NLL ממוצע לשורת קבוצה")
    return finish(fig, "משחקים שהעצים ראו באימון מול משחקי ולידציה")


def explanation(name: str, result: dict[str, Any], *, after: bool) -> str:
    if name == "f5_residuals":
        p = result["residuals"]
        flagged = p.loc[p.flag]
        if after:
            names = ", ".join(flagged.team.unique()) if len(flagged) else "אין"
            return (
                f"ב-{p.team.nunique()} קבוצות התקבלו {len(flagged)} סימונים, בקבוצות הבאות: "
                f"{names}. שארית חיובית בהבקעה פירושה שהקבוצה כבשה יותר מהתחזית. "
                "שארית חיובית בספיגה פירושה שהיריבות כבשו יותר מהתחזית. כשבודקים כ-30 קבוצות "
                "צפויות בערך חריגה אחת או שתיים לכל מדד גם תחת מודל תקין. רצף משחקים תלוי "
                "ומספר הקבוצות שנבדקו מגבילים את המסקנה שאפשר לייחס לקבוצה מסוימת."
            )
        return (
            "נקודה מציגה שערים בפועל פחות תחזית בממוצע לכל קבוצה בעונות העמוקות. לדוגמה, "
            "אם ממוצע ההבקעה היה 1.5 והתחזית 1.3, שארית ההבקעה היא 0.2 שערים למשחק. "
            "לשארית אין גבול קבוע. הספיגה משתמשת בשורה של היריבה, והפס הוא שתי שגיאות תקן "
            "לכל כיוון, לפי סטיית התקן בין משחקים חלקי שורש הספירה. יהלום מסמן פס שאינו כולל אפס."
        )
    if name == "f5_rate_calibration":
        p = result["calibration"]
        if after:
            return (
                f"הקבוצות בתרשים כוללות {p.rows.min()}–{p.rows.max()} שורות כל אחת. "
                f"בקבוצה הנמוכה ניבא F5 בממוצע {p.iloc[0].predicted:.2f} שערים לקבוצה, "
                f"ונרשמו {p.iloc[0].actual:.2f}. הטבלאות מפרידות בית וחוץ ומחשבות את "
                "הגבולות לכל צד. בכדורגל ההפרדה מאפשרת לראות אם אותה רמת ביטחון בתפוקת שערים "
                "מעלה פער שונה אצל מארחת ואצל אורחת, גם כאשר התוצאות במצטבר קרובות לתחזית."
            )
        return (
            "מסדרים את שורות הקבוצה בעונות העמוקות לפי λ ומקבצים לעשירונים בלי לפצל "
            "ערכים זהים. הצירים חיוביים ומודדים שערים לקבוצה. לדוגמה, נקודה סביב 1.2 "
            "בשני הצירים אומרת שבמשחקים שנחזו בהם 1.2 שערים בממוצע גם הובקעו 1.2. "
            "פס אנכי מציג שגיאת תקן אחת של ממוצע השערים. הקו האלכסוני מציין התאמה בין "
            "התחזית לתוצאה הממוצעת בקבוצה."
        )
    if name == "f5_elo_curve":
        p = result["elo"].set_index("bin")
        if after:
            change = p.loc[0, "F5"] - p.loc[-100, "F5"]
            return (
                f"ב-F5 המעבר ממקטע פער Elo שבין מינוס 100 ל-0 למקטע שבין 0 ל-100 "
                f"משנה את התחזית הממוצעת ב-{change:+.2f} שערים לקבוצה. הקו מאפשר להשוות "
                "את השיפוע במקטעים נוספים, ובקצוות כדאי לקרוא את הספירה בריחוף. "
                "אלה ממוצעים של משחקים בפועל עם פערים שונים. גם הביתיות והיריבה משתנות "
                "בין המקטעים, ולכן ההפרש הוא תיאורי ואינו תגובה לשינוי Elo בלבד."
            )
        return (
            "הציר האופקי מציג פער Elo של הקבוצה פחות היריבה. כל נקודה מסכמת מקטע של "
            "100 נקודות בעונות 13–25, למשל פערים מ-0 ועד פחות מ-100 במרכז 50. "
            "הציר האנכי מציג שערים לקבוצה, עם שלוש תחזיות וממוצע בפועל על אותן שורות. "
            "שיפוע שעולה או נחלש לאורך המקטעים מראה כיצד ההבדל בכוח מתבטא בתחזית. "
            "מספר השורות זמין בריחוף לכל מקטע."
        )
    p = result["training"]
    if after:
        last = p.iloc[-1]
        return (
            f"בקיפול האחרון NLL האימון ללא משקלים הוא {last.training:.4f}, עם דעיכה "
            f"{last.weighted:.4f}, והוולידציה {last.validation:.4f}. מדד אימון נמוך יכול "
            "לשקף התאמה למשחקים שכבר נראו. גם הרכב העונות והקושי שלהן משפיעים על ההפרש. "
            "ב-M3 משתמשים ב-λ מחוץ לקיפול, כדי ששלב הכיול יתמודד עם תחזיות של משחקים "
            "שהעץ טרם ראה, כפי שיקרה בעונה עתידית."
        )
    return (
        "כל נקודה מודדת NLL ממוצע לשורת קבוצה, מאפס כלפי מעלה, באחד מ-15 הקיפולים. "
        "למשל, 1.35 מייצג פחות הפתעה מ-1.45. אותו עץ שחוזה את עונת הוולידציה "
        "חוזה כאן גם את משחקי עונות האימון. קו אחד מחשב ממוצע אימון רגיל, והשני "
        "משתמש במשקלי הדעיכה המקוריים שנותנים משקל גבוה יותר לעונות קרובות. "
        "הקו השלישי מודד את התחזיות החתומות של המשחקים מחוץ לאימון."
    )
