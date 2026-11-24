"""Read-only notebook 05 views; no fitting, selection, or implicit stale-cache reuse."""

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_runner import exp005_status, load_goal_inputs
from ligat.viz.presentation import rtl_figure, rtl_text
from ligat.viz.seasons import season_label

COLORS = ("#2563EB", "#9333EA", "#F59E0B", "#0891B2", "#DB2777", "#4F46E5")
BLUE_SCALE = ["#DBEAFE", "#93C5FD", "#2563EB", "#1E3A8A"]


def finish(figure: Any, title: str, *, height: int = 550) -> Any:
    """Shared RTL rendering, preserving embeddings without forbidden direction marks."""
    figure.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=14),
        height=height,
        title=dict(text="<br>".join(textwrap.wrap(title, width=65)), x=0.97, xanchor="right"),
        margin=dict(t=135, b=110, l=85, r=85),
        legend=dict(title=None, orientation="h", y=-65 / (height - 245), yanchor="top"),
    )
    rtl_figure(figure)
    for axis in figure.select_xaxes():
        if axis.title.text and not axis.tickangle:
            axis.title.standoff = 20

    def clean(value: Any, field: str = "") -> Any:
        if isinstance(value, str):
            if field in {"text", "name", "hovertemplate", "hovertext", "ticktext"}:
                value = rtl_text(value)
            return value.replace(chr(0x200E), "").replace(chr(0x200F), "")
        if isinstance(value, list):
            return [clean(v, field) for v in value]
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        return value

    return go.Figure(clean(json.loads(figure.to_json())))


def development_data() -> GoalData:
    """Frozen prematch and matches, explicitly restricted to seasons 8 through 25."""
    return load_goal_inputs(load_goal_experiment())


def pilot_status() -> pd.DataFrame:
    """Validate C5 unit/journal pairs before offering any saved training views."""
    return pd.DataFrame(exp005_status(load_goal_experiment()))


def chain_figure() -> Any:
    figure = go.Figure()
    names = [
        "M0: מידע לפני המשחק",
        "M1: מוטיבציה כמועמד",
        "M2: שני ערכי λ",
        "M3: מטריצת תוצאות",
        "סימולציה: טבלת עונה",
    ]
    for i, name in enumerate(names):
        x = 4 - i
        figure.add_annotation(
            x=x, y=0.5, text=name, showarrow=False, bgcolor="#DBEAFE", borderpad=12
        )
        if i:
            figure.add_annotation(
                x=x + 0.35,
                y=0.5,
                ax=x + 0.65,
                ay=0.5,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
            )
    figure.update_xaxes(visible=False, range=[-0.6, 4.6])
    figure.update_yaxes(visible=False, range=[0, 1])
    return finish(figure, "שני ערכי λ מחברים את המידע המוקדם לסימולציה")


def season_counts(data: GoalData) -> pd.DataFrame:
    counts = data.context.groupby("season_id").size().rename("rows").reset_index()
    counts["season"] = counts.season_id.map(season_label)
    counts["games"] = counts.rows // 2
    return counts


def season_counts_figure(data: GoalData) -> Any:
    counts = season_counts(data)
    figure = go.Figure(
        go.Bar(
            x=counts.season,
            y=counts.rows,
            marker_color=COLORS[0],
            customdata=counts.games,
            hovertemplate="%{x}<br>שורות: %{y}<br>משחקים: %{customdata}<extra></extra>",
        )
    )
    figure.update_xaxes(title="עונה", tickangle=-45)
    figure.update_yaxes(title="שורות קבוצה במשחק", rangemode="tozero")
    return finish(figure, f"לכל משחק שתי שורות: {len(data.y):,} שורות בסך הכול")


def walk_forward_figure() -> Any:
    matrix = [[1 if s < v else (2 if s == v else 0) for s in range(8, 26)] for v in range(11, 26)]
    labels = [season_label(s) for s in range(8, 26)]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels[3:],
            zmin=0,
            zmax=2,
            colorscale=[
                [0, "#F3F4F6"],
                [0.499, "#F3F4F6"],
                [0.5, "#93C5FD"],
                [0.749, "#93C5FD"],
                [0.75, "#2563EB"],
                [1, "#2563EB"],
            ],
            colorbar=dict(tickvals=[0, 1, 2], ticktext=["טרם נצפה", "אימון קודם", "ולידציה"]),
        )
    )
    figure.update_xaxes(title="עונת הנתונים", tickangle=-45)
    figure.update_yaxes(title="עונת הוולידציה", autorange="reversed")
    return finish(figure, "15 קיפולים מתקדמים בזמן; שלוש עונות חימום בלבד", height=850)


def decay_figure(*, daily: bool = False) -> Any:
    ages = np.linspace(0, 8, 161) if daily else np.arange(0, 18)
    grid = [None, 0.5, 1, 2] if daily else [None, 1, 2, 4]
    figure = go.Figure()
    for i, half in enumerate(grid):
        weights = np.ones(len(ages)) if half is None else 0.5 ** (ages / half)
        figure.add_trace(
            go.Scatter(
                x=ages,
                y=weights,
                name="בלי דעיכה" if half is None else f"מחצית חיים {half:g} עונות",
                line=dict(color=COLORS[i]),
            )
        )
    figure.update_xaxes(title="גיל בימים / 365.25" if daily else "מרחק מעונת האימון האחרונה בעונות")
    figure.update_yaxes(title="משקל באימון", range=[0, 1.05])
    return finish(
        figure,
        "דעיכה קצרה נותנת פחות משקל למשחקים ישנים"
        + (" — Maher יומי" if daily else " — אימון לקיפול"),
    )


def poisson_example(data: GoalData) -> tuple[Any, pd.DataFrame]:
    key = data.context.index[0]
    import yaml

    names = {
        r["team_id"]: r["name_he"]
        for r in yaml.safe_load(Path("config/teams.yaml").read_text(encoding="utf-8"))["teams"]
    }
    opponent = next(t for g, t in data.context.index if g == key[0] and t != key[1])
    observed = int(data.y.loc[key])
    nll = float(-poisson.logpmf(observed, 1.6))
    x = list(range(8)) + ["8+"]
    probability = [*poisson.pmf(np.arange(8), 1.6), poisson.sf(7, 1.6)]
    figure = go.Figure(go.Bar(x=[str(v) for v in x], y=probability, marker_color=COLORS[0]))
    figure.add_annotation(
        x=str(observed) if observed < 8 else "8+",
        y=probability[min(observed, 8)],
        text=f"נצפו {observed} שערים; NLL={nll:.3f}",
        showarrow=True,
        ay=-45,
    )
    figure.update_xaxes(title="שערים של קבוצה במשחק")
    figure.update_yaxes(title="הסתברות מדויקת במודל", range=[0, 0.4])
    example = pd.DataFrame(
        [
            dict(
                game_id=key[0],
                team=names[key[1]],
                opponent=names[opponent],
                season=season_label(int(data.context.loc[key, "season_id"])),
                observed=observed,
                lambda_example=1.6,
                nll_example=nll,
            )
        ]
    )
    return finish(figure, "λ של 1.6 משאיר סיכוי גם לאפס וגם לשלושה שערים"), example


def hyperparameter_table() -> pd.DataFrame:
    spec = load_goal_experiment().raw
    meaning = {
        "max_depth": "מספר פיצולים בדרך: אינטראקציות בין כוח, בית ושלב",
        "min_child_weight": "כמה מידע נדרש לפני פיצול: זהירות מול דוגמאות מעטות",
        "n_estimators": "כמה תיקונים קטנים מוסיפים לתחזית",
        "learning_rate": "כמה כל תיקון משנה את λ",
        "subsample": "חלק משחקי האימון בכל עץ; מצמצם התאמת יתר",
        "colsample_bytree": "חלק הפיצ׳רים בכל עץ; מונע הסתמכות על רמז יחיד",
        "reg_lambda": "קנס על גודל תיקוני העלים; לא קנס במדד ההכרעה",
        "depth": "מורכבות הקשרים שעץ יכול ללמוד",
        "l2_leaf_reg": "קנס ריבועי על העלים",
        "iterations": "מספר העצים באימון",
        "one_hot_max_size": "שלב בן ארבע קטגוריות מקודד כן/לא; מזהי קבוצה נשארים ב-CTR",
        "half_life": "כמה עונות עד שמשקל משחק ישן נחצה",
        "library_default": "תצורה נפרדת מלאה של הספרייה, לא ערבוב ברירות מחדל",
    }
    rows = []
    for name, values in [
        ("XGBoost A", spec["xgboost"]["stage_a"]),
        ("XGBoost B", spec["xgboost"]["stage_b"]),
        ("CatBoost", spec["catboost"]),
        ("GLM", {"half_life": spec["glm"]["half_life"]}),
        ("Maher", spec["maher"]),
    ]:
        for parameter, value in values.items():
            rows.append(
                dict(
                    model=name,
                    parameter=parameter,
                    values=str(value),
                    meaning=meaning[parameter],
                    rationale="רשת שנקבעה מראש: קצוות ואמצע; ה-ס15, ה-ש13",
                )
            )
    return pd.DataFrame(rows)


def frozen_prior() -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path("config/m2_maher_prior.yaml").read_text(encoding="utf-8"))
    return dict(value["prior"])
