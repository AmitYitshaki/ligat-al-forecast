"""Seven descriptive charts for notebook 02 chapter B, using already computed tables."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz.league_stats import AWAY_COLOR, DRAW_COLOR, HOME_COLOR, _finish

OBJECTIVES = {
    "champion": "אלופה",
    "upper_line": "קו העליון — סוף השלב הסדיר",
    "safety": "שורדת — הניקוד הנמוך ביותר",
    "relegated_max": "יורדת — הניקוד הגבוה ביותר",
}
COLORS = [HOME_COLOR, "#16a34a", DRAW_COLOR, AWAY_COLOR]
ERA_NAMES = {"12_teams": "12 קבוצות", "16_teams": "16 קבוצות", "14_teams": "14 קבוצות"}
CALENDAR_NAMES = {
    "Saturday": "שבת",
    "Sunday": "ראשון",
    "Monday": "שני",
    "other": "אחר",
    "before_18": "עד 17:59",
    "18_to_20": "18:00–19:59",
    "20_plus": "20:00 ומעלה",
    "unknown": "חסר במקור",
}


def _years(frame: pd.DataFrame) -> list[str]:
    return [f"{int(s) + 1998}/{str(int(s) + 1999)[-2:]}" for s in frame.index]


def _errors(frame: pd.DataFrame, metric: str, scale: float = 1) -> dict[str, Any]:
    return dict(
        type="data",
        symmetric=False,
        array=(frame[f"{metric}_ci_upper"] - frame[metric]) * scale,
        arrayminus=(frame[metric] - frame[f"{metric}_ci_lower"]) * scale,
    )


def chapter_b_chart(kind: str, frame: pd.DataFrame, summary: pd.DataFrame | None = None) -> Any:
    """Show objective lines, uncertainty and denominators without loading data or fitting models."""
    if kind == "thresholds":
        fig = go.Figure()
        for (key, name), color in zip(OBJECTIVES.items(), COLORS, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=_years(frame),
                    y=frame[f"{key}_ppm"],
                    name=name,
                    mode="lines+markers",
                    line_color=color,
                    customdata=frame[f"{key}_games"],
                    hovertemplate=(
                        "%{x}: %{y:.2f} נק׳ למשחק<br>משחקים: %{customdata}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
        fig.update_layout(
            title="ספי האליפות וההישרדות משתנים בין העונות; קו העליון נמדד מוקדם יותר",
            yaxis_title="נקודות למשחק",
        )
    elif kind == "threshold_boxes":
        era = frame.loc[frame.era.eq("14_teams")]
        fig = go.Figure()
        for (key, name), color in zip(OBJECTIVES.items(), COLORS, strict=True):
            fig.add_trace(
                go.Box(
                    y=era[f"{key}_points"],
                    name=name,
                    boxpoints="all",
                    marker_color=color,
                    text=_years(era),
                    customdata=era[f"{key}_games"],
                    hovertemplate=(
                        "%{text}: %{y} נק׳<br>משחקים: %{customdata}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
        median = era.safety_points.median()
        fig.update_layout(
            title=f"חציון השורדת הנמוכה בעידן 14 הקבוצות: {median:.0f} נקודות; אין הבטחת הישרדות",
            yaxis_title="נקודות על המגרש",
            showlegend=False,
        )
    elif kind == "home":
        fig = go.Figure(
            go.Scatter(
                x=_years(frame),
                y=frame.home_win_rate * 100,
                mode="lines+markers",
                name="ניצחון בית",
                line_color=HOME_COLOR,
                error_y=_errors(frame, "home_win_rate", 100),
                customdata=frame.n,
                hovertemplate="%{x}: %{y:.1f}%<br>n=%{customdata}<extra></extra>",
            )
        )
        overall = (frame.home_win_rate * frame.n).sum() / frame.n.sum()
        fig.update_layout(
            title=f"הבית ניצח ב־{overall:.1%} מהמשחקים; שיעור הביתיות משתנה לאורך הזמן",
            yaxis_title="ניצחונות בית (%)",
        )
        for season in (21, 22):
            if season in frame.index:
                fig.add_vline(
                    x=_years(frame.loc[[season]])[0], line_dash="dot", line_color=AWAY_COLOR
                )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1.14,
            showarrow=False,
            text="2019/20–2020/21: קורונה וקהל — הקשר היסטורי לאימות, לא נתון שנמדד כאן",
        )
    elif kind == "eras":
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=["שערים למשחק", "הפרש של 3 ומעלה", "פיזור נקודות למשחק"]
        )
        for col, metric in enumerate(("goals_per_match", "big_margin_rate", "std_ppm"), 1):
            scale = 100 if metric == "big_margin_rate" else 1
            errors = _errors(frame, metric, scale) if metric != "std_ppm" else None
            fig.add_trace(
                go.Bar(
                    x=[ERA_NAMES[e] for e in frame.index],
                    y=frame[metric] * scale,
                    error_y=errors,
                    marker_color=HOME_COLOR,
                    customdata=frame.n_matches,
                    hovertemplate="%{x}: %{y:.2f}<br>n=%{customdata}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_layout(
            title="הבדלי העידנים מופיעים בשערים ובפיזור; בחירת חלון האימון נשארת למחקר",
            showlegend=False,
        )
    elif kind == "season_trends":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["שערים למשחק", "שיעורי תיקו ו־0:0"])
        fig.add_trace(
            go.Scatter(
                x=_years(frame), y=frame.goals_per_match, name="שערים", line_color=HOME_COLOR
            ),
            row=1,
            col=1,
        )
        for metric, name, color in (
            ("draw_rate", "תיקו", DRAW_COLOR),
            ("zero_zero_rate", "0:0", AWAY_COLOR),
        ):
            fig.add_trace(
                go.Scatter(
                    x=_years(frame),
                    y=frame[metric] * 100,
                    name=name,
                    line_color=color,
                    customdata=frame.n_matches,
                    hovertemplate=(
                        "%{x}: %{y:.1f}%<br>n=%{customdata}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=2,
            )
        fig.update_layout(title="שערים ותיקו אינם קבועים לאורך השנים — משתמשים בסיכום העונה הקיים")
    elif kind == "balance":
        if summary is None:
            raise ValueError("Champion counts are required for the balance chart")
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["פיזור בין הקבוצות", "אליפויות לפי מועדון"]
        )
        fig.add_trace(
            go.Scatter(
                x=_years(frame), y=frame.std_ppm, name="סטיית תקן של ppm", line_color=HOME_COLOR
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=summary.team_name, y=summary.titles, name="אליפויות", marker_color=AWAY_COLOR),
            row=1,
            col=2,
        )
        fig.update_layout(
            title=(
                f"{len(summary)} מועדונים זכו ב־{int(summary.titles.sum())} אליפויות; "
                "הפיזור בין הקבוצות משתנה"
            ),
            showlegend=False,
        )
    elif kind == "calendar":
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "ימים — שערים",
                "שעות — שערים",
                "ימים — ניצחונות בית",
                "שעות — ניצחונות בית",
            ],
        )
        for col, dimension in enumerate(("weekday", "kickoff_hour"), 1):
            data = frame.xs(dimension, level="dimension")
            order = (
                ["Saturday", "Sunday", "Monday", "other", "unknown"]
                if dimension == "weekday"
                else ["before_18", "18_to_20", "20_plus", "unknown"]
            )
            data = data.reindex([v for v in order if v in data.index])
            for row, metric in enumerate(("goals_per_match", "home_win_rate"), 1):
                scale = 100 if row == 2 else 1
                fig.add_trace(
                    go.Scatter(
                        x=[CALENDAR_NAMES[v] for v in data.index],
                        y=data[metric] * scale,
                        mode="markers",
                        marker_color=HOME_COLOR,
                        error_y=_errors(data, metric, scale),
                        customdata=data.n,
                        hovertemplate="%{x}: %{y:.2f}<br>n=%{customdata}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
        fig.update_layout(
            title="יום ושעה מציגים קשרים תיאוריים עם טווחי אי־ודאות; אין כאן הוכחה להשפעה",
            showlegend=False,
        )
    else:
        raise ValueError(f"Unknown chapter B chart: {kind}")
    return _finish(fig).update_layout(
        title_x=0.97, title_xanchor="right", margin={"t": 125, "b": 95, "l": 70, "r": 70}
    )
