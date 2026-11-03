"""Hebrew tables and read-only EXP-001 diagnostic charts for notebook 03 B1."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.prematch_b_review import OUTCOMES
from ligat.viz.league_stats import DRAW_COLOR, GOOD_SCALE, _finish
from ligat.viz.presentation import TableDisplay, display_table
from ligat.viz.seasons import season_label

NAMES = {
    "label": "תצורה",
    "games": "משחקים",
    "log_loss": "log loss",
    "rps": "RPS",
    "rank_all": "דירוג ב־15",
    "rank_last5": "דירוג ב־5 האחרונות",
    "last5_log_loss": "log loss ב־5 האחרונות",
    "additions": "מספר תוספות",
    "paired_mean_difference": "פער מזווג מהמובילה",
    "paired_se": "שגיאת תקן לפער",
    "fold": "קיפול",
    "season": "עונה",
    "beta": "β",
    "theta1": "θ₁",
    "theta2": "θ₂",
    "home_probability_change_pp": "שינוי בניצחון בית ל־100 נקודות (נ״א)",
    "rank_loso": "דירוג LOSO",
    "rank_walk_forward": "דירוג walk-forward ברשת הגסה",
    "loso_log_loss": "log loss ב־LOSO",
    "walk_forward_log_loss": "log loss קדימה",
    "accuracy": "דיוק argmax",
    "max_draw": "הסתברות תיקו מרבית",
    "argmax_draws": "תחזיות תיקו ב־argmax",
    "observed_draw": "תיקו בפועל",
    "mean_draw": "הסתברות תיקו ממוצעת",
    "teams": "קבוצות",
    "spearman": "ספירמן",
    "exact_percent": "מקום זהה (%)",
    "within_one_percent": "עד ±1 (%)",
    "within_two_percent": "עד ±2 (%)",
    "team_name": "קבוצה",
    "elo_start": "Elo בתחילת העונה",
    "elo_end": "Elo בסוף העונה",
    "position": "מיקום ספורטיבי",
    "elo_rank": "מיקום לפי Elo",
    "position_difference": "מיקום Elo פחות ספורטיבי",
    "E": "E — ניקוד צפוי",
    "Δ": "Δ — עדכון",
    "E0050 log loss": "E0050 log loss",
    "E0050 RPS": "E0050 RPS",
    "B0 log loss": "B0 log loss",
}
PARAMETERS = {
    "k": "K — תגובה למשחק",
    "home_advantage": "יתרון בית בעדכון",
    "season_regression": "נסיגה בקיץ",
    "use_goal_multiplier": "מכפיל הפרש שערים",
    "promoted_initialization": "אתחול עולה",
    "dead_game_discount": "משקל משחק מת",
}


def review_table(
    frame: pd.DataFrame, *, columns: list[str] | None = None, percent: tuple[str, ...] = ()
) -> TableDisplay:
    """Isolate seasons/numbers in RTL and wrap the explanatory words column."""
    data = frame.copy()
    if "season_id" in data:
        data["season"] = data.season_id.map(lambda sid: season_label(int(sid)))
        data = data.drop(columns="season_id")
        if columns is None:
            data = data[["season", *[name for name in data if name != "season"]]]
    chosen = list(data.columns) if columns is None else columns
    names = {column: NAMES.get(column, column) for column in chosen}
    precision = {column: 4 for column in chosen}
    precision.update(
        {
            column: 6
            for column in (
                "log_loss",
                "rps",
                "last5_log_loss",
                "paired_mean_difference",
                "paired_se",
                "loso_log_loss",
                "walk_forward_log_loss",
            )
        }
    )
    precision.update(
        {
            column: 0
            for column in (
                "games",
                "fold",
                "rank_all",
                "rank_last5",
                "additions",
                "teams",
                "elo_rank",
                "position",
                "position_difference",
                "rank_loso",
                "rank_walk_forward",
                "argmax_draws",
                "קבוצות־עונות",
                "הפרש שערים",
            )
        }
    )
    precision.update(
        {
            "E0050 log loss": 6,
            "E0050 RPS": 6,
            "B0 log loss": 6,
            "מקום זהה (%)": 1,
            "עד ±1 (%)": 1,
            "עד ±2 (%)": 1,
        }
    )
    return display_table(
        data,
        names,
        columns=chosen,
        percent=percent,
        precision=precision,
        show_index=False,
        unicode_minus=tuple(chosen),
    )


def _finish_chart(figure: Any, *, height: int = 560) -> Any:
    _finish(figure)
    figure.update_layout(
        height=height,
        margin=dict(l=85, r=85, t=110, b=100),
        legend_title_text="",
        colorway=[GOOD_SCALE[2], GOOD_SCALE[8], GOOD_SCALE[5]],
    )
    return figure


def marginal_chart(marginals: pd.DataFrame) -> Any:
    """Coarse-grid means; associations across configurations, not causal effects."""
    figure = make_subplots(
        rows=2, cols=3, subplot_titles=list(PARAMETERS.values()), vertical_spacing=0.25
    )
    for index, parameter in enumerate(PARAMETERS):
        subset = marginals.loc[marginals.parameter.eq(parameter)].copy()
        subset["display"] = subset.value.astype(str).replace(
            {
                "True": "עם",
                "False": "בלי",
                "fixed_1450": "1450 קבוע",
                "previous_promoted_mean": "ממוצע עולות",
            }
        )
        numeric = pd.to_numeric(subset.value, errors="coerce")
        if numeric.notna().all():
            subset["display"] = numeric.map(lambda value: f"{value:g}")
        figure.add_trace(
            go.Scatter(
                x=subset.display,
                y=subset.mean_log_loss,
                mode="lines+markers",
                customdata=np.column_stack([subset.configs, subset.best_log_loss]),
                hovertemplate=(
                    "%{x}<br>ממוצע log loss: %{y:.6f}<br>צירופים: %{customdata[0]}"
                    "<br>הטוב ביותר: %{customdata[1]:.6f}<extra></extra>"
                ),
                showlegend=False,
                marker_color=GOOD_SCALE[index % len(GOOD_SCALE)],
            ),
            row=index // 3 + 1,
            col=index % 3 + 1,
        )
        figure.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=subset.display.tolist(),
            row=index // 3 + 1,
            col=index % 3 + 1,
        )
    shown = marginals.loc[marginals.parameter.isin(PARAMETERS), "mean_log_loss"]
    padding = (shown.max() - shown.min()) * 0.08
    figure.update_yaxes(
        title_text="ממוצע log loss",
        range=[float(shown.min() - padding), float(shown.max() + padding)],
        tickformat=".4f",
    )
    figure.update_layout(title_text="EXP-001: ממוצע על הרשת הגסה לפי פרמטר — קשר תיאורי, לא סיבתי")
    return _finish_chart(figure, height=850)


def time_weight_chart(stage2: pd.DataFrame) -> Any:
    data = stage2.copy()
    data["weight"] = data.half_life.map(
        lambda x: "ללא" if pd.isna(x) else ("עונה אחת" if x == 1 else f"{x:g} עונות")
    )
    summary = (
        data.groupby("weight", sort=False)
        .agg(loss=("log_loss", "mean"), configs=("config_id", "size"))
        .reset_index()
    )
    order = ["ללא", "עונה אחת", "2 עונות", "4 עונות"]
    summary["order"] = summary.weight.map({label: index for index, label in enumerate(order)})
    summary = summary.sort_values("order")
    figure = px.line(
        summary,
        x="weight",
        y="loss",
        markers=True,
        custom_data=["configs"],
        labels={"weight": "זמן מחצית למשקל עונה", "loss": "ממוצע log loss"},
        title="משקלי זמן: אותם 10 בסיסי Elo — קשר תיאורי, לא סיבתי",
    )
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=order,
    )
    figure.update_traces(
        hovertemplate=(
            "%{x}<br>ממוצע log loss: %{y:.6f}<br>בסיסי Elo: %{customdata[0]}<extra></extra>"
        )
    )
    return _finish_chart(figure)


def season_score_chart(scores: pd.DataFrame) -> Any:
    figure = go.Figure()
    for column, title in (
        ("log_loss_elo", "E0050 — Elo עם המרה סדורה"),
        ("log_loss_b0", "B0 — שכיחות היסטורית"),
    ):
        figure.add_scatter(
            x=scores.season,
            y=scores[column],
            mode="lines+markers",
            name=title,
            customdata=scores[["games"]],
            hovertemplate=(
                "%{x}<br>log loss: %{y:.6f}<br>משחקים: %{customdata[0]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    figure.update_layout(
        title="E0050 מול B0: עונת ולידציה אחרי עונת ולידציה",
        xaxis_title="עונת ולידציה",
        yaxis_title="log loss — נמוך עדיף",
    )
    figure.update_xaxes(categoryorder="array", categoryarray=scores.season.tolist(), tickangle=-45)
    return _finish_chart(figure)


def calibration_chart(bins: pd.DataFrame) -> Any:
    figure = make_subplots(rows=1, cols=3, subplot_titles=list(OUTCOMES))
    size_reference = 2 * float(bins.games.max()) / 22**2
    for index, outcome in enumerate(OUTCOMES):
        data = bins.loc[bins.outcome.eq(outcome)]
        small = data.games.lt(30)
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color=DRAW_COLOR),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=index + 1,
        )
        figure.add_trace(
            go.Scatter(
                x=data.predicted,
                y=data.observed,
                mode="lines+markers",
                customdata=np.column_stack([data.games, np.where(small, "מדגם קטן", "")]),
                showlegend=False,
                cliponaxis=False,
                marker=dict(
                    color=GOOD_SCALE[index],
                    size=data.games,
                    sizemode="area",
                    sizeref=size_reference,
                    sizemin=5,
                    symbol=np.where(small, "circle-open", "circle"),
                ),
                hovertemplate=(
                    "צפוי: %{x:.1%}<br>נצפה: %{y:.1%}<br>משחקים: %{customdata[0]}"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=index + 1,
        )
    figure.update_xaxes(range=[0, 1], tickformat=".0%", title_text="הסתברות ממוצעת בבין")
    figure.update_yaxes(range=[0, 1], tickformat=".0%", title_text="שכיחות בפועל")
    figure.update_layout(title="כיול E0050 מחוץ למדגם: בית / תיקו / חוץ — 10 בינים קבועים")
    return _finish_chart(figure, height=550)


def draw_histogram(predictions: pd.DataFrame) -> Any:
    figure = px.histogram(
        predictions,
        x="p_draw",
        nbins=30,
        labels={"p_draw": "הסתברות חזויה לתיקו"},
        title="E0050: התפלגות ההסתברות לתיקו במשחקי הוולידציה",
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_layout(yaxis_title="משחקים", bargap=0.04)
    return _finish_chart(figure)


def confusion_chart(matrix: pd.DataFrame) -> Any:
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(OUTCOMES),
            y=list(OUTCOMES),
            colorscale="Blues",
            text=matrix.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="בפועל: %{y}<br>argmax: %{x}<br>משחקים: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="argmax של E0050: אף משחק לא מסווג כתיקו",
        xaxis_title="הקטגוריה בעלת ההסתברות הגבוהה ביותר",
        yaxis_title="מה קרה בפועל",
    )
    figure.update_yaxes(autorange="reversed")
    return _finish_chart(figure)


def alignment_chart(teams: pd.DataFrame) -> Any:
    data = teams.copy()
    data["season"] = data.season_id.map(lambda sid: season_label(int(sid)))
    figure = px.scatter(
        data,
        x="position",
        y="elo_rank",
        hover_name="team_name",
        opacity=0.55,
        custom_data=["season", "elo_end", "position_difference"],
        labels={"position": "מקום ספורטיבי בסיום", "elo_rank": "מקום לפי Elo בסיום"},
        title="בדיקה בדיעבד: Elo בסוף העונה מול המקום הספורטיבי — כל הליגה",
    )
    figure.update_traces(
        hovertemplate=(
            "%{hovertext}<br>%{customdata[0]}<br>ספורטיבי: %{x}<br>Elo: %{y}"
            "<br>דירוג Elo: %{customdata[1]:.1f}"
            "<br>פער מקומות: %{customdata[2]}<extra></extra>"
        )
    )
    figure.add_scatter(
        x=[1, 16],
        y=[1, 16],
        mode="lines",
        line=dict(dash="dash", color=DRAW_COLOR),
        showlegend=False,
        hoverinfo="skip",
    )
    figure.update_xaxes(dtick=1, range=[0.5, 16.5])
    figure.update_yaxes(dtick=1, range=[0.5, 16.5])
    return _finish_chart(figure)
