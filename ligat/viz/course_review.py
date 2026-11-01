"""Chapter C rank transitions and benchmark curves; no data loading or model fitting."""

import re
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz.league_history import ERA_NAMES
from ligat.viz.league_stats import GOOD_SCALE, _finish
from ligat.viz.presentation import rtl_figure

REGION_NAMES = {
    "top": "צמרת 1–3",
    "upper_line": "קו העליון 4–6",
    "middle_table": "אמצע 7–10",
    "bottom": "תחתית 11–14",
}
REGION_COLORS = {
    "top": "#2563eb",
    "upper_line": "#16a34a",
    "middle_table": "#9333ea",
    "bottom": "#ea580c",
}


def transition_chart(matrix: pd.DataFrame, *, at_split: bool = False) -> Any:
    """Identically scaled count heatmaps: integer ranks, fractional credits and row totals."""
    n = float(matrix.sum(axis=1).mean())
    current = "מקום בפיצול" if at_split else "מקום אחרי 10 משחקים"
    figure = go.Figure(
        go.Heatmap(
            x=list(matrix.columns),
            y=list(matrix.index),
            z=matrix.to_numpy(),
            zmin=0,
            zmax=n,
            colorscale=GOOD_SCALE,
            colorbar_title="קבוצות־עונה",
            text=[
                ["" if value == 0 else f"{value:.2f}".rstrip("0").rstrip(".") for value in row]
                for row in matrix.to_numpy()
            ],
            texttemplate="%{text}",
            hovertemplate=current
            + " %{y}<br>מקום סיום %{x}<br>קבוצות־עונה: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            "בפיצול: האליפות כמעט סגורה, התחתית עדיין פתוחה"
            if at_split
            else "אחרי 10 משחקים: צמרת מרוכזת, תחתית מתפזרת"
        ),
        xaxis_title="מקום סיום רשמי",
        yaxis_title=current,
    )
    figure.update_xaxes(tickmode="linear", dtick=1, range=[0.5, 14.5])
    figure.update_yaxes(tickmode="linear", dtick=1, range=[14.5, 0.5])
    return _finish(figure).update_layout(height=800, margin=dict(t=110, b=90, l=100, r=100))


def region_chart(frame: pd.DataFrame) -> Any:
    """Four fixed colors, a percentage axis and a marker for the mechanical split constraint."""
    figure = go.Figure()
    for region, name in REGION_NAMES.items():
        data = frame.xs(region, level="region")
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data.same_region_rate * 100,
                name=name,
                mode="lines",
                line=dict(color=REGION_COLORS[region], width=3),
                customdata=data.n,
                hovertemplate="אחרי %{x} משחקים: %{y:.1f}%<br>"
                "קבוצות־עונה משוקללות: %{customdata}<extra>%{fullData.name}</extra>",
            )
        )
    figure.add_vline(x=26, line_dash="dash", line_color="#777777", annotation_text="פיצול")
    figure.update_layout(
        title="הצמרת מתייצבת אחרי 14 משחקים; השאר לקראת הפיצול",
        xaxis_title="משחקים לכל קבוצה בסדיר",
        yaxis_title="סיימו באותו אזור (%)",
    )
    result = _finish(figure)
    for trace, region in zip(result.data, REGION_NAMES, strict=True):
        trace.line.color = REGION_COLORS[region]
    return result.update_layout(
        height=650, yaxis_range=[0, 100], xaxis_range=[1, 26], margin=dict(t=115, b=100)
    )


def settling_chart(frame: pd.DataFrame) -> Any:
    """14-club season interval as a band; earlier eras are context without error bars."""
    figure = go.Figure()
    for era in ("12_teams", "16_teams", "14_teams"):
        if era not in frame.index.get_level_values("era"):
            continue
        data = frame.xs(era, level="era")
        if era == "14_teams":
            figure.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.spearman_ci_upper,
                    mode="lines",
                    line_width=0,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.spearman_ci_lower,
                    mode="lines",
                    line_width=0,
                    fill="tonexty",
                    fillcolor="rgba(37,99,235,0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data.spearman_mean,
                name=ERA_NAMES[era],
                mode="lines",
                line=dict(
                    color="#2563eb" if era == "14_teams" else "#aaaaaa",
                    width=3 if era == "14_teams" else 1,
                    dash="solid" if era != "16_teams" else "dot",
                ),
                customdata=data.n_seasons,
                hovertemplate="k=%{x}; מתאם %{y:.2f}<br>עונות: %{customdata}"
                "<extra>%{fullData.name}</extra>",
            )
        )
    value = frame.loc[("14_teams", 10), "spearman_mean"]
    figure.add_vline(x=26, line_dash="dash", line_color="#777777", annotation_text="פיצול בעידן 14")
    figure.update_layout(
        title=f"עידן 14: אחרי 10 משחקים מתאם הסדר לסיום {value:.2f}",
        xaxis_title="משחקים לכל קבוצה בסדיר",
        yaxis_title="מתאם סדר הטבלה לסיום",
    )
    result = _finish(figure)
    for trace in result.data:
        plain = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", str(trace.name))
        if plain == ERA_NAMES["14_teams"]:
            trace.line.color = "#2563eb"
        elif trace.showlegend is not False:
            trace.line.color = "#aaaaaa"
    return rtl_figure(
        result.update_layout(yaxis_range=[0, 1], height=650, margin=dict(t=115, b=95))
    )
