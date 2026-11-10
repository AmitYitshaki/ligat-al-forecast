"""R2 illustrations of saved ordinal probabilities, grid interaction and time decay."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.prematch_b_review import OUTCOMES
from ligat.viz.prematch_b import _finish_chart
from ligat.viz.presentation import rtl_figure


def probability_curve_chart(frame: pd.DataFrame) -> Any:
    """Show all three saved last-fold probabilities without a second home offset."""
    figure = go.Figure()
    for outcome in OUTCOMES:
        figure.add_scatter(x=frame.gap, y=frame[outcome], name=outcome, mode="lines")
    figure.update_layout(
        title="המרה סדורה: פער הדירוג מזיז את שלוש ההסתברויות",
        xaxis_title="Elo הבית פחות החוץ, בלי תוספת בית",
        yaxis_title="הסתברות",
    )
    figure.update_xaxes(range=[-300, 300])
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    return _finish_chart(figure)


def interaction_chart(frame: pd.DataFrame) -> Any:
    """Print the coarse-grid mean in each K/regression cell and mark each row minimum."""
    values = frame.pivot(index="k", columns="season_regression", values="log_loss")
    x = [f"{100 * number:g}%" for number in values.columns]
    y = [f"K={number:g}" for number in values.index]
    figure = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=x,
            y=y,
            text=values.to_numpy(),
            texttemplate="%{text:.5f}",
            colorbar=dict(title="log loss"),
            hovertemplate="%{y}<br>נסיגה %{x}<br>ממוצע %{z:.6f}<extra></extra>",
        )
    )
    for row, series in values.iterrows():
        figure.add_annotation(
            x=f"{100 * series.idxmin():g}%",
            y=f"K={row:g}",
            text="★",
            xshift=45,
            showarrow=False,
        )
    figure.update_layout(
        title="תגובה בעונה מול מחיקה בקיץ: ממוצע על 1,280 הצירופים",
        xaxis_title="החלק שנמחק בקיץ",
        yaxis_title="K — גודל התגובה",
    )
    figure.update_yaxes(
        autorange="reversed", type="category", categoryarray=y, categoryorder="array"
    )
    figure.update_xaxes(type="category", categoryarray=x, categoryorder="array")
    return _finish_chart(figure)


def decay_chart(profiles: pd.DataFrame, differences: pd.DataFrame, reference_gain: float) -> Any:
    """Separate the meaning of time weights from the paired observed score changes."""
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["מה כל אפשרות עושה", "מה יצא"], horizontal_spacing=0.15
    )
    for label, group in profiles.groupby("decay", sort=False):
        figure.add_trace(
            go.Scatter(x=group.season, y=group.weight, name=label, mode="lines+markers"),
            row=1,
            col=1,
        )
    labels = ["ללא", "כל עונה", "כל 2 עונות", "כל 4 עונות"]
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=differences.difference,
            mode="markers",
            error_y=dict(type="data", array=differences.se, visible=True),
            showlegend=False,
            customdata=differences[["log_loss", "decay"]],
            hovertemplate=(
                "%{customdata[1]}<br>פער %{y:.6f}<br>log loss %{customdata[0]:.6f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    figure.add_hline(y=0, line_dash="dash", row=1, col=2)
    figure.update_xaxes(title_text="עונת אימון של קיפול 15", tickangle=-45, row=1, col=1)
    figure.update_yaxes(title_text="משקל", range=[0, 1.06], row=1, col=1)
    figure.update_xaxes(title_text="דעיכה — המשקל נחצה", row=1, col=2)
    figure.update_yaxes(title_text="הפסד פחות ללא דעיכה (±SE)", tickformat=".5f", row=1, col=2)
    figure.update_layout(title="דעיכה משנה את ההמרה בלבד: עשרה בסיסי Elo נשארים קבועים")
    figure = _finish_chart(figure, height=820)
    figure.update_layout(
        margin=dict(l=85, r=85, t=180, b=200),
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
    )
    figure.update_xaxes(title_standoff=55, automargin=True, row=1, col=1)
    figure.update_yaxes(title_standoff=35, automargin=True, row=1, col=2)
    figure.add_annotation(
        x=0.5,
        y=-0.38,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            f"קנה מידה: הפער בין B0 ל-E0050 הוא {reference_gain:.3f}; "
            "פסי השגיאה מתארים SE בין 15 קיפולים"
        ),
    )
    return rtl_figure(figure)
