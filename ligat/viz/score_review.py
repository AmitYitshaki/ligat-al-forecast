"""Chapter D score diagnostics; simulation bands describe the fitted null pipeline."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz.league_stats import AWAY_COLOR, DRAW_COLOR, HOME_COLOR, _finish
from ligat.viz.seasons import season_label


def top_scores_chart(scores: pd.DataFrame) -> Any:
    """Actual exact scores, Wilson intervals and the fixed H/D/A colors."""
    fig = go.Figure()
    labels = [f"{h}:{a}" for h, a in scores.index]
    outcomes = [
        "ניצחון בית" if h > a else "תיקו" if h == a else "ניצחון חוץ" for h, a in scores.index
    ]
    for name, color in (
        ("ניצחון בית", HOME_COLOR),
        ("תיקו", DRAW_COLOR),
        ("ניצחון חוץ", AWAY_COLOR),
    ):
        mask = [outcome == name for outcome in outcomes]
        data = scores.loc[mask]
        fig.add_bar(
            x=data.rate * 100,
            y=np.array(labels)[mask],
            orientation="h",
            name=name,
            marker_color=color,
            error_x=dict(
                type="data",
                symmetric=False,
                array=(data.ci_upper - data.rate) * 100,
                arrayminus=(data.rate - data.ci_lower) * 100,
            ),
            customdata=data.n,
            hovertemplate="%{y}: %{x:.2f}%<br>n=%{customdata}<extra></extra>",
        )
    fig.update_layout(
        title="גם התוצאה השכיחה מתארת רק כ־12% מהמשחקים",
        xaxis_title="אחוז משחקים ורווח Wilson של 95%",
        yaxis=dict(
            type="category", categoryorder="array", categoryarray=labels, autorange="reversed"
        ),
    )
    return _finish(fig).update_layout(legend=dict(orientation="h", y=-0.2))


def score_ratios_chart(comparison: pd.DataFrame, null: pd.DataFrame) -> Any:
    """One observed/expected comparison, two baselines, calibrated null 95% ranges."""
    x = np.arange(len(comparison))
    labels = ["כל התיקו" if s == "all_draws" else s for s in comparison.index]
    fig = go.Figure()
    for label, column, shift, color in (
        ("גלובלי", "global_ratio", -0.13, HOME_COLOR),
        ("לכל משחק", "per_game_ratio", 0.13, AWAY_COLOR),
    ):
        fig.add_scatter(
            x=x + shift,
            y=comparison[column],
            mode="markers",
            name=label,
            marker=dict(color=color, size=11),
            customdata=labels,
            hovertemplate="%{customdata}: %{y:.3f}<extra></extra>",
        )
    bands = null.loc[comparison.index]
    # A separate marker at the null mean locates asymmetric bands independently of observations.
    fig.add_scatter(
        x=x + 0.13,
        y=bands.null_mean,
        mode="markers",
        name="95% של תהליך הסימולציה",
        marker=dict(color=DRAW_COLOR, symbol="line-ew", size=12),
        error_y=dict(
            type="data",
            symmetric=False,
            array=bands.null_upper - bands.null_mean,
            arrayminus=bands.null_mean - bands.null_lower,
        ),
        customdata=labels,
        hovertemplate="%{customdata}: מרכז האפס %{y:.3f}<extra></extra>",
    )
    fig.add_hline(y=1, line_dash="dash", line_color=DRAW_COLOR)
    fig.update_layout(
        title="עודף תיקו נבדק מול רעש האומדן, לא רק מול יחס 1",
        xaxis=dict(tickvals=x, ticktext=labels),
        yaxis_title="נצפה / צפוי",
    )
    return _finish(fig).update_layout(legend=dict(orientation="h", y=-0.23), height=600)


def score_stability_chart(deviations: pd.DataFrame) -> Any:
    """Seasonal deviations in percentage points, not unadjusted counts."""
    categories = ["אחר" if c == "other" else c for c in deviations.columns]
    seasons = [season_label(int(s)) for s in deviations.index]
    values = deviations.to_numpy() * 100
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=categories,
            y=seasons,
            zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="נקודות אחוז"),
            hovertemplate="%{y}, %{x}<br>סטייה: %{z:.2f} נק׳ אחוז<extra></extra>",
        )
    )
    fig.update_layout(
        title="התפלגות התוצאות יציבה; צבע מציג סטייה מהממוצע",
        xaxis=dict(type="category", categoryorder="array", categoryarray=categories),
        yaxis=dict(
            type="category", categoryorder="array", categoryarray=seasons, autorange="reversed"
        ),
    )
    fig = _finish(fig).update_layout(height=790, margin=dict(l=160, r=100))
    # Signed deviations require a diverging scale; the general palette is sequential.
    fig.data[0].colorscale = "RdBu_r"
    return fig


def calibrated_dispersion_chart(null: pd.DataFrame, raw: pd.Series) -> Any:
    """Raw versus strength-adjusted metrics; null ranges apply to adjusted metrics only."""
    keys = ["dispersion_home", "dispersion_away", "residual_correlation", "all_draws"]
    labels = ["פיזור בית", "פיזור חוץ", "מתאם", "יחס כל התיקו"]
    data = null.loc[keys]
    fig = go.Figure()
    x = np.arange(4)
    fig.add_scatter(
        x=x - 0.13,
        y=raw.loc[keys],
        mode="markers",
        name="גולמי",
        marker=dict(color=HOME_COLOR, size=11),
    )
    fig.add_scatter(
        x=x + 0.13,
        y=data.observed,
        mode="markers",
        name="אחרי כוח",
        marker=dict(color=AWAY_COLOR, size=11),
    )
    fig.add_scatter(
        x=x + 0.13,
        y=data.null_mean,
        mode="markers",
        name="95% של תהליך הסימולציה",
        marker=dict(color=DRAW_COLOR, symbol="line-ew", size=12),
        error_y=dict(
            type="data",
            symmetric=False,
            array=data.null_upper - data.null_mean,
            arrayminus=data.null_mean - data.null_lower,
        ),
    )
    fig.update_layout(
        title="גולמי מול אחרי כוח: אין עדות לפיזור יתר או לתלות שיורית",
        xaxis=dict(tickvals=x, ticktext=labels),
        yaxis_title="ערך המדד",
    )
    return _finish(fig).update_layout(legend=dict(orientation="h", y=-0.22), height=600)
