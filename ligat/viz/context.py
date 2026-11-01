"""Previous-season value gaps and home advantage; no data access or fitting."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz.league_stats import AWAY_COLOR, HOME_COLOR, _finish

GAP_LABELS = ["פער קטן ביותר", "פער קטן", "בינוני", "גדול", "פער גדול ביותר"]


def _error(frame: pd.DataFrame, metric: str, scale: float = 1) -> dict[str, Any]:
    return dict(
        type="data",
        symmetric=False,
        array=((frame[f"{metric}_ci_upper"] - frame[metric]) * scale).clip(lower=0),
        arrayminus=((frame[metric] - frame[f"{metric}_ci_lower"]) * scale).clip(lower=0),
    )


def upset_chart(frame: pd.DataFrame) -> Any:
    """Weak-side win rates with unchanged Wilson intervals, plus ratio edges in hover."""
    values = frame.xs("value_ratio", level="metric")
    fig = go.Figure()
    edges = frame.attrs["value_ratio_edges"]
    for side, name, color in (
        ("home_weaker", "החלשה בבית", HOME_COLOR),
        ("away_weaker", "החלשה בחוץ", AWAY_COLOR),
    ):
        group = values.xs(side, level="weaker_side").sort_index()
        bins = [int(label[1:]) - 1 for label in group.index]
        ranges = [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in bins]
        fig.add_scatter(
            x=[GAP_LABELS[i] for i in bins],
            y=group.weak_win_rate * 100,
            mode="lines+markers",
            name=name,
            line_color=color,
            error_y=_error(group, "weak_win_rate", 100),
            customdata=np.column_stack([group.n, ranges]),
            hovertemplate="%{x}: %{y:.1f}%<br>n=%{customdata[0]}"
            "<br>יחס שווי גדול/קטן: %{customdata[1]}<extra></extra>",
        )
    home = values.xs("home_weaker", level="weaker_side").sort_index()
    away = values.xs("away_weaker", level="weaker_side").sort_index()
    fig.update_layout(
        title=f"בפער גדול ביותר: החלשה מנצחת {home.weak_win_rate.iloc[-1]:.1%} בבית "
        f"ו־{away.weak_win_rate.iloc[-1]:.1%} בחוץ",
        xaxis=dict(
            title="פער נתח שווי בעונה הקודמת",
            type="category",
            categoryorder="array",
            categoryarray=GAP_LABELS,
        ),
        yaxis_title="ניצחון החלשה (%)",
    )
    return _finish(fig).update_layout(height=700, margin=dict(t=135, b=110, l=75, r=65))
