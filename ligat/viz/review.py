"""Plots for the three approved notebook 02 review diagnostics."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz.league_stats import AWAY_COLOR, HOME_COLOR, _finish
from ligat.viz.seasons import season_label


def playoff_persistence_chart(deviations: pd.DataFrame, summary: pd.DataFrame) -> Any:
    """Show within-season/group deviations, not the mechanical fall/rise in group means."""
    figure = make_subplots(rows=1, cols=2, subplot_titles=["עליון", "תחתון"])
    for col, (stage, color) in enumerate((("upper", HOME_COLOR), ("lower", AWAY_COLOR)), 1):
        data = deviations.loc[deviations.final_stage.eq(stage)].reset_index()
        row = summary.loc[stage]
        figure.add_trace(
            go.Scatter(
                x=data.regular_deviation,
                y=data.playoff_deviation,
                mode="markers",
                marker_color=color,
                name="עונת קבוצה",
                showlegend=col == 1,
                customdata=np.column_stack(
                    [data.team_name, [season_label(int(s)) for s in data.season_id]]
                ),
                hovertemplate="%{customdata[0]}, עונה %{customdata[1]}"
                "<br>סדיר: %{x:.2f}<br>פלייאוף: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        x = np.array([data.regular_deviation.min(), data.regular_deviation.max()])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=row.intercept + row.slope * x,
                mode="lines",
                line_color=color,
                name=f"שיפוע {row.slope:.2f}, מתאם {row.pearson:.2f}",
                showlegend=True,
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="קצב בסדיר פחות ממוצע משתתפות אותו פלייאוף", row=1, col=col)
        figure.update_yaxes(
            title_text="קצב בפלייאוף פחות ממוצע משתתפות אותו פלייאוף", row=1, col=col
        )
    figure.update_layout(
        title="שימור יתרון יחסי: מתאם "
        f"{summary.loc['upper', 'pearson']:.2f} בעליון מול "
        f"{summary.loc['lower', 'pearson']:.2f} בתחתון"
    )
    return _finish(figure).update_layout(height=700, margin=dict(t=140, b=145))
