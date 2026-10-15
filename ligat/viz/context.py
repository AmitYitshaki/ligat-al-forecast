"""Chapter E context plots, including exact reference funnels and exposure-aware labels."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.league_stats.context import funnel_limits_binomial, funnel_limits_poisson
from ligat.viz.league_stats import AWAY_COLOR, DRAW_COLOR, HOME_COLOR, _finish


def _error(frame: pd.DataFrame, metric: str, scale: float = 1) -> dict[str, Any]:
    return dict(
        type="data",
        symmetric=False,
        array=((frame[f"{metric}_ci_upper"] - frame[metric]) * scale).clip(lower=0),
        arrayminus=((frame[metric] - frame[f"{metric}_ci_lower"]) * scale).clip(lower=0),
    )


def _funnel_traces(frame: pd.DataFrame, metric: str, exposure: str, scale: float = 1) -> list[Any]:
    model = frame.attrs["funnel_models"][metric]
    data = frame.loc[frame.n_matches.ge(model["min_matches"]) & frame[exposure].gt(0)]
    fn = funnel_limits_binomial if model["distribution"] == "binomial" else funnel_limits_poisson
    n = np.unique(np.geomspace(data[exposure].min(), data[exposure].max(), 160).astype(int))
    bands = fn(model["rate"], n)
    traces = []
    for field, name, color, dash in (
        ("95", "95% רגיל", HOME_COLOR, "dash"),
        ("998", "99.8% רגיל", DRAW_COLOR, "dot"),
    ):
        for bound in ("lower", "upper"):
            traces.append(
                go.Scatter(
                    x=bands.index,
                    y=bands[f"{bound}{field}"] * scale,
                    mode="lines",
                    line=dict(color=color, dash=dash),
                    name=name,
                    legendgroup=name,
                    showlegend=bound == "lower",
                )
            )
    if model["widened"]:
        for field, name, dash in (("95", "95% מורחב", "solid"), ("998", "99.8% מורחב", "dashdot")):
            for bound in ("lower", "upper"):
                values = model["rate"] + np.sqrt(model["phi"]) * (
                    bands[f"{bound}{field}"] - model["rate"]
                )
                traces.append(
                    go.Scatter(
                        x=bands.index,
                        y=values.clip(lower=0) * scale,
                        mode="lines",
                        line=dict(color="#9333ea", dash=dash),
                        name=name,
                        legendgroup=name,
                        showlegend=bound == "lower",
                    )
                )
    traces.append(
        go.Scatter(
            x=[n.min(), n.max()],
            y=[model["rate"] * scale] * 2,
            mode="lines",
            line=dict(color="black"),
            name="ממוצע הליגה",
        )
    )
    traces.append(
        go.Scatter(
            x=data[exposure],
            y=data[metric] * scale,
            mode="markers",
            marker=dict(color=AWAY_COLOR, size=8),
            name="נצפה",
            text=data.index.astype(str),
            customdata=np.column_stack([data.n_matches, data[exposure]]),
            hovertemplate="%{text}<br>%{y:.3f}<br>משחקים=%{customdata[0]}"
            "<br>חשיפה=%{customdata[1]}<extra></extra>",
        )
    )
    return traces


def chapter_e_chart(kind: str, frame: pd.DataFrame) -> Any:
    """Display contextual associations with uncertainty and sample sizes, never rankings."""
    data = frame.reset_index()
    if kind == "referee_home":
        model = frame.attrs["funnel_models"]["home_win_rate"]
        visible = frame.loc[frame.n_matches.ge(model["min_matches"])]
        inside = visible.home_win_rate.between(
            visible.home_win_rate_lower95, visible.home_win_rate_upper95
        ).sum()
        fig = go.Figure(_funnel_traces(frame, "home_win_rate", "n_matches", 100))
        fig.update_layout(
            title=f"{inside} מתוך {len(visible)} שופטים בתוך גבולות 95% לניצחון בית",
            xaxis_title="מספר משחקים (סולם לוגריתמי)",
            xaxis_type="log",
            yaxis_title="ניצחון בית (%)",
        )
    elif kind in ("referee_events", "venues"):
        if kind == "referee_events":
            metrics = [
                ("red_cards_per_match", "red_cards_matches", "אדומים למשחק"),
                ("penalty_goals_per_match", "penalty_goals_matches", "שערי פנדל למשחק"),
            ]
        else:
            metrics = [
                ("home_win_rate", "n_matches", "ניצחון בית"),
                ("goals_per_match", "n_matches", "שערים למשחק"),
            ]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[x[2] for x in metrics])
        for i, (metric, exposure, _) in enumerate(metrics, start=1):
            for trace in _funnel_traces(frame, metric, exposure):
                trace.showlegend = bool(
                    trace.showlegend is not False
                    and (i == 1 or (kind == "venues" and "מורחב" in trace.name))
                )
                fig.add_trace(trace, row=1, col=i)
            fig.update_xaxes(type="log", title_text="משחקי חשיפה", row=1, col=i)
        if kind == "referee_events":
            red = frame.attrs["funnel_models"]["red_cards_per_match"]
            penalty = frame.attrs["funnel_models"]["penalty_goals_per_match"]
            title = f"פיזור בין שופטים: φ={red['phi']:.2f} באדומים, "
            title += f"{penalty['phi']:.2f} בשערי פנדל; גבולות מורחבים לצד הרגילים"
        else:
            goals = frame.attrs["funnel_models"]["goals_per_match"]
            title = f"{len(frame)} אצטדיונים: פיזור שערים φ={goals['phi']:.2f}; "
            title += "זהות המארחות מערבת חוזק וביתיות"
        fig.update_layout(title=title)
    elif kind == "coaches":
        matched = frame.loc["matched"]
        labels = ["שינוי בקבוצה", "שינוי בבקרה", "הפרש השינויים"]
        coach_metrics = ["mean_diff", "mean_control_diff", "mean_effect"]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[matched[m] for m in coach_metrics],
                marker_color=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[matched[f"{m}_ci_upper"] - matched[m] for m in coach_metrics],
                    arrayminus=[matched[m] - matched[f"{m}_ci_lower"] for m in coach_metrics],
                ),
            )
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(
            title=f"{int(matched.n_changes)} החלפות עם בקרה: שינוי "
            f"{matched.mean_diff:+.2f}, בקרה {matched.mean_control_diff:+.2f}, "
            f"הפרש {matched.mean_effect:+.2f} נקודות למשחק",
            yaxis_title="שינוי בנקודות למשחק; קשר, לא סיבה",
        )
    elif kind == "borrowed":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["נקודות בית למשחק", "ניצחון בית (%)"])
        for i, (metric, scale) in enumerate((("home_ppm", 1), ("home_win_rate", 100)), start=1):
            fig.add_trace(
                go.Bar(
                    x=["שאול" if not home else "ביתי" for home in frame.index],
                    y=frame[metric] * scale,
                    marker_color=[AWAY_COLOR, HOME_COLOR],
                    error_y=_error(frame, metric, scale),
                    customdata=frame.n,
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.2f}<br>n=%{customdata}",
                ),
                row=1,
                col=i,
            )
        fig.update_layout(
            title=f"באצטדיון שאול {frame.loc[False, 'home_ppm']:.2f} "
            f"נקודות מול {frame.loc[True, 'home_ppm']:.2f} בבית; אותן קבוצות־עונות"
        )
    elif kind == "game_types":
        order = [
            k for k in ("city_derby", "shared_venue_other", "big_game", "other") if k in frame.index
        ]
        data = frame.loc[order]
        type_labels = {
            "city_derby": "דרבי עירוני",
            "shared_venue_other": "אצטדיון משותף אחר",
            "big_game": "גדולה מול גדולה",
            "other": "שאר המשחקים",
        }
        fig = make_subplots(rows=1, cols=2, subplot_titles=["תיקו (%)", "שערים למשחק"])
        for i, (metric, scale) in enumerate((("draw_rate", 100), ("goals_per_match", 1)), start=1):
            fig.add_trace(
                go.Bar(
                    x=[type_labels[s] for s in data.index],
                    y=data[metric] * scale,
                    error_y=_error(data, metric, scale),
                    customdata=data.n,
                    marker_color=HOME_COLOR,
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.2f}<br>n=%{customdata}",
                ),
                row=1,
                col=i,
            )
        derby = data.loc["city_derby", "draw_rate"]
        other = data.loc["other", "draw_rate"]
        fig.update_layout(
            title=f"תיקו בדרבי עירוני {derby:.1%} מול {other:.1%} בשאר; "
            "רווחים חופפים אינם הוכחת זהות"
        )
    elif kind == "upsets":
        values = frame.xs("value_ratio", level="metric")
        fig = go.Figure()
        for side, name, color in (
            ("home_weaker", "החלשה בבית", HOME_COLOR),
            ("away_weaker", "החלשה בחוץ", AWAY_COLOR),
        ):
            group = values.xs(side, level="weaker_side").sort_index()
            fig.add_trace(
                go.Scatter(
                    x=group.index,
                    y=group.weak_win_rate * 100,
                    mode="lines+markers",
                    name=name,
                    line_color=color,
                    error_y=_error(group, "weak_win_rate", 100),
                    customdata=group.n,
                    hovertemplate="%{x}: %{y:.1f}%<br>n=%{customdata}",
                )
            )
        home = values.xs("home_weaker", level="weaker_side").sort_index()
        away = values.xs("away_weaker", level="weaker_side").sort_index()
        fig.update_layout(
            title=f"החלשה בבית: {home.weak_win_rate.iloc[0]:.1%} בפער קטן, "
            f"{home.weak_win_rate.iloc[-1]:.1%} בגדול; בחוץ: "
            f"{away.weak_win_rate.iloc[0]:.1%} בקטן, {away.weak_win_rate.iloc[-1]:.1%} בגדול",
            xaxis_title="חמישון יחס שווי t−1 (Q1 הקטן ביותר)",
            yaxis_title="ניצחון החלשה (%)",
        )
    else:
        raise ValueError(f"Unknown chapter E chart: {kind}")
    return _finish(fig).update_layout(
        title_x=0.97, title_xanchor="right", height=700, margin=dict(t=155, b=105, l=75, r=65)
    )
