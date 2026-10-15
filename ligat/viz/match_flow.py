"""Chapter D presentation of descriptive match flow; no I/O, model or feature fitting."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz.league_stats import AWAY_COLOR, DRAW_COLOR, HOME_COLOR, _finish

COLORS = {"win": HOME_COLOR, "draw": DRAW_COLOR, "loss": AWAY_COLOR}
NAMES = {"win": "ניצחון", "draw": "תיקו", "loss": "הפסד"}


def _errors(frame: pd.DataFrame, value: str, scale: float = 1) -> dict[str, Any]:
    return dict(
        type="data",
        symmetric=False,
        array=(frame[f"{value}_ci_upper"] - frame[value]) * scale,
        arrayminus=(frame[value] - frame[f"{value}_ci_lower"]) * scale,
    )


def chapter_d_chart(kind: str, frame: pd.DataFrame) -> Any:
    """Render D1–D11 with data-derived numeric findings, sample sizes and explicit uncertainty."""
    data = frame.reset_index()
    if kind == "scorelines":
        fig = px.imshow(
            frame.to_numpy() * 100,
            x=frame.columns,
            y=frame.index,
            text_auto=".1f",
            color_continuous_scale="Blues",
            labels=dict(x="שערי חוץ", y="שערי בית", color="% משחקים"),
        )
        maximum = frame.stack().idxmax()
        if "ci_lower" in frame.attrs:
            fig.update_traces(
                customdata=np.stack(
                    [
                        frame.attrs["counts"],
                        np.array(frame.attrs["ci_lower"]) * 100,
                        np.array(frame.attrs["ci_upper"]) * 100,
                    ],
                    axis=-1,
                ),
                hovertemplate="בית %{y}, חוץ %{x}: %{z:.2f}%<br>n=%{customdata[0]}"
                "<br>95% [%{customdata[1]:.2f}, %{customdata[2]:.2f}]<extra></extra>",
            )
        fig.update_layout(
            title=f"התוצאה השכיחה: {maximum[0]}:{maximum[1]} "
            f"ב־{frame.loc[maximum[0], maximum[1]]:.1%} מהמשחקים"
        )
    elif kind == "poisson":
        fig = go.Figure()
        fig.add_bar(
            x=data.scoreline,
            y=data.observed * 100,
            name="נצפה",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(data.observed_ci_upper - data.observed) * 100,
                arrayminus=(data.observed - data.observed_ci_lower) * 100,
            ),
            marker_color=HOME_COLOR,
        )
        fig.add_bar(
            x=data.scoreline, y=data.expected * 100, name="Poisson עצמאי", marker_color=AWAY_COLOR
        )
        difference = float(frame.loc["all_draws", "ratio"]) - 1
        direction = "יותר" if difference >= 0 else "פחות"
        fig.update_layout(
            title=f"יש {abs(difference):.1%} {direction} תיקו מהצפוי ב־Poisson עצמאי",
            barmode="group",
            yaxis_title="% מכל המשחקים",
        )
    elif kind == "dispersion":
        fig = go.Figure(
            go.Bar(
                x=data.metric,
                y=data.dispersion,
                error_y=_errors(data, "dispersion"),
                marker_color=HOME_COLOR,
            )
        )
        fig.add_hline(y=1, line_dash="dash", annotation_text="Poisson: שונות = ממוצע")
        row = frame.iloc[0]
        fig.update_layout(
            title=f"פיזור בית {row.dispersion:.2f}; "
            f"מתאם בית–חוץ {row.home_away_correlation:+.3f} "
            f"[{row.home_away_correlation_ci_lower:+.3f}, "
            f"{row.home_away_correlation_ci_upper:+.3f}]",
            yaxis_title="שונות / ממוצע",
        )
    elif kind == "halftime":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["קבוצת בית", "קבוצת חוץ"])
        for i, home in enumerate((True, False), start=1):
            group = frame.xs(home, level="is_home").reindex(["leading", "level", "trailing"])
            z = group[["win_rate", "draw_rate", "loss_rate"]].to_numpy(dtype=float) * 100
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=["ניצחון", "תיקו", "הפסד"],
                    y=["יתרון במחצית", "תיקו במחצית", "פיגור במחצית"],
                    coloraxis="coloraxis",
                    text=z,
                    texttemplate="%{text:.1f}%",
                ),
                row=1,
                col=i,
            )
        home = frame.loc[(True, "leading"), "win_rate"]
        away = frame.loc[(False, "leading"), "win_rate"]
        fig.update_layout(
            title=f"יתרון במחצית מסתיים בניצחון: בית {home:.1%}, חוץ {away:.1%}",
            coloraxis=dict(colorscale="Blues", cmin=0, cmax=100),
        )
    elif kind in ("first_goal", "red_cards"):
        if kind == "first_goal":
            labels = [
                ("בית" if r.is_home else "חוץ")
                + (" הבקיעה ראשונה" if r.scored_first else " ספגה ראשונה")
                for r in data.itertuples()
            ]
            home = frame.loc[(True, True), "win_rate"]
            away = frame.loc[(False, True), "win_rate"]
            title = f"המבקיעה ראשונה מנצחת: בית {home:.1%}, חוץ {away:.1%}"
        else:
            data = frame.xs("cards", level="part").reset_index()
            labels = [f"אדום לקבוצה: {s[0]}, ליריבה: {s[2]}" for s in data.group]
            exposed = data.loc[data.group.str.startswith("1"), "n"].sum()
            title = f"אדום לקבוצה ב־{int(exposed):,} שורות קבוצה×משחק; קשר, לא סיבה"
        fig = go.Figure()
        for result in ("win", "draw", "loss"):
            fig.add_bar(
                x=labels,
                y=data[f"{result}_rate"] * 100,
                name=NAMES[result],
                marker_color=COLORS[result],
                error_y=_errors(data, f"{result}_rate", 100),
                customdata=data.n,
                hovertemplate="%{x}: %{y:.1f}%<br>n=%{customdata}",
            )
        fig.update_layout(title=title, barmode="group", yaxis_title="% משחקים בקבוצה")
    elif kind == "timing":
        fig = go.Figure()
        for home, label, color in ((True, "בית", HOME_COLOR), (False, "חוץ", AWAY_COLOR)):
            group = frame.xs(home, level="is_home").reindex(["leading", "level", "trailing"])
            fig.add_bar(
                x=["מובילה", "בשוויון", "מפגרת"],
                y=group.goals_per_90,
                name=label,
                marker_color=color,
                error_y=_errors(group, "goals_per_90"),
                customdata=np.column_stack([group.n_goals, group.minutes, group.n_matches]),
                hovertemplate="%{x}: %{y:.2f} שערים ל־90<br>שערים=%{customdata[0]}"
                "<br>דקות=%{customdata[1]}<br>משחקים=%{customdata[2]}<extra>%{fullData.name}</extra>",
            )
        home = frame.xs(True, level="is_home")
        away = frame.xs(False, level="is_home")
        fig.update_layout(
            title=f"קצב בפיגור מול שוויון: בית {home.loc['trailing', 'goals_per_90']:.2f} "
            f"מול {home.loc['level', 'goals_per_90']:.2f}; "
            f"חוץ {away.loc['trailing', 'goals_per_90']:.2f} "
            f"מול {away.loc['level', 'goals_per_90']:.2f} שערים ל־90",
            barmode="group",
            yaxis_title="שערים ל־90 דקות חשיפה",
        )
    elif kind == "time_in_state":
        fig = px.scatter(
            data,
            x="minutes_leading",
            y="ppm",
            color="season_id",
            hover_data=["team_name", "n", "minutes_level", "minutes_trailing"],
            labels=dict(minutes_leading="דקות יתרון בממוצע", ppm="נקודות למשחק"),
        )
        fig.update_layout(
            title=f"{len(data)} קבוצות־עונות: "
            f"זמן היתרון נע בין {data.minutes_leading.min():.1f} "
            f"ל־{data.minutes_leading.max():.1f} דקות; תיאור בדיעבד"
        )
    elif kind == "resilience":
        fig = px.scatter(
            data,
            x="points_dropped_per_led_game",
            y="points_won_per_trailed_game",
            text="team_name",
            size="n",
            hover_data=["n", "n_seasons", "n_led", "n_trailed"],
            labels=dict(
                points_dropped_per_led_game="נקודות שאבדו לכל משחק עם יתרון",
                points_won_per_trailed_game="נקודות שהושגו לכל משחק עם פיגור",
            ),
        )
        fig.update_traces(
            textposition="top center",
            error_x=_errors(data, "points_dropped_per_led_game"),
            error_y=_errors(data, "points_won_per_trailed_game"),
        )
        leader = data.loc[data.points_won_per_trailed_game.idxmax()]
        fig.update_layout(
            title=f"{leader.team_name}: {leader.points_won_per_trailed_game:.2f} נקודות "
            f"למשחק עם פיגור, {leader.points_dropped_per_led_game:.2f} אבדו למשחק עם יתרון",
        )
    elif kind == "late_goals":
        data = data.sort_values("late_goal_share")
        fig = go.Figure(
            go.Bar(
                x=data.team_name,
                y=data.late_goal_share * 100,
                marker_color=HOME_COLOR,
                error_y=_errors(data, "late_goal_share", 100),
                customdata=data.n_goals,
                hovertemplate="%{x}: %{y:.1f}%<br>n שערים=%{customdata}",
            )
        )
        leader = data.iloc[-1]
        fig.update_layout(
            title=f"{leader.team_name}: {leader.late_goal_share:.1%} "
            "מהשערים מדקה 76; יש להשוות את רווחי הסמך",
            yaxis_title="% שערים במחצית שנייה, דקה 76+",
        )
    elif kind == "penalties":
        fig = go.Figure()
        for kind_name, label, color in (
            ("penalty", "שערי פנדל", HOME_COLOR),
            ("own_goal", "שערים עצמיים", AWAY_COLOR),
        ):
            metric = f"{kind_name}_goal_share"
            fig.add_trace(
                go.Scatter(
                    x=data.season_id,
                    y=data[metric] * 100,
                    name=label,
                    mode="lines+markers",
                    line_color=color,
                    error_y=_errors(data, metric, 100),
                    customdata=data.n_goals,
                    hovertemplate="עונה %{x}: %{y:.1f}%<br>n שערים=%{customdata}",
                )
            )
        share = data.penalty_n.sum() / data.n_goals.sum()
        fig.update_layout(
            title=f"שערי פנדל הם {share:.1%} מהשערים הידועים; אין מידע על החמצות או שריקות",
            xaxis_title="season_id",
            yaxis_title="% מכל השערים",
        )
    else:
        raise ValueError(f"Unknown chapter D chart: {kind}")
    return _finish(fig).update_layout(
        title_x=0.97, title_xanchor="right", height=700, margin=dict(t=150, b=130, l=75, r=65)
    )
