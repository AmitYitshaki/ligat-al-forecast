"""Plotly presentation of notebook 02 chapter A; no data loading or feature publication."""

from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz.presentation import rtl_figure

HOME_COLOR = "#2563eb"
DRAW_COLOR = "#6b7280"
AWAY_COLOR = "#f59e0b"


def _finish(figure: Any) -> Any:
    figure.update_layout(title_text="<br>".join(wrap(figure.layout.title.text, width=65)))
    figure.update_layout(
        template="plotly_white",
        height=550,
        title_x=0.97,
        title_xanchor="right",
        margin=dict(t=135, b=110),
        font={"family": "Arial", "size": 14},
        legend_title_text="",
    )
    if "coloraxis" in figure.layout.to_plotly_json():
        figure.update_layout(legend=dict(orientation="h", y=-0.18, x=1, xanchor="right"))
    return rtl_figure(figure)


def _diagonal(figure: Any, frame: pd.DataFrame, x: str, y: str) -> None:
    values = frame[[x, y]].to_numpy(dtype=float, na_value=np.nan)
    if values.size and np.isfinite(values).any():
        bounds = [float(np.nanmin(values)), float(np.nanmax(values))]
        figure.add_trace(
            go.Scatter(
                x=bounds,
                y=bounds,
                mode="lines",
                name="שוויון",
                line={"color": DRAW_COLOR, "dash": "dash"},
            )
        )


def chapter_a_chart(kind: str, frame: pd.DataFrame, summary: pd.DataFrame | None = None) -> Any:
    """Present one approved football question with counts and descriptive caveats."""
    data = frame.reset_index()
    hover = [
        c
        for c in ("team_name", "season_id", "matches_played", "final_position")
        if c in data.columns
    ]
    if kind == "position":
        data = data.loc[data.era.eq("14_teams")].copy()
        counts = data.groupby("final_position").size()
        data["n_seasons_at_position"] = data.final_position.map(counts)
        champion_ppm = data.loc[data.final_position.eq(1), "points_per_match"].mean()
        title = f"בעידן 14 הקבוצות האלופה צברה בממוצע {champion_ppm:.2f} נקודות למשחק"
        fig = px.box(
            data,
            x="final_position",
            y="points_per_match",
            points="all",
            hover_data=[*hover, "n_seasons_at_position"],
            title=title,
            labels={
                "final_position": "מיקום סופי רשמי",
                "points_per_match": "נקודות למשחק, ללא קיזוז",
            },
        )
        fig.update_xaxes(type="category")
    elif kind == "attack_defense":
        top = data.loc[data.final_position.le(3), "goals_scored_per_match"].mean()
        bottom = data.loc[data.final_position.ge(data.n_teams - 2), "goals_scored_per_match"].mean()
        fig = px.scatter(
            data,
            x="goals_scored_per_match",
            y="goals_conceded_per_match",
            color="final_position",
            hover_data=hover,
            color_continuous_scale="Viridis_r",
            title=(
                f"שלוש הראשונות כבשו {top:.2f} למשחק, מול {bottom:.2f} בשלוש האחרונות; ממוצע עונות"
            ),
            labels={
                "goals_scored_per_match": "שערי זכות למשחק",
                "goals_conceded_per_match": "שערי חובה למשחק",
                "final_position": "מיקום סופי",
            },
        )
    elif kind == "home_away":
        mean_gap = data.home_away_ppm_gap.mean()
        fig = px.scatter(
            data,
            x="home_points_per_match",
            y="away_points_per_match",
            color="final_position",
            hover_data=hover,
            title=f"בבית נצברו בממוצע {mean_gap:.2f} נקודות למשחק יותר מבחוץ, בממוצע עונות הקבוצות",
            labels={
                "home_points_per_match": "נקודות למשחק בבית",
                "away_points_per_match": "נקודות למשחק בחוץ",
                "final_position": "מיקום סופי",
            },
        )
        _diagonal(fig, data, "home_points_per_match", "away_points_per_match")
    elif kind == "win_style":
        top = data.loc[data.final_position.le(3)]
        title = (
            f"בשלוש הראשונות: {top.big_win_rate.mean():.1%} ניצחונות גדולים ו-"
            f"{top.narrow_win_rate.mean():.1%} צרים מתוך כל המשחקים, בממוצע עונות"
        )
        fig = px.scatter(
            data,
            x="big_win_rate",
            y="narrow_win_rate",
            color="final_position",
            hover_data=[
                *hover,
                "win_rate",
                "big_win_rate_n",
                "big_win_rate_ci_lower",
                "big_win_rate_ci_upper",
                "narrow_win_rate_ci_lower",
                "narrow_win_rate_ci_upper",
            ],
            title=title,
            labels={
                "big_win_rate": "ניצחון בהפרש לפחות 3, מכל המשחקים",
                "narrow_win_rate": "ניצחון בהפרש 1, מכל המשחקים",
                "final_position": "מיקום סופי",
            },
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
    elif kind == "open_style":
        fig = go.Figure()
        for rate, name, color in (
            ("btts_rate", "שתי הקבוצות הבקיעו", HOME_COLOR),
            ("over_2_5_rate", "לפחות 3 שערים", AWAY_COLOR),
        ):
            fig.add_trace(
                go.Scatter(
                    x=data.team_name,
                    y=data[rate],
                    mode="markers",
                    name=name,
                    marker={"color": color},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": (data[f"{rate}_ci_upper"] - data[rate]).clip(lower=0),
                        "arrayminus": (data[rate] - data[f"{rate}_ci_lower"]).clip(lower=0),
                    },
                    customdata=data[["n_seasons", "n_matches"]],
                    hovertemplate="%{x}<br>שיעור=%{y:.1%}<br>עונות=%{customdata[0]}<br>משחקים=%{customdata[1]}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(
            title=(
                f"ב-{len(data)} קבוצות עם לפחות 5 עונות: ממוצע BTTS הוא "
                f"{data.btts_rate.mean():.1%}; רווחי 95% לממוצעי העונות"
            ),
            yaxis_title="ממוצע שיעור עונתי, עם טווח אי-ודאות",
            xaxis_title="קבוצה, ללא דירוג סיבתי",
        )
        fig.update_yaxes(tickformat=".0%")
    elif kind == "persistence":
        if summary is None:
            raise ValueError("Persistence chart needs the descriptive regression summary")
        slope = summary.loc["ppm", "slope"]
        fig = px.scatter(
            data,
            x="ppm_prev",
            y="ppm_t",
            hover_data=["team_name", "season_id"],
            title=f"מקדם ההתמדה בין עונות הוא {slope:.2f}; קשר תיאורי ולא תחזית",
            labels={"ppm_prev": "נקודות למשחק בעונה הקודמת", "ppm_t": "נקודות למשחק בעונה הנוכחית"},
        )
        _diagonal(fig, data, "ppm_prev", "ppm_t")
        _regression_line(
            fig, data, "ppm_prev", float(slope), float(summary.loc["ppm", "intercept"])
        )
    elif kind == "value":
        correlation = (
            summary.loc["prev_season_value_share", "spearman"] if summary is not None else np.nan
        )
        complete = data.dropna(subset=["over_under"])
        high = complete.loc[complete.over_under.gt(0)].nlargest(10, "over_under")
        low = complete.loc[complete.over_under.lt(0)].nsmallest(10, "over_under")
        data["label"] = ""
        for index in [*high.index, *low.index]:
            data.loc[index, "label"] = (
                f"{data.loc[index, 'team_name']} {int(data.loc[index, 'season_id']) + 1998}"
            )
        fig = px.scatter(
            data,
            x="prev_season_value_share",
            y="points_per_match",
            text="label",
            color="over_under",
            hover_data=[*hover, "expected_ppm", "over_under"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title=(f"שווי קודם מול נקודות: ספירמן {correlation:.2f}; החריגות בתוך המדגם")
            if summary is not None
            else "שווי קודם ונקודות: שאריות תיאוריות",
            labels={
                "prev_season_value_share": "נתח השווי בליגה של t−1",
                "points_per_match": "נקודות למשחק",
                "over_under": "שארית",
            },
        )
        fig.update_xaxes(tickformat=".0%")
        if summary is not None:
            relation = summary.loc["prev_season_value_share"]
            _regression_line(
                fig,
                data,
                "prev_season_value_share",
                float(relation.slope),
                float(relation.intercept),
            )
    elif kind == "promoted":
        complete = data.dropna(subset=["diff"])
        below = complete["diff"].lt(0).mean()
        fig = px.scatter(
            data,
            x="imputed_share",
            y="value_share_of_league",
            hover_data=["team_name", "season_id", "diff", "ratio"],
            title=(
                f"ב-{below:.1%} מתוך {len(complete)} עולות עם שווי זמין, "
                "השווי נמוך מערך הכלל; בדיעבד"
            ),
            labels={
                "imputed_share": "ערך הכלל לפי שווי בעונה הקודמת",
                "value_share_of_league": "נתח שווי סגל ממשי באותה עונה",
            },
        )
        _diagonal(fig, data, "imputed_share", "value_share_of_league")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
    elif kind == "consistency":
        data = data.dropna(subset=["std_ppm"])
        data["classification"] = data.is_yoyo.map({True: "יו-יו: לפחות שתי עליות", False: "אחרות"})
        fig = px.scatter(
            data,
            x="mean_ppm",
            y="std_ppm",
            color="classification",
            size="seasons",
            hover_name="team_name",
            hover_data=[
                "seasons",
                "titles",
                "relegations",
                "promotions",
                "mean_ppm_ci_lower",
                "mean_ppm_ci_upper",
            ],
            title=(
                f"מתוך {len(data)} קבוצות עם לפחות 3 עונות, "
                f"{int(data.is_yoyo.sum())} עלו לפחות פעמיים בפיתוח"
            ),
            labels={
                "mean_ppm": "ממוצע נקודות למשחק בין עונות",
                "std_ppm": "סטיית תקן בין עונות, נקודות למשחק",
            },
        )
    else:
        raise ValueError(f"Unknown chapter A chart: {kind}")
    return _finish(fig)


def _regression_line(fig: Any, data: pd.DataFrame, x: str, slope: float, intercept: float) -> None:
    if not data[x].notna().any() or not np.isfinite([slope, intercept]).all():
        return
    bounds = np.array([data[x].min(), data[x].max()], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=bounds,
            y=intercept + slope * bounds,
            mode="lines",
            name="רגרסיה תיאורית",
            line={"color": HOME_COLOR},
        )
    )
