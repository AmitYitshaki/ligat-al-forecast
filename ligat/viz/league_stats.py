"""Plotly presentation of notebook 02 chapter A; no data loading or feature publication."""

from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.viz.notebook02 import COLUMN_NAMES, VALUE_NAMES
from ligat.viz.presentation import rtl_figure
from ligat.viz.seasons import season_label, with_season_labels

HOME_COLOR = "#2563eb"
DRAW_COLOR = "#6b7280"
AWAY_COLOR = "#f59e0b"
# Trim the almost-white yellow endpoint so light values remain visible on white.
GOOD_SCALE = px.colors.sample_colorscale("Viridis", np.linspace(0, 0.90, 12))
POSITION_SCALE = GOOD_SCALE[::-1]


def _finish(figure: Any) -> Any:
    for key in figure.layout.to_plotly_json():
        if key.startswith("coloraxis"):
            coloraxis = figure.layout[key]
            position = coloraxis.colorbar.title.text in ("final_position", "מיקום סופי")
            coloraxis.colorscale = POSITION_SCALE if position else GOOD_SCALE
            if position:
                coloraxis.cmin, coloraxis.cmax = 1, 14
    for number, trace in enumerate(figure.data):
        if trace.type == "heatmap" and trace.coloraxis is None:
            trace.colorscale = GOOD_SCALE
        name = str(trace.name or "")
        outcome = any(word in name for word in ("ניצחון", "ניצחונות", "תיקו", "הפסד"))
        if not outcome:
            color = GOOD_SCALE[3 + (number * 4) % 9]
            marker = getattr(trace, "marker", None)
            if (
                marker is not None
                and isinstance(marker.color, str)
                and marker.color not in (DRAW_COLOR, *GOOD_SCALE)
            ):
                marker.color = color
            elif marker is not None and isinstance(marker.color, (list, tuple, np.ndarray)):
                colors = list(marker.color)
                # A three-outcome vector retains the established football colors.
                outcomes = set((HOME_COLOR, DRAW_COLOR, AWAY_COLOR))
                if colors and all(isinstance(c, str) for c in colors) and set(colors) != outcomes:
                    marker.color = [GOOD_SCALE[3 + (i * 4) % 9] for i in range(len(colors))]
            line = getattr(trace, "line", None)
            if (
                line is not None
                and isinstance(line.color, str)
                and line.color not in (DRAW_COLOR, *GOOD_SCALE)
            ):
                line.color = color
    # Fix categorical order explicitly on each subplot, without changing coordinates.
    # Ranked clubs retain the analytical order; ordinal bins and seasons sort naturally.
    ordinal = ["סדיר", "עליון", "אמצעי", "תחתון", "0–2", "3–5", "6–8", "9–11", "12–14", "15+"]
    for key in figure.layout:
        if not (key.startswith("xaxis") or key.startswith("yaxis")):
            continue
        axis = figure.layout[key]
        coordinate = key[0]
        ref = key.replace("axis", "")
        categories = []
        season_values: set[int] = set()
        for trace in figure.data:
            if (getattr(trace, coordinate + "axis", None) or coordinate) != ref:
                continue
            values = getattr(trace, coordinate, None)
            for value in () if values is None else values:
                if axis.title.text == "season_id" and isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    season_values.add(int(value))
                if isinstance(value, str) and value not in categories:
                    categories.append(value)
        if axis.title.text == "season_id" and season_values:
            ticks = sorted(season_values)
            ticks = sorted(set(ticks[:: max(1, len(ticks) // 8)] + ticks[-1:]))
            axis.title.text = "עונה"
            axis.tickmode, axis.tickvals = "array", ticks
            axis.ticktext = [season_label(s) for s in ticks]
            axis.tickangle = -30
        if not categories or axis.categoryorder:
            continue
        if all(value in ordinal for value in categories):
            categories.sort(key=ordinal.index)
        elif all(value.startswith("Q") and value[1:].isdigit() for value in categories):
            categories.sort(key=lambda value: int(value[1:]))
        elif all(value[:4].isdigit() and "/" in value for value in categories):
            categories.sort()
        axis.categoryorder, axis.categoryarray = "array", categories
        if axis.ticktext is None:
            axis.tickmode, axis.tickvals = "array", categories
            axis.ticktext = [VALUE_NAMES.get(value, value) for value in categories]
    for key in figure.layout:
        if key.startswith("coloraxis") and figure.layout[key].colorbar.title.text == "season_id":
            figure.layout[key].colorbar.title.text = "עונה"
            seasons = sorted(
                {
                    int(v)
                    for trace in figure.data
                    for v in (
                        ()
                        if getattr(getattr(trace, "marker", None), "color", None) is None
                        else trace.marker.color
                    )
                    if isinstance(v, (int, float, np.integer, np.floating))
                }
            )
            if seasons:
                ticks = sorted(set(seasons[:: max(1, len(seasons) // 4)] + seasons[-1:]))
                figure.layout[key].colorbar.tickvals = ticks
                figure.layout[key].colorbar.ticktext = [season_label(s) for s in ticks]
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
    return rtl_figure(figure, labels={**COLUMN_NAMES, **VALUE_NAMES})


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
    data = (with_season_labels(frame) if "season" in frame else frame).reset_index()
    hover = [
        c
        for c in ("team_name", "season", "season_id", "matches_played", "final_position")
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
            hover_data=[
                *hover,
                "n_seasons_at_position",
                *[c for c in ("season", "deduction") if c in data],
            ],
            title=title,
            labels={
                "final_position": "מיקום סופי רשמי",
                "points_per_match": "נקודות למשחק, ללא קיזוז",
            },
        )
        fig.update_traces(marker_color=GOOD_SCALE[-1], line_color=GOOD_SCALE[3])
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(range(1, 15)))
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
            title=f"הבקעה למשחק {top:.2f} בשלוש הראשונות מול {bottom:.2f} בשלוש האחרונות",
            labels={
                "goals_scored_per_match": "שערי זכות למשחק",
                "goals_conceded_per_match": "שערי חובה למשחק",
                "final_position": "מיקום סופי",
            },
        )
        fig.add_vline(
            x=data.goals_scored_per_match.mean(),
            line_dash="dash",
            line_color=DRAW_COLOR,
            annotation_text="ממוצע הבקעה",
        )
        fig.add_hline(
            y=data.goals_conceded_per_match.mean(),
            line_dash="dash",
            line_color=DRAW_COLOR,
            annotation_text="ממוצע ספיגה",
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
            f"בשלוש הראשונות: ניצחונות גדולים {top.big_win_rate.mean():.1%}; "
            f"ניצחונות צרים {top.narrow_win_rate.mean():.1%} מכל המשחקים"
        )
        fig = px.scatter(
            data,
            x="big_win_rate",
            y="narrow_win_rate",
            color="final_position",
            hover_data=[
                *hover,
                "win_rate",
                "big_win_rate",
                "narrow_win_rate",
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
        # Display-only displacement, much smaller than a one-win step. Hover keeps exact rates.
        rng = np.random.default_rng(20261028)
        fig.data[0].x = np.asarray(fig.data[0].x) + rng.uniform(-0.004, 0.004, len(data))
        fig.data[0].y = np.asarray(fig.data[0].y) + rng.uniform(-0.004, 0.004, len(data))
        # Plotly normally hovers x/y directly; preserve exact rates after jitter.
        data["hover_season"] = [season_label(int(s)) for s in data.season_id]
        fig.data[0].customdata = data[
            [
                "team_name",
                "hover_season",
                "final_position",
                "big_win_rate",
                "narrow_win_rate",
                "win_rate",
                "big_win_rate_n",
                "big_win_rate_ci_lower",
                "big_win_rate_ci_upper",
                "narrow_win_rate_ci_lower",
                "narrow_win_rate_ci_upper",
            ]
        ].to_numpy()
        fig.data[0].hovertemplate = (
            "%{customdata[0]} · %{customdata[1]}<br>מיקום רשמי %{customdata[2]}"
            "<br>ניצחונות גדולים %{customdata[3]:.1%}; 95% "
            "[%{customdata[7]:.1%}, %{customdata[8]:.1%}]"
            "<br>ניצחונות צרים %{customdata[4]:.1%}; 95% "
            "[%{customdata[9]:.1%}, %{customdata[10]:.1%}]"
            "<br>כל הניצחונות %{customdata[5]:.1%}; משחקים %{customdata[6]}<extra></extra>"
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
    elif kind == "open_style":
        data = data.sort_values("btts_rate", kind="stable")
        fig = go.Figure()
        for rate, name, color in (
            ("btts_rate", "שתי הקבוצות הבקיעו", GOOD_SCALE[3]),
            ("over_2_5_rate", "לפחות 3 שערים", GOOD_SCALE[-1]),
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
            title=(f"BTTS ממוצע {data.btts_rate.mean():.1%}; רווחי 95% לממוצעי העונות"),
            yaxis_title="ממוצע שיעור עונתי, עם טווח אי-ודאות",
            xaxis_title="קבוצות לפי שיעור BTTS; ללא דירוג סיבתי",
        )
        if summary is not None:
            for metric, color in (("btts_rate", GOOD_SCALE[3]), ("over_2_5_rate", GOOD_SCALE[-1])):
                fig.add_hline(
                    y=float(summary.loc[metric, "league_mean"]), line_dash="dash", line_color=color
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
            color="ppm_t",
            hover_data=["team_name", "season" if "season" in data else "season_id"],
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
        fig = px.scatter(
            data,
            x="prev_season_value_share",
            y="points_per_match",
            color="over_under",
            hover_data=[*hover, "expected_ppm", "over_under"],
            color_continuous_scale=GOOD_SCALE,
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
        rule_label = "אחוזון 30" if frame.attrs.get("rule") == "q30" else "החציון הישן"
        complete = data.dropna(subset=["diff"])
        below = complete["diff"].lt(0).mean()
        fig = px.scatter(
            data,
            x="imputed_share",
            y="value_share_of_league",
            color="value_share_of_league",
            hover_data=[
                "team_name",
                "season" if "season" in data else "season_id",
                "diff",
                "ratio",
            ],
            title=(
                f"{below:.1%} מתוך {len(complete)} העולות מתחת לכלל {rule_label} "
                "(כל העידנים, בדיעבד)"
            ),
            labels={
                "imputed_share": f"{rule_label} בחצי התחתון של t−1",
                "value_share_of_league": "נתח שווי סגל ממשי באותה עונה",
            },
        )
        _diagonal(fig, data, "imputed_share", "value_share_of_league")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.update_coloraxes(colorbar_tickformat=".0%")
    elif kind == "consistency":
        data = data.dropna(subset=["std_ppm"])
        data["classification"] = data.is_yoyo.map({True: "יו-יו: לפחות שתי עליות", False: "אחרות"})
        fig = px.scatter(
            data,
            x="mean_ppm",
            y="std_ppm",
            color="mean_ppm",
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
