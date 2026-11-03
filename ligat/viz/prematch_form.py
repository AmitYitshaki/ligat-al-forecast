"""Notebook 03 B2: readable form tables and saved-experiment diagnostics."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.prematch_form_review import METRICS, SELECTED_P, STRATA
from ligat.viz.league_stats import DRAW_COLOR, GOOD_SCALE
from ligat.viz.prematch_b import _finish_chart
from ligat.viz.presentation import TableDisplay, display_table
from ligat.viz.seasons import season_label

NAMES = {
    "season": "עונה",
    "metric": "מדד",
    "stratum": "סוג משחקים",
    "rho": "ρ",
    "league_mean": "ממוצע ליגה בעונה הקודמת",
    "promoted_value": "ערך פתיחה לעולה",
    "promoted_observations": "עולות קודמות",
    "promoted_fallback": "גיבוי לעולה",
    "method": "שיטה",
    "score": "ציון",
    "recent_score": "ציון ב־5 האחרונות",
    "poisson_nll": "NLL ב־15",
    "last5_poisson_nll": "NLL ב־5 האחרונות",
    "log_loss": "log loss ב־15",
    "last5_log_loss": "log loss ב־5 האחרונות",
    "z": "תוספת מעל Elo ביחידות SE",
    "seasons_better": "עונות עם תוספת מעל Elo",
    "rank_all": "דירוג ב־15",
    "rank_last5": "דירוג ב־5 האחרונות",
    "within_one_se": "קרוב למוביל ב־15",
    "improvement": "שיפור מעל Elo",
    "gamma": "γ ביתיות",
    "a": "a התקפה",
    "d": "d הגנת היריבה",
    "b": "b Elo",
    "beta": "β Elo",
    "beta_form": "β כושר",
    "attack_change_percent": "תוספת שערים בהתקפה (%)",
    "defense_change_percent": "תוספת שערים מול ההגנה (%)",
    "home_win_change_pp": "תוספת לניצחון בית (נ״א)",
    "quantity": "כמות",
    "median": "חציון",
    "minimum": "מינימום",
    "maximum": "מקסימום",
    "games": "משחקים",
    "predicted": "חזוי",
    "observed": "בפועל",
    "ratio": "יחס בפועל / חזוי",
    "gap_low": "גבול תחתון לפער λ",
    "gap_high": "גבול עליון לפער λ",
}


def form_table(
    frame: pd.DataFrame, *, columns: list[str] | None = None, percent: tuple[str, ...] = ()
) -> TableDisplay:
    """Hebrew labels, explicit percent units and isolated negative coefficients."""
    data = frame.copy()
    if "season_id" in data:
        data["season"] = data.season_id.map(lambda sid: season_label(int(sid)))
        data = data.drop(columns="season_id")
        if columns is None:
            data = data[["season", *[column for column in data if column != "season"]]]
    if "metric" in data:
        data["metric"] = data.metric.replace(METRICS)
    if "stratum" in data:
        data["stratum"] = data.stratum.replace(STRATA)
    if "quantity" in data:
        data["quantity"] = data.quantity.map(lambda value: NAMES.get(value, value))
    chosen = list(data.columns) if columns is None else columns
    precision = {column: 4 for column in chosen}
    precision.update(
        {
            name: 6
            for name in (
                "poisson_nll",
                "last5_poisson_nll",
                "log_loss",
                "last5_log_loss",
                "score",
                "recent_score",
                "improvement",
            )
        }
    )
    precision.update(
        {
            name: 0
            for name in (
                "rank_all",
                "rank_last5",
                "games",
                "seasons_better",
                "promoted_observations",
            )
        }
    )
    return display_table(
        data,
        {name: NAMES.get(name, name) for name in chosen},
        columns=chosen,
        percent=percent,
        precision=precision,
        show_index=False,
        values={"True": "כן", "False": "לא"},
        unicode_minus=tuple(chosen),
    )


def rho_chart(coefficients: pd.DataFrame) -> Any:
    """All nine rho trajectories; the early fallback is visible, not treated as a fit."""
    metrics, strata = list(METRICS), list(STRATA)
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[f"{METRICS[m]} — {STRATA[s]}" for m in metrics for s in strata],
        vertical_spacing=0.13,
    )
    for row, metric in enumerate(metrics, 1):
        for column, stratum in enumerate(strata, 1):
            data = coefficients.loc[
                (coefficients.metric == metric) & (coefficients.stratum == stratum)
            ].sort_values("season_id")
            seasons = data.season_id.map(lambda s: season_label(int(s)))
            figure.add_trace(
                go.Scatter(
                    x=seasons,
                    y=data.rho,
                    mode="lines+markers",
                    showlegend=False,
                    marker=dict(
                        color=GOOD_SCALE[(row - 1) * 4],
                        symbol=np.where(data.rho_fallback, "circle-open", "circle"),
                    ),
                    customdata=np.column_stack(
                        [
                            data.observations,
                            data.season_pairs,
                            np.where(data.rho_fallback, "גיבוי 0.5", "אומדן מעונות קודמות"),
                        ]
                    ),
                    hovertemplate=(
                        "%{x}<br>ρ: %{y:.3f}<br>קבוצות־עונות: %{customdata[0]}"
                        "<br>זוגות עונות: %{customdata[1]}"
                        "<br>%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
            positions = list(range(0, len(seasons), 4))
            if len(seasons) - 1 - positions[-1] < 3:
                positions.pop()
            positions.append(len(seasons) - 1)
            ticks = seasons.iloc[positions].tolist()
            figure.update_xaxes(
                categoryorder="array",
                categoryarray=seasons.tolist(),
                tickvals=ticks,
                tickangle=-35,
                row=row,
                col=column,
            )
    figure.update_yaxes(range=[0, 1], title_text="ρ")
    figure.update_layout(title="ערכי הפתיחה: כמה מהפער מהליגה נשמר — לפי עונת התחזית")
    return _finish_chart(figure, height=1150)


def haifa_chart(frame: pd.DataFrame, metric: str) -> Any:
    """Venue estimates only at that venue's matches; actual goals are comparison dots."""
    if metric not in ("gf", "ga"):
        raise ValueError("Haifa chart metric must be gf or ga")
    figure = go.Figure()
    for home, label, color in ((True, "בית", GOOD_SCALE[7]), (False, "חוץ", GOOD_SCALE[1])):
        part = frame.loc[frame.is_home.eq(home)]
        figure.add_scatter(
            x=part.as_of,
            y=part[f"{metric}_venue_ewma5"],
            mode="lines+markers",
            name=f"EWMA 5 — {label}",
            line_color=color,
            hovertemplate=(
                "%{x|%d.%m.%Y}<br>פיצ׳ר לפני המשחק: %{y:.3f}<extra>%{fullData.name}</extra>"
            ),
        )
    figure.add_scatter(
        x=frame.as_of,
        y=frame[f"{metric}_all_ewma20"],
        mode="lines",
        name="EWMA 20 — כל המגרשים",
        line=dict(color=DRAW_COLOR, dash="dash"),
        hovertemplate="%{x|%d.%m.%Y}<br>פיצ׳ר לפני המשחק: %{y:.3f}<extra>%{fullData.name}</extra>",
    )
    actual = "goals_scored" if metric == "gf" else "goals_against"
    for home, label, color in ((True, "בית", GOOD_SCALE[7]), (False, "חוץ", GOOD_SCALE[1])):
        part = frame.loc[frame.is_home.eq(home)]
        figure.add_scatter(
            x=part.as_of,
            y=part[actual],
            mode="markers",
            name=f"בפועל — {label}",
            marker=dict(color=color, symbol="x", size=10),
            hovertemplate="%{x|%d.%m.%Y}<br>שערים בפועל: %{y}<extra>%{fullData.name}</extra>",
        )
    title = "התקפה" if metric == "gf" else "הגנה"
    figure.update_layout(
        title=f"מכבי חיפה {season_label(25)}: {title} לפני המשחק מול השערים בפועל",
        xaxis_title="תאריך משחק",
        yaxis_title="שערים למשחק",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    figure.update_xaxes(tickformat="%m/%Y")
    return _finish_chart(figure, height=650)


def addition_chart(candidates: pd.DataFrame) -> Any:
    """Paired-fold improvement/SE, not a p-value or a new candidate selection."""
    data = candidates.sort_values("z")
    figure = px.bar(
        data,
        x="z",
        y="method",
        color="family",
        orientation="h",
        color_discrete_map={"מידע מתוך העונה": GOOD_SCALE[7], "עונות קודמות בלבד": GOOD_SCALE[1]},
        custom_data=["improvement", "improvement_se", "seasons_better"],
        labels={"z": "שיפור מעל Elo / שגיאת תקן מזווגת", "method": "מועמד", "family": "מקור המידע"},
        title="EXP-002 P: מידע מתוך העונה מוסיף מעל Elo; prev/hist אינם מוסיפים",
    )
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>שיפור/SE: %{x:.3f}<br>שיפור NLL: %{customdata[0]:.6f}"
            "<br>SE: %{customdata[1]:.6f}"
            "<br>עונות עם שיפור: %{customdata[2]}<extra></extra>"
        )
    )
    figure.add_vline(x=0, line_color=DRAW_COLOR)
    figure.add_vline(x=1, line_dash="dash", line_color=DRAW_COLOR, annotation_text="שגיאת תקן אחת")
    figure.update_yaxes(categoryorder="array", categoryarray=data.method.tolist())
    return _finish_chart(figure, height=950)


def recent_scatter(candidates: pd.DataFrame, references: pd.DataFrame) -> Any:
    """Make the ה-ת6 disagreement and pre-approved exception visible."""
    figure = go.Figure()
    for variant, symbol, color, label in (
        ("all", "circle", GOOD_SCALE[2], "כל המגרשים"),
        ("venue", "diamond", GOOD_SCALE[7], "בית וחוץ בנפרד"),
    ):
        data = candidates.loc[candidates.variant.eq(variant)]
        figure.add_scatter(
            x=data.poisson_nll,
            y=data.last5_poisson_nll,
            mode="markers",
            name=label,
            marker=dict(symbol=symbol, color=color, size=10),
            customdata=data[["method", "rank_all", "rank_last5"]],
            hovertemplate=(
                "%{customdata[0]}<br>NLL ב־15: %{x:.6f}<br>NLL ב־5: %{y:.6f}"
                "<br>דירוג ב־15: %{customdata[1]}"
                "<br>דירוג ב־5: %{customdata[2]}<extra></extra>"
            ),
        )
    for identifier, name, position, symbol in (
        ("P_ewma5_all_elo", "ewma5/all — מוביל והצעת פשטות", "bottom right", "circle-open"),
        (SELECTED_P, "ewma5/venue — נבחר בידי עמית", "top right", "star"),
    ):
        data = candidates.loc[candidates.config_id.eq(identifier)]
        figure.add_scatter(
            x=data.poisson_nll,
            y=data.last5_poisson_nll,
            mode="markers+text",
            name=name,
            text=[name],
            textposition=position,
            marker=dict(size=18, symbol=symbol, color=GOOD_SCALE[4]),
            hovertemplate="NLL ב־15: %{x:.6f}<br>NLL ב־5: %{y:.6f}<extra>%{fullData.name}</extra>",
        )
    reference = references.set_index("method").loc["P_reference_elo"]
    figure.add_vline(
        x=reference.score, line_color=DRAW_COLOR, line_dash="dash", annotation_text="Elo בלבד — 15"
    )
    figure.add_hline(
        y=reference.recent_score,
        line_color=DRAW_COLOR,
        line_dash="dash",
        annotation_text="Elo בלבד — 5",
    )
    figure.update_layout(
        title="דגל ה־ת6: המוביל ב־15 שונה מהמוביל ב־5 האחרונות — NLL נמוך עדיף",
        xaxis_title="NLL על 15 עונות",
        yaxis_title="NLL על 5 האחרונות",
    )
    return _finish_chart(figure, height=700)
