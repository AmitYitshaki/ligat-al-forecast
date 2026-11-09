"""Hebrew RTL tables and B-style figures for notebook 03 C6a; no data loading."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.analysis.prematch_context_review import METRICS, WORDS
from ligat.viz.league_stats import DRAW_COLOR, GOOD_SCALE
from ligat.viz.prematch_b import _finish_chart
from ligat.viz.presentation import TableDisplay, display_table
from ligat.viz.seasons import season_label

NAMES = {
    "column": "עמודה",
    "newcomer_policy": "הטיפול בעולה",
    "date": "תאריך ההשוואה",
    "clubs": "מועדונים משותפים",
    "spearman": "ספירמן",
    "source": "מקור",
    "value_rank": "מיקום בשווי הממוצע",
    "first_season": "עונת הקיפול הראשון",
    "first_training": "עונות אימון בקיפול הראשון",
    "first_improvement": "תוספת בקיפול הראשון",
    "comparison": "השוואה",
    "mean_spearman": "ספירמן ממוצע",
    "min_spearman": "מינימום בעונה",
    "max_spearman": "מקסימום בעונה",
    "seasons": "עונות",
    "mean": "יתרון הממוצע בספירמן",
    "std": "סטיית תקן בין עונות",
    "se": "SE",
    "better_seasons": "עונות לטובת הממוצע",
    "season": "עונה",
    "team_name": "קבוצה",
    "total_million": "שווי כולל (מיליון €)",
    "average_million": "שווי ממוצע (מיליון €)",
    "metric": "מדד",
    "teams": "קבוצות־עונות",
    "method": "שיטה",
    "ppm": "נקודות למשחק",
    "route": "מסלול",
    "candidate": "מועמד",
    "formula": "נוסחה: בית מול חוץ",
    "training_start": "אימון מעונה",
    "score": "ציון ב־15",
    "recent_score": "ציון ב־5 האחרונות",
    "rank_all": "דירוג ב־15",
    "rank_last5": "דירוג ב־5",
    "improvement": "תוספת מעל הייחוס",
    "better": "קיפולים טובים",
    "folds": "קיפולים",
    "recent_improvement": "תוספת ב־5 האחרונות",
    "recent_better": "טובים ב־5",
    "adds": "מוסיף לפי הכלל המקורי",
    "condition": "תנאי החלופה מתקיים",
    "disagreement": "דגל ה־ת6",
    "approved": "הכרעת עמית",
    "training_seasons": "עונות אימון",
    "elo_coefficient": "מקדם Elo",
    "value_coefficient": "מקדם השווי",
    "q25": "רבעון 25 של הפער",
    "q75": "רבעון 75 של הפער",
    "home_win_low": "ניצחון בית ברבעון 25 (%)",
    "home_win_high": "ניצחון בית ברבעון 75 (%)",
    "change_pp": "שינוי בנקודות אחוז",
    "quantity": "בדיקה",
    "value": "ערך מחושב",
    "expected": "ערך בקרה",
    "difference": "פער מהבקרה",
    "passed": "עבר",
    **METRICS,
}


def context_table(frame: pd.DataFrame, *, columns: list[str] | None = None) -> TableDisplay:
    """Use the shared bidi-aware table renderer, retaining explicit explanatory words."""
    data = frame.copy()
    if WORDS not in data:
        raise ValueError("Every C methods/metrics table requires the words column")
    if "season_id" in data:
        data["season"] = data.season_id.map(lambda s: season_label(int(s)))
        data = data.drop(columns="season_id")
        if columns is None:
            data = data[["season", *[c for c in data if c != "season"]]]
    if "first_season" in data:
        data["first_season"] = data.first_season.map(lambda s: season_label(int(s)))
    if "metric" in data:
        data["metric"] = data.metric.replace(METRICS)
    chosen = list(data.columns) if columns is None else columns
    precision = {c: 4 for c in chosen}
    precision.update(
        {
            c: 6
            for c in (
                "score",
                "recent_score",
                "improvement",
                "se",
                "recent_improvement",
                "value",
                "expected",
                "difference",
                "first_improvement",
            )
        }
    )
    precision.update(
        {
            c: 0
            for c in (
                "clubs",
                "seasons",
                "better_seasons",
                "teams",
                "better",
                "recent_better",
                "folds",
                "training_start",
                "training_seasons",
                "rank_all",
                "rank_last5",
                "value_rank",
                "first_training",
            )
        }
    )
    return display_table(
        data,
        {c: NAMES.get(c, c) for c in chosen},
        columns=chosen,
        precision=precision,
        show_index=False,
        unicode_minus=tuple(chosen),
    )


def context_matrix_table(matrix: pd.DataFrame) -> TableDisplay:
    """A signed matrix also has a row explanation, including the rank's opposite direction."""
    frame = matrix.rename_axis("metric").reset_index()
    frame[WORDS] = frame.metric.map(METRICS) + " מול כל מדד אחר; מיקום נמוך יותר הוא טוב יותר"
    return context_table(frame)


def depth_display(frame: pd.DataFrame) -> TableDisplay:
    """Compact mean (SE), in thousandths, with the actual counts of retained folds."""
    rows = []
    for row in frame.to_dict("records"):
        record = {"מסלול": row["route"], "שיטה": row["method"], WORDS: row[WORDS]}
        for minimum in (2, 4, 6, 8):
            mean, se = row[f"mean_{minimum}"] * 1000, row[f"se_{minimum}"] * 1000
            marker = " ✓" if minimum > 2 and mean > se else ""
            record[f"לפחות {minimum} עונות ({row[f'folds_{minimum}']})"] = (
                f"{mean:+.2f} ({se:.2f}){marker}".replace("-", "−")
            )
        record["5 האחרונות — תיאורי"] = (
            f"{row['recent_improvement'] * 1000:+.2f}, {row['recent_better']}/5".replace("-", "−")
        )
        rows.append(record)
    data = pd.DataFrame(rows)
    return display_table(data, {c: c for c in data}, show_index=False)


def distribution_chart(frame: pd.DataFrame) -> Any:
    """Five distinct scales on equal-weight team-season rows; no repeated-game weighting."""
    known = frame.loc[frame.prev_season_known]
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=list(METRICS.values()),
        horizontal_spacing=0.09,
        vertical_spacing=0.18,
    )
    for i, metric in enumerate(METRICS):
        for promoted, label, color in (
            (False, "ותיקות", GOOD_SCALE[7]),
            (True, "עולות", GOOD_SCALE[3]),
        ):
            part = known.loc[known.promoted_in.eq(promoted)]
            figure.add_trace(
                go.Box(
                    y=part[metric],
                    x=[label] * len(part),
                    name=label,
                    legendgroup=label,
                    showlegend=i == 0,
                    marker_color=color,
                    boxpoints="outliers",
                    customdata=np.column_stack(
                        [part.season_id.map(lambda s: season_label(int(s))), part.team_name]
                    ),
                    hovertemplate="%{x}<br>%{y:.3f}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
                ),
                row=i // 3 + 1,
                col=i % 3 + 1,
            )
    figure.update_layout(title="מדדי t−1: עולות מול ותיקות — שורה אחת לכל קבוצה־עונה")
    figure.update_xaxes(categoryorder="array", categoryarray=["ותיקות", "עולות"])
    return _finish_chart(figure, height=900)


def context_heatmap(matrix: pd.DataFrame) -> Any:
    """Show signed Spearman, so the opposite meaning of numerical rank remains visible."""
    labels = [METRICS[c] for c in matrix]
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            texttemplate="%{z:.2f}",
            colorscale=GOOD_SCALE,
            hovertemplate="%{x}<br>%{y}<br>ספירמן: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(title="מדדי עבר: שלושה מתארים תוצאות, והשווי מתאר את השוק")
    figure.update_yaxes(autorange="reversed")
    return _finish_chart(figure, height=620)


def value_comparison_chart(paired: pd.DataFrame) -> Any:
    """Average-minus-total next-season rank association for the same continuing clubs."""
    data = paired.sort_values("season_id")
    labels = data.season_id.map(lambda s: season_label(int(s)))
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=data.improvement,
            marker_color=GOOD_SCALE[7],
            customdata=data.teams,
            hovertemplate=(
                "%{x}<br>תוספת לממוצע: %{y:+.3f}<br>מועדונים: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_color=DRAW_COLOR, line_width=2)
    figure.update_layout(
        title="שווי ממוצע מול סכום: התוספת בספירמן לעונה הבאה",
        yaxis_title="ממוצע פחות סכום",
        xaxis_title="עונת השווי",
    )
    figure.update_xaxes(categoryorder="array", categoryarray=labels.tolist(), tickangle=-35)
    return _finish_chart(figure, height=580)


def alone_chart(frame: pd.DataFrame, references: pd.DataFrame) -> Any:
    """Both saved windows, each with its own B0 and Elo lines; lower loss is better."""
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=["15 קיפולים", "5 אחרונים"], horizontal_spacing=0.2
    )
    refs = references.set_index("method")
    for column, score, color in ((1, "score", GOOD_SCALE[7]), (2, "recent_score", GOOD_SCALE[3])):
        figure.add_trace(
            go.Bar(
                x=frame[score],
                y=frame.method,
                orientation="h",
                name="המדד לבדו",
                marker_color=color,
                showlegend=False,
                hovertemplate="%{y}<br>log loss: %{x:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        for reference, line_color in (("B0", DRAW_COLOR), ("Elo", GOOD_SCALE[0])):
            figure.add_vline(
                x=float(refs.at[reference, score]),
                line_color=line_color,
                line_dash="dash",
                annotation_text=reference,
                row=1,
                col=column,
            )
        figure.update_xaxes(
            range=[1.0, 1.095], title_text="log loss — קטן יותר טוב יותר", row=1, col=column
        )
        figure.update_yaxes(
            categoryorder="array",
            categoryarray=frame.method.tolist(),
            autorange="reversed",
            row=1,
            col=column,
        )
    figure.update_layout(title="EXP-003, מסלול A: מה מדד העונה הקודמת אומר לבדו")
    return _finish_chart(figure, height=650)


def save_context_figures(figures: dict[str, Any], directory: Path) -> dict[str, tuple[str, str]]:
    """Export only presentation artifacts, with the exact B2 image dimensions and scale."""
    if not directory.resolve().is_relative_to(Path("reports/figures").resolve()):
        raise ValueError("C figures must stay inside reports/figures")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, figure in figures.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid C figure name")
        html, png = directory / f"{name}.html", directory / f"{name}.png"
        figure.write_html(html, include_plotlyjs=True)
        figure.write_image(png, width=1800, height=figure.layout.height or 650, scale=1.5)
        paths[name] = (str(html), str(png))
    return paths


def added_chart(frame: pd.DataFrame, route: str) -> Any:
    """Display mean ± one paired SE in thousandths; these are not 95% intervals."""
    if route not in ("O", "P"):
        raise ValueError("Added-information chart requires route O or P")
    data = frame.loc[frame.route.eq(route)].copy()
    figure = go.Figure(
        go.Scatter(
            x=data.improvement * 1000,
            y=data.method,
            mode="markers+text",
            text=[f"{v:+.2f}".replace("-", "−") for v in data.improvement * 1000],
            textposition="top center",
            marker=dict(color=GOOD_SCALE[7], size=11),
            error_x=dict(type="data", array=data.se * 1000, visible=True, color=DRAW_COLOR),
            customdata=np.column_stack([data.better, data.recent_improvement * 1000]),
            hovertemplate=(
                "%{y}<br>תוספת ×1000: %{x:+.3f}"
                "<br>טובים מתוך 15: %{customdata[0]}"
                "<br>5 אחרונים ×1000: %{customdata[1]:+.3f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color="black", line_width=2)
    metric = "log loss" if route == "O" else "Poisson NLL"
    figure.update_layout(
        title=f"מסלול {route}: התוספת מעל הייחוס — ממוצע ±SE",
        xaxis_title=f"ייחוס פחות מועמד ×1000 ({metric})",
    )
    figure.update_yaxes(
        categoryorder="array", categoryarray=data.method.tolist(), autorange="reversed"
    )
    return _finish_chart(figure, height=650)


def value_fold_chart(frame: pd.DataFrame) -> Any:
    """Highlight 2009/10 while retaining the separate O/P route colors."""
    figure = go.Figure()
    for route, color in (("O", GOOD_SCALE[7]), ("P", GOOD_SCALE[3])):
        data = frame.loc[frame.route.eq(route)].sort_values("season_id")
        labels = data.season_id.map(lambda s: season_label(int(s)))
        for highlighted in (False, True):
            mask = data.season_id.eq(11) if highlighted else data.season_id.ne(11)
            figure.add_trace(
                go.Bar(
                    x=labels.loc[mask],
                    y=data.loc[mask, "improvement"] * 1000,
                    name=route,
                    legendgroup=route,
                    showlegend=not highlighted,
                    offsetgroup=route,
                    marker_color=DRAW_COLOR if highlighted else color,
                    customdata=data.loc[mask, "training_seasons"],
                    hovertemplate=(
                        "%{x}<br>תוספת ×1000: %{y:+.3f}<br>עונות אימון: %{customdata}<extra>"
                    )
                    + route
                    + "</extra>",
                )
            )
    figure.add_hline(y=0, line_color="black", line_width=2)
    figure.update_layout(
        barmode="group",
        title="שווי ממוצע: הקיפול הדל של 2009/2010 בולט בהפסד",
        yaxis_title="תוספת מעל הייחוס ×1000",
        xaxis_title="עונת הולידציה",
    )
    seasons = sorted(frame.season_id.unique())
    figure.update_xaxes(
        categoryorder="array", categoryarray=[season_label(int(s)) for s in seasons], tickangle=-35
    )
    return _finish_chart(figure, height=650)
