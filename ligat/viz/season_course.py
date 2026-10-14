"""Seven chapter C charts from approved descriptive tables; no data loading."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz.league_history import ERA_NAMES, _errors
from ligat.viz.league_stats import AWAY_COLOR, DRAW_COLOR, HOME_COLOR, _finish

STAGE_NAMES = {"regular": "סדיר", "upper": "עליון", "lower": "תחתון", "middle": "אמצעי"}
ERA_COLORS = {"12_teams": DRAW_COLOR, "16_teams": AWAY_COLOR, "14_teams": HOME_COLOR}


def _checkpoint(frame: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    k = (
        10
        if 10 in frame.index.get_level_values("k")
        else int(frame.index.get_level_values("k").max())
    )
    return k, frame.xs(k, level="k")


def chapter_c_chart(kind: str, frame: pd.DataFrame, summary: pd.DataFrame | None = None) -> Any:
    """Present a numeric finding with stage/era denominators and descriptive uncertainty."""
    if kind == "settling":
        k, checkpoint = _checkpoint(frame)
        fig = go.Figure()
        for era, group in frame.groupby(level="era", observed=True):
            data = group.droplevel("era")
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.spearman_mean,
                    mode="lines",
                    name=ERA_NAMES[era],
                    line_color=ERA_COLORS[era],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=data.spearman_ci_upper - data.spearman_mean,
                        arrayminus=data.spearman_mean - data.spearman_ci_lower,
                    ),
                    customdata=data.n_seasons,
                    hovertemplate=(
                        "k=%{x}: %{y:.2f}<br>עונות=%{customdata}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
        fig.update_layout(
            title=(
                f"אחרי {k} משחקים המתאם לסיום כבר "
                f"{checkpoint.spearman_mean.min():.2f}–{checkpoint.spearman_mean.max():.2f}"
            ),
            xaxis_title="משחקים לכל קבוצה בשלב הסדיר",
            yaxis_title="מתאם ספירמן",
            yaxis_range=[-1, 1],
        )
    elif kind == "leaders":
        eras = frame.index.get_level_values("era").unique()
        fig = make_subplots(rows=1, cols=len(eras), subplot_titles=[ERA_NAMES[e] for e in eras])
        for col, era in enumerate(eras, 1):
            data = frame.xs(era, level="era")
            for metric, name, color in (
                ("leader_won_title_rate", "מובילה זכתה", HOME_COLOR),
                ("bottom_relegated_rate", "אחרונה ירדה", AWAY_COLOR),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[metric] * 100,
                        name=name,
                        mode="lines",
                        line_color=color,
                        error_y=_errors(data, metric, 100),
                        customdata=data[f"{metric}_n"],
                        showlegend=col == 1,
                        hovertemplate=(
                            "k=%{x}: %{y:.1f}%<br>מקרים ללא שוויון=%{customdata}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
        k, check = _checkpoint(frame)
        fig.update_layout(
            title=(
                f"אחרי {k} משחקים המובילה זכתה ב־{int(check.leader_won_title_rate_count.sum())}"
                f"/{int(check.leader_won_title_rate_n.sum())}; האחרונה ירדה ב־"
                f"{int(check.bottom_relegated_rate_count.sum())}/{int(check.bottom_relegated_rate_n.sum())}"
            )
        )
        fig.update_yaxes(range=[0, 100], title_text="אחוז")
    elif kind == "movement":
        if summary is None or summary.empty:
            raise ValueError("Movement summary is required")
        data = frame.reset_index()
        fig = go.Figure()
        for stage, color in (("upper", HOME_COLOR), ("lower", AWAY_COLOR)):
            group = data.loc[data.group.eq(stage)]
            fig.add_trace(
                go.Box(
                    y=group.move,
                    name=STAGE_NAMES[stage],
                    boxpoints="all",
                    marker_color=color,
                    text=group.team_name,
                    customdata=group.season_id,
                    hovertemplate=(
                        "%{text}, עונה %{customdata}: תזוזה %{y}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
        row = summary.iloc[0]
        fig.update_layout(
            title=(
                f"המובילה בפיצול זכתה ב־{int(row.leader_won_count)} מתוך {int(row.leader_n)} עונות"
            ),
            yaxis_title="דירוג בפיצול פחות מיקום סופי; חיובי = שיפור",
        )
    elif kind == "stages":
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["שערים למשחק", "תיקו (%)", "0:0 (%)", "שתי הקבוצות מבקיעות (%)"],
        )
        data = frame.reindex([s for s in STAGE_NAMES if s in frame.index])
        for i, metric in enumerate(("goals_per_match", "draw_rate", "zero_zero_rate", "btts_rate")):
            scale = 1 if i == 0 else 100
            fig.add_trace(
                go.Bar(
                    x=[STAGE_NAMES[s] for s in data.index],
                    y=data[metric] * scale,
                    marker_color=HOME_COLOR,
                    error_y=_errors(data, metric, scale),
                    customdata=data.n_matches,
                    hovertemplate="%{x}: %{y:.2f}<br>n=%{customdata}<extra></extra>",
                ),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )
        fig.update_layout(
            title=(
                f"בעליון {frame.loc['upper', 'goals_per_match']:.2f} שערים מול "
                f"{frame.loc['regular', 'goals_per_match']:.2f} בסדיר; היריבות שונות"
            ),
            showlegend=False,
        )
    elif kind == "form":
        data = frame.reset_index()
        fig = px.scatter(
            data,
            x="ppm_regular",
            y="ppm_playoff",
            color="final_stage",
            color_discrete_map={"upper": HOME_COLOR, "lower": AWAY_COLOR, "middle": DRAW_COLOR},
            hover_data=["team_name", "season_id", "n_regular", "n_playoff", "ppm_change"],
            labels={
                "ppm_regular": "נקודות למשחק — סדיר",
                "ppm_playoff": "נקודות למשחק — פלייאוף",
                "final_stage": "שלב",
            },
        )
        limits = [
            float(data[["ppm_regular", "ppm_playoff"]].min().min()),
            float(data[["ppm_regular", "ppm_playoff"]].max().max()),
        ]
        fig.add_trace(
            go.Scatter(
                x=limits,
                y=limits,
                mode="lines",
                name="קצב זהה",
                line=dict(color=DRAW_COLOR, dash="dash"),
            )
        )
        fig.update_layout(
            title=(
                f"{int(data.ppm_change.gt(0).sum())} מתוך {len(data)} קבוצות־עונות "
                "שיפרו את קצב הנקודות בפלייאוף"
            )
        )
    elif kind == "last_rounds":
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["שערים למשחק", "תיקו (%)", "ניצחון בית (%)", "הפתעות בקרב הזכאיות (%)"],
        )
        data = frame.reset_index()
        for i, metric in enumerate(("goals_per_match", "draw_rate", "home_win_rate", "upset_rate")):
            for period, name, color in (
                ("last", "3 המחזורים האחרונים", AWAY_COLOR),
                ("rest", "שאר המחזורים", HOME_COLOR),
            ):
                group = data.loc[data.period.eq(period)]
                scale = 1 if i == 0 else 100
                fig.add_trace(
                    go.Bar(
                        x=[STAGE_NAMES[s] for s in group.stage],
                        y=group[metric] * scale,
                        name=name,
                        marker_color=color,
                        error_y=_errors(group, metric, scale),
                        showlegend=i == 0,
                        customdata=group.upset_n if metric == "upset_rate" else group.n,
                        hovertemplate=(
                            "%{x}: %{y:.2f}<br>n=%{customdata}<extra>%{fullData.name}</extra>"
                        ),
                    ),
                    row=i // 2 + 1,
                    col=i % 2 + 1,
                )
        last = frame.loc[("upper", "last")]
        title = f"בסוף העליון {last.goals_per_match:.2f} שערים; אין קבוצת שאר בתצוגה זו"
        if ("upper", "rest") in frame.index:
            rest = frame.loc[("upper", "rest")]
            title = (
                f"בסוף העליון {last.goals_per_match:.2f} שערים מול "
                f"{rest.goals_per_match:.2f} בשאר; רמז למחקר מוטיבציה"
            )
        fig.update_layout(
            title=title,
            barmode="group",
        )
    elif kind == "race":
        data = frame.reset_index()
        if data.season_id.nunique() != 1:
            raise ValueError("Choose exactly one season for the points race")
        data = data.loc[~data.games_short]
        fig = px.line(
            data,
            x="k",
            y="points",
            color="team_name",
            markers=False,
            hover_data=["games_played"],
            labels={
                "k": "משחקים לכל קבוצה",
                "points": "נקודות מצטברות — ללא עונשים",
                "team_name": "קבוצה",
            },
        )
        split = int(data.regular_games.max())
        fig.add_vline(x=split, line_dash="dash", line_color=DRAW_COLOR, annotation_text="פיצול")
        final = data.sort_values("k").groupby("team_id").tail(1).points.sort_values(ascending=False)
        season = int(data.season_id.iloc[0])
        fig.update_layout(
            title=(
                f"{season + 1998}/{str(season + 1999)[-2:]}: "
                f"המובילות צברו {int(final.iloc[0])} ו־{int(final.iloc[1])} נקודות על המגרש"
            )
        )
    else:
        raise ValueError(f"Unknown chapter C chart: {kind}")
    return _finish(fig).update_layout(
        title_x=0.97, title_xanchor="right", height=650, margin=dict(t=130, b=85, l=65, r=65)
    )
