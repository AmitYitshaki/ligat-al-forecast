"""Notebook-04 presentation: Tailwind palette, explicit Hebrew labels and named clubs."""

from collections.abc import Mapping, Sequence
from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.basedatatypes import BasePlotlyType  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from ligat.viz.motivation_style import normalize_minus as normalize_minus
from ligat.viz.motivation_style import two_se, verify_rtl_labels
from ligat.viz.presentation import TableDisplay, display_table, rtl_figure
from ligat.viz.seasons import season_label

BLUE = "#2563EB"
GRAY = "#6B7280"
BLUE_SCALE = [[0.0, "#DBEAFE"], [0.5, "#2563EB"], [1.0, "#1E3A8A"]]
STAGE_COLORS = {"regular": "#4F46E5", "upper": "#F59E0B", "middle": "#9333EA", "lower": "#0891B2"}
VALUES = {
    "none": "ללא",
    "4 seasons": "המשקל נחצה כל 4 עונות",
    "diff": "בית פחות חוץ",
    "sides": "שני הצדדים",
    "regular": "סדירה",
    "upper": "עליון",
    "middle": "אמצעי",
    "lower": "תחתון",
    "all": "כל השלבים",
    "title": "אליפות",
    "cut": "פלייאוף עליון",
    "survival": "הישארות",
    "survival_direct": "ללא ירידה ישירה",
    "zero": "ציון אפס",
    "places_only": "רק קווי המקומות פתוחים",
    "direct_only": "רק הישארות ישירה",
    "practical_zero": "אפס מעשי במסלולים",
    "other_mc": "רעש Monte Carlo",
    "contrast": "הפרש בתוך שכבות",
    "slope": "שיפוע בתוך שכבות",
    "dead": "קבוצה ללא יעד",
    "dead_vs_alive_opponent": "ללא יעד מול יריבה חיה",
    "alive_vs_dead_opponent": "חיה מול יריבה ללא יעד",
    "home_minus_away": "פער בית פחות חוץ",
    "transparent": "פיצ'רים שקופים",
    "title_close": "מאבק אליפות צמוד",
    "cut_close": "מאבק עליון צמוד",
    "releg_close": "מאבק הישארות צמוד",
    "incumbent": "ותיקה בדוגמה",
    "promoted": "עולה בדוגמה",
    "res_pts": "נקודות",
    "res_gf": "הבקעה",
    "res_ga": "ספיגה",
    "gap_to_safety": "פער מוחלט מקו ההישארות",
    "gap_to_leader": "פער מהפסגה",
    "gap_to_cut": "פער מוחלט מהעליון",
    "games_remaining_stage": "משחקים שנותרו בשלב",
}
LABELS = {
    "games_remaining_stage": "משחקים שנותרו בשלב",
    "p_win": "סיכוי בניצחון",
    "p_draw": "סיכוי בתיקו",
    "p_loss": "סיכוי בהפסד",
    "lev_raw": "חשיבות גולמית",
    "lev": "חשיבות",
    "mc_se": "SE Monte Carlo",
    "m_home": "רמת הבית",
    "m_away": "רמת החוץ",
    "m_diff": "בית פחות חוץ",
    "team_name": "קבוצה",
    "opponent_name": "יריבה",
    "home_name": "בית",
    "away_name": "חוץ",
    "season": "עונה",
    "as_of": "בוקר המשחק",
    "date": "תאריך המשחק",
    "round": "מחזור",
    "stage": "שלב",
    "n": "n — שורות",
    "n_games": "משחקים",
    "n_seasons": "עונות",
    "n_used": "שורות עם השוואה",
    "n_excluded": "שורות שהוחרגו",
    "estimate": "אומדן",
    "se": "SE עונתי",
    "metric": "שארית",
    "representation": "דרך מדידה",
    "category": "קטגוריה / רמה",
    "method": "סוג האומדן",
    "last5_estimate": "אומדן ב-5 האחרונות",
    "last5_se": "SE ב-5 האחרונות",
    "last5_n": "שורות ב-5 האחרונות",
    "last5_n_used": "שורות עם השוואה ב-5 האחרונות",
    "sign_same": "עונות באותו סימן",
    "sign_eligible": "עונות עם לפחות 30 שורות",
    "level": "ייצוג",
    "value": "רמה",
    "fraction_stage": "% מתוך השלב",
    "fraction_all": "% מכל השורות",
    "target": "יעד",
    "fraction": "שיעור",
    "remaining": "משחקים שנותרו בשלב",
    "mean": "ממוצע חשיבות",
    "median": "חציון",
    "zero_fraction": "שיעור אפסים",
    "above_02": "שיעור מעל 0.2",
    "q25": "אחוזון 25",
    "q50": "חציון",
    "q75": "אחוזון 75",
    "q90": "אחוזון 90",
    "q99": "אחוזון 99",
    "feature": "פיצ'ר שקוף",
    "rho": "Spearman",
    "close": "מאבק",
    "is_close": "צמוד לפי הטבלה",
    "zero_group": "סיבת ציון אפס",
    "N": "מסלולים",
    "is_home": "ביתית",
    "leverage_max": "חשיבות מרבית",
    "nothing_to_play": "אין על מה לשחק",
    "title_decided": "האליפות הוכרעה",
    "cut_decided": "העליון הוכרע",
    "relegation_decided": "ההישארות הוכרעה",
    "places_decided": "קווי המקומות הוכרעו",
    "points": "נקודות בבוקר",
    "position": "מקום בבוקר",
    "games_played": "משחקים ששוחקו",
    "elo_pre": "Elo בבוקר",
    "role": "תפקיד בדוגמה",
    "previous": "מקום קודם בדוגמה",
    "lottery": "מפתח הגרלה",
    "goal_difference": "הפרש שערים",
    "wins": "ניצחונות",
    "goals_for": "שערי זכות",
    "goals_against": "שערי חובה",
    "n_ratio": "שורות עם יחס מוגדר",
    "paired_zero": "SE מזווג אפס",
    "infinite_ratio": "יחס אינסופי",
    "undefined_ratio": "יחס 0/0 — לא מוגדר",
    "file": "קובץ קלט",
    "sha256": "SHA-256",
    "manifest": "מקור האימות",
    "verified": "אומת",
    "check": "מה נבדק",
    "explanation": "למה זה חשוב בכדורגל",
    "passed": "עבר",
    "seconds": "שניות חישוב",
    "wall_seconds": "שניות בריצה הנוכחית",
    "cache_hit": "נטען מהמטמון",
    "computed_at": "מועד החישוב (UTC)",
    "max_error": "הפרש מרבי",
    "dates": "תאריכים",
    "target_checks": "בדיקות יעד שהוכרע",
    "random_tables": "השוואות טבלאות אקראיות",
    "replayed_formats": "פורמטים ששוחזרו",
    "first_day_error": "הפרש הסתברויות ביום הראשון",
    "max_elo_error": "הפרש Elo מרבי",
    "split_lottery_paths": "טבלאות שהגיעו להגרלה בפיצול",
    "split_rank_paths": "טבלאות שנבדקו בפיצול",
    "final_lottery_paths": "טבלאות שהגיעו להגרלה בסיום",
    "final_rank_paths": "טבלאות שנבדקו בסיום",
    "split_rate": "שיעור הגרלות בפיצול",
    "final_rate": "שיעור הגרלות בסיום",
    "fold": "קיפול",
    "train_first": "אימון מעונה",
    "train_last": "אימון עד עונה",
    "train_seasons": "עונות אימון",
    "deep": "קיפול עמוק",
    "candidate": "תצורה",
    "form": "צורת הייצוג",
    "decay": "דעיכה באימון",
    "words": "מה זה במילים",
    "formula": "נוסחה",
    "information": "מידע שמשמש לחישוב",
    "example": "דוגמה כדורגלנית",
    "validation": "איפה נבדק",
    "term": "מושג",
    "definition": "הסבר",
    "status": "מעמד",
    "finding": "ממצא",
    "data_scientist": "מבט מדעי הנתונים",
    "football": "מבט כדורגלני",
    "gap_to_safety": "פער מקו ההישארות",
    "gap_to_leader": "פער מהפסגה",
    "gap_to_cut": "פער מהעליון",
    "xpts": "נקודות צפויות",
    "pts": "נקודות בפועל",
    "res_pts": "שארית נקודות",
    "res_gf": "שארית הבקעה",
    "res_ga": "שארית ספיגה",
}
for _target, _name in (
    ("title", "אליפות"),
    ("cut", "עליון"),
    ("survival", "הישארות"),
    ("survival_direct", "ללא ירידה ישירה"),
):
    VALUES[f"Comp / {_target}"] = f"רכיבים / {_name}"
    for _suffix, _label in (
        ("p_win", "סיכוי בניצחון"),
        ("p_draw", "סיכוי בתיקו"),
        ("p_loss", "סיכוי בהפסד"),
        ("lev_raw", "חשיבות גולמית"),
        ("lev", "חשיבות"),
        ("mc_se", "SE Monte Carlo"),
    ):
        LABELS[f"{_target}_{_suffix}"] = f"{_name}: {_label}"

VALUES.update(
    {
        "yes": "כן",
        "no": "לא",
        "unclear": "לא ברור",
        "near_zero_alive": "L5=0.25 (זהה ל-L3=0.5)",
        "near_zero_vs_dead": "כמעט כלום מול מתות בלבד",
        "all_games": "כל המשחקים",
        "live_games": "שתי הקבוצות חיות",
        "joint": "משותף עם הדגל",
        "B1": "2009/2010–2013/2014",
        "B2": "2014/2015–2018/2019",
        "B3": "2019/2020–2023/2024",
        "alive_only_importance": "סלי חשיבות — חיות בלבד",
        "alive_importance_difference": "פער קבוצה–יריבה — שורות חיות",
        "alive_up_to_001": "חיה עם חשיבות עד 0.01",
        "up_to_001": "עד 0.01",
        "001_to_005": "0.01–0.05",
        "005_to_02": "0.05–0.2",
        "above_02": "מעל 0.2",
        "importance_diff": "חשיבות קבוצה פחות יריבה",
    }
)
LABELS.update(
    {
        "comparisons": "השוואות מתוכננות",
        "estimated": "אומדנים מחושבים",
        "estimable": "השוואות כשירות",
        "beyond_two_se": "מעבר לפעמיים SE",
        "comparison_id": "השוואה",
        "predictor": "משתנה",
        "centered_corr": "מתאם לאחר מרכוז",
        "median_diff": "פער טיפוסי — חציון",
        "p90_diff": "פער טיפוסי — P90",
        "median_effect": "נקודות — פער חציוני",
        "p90_effect": "נקודות — P90",
        "central_effect": "אומדן מרכזי בנקודות",
        "central_se": "SE בנקודות",
        "beyond_flag": "מעבר לדגל?",
        "blocks_same": "בלוקים באותו כיוון",
        "blocks_eligible": "בלוקים כשירים",
        "blocks_small": "בלוקים קטנים מ-30",
        "block": "בלוק",
        "what_captured": "מה הדרך תופסת",
        "next_test": "מה נשאר ל-EXP-004",
        "football_reading": "איך קוראים בכדורגל",
        "remaining_fixtures": "משחקים שנותרו בלוח הידוע",
        "finished": "סיימה את משחקיה",
        "other_games": "משחקים נוספים מעבר לתצוגה",
        "mc_lower": "חשיבות פחות 2·SE",
        "mc_upper": "חשיבות ועוד 2·SE",
        "family": "משפחת הפיילוט",
        "current": "שחזור נוכחי",
        "reference": "פיילוט",
        "sampled_paths": "טבלאות סיום במדגם",
        "old_tied_paths": "טבלאות עם הגרלה — החוק הישן",
        "new_tied_paths": "טבלאות עם הגרלה — החוק המאוחר",
        "resolved_paths": "טבלאות שהשוויון בהן נפתר",
        "old_tied_pairs": "זוגות שווים — החוק הישן",
        "new_tied_pairs": "זוגות שווים — החוק המאוחר",
        "resolved_pairs": "זוגות שהוכרעו",
        "resolved_fraction": "שיעור הזוגות שהוכרעו",
        "checked_rows": "שורות קבוצה שנבדקו",
        "checked_games": "משחקים שנבדקו",
        "checked_seasons": "עונות שנבדקו",
        "n_nonzero": "משחקים עם פער שאינו אפס",
        "last5_n_games": "משחקים ב-5 האחרונות",
        "last5_n_seasons": "עונות בתקופה האחרונה",
    }
)


def format_date(value: Any, *, with_time: bool = False) -> str:
    """Unpadded day/month and optional local Israel time, including DST conversion."""
    if pd.isna(value):
        return "NA"
    stamp = pd.Timestamp(value)
    stamp = (
        stamp.tz_localize("Asia/Jerusalem")
        if stamp.tzinfo is None
        else stamp.tz_convert("Asia/Jerusalem")
    )
    result = f"{stamp.day}.{stamp.month}.{stamp.year}"
    return result + (f" {stamp.hour:02}:{stamp.minute:02}" if with_time else "")


def timing_line(
    seconds: float,
    *,
    computed_at: Any = None,
    cache_hit: bool = False,
    wall_seconds: float | None = None,
) -> TableDisplay:
    """A short local-time status sentence, rather than an artificial sample-size table."""
    when = pd.Timestamp.now(tz="UTC") if computed_at is None else computed_at
    prefix = "נטען מהמטמון; " if cache_hit else ""
    current = (
        f"; קריאה כעת: {wall_seconds:.3f} שניות" if cache_hit and wall_seconds is not None else ""
    )
    return TableDisplay(
        f'<p dir="rtl">{prefix}חושב בתאריך {format_date(when, with_time=True)}, '
        f"{seconds:.3f} שניות{current}.</p>"
    )


def table(
    frame: pd.DataFrame,
    *,
    names: Mapping[int, str],
    columns: Sequence[str],
    percent: Sequence[str] = (),
    labels: Mapping[str, str] | None = None,
) -> TableDisplay:
    """Format explicit columns; convert club keys to config names and preserve missing values."""
    data = (
        frame.reset_index()
        if any(n in ("game_id", "team_id") for n in frame.index.names)
        else frame.copy()
    )
    for source, destination in (
        ("team_id", "team_name"),
        ("opponent_id", "opponent_name"),
        ("home_team_id", "home_name"),
        ("away_team_id", "away_name"),
    ):
        if source in data:
            data[destination] = data[source].map(names)
            if data[destination].isna().any():
                raise ValueError(f"Unmapped club in {source}")
    if "season_id" in data:
        data["season"] = [season_label(int(s)) for s in data.season_id]
    for column in ("train_first", "train_last"):
        if column in data:
            data[column] = [season_label(int(s)) for s in data[column]]
    for column in ("as_of", "date", "computed_at"):
        if column in data:
            data[column] = data[column].map(
                lambda value, column=column: format_date(value, with_time=column != "as_of")
            )
    for column in ("q25", "median", "q75"):
        if column in data and np.isinf(pd.to_numeric(data[column], errors="coerce")).any():
            data[column] = data[column].map(lambda v: "∞" if pd.notna(v) and np.isinf(v) else v)
    for column in ("split_rate", "final_rate"):
        if column in data:
            data[column] = data[column].map(
                lambda value: "אין פיצול" if pd.isna(value) else f"{value:.5%}"
            )
    if "stratum" in data:
        data["stratum"] = data.stratum.map(
            lambda value: (
                str(value)
                .replace("regular", "סדירה")
                .replace("upper", "עליון")
                .replace("middle", "אמצעי")
                .replace("lower", "תחתון")
                .replace("<=", "≤")
                .replace("2-3", "2–3")
                .replace("4-6", "4–6")
            )
        )
    for column in columns:
        if column in data and (
            column.startswith("n_")
            or column in {"n", "N", "checked_rows", "checked_games", "checked_seasons"}
        ):
            numeric = pd.to_numeric(data[column], errors="coerce")
            if numeric.notna().all() and (numeric % 1 == 0).all():
                data[column] = numeric.astype(int)
        if column in data and (column.endswith("_error") or column == "error"):
            data[column] = data[column].map(
                lambda value: (
                    "0 (בדיוק המחשב)"
                    if isinstance(value, (float, int)) and abs(value) < 1e-12
                    else value
                )
            )
    if "season_id" in data:
        data = data.sort_values("season_id", kind="stable")
    rendered = display_table(
        data,
        dict(LABELS)
        | {
            "stratum": "שכבה",
            "releg_close": "מאבק הישארות צמוד",
            "median_effect_se": "SE בפער החציוני",
            "p90_effect_se": "SE בפער P90",
        }
        | dict(labels or {}),
        columns=list(columns),
        percent=percent,
        values=VALUES,
        show_index=False,
        precision={
            c: 0
            if c
            in {
                "elo_pre",
                "compared_rows",
                "excluded_rows",
                "comparisons",
                "estimated",
                "estimable",
                "beyond_two_se",
            }
            or (c.startswith("n_") and not c.endswith("pct"))
            else 4
            for c in columns
            if c
            not in (
                "n",
                "n_games",
                "n_seasons",
                "n_used",
                "n_excluded",
                "N",
                "n_ratio",
                "infinite_ratio",
                "undefined_ratio",
                "paired_zero",
                "round",
                "position",
                "points",
                "games_played",
                "games_remaining_stage",
                "remaining",
                "dates",
                "target_checks",
                "random_tables",
                "replayed_formats",
                "fold",
                "train_seasons",
                "last5_n",
                "last5_n_used",
                "sign_same",
                "sign_eligible",
                "previous",
                "split_lottery_paths",
                "split_rank_paths",
                "final_lottery_paths",
                "final_rank_paths",
                "n_nonzero",
                "blocks_same",
                "blocks_eligible",
                "blocks_small",
                "remaining_fixtures",
                "other_games",
                "sampled_paths",
                "old_tied_paths",
                "new_tied_paths",
                "resolved_paths",
                "old_tied_pairs",
                "new_tied_pairs",
                "resolved_pairs",
                "checked_rows",
                "checked_games",
                "checked_seasons",
                "last5_n_games",
                "last5_n_seasons",
            )
        },
        unicode_minus=tuple(columns),
    )
    return rendered


def _finish(fig: Any, title: str, *, height: int = 550) -> Any:
    title = "<br>".join(
        "<br>".join(wrap(part, 65, break_long_words=False)) for part in title.split("<br>")
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=14),
        title=dict(text=title, x=0.97, xanchor="right"),
        height=height,
        margin=dict(t=135, b=170, l=105, r=45),
        legend=dict(title_text="", orientation="h", y=-125 / max(height - 305, 100)),
    )
    has_errors = False
    for trace in fig.data:
        for axis in ("error_x", "error_y"):
            error = getattr(trace, axis, None)
            if error is not None and error.array is not None:
                has_errors = True
                if not isinstance(trace.meta, dict) or axis[-1] not in trace.meta.get("two_se", []):
                    raise ValueError(f"{axis}: estimate bars must be built with two_se()")
    if has_errors:
        fig.update_layout(legend_title_text="פס = פעמיים SE", showlegend=True)
        if not any(trace.showlegend is not False for trace in fig.data):
            fig.data[0].showlegend = True
            fig.data[0].name = "אומדנים"

    def clean(value: Any, *, format_numbers: bool = True) -> Any:
        if isinstance(value, str):
            # Keep the shared RTL embeddings, season formatting and hover policy.
            # Only boundary marks expressly forbidden in this notebook are omitted.
            text = normalize_minus(value) if format_numbers and "\u202b" not in value else value
            return text.replace("\u200e", "").replace("\u200f", "")
        if isinstance(value, BasePlotlyType):
            for key in value.to_plotly_json():
                item = value[key]
                if item is not None:
                    value[key] = clean(item, format_numbers=format_numbers)
            return value
        if isinstance(value, dict):
            return {key: clean(item, format_numbers=format_numbers) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, format_numbers=format_numbers) for item in value]
        if isinstance(value, np.ndarray) and value.dtype.kind in "UOS":
            return [clean(item, format_numbers=format_numbers) for item in value.tolist()]
        return value

    clean(fig.layout)
    for trace in fig.data:
        clean(trace)
    rtl_figure(fig)
    clean(fig.layout, format_numbers=False)
    for trace in fig.data:
        clean(trace, format_numbers=False)
    verify_rtl_labels(fig.to_plotly_json())
    return fig


def importance_histogram(frame: pd.DataFrame, summary: pd.DataFrame) -> Any:
    """Full frozen distribution; its population counts are explicit rather than inferential."""
    s = summary.iloc[0]
    fig = go.Figure(
        go.Histogram(
            x=frame.leverage_max,
            nbinsx=40,
            marker_color=BLUE,
            hovertemplate="חשיבות: %{x:.3f}<br>שורות: %{y}<extra></extra>",
        )
    )
    fig.update_xaxes(title="חשיבות תחרותית (0–1)")
    fig.update_yaxes(title="שורות קבוצה–משחק", rangemode="tozero")
    return _finish(
        fig,
        f"{s.zero_fraction:.1%} מהשורות: ציון אפס<br>"
        f"n={int(s.n):,}; משחקים={int(s.n_games):,}; עונות={int(s.n_seasons)}",
    )


def importance_by_remaining(frame: pd.DataFrame) -> Any:
    """Display precomputed stage means, using the fixed stage colors."""
    fig = go.Figure()
    for stage, color in STAGE_COLORS.items():
        part = frame.loc[frame.stage.eq(stage)].sort_values("remaining")
        fig.add_trace(
            go.Scatter(
                x=part.remaining,
                y=part["mean"],
                mode="lines+markers",
                name=VALUES[stage],
                line=dict(color=color),
                customdata=part[["n", "n_games", "n_seasons"]],
                hovertemplate=(
                    "נותרו %{x} משחקים<br>חשיבות %{y:.3f}<br>n=%{customdata[0]}"
                    "<br>משחקים=%{customdata[1]}<br>עונות=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title="משחקים שנותרו בשלב (כולל הנוכחי)", autorange="reversed")
    fig.update_yaxes(title="חשיבות תחרותית ממוצעת", rangemode="tozero")
    return _finish(fig, "אותו מרחק מהסוף יכול לשאת חשיבות שונה בכל שלב")


def gap_scatter(frame: pd.DataFrame, *, names: Mapping[int, str]) -> Any:
    """Display provided alive-team pairs; no outcomes or model fitting enter the graph."""
    part = frame.loc[~frame.nothing_to_play].dropna(subset=["gap_to_safety"])
    custom = np.column_stack(
        [part.index.get_level_values("team_id").map(names), part.games_remaining_stage]
    )
    fig = go.Figure(
        go.Scattergl(
            x=part.gap_to_safety.abs(),
            y=part.leverage_max,
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.35,
                color=part.games_remaining_stage,
                colorscale=BLUE_SCALE,
                colorbar=dict(
                    title="משחקים שנותרו",
                    orientation="h",
                    y=-0.5,
                    yanchor="top",
                    x=0.5,
                    xanchor="center",
                    len=0.65,
                    thickness=12,
                ),
            ),
            customdata=custom,
            hovertemplate=(
                "%{customdata[0]}<br>פער מוחלט: %{x} נקודות<br>חשיבות: %{y:.3f}"
                "<br>נותרו: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="פער מוחלט מקו ההישארות (נקודות)")
    fig.update_yaxes(title="חשיבות תחרותית (0–1)")
    return _finish(
        fig,
        "פער דומה בטבלה אינו קובע לבדו את חשיבות המשחק<br>"
        f"n={len(part):,}; משחקים={part.index.get_level_values('game_id').nunique():,}; "
        f"עונות={part.season_id.nunique()}",
    )


def residual_forest(
    frame: pd.DataFrame, title: str = "אותם ייצוגים נבדקים באותן שכבות — אומדן ±2SE"
) -> Any:
    """Show all category contrasts and their season-bootstrap SE, without significance claims."""
    data = frame.loc[frame.method.eq("contrast")].copy()
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, subplot_titles=["נקודות", "הבקעה", "ספיגה"]
    )
    labels = [
        VALUES[str(r.category)]
        if r.category == "near_zero_alive"
        else f"{VALUES.get(r.representation, r.representation)}: "
        f"{VALUES.get(str(r.category), str(r.category))}"
        for r in data.drop_duplicates(["representation", "category"]).itertuples()
    ]
    for col, metric in enumerate(("res_pts", "res_gf", "res_ga"), 1):
        part = data.loc[data.metric.eq(metric)]
        ticks = [
            VALUES[str(r.category)]
            if r.category == "near_zero_alive"
            else f"{VALUES.get(r.representation, r.representation)}: "
            f"{VALUES.get(str(r.category), str(r.category))}"
            for r in part.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=part.estimate,
                y=ticks,
                mode="markers",
                marker=dict(color=BLUE, size=7),
                **two_se(part.se, axis="x"),
                showlegend=False,
                customdata=part[["n", "n_used", "n_excluded", "n_games", "n_seasons"]],
                hovertemplate="%{y}<br>אומדן=%{x:+.3f}<br>n=%{customdata[0]}<br>בהשוואה=%{customdata[1]}<br>הוחרגו=%{customdata[2]}<br>משחקים=%{customdata[3]}<br>עונות=%{customdata[4]}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=0, line_dash="dash", line_color=GRAY, row=1, col=col)
        fig.update_xaxes(title_text="הפרש בתוך שכבות", row=1, col=col)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    return _finish(fig, title, height=550 if len(labels) < 4 else 850)


def title_timeline(frame: pd.DataFrame, team_name: str) -> Any:
    """Contrast simulated title importance with the mathematical clinch and dead-team dates."""
    frame = frame.sort_values("as_of")
    fig = go.Figure(
        go.Scatter(
            x=frame.as_of,
            y=frame.title_lev,
            mode="lines+markers",
            name="חשיבות לאליפות",
            line=dict(color=BLUE),
            hovertemplate="%{x|%-d.%-m.%Y}<br>חשיבות=%{y:.4f}<extra></extra>",
        )
    )
    for i, (flag, label) in enumerate(
        (("title_decided", "האליפות הוכרעה"), ("nothing_to_play", "אין על מה לשחק"))
    ):
        part = frame.loc[frame[flag]]
        if not part.empty:
            day = part.as_of.min().isoformat()
            fig.add_shape(
                type="line",
                x0=day,
                x1=day,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=GRAY, dash="dash"),
            )
            fig.add_annotation(x=day, y=1.04 + 0.08 * i, yref="paper", text=label, showarrow=False)
    fig.update_xaxes(title="בוקר המשחק", tickformat="%-d.%-m.%Y")
    fig.update_yaxes(title="חשיבות לאליפות (0–1)", rangemode="tozero")
    zero_day = frame.loc[frame.title_lev.eq(0), "as_of"].min()
    flag_day = frame.loc[frame.title_decided, "as_of"].min()
    answer = (
        "החשיבות מגיעה לאפס לפני שדגל האליפות נדלק"
        if pd.notna(zero_day) and pd.notna(flag_day) and zero_day < flag_day
        else "חשיבות האליפות וההכרעה המתמטית לאורך העונה"
    )
    return _finish(
        fig,
        f"{team_name}: {answer}<br>n={len(frame)}; משחקים={len(frame)}; עונות=1",
    )


def seasonal_estimates(frame: pd.DataFrame) -> Any:
    """Annual continuous slope only; the uninformative small-sample flag line is omitted."""
    part = frame.loc[frame.representation.eq("Lc") & frame.metric.eq("res_pts")].sort_values(
        "season_id"
    )
    fig = go.Figure(
        go.Scatter(
            x=[season_label(int(s)) for s in part.season_id],
            y=part.estimate,
            mode="lines+markers",
            line=dict(color=BLUE),
            customdata=part[["n_used", "n_games", "n_seasons"]],
            hovertemplate="%{x}<br>שיפוע=%{y:+.3f}<br>n=%{customdata[0]}<br>משחקים=%{customdata[1]}<br>עונות=%{customdata[2]}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=GRAY)
    fig.update_xaxes(title="עונה")
    fig.update_yaxes(title="נקודות ליחידת פער חשיבות")
    return _finish(fig, "שיפוע החשיבות משנה כיוון בין עונות; הבלוקים הם בדיקת היציבות")


def block_estimates(frame: pd.DataFrame) -> Any:
    """Compare the same estimand in each block; points and slope units have separate panels."""
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.15,
        subplot_titles=["דגל: מתה מול השאר", "Lc: כל המשחקים", "Lc: משחקים חיים"],
    )
    for col, key, color in ((1, "dead", BLUE), (2, "all_Lc", "#9333EA"), (3, "live_Lc", "#0891B2")):
        part = frame.loc[frame.comparison_id.eq(key) & frame.metric.eq("res_pts")].sort_values(
            "block"
        )
        fig.add_trace(
            go.Scatter(
                x=[f"בלוק {b[-1]}" for b in part.block],
                y=part.estimate,
                mode="markers",
                marker=dict(color=color, size=9),
                **two_se(part.se),
                customdata=np.column_stack(
                    [part.block.map(VALUES), part[["n", "n_used", "n_games", "n_seasons"]]]
                ),
                showlegend=False,
                hovertemplate=(
                    "%{x}: %{customdata[0]}<br>אומדן=%{y:+.3f}<br>n=%{customdata[1]}"
                    "<br>בשימוש=%{customdata[2]}<br>משחקים=%{customdata[3]}"
                    "<br>עונות=%{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color=GRAY, row=1, col=col)
        fig.update_yaxes(
            title_text="נקודות" if key == "dead" else "נקודות ליחידת פער", row=1, col=col
        )
        fig.update_xaxes(title_text="בלוק של חמש עונות", tickangle=0, row=1, col=col)
    return _finish(
        fig, "הדגל שלילי בשלושת הבלוקים; שיפוע בין חיות אינו יציב באותו אופן", height=850
    )


def slope_readings(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate each declared slope and typical difference into a plain football sentence."""
    result = frame.copy()
    readings = []
    for row in result.itertuples():
        context = (
            "כשהבית חיה והחוץ מתה"
            if row.predictor == "L2_diff"
            else f"בפער מדד חציוני {row.median_diff:.4f} לטובת הבית"
        )
        control = " כשהדגל כבר בפנים" if row.comparison_id == "joint_Lc" else ""
        if row.comparison_id == "joint_L2":
            control = " כשפער החשיבות מוחזק קבוע"
        direction = "יותר" if row.median_effect >= 0 else "פחות"
        readings.append(
            f"{context}{control}, שארית הבית קשורה "
            f"לכ־{abs(row.median_effect):.3f} נקודות {direction} מציפיית Elo; קשר תיאורי."
        )
    result["football_reading"] = readings
    return result


def summary_words(frame: pd.DataFrame) -> pd.DataFrame:
    """Present the fixed summary examples without choosing categories by measured effects."""
    result = frame.copy()
    captures = {
        "L2": "מתה מול יריבה חיה: חזק בעשור הראשון, חלש ולא ודאי בחמש האחרונות",
        "L3": "כמעט בלי חשיבות מול שאר החיות",
        "L5": "חשיבות מעל 0.2 מול שאר החיות",
        "Lc": "פער חשיבות כשהדגל כבר בפנים",
        "Comp": "הישארות מובילה מעל 0.2, חיות בלבד",
        "transparent": "מאבק הישארות צמוד, חיות בלבד",
    }
    result["what_captured"] = result.representation.map(captures)
    result["next_test"] = "תרומה מחוץ למדגם מעל E0050 לפי הרשת הקבועה"
    return result


def decomposition_plot(frame: pd.DataFrame) -> Any:
    """Matched dead/control residual means, with the same season-bootstrap SE."""
    data = frame.loc[frame.scope.isin(["B1", "B2", "B3"]) & frame.metric.eq("res_pts")]
    fig = go.Figure()
    for field, label, color in (
        ("dead_mean", "ללא יעד", BLUE),
        ("control_mean", "קבוצות ההשוואה החיות", GRAY),
    ):
        fig.add_trace(
            go.Scatter(
                x=[VALUES[key] for key in ("B1", "B2", "B3")],
                y=data[field],
                **two_se(data[f"{field}_se"]),
                mode="markers",
                marker=dict(color=color, size=12),
                name=label,
                customdata=data[["n_used", "n_excluded", "n_games", "n_seasons"]],
                hovertemplate="%{x}<br>שארית %{y:.3f} נקודות<br>שורות מתות שימושיות: "
                "%{customdata[0]}<br>הוחרגו: %{customdata[1]}<br>משחקים: %{customdata[2]}"
                "<br>עונות: %{customdata[3]}<extra>%{fullData.name}</extra>",
            )
        )
    fig.add_hline(y=0, line_color=GRAY, line_dash="dash")
    recent = data.loc[data.scope.eq("B3")].iloc[0]
    fig.add_annotation(
        x=VALUES["B3"], y=float(recent.dead_mean), text="המתות על קו Elo", showarrow=True, ay=-65
    )
    fig.update_yaxes(title="שארית נקודות מול E0050, ממוצע ±2SE")
    fig.update_xaxes(title="בלוקים קבועים של חמש עונות; הטווחים בטבלה")
    return _finish(fig, "מי זזה מול Elo: המתות או קבוצות ההשוואה?")


def comparison_reading(n_used: int, estimate: float, se: float) -> str:
    """Apply the same declared minimum-30 coverage rule before interpreting uncertainty."""
    if n_used < 30:
        return "לא מספיק משחקים להשוואה"
    if not np.isfinite(estimate) or not np.isfinite(se):
        return "אי־הוודאות אינה ניתנת לאמידה"
    return "מעבר לפעמיים SE; תיאורי בלבד" if abs(estimate) > 2 * se else "אין הפרדה ברורה מהרעש"


def stage_residual_plot(frame: pd.DataFrame) -> Any:
    """OOF total-goals residuals by stage; hollow markers flag fewer than 30 fixtures."""
    fig = go.Figure()
    for population, name, color in (
        ("clean", "בלי מתה ובלי מאבק הישארות", BLUE),
        ("all_games", "כל המשחקים", GRAY),
    ):
        data = frame.loc[frame.population.eq(population)]
        fig.add_trace(
            go.Scatter(
                x=[VALUES.get(stage, stage) for stage in data.stage],
                y=data.goals_residual,
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=11,
                    symbol=["circle-open" if n < 30 else "circle" for n in data.n_games],
                ),
                **two_se(data.goals_se),
                customdata=data[["n_games", "n_seasons"]],
                text=[
                    "לא מספיק משחקים" if n < 30 else "כיסוי של שלושים משחקים ומעלה"
                    for n in data.n_games
                ],
                hovertemplate="%{x}<br>שארית שערים %{y:.3f}<br>משחקים %{customdata[0]}"
                "<br>עונות %{customdata[1]}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color=GRAY)
    fig.update_xaxes(title="שלב העונה")
    fig.update_yaxes(title="בפועל פחות λ בית וחוץ יחד (שערים למשחק), ±2SE")
    return _finish(fig, "האם מודל השערים מפספס בפלייאוף גם בלי מתות ומאבקי הישארות?")


def chapter4_takeaways(
    evidence: Mapping[str, pd.DataFrame],
    stage_review: pd.DataFrame | None = None,
    period_review: pd.DataFrame | None = None,
) -> TableDisplay:
    """Keep the game-level flag and the descriptive/predictive distinction explicit."""
    multi = evidence["multiplicity"].iloc[0]
    recent = (
        evidence["blocks"]
        .loc[
            lambda x: (
                x.comparison_id.eq("dead_vs_alive_opponent")
                & x.metric.eq("res_pts")
                & x.block.eq("B3")
            )
        ]
        .iloc[0]
    )
    stage_text = (
        "<li><b>מבט מדעי הנתונים:</b> במשחקים נקיים, λ קרוב לממוצע בסדיר ומעריך "
        "בחסר בפלייאוף על פני כל התקופה. בחמש האחרונות האומדן בעליון קרוב לאפס, "
        "ובתחתון אין מספיק משחקים נקיים להשוואה. אין כאן הסבר מספק לפי שלב "
        "למיעוט התיקו בין שתי מתות."
        "<br><b>מבט כדורגלני:</b> ההסבר לא ידוע; זו שאלה למחברת 05. כיול λ לפי "
        "שלב ייבדק ב-M2, ותיקון התיקו לפי שלב וסוג משחק ב-M3 במחברת 06.</li>"
        if stage_review is not None
        else ""
    )
    if period_review is not None:
        latest = period_review.loc[period_review.period.eq("last5")].set_index("game_type")
        both, one = latest.loc["both_dead"], latest.loc["one_dead"]
        stage_text = (
            "<li><b>מבט מדעי הנתונים:</b> בחמש האחרונות בקטגוריות המתות, עודף השערים הוא "
            f"{both.res_total_goals:+.2f} (SE {both.res_total_goals_se:.2f}) בין שתיים, ו-"
            f"{one.res_total_goals:+.2f} (SE {one.res_total_goals_se:.2f}) בצד מת יחיד "
            f"ללא מאבק הישארות. יחד {int(both.n_games + one.n_games)} משחקים בקטגוריות אלה; "
            "הקטגוריה עם צד מת יחיד קטנה משלושים. במשחקי העליון הנקיים האומדן "
            "קרוב לאפס; בתחתון הנקי המדגם קטן. לכן אין כאן הסבר מספק של שלב "
            "לעודף במשחקי המתות בעונות האחרונות. חסר התיקו בין שתי מתות קטן "
            f"ל-{both.res_draw:+.2f} (SE {both.res_draw_se:.2f})."
            "<br><b>מבט כדורגלני:</b> כשקבוצה כבר אינה צריכה כלום, המשחקים נראים "
            "פתוחים יותר, עם יותר שערים לשני הצדדים. זה תיאור ולא הוכחה סיבתית. "
            "הדגל ייבדק בכל מקרה בתוך ההשוואה המשולשת של מודל השערים במחברת 05 (ה-מ9), "
            "והתיקו במחברת 06.</li>"
        )
    return TableDisplay(
        normalize_minus(
            '<div dir="rtl"><p><b>התשובה בקצרה:</b> כנראה שאין כאן מדד מוטיבציה '
            "שמוסיף יכולת חיזוי של בית/תיקו/חוץ מעל Elo; EXP-004 יכריע בחלק B.</p>"
            "<ul><li><b>מבט מדעי הנתונים:</b> הדגל ברמת המשחק חזק בעשור הראשון, חלש ולא "
            "ודאי בחמש העונות האחרונות. במתה מול חיה בעונות 2019/2020–2023/2024: "
            f"{recent.estimate:+.2f} נקודות, SE {recent.se:.2f}, {int(recent.n)} משחקים. "
            "<br><b>מבט כדורגלני:</b> רק כשצד אחד כבר סיים את מטרותיו מתקבל פער בדגל "
            "בית פחות חוץ; כששתיהן חיות או שתיהן מתות, הפער מתאפס.</li>"
            "<li><b>מבט מדעי הנתונים:</b> הפירוק מפריד בין שארית המתות לשארית החיות "
            "באותם משקלים; בעונות 2019/2020–2023/2024 ההשוואה רגישה לשכבות קטנות. "
            "<br><b>מבט כדורגלני:</b> קבוצות חיות במחזור הסיום נלחמות על הישארות; "
            "ארבע שורות אינן בסיס יציב לייחס את כל ההפרש לוויתור של היריבה.</li>"
            "<li><b>מבט מדעי הנתונים:</b> מעבר לדגל אין יתרון נקודות ברור; בספיגה "
            "ובאופי המשחק נשאר תיאור מעניין. <br><b>מבט כדורגלני:</b> מאבק הישארות "
            "קשור לפחות שערים; בין שתי מתות יש פחות תיקו. השערים ייבדקו במחברת 05, "
            "והתיקו במחברת 06.</li>"
            f"{stage_text}"
            "<li><b>ריבוי השוואות:</b> "
            f"נספרו {int(multi.comparisons)} בדיקות מרכזיות; "
            f"{int(multi.get('estimated', multi.comparisons))} עם אומדן מספרי, "
            f"ומהן {int(multi.estimable)} כשירות עם SE חיובי ולפחות שלושים תצפיות שימושיות. "
            f"{int(multi.beyond_two_se)} מהכשירות עוברות |אומדן| &gt; 2·SE. "
            "נספרים האומדנים הייחודיים, שני ממוצעי הפירוק בכל מדד, הרגישות, שאריות "
            "המשחקים הגולמיות ובתוך שכבות מול כל שאר המשחקים ומול סוג כל השאר בלבד, "
            "שני מקדמי הספיגה המשותפים ושאריות לפי שלב ואוכלוסייה ולפי שתי התקופות "
            "של סוגי המשחקים. תאים ריקים או "
            "עם פחות משלושים תצפיות אינם כשירים לפירוש. בלוקים, עונות "
            "בודדות והצגה חוזרת אינן נספרות שוב. בספירה המורחבת כאן נכללים גם "
            "שמונת אומדני המדרגות ושבעת התרגומים לפער טיפוסי; האחרונים הם "
            "שינוי יחידות של אותם שיפועים, ולא ראיות נפרדות. "
            "<br><b>מבט כדורגלני:</b> ההשוואות תלויות; התיאור אינו מסקנה סיבתית "
            "ואינו בדיקת חיזוי. אף ייצוג לא נכנס היא תוצאה לגיטימית בחלק B.</li></ul></div>"
        )
    )


def walk_forward(plan: pd.DataFrame) -> Any:
    """Display the planned past-only folds; this diagram does not train any model."""
    years = list(range(8, 26))
    z = [
        [1 if s < int(row.season_id) else 2 if s == int(row.season_id) else 0 for s in years]
        for row in plan.itertuples()
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[season_label(s) for s in years],
            y=[f"קיפול {r.fold} — {'עמוק' if r.deep else 'דל'}" for r in plan.itertuples()],
            zmin=0,
            zmax=2,
            colorscale=[[0, "#DBEAFE"], [0.5, "#60A5FA"], [1, "#1E3A8A"]],
            showscale=False,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )
    for label, color in (("טרם נראה", "#DBEAFE"), ("אימון", "#60A5FA"), ("ולידציה", "#1E3A8A")):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                marker=dict(symbol="square", size=14, color=color),
            )
        )
    fig.update_xaxes(title="עונת הנתונים", tickangle=-45)
    fig.update_yaxes(title="קיפול והעומק שלו", autorange="reversed")
    return _finish(fig, "בכל קיפול לומדים רק מהעבר ובודקים בעונה הבאה", height=1150)
