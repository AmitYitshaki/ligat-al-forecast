"""Notebook 03 presentation: Hebrew season labels, diagnostic tables and stage charts."""

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]

from ligat.viz.league_stats import _finish
from ligat.viz.presentation import TableDisplay, display_table
from ligat.viz.seasons import season_label, with_season_labels

STAGES = {"regular": "סדיר", "upper": "עליון", "middle": "אמצעי", "lower": "תחתון"}
NAMES = {
    "season": "עונה",
    "n_teams": "קבוצות",
    "regular_rounds": "מחזורים בסדיר",
    "playoff_format": "פלייאוף: קבוצות / משחקים לקבוצה",
    "points_halved": "חצייה",
    "relegated": "יורדות ישירות",
    "relegation_test": "מקום במבחן",
    "upper_cut": "קו העליון",
    "last_safe": "המקום הבטוח האחרון",
    "tiebreak": "שוברי שוויון",
    "scope": "אימות",
    "category": "סיווג",
    "rows": "שורות",
    "counter_rows": "נבדקו מונים",
    "rank_rows": "נבדק מיקום",
    "display_rows": "סדר תצוגה מותר",
    "example": "משחק",
    "date": "תאריך",
    "round": "מחזור",
    "fixture": "מפגש",
    "team_name": "קבוצה",
    "basis": "בסיס הטבלה",
    "games_played": "משחקים ששוחקו",
    "points": "נקודות",
    "sporting_points": "נקודות ספורטיביות",
    "official_points": "נקודות רשמיות",
    "deduction": "שינוי מנהלתי",
    "technical_points": "תוספת נקודות טכניות באימות",
    "sporting_position": "מיקום ספורטיבי",
    "official_position": "מיקום רשמי",
    "rank_changed": "המיקום שונה",
    "officially_relegated": "ירדה בפועל",
    "sporting_group": "שיבוץ ספורטיבי מחושב",
    "actual_group": "שיבוץ מהלוח",
    "cause": "סיבה",
    "stage": "שלב",
    "games": "משחקים",
    "neither": "שתיהן עם יעד פתוח",
    "one": "קבוצה אחת בלי יעד",
    "both": "שתיהן בלי יעד",
    "at_least_one_share": "לפחות אחת בלי יעד",
    "both_share": "שתיהן בלי יעד (%)",
    "rounds_to_stage_end": "משחקים בשלב אחרי המשחק",
    "team_games": "שורות קבוצה×משחק",
    "definition": "הגדרת הדגל",
    "dead_team_rows": "שורות קבוצה מסומנות",
    "dead_team_share": "שיעור שורות קבוצה מסומנות",
    "feature": "פיצ׳ר",
    "block": "בלוק",
    "status": "מעמד",
    "availability": "זמינות",
    "models": "מודלים",
    "decision": "החלטה",
    "model_status": "חריג למעמד לפי מודל",
    "game_id": "מזהה משחק",
    "team_id": "מזהה קבוצה",
}
VALUES = STAGES | {
    "round": "טבלת מחזור",
    "final": "טבלת סיום",
    "match": "זהה בספורטיבי",
    "postponed": "מחזור עם משחק שחוצה מחזורים — הוחרג",
    "source_artifact": "ארטיפקט מקור",
    "technical": "פער טכני ששוחזר באימות",
    "deduction": "הורדת נקודות",
    "display_order": "סדר תצוגה",
    "target": "המשחק הנחזה",
    "delayed": "משחק קודם ששוחק אחריו",
    "morning": "בוקר המשחק לפי as_of",
    "official_round_end": "הרשמית אחרי המחזור (כוללת תוצאות עתידיות)",
    "none": "ללא פלייאוף",
    "True": "כן",
    "False": "לא",
}


def review_table(
    frame: pd.DataFrame, *, percent: tuple[str, ...] = (), columns: list[str] | None = None
) -> TableDisplay:
    """Show readable seasons and hide internal season IDs without altering the frame."""
    data = with_season_labels(frame)
    if "season_id" in data:
        data = data.drop(columns="season_id")
        data = data[["season", *[column for column in data if column != "season"]]]
    for column in ("playoff_format", "tiebreak"):
        if column in data:
            replacements = STAGES | {
                "none": "—",
                "points": "נקודות",
                "goal_difference": "הפרש",
                "wins": "ניצחונות",
                "goals_for": "שערי זכות",
                "head_to_head": "ראש בראש",
            }
            for source, target in replacements.items():
                data[column] = data[column].str.replace(source, target, regex=False)
    precision = {column: 0 for column in data if column not in percent}
    precision.update({"official_position": 0})
    # Nonexistent format elements are not missing observations. Keep this local
    # to presentation; the source frame and nullable numeric dtypes stay intact.
    for column in ("relegation_test", "upper_cut"):
        if column in data:
            data[column] = data[column].astype(object).where(data[column].notna(), "—")
    if "sporting_position" in data:
        data["sporting_position"] = data.sporting_position.map(
            lambda value: value if pd.isna(value) else f"{value:g}"
        )
    return display_table(
        data,
        NAMES,
        columns=columns,
        values=VALUES,
        percent=percent,
        precision=precision,
        show_index=False,
        unicode_minus=("deduction", "technical_points"),
    )


def dead_games_chart(summary: pd.DataFrame) -> Any:
    """Compare the share with at least one mathematically dead team across seasons/stages."""
    frame = summary.loc[summary.stage.ne("middle")].copy()
    frame["season"] = frame.season_id.map(lambda sid: season_label(int(sid)))
    frame["stage"] = frame.stage.map(STAGES)
    frame["share_percent"] = 100 * frame.at_least_one_share
    colors = {"סדיר": "#0072B2", "עליון": "#D55E00", "תחתון": "#009E73"}
    figure = px.line(
        frame,
        x="season",
        y="share_percent",
        color="stage",
        markers=True,
        custom_data=["games", "one", "both"],
        color_discrete_map=colors,
        labels={
            "season": "עונה",
            "share_percent": "% משחקים עם לפחות קבוצה אחת בלי יעד",
            "stage": "שלב",
        },
        category_orders={"season": sorted(frame.season.unique()), "stage": list(STAGES.values())},
    )
    figure.update_traces(
        hovertemplate=(
            "עונה %{x}<br>%{y:.1f}%<br>משחקים %{customdata[0]}"
            "<br>אחת בלי יעד %{customdata[1]}<br>שתיים בלי יעד %{customdata[2]}"
            "<extra>%{fullData.name}</extra>"
        )
    )
    figure.update_layout(
        title="משחקים בלי יעד פתוח נפוצים יותר בפלייאוף מאשר בסדיר", yaxis_rangemode="tozero"
    )
    figure.update_xaxes(
        categoryorder="array",
        categoryarray=sorted(frame.season.unique()),
        tickangle=-35,
        title_standoff=55,
        automargin=True,
    )
    figure = _finish(figure)
    # This categorical comparison deliberately overrides the common sequential
    # palette, which _finish otherwise applies even to explicitly supplied colors.
    for trace in figure.data:
        color = next(color for stage, color in colors.items() if stage in trace.name)
        trace.line.color = color
        trace.marker.color = color
    return figure
