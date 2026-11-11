"""Readable, descriptive match checks for notebook 03; no model or data writes."""

import re
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ligat.analysis.prematch_final_review import FEATURE_NAMES, TARGET_NAMES, WORDS
from ligat.analysis.prematch_match_review import MatchReview
from ligat.viz.league_stats import GOOD_SCALE
from ligat.viz.prematch import STAGES
from ligat.viz.prematch_b import _finish_chart
from ligat.viz.presentation import TableDisplay, display_table

LABELS = {
    "target": "יעד",
    "feature": "מדד",
    "games": "משחקים",
    "bin": "עשירון",
    "variance": "שונות בפועל",
    "noise": "רעש Poisson משוער",
    "signal_fraction": "חלק שונות נותר",
    "heuristic_correlation": "שורש החלק הנותר",
    "elo_low": "Elo מזערי",
    "elo_high": "Elo מרבי",
    "elo_mean": "Elo ממוצע",
    "mean_gd": "הפרש שערים ממוצע",
    "home_win_percent": "ניצחון בית (%)",
    "draw_percent": "תיקו (%)",
    "away_win_percent": "ניצחון חוץ (%)",
    "pearson": "Pearson",
    "eta": "יחס מתאם η",
    "groups": "קבוצות",
    "pearson_early": "2007/2008–2018/2019",
    "pearson_recent": "2019/2020–2023/2024",
    "change": "שינוי",
    "games_early": "משחקים מוקדם",
    "games_recent": "משחקים מאוחר",
    "period": "תקופה",
    "value_elo": "חפיפה שווי–Elo",
    "partial_value": "מתאם חלקי",
    "change_difference": "הפרש ההתחזקויות",
    "se": "שגיאת תקן SE",
    "descriptive_z": "הפרש ביחידות SE",
    "repeats": "חזרות",
    "seed": "זרע אקראי",
    "pearson_home": "מתאם בבית",
    "pearson_away": "מתאם בחוץ",
    "group": "קבוצה",
    "outcome": "תוצאה",
    "observed": "נצפה",
    "expected": "צפוי ללא קשר",
    "contribution": "תרומה ל־χ²",
    "chi2": "χ²",
    "dof": "דרגות חופש",
    "p_value": "p תיאורי",
    "cramers_v": "קרמר V",
    "expected_promoted_home": "ניצחונות בית צפויים לעולות",
    "draws": "תוצאות תיקו",
    "matched_dead": "מתים מותאמים",
    "all_dead": "כל המתים",
    "excluded_dead": "ללא התאמה",
    "difference_pp": "לא מתים פחות מתים (נ״א)",
    "se_pp": "SE (נ״א)",
    "stage": "שלב",
    "rounds_to_stage_end": "משחקים לאחר הנוכחי",
    "dead_n": "משחקים מתים",
    "alive_n": "משחקי השוואה",
    "weight": "משקל שכבה",
    "alive_rate": "שיעור תיקו בבקרה",
    "minimum_controls": "מינימום משחקי השוואה בשכבה",
    "dead_draw_percent": "תיקו במתים (%)",
    "alive_draw_percent": "תיקו בלא־מתים מותאמים (%)",
    WORDS: "מה זה, במילים",
}
COUNTS = {
    "games",
    "bin",
    "groups",
    "games_early",
    "games_recent",
    "repeats",
    "seed",
    "observed",
    "dof",
    "draws",
    "matched_dead",
    "all_dead",
    "excluded_dead",
    "dead_n",
    "alive_n",
    "rounds_to_stage_end",
    "minimum_controls",
}


def match_table(frame: pd.DataFrame, columns: list[str]) -> TableDisplay:
    """Unique explanations, isolated Unicode signs and integer counts in RTL tables."""
    if WORDS not in columns or WORDS not in frame or frame[WORDS].duplicated().any():
        raise ValueError("Every match table needs different words for every row")
    data = frame.copy()
    for column, names in (("feature", FEATURE_NAMES), ("target", TARGET_NAMES), ("stage", STAGES)):
        if column in data:
            data[column] = data[column].astype(object).replace(names)
    data[WORDS] = data[WORDS].map(
        lambda text: re.sub(
            r"\d{4}/\d{4}–\d{4}/\d{4}",
            lambda match: "\u2066" + match.group() + "\u2069",
            re.sub(
                r"\b(?:regular|upper|lower)\b",
                lambda match: STAGES[match.group()],
                str(text),
            ),
        )
    )
    if "p_value" in data:
        data["p_value"] = data.p_value.map(lambda x: f"{x:.3g}")
    return display_table(
        data,
        {
            c: "\u2066" + LABELS.get(c, c) + "\u2069"
            if c in {"pearson_early", "pearson_recent"}
            else LABELS.get(c, c)
            for c in columns
        },
        columns=columns,
        show_index=False,
        precision={c: 0 if c in COUNTS else 4 for c in columns},
        unicode_minus=tuple(columns),
    )


def period_chart(periods: pd.DataFrame) -> Any:
    """Disjoint windows on the same zero-based correlation scale."""
    labels = periods.feature.map(FEATURE_NAMES)
    figure = go.Figure()
    for column, name, color in (
        ("pearson_early", "2007/2008–2018/2019", GOOD_SCALE[3]),
        ("pearson_recent", "2019/2020–2023/2024", GOOD_SCALE[7]),
    ):
        figure.add_trace(
            go.Scatter(
                x=periods[column],
                y=labels,
                mode="markers",
                name=name,
                marker=dict(size=13, color=color),
                hovertemplate="%{y}<br>Pearson: %{x:.5f}<extra>" + name + "</extra>",
            )
        )
    figure.update_layout(
        title="השווי התחזק יותר, אבל Elo חזק ממנו בשתי התקופות",
        xaxis_title="Pearson מול הפרש שערים — תיאור בלבד",
    )
    figure.update_xaxes(range=[0, 0.5])
    figure.update_yaxes(categoryorder="array", categoryarray=labels.tolist(), autorange="reversed")
    _finish_chart(figure, height=560)
    figure.update_layout(margin=dict(l=280, r=40, t=95, b=90))
    return figure


def match_description(review: MatchReview, section: str) -> str:
    """Compute numerical prose from the same tables that the notebook displays."""
    tables = review.tables
    if section == "ceiling":
        row = tables["ceilings"].iloc[0]
        line = tables["linearity"].iloc[0]
        text = (
            f"שונות הפרש השערים {row.variance:.4f}, רעש משוער {row.noise:.4f}; "
            f"החלק הנותר {row.signal_fraction:.4f}, ושורשו {row.heuristic_correlation:.4f}. "
            f"Pearson הוא {line.pearson:.4f}, וריבועו {100 * line.pearson**2:.1f}% מהשונות "
            f"בקירוב ליניארי בתוך המדגם. יחס המתאם η ב־{int(line.groups)} קבוצות "
            f"הוא {line.eta:.4f}. "
            "הקרבה תואמת קשר ליניארי בעיקרו; אינה מבחן שמוכיח ליניאריות. "
            "זה אומדן רעש תחת Poisson עצמאי, ולא תקרה מוכחת או R² מחוץ למדגם. "
            "עודף תיקו והבדלי קצב בין משחקים עשויים לשנות את אומדן הרעש; אין כאן טווח סמך לתקרה."
        )
    elif section == "periods":
        p = tables["periods"].set_index("feature")
        row = tables["bootstrap"].iloc[0]
        text = (
            f"השווי התחזק ב־{p.at['value_avg_log_ratio', 'change']:.4f}, ופער Elo "
            f"ב־{p.at['elo_diff', 'change']:.4f}; ההפרש {row.change_difference:.4f}, "
            f"SE {row.se:.4f}, כלומר {row.descriptive_z:.2f} שגיאות תקן. "
            f"{int(row.repeats):,} דגימות משחקים מזווגות בתוך כל תקופה, זרע {int(row.seed)}. "
            "הדגימה מניחה משחקים עצמאיים ואינה מתקנת תלות בין משחקי אותו מועדון. "
            "רוב ההתחזקות משותפת ל־Elo: מתאם חלקי קטן אינו עדות לתרומה גדולה במודל."
        )
    elif section == "dead":
        d, c = tables["dead"], tables["dead_comparison"].iloc[0]
        text = (
            f"בכל {int(d.iloc[0].games)} המשחקים המתים: {d.iloc[0].draw_percent:.2f}% תיקו. "
            f"בהשוואה ההוגנת נכללים {int(c.matched_dead)}: "
            f"{d.iloc[1].draw_percent:.2f}% מול {d.iloc[3].draw_percent:.2f}% בלא־מתים מותאמים. "
            f"הפער {c.difference_pp:.2f} נקודות אחוז, SE {c.se_pp:.2f}, "
            f"כ־{c.descriptive_z:.2f} שגיאות תקן. "
            f"{int(c.excluded_dead)} משחקים ללא בקרה באותה שכבה אינם משתתפים בהפרש. "
            f"בהשוואה הגסה לשלושת האחרונים: {int(d.iloc[2].games)} משחקים לא מתים, "
            f"{d.iloc[2].draw_percent:.2f}% תיקו. "
        )
        sensitivity = tables["dead_sensitivity"].iloc[1]
        removed = tables["dead_excluded_strata"]
        sensitivity_gap = f"{sensitivity.difference_pp:.2f}".replace("-", "−")
        text += (
            f"בבדיקת הרגישות מסירים {len(removed)} שכבות עם פחות מ־"
            f"{int(sensitivity.minimum_controls)} משחקי השוואה. נשארים "
            f"{int(sensitivity.matched_dead)} מתים: {sensitivity.dead_draw_percent:.2f}% "
            f"מול {sensitivity.alive_draw_percent:.2f}%; לא־מתים פחות מתים = "
            f'<bdi dir="ltr">{sensitivity_gap}</bdi> נ״א, '
            f"SE {sensitivity.se_pp:.2f}. "
            "אין ראיה אמינה שמשחקים מתים מסתיימים בפחות תיקו. "
            "הפער נשען כמעט כולו על המחזור האחרון בפלייאופים, שבו כמעט כל המשחקים מתים "
            "וקבוצת ההשוואה זעירה. במחזור האחרון בתחתון, המשחקים שעוד חשובים הם "
            "קרבות ירידה, שבהם תיקו לפעמים מספיק לשני הצדדים; זה הסבר כדורגלני אפשרי, "
            "ולא עובדה שנבדקה לכל משחק."
        )
    else:
        raise ValueError("Unknown match description section")
    return '<div dir="rtl">' + text + "</div>"
