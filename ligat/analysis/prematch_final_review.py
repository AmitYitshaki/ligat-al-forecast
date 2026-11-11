"""Read-only notebook C9–C14: the frozen M0 table and descriptive C5 associations."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.analysis.prematch_correlations import (
    TARGET_COLUMNS,
    OutcomeAssociation,
    contrast_elo_correlations,
    feature_catalog,
    feature_outcome_chi2,
    feature_pair_correlations,
    feature_target_correlations,
    game_targets,
)
from ligat.analysis.prepare import AnalysisTables
from ligat.features.pipeline import (
    ELO_COLUMNS,
    PREVIOUS_VALUE_COLUMNS,
    SELECTED_CONTEXT_COLUMNS,
    SELECTED_FORM_COLUMNS,
    TABLE_COLUMNS,
    load_prematch_config,
    validate_prematch_frame,
)
from ligat.features.registry import PREMATCH_CONTRAST_COLUMNS, PREMATCH_OPPONENT_COLUMNS
from ligat.features.table_state import game_rows

WORDS = "מה השיטה עושה, במילים"
FEATURE_NAMES = {
    "is_home": "הקבוצה בבית",
    "games_played": "משחקים קודמים",
    "wins": "ניצחונות קודמים",
    "draws": "תיקו קודמים",
    "losses": "הפסדים קודמים",
    "goals_for": "שערי זכות מצטברים לפני המשחק",
    "goals_against": "שערי חובה מצטברים לפני המשחק",
    "goal_difference": "הפרש שערים מצטבר לפני המשחק",
    "points": "נקודות בטבלה לפני המשחק",
    "position": "מיקום בטבלה לפני המשחק",
    "group_position": "מיקום בתוך קבוצת הטבלה",
    "points_per_game": "קצב נקודות לפני המשחק",
    "position_pct": "מיקום יחסי לפני המשחק",
    "gap_to_leader": "פער מהמובילה",
    "gap_to_safety": "פער מקו ההישרדות",
    "gap_to_cut": "פער מקו העליון",
    "games_remaining_stage": "משחקים שנותרו, כולל הנוכחי",
    "games_behind": "משחקים חסרים ביחס למרב בשלב",
    "title_decided": "יעד אליפות מוכרע",
    "cut_decided": "יעד עליון מוכרע",
    "relegation_decided": "יעד הישרדות מוכרע",
    "places_decided": "קווי מקום מוכרעים",
    "nothing_to_play": "הקבוצה בלי יעד פתוח",
    "rounds_to_stage_end": "משחקים בשלב אחרי הנוכחי",
    "elo_pre": "Elo של הקבוצה",
    "elo_opponent_pre": "Elo של היריבה",
    "elo_diff": "פער Elo",
    "elo_expected": "ציפיית Elo",
    "gf_venue_ewma5": "כושר הבקעה לפי מגרש",
    "ga_venue_ewma5": "כושר ספיגה לפי מגרש",
    "pts_venue_ewma5": "כושר נקודות לפי מגרש",
    "opp_gf_venue_ewma5": "כושר הבקעה של היריבה",
    "opp_ga_venue_ewma5": "כושר ספיגה של היריבה",
    "prev_value_share_avg": "נתח שווי ממוצע אשתקד",
    "prev_position_pct": "מיקום יחסי אשתקד",
    "prev_ppm": "נקודות למשחק אשתקד",
    "prev_gd_per_game": "הפרש שערים למשחק אשתקד",
    "promoted_in": "הקבוצה עולה",
    "squad_value_imputed": "שווי הושלם",
    "prev_season_known": "עונה קודמת ידועה",
    "league_size": "קבוצות בליגה",
    "value_avg_log_ratio": "יחס שווי ממוצע אשתקד",
    "position_log_ratio": "יחס מיקום אשתקד; נמוך טוב",
    "prev_ppm_diff": "פער נקודות למשחק אשתקד",
    "prev_gd_diff": "פער הפרש שערים אשתקד",
    "stage": "שלב המשחק",
    "away_stage": "שלב המשחק של האורחת",
    "away_promoted_in": "האורחת עולה",
    "away_nothing_to_play": "האורחת בלי יעד פתוח",
    "date": "תאריך המשחק",
    "as_of": "המידע נכון לבוקר המשחק",
}
TARGET_NAMES = {
    "goals_scored": "שערי הקבוצה במשחק",
    "goals_conceded": "שערי היריבה במשחק",
    "goal_difference_game": "הפרש שערים במשחק",
    "points_game": "נקודות 3/1/0 במשחק",
}
TREND_FEATURES = (
    "elo_expected",
    "elo_diff",
    "prev_gd_diff",
    "prev_ppm_diff",
    "value_avg_log_ratio",
)


@dataclass(frozen=True)
class PrematchFinalReview:
    """Aggregates only; observed per-game targets remain temporary in-memory inputs."""

    frame: pd.DataFrame
    catalog: pd.DataFrame
    pairs: pd.DataFrame
    associations: pd.DataFrame
    matrices: dict[str, pd.DataFrame]
    outcomes: OutcomeAssociation
    sha256: str


def load_final_prematch(root: Path) -> tuple[pd.DataFrame, str]:
    """Verify v1, the four pinned source blocks and original gap columns before reading."""
    path = root / "data/derived/prematch/prematch_v1.parquet"
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = load_prematch_config(root / "config/prematch.yaml")
    if (
        metadata.get("component") != "prematch"
        or metadata.get("version") != "1"
        or metadata.get("artifact_sha256") != digest
        or metadata.get("inputs") != spec.input_records()
        or metadata.get("config_sha256") != spec.sha256
    ):
        raise ValueError("Final review prematch checksum/provenance mismatch")
    for item in spec.inputs.values():
        for target, expected in (
            (root / item.path, item.sha256),
            ((root / item.path).with_suffix(".metadata.json"), item.metadata_sha256),
        ):
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Final review frozen input checksum mismatch: {target}")
    frame = pd.read_parquet(path)
    validate_prematch_frame(frame)
    if metadata.get("rows") != len(frame) or metadata.get("columns") != list(frame):
        raise ValueError("Final review prematch dimensions differ from sidecar")
    gaps = ["gap_to_leader", "gap_to_safety", "gap_to_cut"]
    original = pd.read_parquet(root / spec.inputs["table_state"].path, columns=gaps)
    if not frame[gaps].equals(original.reindex(frame.index)):
        raise ValueError("Final review gap values or missingness differ from frozen table_state")
    return frame, digest


def analyze_final_prematch(
    frame: pd.DataFrame, tables: AnalysisTables, sha256: str
) -> PrematchFinalReview:
    """Delegate all C5 statistics to C5; append seasonal ranges for the stability display."""
    if not tables.match_table.season_id.isin(range(8, 26)).all():
        raise ValueError("Final review requires development seasons 8–25")
    targets = game_targets(frame, game_rows(tables))
    catalog = feature_catalog(frame)
    associations = feature_target_correlations(frame, targets)
    # C5 exposes seasonal medians and sign counts, but not the endpoints. These
    # extra display columns describe the requested range; C5 estimates stay intact.
    combined = frame[catalog.feature.tolist()].astype(float).join(targets)
    seasonal = combined.groupby(frame.season_id).corr()
    for bound in ("min", "max"):
        by_feature = getattr(seasonal.groupby(level=1), bound)()
        associations[f"pearson_{bound}_season"] = [
            by_feature.at[row.feature, row.target] for row in associations.itertuples()
        ]
    return PrematchFinalReview(
        frame,
        catalog,
        feature_pair_correlations(frame),
        associations,
        contrast_elo_correlations(frame),
        feature_outcome_chi2(frame, targets),
        sha256,
    )


def check_c5_reference(review: PrematchFinalReview, path: Path) -> None:
    """Compare every original C5 numeric aggregate to the saved descriptive report."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, actual, keys in (
        ("feature_pairs", review.pairs, ["feature_a", "feature_b"]),
        ("feature_targets", review.associations, ["feature", "target"]),
        ("outcome_chi2", review.outcomes.statistics, ["feature"]),
        ("outcome_percentages", review.outcomes.percentages, ["feature", "category"]),
    ):
        expected = pd.DataFrame(payload[key]).set_index(keys).sort_index()
        data = actual.set_index(keys).sort_index()
        if not data.index.equals(expected.index):
            raise ValueError(f"C5 reference population changed: {key}")
        for column in data.columns.intersection(expected.columns):
            if pd.api.types.is_numeric_dtype(data[column]) and column != "expected_below_5":
                if not np.allclose(
                    data[column].to_numpy(float),
                    expected[column].to_numpy(float),
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                ):
                    raise ValueError(f"C5 reference mismatch: {key}.{column}; stop and report")


def column_groups() -> pd.DataFrame:
    """The six groups from DATA_DICTIONARY, counting columns rather than model features."""
    groups = [
        ("טבלה", TABLE_COLUMNS, "table_state_v2", "מצב ספורטיבי לפני המשחק, תאריך, שלב ודגלים"),
        ("Elo", ELO_COLUMNS, "elo_E0050_prematch", "דירוג הקבוצה והיריבה, הפער וציפיית Elo"),
        ("כושר הקבוצה", SELECTED_FORM_COLUMNS, "form_v1", "EWMA5 לפי מגרש: הבקעה, ספיגה ונקודות"),
        (
            "כושר היריבה",
            PREMATCH_OPPONENT_COLUMNS,
            "שורת היריבה",
            "ההתקפה וההגנה של היריבה באותו משחק",
        ),
        (
            "הקשר עונתי",
            SELECTED_CONTEXT_COLUMNS,
            "season_context_v1",
            "עבר העונה, דגלים וגודל הליגה; ללא שווי כולל",
        ),
        (
            "הפרשים",
            PREMATCH_CONTRAST_COLUMNS,
            "שתי שורות המשחק",
            "ארבע השוואות בין הקבוצה ליריבה; משנות סימן במעבר צד",
        ),
    ]
    return pd.DataFrame(
        [
            {"group": name, "columns": len(cols), "source": source, WORDS: words}
            for name, cols, source, words in groups
        ]
    )


def match_example(frame: pd.DataFrame, team_config: Path) -> pd.DataFrame:
    """A season-25 fixture side by side, with names exclusively from teams.yaml."""
    teams = yaml.safe_load(team_config.read_text(encoding="utf-8"))["teams"]
    names = {int(t["team_id"]): t["name_he"] for t in teams}
    home = frame.loc[
        frame.season_id.eq(25) & frame.is_home & frame.games_played.ge(10)
    ].sort_values(["date", "opponent_id"])
    if home.empty:
        raise ValueError("Season 25 example is unavailable")
    key = home.index[0]
    a = frame.loc[key]
    b = frame.loc[(key[0], int(a.opponent_id))]
    if b.is_home or int(b.opponent_id) != key[1]:
        raise ValueError("Example fixture does not contain mirrored opponents")
    contrasts = [*PREMATCH_CONTRAST_COLUMNS, "elo_diff"]
    if not np.allclose(a[contrasts].astype(float), -b[contrasts].astype(float), atol=1e-12):
        raise ValueError("Example contrasts are not antisymmetric")
    if a.opp_gf_venue_ewma5 != b.gf_venue_ewma5 or b.opp_gf_venue_ewma5 != a.gf_venue_ewma5:
        raise ValueError("Example opponent attack is not mirrored")
    columns = [
        "date",
        "as_of",
        "stage",
        "is_home",
        "points",
        "goals_for",
        "goals_against",
        "elo_diff",
        "elo_expected",
        "gf_venue_ewma5",
        "ga_venue_ewma5",
        "opp_gf_venue_ewma5",
        "opp_ga_venue_ewma5",
        *PREMATCH_CONTRAST_COLUMNS,
    ]
    return pd.DataFrame(
        [
            {
                "feature": c,
                "home": a[c],
                "away": b[c],
                "home_name": names[int(key[1])],
                "away_name": names[int(a.opponent_id)],
                WORDS: (
                    "ציפיית Elo עם תוספת הבית 30; אינה הסתברות ניצחון"
                    if c == "elo_expected"
                    else FEATURE_NAMES[c]
                ),
            }
            for c in columns
        ]
    )


def missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Distinguish the initial-season absence from meaningful table-gap missingness."""
    rows = []
    for feature in [
        *PREVIOUS_VALUE_COLUMNS,
        *PREMATCH_CONTRAST_COLUMNS,
        "gap_to_leader",
        "gap_to_safety",
        "gap_to_cut",
    ]:
        missing = frame[feature].isna()
        rows.append(
            dict(
                feature=feature,
                missing=int(missing.sum()),
                missing_first=int((missing & frame.season_id.eq(8)).sum()),
                missing_later=int((missing & frame.season_id.ne(8)).sum()),
                **{
                    WORDS: "אין t−1 ב-2006/2007"
                    if feature in (*PREVIOUS_VALUE_COLUMNS, *PREMATCH_CONTRAST_COLUMNS)
                    else "פער אינו מוגדר בכל שלב; נשמר כמו ב-table_state_v2"
                },
            )
        )
    return pd.DataFrame(rows)


def overlap_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify the first three overlap statements directly, without removing any column."""
    home = frame.loc[frame.is_home]
    keys = pd.MultiIndex.from_arrays([home.index.get_level_values("game_id"), home.opponent_id])
    stage_differences = int((home.stage.to_numpy() != frame.reindex(keys).stage.to_numpy()).sum())
    remaining = frame.games_remaining_stage - frame.rounds_to_stage_end
    exceptions = frame.promoted_in.ne(frame.squad_value_imputed)
    seasons = sorted(frame.loc[exceptions, "season_id"].unique().tolist())
    if not remaining.eq(1).all() or stage_differences or exceptions.sum() != 35 or seasons != [11]:
        raise ValueError("C5 overlap reference changed; stop and report")
    return pd.DataFrame(
        [
            {
                "pair": "games_remaining_stage / rounds_to_stage_end",
                "value": int(remaining.iloc[0]),
                WORDS: "הראשון פחות השני תמיד 1: כולל המשחק הנוכחי לעומת אחריו",
            },
            {
                "pair": "stage בית / stage חוץ",
                "value": stage_differences,
                WORDS: "אפס משחקים עם שלב שונה בין שתי השורות",
            },
            {
                "pair": "promoted_in / squad_value_imputed",
                "value": int(exceptions.sum()),
                WORDS: "שונים רק ב-35 שורות של 2009/2010, עקב השלמת השווי של הפועל פ״ת",
            },
            {
                "pair": "elo_diff / elo_expected",
                "value": np.nan,
                WORDS: (
                    "ציפיית Elo היא המרה לוגיסטית של הפער, עם יתרון הבית וכיוון השורה; "
                    "אינה מדד כוח עצמאי"
                ),
            },
        ]
    )


def cross_family_pairs(review: PrematchFinalReview) -> pd.DataFrame:
    """Show exactly the nine flagged cross-family pairs; keep all features intact."""
    result = review.pairs.loc[review.pairs.flagged & ~review.pairs.same_family].copy()
    result[WORDS] = (
        result.feature_a.map(FEATURE_NAMES) + " מול " + result.feature_b.map(FEATURE_NAMES)
    )
    return result.sort_values("pearson", key=lambda s: s.abs(), ascending=False)


def target_leaders(review: PrematchFinalReview, target: str, count: int) -> pd.DataFrame:
    """Rank only the display by absolute Pearson; no selection criterion is created."""
    if target not in TARGET_COLUMNS:
        raise ValueError("Unknown descriptive target")
    result = review.associations.loc[review.associations.target.eq(target)].copy()
    result[WORDS] = result.feature.map(FEATURE_NAMES) + " מול " + TARGET_NAMES[target]
    return result.sort_values("pearson", key=lambda s: s.abs(), ascending=False).head(count)


def target_trends(review: PrematchFinalReview) -> pd.DataFrame:
    """The five predeclared correlations in the two requested windows."""
    part = review.associations.loc[review.associations.target.eq("goal_difference_game")]
    result = part.set_index("feature").loc[list(TREND_FEATURES)].reset_index()
    result["change"] = result.pearson_last5 - result.pearson
    result[WORDS] = result.feature.map(FEATURE_NAMES) + " מול הפרש שערים במשחק; קשר תיאורי בלבד"
    return result


def outcome_statistics(review: PrematchFinalReview, *, excluded: bool = False) -> pd.DataFrame:
    """Preserve excluded features in their own table, without zero-filling missing tests."""
    result = review.outcomes.statistics.copy()
    result = result.loc[
        result.excluded_reason.ne("") if excluded else result.excluded_reason.eq("")
    ]
    result[WORDS] = result.feature.map(FEATURE_NAMES)
    return result.sort_values("cramers_v", ascending=False)


def dead_game_summary(review: PrematchFinalReview) -> pd.DataFrame:
    """Home-row Elo comparisons for dead-home, dead-away, their union and all games."""
    home = review.frame.loc[review.frame.is_home]
    keys = pd.MultiIndex.from_arrays([home.index.get_level_values("game_id"), home.opponent_id])
    away_dead = pd.Series(review.frame.reindex(keys).nothing_to_play.to_numpy(), index=home.index)
    union = home.nothing_to_play | away_dead
    statistics = review.outcomes.statistics.set_index("feature")
    percentages = review.outcomes.percentages
    rows = []
    for feature, mask in (
        ("nothing_to_play", home.nothing_to_play),
        ("away_nothing_to_play", away_dead),
    ):
        result = percentages.loc[
            percentages.feature.eq(feature) & percentages.category.eq("True")
        ].iloc[0]
        rows.append(
            {
                "feature": feature,
                "games": int(mask.sum()),
                "draw_percent": result.D_percent,
                "mean_elo_diff": float(home.loc[mask, "elo_diff"].mean()),
                "cramers_v": statistics.at[feature, "cramers_v"],
                "union_games": int(union.sum()),
                WORDS: "פער Elo מנקודת מבט הבית; לא בקרה סיבתית לחוזק היריבה",
            }
        )
    return pd.DataFrame(rows)


def final_controls(review: PrematchFinalReview) -> pd.DataFrame:
    """Stop on any disagreement with the declared C5 and C6b QA anchors."""
    dead = dead_game_summary(review).set_index("feature")
    pairs = review.pairs
    records = [
        ("פיצ׳רים מספריים ובוליאניים", len(review.catalog), 45, 0),
        ("זוגות", len(pairs), 990, 0),
        ("זוגות מסומנים", int(pairs.flagged.sum()), 48, 0),
        ("מסומנים בין משפחות", len(cross_family_pairs(review)), 9, 0),
        ("בתוך משפחה", int((pairs.flagged & pairs.same_family).sum()), 39, 0),
        ("משחקי בית", review.outcomes.home_rows, 4315, 0),
        ("Pearson מרבי ליעד", review.associations.pearson.abs().max(), 0.3933, 0.00005),
        ("Cramér's V מרבי", review.outcomes.statistics.cramers_v.max(), 0.2272, 0.00005),
        ("תיקו כשהבית מתה (%)", dead.at["nothing_to_play", "draw_percent"], 18.39, 0.005),
        ("תיקו כשהאורחת מתה (%)", dead.at["away_nothing_to_play", "draw_percent"], 18.125, 1e-10),
        ("Elo ממוצע כשהבית מתה", dead.at["nothing_to_play", "mean_elo_diff"], 5.9, 0.1),
        ("Elo ממוצע כשהאורחת מתה", dead.at["away_nothing_to_play", "mean_elo_diff"], -13.8, 0.1),
        ("משחקים עם לפחות קבוצה מתה", dead.union_games.iloc[0], 245, 0),
    ]
    result = pd.DataFrame(
        [
            {
                "quantity": name,
                "value": float(value),
                "expected": expected,
                "passed": abs(value - expected) <= tolerance,
                WORDS: f"{name}: הערך המחושב מושווה לעוגן האימות שנקבע מראש",
            }
            for name, value, expected, tolerance in records
        ]
    )
    if (
        not result.passed.all()
        or review.sha256 != "246a638d7754609b2efecf77120857857b471d03a1d42c7420fd96a327d69e17"
    ):
        raise ValueError("C6b reference control discrepancy; stop and report")
    return result


def m0_deliverables() -> pd.DataFrame:
    """Describe existing artifacts and approved defaults, not a new feature decision."""
    items = [
        (
            "table_state_v2",
            "טבלה ודגל",
            "טבלה לפי as_of, חוקים ודגל מתמטי; נקודות ספורטיביות, בלי הורדות",
        ),
        ("elo_E0050_prematch", "דירוג Elo", "דירוג תוצאות בלבד לפני כל משחק; ייחוס E0050 שמור"),
        ("form_v1", "כושר", "66 מדדי עבר; ההבקעה והספיגה ב-ewma5/venue נבחרו כייחוס"),
        (
            "season_context_v1",
            "הקשר t−1",
            "תשעה שדות קבועים לעונה; value_avg ברירת מחדל ל-M2 לפי ה־ת14",
        ),
        (
            "prematch_v1",
            "שכבה מאוחדת",
            "51 עמודות מארבעת הבלוקים; כל רכיב קורא את הפלט השמור; אין בו יעד",
        ),
        (
            "E0050",
            "log loss = 1.0204",
            "סרגל EXP-001, אימון החל מ-2006/2007; אינו הציון מעט השונה של EXP-003",
        ),
        ("Poisson", "NLL = 1.4095", "סרגל EXP-002 לשערים עם Elo ו-ewma5/venue; אימון מ-2006/2007"),
        (
            "מעמדי פיצ׳רים",
            "קיים מול מועמד",
            "value_avg והתקפה/הגנה ewma5/venue ברירות מחדל ל-M2; "
            "פערי ה-registry מפורטים ב-S4 D01; pts_venue_ewma5 מועמד לפי S4 D02. "
            "gd, ppm, position ועמודות Elo נשארים 🔬 כקלטים לבדיקה; E0050 הוא סרגל קבוע",
        ),
    ]
    return pd.DataFrame([{"artifact": a, "status": s, WORDS: words} for a, s, words in items])


def planning_decisions() -> pd.DataFrame:
    """Concise reading aid to the fourteen approved VISION §2.4 decisions."""
    words = [
        "Elo מכויל במחברת 03 כדי לספק סרגל ל-M1; שכבת ההמרה סדורה עם שלושה פרמטרים לפי ה־ת11",
        "Elo נשען על תוצאות בלבד; אתחול עולות ורשת היפר־פרמטרים נבדקו בניסוי",
        "דגל אין על מה לשחק מחושב מהטבלה ומהחוקים, בלי Elo ובלי מודל",
        "אם לפחות קבוצה אחת מתה, אותה הנחה חלה על עדכון Elo של שתיהן",
        "M3 מקבל λ בלבד ופרמטרי תיקון; B4 ישיר בודק אובדן מידע בשרשרת",
        "מדווחים גם חמש עונות ולידציה אחרונות; אי־הסכמה עוברת להכרעת עמית",
        "בכל רכיב נלמד נדרש ניסוי משקלי זמן, כולל ללא דעיכה",
        "התקפת הקבוצה מוצלבת עם הגנת היריבה; שתיהן יתעדכנו במסלול M4",
        "רשת ההנחה 0 / 0.25 / 0.5 / 0.75 / 1; ליטוש רק לפי הכלל שנקבע מראש",
        "הדגל בודק גם קווי מקומות 1–4; קו 4 נכלל תמיד כשמרנות לגבי אירופה",
        "פרטי Elo, משחקים באותו יום, snapshots, שלושה פרמטרי המרה ושופט משותף",
        "כושר אינו חוצה קיץ; ערך פתיחה מעונות קודמות וחלונות בתוך העונה; EXP-002 בשני מסלולים",
        "שווי t−1 עבר את השער; לעולה מיקום 0.75, נקודות/הפרש מהעולות הקודמות; אין t−1 ב-2006/2007",
        "עומק אימון מראש בניסויים חדשים; value_avg ברירת מחדל בחריגה מתועדת; "
        "gd/ppm/position מועמדים",
    ]
    return pd.DataFrame(
        [
            {"decision": f"ה־ת{i}", WORDS: text, "source": "docs/VISION.md §2.4"}
            for i, text in enumerate(words, 1)
        ]
    )
