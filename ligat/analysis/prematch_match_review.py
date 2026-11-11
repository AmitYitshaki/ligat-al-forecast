"""Saved-data descriptive checks for notebook 03; no training or feature writes."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import chi2_contingency  # type: ignore[import-untyped]

from ligat.analysis.prematch_correlations import game_targets
from ligat.analysis.prematch_final_review import FEATURE_NAMES, TREND_FEATURES, WORDS


def _development(frame: pd.DataFrame) -> None:
    if frame.empty or frame.season_id.isna().any() or not frame.season_id.isin(range(8, 26)).all():
        raise ValueError("Match review requires development seasons 8–25")


def match_view(frame: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join sporting targets in memory and keep exactly one home observation per fixture."""
    _development(frame)
    targets = game_targets(frame, rows)
    home = frame.loc[frame.is_home].copy()
    if home.index.get_level_values("game_id").duplicated().any():
        raise ValueError("Exactly one home row per fixture is required")
    opponent_keys = pd.MultiIndex.from_arrays(
        [home.index.get_level_values("game_id"), home.opponent_id]
    )
    away = frame.reindex(opponent_keys)
    if away.is_home.isna().any() or away.is_home.any():
        raise ValueError("Every home row must have its mirrored away row")
    result = home.join(targets)
    result["dead_union"] = home.nothing_to_play.to_numpy() | away.nothing_to_play.to_numpy()
    result["total_goals"] = result.goals_scored + result.goals_conceded
    result["outcome"] = np.select(
        [result.goal_difference_game.gt(0), result.goal_difference_game.eq(0)], [0, 1], 2
    )
    return result, targets


def poisson_noise_limits(games: pd.DataFrame) -> pd.DataFrame:
    """Plug observed means into the independent-Poisson variance decomposition.

    These are heuristic signal/noise calculations, not identified bounds or OOF R².
    Population moments (ddof=0) use the same set of home fixture observations.
    """
    _development(games)
    rows = []
    for target, noise_column, words in (
        ("goal_difference_game", "total_goals", "הפרש שערים: סכום קצבי שני הצדדים הוא רעש Poisson"),
        ("goals_scored", "goals_scored", "שערי הבית: התוחלת משמשת אומדן לשונות ההגרלה"),
        ("goals_conceded", "goals_conceded", "שערי החוץ: אותו חישוב לכל צד בנפרד"),
    ):
        variance = float(games[target].var(ddof=0))
        noise = float(games[noise_column].mean())
        if variance <= 0 or noise < 0:
            raise ValueError("Positive outcome variance and nonnegative counts are required")
        raw = 1 - noise / variance
        rows.append(
            dict(
                target=target,
                games=len(games),
                variance=variance,
                noise=noise,
                raw_signal_fraction=raw,
                signal_fraction=max(0.0, raw),
                heuristic_correlation=float(np.sqrt(max(0.0, raw))),
                **{WORDS: words},
            )
        )
    return pd.DataFrame(rows)


def elo_bins(games: pd.DataFrame, *, groups: int = 10) -> pd.DataFrame:
    """Equal-count Elo groups; no observation is duplicated or removed to force bins."""
    _development(games)
    bins = pd.qcut(games.elo_diff, groups, labels=False, duplicates="drop")
    if bins.nunique() < 2:
        raise ValueError("At least two distinct Elo bins are required")
    rows = []
    for group, part in games.assign(bin=bins).groupby("bin", sort=True):
        rows.append(
            dict(
                bin=int(group) + 1,
                games=len(part),
                elo_low=float(part.elo_diff.min()),
                elo_high=float(part.elo_diff.max()),
                elo_mean=float(part.elo_diff.mean()),
                mean_gd=float(part.goal_difference_game.mean()),
                home_win_percent=float(100 * part.outcome.eq(0).mean()),
                draw_percent=float(100 * part.outcome.eq(1).mean()),
                away_win_percent=float(100 * part.outcome.eq(2).mean()),
                **{WORDS: f"קבוצה {int(group) + 1}: {len(part)} משחקים, מהפער הנמוך לגבוה"},
            )
        )
    return pd.DataFrame(rows)


def grouped_linearity(games: pd.DataFrame) -> pd.DataFrame:
    """Correlation ratio η from twenty bin means, on the original outcome scale."""
    bins = elo_bins(games, groups=20)
    mean = float(games.goal_difference_game.mean())
    total = float(((games.goal_difference_game - mean) ** 2).sum())
    between = float(np.sum(bins.games * (bins.mean_gd - mean) ** 2))
    pearson = float(games.elo_diff.corr(games.goal_difference_game))
    eta = float(np.sqrt(between / total))
    return pd.DataFrame(
        [
            dict(
                pearson=pearson,
                eta=eta,
                groups=len(bins),
                games=len(games),
                **{WORDS: "η של ממוצעי קבוצות שומר את שונות המשחקים; אינו מתאם בין 20 נקודות בלבד"},
            )
        ]
    )


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator else float("nan")


def period_checks(
    games: pd.DataFrame, *, repeats: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Disjoint periods, shared complete cases and paired within-period game bootstrap."""
    _development(games)
    if repeats < 2:
        raise ValueError("At least two bootstrap replicates are required")
    columns = [*TREND_FEATURES, "goal_difference_game"]
    samples = [
        games.loc[games.season_id.between(a, b), columns].dropna().astype(float)
        for a, b in ((9, 20), (21, 25))
    ]
    if any(len(s) < 10 for s in samples):
        raise ValueError("Two nonempty development periods of at least ten games are required")
    periods, overlap = [], []
    for feature in TREND_FEATURES:
        rs = [
            _correlation(s[feature].to_numpy(), s.goal_difference_game.to_numpy()) for s in samples
        ]
        periods.append(
            dict(
                feature=feature,
                pearson_early=rs[0],
                pearson_recent=rs[1],
                change=rs[1] - rs[0],
                games_early=len(samples[0]),
                games_recent=len(samples[1]),
                **{WORDS: FEATURE_NAMES[feature] + ": אותן שורות בשתי תקופות נפרדות"},
            )
        )
    for label, s in zip(("2007/2008–2018/2019", "2019/2020–2023/2024"), samples, strict=True):
        corr = s[["value_avg_log_ratio", "elo_diff", "goal_difference_game"]].corr()
        a, b, c = float(corr.iloc[0, 2]), float(corr.iloc[1, 2]), float(corr.iloc[0, 1])
        denominator = np.sqrt((1 - c * c) * (1 - b * b))
        partial = float((a - c * b) / denominator) if denominator > 0 else np.nan
        overlap.append(
            dict(
                period=label,
                games=len(s),
                value_elo=c,
                partial_value=partial,
                **{WORDS: label + ": קשר השווי לשערים אחרי הסרת החלק הליניארי המשותף ל־Elo"},
            )
        )
    rng = np.random.default_rng(seed)
    arrays = [
        s[["value_avg_log_ratio", "elo_diff", "goal_difference_game"]].to_numpy() for s in samples
    ]
    changes = []
    for _ in range(repeats):
        values = []
        for data in arrays:
            draw = data[rng.integers(len(data), size=len(data))]
            values.append(
                _correlation(draw[:, 0], draw[:, 2]) - _correlation(draw[:, 1], draw[:, 2])
            )
        changes.append(values[1] - values[0])
    if not np.isfinite(changes).all():
        raise ValueError("Undefined bootstrap correlations; do not replace them with zero")
    index = pd.DataFrame(periods).set_index("feature")
    gap = float(index.at["value_avg_log_ratio", "change"] - index.at["elo_diff", "change"])
    se = float(np.std(changes, ddof=1))
    bootstrap = pd.DataFrame(
        [
            dict(
                change_difference=gap,
                se=se,
                descriptive_z=gap / se,
                repeats=repeats,
                seed=seed,
                **{WORDS: "שינוי מתאם השווי פחות שינוי פער Elo; דגימות מזווגות בתוך כל תקופה"},
            )
        ]
    )
    return pd.DataFrame(periods), pd.DataFrame(overlap), bootstrap


def venue_checks(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Separate within-venue correlations; raw contrasts remain mirrored controls."""
    _development(frame)
    if not frame.index.equals(targets.index):
        raise ValueError("Targets must align exactly with feature rows")
    rows = []
    for feature, target in (
        ("gf_venue_ewma5", "goals_scored"),
        ("opp_ga_venue_ewma5", "goals_scored"),
        ("ga_venue_ewma5", "goals_conceded"),
        ("elo_diff", "goal_difference_game"),
    ):
        row: dict[str, Any] = dict(
            feature=feature,
            target=target,
            **{WORDS: FEATURE_NAMES[feature] + " מול " + target + "; בית וחוץ בנפרד"},
        )
        for is_home, side in ((True, "home"), (False, "away")):
            mask = frame.is_home.eq(is_home)
            row["pearson_" + side] = float(frame.loc[mask, feature].corr(targets.loc[mask, target]))
            row["games_" + side] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def promotion_chi_example(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed, independence expectation and per-cell χ² contributions for home promotion."""
    _development(games)
    observed = (
        pd.crosstab(games.promoted_in, games.outcome)
        .reindex(index=[False, True], columns=[0, 1, 2], fill_value=0)
        .to_numpy()
    )
    if not (observed.sum(axis=0) > 0).all() or not (observed.sum(axis=1) > 0).all():
        raise ValueError("Both promotion groups and all three outcomes must be represented")
    statistic, p_value, dof, expected = chi2_contingency(observed, correction=False)
    contributions = (observed - expected) ** 2 / expected
    rows = []
    for i, group in enumerate(("מארחת ותיקה", "מארחת עולה")):
        for j, outcome in enumerate(("ניצחון בית", "תיקו", "ניצחון חוץ")):
            rows.append(
                dict(
                    group=group,
                    outcome=outcome,
                    observed=int(observed[i, j]),
                    expected=float(expected[i, j]),
                    contribution=float(contributions[i, j]),
                    **{
                        WORDS: f"{group}, {outcome}: נצפה {observed[i, j]}, "
                        f"צפוי {expected[i, j]:.1f}"
                    },
                )
            )
    summary = pd.DataFrame(
        [
            dict(
                chi2=float(statistic),
                dof=int(dof),
                p_value=float(p_value),
                cramers_v=float(np.sqrt(statistic / observed.sum())),
                games=len(games),
                expected_promoted_home=float(expected[1, 0]),
                **{WORDS: "קשר בין עליית המארחת לתוצאת המשחק; אינו מנטרל את Elo"},
            )
        ]
    )
    return pd.DataFrame(rows), summary


def matched_dead_games(
    games: pd.DataFrame, *, minimum_controls: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize alive games to supported dead-game stage/remaining-game strata.

    Unsupported dead games remain in the descriptive full count, and are excluded
    from both sides of the matched contrast. SE uses independent-game Bernoulli
    variance with fixed empirical weights, not a causal or season-cluster estimate.
    A control-count restriction removes entire strata from both groups before
    recomputing all weights, rates and uncertainty on the remaining support.
    """
    _development(games)
    if isinstance(minimum_controls, bool) or not isinstance(minimum_controls, int):
        raise ValueError("minimum_controls must be a positive integer")
    if minimum_controls < 1:
        raise ValueError("minimum_controls must be a positive integer")
    work = games.assign(draw=games.outcome.eq(1))
    keys = ["stage", "rounds_to_stage_end"]
    dead, alive = work.loc[work.dead_union], work.loc[~work.dead_union]
    counts = dead.groupby(keys, observed=True).agg(
        dead_n=("draw", "size"), dead_draws=("draw", "sum")
    )
    controls = alive.groupby(keys, observed=True).agg(
        alive_n=("draw", "size"), alive_draws=("draw", "sum")
    )
    supported = counts.join(controls, how="inner").copy()
    supported = supported.loc[supported.alive_n.ge(minimum_controls)].copy()
    if supported.empty:
        raise ValueError("No common support for the dead-game comparison")
    n = int(supported.dead_n.sum())
    supported["weight"] = supported.dead_n / n
    supported["alive_rate"] = supported.alive_draws / supported.alive_n
    matched_rate = float(np.sum(supported.weight * supported.alive_rate))
    dead_rate = float(supported.dead_draws.sum() / n)
    control_variance = float(
        np.sum(
            supported.weight**2
            * supported.alive_rate
            * (1 - supported.alive_rate)
            / supported.alive_n
        )
    )
    se = float(np.sqrt(dead_rate * (1 - dead_rate) / n + control_variance))
    late = alive.loc[alive.rounds_to_stage_end.le(2)]
    late_reference = alive.loc[alive.rounds_to_stage_end.le(3)]
    descriptions = [
        (
            "כל המשחקים המתים",
            len(dead),
            int(dead.draw.sum()),
            float(dead.draw.mean()),
            "כולל משחקים שאין להם קבוצת השוואה",
        ),
        (
            "משחקים מתים עם התאמה",
            n,
            int(supported.dead_draws.sum()),
            dead_rate,
            "מדגם התמיכה המשותפת בלבד",
        ),
        (
            "לא מתים בשלושת האחרונים בשלב",
            len(late),
            int(late.draw.sum()),
            float(late.draw.mean()),
            "השוואה גסה שאינה מתקנת את הרכב השלבים",
        ),
        (
            "לא מתים מותאמים",
            int(supported.alive_n.sum()),
            np.nan,
            matched_rate,
            "שקלול לפי פיזור המשחקים המתים בתמיכה המשותפת",
        ),
    ]
    descriptions.append(
        (
            "לא מתים, מונה 0–3 (ארבעת האחרונים)",
            len(late_reference),
            int(late_reference.draw.sum()),
            float(late_reference.draw.mean()),
            "להבהרת הייחוס 689: מונה 0–3 מייצג ארבעה משחקים, ולא שלושה",
        )
    )
    summary = pd.DataFrame(
        [
            dict(group=label, games=size, draws=draws, draw_percent=100 * rate, **{WORDS: words})
            for label, size, draws, rate, words in descriptions
        ]
    )
    comparison = pd.DataFrame(
        [
            dict(
                matched_dead=n,
                all_dead=len(dead),
                excluded_dead=len(dead) - n,
                difference_pp=100 * (matched_rate - dead_rate),
                se_pp=100 * se,
                descriptive_z=(matched_rate - dead_rate) / se,
                minimum_controls=minimum_controls,
                **{WORDS: "לא מתים פחות מתים, באותו שלב ובאותו מרחק מהסוף; קשר תיאורי"},
            )
        ]
    )
    supported = supported.reset_index()
    supported[WORDS] = [
        f"{r.stage}, עוד {r.rounds_to_stage_end} משחקים: {r.dead_n} מתים מול {r.alive_n} לא מתים"
        for r in supported.itertuples()
    ]
    return summary, comparison, supported


@dataclass(frozen=True)
class MatchReview:
    """Small descriptive tables, retained in memory; no feature artifact is produced."""

    tables: dict[str, pd.DataFrame]


def analyze_match_review(frame: pd.DataFrame, rows: pd.DataFrame) -> MatchReview:
    games, targets = match_view(frame, rows)
    periods, overlap, bootstrap = period_checks(games)
    chi_cells, chi_summary = promotion_chi_example(games)
    dead, dead_comparison, dead_strata = matched_dead_games(games)
    sensitive, sensitive_comparison, sensitive_strata = matched_dead_games(
        games, minimum_controls=20
    )
    sensitivity = []
    for summary, comparison, words in (
        (dead, dead_comparison, "כל שכבה עם לפחות משחק השוואה אחד"),
        (sensitive, sensitive_comparison, "בדיקת רגישות: לפחות 20 משחקי השוואה בכל שכבה"),
    ):
        row = comparison.iloc[0].to_dict()
        row.update(
            dead_draw_percent=float(summary.iloc[1].draw_percent),
            alive_draw_percent=float(summary.iloc[3].draw_percent),
            **{WORDS: words},
        )
        sensitivity.append(row)
    excluded = dead_strata.loc[dead_strata.alive_n.lt(20)].copy()
    return MatchReview(
        dict(
            ceilings=poisson_noise_limits(games),
            deciles=elo_bins(games),
            linearity=grouped_linearity(games),
            periods=periods,
            overlap=overlap,
            bootstrap=bootstrap,
            venue=venue_checks(frame, targets),
            chi_cells=chi_cells,
            chi_summary=chi_summary,
            dead=dead,
            dead_comparison=dead_comparison,
            dead_strata=dead_strata,
            dead_sensitivity=pd.DataFrame(sensitivity),
            dead_excluded_strata=excluded,
            dead_sensitive_strata=sensitive_strata,
        )
    )
