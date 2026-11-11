"""Descriptive prematch associations only: no feature selection, fitting or file writes."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import chi2_contingency  # type: ignore[import-untyped]

from ligat.features.pipeline import (
    DEVELOPMENT_SEASONS,
    ELO_COLUMNS,
    SELECTED_CONTEXT_COLUMNS,
    SELECTED_FORM_COLUMNS,
    TABLE_COLUMNS,
)
from ligat.features.registry import PREMATCH_CONTRAST_COLUMNS, PREMATCH_OPPONENT_COLUMNS

EXCLUDED_COLUMNS = frozenset({"season_id", "date", "as_of", "round", "stage", "opponent_id"})
TARGET_COLUMNS = ("goals_scored", "goals_conceded", "goal_difference_game", "points_game")
CONTRAST_ELO_COLUMNS = (*PREMATCH_CONTRAST_COLUMNS, "elo_diff")
PAIR_FLAG_LIMIT = 0.8
PEARSON_LEAKAGE_LIMIT = 0.6
CRAMERS_LEAKAGE_LIMIT = 0.4
QUANTILE_GROUPS = 5
OUTCOMES = ("H", "D", "A")
FEATURE_FAMILIES = {
    **dict.fromkeys(TABLE_COLUMNS, "table"),
    **dict.fromkeys(ELO_COLUMNS, "elo"),
    **dict.fromkeys(SELECTED_FORM_COLUMNS, "form"),
    **dict.fromkeys(PREMATCH_OPPONENT_COLUMNS, "opponent_form"),
    **dict.fromkeys(SELECTED_CONTEXT_COLUMNS, "context"),
    **dict.fromkeys(PREMATCH_CONTRAST_COLUMNS, "contrast"),
}


class LeakageDetectedError(RuntimeError):
    """Stop before producing a report when a predeclared association limit is crossed."""

    def __init__(self, feature: str, target: str, statistic: str) -> None:
        self.feature = feature
        self.target = target
        self.statistic = statistic
        super().__init__(f"Potential leakage: {feature} / {target} ({statistic}); stop and report")


def _validate_frame(frame: pd.DataFrame) -> None:
    if (
        frame.empty
        or frame.index.names != ["game_id", "team_id"]
        or not frame.index.is_unique
        or frame.columns.duplicated().any()
        or frame.index.to_frame().isna().any().any()
    ):
        raise ValueError("Prematch analysis requires unique, nonmissing (game_id, team_id) keys")
    if (
        "season_id" not in frame
        or frame.season_id.isna().any()
        or not frame.season_id.isin(DEVELOPMENT_SEASONS).all()
    ):
        raise ValueError("Prematch analysis accepts development seasons 8–25 only")
    if set(TARGET_COLUMNS) & set(frame):
        raise ValueError("Observed targets must remain separate from the prematch feature table")


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_frame(frame)
    columns = [
        c
        for c in frame
        if c not in EXCLUDED_COLUMNS
        and (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))
    ]
    unknown = set(columns) - FEATURE_FAMILIES.keys()
    if unknown:
        raise ValueError(f"Unknown numeric feature families: {sorted(unknown)}")
    numeric = frame.loc[:, columns].astype(float)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Prematch numeric values must be finite or missing")
    return numeric


def feature_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """List numeric/bool features, treating bool as 0/1 and recording global constants."""
    numeric = _numeric(frame)
    return pd.DataFrame(
        [
            {
                "feature": c,
                "family": FEATURE_FAMILIES[c],
                "n": int(numeric[c].notna().sum()),
                "unique_values": int(numeric[c].nunique()),
                "constant": numeric[c].nunique() < 2,
            }
            for c in numeric
        ],
        columns=["feature", "family", "n", "unique_values", "constant"],
    )


def game_targets(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Join projected sporting game results in memory; points are always 3/1/0, never halved."""
    _validate_frame(frame)
    required = {"game_id", "team_id", "season_id", "goals_scored", "goals_conceded"}
    if (
        not required <= set(rows)
        or rows.season_id.isna().any()
        or not rows.season_id.isin(DEVELOPMENT_SEASONS).all()
    ):
        raise ValueError("Target rows require projected goals and development seasons 8–25")
    keys = rows.loc[:, ["game_id", "team_id"]]
    if keys.isna().any().any() or not np.equal(keys.to_numpy(float) % 1, 0).all():
        raise ValueError("Target keys must be nonmissing integer IDs")
    projected = rows.set_index(["game_id", "team_id"]).copy()
    projected.index = pd.MultiIndex.from_frame(keys.astype("int64"))
    if not projected.index.is_unique or not projected.sort_index().index.equals(
        frame.sort_index().index
    ):
        raise ValueError("Target rows must have exactly the same unique team-game keys")
    projected = projected.reindex(frame.index)
    if not np.array_equal(projected.season_id.to_numpy(int), frame.season_id.to_numpy(int)):
        raise ValueError("Target season IDs disagree with prematch")
    targets = projected.loc[:, ["goals_scored", "goals_conceded"]].copy()
    goals = targets.to_numpy(float)
    if not np.isfinite(goals).all() or (goals < 0).any() or (goals % 1 != 0).any():
        raise ValueError("Observed goals must be complete nonnegative integers")
    targets = targets.astype("int64")
    targets["goal_difference_game"] = targets.goals_scored - targets.goals_conceded
    targets["points_game"] = np.select(
        [targets.goal_difference_game.gt(0), targets.goal_difference_game.eq(0)], [3, 1], 0
    )
    _validated_targets(frame, targets)
    return targets


def _validated_targets(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    _validate_frame(frame)
    if (
        tuple(targets.columns) != TARGET_COLUMNS
        or not targets.index.is_unique
        or not targets.sort_index().index.equals(frame.sort_index().index)
    ):
        raise ValueError("Expected four separate targets with exactly the prematch keys")
    result = targets.reindex(frame.index)
    values = result.to_numpy(float)
    if not np.isfinite(values).all() or (values % 1 != 0).any() or (values[:, :2] < 0).any():
        raise ValueError("Target values must be finite integers with nonnegative goals")
    grouped = result.groupby(level="game_id")
    if not grouped.size().eq(2).all():
        raise ValueError("Target integrity requires exactly two team rows per game")
    other_conceded = grouped.goals_conceded.transform("sum") - result.goals_conceded
    difference = result.goals_scored - result.goals_conceded
    points = np.select([difference.gt(0), difference.eq(0)], [3, 1], 0)
    if (
        not result.goals_scored.eq(other_conceded).all()
        or not result.goal_difference_game.eq(difference).all()
        or not result.points_game.eq(points).all()
        or not grouped.goal_difference_game.sum().eq(0).all()
        or not grouped.points_game.sum().isin([2, 3]).all()
    ):
        raise ValueError("Mirrored goals, antisymmetric goal difference or 3/1/0 points disagree")
    return result.astype(float)


def _kept_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric(frame)
    return numeric.loc[:, numeric.nunique().ge(2)]


def _season_correlations(frame: pd.DataFrame, numeric: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(s): numeric.loc[frame.season_id.eq(s)].corr(method="pearson", min_periods=2)
        for s in sorted(frame.season_id.unique())
    }


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def feature_pair_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise-complete Pearson/Spearman, recent Pearson and season range; no p-values."""
    numeric = _kept_numeric(frame)
    pearson = numeric.corr(method="pearson", min_periods=2)
    spearman = numeric.corr(method="spearman", min_periods=2)
    recent = numeric.loc[frame.season_id.ge(21)].corr(method="pearson", min_periods=2)
    seasons = _season_correlations(frame, numeric)
    observed = numeric.notna().astype("int64")
    counts = observed.T @ observed
    records = []
    for a, b in combinations(numeric.columns, 2):
        r = float(pearson.loc[a, b])
        seasonal = _finite([float(m.loc[a, b]) for m in seasons.values()])
        records.append(
            {
                "feature_a": a,
                "feature_b": b,
                "family_a": FEATURE_FAMILIES[a],
                "family_b": FEATURE_FAMILIES[b],
                "same_family": FEATURE_FAMILIES[a] == FEATURE_FAMILIES[b],
                "n": int(counts.loc[a, b]),
                "pearson": r,
                "spearman": float(spearman.loc[a, b]),
                "pearson_last5": float(recent.loc[a, b]),
                "pearson_min_season": min(seasonal, default=float("nan")),
                "pearson_max_season": max(seasonal, default=float("nan")),
                "flagged": bool(abs(r) > PAIR_FLAG_LIMIT),
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "feature_a",
            "feature_b",
            "family_a",
            "family_b",
            "same_family",
            "n",
            "pearson",
            "spearman",
            "pearson_last5",
            "pearson_min_season",
            "pearson_max_season",
            "flagged",
        ],
    )


def feature_target_correlations(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Describe each feature against game results; stop on global |Pearson| > 0.6."""
    numeric = _kept_numeric(frame)
    targets = _validated_targets(frame, targets)
    combined = numeric.join(targets, validate="one_to_one")
    pearson = combined.corr(method="pearson", min_periods=2)
    spearman = combined.corr(method="spearman", min_periods=2)
    recent = combined.loc[frame.season_id.ge(21)].corr(method="pearson", min_periods=2)
    seasons = _season_correlations(frame, combined)
    records = []
    for feature in numeric:
        for target in TARGET_COLUMNS:
            r = float(pearson.loc[feature, target])
            if abs(r) > PEARSON_LEAKAGE_LIMIT:
                raise LeakageDetectedError(feature, target, "pearson")
            seasonal = _finite([float(m.loc[feature, target]) for m in seasons.values()])
            records.append(
                {
                    "feature": feature,
                    "family": FEATURE_FAMILIES[feature],
                    "target": target,
                    "n": int(numeric[feature].notna().sum()),
                    "pearson": r,
                    "spearman": float(spearman.loc[feature, target]),
                    "pearson_last5": float(recent.loc[feature, target]),
                    "median_season_pearson": float(np.median(seasonal))
                    if seasonal
                    else float("nan"),
                    "seasons_same_sign": sum(np.sign(v) == np.sign(r) for v in seasonal),
                    "seasons_counted": len(seasonal),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "feature",
            "family",
            "target",
            "n",
            "pearson",
            "spearman",
            "pearson_last5",
            "median_season_pearson",
            "seasons_same_sign",
            "seasons_counted",
        ],
    )


def contrast_elo_correlations(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The four prior-strength contrasts and Elo, in two periods and two correlations."""
    numeric = _numeric(frame).loc[:, list(CONTRAST_ELO_COLUMNS)]
    return {
        f"{method}_{period}": numeric.loc[mask].corr(method=method, min_periods=2)
        for period, mask in (("all", frame.season_id.notna()), ("last5", frame.season_id.ge(21)))
        for method in ("pearson", "spearman")
    }


def correlation_sanity_checks(frame: pd.DataFrame) -> dict[str, float]:
    """Check Elo orientation and equal Pearson matrices on home/away contrast rows."""
    numeric = _numeric(frame)
    r_own = float(numeric.elo_diff.corr(numeric.elo_pre))
    r_opponent = float(numeric.elo_diff.corr(numeric.elo_opponent_pre))
    columns = list(CONTRAST_ELO_COLUMNS)
    home = numeric.loc[frame.is_home, columns].corr().to_numpy()
    away = numeric.loc[~frame.is_home, columns].corr().to_numpy()
    if (
        not r_own > 0
        or not r_opponent < 0
        or not np.allclose(home, away, rtol=0, atol=1e-12, equal_nan=True)
    ):
        raise ValueError("Elo orientation or home/away contrast Pearson sanity check failed")
    difference = np.abs(home - away)
    return {
        "elo_diff_vs_elo_pre": r_own,
        "elo_diff_vs_elo_opponent_pre": r_opponent,
        "max_home_away_difference": float(np.nanmax(difference)),
    }


@dataclass(frozen=True)
class OutcomeAssociation:
    """Aggregate chi-square results and category tables; no per-game targets are retained."""

    statistics: pd.DataFrame
    percentages: pd.DataFrame
    contingencies: dict[str, pd.DataFrame]
    bin_edges: dict[str, tuple[float, ...]]
    home_rows: int


def _home_inputs(
    frame: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    targets = _validated_targets(frame, targets)
    if (
        not pd.api.types.is_bool_dtype(frame.is_home)
        or frame.is_home.isna().any()
        or not frame.groupby(level="game_id").is_home.sum().eq(1).all()
    ):
        raise ValueError("Chi-square requires exactly one home row per game")
    home = frame.loc[frame.is_home].copy()
    keys = pd.MultiIndex.from_arrays(
        [home.index.get_level_values("game_id"), home.opponent_id], names=frame.index.names
    )
    if (frame.index.get_indexer(keys) < 0).any():
        raise ValueError("Away features require the opponent row in the same game")
    away = frame.reindex(keys).copy()
    if away.is_home.any():
        raise ValueError("Away features must come from the away team")
    away.index = home.index
    for source in ("promoted_in", "nothing_to_play", "stage"):
        home[f"away_{source}"] = away[source]
    difference = targets.loc[home.index, "goal_difference_game"]
    outcomes = pd.Series(
        np.select([difference.gt(0), difference.eq(0)], ["H", "D"], "A"), index=home.index
    )
    return home, outcomes, home.season_id.ge(21)


def _contingency(categories: pd.Series, outcomes: pd.Series) -> pd.DataFrame:
    counts = pd.crosstab(categories, outcomes, dropna=False).reindex(
        columns=list(OUTCOMES), fill_value=0
    )
    return counts.loc[counts.sum(axis=1).gt(0)].astype("int64")


def _chi_statistics(counts: pd.DataFrame) -> dict[str, float]:
    observed = counts.loc[:, counts.sum().gt(0)]
    if min(observed.shape) < 2:
        return {}
    chi2, p_value, dof, expected = chi2_contingency(observed.to_numpy(), correction=False)
    return {
        "chi2": float(chi2),
        "dof": float(dof),
        "p_value": float(p_value),
        "cramers_v": float(np.sqrt(chi2 / (observed.to_numpy().sum() * (min(observed.shape) - 1)))),
        "min_expected": float(expected.min()),
    }


def feature_outcome_chi2(frame: pd.DataFrame, targets: pd.DataFrame) -> OutcomeAssociation:
    """Home-only association to H/D/A; fixed full-sample quintiles and a V > 0.4 stop."""
    numeric = _numeric(frame)
    home, outcomes, recent = _home_inputs(frame, targets)
    names = [*numeric.columns, "stage", "away_promoted_in", "away_nothing_to_play", "away_stage"]
    records = []
    percentages = []
    contingencies = {}
    edges = {}
    for name in names:
        source = name.removeprefix("away_")
        family = FEATURE_FAMILIES[source]
        values = home[name]
        eligible = values.notna()
        n = int(eligible.sum())
        categorical = pd.api.types.is_bool_dtype(values) or isinstance(
            values.dtype, pd.CategoricalDtype
        )
        reason = ""
        if values.nunique() < 2:
            categories = values
            reason = "constant in home rows" if n else "no nonmissing home values"
        elif categorical:
            categories = values
        else:
            categories, boundaries = pd.qcut(
                values, q=QUANTILE_GROUPS, duplicates="drop", retbins=True
            )
            edges[name] = tuple(float(v) for v in boundaries)
            if categories.nunique() < 2:
                reason = "fewer than 2 categories after qcut"
        counts = _contingency(categories.loc[eligible], outcomes.loc[eligible])
        if int(counts.to_numpy().sum()) != n:
            raise ValueError(f"Chi-square contingency count mismatch: {name}")
        contingencies[name] = counts
        statistics = _chi_statistics(counts) if not reason else {}
        if not statistics and not reason:
            reason = "fewer than 2 observed outcome categories"
        if statistics.get("cramers_v", 0) > CRAMERS_LEAKAGE_LIMIT:
            raise LeakageDetectedError(name, "H/D/A", "cramers_v")
        recent_counts = _contingency(
            categories.loc[eligible & recent], outcomes.loc[eligible & recent]
        )
        # Subsetting precomputed categories keeps the original quintile boundaries.
        recent_stats = _chi_statistics(recent_counts) if not reason else {}
        records.append(
            {
                "feature": name,
                "family": family,
                "n": n,
                "categories": len(counts),
                **{
                    key: statistics.get(key, float("nan"))
                    for key in ("chi2", "dof", "p_value", "cramers_v", "min_expected")
                },
                "expected_below_5": statistics["min_expected"] < 5 if statistics else pd.NA,
                "cramers_v_last5": recent_stats.get("cramers_v", float("nan")),
                "excluded_reason": reason,
            }
        )
        baseline = counts.sum()
        for category, row in counts.iterrows():
            total = int(row.sum())
            percentages.append(
                {
                    "feature": name,
                    "family": family,
                    "category": str(category),
                    "n": total,
                    **{
                        f"{outcome}_percent": float(row[outcome] / total * 100)
                        for outcome in OUTCOMES
                    },
                    "baseline_n": n,
                    **{
                        f"baseline_{outcome}_percent": float(baseline[outcome] / n * 100)
                        for outcome in OUTCOMES
                    },
                }
            )
    return OutcomeAssociation(
        pd.DataFrame(records), pd.DataFrame(percentages), contingencies, edges, len(home)
    )
