"""Development-only descriptions of previous-season strength; no forecasting features."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

from .core import MEAN_BOOTSTRAP_DRAWS, _bootstrap_count, _selection, _subset, rate_interval
from .match_flow import _mean_ci


def _t1_games(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    """Attach only previous-season strengths, matching the upset diagnostic lookup."""
    games = _subset(tables.match_table, selected)
    current = _subset(tables.team_seasons, selected)
    previous = _subset(tables.team_seasons, list(DEVELOPMENT_SEASONS)).final_position
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.season_id, games[f"{side}_team_id"]])
        prior = pd.MultiIndex.from_arrays([games.season_id - 1, games[f"{side}_team_id"]])
        games[f"{side}_share"] = current.prev_season_value_share.reindex(keys).to_numpy(
            dtype=float, na_value=np.nan
        )
        games[f"{side}_rank"] = previous.reindex(prior).to_numpy(dtype=float, na_value=np.nan)
        games[f"{side}_promoted"] = current.promoted_in.reindex(keys).fillna(False).to_numpy(bool)
    return games


def _strength_sample(tables: AnalysisTables, selected: list[int]) -> pd.DataFrame:
    games = _t1_games(tables, selected)
    columns = [f"{side}_{metric}" for side in ("home", "away") for metric in ("share", "rank")]
    positive = games[columns].gt(0).all(axis=1) & np.isfinite(games[columns]).all(axis=1)
    valid = positive & ~games.home_promoted & ~games.away_promoted & games.season_id.ne(8)
    sample = games.loc[valid].copy()
    sample["lv"] = np.log(sample.home_share / sample.away_share)
    sample["rk"] = sample.away_rank - sample.home_rank
    sample["lrk"] = np.log(sample.away_rank / sample.home_rank)
    return sample


def t1_strength_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = 2000,
    seed: int = 1,
) -> pd.DataFrame:
    """Five H/D/A descriptions on identical held-out-season samples (LOSO).

    LOSO trains on all other development seasons, including later seasons: this is
    retrospective EDA, not chronological model selection. Rank is the prior official
    final position and can include deductions. Promoted teams and season 8 are excluded.
    RPS uses ordered H/D/A cumulative probabilities, divided by K-1=2.
    Paired loss differences resample whole seasons; season match counts are retained,
    so both the point estimate and each bootstrap draw are means per match.
    """
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    sample = _strength_sample(tables, selected).sort_index()
    if sample.season_id.nunique() < 2:
        raise ValueError("LOSO needs at least two eligible development seasons")
    y = np.select(
        [sample.home_goals.gt(sample.away_goals), sample.home_goals.eq(sample.away_goals)],
        [0, 1],
        default=2,
    )
    models = {
        "baseline": [],
        "value": ["lv"],
        "rank_difference": ["rk"],
        "rank_log": ["lrk"],
        "value_rank_log": ["lv", "lrk"],
    }
    losses = pd.DataFrame(index=sample.index)
    rows = []
    for name, features in models.items():
        probabilities = np.empty((len(sample), 3))
        for held_out in sorted(sample.season_id.unique()):
            test = sample.season_id.eq(held_out).to_numpy()
            training = ~test
            if features:
                estimator = LogisticRegression(C=1e6, max_iter=1000)
                estimator.fit(sample.loc[training, features], y[training])
                probabilities[test] = 0
                positions = np.flatnonzero(test)
                probabilities[np.ix_(positions, estimator.classes_.astype(int))] = (
                    estimator.predict_proba(sample.loc[test, features])
                )
            else:
                probabilities[test] = np.bincount(y[training], minlength=3) / training.sum()
        per_game = -np.log(np.maximum(probabilities[np.arange(len(y)), y], np.finfo(float).eps))
        cumulative = probabilities.cumsum(axis=1)[:, :-1]
        target = np.eye(3)[y].cumsum(axis=1)[:, :-1]
        rps = ((cumulative - target) ** 2).sum(axis=1) / 2
        losses[name] = per_game
        rows.append(
            dict(
                model=name,
                n=len(sample),
                n_seasons=sample.season_id.nunique(),
                log_loss=float(per_game.mean()),
                rps=float(rps.mean()),
            )
        )
    losses["season_id"] = sample.season_id
    totals = losses.groupby("season_id").sum()
    counts = sample.groupby("season_id").size().reindex(totals.index).to_numpy()
    draws = np.random.default_rng(seed).integers(len(counts), size=(n_bootstrap, len(counts)))
    comparisons = []
    for first, reference in (
        ("value", "baseline"),
        ("value", "rank_log"),
        ("value_rank_log", "rank_log"),
    ):
        differences = (totals[first] - totals[reference]).to_numpy()
        sampled = differences[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low, high = np.quantile(sampled, [0.025, 0.975])
        comparisons.append(
            dict(
                first=first,
                reference=reference,
                difference=float(differences.sum() / counts.sum()),
                ci_lower=float(low),
                ci_upper=float(high),
                seasons_better=int((differences < 0).sum()),
                n_seasons=len(counts),
            )
        )
    result = pd.DataFrame(rows).set_index("model")
    result.attrs.update(
        n_matches=len(sample),
        n_seasons=len(counts),
        seed=seed,
        n_bootstrap=n_bootstrap,
        correlation_lv_rk=float(sample.lv.corr(sample.rk)),
        comparisons=comparisons,
        seasonal_losses=losses.groupby("season_id").mean().reset_index().to_dict("records"),
        method="LOSO on development seasons; not chronological walk-forward",
        target_order=["home_win", "draw", "away_win"],
    )
    return result


def _rates(games: pd.DataFrame, n_bootstrap: int) -> dict[str, Any]:
    row: dict[str, Any] = dict(n=len(games), n_matches=len(games))
    for label, flag in {
        "home_win_rate": games.home_goals.gt(games.away_goals),
        "draw_rate": games.home_goals.eq(games.away_goals),
        "away_win_rate": games.home_goals.lt(games.away_goals),
        "zero_zero_rate": games.total_goals.eq(0),
    }.items():
        rate, lo, hi = rate_interval(int(flag.sum()), len(games))
        row.update(
            {
                label: rate,
                f"{label}_count": int(flag.sum()),
                f"{label}_ci_lower": lo,
                f"{label}_ci_upper": hi,
            }
        )
    row["goals_per_match"] = games.total_goals.mean()
    row["goals_per_match_ci_lower"], row["goals_per_match_ci_upper"] = _mean_ci(
        games, "total_goals", n_bootstrap
    )
    points = games.assign(
        home_points=3 * games.home_goals.gt(games.away_goals).astype(int)
        + games.home_goals.eq(games.away_goals).astype(int)
    )
    row["home_ppm"] = points.home_points.mean()
    row["home_ppm_ci_lower"], row["home_ppm_ci_upper"] = _mean_ci(
        points, "home_points", n_bootstrap
    )
    return row


def upset_rates(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    n_bootstrap: int = MEAN_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Previous-season value ratios/rank gaps only; unknown strength is never a forced weak side."""
    selected = _selection(seasons)
    _bootstrap_count(n_bootstrap)
    games = _subset(tables.match_table, selected)
    seasons_frame = _subset(tables.team_seasons, selected).reset_index()
    # t-1 lookup may use a development season outside a narrow display selection, never holdouts.
    previous = _subset(tables.team_seasons, [s for s in DEVELOPMENT_SEASONS]).reset_index()
    positions = previous.set_index(["season_id", "team_id"]).final_position
    current = seasons_frame.set_index(["season_id", "team_id"])
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([games.season_id, games[f"{side}_team_id"]])
        prior_keys = pd.MultiIndex.from_arrays([games.season_id - 1, games[f"{side}_team_id"]])
        games[f"{side}_share"] = current.prev_season_value_share.reindex(keys).to_numpy(
            dtype=float, na_value=np.nan
        )
        games[f"{side}_rank"] = positions.reindex(prior_keys).to_numpy(dtype=float, na_value=np.nan)
        games[f"{side}_promoted"] = (
            current.promoted_in.reindex(keys).fillna(False).to_numpy(dtype=bool)
        )
    rows, edges = [], []
    for metric, home, away in (
        ("value_ratio", games.home_share, games.away_share),
        ("rank_gap", -games.home_rank, -games.away_rank),
    ):
        valid = home.notna() & away.notna()
        if metric == "value_ratio":
            valid &= home.gt(0) & away.gt(0)
        equal = valid & home.eq(away)
        bins = pd.Series("unknown_both", index=games.index, dtype="string")
        bins.loc[equal] = "equal_strength"
        known = valid & ~equal
        if metric == "value_ratio" and known.any():
            ratios = np.maximum(
                home.loc[known] / away.loc[known], away.loc[known] / home.loc[known]
            )
            quantiles, breaks = pd.qcut(ratios, 5, labels=False, retbins=True, duplicates="drop")
            bins.loc[known] = quantiles.map(lambda x: f"Q{int(x) + 1}" if pd.notna(x) else "Q1")
            edges = breaks.tolist()
        elif metric == "rank_gap":
            gap = (home - away).abs()
            for label, flag in (
                ("gap_1_3", gap.le(3)),
                ("gap_4_7", gap.between(4, 7)),
                ("gap_8_plus", gap.ge(8)),
            ):
                bins.loc[known & flag] = label
        one_missing = home.isna() ^ away.isna()
        veteran_promoted = one_missing & (
            (home.isna() & games.home_promoted) | (away.isna() & games.away_promoted)
        )
        bins.loc[one_missing & ~veteran_promoted] = "missing_source"
        bins.loc[veteran_promoted] = "veteran_vs_promoted"
        orientation = pd.Series("unknown", index=games.index)
        orientation.loc[known & home.lt(away)] = "home_weaker"
        orientation.loc[known & home.gt(away)] = "away_weaker"
        orientation.loc[veteran_promoted & games.home_promoted] = "home_promoted"
        orientation.loc[veteran_promoted & games.away_promoted] = "away_promoted"
        grouped = games.assign(gap_bin=bins, weaker_side=orientation)
        for (label, side), group in grouped.groupby(["gap_bin", "weaker_side"], observed=True):
            row = dict(metric=metric, gap_bin=label, weaker_side=side, **_rates(group, n_bootstrap))
            for name in ("weak_win", "strong_win", "veteran_win", "promoted_win"):
                row[f"{name}_rate"] = row[f"{name}_rate_ci_lower"] = row[
                    f"{name}_rate_ci_upper"
                ] = float("nan")
            if side in ("home_weaker", "away_weaker", "home_promoted", "away_promoted"):
                home_labeled = side.startswith("home")
                prefix = "promoted" if "promoted" in side else "weak"
                opponent = "veteran" if prefix == "promoted" else "strong"
                for dest, origin in (
                    (prefix, "home" if home_labeled else "away"),
                    (opponent, "away" if home_labeled else "home"),
                ):
                    for suffix in ("rate", "rate_ci_lower", "rate_ci_upper"):
                        row[f"{dest}_win_{suffix}"] = row[f"{origin}_win_{suffix}"]
            rows.append(row)
    frame = pd.DataFrame(rows).set_index(["metric", "gap_bin", "weaker_side"], drop=True)
    frame.attrs.update(
        value_ratio_edges=edges,
        n_matches_per_metric=len(games),
        rank_bins="1–3 / 4–7 / 8+ places; full ties separate",
        unknown_policy="home/draw/away only; veteran/promoted explicit; no imputation",
    )
    return frame
