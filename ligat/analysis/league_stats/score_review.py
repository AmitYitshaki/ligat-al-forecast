"""Retrospective score diagnostics and calibrated Poisson null; development data only.

These random season halves diagnose assumptions, never forecast quality or prematch
features. All calculation is pure; the optional cache lives in a separate I/O module.
"""

from collections.abc import Callable, Iterable
from dataclasses import replace

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import (  # type: ignore[import-untyped]
    chi2_contingency,
    linregress,
    poisson,
    random_table,
    skellam,
)

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

from .core import _selection, _subset
from .match_flow import observed_vs_poisson
from .poisson_check import (
    DIAGNOSTIC_SEED,
    FIT_ITERATIONS,
    SCORELINES,
    _fit_strengths,
    poisson_crossfit_predictions,
)

CATEGORIES = ("1:1", "1:0", "0:0", "0:1", "2:0", "2:1", "1:2", "0:2", "2:2", "other")
NULL_SEED = 20261029
NULL_REPLICATES = 200
ALGORITHM_VERSION = "score-review-1"


def poisson_baseline_comparison(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """One observed table against global and unchanged half-season Poisson baselines."""
    from .poisson_check import poisson_crossfit_comparison

    selected = _selection(seasons)
    global_rates = observed_vs_poisson(tables, selected)
    fitted_rates = poisson_crossfit_comparison(tables, selected)
    result = global_rates[["observed"]].copy()
    result["global_expected"] = global_rates.expected
    result["global_ratio"] = global_rates.ratio
    result["per_game_expected"] = fitted_rates.expected
    result["per_game_ratio"] = fitted_rates.ratio
    return result


def score_coverage(
    tables: AnalysisTables, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.DataFrame:
    """Coverage of finite exact-score grids, before pooling the tail."""
    games = _subset(tables.match_table, _selection(seasons))
    rows = []
    for maximum in range(3, 8):
        outside = int((games.home_goals.gt(maximum) | games.away_goals.gt(maximum)).sum())
        rows.append(
            dict(maximum=maximum, n=len(games), outside=outside, coverage=1 - outside / len(games))
        )
    result = pd.DataFrame(rows).set_index("maximum")
    outside = games.loc[games.home_goals.ge(5) | games.away_goals.ge(5)]
    result.attrs.update(
        maximum=int(games[["home_goals", "away_goals"]].max().max()),
        outside_scores=outside.groupby(["home_goals", "away_goals"]).size().to_dict(),
        strong_tail_5=float(poisson.sf(4, 3)),
        strong_tail_6=float(poisson.sf(5, 3)),
    )
    return result


def score_stability(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    permutations: int = 5000,
    seed: int = NULL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season/category deviations, one global test, and nine descriptive time trends.

    Conditional random contingency tables are exactly label permutations preserving
    season sizes and category totals. The plus-one Monte Carlo p avoids zero p-values.
    The nine regression p-values are exploratory, without a confirmatory interpretation.
    """
    games = _subset(tables.match_table, _selection(seasons)).copy()
    if permutations < 1:
        raise ValueError("permutations must be positive")
    labels = (
        games.home_goals.astype(int).astype(str) + ":" + games.away_goals.astype(int).astype(str)
    )
    games["category"] = labels.where(labels.isin(CATEGORIES[:-1]), "other")
    counts = pd.crosstab(games.season_id, games.category).reindex(columns=CATEGORIES, fill_value=0)
    if len(counts) < 2 or (counts.sum() == 0).any():
        raise ValueError("Stability needs at least two seasons and nonempty categories")
    statistic, p, dof, expected = chi2_contingency(counts, correction=False)
    sampled = random_table.rvs(
        counts.sum(axis=1).to_numpy(),
        counts.sum(axis=0).to_numpy(),
        size=permutations,
        random_state=np.random.default_rng(seed),
    )
    null = (((sampled - expected) ** 2) / expected).sum(axis=(1, 2))
    rates = counts.div(counts.sum(axis=1), axis=0)
    deviations = rates.subtract(counts.sum() / counts.to_numpy().sum(), axis=1)
    deviations.attrs.update(
        counts=counts.to_numpy().tolist(),
        chi2=float(statistic),
        dof=int(dof),
        p_asymptotic=float(p),
        p_permutation=float((1 + np.count_nonzero(null >= statistic)) / (permutations + 1)),
        permutations=permutations,
        seed=seed,
        seasonal_standard_error=float(np.sqrt(0.12 * 0.88 / 240)),
    )
    trends = []
    for label in CATEGORIES[:-1]:
        fit = linregress(counts.index.to_numpy(float), rates[label].to_numpy())
        trends.append(dict(scoreline=label, slope_pp=float(fit.slope * 100), p=float(fit.pvalue)))
    return deviations, pd.DataFrame(trends).set_index("scoreline")


def _validated_games(frame: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    selected = _selection(seasons)
    games = _subset(frame, selected).sort_index().copy()
    if games.empty:
        raise ValueError("Nonempty played development matches required")
    return games


def poisson_truth(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    iterations: int = FIT_ITERATIONS,
) -> pd.DataFrame:
    """Full-season strengths define the plug-in null, not a forecasting prediction."""
    games = _validated_games(tables.match_table, seasons)
    if iterations < 1 or games[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Positive iterations and complete scores required")
    # Deliberately discard observed goals from the simulation input contract.
    output = games[["season_id", "home_team_id", "away_team_id"]].copy()
    output["lambda_home"], output["lambda_away"] = np.nan, np.nan
    for _, season in games.groupby("season_id", observed=True):
        h, a, attack, defence = _fit_strengths(season, iterations)
        for side, other, mean in (("home", "away", h), ("away", "home", a)):
            output.loc[season.index, f"lambda_{side}"] = (
                mean
                * attack.reindex(season[f"{side}_team_id"]).to_numpy()
                * defence.reindex(season[f"{other}_team_id"]).to_numpy()
            )
    return output


def simulate_poisson_goals(
    truth: pd.DataFrame,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    seed: int = NULL_SEED,
) -> pd.DataFrame:
    """Draw ONLY from supplied lambdas; real goal columns, if present, are never read."""
    frame = _validated_games(truth, seasons)
    rates = frame[["lambda_home", "lambda_away"]].to_numpy(float)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Truth lambdas must be finite and nonnegative")
    result = frame[["season_id", "home_team_id", "away_team_id"]].copy()
    result[["home_goals", "away_goals"]] = np.random.default_rng(seed).poisson(rates)
    return result


def poisson_diagnostic_metrics(
    predictions: pd.DataFrame, seasons: Iterable[int] = DEVELOPMENT_SEASONS
) -> pd.Series:
    """Dispersion, standardized residual correlation, score ratios, balance thirds.

    Thirds contain equal match counts (stable index breaks equal-strength ties).
    Draws use the full Skellam probability, not equality of two pooled 5+ buckets.
    Zero lambdas cannot define standardized residuals and are rejected explicitly.
    """
    games = _validated_games(predictions, seasons)
    h, a = games.lambda_home.to_numpy(float), games.lambda_away.to_numpy(float)
    if not np.isfinite(np.r_[h, a]).all() or (np.r_[h, a] <= 0).any():
        raise ValueError("Diagnostics require finite strictly positive lambdas")
    yh, ya = games.home_goals.to_numpy(float), games.away_goals.to_numpy(float)
    rh, ra = (yh - h) / np.sqrt(h), (ya - a) / np.sqrt(a)
    values = {
        "dispersion_home": float(np.sum((yh - h) ** 2) / h.sum()),
        "dispersion_away": float(np.sum((ya - a) ** 2) / a.sum()),
        "residual_correlation": float(np.corrcoef(rh, ra)[0, 1]),
    }
    for label, hg, ag in SCORELINES:
        values[label] = float(
            np.mean((yh == hg) & (ya == ag)) / np.mean(poisson.pmf(hg, h) * poisson.pmf(ag, a))
        )
    draw = skellam.pmf(0, h, a)
    flag = yh == ya
    values["all_draws"] = float(flag.sum() / draw.sum())
    order = np.argsort(np.abs(h - a), kind="stable")
    for label, positions in zip(
        ("balanced", "middle", "unbalanced"), np.array_split(order, 3), strict=True
    ):
        values[f"draw_{label}"] = float(flag[positions].sum() / draw[positions].sum())
    return pd.Series(values, name="observed")


def summarize_poisson_null(observed: pd.Series, simulations: pd.DataFrame) -> pd.DataFrame:
    """Two-sided deviations about the simulation mean, plus-one empirical MC p.

    The 95% quantiles describe the null pipeline, not a CI on observed effects.
    """
    mean, sd = simulations.mean(), simulations.std(ddof=1)
    deviation = (observed - mean).abs()
    p = (1 + simulations.subtract(mean).abs().ge(deviation).sum()) / (len(simulations) + 1)
    return pd.DataFrame(
        dict(
            observed=observed,
            null_mean=mean,
            null_sd=sd,
            z=(observed - mean) / sd,
            p_mc=p,
            null_lower=simulations.quantile(0.025),
            null_upper=simulations.quantile(0.975),
        )
    )


def calibrated_poisson_check(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    replicates: int = NULL_REPLICATES,
    seed: int = NULL_SEED,
    observed_seed: int = DIAGNOSTIC_SEED,
    iterations: int = FIT_ITERATIONS,
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-season plug-in null; refit both season halves on EVERY simulated dataset.

    Independent child seeds generate goals and random complementary halves. The
    observed diagnostic uses the existing split seed, preserving earlier chapters.
    Full-season truth remains noisy: method artifacts can be slightly overstated.
    """
    selected = _selection(seasons)
    if replicates < 2:
        raise ValueError("At least two null replicates required")
    observed = poisson_diagnostic_metrics(
        poisson_crossfit_predictions(tables, selected, seed=observed_seed, iterations=iterations),
        selected,
    )
    truth = poisson_truth(tables, selected, iterations=iterations)
    seeds = np.random.SeedSequence(seed).spawn(replicates)
    rows = []
    for i, child in enumerate(seeds, 1):
        goal_seed, split_seed = map(int, child.generate_state(2))
        simulated = simulate_poisson_goals(truth, selected, seed=goal_seed)
        predictions = poisson_crossfit_predictions(
            replace(tables, match_table=simulated), selected, seed=split_seed, iterations=iterations
        )
        rows.append(poisson_diagnostic_metrics(predictions, selected))
        if progress is not None:
            progress(i)
    samples = pd.DataFrame(rows).reset_index(drop=True)
    result = summarize_poisson_null(observed, samples)
    result.attrs.update(
        replicates=replicates,
        seed=seed,
        observed_seed=observed_seed,
        iterations=iterations,
        algorithm=ALGORITHM_VERSION,
        split="random complementary halves within each season; not chronological OOF",
        calibrated_draw_excess=float(
            result.loc["all_draws", "observed"] / result.loc["all_draws", "null_mean"] - 1
        ),
    )
    return result, samples
