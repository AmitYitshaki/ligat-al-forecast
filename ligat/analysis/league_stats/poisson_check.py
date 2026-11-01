"""Retrospective two-fold Poisson diagnostic; never a chronological forecasting backtest."""

from collections.abc import Iterable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import poisson, skellam  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables

from .core import _selection, _subset, rate_interval

DIAGNOSTIC_SEED = 0
FIT_ITERATIONS = 100
PSEUDOCOUNT = 0.5
SCORELINES = (("0:0", 0, 0), ("1:0", 1, 0), ("0:1", 0, 1), ("1:1", 1, 1), ("2:2", 2, 2))


def _fit_strengths(
    training: pd.DataFrame, iterations: int
) -> tuple[float, float, pd.Series, pd.Series]:
    """Alternate attack/defence updates with a fixed pseudocount, using training goals only."""
    teams = pd.Index(pd.unique(pd.concat([training.home_team_id, training.away_team_id])))
    home = teams.get_indexer(training.home_team_id)
    away = teams.get_indexer(training.away_team_id)
    hg, ag = training.home_goals.to_numpy(float), training.away_goals.to_numpy(float)
    attack, defence = np.ones(len(teams)), np.ones(len(teams))
    home_mean, away_mean = float(hg.mean()), float(ag.mean())
    scored = np.bincount(home, weights=hg, minlength=len(teams)) + np.bincount(
        away, weights=ag, minlength=len(teams)
    )
    conceded = np.bincount(away, weights=hg, minlength=len(teams)) + np.bincount(
        home, weights=ag, minlength=len(teams)
    )
    for _ in range(iterations):
        lh, la = home_mean * attack[home] * defence[away], away_mean * attack[away] * defence[home]
        expected = np.bincount(home, weights=lh, minlength=len(teams)) + np.bincount(
            away, weights=la, minlength=len(teams)
        )
        attack *= (scored + PSEUDOCOUNT) / (expected + PSEUDOCOUNT)
        lh, la = home_mean * attack[home] * defence[away], away_mean * attack[away] * defence[home]
        expected = np.bincount(away, weights=lh, minlength=len(teams)) + np.bincount(
            home, weights=la, minlength=len(teams)
        )
        defence *= (conceded + PSEUDOCOUNT) / (expected + PSEUDOCOUNT)
    return home_mean, away_mean, pd.Series(attack, index=teams), pd.Series(defence, index=teams)


def poisson_crossfit_predictions(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    seed: int = DIAGNOSTIC_SEED,
    iterations: int = FIT_ITERATIONS,
) -> pd.DataFrame:
    """Predict each match with strengths fitted on its season's complementary random half.

    This deliberately uses later matches within development seasons to diagnose a
    distributional assumption retrospectively. It is NOT prematch data, M2 evaluation,
    or chronological OOF. A held-out match's score cannot affect its own lambdas.
    Both sides of a game always share a fold. Unseen clubs receive neutral strength.
    """
    selected = _selection(seasons)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    games = _subset(tables.match_table, selected).sort_index()
    if games.empty or games[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Poisson diagnostic requires complete played development matches")
    rng = np.random.default_rng(seed)
    output = games[["season_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]].copy()
    output["fold"] = -1
    output["lambda_home"] = np.nan
    output["lambda_away"] = np.nan
    for _, season in games.groupby("season_id", observed=True):
        folds = rng.integers(0, 2, len(season))
        if len(np.unique(folds)) != 2:
            raise ValueError("Each season needs two nonempty diagnostic halves")
        output.loc[season.index, "fold"] = folds
        for fold in (0, 1):
            training, held_out = season.loc[folds != fold], season.loc[folds == fold]
            h, a, attack, defence = _fit_strengths(training, iterations)
            for side, other, baseline in (("home", "away", h), ("away", "home", a)):
                strength = attack.reindex(held_out[f"{side}_team_id"]).fillna(1).to_numpy()
                opponent = defence.reindex(held_out[f"{other}_team_id"]).fillna(1).to_numpy()
                output.loc[held_out.index, f"lambda_{side}"] = baseline * strength * opponent
    output.attrs.update(
        method="two random complementary halves within season",
        seed=seed,
        iterations=iterations,
        chronological=False,
    )
    return output


def poisson_crossfit_comparison(
    tables: AnalysisTables,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    seed: int = DIAGNOSTIC_SEED,
    iterations: int = FIT_ITERATIONS,
) -> pd.DataFrame:
    """Compare observed and held-out Poisson rates; draws include the full infinite tail."""
    games = poisson_crossfit_predictions(tables, seasons, seed=seed, iterations=iterations)
    home, away = games.lambda_home.to_numpy(), games.lambda_away.to_numpy()
    draw_probability = np.empty(len(games))
    both = (home > 0) & (away > 0)
    draw_probability[both] = skellam.pmf(0, home[both], away[both])
    draw_probability[~both] = np.exp(-home[~both] - away[~both])
    rows = []
    for label, h, a in (*SCORELINES, ("all_draws", -1, -1)):
        flag = (
            games.home_goals.eq(games.away_goals)
            if h < 0
            else (games.home_goals.eq(h) & games.away_goals.eq(a))
        )
        probabilities = draw_probability if h < 0 else poisson.pmf(h, home) * poisson.pmf(a, away)
        observed, lower, upper = rate_interval(int(flag.sum()), len(games))
        expected = float(probabilities.mean())
        rows.append(
            dict(
                scoreline=label,
                n=len(games),
                count=int(flag.sum()),
                observed=observed,
                expected=expected,
                observed_ci_lower=lower,
                observed_ci_upper=upper,
                ratio=observed / expected if expected > 0 else np.nan,
            )
        )
    result = pd.DataFrame(rows).set_index("scoreline", drop=True)
    result.attrs.update(games.attrs)
    return result
