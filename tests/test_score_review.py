"""Tail mass, conditional permutations and simulation calibration regressions."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, poisson

from ligat.analysis.league_stats.match_flow import poisson_reference, scoreline_distribution
from ligat.analysis.league_stats.poisson_check import poisson_crossfit_predictions
from ligat.analysis.league_stats.score_review import (
    CATEGORIES,
    calibrated_poisson_check,
    poisson_baseline_comparison,
    poisson_diagnostic_metrics,
    poisson_truth,
    score_coverage,
    score_stability,
    simulate_poisson_goals,
    summarize_poisson_null,
)
from ligat.analysis.prepare import AnalysisTables
from ligat.analysis.score_cache import cached_poisson_check


def _tables(games: pd.DataFrame) -> AnalysisTables:
    return AnalysisTables(games, *[pd.DataFrame() for _ in range(5)])


@pytest.fixture
def synthetic() -> AnalysisTables:
    # Known independent Poisson with genuine heterogeneous attack and defence.
    rng = np.random.default_rng(17)
    n = 600
    home = np.arange(n) % 6
    away = (home + 1 + (np.arange(n) // 6) % 5) % 6
    attack = np.array([0.6, 0.8, 1, 1.1, 1.3, 1.5])
    defence = np.array([1.4, 1.2, 1, 1, 0.8, 0.7])
    h, a = 1.4 * attack[home] * defence[away], attack[away] * defence[home]
    return _tables(
        pd.DataFrame(
            dict(
                season_id=np.repeat([12, 13], n // 2),
                home_team_id=home,
                away_team_id=away,
                home_goals=rng.poisson(h),
                away_goals=rng.poisson(a),
                lambda_home=h,
                lambda_away=a,
            )
        )
    )


def test_tail_buckets_keep_all_games_and_probability(synthetic: AnalysisTables) -> None:
    games = synthetic.match_table.copy()
    games.loc[0, ["home_goals", "away_goals"]] = [6, 0]
    games.loc[1, ["home_goals", "away_goals"]] = [7, 1]
    source = replace(synthetic, match_table=games)
    observed = scoreline_distribution(source)
    reference = poisson_reference(source)
    assert observed.shape == reference.shape == (6, 6)
    assert list(observed.index) == ["0", "1", "2", "3", "4", "5+"]
    assert observed.to_numpy().sum() == pytest.approx(1)
    assert reference.to_numpy().sum() == pytest.approx(1)
    assert np.sum(observed.attrs["counts"]) == len(games)
    assert observed.loc["5+", "0"] >= 1 / len(games)
    assert observed.loc["5+", "1"] >= 1 / len(games)
    assert reference.loc["5+"].sum() == pytest.approx(
        1 - poisson.pmf(np.arange(5), games.home_goals.mean()).sum()
    )
    assert reference["5+"].sum() == pytest.approx(
        1 - poisson.pmf(np.arange(5), games.away_goals.mean()).sum()
    )
    assert reference.attrs["tail_mass"] == pytest.approx(
        1 - reference.iloc[:5, :5].to_numpy().sum()
    )
    assert score_coverage(source).loc[7, "outside"] == int(
        (games[["home_goals", "away_goals"]] > 7).any(axis=1).sum()
    )


def test_stability_categories_and_scipy(synthetic: AnalysisTables) -> None:
    deviations, trends = score_stability(synthetic, permutations=49)
    counts = np.array(deviations.attrs["counts"])
    statistic, p, dof, _ = chi2_contingency(counts, correction=False)
    assert counts.sum() == len(synthetic.match_table)
    assert list(deviations.columns) == list(CATEGORIES)
    labels = (
        synthetic.match_table.home_goals.astype(str)
        + ":"
        + synthetic.match_table.away_goals.astype(str)
    )
    assert counts[:, -1].sum() == (~labels.isin(CATEGORIES[:-1])).sum()
    assert deviations.attrs["chi2"] == pytest.approx(statistic)
    assert deviations.attrs["p_asymptotic"] == pytest.approx(p)
    assert deviations.attrs["dof"] == dof
    assert len(trends) == 9
    pd.testing.assert_frame_equal(deviations, score_stability(synthetic, permutations=49)[0])


def test_simulation_reads_lambdas_only(synthetic: AnalysisTables) -> None:
    truth = poisson_truth(synthetic)
    assert "home_goals" not in truth and "away_goals" not in truth
    original = simulate_poisson_goals(truth, seed=41)
    contaminated = truth.assign(home_goals=999, away_goals=-999)
    pd.testing.assert_frame_equal(original, simulate_poisson_goals(contaminated, seed=41))
    assert not original.equals(simulate_poisson_goals(truth, seed=42))


def test_standardized_residual_correlation_manual(synthetic: AnalysisTables) -> None:
    games = synthetic.match_table.head(6).copy()
    metrics = poisson_diagnostic_metrics(games)
    h, a = games.lambda_home.to_numpy(), games.lambda_away.to_numpy()
    x, y = (games.home_goals - h) / np.sqrt(h), (games.away_goals - a) / np.sqrt(a)
    manual = np.sum((x - x.mean()) * (y - y.mean())) / np.sqrt(
        np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)
    )
    assert metrics.residual_correlation == pytest.approx(manual)
    assert metrics.dispersion_home == pytest.approx(((games.home_goals - h) ** 2).sum() / h.sum())


def test_null_reproducible_and_independent_poisson_has_no_spurious_signal(
    synthetic: AnalysisTables,
) -> None:
    result, samples = calibrated_poisson_check(synthetic, replicates=30, iterations=30)
    again, repeated = calibrated_poisson_check(synthetic, replicates=30, iterations=30)
    pd.testing.assert_frame_equal(result, again)
    pd.testing.assert_frame_equal(samples, repeated)
    assert np.isfinite(result.z).all()
    assert result.z.abs().max() < 4  # fixed known-Poisson fixture, every requested metric
    assert result.p_mc.between(1 / 31, 1).all()
    assert result.null_lower.le(result.null_upper).all()


def test_monte_carlo_p_counts_two_sided_deviation() -> None:
    samples = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    result = summarize_poisson_null(pd.Series({"x": 3.5}), samples)
    assert result.loc["x", "p_mc"] == pytest.approx(3 / 6)


def test_cache_reuses_only_identical_input(synthetic: AnalysisTables, tmp_path: Path) -> None:
    path = tmp_path / "null.json"
    first, _ = cached_poisson_check(synthetic, path, replicates=3, iterations=10)
    second, _ = cached_poisson_check(synthetic, path, replicates=3, iterations=10)
    assert not first.attrs["cache_hit"] and second.attrs["cache_hit"]
    pd.testing.assert_frame_equal(first, second)
    changed = synthetic.match_table.copy()
    changed.loc[0, "home_goals"] += 1
    third, _ = cached_poisson_check(
        replace(synthetic, match_table=changed), path, replicates=3, iterations=10
    )
    assert not third.attrs["cache_hit"]
    assert third.attrs["cache_key"] != first.attrs["cache_key"]


@pytest.mark.parametrize("seasons", [[26], [27], [28], [8, 26], [], [8, 8]])
@pytest.mark.parametrize(
    "name",
    ["coverage", "stability", "truth", "simulate", "metrics", "calibration", "cache", "comparison"],
)
def test_all_entry_points_reject_non_development(
    name: str, seasons: list[int], tmp_path: Path
) -> None:
    source = _tables(pd.DataFrame())
    with pytest.raises(ValueError, match="development"):
        if name == "coverage":
            score_coverage(source, seasons)
        elif name == "stability":
            score_stability(source, seasons)
        elif name == "truth":
            poisson_truth(source, seasons)
        elif name == "simulate":
            simulate_poisson_goals(source.match_table, seasons)
        elif name == "metrics":
            poisson_diagnostic_metrics(source.match_table, seasons)
        elif name == "cache":
            cached_poisson_check(source, tmp_path / "x.json", seasons)
        elif name == "comparison":
            poisson_baseline_comparison(source, seasons)
        else:
            calibrated_poisson_check(source, seasons)


def test_existing_crossfit_is_unchanged_by_heldout_score(synthetic: AnalysisTables) -> None:
    original = poisson_crossfit_predictions(synthetic)
    changed = synthetic.match_table.copy()
    game = original.index[0]
    changed.loc[game, ["home_goals", "away_goals"]] = [7, 6]
    altered = poisson_crossfit_predictions(replace(synthetic, match_table=changed))
    assert (
        original.loc[game, ["lambda_home", "lambda_away"]].tolist()
        == altered.loc[game, ["lambda_home", "lambda_away"]].tolist()
    )
