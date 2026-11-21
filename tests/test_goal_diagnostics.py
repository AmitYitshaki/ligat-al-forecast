"""Complete Poisson tails, home/away pairing and integration with review rules."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import poisson

from ligat.evaluate.goal_diagnostics import (
    goal_outcome_metrics,
    outcome_probabilities,
    poisson_score_matrix,
)
from ligat.evaluate.goal_metrics import poisson_outcomes
from ligat.evaluate.metrics import log_loss, rps
from ligat.evaluate.selection import paired_statistics
from ligat.evaluate.selection_rules import (
    LeakageSuspectedError,
    leakage_check,
    metric_disagreement_entry,
)
from tests.test_goal_reporting import evaluation


def test_matrix_conserves_mass_and_both_full_poisson_tails():
    home, away = np.array([0, 0.1, 1.5, 3, 20, 1000]), np.array([0, 2, 1.2, 3, 15, 1000])
    matrix = poisson_score_matrix(home, away)
    assert matrix.shape == (6, 6, 6)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    np.testing.assert_allclose(matrix.sum(axis=(1, 2)), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        matrix[:, 5, :].sum(axis=1), 1 - poisson.cdf(4, home), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        matrix[:, :, 5].sum(axis=1), 1 - poisson.cdf(4, away), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        matrix[:, :5, :5],
        poisson.pmf(np.arange(5)[None, :], home[:, None])[:, :, None]
        * poisson.pmf(np.arange(5)[None, :], away[:, None])[:, None, :],
        atol=1e-12,
        rtol=0,
    )


def test_home_draw_away_for_one_point_five_and_one_point_two_matches_manual_scipy():
    h = [poisson.pmf(i, 1.5) for i in range(5)] + [1 - poisson.cdf(4, 1.5)]
    a = [poisson.pmf(i, 1.2) for i in range(5)] + [1 - poisson.cdf(4, 1.2)]
    manual = [
        sum(h[i] * a[j] for i in range(6) for j in range(6) if condition(i, j))
        for condition in (lambda i, j: i > j, lambda i, j: i == j, lambda i, j: i < j)
    ]
    actual = outcome_probabilities(poisson_score_matrix([1.5], [1.2]))[0]
    np.testing.assert_allclose(actual, manual, atol=1e-12, rtol=0)
    legacy, _ = poisson_outcomes([[1.5, 1.2]])
    assert np.max(np.abs(actual - legacy[0])) < 1e-3


def test_exchanging_home_and_away_transposes_matrix_and_reverses_win_probabilities():
    h, a = [0.2, 1.5, 3, 8], [4, 1.2, 2, 0.3]
    matrix = poisson_score_matrix(h, a)
    reverse = poisson_score_matrix(a, h)
    np.testing.assert_array_equal(reverse, matrix.transpose(0, 2, 1))
    np.testing.assert_allclose(
        outcome_probabilities(reverse), outcome_probabilities(matrix)[:, ::-1], atol=1e-12, rtol=0
    )


def test_five_plus_five_plus_is_included_in_the_draw():
    matrix = poisson_score_matrix([3], [3])
    p = outcome_probabilities(matrix)[0]
    lower_diagonal = sum(matrix[0, i, i] for i in range(5))
    assert p[1] == pytest.approx(lower_diagonal + poisson.sf(4, 3) ** 2, abs=1e-12)
    only_tail = np.zeros((1, 6, 6))
    only_tail[0, 5, 5] = 1
    np.testing.assert_array_equal(outcome_probabilities(only_tail), [[0, 1, 0]])


def test_three_three_records_known_draw_bias_from_five_plus_aggregation():
    # H-SH29 maps even unequal scores >=5 to a draw: this is its known approximation cost.
    masses = poisson.pmf(np.arange(11), 3)
    collapsed_draw = sum(masses[:5] ** 2) + poisson.sf(4, 3) ** 2
    legacy_draw = sum(masses**2) / poisson.cdf(10, 3) ** 2
    expected_gap = collapsed_draw - legacy_draw
    assert expected_gap == pytest.approx(0.020784326692932426, abs=1e-12)
    new = outcome_probabilities(poisson_score_matrix([3], [3]))[0]
    old, _ = poisson_outcomes([[3, 3]])
    np.testing.assert_allclose(
        new - old[0], [-expected_gap / 2, expected_gap, -expected_gap / 2], atol=1e-12, rtol=0
    )


def test_legacy_conversion_sanity_over_positive_rates_up_to_two():
    # Amit approved 0.0015 here; at 2/2 the draw gap is already 0.001309149.
    rates = np.r_[0.001, np.linspace(0.05, 2, 40)]
    home, away = np.meshgrid(rates, rates)
    pairs = np.column_stack((home.ravel(), away.ravel()))
    actual = outcome_probabilities(poisson_score_matrix(pairs[:, 0], pairs[:, 1]))
    legacy, _ = poisson_outcomes(pairs)
    differences = np.abs(actual - legacy)
    assert (differences < 0.0015).all()
    assert np.unravel_index(differences.argmax(), differences.shape) == (len(pairs) - 1, 1)
    np.testing.assert_allclose(
        differences[-1],
        [0.000654574566334, 0.001309149132668, 0.000654574566334],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("home", "away"),
    [
        ([], []),
        ([1], [1, 2]),
        (1, 2),
        ([np.nan], [1]),
        ([np.inf], [1]),
        ([-1], [1]),
        ([[1]], [[1]]),
    ],
)
def test_matrix_rejects_invalid_rates(home, away):
    with pytest.raises(ValueError, match="rate arrays"):
        poisson_score_matrix(home, away)


@pytest.mark.parametrize("damage", ["shape", "negative", "nan", "mass", "empty"])
def test_outcomes_reject_invalid_matrices_without_normalizing(damage):
    matrix = poisson_score_matrix([1], [1])
    if damage == "shape":
        matrix = matrix[:, :5, :5]
    elif damage == "negative":
        matrix[0, 0, 0] = -0.1
    elif damage == "nan":
        matrix[0, 0, 0] = np.nan
    elif damage == "mass":
        matrix *= 0.9
    else:
        matrix = matrix[:0]
    with pytest.raises(ValueError, match="unit mass"):
        outcome_probabilities(matrix)


def test_game_pairing_uses_home_flag_and_keys_instead_of_team_or_row_order():
    a = evaluation([1.5, 1.2, 2, 0.8], goals=[2, 1, 0, 0], seasons=[11, 21])
    a = replace(a, per_team=a.per_team.sample(frac=1, random_state=5))
    result = goal_outcome_metrics(a)
    assert result.per_game.lambda_home.tolist() == [1.5, 2]
    assert result.per_game.lambda_away.tolist() == [1.2, 0.8]
    assert result.per_game.home_team_id.tolist() == [20, 20]
    assert result.per_game.away_team_id.tolist() == [10, 10]
    assert result.per_game.outcome.tolist() == [0, 1]
    probabilities = outcome_probabilities(poisson_score_matrix([1.5, 2], [1.2, 0.8]))
    np.testing.assert_allclose(
        result.per_game.log_loss, log_loss([0, 1], probabilities), atol=1e-12
    )
    np.testing.assert_allclose(result.per_game.rps, rps([0, 1], probabilities), atol=1e-12)
    assert result.per_fold.games.tolist() == [1, 1]
    assert result.summary.loc["all", "games"] == 2
    assert result.summary.loc["deep", "games"] == result.summary.loc["last5", "games"] == 1


def test_outcome_scope_scores_pool_games_across_unequal_folds():
    result = goal_outcome_metrics(
        evaluation([1, 1, 2, 2, 2, 2, 2, 2], goals=[1] * 8, seasons=[12, 21, 21, 21])
    )
    for metric in ("log_loss", "rps"):
        first, second = result.per_fold[metric]
        assert result.summary.loc["all", metric] == pytest.approx((first + 3 * second) / 4)
        assert result.summary.loc["deep", metric] == result.summary.loc["all", metric]
        assert result.summary.loc["last5", metric] == second


@pytest.mark.parametrize(
    "damage", ["missing", "two_home", "missing_home", "fold", "nll", "rate", "season"]
)
def test_game_pairing_rejects_incomplete_or_inconsistent_team_rows(damage):
    a = evaluation([1.5, 1.2, 2, 0.8])
    rows = a.per_team.copy()
    if damage == "missing":
        rows = rows.iloc[1:]
    elif damage == "two_home":
        rows["is_home"] = True
    elif damage == "missing_home":
        rows = rows.drop(columns="is_home")
    elif damage == "fold":
        rows.iloc[0, rows.columns.get_loc("fold")] = 9
    elif damage == "nll":
        rows["poisson_nll"] += 1
    elif damage == "rate":
        rows["lambda"] = 0
    else:
        rows["season_id"] = 26
    with pytest.raises(ValueError):
        goal_outcome_metrics(replace(a, per_team=rows))


def test_derived_fold_scores_feed_leakage_and_rule10_without_adapters():
    seasons = [11, 12, 21, 22, 23, 24, 25]
    a = evaluation([1] * 14, goals=[1] * 14, seasons=seasons)
    b = evaluation([1.5] * 14, goals=[1] * 14, seasons=seasons)
    ha, hb = goal_outcome_metrics(a), goal_outcome_metrics(b)
    assert leakage_check(ha.per_fold, score="log_loss").flagged_folds.empty
    nll = paired_statistics(a.per_fold, b.per_fold, score="poisson_nll")
    for metric in ("log_loss", "rps"):
        hda = paired_statistics(ha.per_fold, hb.per_fold, score=metric)
        assert hda.n == nll.n == 6
        assert not metric_disagreement_entry(nll, hda)
    suspicious = goal_outcome_metrics(evaluation([8, 0.2] * 2, goals=[5, 0] * 2))
    with pytest.raises(LeakageSuspectedError):
        leakage_check(suspicious.per_fold, score="log_loss")
