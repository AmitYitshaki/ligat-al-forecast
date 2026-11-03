"""Manual scoring, target isolation, approved folds and paired comparison rules."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.league_stats import context, t1_strength_comparison
from ligat.analysis.prepare import AnalysisTables
from ligat.evaluate.backtest import (
    RETROSPECTIVE_LABEL,
    EvaluationResult,
    development_folds,
    evaluate,
    summarize,
)
from ligat.evaluate.comparison import compare
from ligat.evaluate.metrics import EPS, log_loss, rps, validate_probabilities
from ligat.models.baselines import B0


def games() -> pd.DataFrame:
    rows = []
    for season in range(8, 26):
        for n in range(season - 6):
            outcome = (season + n) % 3
            rows.append(
                {
                    "game_id": len(rows) + 1,
                    "season_id": season,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_goals": int(outcome == 0),
                    "away_goals": int(outcome == 2),
                    "result": "observed",
                    "points": 3,
                    "home_points": 3,
                    "elo_diff": float(n),
                    "unrequested_postmatch": 999,
                }
            )
    return pd.DataFrame(rows).set_index("game_id")


def fake_result(losses: np.ndarray, counts: np.ndarray | None = None) -> EvaluationResult:
    counts = np.ones(15, dtype=int) if counts is None else counts
    records = [
        {
            "fold": i + 1,
            "season_id": i + 11,
            "game_id": f"{i}-{n}",
            "outcome": n % 3,
            "log_loss": float(loss),
            "rps": float(loss / 5),
        }
        for i, (loss, count) in enumerate(zip(losses, counts, strict=True))
        for n in range(int(count))
    ]
    scored = pd.DataFrame(records)
    folds, summary = summarize(scored)
    return EvaluationResult(scored, folds, summary, "walk_forward", "synthetic")


@pytest.mark.parametrize(
    "outcome,expected_loss,expected_rps",
    [(0, 0.69314718056, 0.145), (1, 1.20397280433, 0.145), (2, 1.60943791243, 0.445)],
)
def test_manual_scores(outcome: int, expected_loss: float, expected_rps: float) -> None:
    p = np.array([[0.5, 0.3, 0.2]])
    assert log_loss([outcome], p)[0] == pytest.approx(expected_loss)
    assert rps([outcome], p)[0] == pytest.approx(expected_rps)


def test_eps_clips_only_observed_probability_without_renormalizing() -> None:
    p = np.array([[0.0, 0.3, 0.7], [0.5, 0.3, 0.2]])
    original = p.copy()
    assert log_loss([0, 1], p)[0] == pytest.approx(-np.log(EPS))
    assert log_loss([0, 1], p, eps=0.4).tolist() == pytest.approx([-np.log(0.4)] * 2)
    np.testing.assert_array_equal(p, original)
    near = np.array([[0.5, 0.3, 0.2000000005]])
    np.testing.assert_array_equal(validate_probabilities(near), near)


@pytest.mark.parametrize(
    "p",
    [
        [],
        [0.5, 0.3, 0.2],
        [[0.5, 0.5]],
        [[0.3, 0.3, 0.3, 0.1]],
        [[-0.1, 0.5, 0.6]],
        [[np.nan, 0.5, 0.5]],
        [[np.inf, 0, 0]],
        [[0, 0, 0]],
        [[0.5, 0.3, 0.21]],
    ],
)
def test_invalid_probability_matrices_fail(p: list) -> None:
    for metric in (log_loss, rps):
        with pytest.raises(ValueError):
            metric([0], p)


@pytest.mark.parametrize("y", [[-1], [3], [0.2], [np.nan], [True], ["0"], [[0]], [0, 1]])
def test_invalid_outcomes_fail(y: list) -> None:
    for metric in (log_loss, rps):
        with pytest.raises(ValueError):
            metric(y, [[0.5, 0.3, 0.2]])


@pytest.mark.parametrize("eps", [0, -0.1, 1, np.inf, np.nan])
def test_invalid_epsilon_fails(eps: float) -> None:
    with pytest.raises(ValueError):
        log_loss([0], [[0.5, 0.3, 0.2]], eps=eps)


def test_b0_frequencies_and_refit_do_not_carry_previous_training() -> None:
    model = B0()
    with pytest.raises(ValueError, match="fitted"):
        model.predict(pd.DataFrame(index=range(2)))
    model.fit(pd.DataFrame({"outcome": [0, 0, 1, 2]}))
    np.testing.assert_array_equal(
        model.predict(pd.DataFrame(index=range(2))), [[0.5, 0.25, 0.25]] * 2
    )
    model.fit(pd.DataFrame({"outcome": [1, 1]}))
    np.testing.assert_array_equal(model.predict(pd.DataFrame(index=range(1))), [[0, 1, 0]])


def test_walk_forward_is_exactly_15_folds_and_predict_sees_only_requested_inputs() -> None:
    data = games()
    original = data.copy(deep=True)
    seen = []

    class Spy(B0):
        def fit(self, train: pd.DataFrame) -> None:
            self.training = set(train.season_id)
            assert "outcome" in train
            super().fit(train)

        def predict(self, validation: pd.DataFrame) -> np.ndarray:
            year = int(validation.season_id.iloc[0])
            assert self.training == set(range(8, year))
            assert set(validation) == {
                "game_id",
                "season_id",
                "home_team_id",
                "away_team_id",
                "elo_diff",
            }
            seen.append(year)
            return super().predict(validation)

    result = evaluate(Spy, data, feature_columns=["elo_diff"])
    assert seen == list(range(11, 26))
    assert len(result.per_fold) == 15
    assert result.per_fold.season_id.tolist() == list(range(11, 26))
    assert set(result.per_game.season_id) == set(range(11, 26))
    assert not result.per_game.game_id.duplicated().any()
    assert int(result.summary.games) == len(data.loc[data.season_id.ge(11)])
    assert_frame_equal(data, original)


@pytest.mark.parametrize("column", ["home_goals", "away_goals", "result", "points", "home_points"])
def test_cheating_predictor_cannot_read_validation_targets(column: str) -> None:
    class Cheater(B0):
        def predict(self, validation: pd.DataFrame) -> np.ndarray:
            validation[column]
            return super().predict(validation)

    with pytest.raises(KeyError, match=column):
        evaluate(Cheater, games())
    with pytest.raises(ValueError, match="target"):
        evaluate(B0, games(), feature_columns=[column])


@pytest.mark.parametrize("label,year", [("final_validation", 26), ("test", 27), ("live", 28)])
def test_heldout_games_and_fold_rows_fail_before_fitting(label: str, year: int) -> None:
    data = games().reset_index()
    heldout = data.iloc[[0]].assign(game_id=999, season_id=year)
    with pytest.raises(ValueError, match="development"):
        evaluate(B0, pd.concat([data, heldout]))
    plan = development_folds(data)
    extra = pd.DataFrame(
        [{"fold": label, "train_seasons": tuple(range(8, 26)), "validation_season": year}]
    )
    with pytest.raises(ValueError, match="forbidden"):
        evaluate(B0, data, folds=pd.concat([plan, extra], ignore_index=True))


def test_loso_omits_validation_from_training_and_is_never_a_selection_mode() -> None:
    data = games()
    seen = []

    class Spy(B0):
        def fit(self, train: pd.DataFrame) -> None:
            self.training = set(train.season_id)
            super().fit(train)

        def predict(self, validation: pd.DataFrame) -> np.ndarray:
            year = int(validation.season_id.iloc[0])
            assert self.training == set(range(8, 26)) - {year}
            seen.append(year)
            return super().predict(validation)

    result = evaluate(Spy, data, mode="loso")
    assert seen == list(range(8, 26))
    assert result.label == RETROSPECTIVE_LABEL
    assert len(result.per_fold) == 18
    with pytest.raises(ValueError, match="not for selection"):
        compare({"B0": result})
    stability = compare({"B0": result}, for_selection=False)
    assert stability.label == RETROSPECTIVE_LABEL
    assert "within_one_se" not in stability.rankings
    assert stability.rankings.loc["B0", "rank_all"] == 1


def test_weighted_means_and_unweighted_fold_dispersion() -> None:
    scored = pd.DataFrame(
        {
            "fold": [1, 2, 2, 2],
            "season_id": [11, 12, 12, 12],
            "game_id": [1, 2, 3, 4],
            "log_loss": [0.2, 0.8, 0.8, 0.8],
            "rps": [0.1, 0.3, 0.3, 0.3],
        }
    )
    folds, summary = summarize(scored)
    assert folds.games.tolist() == [1, 3]
    assert summary.log_loss == pytest.approx(0.65)
    assert summary.rps == pytest.approx(0.25)
    assert summary.log_loss_std == pytest.approx(np.std([0.2, 0.8], ddof=1))
    assert summary.log_loss_min == 0.2 and summary.log_loss_max == pytest.approx(0.8)


def test_last_five_summary_and_disagreement_never_select_automatically() -> None:
    first = fake_result(np.array([0.9] * 10 + [1.2] * 5))
    second = fake_result(np.array([1.05] * 10 + [1.0] * 5))
    report = compare({"A": first, "B": second})
    assert report.all_winners == ("A",)
    assert report.last5_winners == ("B",)
    assert report.disagreement and report.requires_decision
    assert report.rankings.loc["A", "folds_best"] == 10
    assert report.rankings.loc["B", "folds_best"] == 5
    assert not hasattr(report, "chosen")
    assert second.summary.last5_log_loss == pytest.approx(1)
    assert not any("last5" in c and "se" in c for c in report.rankings)


def test_paired_one_se_distinguishes_consistent_small_loss_from_noisy_near_zero() -> None:
    counts = np.arange(1, 16)
    best = fake_result(np.ones(15), counts)
    constant = fake_result(np.full(15, 1.00001), counts)
    differences = np.array([-0.01, 0.01] * 7 + [0.001])
    noisy = fake_result(1 + differences, counts)
    ranked = compare({"best": best, "constant": constant, "noisy": noisy}).rankings
    assert bool(ranked.loc["best", "within_one_se"])
    assert not bool(ranked.loc["constant", "within_one_se"])
    assert bool(ranked.loc["noisy", "within_one_se"])
    assert ranked.loc["noisy", "paired_mean_difference"] == pytest.approx(
        np.average(differences, weights=counts)
    )
    assert ranked.loc["noisy", "paired_se"] == pytest.approx(
        np.std(differences, ddof=1) / np.sqrt(15)
    )


def test_comparison_requires_identical_game_keys_and_targets() -> None:
    first = fake_result(np.ones(15))
    missing = fake_result(np.ones(15))
    missing.per_game.drop(index=0, inplace=True)
    with pytest.raises(ValueError, match="same games"):
        compare({"a": first, "b": missing})
    wrong_target = fake_result(np.ones(15))
    wrong_target.per_game.loc[0, "outcome"] = 2
    with pytest.raises(ValueError, match="same games"):
        compare({"a": first, "b": wrong_target})


def test_exact_ties_require_review_without_an_arbitrary_reference() -> None:
    best = fake_result(np.ones(15))
    other = fake_result(np.full(15, 1.1))
    report = compare({"A": best, "B": best, "C": other})
    assert report.requires_decision
    assert report.all_winners == ("A", "B")
    assert bool(report.rankings.loc["A", "within_one_se"])
    assert pd.isna(report.rankings.loc["C", "within_one_se"])


def test_bad_predictor_shape_is_rejected() -> None:
    class Bad(B0):
        def predict(self, validation: pd.DataFrame) -> np.ndarray:
            return np.ones((len(validation), 2))

    with pytest.raises(ValueError, match="n×3"):
        evaluate(Bad, games())


def test_overlapping_or_missing_training_seasons_are_rejected() -> None:
    data = games()
    plan = development_folds(data)
    plan.at[0, "train_seasons"] = (8, 9, 11)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(B0, data, folds=plan)
    with pytest.raises(ValueError, match="18 development"):
        evaluate(B0, data.loc[data.season_id.ne(8)])


def test_notebook02_baseline_is_identical_on_the_original_sample(
    real_tables: AnalysisTables,
) -> None:
    sample = context._strength_sample(real_tables, list(range(8, 26))).sort_index()
    y = np.select(
        [sample.home_goals.gt(sample.away_goals), sample.home_goals.eq(sample.away_goals)],
        [0, 1],
        2,
    )
    probabilities = np.empty((len(sample), 3))
    for held_out in sorted(sample.season_id.unique()):
        test = sample.season_id.eq(held_out).to_numpy()
        probabilities[test] = np.bincount(y[~test], minlength=3) / (~test).sum()
    scored = evaluate(B0, sample, mode="loso")
    actual = scored.per_game.set_index("game_id").reindex(sample.index)
    np.testing.assert_array_equal(actual[["p_home", "p_draw", "p_away"]].to_numpy(), probabilities)
    previous = t1_strength_comparison(real_tables, n_bootstrap=20).loc["baseline"]
    assert log_loss(y, probabilities).mean() == pytest.approx(previous.log_loss, abs=1e-14)
    assert rps(y, probabilities).mean() == pytest.approx(previous.rps, abs=1e-14)
    assert log_loss(y, probabilities).mean() == pytest.approx(1.083536, abs=0.0000005)
    assert rps(y, probabilities).mean() == pytest.approx(0.230552, abs=0.0000005)
