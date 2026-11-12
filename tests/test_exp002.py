"""EXP-002 arithmetic, reference fits, strict target isolation and blind persistence."""

from math import factorial, log
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import yaml
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from statsmodels.miscmodels.ordinal_model import OrderedModel

from ligat.analysis.prepare import AnalysisTables
from ligat.data.repository import Repository
from ligat.evaluate.backtest import TARGET_COLUMNS, evaluate
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.comparison import compare
from ligat.evaluate.form_experiment import (
    FormCandidate,
    FormExperimentInput,
    candidate_games,
    load_form_experiment,
    load_form_inputs,
    prepare_form_experiment,
    zero_counts,
)
from ligat.evaluate.form_experiment_artifacts import FormExperimentProvenance, save_form_experiment
from ligat.evaluate.goal_backtest import GoalEvaluationResult, evaluate_goals, summarize_goals
from ligat.evaluate.goal_metrics import poisson_nll, poisson_outcomes
from ligat.features.registry import FORM_COLUMNS
from ligat.models.baselines import B0
from ligat.models.form_models import FormOrderedPredictor, FormPoissonPredictor, fit_poisson
from ligat.models.ordered_logit import _matrix_logs, fit_ordered_logit, ordered_probabilities
from tests.test_evaluation import games
from tests.test_form import season_fixtures, tables_from


def ordinal_sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2002)
    x = rng.normal(size=(3000, 2))
    eta = x @ np.array([1.1, -0.7])
    cumulative = np.column_stack((expit(-0.8 - eta), expit(0.7 - eta)))
    y = 2 - (rng.random(len(x))[:, None] > cumulative).sum(axis=1)
    return x, y


def test_poisson_nll_hand_includes_log_factorial() -> None:
    y = np.array([0, 1, 3, 5])
    lam = np.array([1.0, 2.0, 2.5, 4.0])
    expected = np.array(
        [rate - g * log(rate) + log(factorial(g)) for g, rate in zip(y, lam, strict=True)]
    )
    np.testing.assert_allclose(poisson_nll(y, lam), expected, atol=1e-14)


@pytest.mark.parametrize(
    "y,lam", [([-1], [1]), ([0.5], [1]), ([0], [0]), ([1], [np.inf]), ([1, 2], [1])]
)
def test_poisson_nll_rejects_invalid_targets_and_rates(y: list[float], lam: list[float]) -> None:
    with pytest.raises(ValueError):
        poisson_nll(y, lam)


def test_poisson_conversion_grid_normalization_orientation_and_tail() -> None:
    rates = np.array([[2.0, 0.5], [0.5, 2.0], [1.0, 1.0]])
    p, tail = poisson_outcomes(rates)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15)
    assert p[0, 0] > p[0, 2]
    np.testing.assert_allclose(p[0], p[1, ::-1], atol=1e-15)
    assert p[2, 0] == pytest.approx(p[2, 2])
    from scipy.stats import poisson

    np.testing.assert_allclose(
        tail, 1 - poisson.cdf(10, rates[:, 0]) * poisson.cdf(10, rates[:, 1]), atol=1e-15
    )


def test_poisson_mle_matches_independent_statsmodels_glm() -> None:
    rng = np.random.default_rng(282)
    x = np.column_stack((np.ones(1200), rng.normal(size=(1200, 3))))
    y = rng.poisson(np.exp(x @ np.array([0.2, 0.1, 0.3, -0.2])))
    actual = fit_poisson(x, y)
    reference = sm.GLM(y, x, family=sm.families.Poisson()).fit(tol=1e-12, maxiter=1000)
    np.testing.assert_allclose(actual.coefficients, reference.params, atol=1e-6, rtol=0)
    assert actual.converged and actual.max_abs_gradient < 1e-6


def test_poisson_home_baseline_matches_two_group_mle() -> None:
    x = np.column_stack((np.ones(8), np.tile([1.0, 0.0], 4)))
    y = np.array([1, 0, 3, 1, 2, 1, 2, 2])
    fit = fit_poisson(x, y)
    np.testing.assert_allclose(fit.coefficients, [log(1), log(2)], atol=1e-7)


def test_zero_attack_and_crossed_venue_design_remain_finite_and_fit() -> None:
    frame = pd.DataFrame(
        {
            "season_id": [8, 8, 9, 9],
            "home_gf": [0.0, 1.0, 2.0, 3.0],
            "away_gf": [4.0, 2.0, 1.0, 0.0],
            "home_ga": [0.5, 2.0, 3.0, 4.0],
            "away_ga": [5.0, 4.0, 2.0, 1.0],
            "elo_diff": [10.0, -20.0, 30.0, -50.0],
        }
    )
    model = FormPoissonPredictor(with_elo=True, with_form=True, log_offset=0.1)
    design = model.design(frame)
    assert design[0, 2] == pytest.approx(log(0.1))
    assert design[0, 3] == pytest.approx(log(5.1))  # Home attack vs AWAY defense.
    assert design[1, 2] == pytest.approx(log(4.1))
    assert design[1, 3] == pytest.approx(log(0.6))  # Away attack vs HOME defense.
    assert design[0, 4] == 0.025 and design[1, 4] == -0.025
    model.fit(frame, pd.DataFrame({"home_goals": [1, 2, 2, 1], "away_goals": [2, 1, 0, 1]}))
    assert model.fit_result is not None and model.fit_result.converged
    assert np.isfinite(model.predict_goals(frame)).all()


def test_matrix_ordinal_matches_statsmodels_and_allows_negative_form_coefficient() -> None:
    x, y = ordinal_sample()
    actual = fit_ordered_logit(x, y)
    reference = OrderedModel(2 - y, x, distr="logit").fit(
        method="bfgs", disp=False, gtol=1e-7, maxiter=1000
    )
    assert reference.mle_retvals["converged"]
    np.testing.assert_allclose(
        [actual.beta, *actual.additional_betas, actual.theta1, actual.delta],
        reference.params,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ordered_probabilities(actual, x),
        reference.model.predict(reference.params)[:, ::-1],
        atol=1e-6,
    )
    assert actual.beta > 0 and actual.additional_betas[0] < 0


def test_one_column_matrix_path_is_exactly_the_original_vector_path() -> None:
    x, y = ordinal_sample()
    for weights in (None, np.arange(len(y)) % 3 + 1):
        vector = fit_ordered_logit(x[:, 0], y, weights)
        matrix = fit_ordered_logit(x[:, :1], y, weights)
        assert vector == matrix
        np.testing.assert_array_equal(
            ordered_probabilities(vector, x[:, 0]), ordered_probabilities(matrix, x[:, :1])
        )


def test_matrix_gradient_matches_finite_difference() -> None:
    x = np.array([[-1.0, 0.5], [0.0, -2.0], [2.0, 1.0]])
    parameters = np.array([1.1, -0.7, -0.8, np.log(1.5)])
    _, scores = _matrix_logs(parameters, x)
    numeric = approx_derivative(lambda p: _matrix_logs(p, x)[0].ravel(), parameters)
    np.testing.assert_allclose(scores.reshape(-1, 4), numeric, atol=1e-8, rtol=1e-7)


def test_matrix_precision_loss_accepts_only_recomputed_small_gradient() -> None:
    x, y = ordinal_sample()
    actual = fit_ordered_logit(x, y)
    params = np.array([actual.beta, *actual.additional_betas, actual.theta1, actual.delta])
    fake = SimpleNamespace(x=params, success=False, status=2, nit=1)
    with patch("ligat.models.ordered_logit.minimize", return_value=fake):
        assert fit_ordered_logit(x, y).accepted_by_gradient_check
    fake.x = np.array([1.0, 0.0, -0.8, 0.0])
    with patch("ligat.models.ordered_logit.minimize", return_value=fake):
        with pytest.raises(RuntimeError, match="converge"):
            fit_ordered_logit(x, y)
    fake.x = params
    fake.status = 1
    with patch("ligat.models.ordered_logit.minimize", return_value=fake):
        with pytest.raises(RuntimeError, match="converge"):
            fit_ordered_logit(x, y)


def test_goal_judge_keeps_targets_separate_and_training_in_past_seasons() -> None:
    fitted = []
    predicted = []

    class Spy:
        def fit(self, features: pd.DataFrame, goals: pd.DataFrame) -> None:
            assert not set(features) & TARGET_COLUMNS
            assert set(goals) == {"home_goals", "away_goals"}
            self.seasons = set(features.season_id)
            assert features.index.equals(goals.index)
            fitted.append(self.seasons)

        def predict_goals(self, features: pd.DataFrame) -> np.ndarray:
            assert not set(features) & TARGET_COLUMNS
            assert "unrequested_postmatch" not in features
            assert max(self.seasons) < int(features.season_id.min())
            predicted.append(set(features.season_id))
            return np.full((len(features), 2), 1.2)

    result = evaluate_goals(Spy, games(), feature_columns=["elo_diff"])
    assert len(fitted) == len(predicted) == 15
    assert fitted[0] == {8, 9, 10} and predicted[-1] == {25}
    assert len(result.per_team) == 2 * len(result.per_game)
    assert result.summary.poisson_nll == pytest.approx(result.per_team.poisson_nll.mean())
    assert result.per_fold.rows.eq(2 * result.per_fold.games).all()
    for target in TARGET_COLUMNS & set(games()):
        with pytest.raises(ValueError, match="target"):
            evaluate_goals(Spy, games(), feature_columns=[target])
    heldout = games().reset_index()
    heldout.loc[0, "season_id"] = 26
    with pytest.raises(ValueError, match="development"):
        evaluate_goals(Spy, heldout, feature_columns=[])


def goal_result(losses: np.ndarray, *, hda_loss: float) -> GoalEvaluationResult:
    rows = []
    for i, loss in enumerate(losses):
        for j in range(i + 1):
            rows.append(
                dict(
                    fold=i + 1,
                    season_id=i + 11,
                    game_id=i * 100 + j,
                    outcome=1,
                    home_goals=1,
                    away_goals=1,
                    poisson_nll=loss,
                    log_loss=hda_loss,
                    rps=0.2,
                )
            )
    scored = pd.DataFrame(rows)
    folds, summary = summarize_goals(scored)
    return GoalEvaluationResult(scored, folds, summary, pd.DataFrame())


def test_goal_comparison_selects_nll_and_uses_paired_one_se() -> None:
    best = goal_result(np.ones(15), hda_loss=1.2)
    consistently_worse = goal_result(np.ones(15) + 0.001, hda_loss=1.0)
    noisy = goal_result(np.ones(15) + np.tile([-0.1, 0.1, 0], 5), hda_loss=1.1)
    compared = compare(
        {"best": best, "consistent": consistently_worse, "noisy": noisy}, metric="poisson_nll"
    )
    assert compared.all_winners == ("best",)
    assert not compared.rankings.loc["consistent", "within_one_se"]
    assert compared.rankings.loc["noisy", "within_one_se"]
    assert compared.rankings.loc["best", "rows"] == 2 * len(best.per_game)
    # The legacy path still ranks these same outcomes by H/D/A log loss.
    assert compare({"best": best, "consistent": consistently_worse}).all_winners == ("consistent",)


def test_declared_grid_and_candidate_venue_join() -> None:
    spec = load_form_experiment()
    assert len(spec.grid) == 22 and len(spec.pilot) == 2 and spec.log_offset == 0.1
    index = pd.MultiIndex.from_tuples([(1, 10), (1, 20)], names=["game_id", "team_id"])
    form = pd.DataFrame(1.0, index=index, columns=FORM_COLUMNS)
    form["gf_venue_w10"] = [2.0, 3.0]
    form["ga_venue_w10"] = [4.0, 5.0]
    form["pts_venue_w10"] = [1.0, 2.0]
    prepared = FormExperimentInput(
        pd.DataFrame([dict(game_id=1, home_team_id=10, away_team_id=20, elo_diff=40.0)]), form
    )
    c = FormCandidate("w10", "venue")
    frame = candidate_games(prepared, c)
    design = FormPoissonPredictor(with_elo=True, with_form=True, log_offset=0.1).design(frame)
    np.testing.assert_allclose(design[:, 2:4], np.log([[2.1, 5.1], [3.1, 4.1]]))
    assert frame.points_difference.iloc[0] == -1
    assert c.complexity == 1 and FormCandidate("w10", "all").complexity == 0
    assert zero_counts(prepared, spec).shape == (22, 5)


def test_configuration_cannot_silently_change_methodology(tmp_path: Path) -> None:
    original = yaml.safe_load(Path("config/exp002.yaml").read_text(encoding="utf-8"))
    original["poisson"]["log_offset"] = 0.01
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(original), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        load_form_experiment(path)


def test_nullable_clean_keys_match_numeric_artifact_keys_but_missing_keys_fail() -> None:
    from dataclasses import replace

    tables = tables_from(season_fixtures(25))
    rows = tables.team_matches.reset_index().astype(
        {"game_id": "Int64", "team_id": "Int64", "opponent_id": "Int64"}
    )
    tables = replace(tables, team_matches=rows.set_index(["team_id", "game_id"]))
    index = pd.MultiIndex.from_frame(rows[["game_id", "team_id"]].astype("int64"))
    form = pd.DataFrame(1.0, index=index, columns=FORM_COLUMNS)
    elo = pd.DataFrame({"elo_diff": np.zeros(len(index))}, index=index)
    prepared = prepare_form_experiment(tables, form, elo)
    assert len(prepared.games) * 2 == len(form)
    with pytest.raises(ValueError, match="keys"):
        prepare_form_experiment(tables, form.iloc[:-1], elo)


@pytest.fixture(scope="module")
def real_prepared(real_tables: AnalysisTables) -> FormExperimentInput:
    spec = load_form_experiment()
    repository = Repository()
    artifacts = [Path(spec.raw["inputs"][key]["path"]) for key in ("form", "elo")]
    required = [
        path
        for artifact in artifacts
        for path in (artifact, artifact.with_suffix(".metadata.json"))
    ]
    required.extend(
        repository.root / f"{name}.parquet"
        for name in (
            "matches",
            "game_details",
            "goals",
            "red_cards",
            "coaches",
            "final_tables",
            "squad_values",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("Local EXP-002 regression inputs are unavailable: " + ", ".join(missing))
    form, elo = load_form_inputs(spec)
    return prepare_form_experiment(real_tables, form, elo)


def test_real_b0_regression_matches_exact_stored_reference(
    real_prepared: FormExperimentInput,
) -> None:
    # reports/experiments/EXP-001/execution.json, B0 baseline (full stored precision).
    result = evaluate(B0, real_prepared.games)
    assert result.summary.log_loss == pytest.approx(1.0849744118664024, abs=1e-12, rel=0)


def test_real_e0050_single_matrix_column_matches_exact_exp001_reference(
    real_prepared: FormExperimentInput,
) -> None:
    # EXP-001/stage1/comparison.csv, E0050 / elo-b24989..., no time weighting.
    result = evaluate(
        lambda: FormOrderedPredictor(with_form=False),
        real_prepared.games,
        feature_columns=["elo_diff"],
    )
    assert result.summary.log_loss == pytest.approx(1.020423847325146, abs=1e-9, rel=0)


def test_blind_save_and_full_save_share_manifest_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_form_experiment()
    evaluation = goal_result(np.ones(15), hda_loss=1.2)
    fit = evaluation.per_fold[["fold", "season_id"]].assign(
        training_seasons=[(8, 9, 10)] * 15,
        converged=True,
        iterations=5,
        accepted_by_gradient_check=False,
        max_abs_gradient=1e-10,
        coefficients=[(0.2, 0.1, 0.3, 0.4, 0.5)] * 15,
    )
    from ligat.evaluate.form_experiment import FormRun

    run = FormRun("P", FormCandidate("w10", "all"), True, evaluation, fit, 0.5)
    # A real goal evaluation has a tail diagnostic and per-team frame.
    run.evaluation.per_game["omitted_mass"] = 1e-6
    provenance = FormExperimentProvenance("a" * 40, "b" * 64, {"code.py": "c" * 64})
    monkeypatch.chdir(tmp_path)
    directory = Path("reports/experiments/EXP-002/pilot")
    manifest = save_form_experiment([run], spec, provenance, directory=directory, stage="pilot")
    metadata = verify_calibration(manifest)
    assert metadata["pilot_blind"]
    assert not {"fold_scores.parquet", "comparison_P.csv", "fits.parquet"} & {
        p.name for p in directory.iterdir()
    }
    diagnostic = pd.read_csv(directory / "convergence.csv")
    assert not {"log_loss", "poisson_nll", "rps", "coefficients", "beta"} & set(diagnostic)
    with pytest.raises(FileExistsError):
        save_form_experiment([run], spec, provenance, directory=directory, stage="pilot")
    full = save_form_experiment(
        [run], spec, provenance, directory=directory.parent / "full", stage="full"
    )
    assert verify_calibration(full)["stage"] == "full"
    assert (full.parent / "comparison_P.csv").exists()
    with pytest.raises(ValueError, match="EXP-002"):
        save_form_experiment([run], spec, provenance, directory=Path("data/raw"), stage="pilot")
