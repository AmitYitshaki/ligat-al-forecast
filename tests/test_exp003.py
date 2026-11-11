"""Declared context contrasts, policy isolation, blind persistence and frozen regressions."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prepare import load_analysis_tables
from ligat.evaluate.backtest import DevelopmentPolicy, EvaluationResult, policy_folds, summarize
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.context_experiment import (
    CANDIDATES,
    ContextConfiguration,
    ContextExperimentInput,
    ContextInputs,
    ContextOrderedPredictor,
    ContextPoissonPredictor,
    ContextRun,
    context_comparisons,
    context_configurations,
    load_context_experiment,
    load_context_inputs,
    prepare_context_experiment,
    run_context_configuration,
)
from ligat.evaluate.context_experiment_artifacts import ContextProvenance, save_context_experiment
from ligat.evaluate.goal_backtest import GoalEvaluationResult, summarize_goals
from ligat.features.registry import FORM_COLUMNS, SEASON_CONTEXT_COLUMNS
from ligat.features.table_state import game_rows
from ligat.models.baselines import B0
from ligat.models.ordered_logit import OrderedFit, fit_ordered_logit, ordered_probabilities
from tests.test_form import season_fixtures, tables_from


def fake_inputs():
    tables = tables_from(season_fixtures(25))
    rows = game_rows(tables)
    index = pd.MultiIndex.from_frame(rows[["game_id", "team_id"]].astype("int64")).sort_values()
    identity = rows.set_index(["game_id", "team_id"]).reindex(index)
    form = pd.DataFrame(1.0, index=index, columns=FORM_COLUMNS)
    elo = pd.DataFrame(
        {
            "elo_pre": 1500.0,
            "elo_opponent_pre": 1400.0,
            "elo_diff": np.where(identity.is_home, 100.0, -100.0),
            "elo_expected": 0.5,
        },
        index=index,
    )
    context = pd.DataFrame(0.5, index=index, columns=SEASON_CONTEXT_COLUMNS[:5])
    context["promoted_in"] = (
        identity.season_id.eq(9) & (index.get_level_values("team_id") == 5)
    ).to_numpy()
    context["squad_value_imputed"] = context.promoted_in
    context["prev_season_known"] = identity.season_id.ne(8).to_numpy()
    context["league_size"] = 4
    context.loc[context.promoted_in, "prev_position_pct"] = 0.75
    context.loc[~context.prev_season_known, SEASON_CONTEXT_COLUMNS[:5]] = np.nan
    audit = pd.DataFrame({"season_id": list(range(8, 26)), "promoted_position_mean": 0.6})
    return tables, ContextInputs(context, elo, form, audit)


def test_declared_grid_is_exactly_33_with_eight_sensitivity_and_four_pilot():
    spec = load_context_experiment()
    configs = context_configurations()
    assert len(configs) == len({c.config_id for c in configs}) == 33
    assert sum(c.sensitivity for c in configs) == 8
    assert sum(c.route == "A" for c in configs) == 9
    assert len(CANDIDATES) == 7
    assert set(spec.raw["pilot"]["configurations"]) <= {c.config_id for c in configs}
    assert ContextConfiguration("O", candidate="value_avg+position").columns == (
        "z_value_avg",
        "z_position",
    )


def test_position_mean_changes_only_promoted_team_rows_and_season_8_stays_missing():
    tables, inputs = fake_inputs()
    original = inputs.context.copy(deep=True)
    prepared = prepare_context_experiment(tables, inputs)
    frame = prepared.team_context
    assert_frame_equal(inputs.context, original)
    assert frame.loc[frame.promoted_in, "position_mean"].eq(0.6).all()
    pd.testing.assert_series_equal(
        frame.loc[~frame.promoted_in, "position_mean"],
        frame.loc[~frame.promoted_in, "prev_position_pct"],
        check_names=False,
    )
    assert frame.loc[frame.season_id.eq(8), SEASON_CONTEXT_COLUMNS[:5]].isna().all().all()
    assert (
        prepared.games.loc[prepared.games.season_id.eq(8), ["z_value_avg", "z_position", "z_gd"]]
        .isna()
        .all()
        .all()
    )


def test_poisson_context_is_exactly_antisymmetric_and_combination_has_two_columns():
    features = pd.DataFrame(
        {
            "home_gf": [0, 2],
            "away_gf": [1, 3],
            "home_ga": [1, 1],
            "away_ga": [2, 2],
            "elo_diff": [100, -100],
            "z_value_avg": [0.3, -0.7],
            "z_position": [-0.2, 0.5],
        }
    )
    model = ContextPoissonPredictor(("z_value_avg", "z_position"))
    design = model.design(features)
    assert design.shape == (4, 7)
    np.testing.assert_array_equal(design[::2, -2:], -design[1::2, -2:])
    np.testing.assert_array_equal(design[::2, -2:], features[["z_value_avg", "z_position"]])
    assert design[0, 2] == pytest.approx(np.log(0.1))
    assert design[0, 3] == pytest.approx(np.log(2.1))


@pytest.mark.parametrize("combined", [False, True])
def test_alone_route_can_learn_negative_candidate_slope(combined):
    rng = np.random.default_rng(720)
    x = rng.normal(size=(2500, 2 if combined else 1))
    truth = OrderedFit(-1.2, -1, np.log(2), 0, True, 0, False, 0.0, (0.6,) if combined else ())
    p = ordered_probabilities(truth, x)
    y = np.array([rng.choice(3, p=row) for row in p])
    configuration = ContextConfiguration(
        "A", candidate="value_avg+position" if combined else "position"
    )
    frame = pd.DataFrame(x, columns=configuration.columns).assign(season_id=9, outcome=y)
    model = ContextOrderedPredictor(configuration)
    model.fit(frame)
    assert model.fit_result.beta < 0
    if combined:
        assert (
            len(model.fit_result.additional_betas) == 1 and model.fit_result.additional_betas[0] > 0
        )
    with pytest.raises(RuntimeError, match="Negative"):
        fit_ordered_logit(x, y)  # Existing first/Elo sign guard remains the default.


def test_added_candidate_is_free_to_be_negative_while_elo_is_positive():
    rng = np.random.default_rng(721)
    x = rng.normal(size=(2500, 2))
    truth = OrderedFit(1.4, -1, np.log(2), 0, True, 0, False, 0.0, (-0.8,))
    p = ordered_probabilities(truth, x)
    y = np.array([rng.choice(3, p=row) for row in p])
    model = ContextOrderedPredictor(ContextConfiguration("O", candidate="gd"))
    model.fit(
        pd.DataFrame({"elo_diff": 400 * x[:, 0], "z_gd": x[:, 1], "season_id": 9, "outcome": y})
    )
    assert model.fit_result.beta > 0 and model.fit_result.additional_betas[0] < 0


def test_poisson_uses_existing_fit_and_accepts_negative_candidate_coefficient():
    rng = np.random.default_rng(722)
    n = 1200
    features = pd.DataFrame(
        {name: rng.lognormal(0, 0.4, n) for name in ("home_gf", "away_gf", "home_ga", "away_ga")}
    )
    features["elo_diff"] = rng.normal(0, 200, n)
    features["z_gd"] = rng.normal(size=n)
    features["season_id"] = 9
    model = ContextPoissonPredictor(("z_gd",))
    truth = np.array([0.1, 0.15, 0.2, 0.15, 1.2, -0.6])
    goals = pd.DataFrame(
        rng.poisson(np.exp(model.design(features) @ truth)).reshape(-1, 2),
        columns=["home_goals", "away_goals"],
    )
    model.fit(features, goals)
    assert model.fit_result.coefficients[4] > 0 and model.fit_result.coefficients[-1] < 0


def policy_games():
    rows = []
    for season in range(8, 26):
        for i, (home, away) in enumerate([(1, 0), (0, 0), (0, 1)]):
            rows.append(
                {
                    "game_id": season * 10 + i,
                    "season_id": season,
                    "home_goals": home,
                    "away_goals": away,
                    "elo_diff": 0.0,
                    "z_value_avg": np.nan if season == 8 else 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("sensitivity", [False, True])
def test_all_context_training_and_validation_obey_season_policy(monkeypatch, sensitivity):
    observed = []

    class Spy(B0):
        fit_result = None

        def __init__(self, configuration):
            super().__init__()

        def fit(self, train):
            self.training_seasons = tuple(sorted(train.season_id.unique()))
            observed.append(self.training_seasons)
            assert 8 not in self.training_seasons
            if sensitivity:
                assert 11 not in self.training_seasons
            super().fit(train)

    monkeypatch.setattr("ligat.evaluate.context_experiment.ContextOrderedPredictor", Spy)
    games = policy_games()
    prepared = ContextExperimentInput(games, pd.DataFrame(), "a" * 64)
    run = run_context_configuration(
        prepared,
        load_context_experiment(),
        ContextConfiguration("O", candidate="value_avg", sensitivity=sensitivity),
    )
    assert len(observed) == (14 if sensitivity else 15)
    assert observed[0] == (9, 10)
    assert set(run.evaluation.per_game.season_id) == (
        set(range(12, 26)) if sensitivity else set(range(11, 26))
    )
    plan = policy_folds(
        games,
        DevelopmentPolicy(
            9,
            (11,) if sensitivity else (),
            tuple(s for s in range(11, 26) if not sensitivity or s != 11),
        ),
    )
    assert plan.n_train_matches.iloc[0] == 6


def test_b0_reference_also_excludes_season_8(monkeypatch):
    observed = []
    original = B0.fit

    def fit(self, train):
        observed.append(set(train.season_id))
        return original(self, train)

    monkeypatch.setattr(B0, "fit", fit)
    run_context_configuration(
        ContextExperimentInput(policy_games(), pd.DataFrame(), "a" * 64),
        load_context_experiment(),
        ContextConfiguration("A", reference="b0"),
    )
    assert len(observed) == 15 and all(8 not in seasons for seasons in observed)


def test_checksum_failure_in_second_input_precedes_every_dataframe_read(tmp_path, monkeypatch):
    spec = load_context_experiment()
    raw = json.loads(json.dumps(spec.raw))
    monkeypatch.chdir(tmp_path)
    for name in ("context", "elo", "form"):
        path = Path("data/derived") / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen")
        sha = hashlib.sha256(b"frozen").hexdigest()
        sidecar = path.with_suffix(".metadata.json")
        sidecar.write_text(json.dumps({"artifact_sha256": sha, "seasons": list(range(8, 26))}))
        raw["inputs"][name] = {
            "path": str(path),
            "sha256": sha,
            "metadata_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        }
    Path(raw["inputs"]["elo"]["path"]).write_bytes(b"modified")
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pytest.fail("read before all checksums")
    )
    with pytest.raises(ValueError, match="checksum"):
        load_context_inputs(replace(spec, raw=raw))


def test_held_out_matches_are_rejected_before_preparing_context():
    tables, inputs = fake_inputs()
    with pytest.raises(ValueError, match="8–25"):
        prepare_context_experiment(
            replace(tables, team_matches=tables.team_matches.assign(season_id=26)), inputs
        )


def toy_evaluation(seasons, *, improvement=0, goals=False):
    frame = policy_games().query("season_id in @seasons").copy()
    frame["fold"] = frame.season_id - 10
    frame["outcome"] = np.tile([0, 1, 2], len(seasons))
    frame[["p_home", "p_draw", "p_away"]] = 1 / 3
    frame["log_loss"] = 1.2 - improvement
    frame["rps"] = 0.2
    if goals:
        frame["poisson_nll"] = 1.5 - improvement
        folds, summary = summarize_goals(frame)
        return GoalEvaluationResult(
            frame,
            folds,
            summary,
            pd.DataFrame(
                {"game_id": np.repeat(frame.game_id, 2), "season_id": np.repeat(frame.season_id, 2)}
            ),
        )
    folds, summary = summarize(frame)
    return EvaluationResult(frame, folds, summary, "walk_forward", "synthetic")


def toy_run(configuration, seasons=(11, 18, 25), improvement=0):
    evaluation = toy_evaluation(seasons, improvement=improvement, goals=configuration.route == "P")
    fits = pd.DataFrame(
        {
            "fold": [s - 10 for s in seasons],
            "season_id": seasons,
            "training_seasons": [tuple(range(9, s)) for s in seasons],
            "converged": True,
            "iterations": 2,
            "accepted_by_gradient_check": False,
            "max_abs_gradient": 1e-8,
            "elo_coefficient": 1.0,
        }
    )
    if configuration.route == "P":
        fits["coefficients"] = [
            (0.1, 0.1, 0.2, 0.2, 1.0, *([-0.1] * len(configuration.columns)))
        ] * len(seasons)
    else:
        fits["beta"] = 1.0
        fits["additional_betas"] = [tuple(-0.1 for _ in configuration.columns)] * len(seasons)
        fits["theta1"] = -1.0
        fits["delta"] = 0.0
    return ContextRun(configuration, evaluation, fits, 0.01)


def test_sensitivity_pairing_uses_sqrt_14_and_pilot_artifacts_stay_in_derived(
    tmp_path, monkeypatch
):
    spec = load_context_experiment()
    seasons = tuple(range(12, 26))
    ref = toy_run(ContextConfiguration("O", reference="elo", sensitivity=True), seasons)
    candidate = toy_run(
        ContextConfiguration("O", candidate="value_avg", sensitivity=True),
        seasons,
        improvement=0.01,
    )
    pattern = np.linspace(-0.01, 0.02, 14)
    frame = candidate.evaluation.per_game.copy()
    frame["log_loss"] = frame.season_id.map(dict(zip(seasons, 1.2 - pattern, strict=True)))
    folds, summary = summarize(frame)
    candidate = replace(
        candidate,
        evaluation=replace(candidate.evaluation, per_game=frame, per_fold=folds, summary=summary),
    )
    comparisons, _ = context_comparisons([ref, candidate])
    row = comparisons.loc[comparisons.config_id.eq(candidate.config_id)].iloc[0]
    assert row.se == pytest.approx(pattern.std(ddof=1) / np.sqrt(14))
    assert row.mean_improvement == pytest.approx(pattern.mean())
    assert row.last5_folds == 5
    runs = [
        toy_run(ContextConfiguration("O", reference="elo")),
        toy_run(ContextConfiguration("O", candidate="gd"), improvement=0.01),
        toy_run(ContextConfiguration("P", reference="poisson")),
        toy_run(ContextConfiguration("P", candidate="value_avg"), improvement=0.01),
    ]
    monkeypatch.chdir(tmp_path)
    manifest = save_context_experiment(
        runs, spec, ContextProvenance("a" * 40, "b" * 64, {"code.py": "c" * 64}), stage="pilot"
    )
    meta = verify_calibration(manifest)
    assert meta["configurations"] == 4 and meta["fold_rows"] == 12 and meta["pilot_blind"]
    assert Path("data/derived/exp003/pilot/predictions.parquet").is_file()
    assert not list(manifest.parent.glob("*predictions*.parquet"))
    assert pd.read_parquet(manifest.parent / "fold_scores.parquet").shape[0] == 12
    coefficients = pd.read_parquet(manifest.parent / "coefficients.parquet")
    assert coefficients.loc[coefficients.coefficient.eq("z_gd"), "sign"].eq("negative").all()
    with pytest.raises(FileExistsError):
        save_context_experiment(
            runs, spec, ContextProvenance("a" * 40, "b" * 64, {"code.py": "c" * 64}), stage="pilot"
        )
    Path("data/derived/exp003/pilot/predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_calibration(manifest)


def test_leakage_threshold_stops_without_reporting_the_score(monkeypatch):
    result = toy_evaluation(tuple(range(11, 26)))
    result.summary["log_loss"] = 0.5
    monkeypatch.setattr("ligat.evaluate.context_experiment.evaluate", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="without displaying the score") as error:
        run_context_configuration(
            ContextExperimentInput(policy_games(), pd.DataFrame(), "a" * 64),
            load_context_experiment(),
            ContextConfiguration("O", candidate="gd"),
        )
    assert "0.5" not in str(error.value)


@pytest.fixture(scope="module")
def real_prepared():
    spec = load_context_experiment()
    if not all(Path(entry["path"]).exists() for entry in spec.raw["inputs"].values()):
        pytest.skip("Frozen EXP-003 inputs are local only")
    inputs = load_context_inputs(spec)
    return prepare_context_experiment(
        load_analysis_tables(seasons=list(range(8, 26))), inputs
    ), spec


def test_regression_from_season_8_reproduces_exact_elo(real_prepared):
    prepared, spec = real_prepared
    run = run_context_configuration(
        prepared, spec, ContextConfiguration("O", reference="elo"), regression_training_start=8
    )
    assert abs(float(run.evaluation.summary.log_loss) - 1.020423847325146) <= 1e-9
    assert len(run.evaluation.per_game) == 3721


def test_regression_from_season_8_reproduces_exact_saved_poisson_reference(real_prepared):
    prepared, spec = real_prepared
    path = Path("reports/experiments/EXP-002/full/fold_scores.parquet")
    if not path.exists():
        pytest.skip("Saved EXP-002 folds are local only")
    saved = pd.read_parquet(path)
    saved = saved.loc[saved.config_id.eq("P_ewma5_venue_elo")]
    expected = float(np.average(saved.poisson_nll, weights=saved.rows))
    run = run_context_configuration(
        prepared, spec, ContextConfiguration("P", reference="poisson"), regression_training_start=8
    )
    assert abs(float(run.evaluation.summary.poisson_nll) - expected) <= 1e-9
    assert len(run.evaluation.per_game) == 3721
