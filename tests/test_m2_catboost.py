"""CatBoost Poisson contracts, native parity, persistence and blind real-fold timing."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.features.prematch_v2 import validate_prematch_v2_frame
from ligat.features.registry import M2_BASE_COLUMNS
from ligat.models.goal_model import TeamGoalModel
from ligat.models.m2_catboost import CatBoostGoalConfig, CatBoostGoalModel
from ligat.models.m2_identity import IDENTITY_COLUMNS, TeamIdentity
from ligat.models.m2_xgb import STAGE_DTYPE, XGBGoalConfig, XGBGoalModel, _features
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_m2_identity import identity_rows as identity_rows
from tests.test_m2_xgb import sample


def config(**kwargs):
    return CatBoostGoalConfig(
        **(dict(depth=2, l2_leaf_reg=3, iterations=32, learning_rate=0.1) | kwargs)
    )


@pytest.mark.slow
def test_known_poisson_mean_is_lambda_without_missing_or_double_exponent():
    X, y, context = sample(4000, constant=True)
    model = CatBoostGoalModel(config(iterations=120)).fit(X, y, context=context)
    rates = model.predict(X, context=context)
    assert rates.mean() == pytest.approx(1.3, abs=0.07)
    assert rates.index.equals(X.index) and rates.name == "lambda"
    assert np.isfinite(rates).all() and rates.gt(0).all()
    contract: TeamGoalModel = model
    assert contract.refit_policy == "per_fold" and contract.supports_sample_weight


def test_native_poisson_parity_categories_boolean_and_numeric_nan(tmp_path):
    X, y, context = sample()
    X.iloc[0, 0] = np.nan
    model = CatBoostGoalModel(config()).fit(X, y, context=context)
    prepared, _ = _features(X)
    assert np.isnan(prepared.iloc[0, 0])
    assert set(prepared.is_home) == {0.0, 1.0}
    assert prepared.stage.dtype == STAGE_DTYPE
    native = CatBoostRegressor(
        depth=2,
        l2_leaf_reg=3,
        iterations=32,
        learning_rate=0.1,
        random_seed=0,
        loss_function="Poisson",
        task_type="CPU",
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ).fit(prepared, y, cat_features=["stage"])
    rates = model.predict(X, context=context)
    np.testing.assert_array_equal(rates, native.predict(prepared))
    np.testing.assert_array_equal(rates, native.predict(prepared, prediction_type="Exponent"))
    np.testing.assert_allclose(
        rates, np.exp(native.predict(prepared, prediction_type="RawFormulaVal"))
    )
    path = tmp_path / "stage.cbm"
    model.save(path)
    restored = CatBoostGoalModel.load(path)
    pd.testing.assert_series_equal(rates, restored.predict(X, context=context))


def test_same_seed_is_exact_and_unit_weights_equal_unweighted():
    X, y, context = sample()
    a = CatBoostGoalModel(config(seed=11)).fit(X, y, context=context)
    b = CatBoostGoalModel(config(seed=11)).fit(X, y, context=context)
    c = CatBoostGoalModel(config(seed=11)).fit(X, y, pd.Series(1.0, index=X.index), context=context)
    pd.testing.assert_series_equal(a.predict(X, context=context), b.predict(X, context=context))
    pd.testing.assert_series_equal(a.predict(X, context=context), c.predict(X, context=context))
    weights = pd.Series(np.where(y > 1, 5.0, 0.2), index=X.index)
    d = CatBoostGoalModel(config(seed=11)).fit(X, y, weights, context=context)
    assert not np.array_equal(a.predict(X, context=context), d.predict(X, context=context))


def test_all_four_stages_and_middle_absent_from_training():
    X, y, context = sample()
    model = CatBoostGoalModel(config()).fit(X, y, context=context)
    X["stage"] = np.resize(STAGE_DTYPE.categories, len(X))
    assert model.predict(X, context=context).gt(0).all()
    assert (
        CatBoostGoalModel(config())
        .fit(X, y, context=context)
        .predict(X, context=context)
        .gt(0)
        .all()
    )
    X.loc[X.index[0], "stage"] = None
    with pytest.raises(ValueError, match="categorical values"):
        model.predict(X, context=context)


def test_shared_identity_is_identical_to_xgboost_and_survives_reload(identity_rows, tmp_path):
    X, y, context, vocabulary = identity_rows
    train = context.season_id.lt(13)
    model = CatBoostGoalModel(config(), team_ids=vocabulary).fit(
        X.loc[train], y.loc[train], context=context.loc[train]
    )
    xgb = XGBGoalModel(XGBGoalConfig.library_default(), team_ids=vocabulary).fit(
        X.loc[train], y.loc[train], context=context.loc[train]
    )
    assert isinstance(model.identity, TeamIdentity) and model.identity == xgb.identity
    encoded = model.identity.transform(X, context=context)
    pd.testing.assert_frame_equal(encoded, xgb.identity.transform(X, context=context))
    assert all(encoded[c].dtype == model.identity.dtype for c in IDENTITY_COLUMNS)
    assert "unknown" in encoded.team_id.cat.categories
    path = tmp_path / "identity.cbm"
    model.save(path)
    loaded = CatBoostGoalModel.load(path)
    assert loaded.identity == model.identity
    pd.testing.assert_series_equal(
        model.predict(X, context=context), loaded.predict(X, context=context)
    )
    with pytest.raises(ValueError, match="opt-in"):
        CatBoostGoalModel(config()).fit(X, y, context=context)


def test_library_default_omits_tunables_and_records_complete_actual_parameters(tmp_path):
    X, y, context = sample(128)
    X = X.drop(columns="stage")
    model = CatBoostGoalModel(CatBoostGoalConfig.library_default())
    assert not {"depth", "l2_leaf_reg", "iterations", "learning_rate"} & set(
        model._new_estimator().get_params()
    )
    model.fit(X, y, context=context)
    actual = model.actual_params
    assert actual == model._estimator.get_all_params()
    assert actual["learning_rate"] == pytest.approx(0.03)
    assert actual["depth"] == 6 and actual["l2_leaf_reg"] == 3
    assert actual["iterations"] == 1000 and actual["random_seed"] == 0
    assert actual["loss_function"] == "Poisson" and actual["task_type"] == "CPU"
    actual["depth"] = -100
    assert model.actual_params["depth"] == 6
    path = tmp_path / "default.cbm"
    model.save(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    assert metadata["actual_params"] == model.actual_params
    loaded = CatBoostGoalModel.load(path)
    assert loaded.actual_params == model.actual_params
    pd.testing.assert_series_equal(
        model.predict(X, context=context), loaded.predict(X, context=context)
    )
    print(json.dumps({"library_default_actual": model.actual_params}))


@pytest.mark.parametrize("damage", ["config", "actual_params", "features", "model", "sidecar"])
def test_persistence_rejects_mismatched_sidecar_or_native_model(tmp_path, damage):
    X, y, context = sample()
    model = CatBoostGoalModel(config()).fit(X, y, context=context)
    path = tmp_path / "model.cbm"
    model.save(path)
    sidecar = path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    if damage == "config":
        metadata["config"]["seed"] = 7
    elif damage == "actual_params":
        metadata["actual_params"]["learning_rate"] = 0.5
    elif damage == "features":
        metadata["features"][0][1] = "int64"
    elif damage == "sidecar":
        metadata = []
    else:
        CatBoostGoalModel(config(seed=2)).fit(X, y, context=context).save(tmp_path / "other.cbm")
        path.write_bytes((tmp_path / "other.cbm").read_bytes())
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        CatBoostGoalModel.load(path)


def test_config_ids_labels_normalization_and_frozen_state():
    cfg = config()
    assert cfg == config(depth=np.int64(2), l2_leaf_reg=3.0)
    assert cfg.config_id == CatBoostGoalConfig(**asdict(cfg)).config_id
    assert cfg.config_id != replace(cfg, seed=1).config_id
    assert cfg.config_id != XGBGoalConfig.library_default().config_id
    assert cfg.config_id.startswith("catboost-grid-")
    assert cfg.label == "catboost · d2 · l2=3 · it32 · lr0.1 · seed0"
    assert CatBoostGoalConfig.library_default().label == "ברירת המחדל של CatBoost"
    assert CatBoostGoalConfig.library_default(seed=2).label.endswith("seed2")
    assert cfg.label != replace(cfg, learning_rate=0.10000000001).label
    code = (
        "from ligat.models.m2_catboost import CatBoostGoalConfig; "
        "print(CatBoostGoalConfig(2, 3, 32, 0.1).config_id)"
    )
    assert subprocess.check_output([sys.executable, "-c", code], text=True).strip() == cfg.config_id
    with pytest.raises(FrozenInstanceError):
        cfg.depth = 3


@pytest.mark.parametrize(
    "changes",
    [
        {"depth": None},
        {"depth": 0},
        {"depth": 17},
        {"depth": True},
        {"iterations": 0},
        {"l2_leaf_reg": -1},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"preset": "other"},
        {"preset": "library_default"},
    ],
)
def test_invalid_or_partial_configs_are_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize(
    "damage",
    [
        "target_keys",
        "context_keys",
        "target_fraction",
        "target_nan",
        "weights_keys",
        "weights_nan",
        "weights_negative",
        "weights_zero",
        "zero_goals",
        "feature_inf",
    ],
)
def test_bad_training_inputs_fail_before_native_fit(damage):
    X, y, context = sample()
    weights = pd.Series(1.0, index=X.index)
    if damage == "target_keys":
        y = y.iloc[::-1]
    elif damage == "context_keys":
        context = context.iloc[::-1]
    elif damage in ("target_fraction", "target_nan"):
        y = y.astype(float)
        y.iloc[0] = 0.5 if damage == "target_fraction" else np.nan
    elif damage == "weights_keys":
        weights = weights.iloc[::-1]
    elif damage == "weights_nan":
        weights.iloc[0] = np.nan
    elif damage == "weights_negative":
        weights.iloc[0] = -1
    elif damage == "weights_zero":
        weights[:] = 0
    elif damage == "zero_goals":
        y[:] = 0
    else:
        X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        CatBoostGoalModel(config()).fit(X, y, weights, context=context)


def test_unfitted_failed_refit_and_prediction_schema_are_guarded(tmp_path):
    X, y, context = sample()
    model = CatBoostGoalModel(config())
    for operation in (
        lambda: model.predict(X, context=context),
        lambda: model.save(tmp_path / "model.cbm"),
        lambda: model.actual_params,
    ):
        with pytest.raises(RuntimeError):
            operation()
    model.fit(X, y, context=context)
    for changed in (X.iloc[:, ::-1], X.assign(is_home=X.is_home.astype(int))):
        with pytest.raises(ValueError, match="schema"):
            model.predict(changed, context=context)
    with pytest.raises(ValueError, match="align"):
        model.predict(X, context=context.iloc[::-1])
    with pytest.raises(ValueError):
        model.fit(X, y.iloc[::-1], context=context)
    with pytest.raises(RuntimeError):
        model.predict(X, context=context)
    with pytest.raises(RuntimeError):
        _ = model.actual_params


def test_catboost_through_judge_with_half_life(goal_sources):
    data = prepare(goal_sources)
    result = evaluate_team_goals(
        lambda: CatBoostGoalModel(config(iterations=8)),
        data,
        feature_columns=("elo_diff",),
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        half_life=2,
    )
    assert result.refit_policy == "per_fold" and len(result.per_fold) == 1
    assert result.per_team["lambda"].gt(0).all()
    assert result.per_team.half_life.eq(2).all() and result.per_team.season_id.eq(15).all()


@pytest.mark.real_data
def test_one_real_fold_reports_time_lambda_range_and_training_goal_mean_without_scores(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Published prematch v2 is local only")
    seasons = tuple(range(8, 26))
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    validate_prematch_v2_frame(prematch)
    data = prepare_goal_data(
        prematch, real_tables.match_table, feature_columns=M2_BASE_COLUMNS, seasons=seasons
    )
    reports = []

    class TimedModel(CatBoostGoalModel):
        def fit(self, X, y, sample_weight=None, *, context):
            started = perf_counter()
            super().fit(X, y, sample_weight, context=context)
            reports.append(
                {"fit_seconds": perf_counter() - started, "training_goal_mean": float(y.mean())}
            )
            return self

    cfg = config(depth=4, iterations=300, learning_rate=0.05)
    result = evaluate_team_goals(
        lambda: TimedModel(cfg),
        data,
        feature_columns=M2_BASE_COLUMNS,
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
    )
    rates = result.per_team["lambda"]
    assert len(reports) == 1 and reports[0]["fit_seconds"] > 0
    assert set(result.per_team.season_id) == {15} and rates.between(0.2, 5).all()
    print(
        json.dumps(
            reports[0]
            | {
                "season_id": 15,
                "lambda_min": float(rates.min()),
                "lambda_max": float(rates.max()),
                "lambda_mean": float(rates.mean()),
                "rows": len(rates),
                "config_id": cfg.config_id,
            }
        )
    )
