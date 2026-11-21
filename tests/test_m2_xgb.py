"""Synthetic contracts and one blind real-fold engineering check for M2 Poisson trees."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.features.prematch_v2 import validate_prematch_v2_frame
from ligat.features.registry import M2_BASE_COLUMNS
from ligat.models.goal_model import TeamGoalModel
from ligat.models.m2_xgb import STAGE_DTYPE, XGBGoalConfig, XGBGoalModel
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare


def config(**kwargs):
    values = dict(
        max_depth=2,
        min_child_weight=1,
        n_estimators=32,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_lambda=1,
        seed=0,
    )
    return XGBGoalConfig(**(values | kwargs))


def sample(n=512, *, constant=False):
    rng = np.random.default_rng(14101)
    index = pd.MultiIndex.from_arrays(
        [np.repeat(np.arange(n // 2), 2), np.tile([20, 10], n // 2)], names=["game_id", "team_id"]
    )
    X = pd.DataFrame(
        {
            "elo_diff": rng.normal(size=n),
            "is_home": np.tile([True, False], n // 2),
            "stage": np.resize(["regular", "upper", "lower"], n),
        },
        index=index,
    )
    lam = np.full(n, 1.3) if constant else np.exp(0.2 + 0.3 * X.elo_diff + 0.1 * X.is_home)
    y = pd.Series(rng.poisson(lam), index=index, name="goals_scored")
    context = pd.DataFrame({"season_id": 14}, index=index)
    return X, y, context


def test_lambda_is_positive_finite_and_preserves_requested_index_without_mutation():
    X, y, context = sample()
    original = X.copy(deep=True)
    model: TeamGoalModel = XGBGoalModel(config())
    assert model.fit(X, y, context=context) is model
    assert model.refit_policy == "per_fold" and model.supports_sample_weight
    shuffled = X.sample(frac=1, random_state=5)
    predictions = model.predict(shuffled, context=context.loc[shuffled.index])
    assert predictions.index.equals(shuffled.index) and predictions.name == "lambda"
    assert np.isfinite(predictions).all() and predictions.gt(0).all()
    pd.testing.assert_frame_equal(X, original)


def test_poisson_mean_is_returned_without_an_extra_exponential():
    X, y, context = sample(4000, constant=True)
    model = XGBGoalModel(config(max_depth=1, n_estimators=64, subsample=1, colsample_bytree=1))
    model.fit(X, y, context=context)
    mean = model.predict(X, context=context).mean()
    assert mean == pytest.approx(1.3, abs=0.06)
    assert abs(mean - np.exp(1.3)) > 1


def test_same_seed_is_bitwise_reproducible_and_different_seed_changes_predictions():
    X, y, context = sample()
    results = []
    for seed in (0, 0, 1):
        model = XGBGoalModel(config(seed=seed)).fit(X, y, context=context)
        results.append(model.predict(X, context=context).to_numpy())
    np.testing.assert_array_equal(results[0], results[1])
    assert not np.array_equal(results[0], results[2])


def test_unit_weights_equal_none_and_nonuniform_weights_change_predictions():
    X, y, context = sample()
    results = []
    for weights in (
        None,
        pd.Series(1.0, index=X.index),
        pd.Series(np.where(y >= 2, 20, 0.05), index=X.index),
    ):
        model = XGBGoalModel(config()).fit(X, y, weights, context=context)
        results.append(model.predict(X, context=context).to_numpy())
    np.testing.assert_array_equal(results[0], results[1])
    assert np.mean(results[2]) > np.mean(results[0]) + 0.2


def test_raw_weight_scale_matches_native_xgboost_without_normalization():
    X, y, context = sample()
    cfg = config(reg_lambda=10)
    weights = pd.Series(np.where(y >= 2, 20, 0.05), index=X.index)
    actual = XGBGoalModel(cfg).fit(X, y, weights, context=context).predict(X, context=context)
    prepared = X.assign(is_home=X.is_home.astype(float), stage=X.stage.astype(STAGE_DTYPE))
    parameters = asdict(cfg)
    parameters.pop("preset")
    parameters["random_state"] = parameters.pop("seed")
    reference = XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        enable_categorical=True,
        **parameters,
    )
    reference.fit(prepared, y, sample_weight=weights)
    np.testing.assert_array_equal(actual.to_numpy(), reference.predict(prepared))
    reference.fit(prepared, y, sample_weight=weights / weights.mean())
    assert not np.array_equal(actual.to_numpy(), reference.predict(prepared))


def test_nan_and_nullable_boolean_features_remain_supported():
    X, y, context = sample()
    X.loc[X.index[::3], "elo_diff"] = np.nan
    X["is_home"] = X.is_home.astype("boolean")
    X.loc[X.index[::7], "is_home"] = pd.NA
    model = XGBGoalModel(config()).fit(X, y, context=context)
    assert model.predict(X, context=context).gt(0).all()
    assert X.elo_diff.isna().any() and X.is_home.isna().any()


def test_fixed_stage_categories_include_unobserved_middle_at_prediction():
    X, y, context = sample()
    assert "middle" not in set(X.stage)
    model = XGBGoalModel(config()).fit(X, y, context=context)
    for stage in STAGE_DTYPE.categories:
        assert model.predict(X.assign(stage=stage), context=context).gt(0).all()
    with pytest.raises(ValueError, match="stage"):
        model.predict(X.assign(stage="fifth"), context=context)
    with pytest.raises(ValueError, match="stage"):
        XGBGoalModel(config()).fit(X.assign(stage="fifth"), y, context=context)


@pytest.mark.parametrize("damage", ["missing", "extra", "order", "dtype"])
def test_prediction_rejects_changed_columns_order_and_numeric_dtype(damage):
    X, y, context = sample()
    model = XGBGoalModel(config()).fit(X, y, context=context)
    changed = {
        "missing": X.drop(columns="is_home"),
        "extra": X.assign(promoted_in=False),
        "order": X.iloc[:, ::-1],
        "dtype": X.astype({"elo_diff": "float32"}),
    }[damage]
    with pytest.raises(ValueError, match="schema"):
        model.predict(changed, context=context)


def test_json_reload_preserves_exact_predictions_and_unobserved_stage(tmp_path):
    X, y, context = sample()
    model = XGBGoalModel(config()).fit(X, y, context=context)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = XGBGoalModel.load(path)
    assert loaded.config == model.config
    for features in (X.iloc[::-1], X.assign(stage="middle")):
        aligned = context.loc[features.index]
        pd.testing.assert_series_equal(
            loaded.predict(features, context=aligned),
            model.predict(features, context=aligned),
            check_exact=True,
        )
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    assert meta["stage_categories"] == ["regular", "upper", "middle", "lower"]
    assert [row[0] for row in meta["features"]] == list(X.columns)


@pytest.mark.parametrize(
    "damage", ["config", "dtype", "order", "stage", "model", "missing", "version"]
)
def test_mismatched_model_or_sidecar_is_rejected_on_load(tmp_path, damage):
    X, y, context = sample()
    path = tmp_path / "model.json"
    XGBGoalModel(config()).fit(X, y, context=context).save(path)
    sidecar = path.with_suffix(".metadata.json")
    meta = json.loads(sidecar.read_text())
    if damage == "config":
        meta["config"]["seed"] = 1
    elif damage == "dtype":
        meta["features"][0][1] = "float32"
    elif damage == "order":
        meta["features"].reverse()
    elif damage == "stage":
        meta["stage_categories"].reverse()
    elif damage == "model":
        XGBGoalModel(config(seed=3)).fit(X, y, context=context).save(tmp_path / "other.json")
        path.write_bytes((tmp_path / "other.json").read_bytes())
    elif damage == "missing":
        del meta["config_id"]
    else:
        meta["xgboost_version"] = "other"
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        XGBGoalModel.load(path)


def test_config_is_frozen_canonical_and_ids_are_stable_across_processes():
    cfg = config()
    assert (
        cfg == config(min_child_weight=1.0)
        and cfg.config_id == config(min_child_weight=1.0).config_id
    )
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 2
    code = (
        "from ligat.models.m2_xgb import XGBGoalConfig; print(XGBGoalConfig(**"
        + repr(asdict(cfg))
        + ").config_id)"
    )
    assert subprocess.check_output([sys.executable, "-c", code], text=True).strip() == cfg.config_id
    assert XGBGoalConfig.library_default().preset == "library_default"
    assert (
        XGBGoalConfig.library_default().config_id
        != replace(XGBGoalConfig.library_default(), preset="grid").config_id
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_depth", 3),
        ("min_child_weight", 10),
        ("n_estimators", 33),
        ("learning_rate", 0.05),
        ("subsample", 1),
        ("colsample_bytree", 1),
        ("reg_lambda", 10),
        ("seed", 1),
    ],
)
def test_every_variable_hyperparameter_changes_config_id(field, value):
    assert replace(config(), **{field: value}).config_id != config().config_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_depth", True),
        ("max_depth", 0),
        ("n_estimators", 1.5),
        ("seed", -1),
        ("learning_rate", 0),
        ("subsample", 1.1),
        ("reg_lambda", np.inf),
        ("min_child_weight", -1),
        ("max_delta_step", 0.8),
        ("preset", "missing"),
        ("preset", "library_default"),
    ],
)
def test_invalid_or_partial_configurations_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})


@pytest.mark.parametrize(
    "damage",
    [
        "target_keys",
        "context_keys",
        "target_nan",
        "target_fraction",
        "weights_keys",
        "weights_negative",
        "weights_zero",
        "weights_nan",
        "infinite_feature",
        "team_id",
    ],
)
def test_bad_training_inputs_fail_before_fitting(damage):
    X, y, context = sample()
    weights = pd.Series(1.0, index=X.index)
    if damage == "target_keys":
        y = y.iloc[::-1]
    elif damage == "context_keys":
        context = context.iloc[::-1]
    elif damage == "target_nan":
        y = y.astype(float)
        y.iloc[0] = np.nan
    elif damage == "target_fraction":
        y = y.astype(float)
        y.iloc[0] = 0.5
    elif damage == "weights_keys":
        weights = weights.iloc[::-1]
    elif damage == "weights_negative":
        weights.iloc[0] = -1
    elif damage == "weights_zero":
        weights[:] = 0
    elif damage == "weights_nan":
        weights.iloc[0] = np.nan
    elif damage == "infinite_feature":
        X.iloc[0, 0] = np.inf
    else:
        X["team_id"] = 1
    with pytest.raises(ValueError):
        XGBGoalModel(config()).fit(X, y, weights, context=context)


def test_unfitted_model_and_failed_refit_cannot_return_stale_predictions(tmp_path):
    X, y, context = sample()
    model = XGBGoalModel(config())
    with pytest.raises(RuntimeError):
        model.predict(X, context=context)
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "model.json")
    model.fit(X, y, context=context)
    with pytest.raises(ValueError):
        model.fit(X, y.iloc[::-1], context=context)
    with pytest.raises(RuntimeError):
        model.predict(X, context=context)


def test_poisson_trees_work_through_judge_with_season_decay(goal_sources):
    data = prepare(goal_sources)
    result = evaluate_team_goals(
        lambda: XGBGoalModel(config(n_estimators=8)),
        data,
        feature_columns=("elo_diff",),
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        half_life=2,
    )
    assert result.refit_policy == "per_fold" and len(result.per_fold) == 1
    assert result.per_team["lambda"].gt(0).all()
    assert result.per_team.half_life.eq(2).all() and result.per_team.season_id.eq(15).all()


@pytest.mark.real_data
def test_one_real_fold_runs_with_base_features_and_records_fit_time_without_scores(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Published prematch v2 is local only")
    seasons = tuple(range(8, 26))
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    validate_prematch_v2_frame(prematch)
    data = prepare_goal_data(
        prematch, real_tables.match_table, feature_columns=M2_BASE_COLUMNS, seasons=seasons
    )
    elapsed = []

    class TimedModel(XGBGoalModel):
        def fit(self, X, y, sample_weight=None, *, context):
            started = perf_counter()
            super().fit(X, y, sample_weight, context=context)
            elapsed.append(perf_counter() - started)
            return self

    cfg = config(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8)
    result = evaluate_team_goals(
        lambda: TimedModel(cfg),
        data,
        feature_columns=M2_BASE_COLUMNS,
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        blind=True,
    )
    rates = result.per_team["lambda"]
    assert len(elapsed) == 1 and elapsed[0] > 0
    assert set(result.per_team.season_id) == {15}
    assert rates.between(0.2, 5).all()
    print(
        json.dumps(
            {
                "season_id": 15,
                "fit_seconds": elapsed[0],
                "lambda_min": float(rates.min()),
                "lambda_max": float(rates.max()),
                "rows": len(rates),
                "config_id": cfg.config_id,
            }
        )
    )
