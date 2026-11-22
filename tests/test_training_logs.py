"""Training journal parity, native metric arithmetic and bound persistence."""

import gzip
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from ligat.models.m2_catboost import CatBoostGoalModel
from ligat.models.m2_glm import GLMGoalConfig, GLMGoalModel
from ligat.models.m2_maher import MaherGoalConfig, MaherGoalModel, day_weights
from ligat.models.m2_xgb import XGBGoalModel
from ligat.models.training_log import write_training_log
from tests.test_m2_catboost import config as cat_config
from tests.test_m2_maher import synthetic_league
from tests.test_m2_maher_prior import prior
from tests.test_m2_maher_season_reference import current_roster
from tests.test_m2_xgb import config as xgb_config
from tests.test_m2_xgb import sample


def catboost_training_logarithm(rates):
    """Vectorized mathematical FastLogf approximation used by CatBoost 1.2.10.

    Source: library/cpp/fast_log/fast_log.h in the CatBoost v1.2.10 source.
    This is test arithmetic only; production records the original native curve.
    """
    bits = rates.astype(np.float32).view(np.uint32)
    mantissa = ((bits & 0x007FFFFF) | 0x3F000000).view(np.float32)
    log2 = (
        bits.astype(np.float32) * np.float32(2**-23)
        - np.float32(124.22551499)
        - np.float32(1.498030302) * mantissa
        - np.float32(1.72587999) / (np.float32(0.3520887068) + mantissa)
    )
    return np.float32(0.69314718) * log2


@pytest.mark.parametrize("backend", ["xgb", "catboost", "glm", "maher"])
@pytest.mark.parametrize("weighted", [False, True])
def test_training_journal_is_observational_and_survives_save(backend, weighted, tmp_path):
    X, y, context = sample()
    kwargs = {}
    weights = pd.Series(np.linspace(0.1, 2, len(y)), index=y.index) if weighted else None
    if backend == "xgb":

        def factory(enabled):
            return XGBGoalModel(xgb_config(), record_training_log=enabled)

        suffix = ".json"
    elif backend == "catboost":

        def factory(enabled):
            return CatBoostGoalModel(cat_config(), record_training_log=enabled)

        suffix = ".cbm"
    elif backend == "glm":

        def factory(enabled):
            return GLMGoalModel(GLMGoalConfig("r1"), record_training_log=enabled)

        suffix = ".json"
    else:
        X, y, context, *_ = synthetic_league()
        weights = None
        kwargs = {"as_of": context.as_of.max() + pd.Timedelta(days=1)}

        def factory(enabled):
            return MaherGoalModel(MaherGoalConfig(), record_training_log=enabled)

        suffix = ".json"
    on, off = factory(True), factory(False)
    on.fit(X, y, weights, context=context, **kwargs)
    off.fit(X, y, weights, context=context, **kwargs)
    rates = on.predict(X, context=context)
    np.testing.assert_array_equal(rates, off.predict(X, context=context))
    if backend == "xgb":
        assert on._estimator.get_booster().get_dump() == off._estimator.get_booster().get_dump()
    if backend == "catboost":
        assert on.actual_params == off.actual_params
        np.testing.assert_array_equal(
            on._estimator.get_leaf_values(), off._estimator.get_leaf_values()
        )
    log = on.training_log
    assert log["dataset"] == "training" and log["converged"]
    assert len(log["train_loss"]) == log["iterations"]
    assert not off.training_log["train_loss"]
    terms = rates.to_numpy() - y.to_numpy() * np.log(rates.to_numpy())
    if backend == "xgb" or (backend == "glm" and weighted):
        terms += gammaln(y.to_numpy() + 1)
    if backend == "catboost":
        terms = rates.to_numpy() - y.to_numpy() * catboost_training_logarithm(rates.to_numpy())
    expected = np.average(terms, weights=weights)
    assert log["train_loss"][-1] == pytest.approx(expected, abs=2e-6)
    path = tmp_path / (backend + suffix)
    on.save(path)
    loaded = type(on).load(path)
    assert loaded.training_log == log
    np.testing.assert_array_equal(rates, loaded.predict(X, context=context))
    write_training_log(on, tmp_path / "manual.json.gz")
    assert json.loads(gzip.decompress((tmp_path / "manual.json.gz").read_bytes())) == log
    log["train_loss"].clear()
    assert on.training_log["train_loss"]


def test_synthetic_training_log_overhead():
    X, y, context = sample()
    MX, my, mc, *_ = synthetic_league()
    report = {}
    for backend, model, cfg, features, targets, ctx, kwargs in (
        ("xgb", XGBGoalModel, xgb_config(), X, y, context, {}),
        ("catboost", CatBoostGoalModel, cat_config(), X, y, context, {}),
        (
            "maher",
            MaherGoalModel,
            MaherGoalConfig(),
            MX,
            my,
            mc,
            {"as_of": mc.as_of.max() + pd.Timedelta(days=1)},
        ),
    ):
        times = {True: [], False: []}
        for repeat in range(4):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                start = perf_counter()
                model(cfg, record_training_log=enabled).fit(
                    features, targets, context=ctx, **kwargs
                )
                times[enabled].append(perf_counter() - start)
        report[backend] = {
            "without_seconds": median(times[False]),
            "with_seconds": median(times[True]),
            "overhead_percent": 100 * (median(times[True]) / median(times[False]) - 1),
        }
    target = Path("data/reports/T-014") / (
        os.environ.get("LIGAT_AUDIT_ROUND", "C5") + "_synthetic_overhead.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def test_catboost_training_log_writes_nothing(monkeypatch, tmp_path):
    X, y, context = sample()
    monkeypatch.chdir(tmp_path)
    log = CatBoostGoalModel(cat_config()).fit(X, y, context=context).training_log
    assert log["dataset"] == "training"
    assert list(tmp_path.iterdir()) == []


def test_maher_iteration_limit_does_not_publish_converged_log(monkeypatch):
    X, y, context, *_ = synthetic_league()
    monkeypatch.setattr("ligat.models.m2_maher._MAX_ITERATIONS", 1)
    model = MaherGoalModel(MaherGoalConfig())
    with pytest.raises(RuntimeError, match="did not converge"):
        model.fit(X, y, context=context, as_of=context.as_of.max() + pd.Timedelta(days=1))
    with pytest.raises(RuntimeError, match="Fit or load"):
        _ = model.training_log


def test_maher_last_training_loss_contains_weighted_likelihood_and_prior():
    X, y, context, *_ = synthetic_league()
    state, frozen = current_roster(), prior()
    as_of = state.start(12).tz_convert("Asia/Jerusalem")
    model = MaherGoalModel(MaherGoalConfig(0.5, frozen), roster=state).fit(
        X, y, context=context, as_of=as_of
    )
    params = model.parameters
    eta = (
        params.log_alpha.reindex(X.team_id).to_numpy()
        + params.log_beta.reindex(X.opponent_id).to_numpy()
        + model.log_gamma * X.is_home.to_numpy()
    )
    weights = day_weights(context.date, as_of, 0.5)
    penalty = 0.0
    league = params.loc[list(state.teams(12))]
    for team in state.promoted(12):
        da = params.loc[team, "log_alpha"] - league.log_alpha.mean() - frozen.a_star
        db = params.loc[team, "log_beta"] - league.log_beta.mean() - frozen.b_star
        penalty += 0.5 * model.penalty_strength * (da**2 + db**2)
    expected = (weights @ (np.exp(eta) - y.to_numpy() * eta) + penalty) / weights.sum()
    assert model.training_log["final_loss"] == pytest.approx(expected, abs=1e-12)
    assert model.training_log["train_loss"][-1] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("weighted", [True, False])
def test_xgboost_reuses_training_dmatrix_for_training_metric(monkeypatch, weighted):
    from xgboost import XGBRegressor

    X, y, context = sample()
    weights = pd.Series(np.linspace(0.1, 2, len(y)), index=y.index) if weighted else None
    original = XGBRegressor._create_dmatrix
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(XGBRegressor, "_create_dmatrix", counted)
    XGBGoalModel(xgb_config()).fit(X, y, weights, context=context)
    assert len(calls) == 1
