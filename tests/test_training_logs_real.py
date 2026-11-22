"""Blind fold-15 training journals, timing overhead and legacy callback parity."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.features.registry import M2_BASE_COLUMNS
from ligat.models.form_models import fit_poisson
from ligat.models.m2_catboost import CatBoostGoalConfig, CatBoostGoalModel
from ligat.models.m2_glm import GLMGoalConfig, GLMGoalModel, glm_design
from ligat.models.m2_maher import MAHER_COLUMNS, MaherGoalConfig, MaherGoalModel
from ligat.models.m2_maher_prior import load_promoted_prior, season_roster
from ligat.models.m2_xgb import XGBGoalConfig, XGBGoalModel
from ligat.models.training_log import journal_bytes
from tests.test_m2_maher_prior import real_prior_inputs as real_prior_inputs


def report_path(name):
    target = Path("data/reports/T-014") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@pytest.mark.real_data
def test_real_training_logs_and_overhead_are_blind(real_prior_inputs, real_tables):
    _, frame = real_prior_inputs
    columns = tuple(dict.fromkeys((*M2_BASE_COLUMNS, *MAHER_COLUMNS)))
    data = prepare_goal_data(
        frame, real_tables.match_table, feature_columns=columns, seasons=tuple(range(8, 26))
    )
    roster = season_roster(data)
    factories = {
        "xgb": lambda enabled: XGBGoalModel(
            XGBGoalConfig(4, 10, 400, 0.05, 0.8, 0.8, 1), record_training_log=enabled
        ),
        "catboost": lambda enabled: CatBoostGoalModel(
            CatBoostGoalConfig(4, 3, 300, 0.05), record_training_log=enabled
        ),
        "glm": lambda enabled: GLMGoalModel(GLMGoalConfig("base", 9), record_training_log=enabled),
        "maher": lambda enabled: MaherGoalModel(
            MaherGoalConfig(1, load_promoted_prior()), roster=roster, record_training_log=enabled
        ),
    }
    report = {}
    for backend, build in factories.items():
        fits, logs = [], []

        class Observed:
            def __init__(self, enabled, builder=build, timings=fits, records=logs):
                self.timings, self.records = timings, records
                self.model = builder(enabled)
                self.refit_policy = self.model.refit_policy
                self.supports_sample_weight = self.model.supports_sample_weight
                self.requires_as_of = getattr(self.model, "requires_as_of", False)

            def fit(self, X, y, sample_weight=None, *, context, **kwargs):
                start = perf_counter()
                if self.requires_as_of:
                    self.model.fit(X, y, context=context, **kwargs)
                else:
                    self.model.fit(X, y, sample_weight, context=context)
                self.timings.append(perf_counter() - start)
                self.records.append(self.model.training_log)
                return self

            def predict(self, X, *, context):
                return self.model.predict(X, context=context)

        features = MAHER_COLUMNS if backend == "maher" else M2_BASE_COLUMNS
        results, seconds, recorded = [], [], []
        for enabled in (False, True):
            fits.clear()
            logs.clear()
            with threadpool_limits(limits=1):
                results.append(
                    evaluate_team_goals(
                        lambda e=enabled: Observed(e),
                        data,
                        feature_columns=features,
                        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
                        blind=True,
                    )
                )
            seconds.append(sum(fits))
            if enabled:
                recorded = list(logs)
        np.testing.assert_array_equal(results[0].per_team["lambda"], results[1].per_team["lambda"])
        for log in recorded:
            assert len(log["train_loss"]) == log["iterations"]
        report[backend] = {
            "compressed_bytes": len(journal_bytes(recorded, compressed=True)),
            "without_seconds": seconds[0],
            "with_seconds": seconds[1],
            "overhead_percent": 100 * (seconds[1] / seconds[0] - 1),
            "fits": len(recorded),
            "curves": [
                {
                    "length": len(log["train_loss"]),
                    "first": log["train_loss"][0],
                    "last": log["train_loss"][-1],
                    "monotone": bool(np.all(np.diff(log["train_loss"]) <= 1e-12)),
                }
                for log in recorded
            ],
        }
    report_path("C5_real_training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


@pytest.mark.real_data
def test_real_closed_ruler_coefficients_are_bitwise_unchanged_by_callback(real_prior_inputs):
    _, frame = real_prior_inputs
    # M0 features only; no candidate validation target or score is used.
    data, _ = real_prior_inputs
    for recipe, start in (("r0", 8), ("r1", 8), ("ruler", 8), ("ruler", 9)):
        selected = data.context.season_id.between(start, 14)
        X = frame.loc[data.X.index[selected]]
        design = glm_design(X, GLMGoalConfig(recipe, start)).to_numpy()
        goals = data.y.loc[selected].to_numpy()
        losses = []
        with threadpool_limits(limits=1):
            a = fit_poisson(design, goals)
            b = fit_poisson(design, goals, callback=losses.append)
        assert a == b
        assert len(losses) == b.iterations


@pytest.mark.real_data
@pytest.mark.slow
def test_warmed_alternating_heavy_fold15_overhead(real_prior_inputs, real_tables):
    """Five paired, alternating measurements after one excluded warm-up per config."""
    _, frame = real_prior_inputs
    columns = tuple(dict.fromkeys((*M2_BASE_COLUMNS, *MAHER_COLUMNS)))
    data = prepare_goal_data(
        frame, real_tables.match_table, feature_columns=columns, seasons=tuple(range(8, 26))
    )
    roster = season_roster(data)
    configurations = (
        ("xgb", XGBGoalModel, XGBGoalConfig(8, 1, 1000, 0.05, 0.8, 0.8, 1), {}),
        ("catboost_default", CatBoostGoalModel, CatBoostGoalConfig.library_default(), {}),
        ("catboost_grid", CatBoostGoalModel, CatBoostGoalConfig(8, 1, 1000, 0.05), {}),
        ("glm", GLMGoalModel, GLMGoalConfig("base", 9), {}),
        ("maher", MaherGoalModel, MaherGoalConfig(0.5, load_promoted_prior()), {"roster": roster}),
    )
    report = {}
    for backend, constructor, config, options in configurations:
        timings = []

        class Timed:
            def __init__(self, enabled, cls=constructor, cfg=config, opts=options, times=timings):
                self.model = cls(cfg, record_training_log=enabled, **opts)
                self.times = times
                self.refit_policy = self.model.refit_policy
                self.supports_sample_weight = self.model.supports_sample_weight
                self.requires_as_of = getattr(self.model, "requires_as_of", False)

            def fit(self, X, y, sample_weight=None, *, context, **kwargs):
                started = perf_counter()
                if self.requires_as_of:
                    self.model.fit(X, y, context=context, **kwargs)
                else:
                    self.model.fit(X, y, sample_weight, context=context)
                self.times.append(perf_counter() - started)
                return self

            def predict(self, X, *, context):
                return self.model.predict(X, context=context)

        def run(enabled, times=timings, name=backend, wrapper=Timed):
            times.clear()
            with threadpool_limits(limits=1):
                result = evaluate_team_goals(
                    lambda: wrapper(enabled),
                    data,
                    feature_columns=MAHER_COLUMNS if name == "maher" else M2_BASE_COLUMNS,
                    policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
                    blind=True,
                )
            return sum(times), result.per_team["lambda"].to_numpy()

        run(True)
        samples = {True: [], False: []}
        for pair in range(5):
            rates = {}
            for enabled in (True, False) if pair % 2 == 0 else (False, True):
                seconds, values = run(enabled)
                samples[enabled].append(seconds)
                rates[enabled] = values
            np.testing.assert_array_equal(rates[True], rates[False])
        entry = {}
        for enabled in (True, False):
            values = samples[enabled]
            entry["with_log" if enabled else "without_log"] = {
                "samples": values,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            }
        entry["overhead_percent"] = 100 * (np.median(samples[True]) / np.median(samples[False]) - 1)
        entry["label"] = config.label
        report[backend] = entry
        report_path("C5_overhead.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({backend: entry}), flush=True)
