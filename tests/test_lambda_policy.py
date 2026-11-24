"""The M3 policy is explicit, causal, and can reuse evaluation units read-only."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ligat.evaluate.goal_inputs import GoalData
from ligat.evaluate.goals_experiment import GoalUnit
from ligat.evaluate.goals_runner import goal_units_status, run_goal_unit, run_goal_units
from ligat.evaluate.lambda_layer import LambdaLayerConfig, lambda_layer_status, write_lambda_layer
from ligat.models.m2_maher import MAHER_COLUMNS, MaherGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import LoggedMean, provenance


class DailyMean(LoggedMean):
    def __init__(self):
        super().__init__("per_matchday")


def daily_factory(unit):
    return DailyMean()


def policy_config(policy):
    return LambdaLayerConfig("maher", asdict(MaherGoalConfig()), MAHER_COLUMNS, policy)


def test_policy_is_required_and_bound_to_the_layer_signature():
    with pytest.raises(TypeError):
        LambdaLayerConfig("maher", asdict(MaherGoalConfig()), MAHER_COLUMNS)
    with pytest.raises(ValueError, match="refit_policy"):
        policy_config("invalid")
    a, b = policy_config("as_evaluated"), policy_config("prior_seasons_only")
    assert a.config_id != b.config_id
    assert a.units()[0].unit_id != b.units()[0].unit_id
    for unit in a.units():
        expected = GoalUnit("maher", asdict(MaherGoalConfig()), MAHER_COLUMNS, unit.season)
        assert unit.unit_id == expected.unit_id


def test_as_evaluated_reuses_completed_full_units_without_fit_or_mutation(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, MAHER_COLUMNS)
    cfg, version = policy_config("as_evaluated"), provenance(data)
    evaluation, layer = tmp_path / "evaluation", tmp_path / "layer"
    run_goal_units(
        cfg.units(), data, version, evaluation, n_jobs=1, stage="full", model_factory=daily_factory
    )
    before = {p: p.read_bytes() for p in evaluation.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner._model", lambda *a: pytest.fail("Export refitted a full unit")
    )
    frame = write_lambda_layer(cfg, data, layer, evaluation_root=evaluation, provenance=version)
    assert frame.trained_on_current_season.any()
    assert frame.train_end_season.lt(frame.season_id).all()
    assert frame.train_end_date.lt(frame.as_of).all()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert (
        lambda_layer_status(cfg, data, layer, evaluation_root=evaluation, provenance=version)[
            "status"
        ]
        == "complete"
    )
    original = Path.read_bytes
    import ligat.evaluate.lambda_layer as module

    own = Path(module.__file__).resolve()
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: original(p) + (b"\n# changed transform" if p.resolve() == own else b""),
    )
    assert all(
        s["status"] == "complete"
        for s in goal_units_status(cfg.units(), data, version, evaluation, stage="full")
    )
    assert (
        lambda_layer_status(cfg, data, layer, evaluation_root=evaluation, provenance=version)[
            "status"
        ]
        == "outdated"
    )


def test_incomplete_evaluation_is_not_trained_by_the_exporter(goal_sources, tmp_path, monkeypatch):
    data = prepare(goal_sources, MAHER_COLUMNS)
    cfg = policy_config("as_evaluated")
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner._model",
        lambda *a: pytest.fail("Export trained missing D3 unit"),
    )
    with pytest.raises(ValueError, match="completed evaluation units"):
        write_lambda_layer(
            cfg,
            data,
            tmp_path / "layer",
            evaluation_root=tmp_path / "evaluation",
            provenance=provenance(data),
        )
    with pytest.raises(ValueError, match="evaluation_root"):
        write_lambda_layer(cfg, data, tmp_path / "layer")


@pytest.mark.slow
def test_actual_maher_as_evaluated_uses_only_results_before_as_of(goal_sources, tmp_path):
    prematch, games = goal_sources
    data = prepare((prematch.assign(promoted_in=False), games), (*MAHER_COLUMNS, "promoted_in"))
    unit = policy_config("as_evaluated").units()[0]
    original = run_goal_unit(
        unit, data, provenance(data), tmp_path / "a", stage="synthetic"
    ).evaluation.per_team
    cutoff = sorted(original.as_of.unique())[1]
    future_y = data.y.copy()
    future_y.loc[data.context.season_id.eq(11) & data.context.date.ge(cutoff)] += 1
    future = GoalData(data.X, future_y, data.context)
    altered = run_goal_unit(
        unit, future, provenance(future), tmp_path / "b", stage="synthetic"
    ).evaluation.per_team
    mask = original.as_of.le(cutoff)
    np.testing.assert_array_equal(original.loc[mask, "lambda"], altered.loc[mask, "lambda"])
    past_y = data.y.copy()
    past_y.loc[data.context.season_id.eq(11) & data.context.date.lt(cutoff)] += 1
    past = GoalData(data.X, past_y, data.context)
    changed = run_goal_unit(
        unit, past, provenance(past), tmp_path / "c", stage="synthetic"
    ).evaluation.per_team
    assert not np.allclose(
        original.loc[original.as_of.eq(cutoff), "lambda"],
        changed.loc[changed.as_of.eq(cutoff), "lambda"],
    )
    assert original.train_max_date.lt(original.as_of).all()
