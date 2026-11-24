"""Derived curve caches validate C5 evidence and fitting remains entirely in memory."""

import builtins
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from ligat.evaluate.goals_experiment import GoalUnit, load_goal_experiment
from ligat.evaluate.goals_runner import run_goal_unit, run_goal_units
from ligat.models.m2_catboost import CatBoostGoalConfig, CatBoostGoalModel
from ligat.models.m2_glm import GLMGoalConfig, GLMGoalModel
from ligat.models.m2_maher import MaherGoalConfig, MaherGoalModel
from ligat.models.m2_xgb import XGBGoalConfig, XGBGoalModel
from ligat.viz import goal_training_curves as curves
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import LoggedMean, fake_factory, provenance, units


def test_curve_cache_checks_sources_and_preserves_float32(goal_sources, tmp_path, monkeypatch):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    run_goal_units(units(), data, version, tmp_path, n_jobs=1, model_factory=fake_factory)
    spec = replace(
        load_goal_experiment(), raw=load_goal_experiment().raw | {"output": str(tmp_path)}
    )
    monkeypatch.setattr(curves, "load_goal_inputs", lambda spec: data)
    monkeypatch.setattr(curves, "goal_provenance", lambda *a, **k: version)
    monkeypatch.setattr(curves, "pilot_units", lambda spec: units())
    original_status = curves.goal_units_status
    verifications = []

    def verified(*args, **kwargs):
        verifications.append(True)
        return original_status(*args, **kwargs)

    monkeypatch.setattr(curves, "goal_units_status", verified)
    first = curves.training_curves(spec)
    path = tmp_path / "pilot" / "training_curves.parquet"
    before = path.stat().st_mtime_ns
    second = curves.training_curves(spec)
    assert path.stat().st_mtime_ns == before
    assert len(verifications) == 1
    pd.testing.assert_frame_equal(first, second)
    assert str(first.train_loss.dtype) == "float32"
    assert len(first) == 4 and set(first.unit_id) == {u.unit_id for u in units()}
    unit_path = tmp_path / "pilot" / (units()[0].unit_id + ".json")
    saved = json.loads(unit_path.read_bytes())
    saved["payload"]["per_team"][0]["lambda"] += 1
    unit_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="outdated"):
        curves.training_curves(spec)
    assert len(verifications) == 2
    assert path.stat().st_mtime_ns == before


def test_zero_iteration_fit_is_missing_loss_not_invented_zero(goal_sources, tmp_path, monkeypatch):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)

    class Zero(LoggedMean):
        @property
        def training_log(self):
            return dict(
                dataset="training",
                model="glm",
                train_loss=[],
                iterations=0,
                converged=True,
                recorded=True,
            )

    one = units()[:1]
    run_goal_units(one, data, version, tmp_path, n_jobs=1, model_factory=lambda u: Zero())
    spec = replace(
        load_goal_experiment(), raw=load_goal_experiment().raw | {"output": str(tmp_path)}
    )
    monkeypatch.setattr(curves, "load_goal_inputs", lambda spec: data)
    monkeypatch.setattr(curves, "goal_provenance", lambda *a, **k: version)
    monkeypatch.setattr(curves, "pilot_units", lambda spec: one)
    result = curves.training_curves(spec)
    assert len(result) == 1 and result.iteration.iloc[0] == 0
    assert result.train_loss.isna().all() and result.converged.all()


def test_validation_views_cannot_bypass_full_stage(tmp_path):
    spec = load_goal_experiment()
    with pytest.raises(ValueError, match="only to full"):
        curves.full_validation_table(spec, stage="pilot")


@pytest.mark.slow
def test_four_native_fits_open_no_files_and_publish_one_journal_each(
    goal_sources, tmp_path, monkeypatch
):
    import ligat.evaluate.goals_runner as runner
    from ligat.models.m2_maher import MAHER_COLUMNS

    prematch, games = goal_sources
    data = prepare(
        (prematch.assign(promoted_in=False), games),
        tuple(dict.fromkeys(("is_home", *MAHER_COLUMNS, "promoted_in"))),
    )
    version = provenance(data)
    configs = [
        ("xgb", XGBGoalConfig(2, 1, 4, 0.1, 0.8, 0.8, 1), ("is_home",)),
        ("catboost", CatBoostGoalConfig(2, 3, 4, 0.05), ("is_home",)),
        ("glm", GLMGoalConfig("r0"), ("is_home",)),
        ("maher", MaherGoalConfig(1), MAHER_COLUMNS),
    ]

    def forbidden(*a, **k):
        pytest.fail("A native fit opened a file")

    fitted = []
    for cls in (XGBGoalModel, CatBoostGoalModel, GLMGoalModel, MaherGoalModel):
        original = cls.fit

        def guarded(self, *args, _original=original, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(builtins, "open", forbidden)
                patch.setattr(Path, "open", forbidden)
                result = _original(self, *args, **kwargs)
            fitted.append(type(self).__name__)
            return result

        monkeypatch.setattr(cls, "fit", guarded)
    published = []
    original_atomic = runner.atomic_bytes

    def write(path, raw):
        if path.name.endswith(".trainlog.json.gz"):
            published.append(path)
        original_atomic(path, raw)

    monkeypatch.setattr(runner, "atomic_bytes", write)
    for backend, cfg, features in configs:
        run_goal_unit(GoalUnit(backend, asdict(cfg), features, 15), data, version, tmp_path)
    assert set(fitted) == {"XGBGoalModel", "CatBoostGoalModel", "GLMGoalModel", "MaherGoalModel"}
    assert len(published) == len(set(published)) == 4


def test_diagnostic_neighbors_change_one_grid_parameter_and_skip_boundaries():
    spec = load_goal_experiment()
    selected = XGBGoalConfig(2, 1, 150, 0.05, 0.8, 0.8, 1)
    before = asdict(selected)
    neighbors = curves.diagnostic_configurations(selected, spec)
    assert len(neighbors) == 4
    assert len({c.config_id for c in neighbors}) == 4
    assert neighbors[0] == selected and asdict(selected) == before
    changes = []
    for neighbor in neighbors[1:]:
        changed = {k for k, v in asdict(neighbor).items() if v != before[k]}
        assert len(changed) == 1
        changes.append(changed.pop())
        assert neighbor.max_depth in spec.raw["xgboost"]["stage_a"]["max_depth"]
        assert neighbor.learning_rate in spec.raw["xgboost"]["stage_a"]["learning_rate"]
    assert set(changes) == {"max_depth", "learning_rate", "n_estimators"}
    edge = replace(selected, max_depth=8, learning_rate=0.1, n_estimators=1000)
    boundary = curves.diagnostic_configurations(edge, spec)
    assert len(boundary) == 3  # selected, depth down, learning-rate down; max trees deduplicated
    assert all(c.max_depth <= 8 and c.learning_rate <= 0.1 for c in boundary)


def test_diagnostic_requires_frozen_report_before_any_source_or_training(monkeypatch):
    def missing(spec):
        raise ValueError("D3 report not frozen")

    def forbidden(*args, **kwargs):
        pytest.fail("Sources accessed before checking the frozen D3 report")

    monkeypatch.setattr(curves, "frozen_full_report", missing)
    monkeypatch.setattr(curves, "_sources", forbidden)
    with pytest.raises(ValueError, match="not frozen"):
        curves.run_learning_diagnostics(load_goal_experiment())


def test_diagnostic_loader_rejects_modified_record(tmp_path, monkeypatch):
    import hashlib

    spec = replace(
        load_goal_experiment(), raw=load_goal_experiment().raw | {"output": str(tmp_path)}
    )
    full = tmp_path / "full"
    full.mkdir()
    (full / "ladder_report.json").write_bytes(b"frozen report fixture")
    target = tmp_path / "diagnostics" / "learning_curves"
    target.mkdir(parents=True)
    identity = dict(
        diagnostic=True,
        frozen_report_sha256=hashlib.sha256((full / "ladder_report.json").read_bytes()).hexdigest(),
        code_sha256=curves._diagnostic_signature(),
    )
    output = dict(identity=identity, train_nll=[1.4], validation_nll=[1.5])
    record = target / (curves._payload_digest(identity) + ".json")
    record.write_text(json.dumps(dict(payload=output, sha256=curves._payload_digest(output))))
    summary = dict(diagnostic=True, outputs=[output])
    (target / "manifest.json").write_text(
        json.dumps(dict(payload=summary, sha256=curves._payload_digest(summary)))
    )
    monkeypatch.setattr(curves, "frozen_full_report", lambda spec: {})
    assert curves.load_learning_diagnostics(spec) == summary
    output["validation_nll"][0] = 0.5
    record.write_text(json.dumps(dict(payload=output, sha256=curves._payload_digest(output))))
    with pytest.raises(ValueError, match="fit record"):
        curves.load_learning_diagnostics(spec)
