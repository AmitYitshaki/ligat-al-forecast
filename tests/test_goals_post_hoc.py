"""Post-hoc plans are fixed, signed, resumable and keep fitting in memory."""

import builtins
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ligat.evaluate import goals_post_hoc as post
from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.features import registry
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_runner import LoggedMean, provenance, units


def selected_report():
    return {
        "winners": {
            "S2": {
                "backend": "xgb",
                "parameters": asdict(XGBGoalConfig(2, 1, 150, 0.05, 0.8, 0.8, 1)),
                "half_life": None,
            }
        }
    }


def test_declared_post_hoc_plan_applies_registry_and_drops():
    spec = load_goal_experiment()
    report = selected_report()
    plan = post.diagnostic_plan(spec, report)
    assert len(plan) == 14 * len(spec.seasons)
    p1 = next(u for name, u in plan if name == "P1")
    p2 = next(u for name, u in plan if name == "P2")
    expected = set(registry.M2_BASE_COLUMNS)
    for name in ["F1-A", "F2", "F3", "F4", "F5"]:
        expected.update(registry.M2_FAMILY_COLUMNS[name])
    expected -= set(spec.raw["correlation_drops"])
    assert set(p2.features) == expected
    assert set(p1.features) == expected - {"elo_pre", "elo_diff"}
    assert p1.parameters == p2.parameters == report["winners"]["S2"]["parameters"]
    assert not {"team_id", "opponent_id"} & set(p2.features)
    assert all(not set(spec.raw["correlation_drops"]) & set(u.features) for _, u in plan)
    for name, u in plan:
        if name == "P4":
            assert u.parameters["one_hot_max_size"] == spec.raw["catboost"]["one_hot_max_size"]


def test_diagnostic_signature_binds_plan_code(monkeypatch):
    first = post._signature()
    assert first == post._signature()

    def changed(spec, report):
        return []

    monkeypatch.setattr(post, "diagnostic_plan", changed)
    assert post._signature() != first


def test_diagnostic_units_resume_verify_binding_and_write_one_journal(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, ("is_home",))
    version = provenance(data)
    called = []
    journals = []
    import ligat.evaluate.goals_runner as runner

    original_atomic = runner.atomic_bytes

    def published(path, raw):
        if path.name.endswith(".trainlog.json.gz"):
            journals.append(path)
        original_atomic(path, raw)

    monkeypatch.setattr(runner, "atomic_bytes", published)

    class InMemory(LoggedMean):
        def fit(self, *args, **kwargs):
            def forbidden(*a, **k):
                pytest.fail("File opened during diagnostic fit")

            with monkeypatch.context() as patch:
                patch.setattr(builtins, "open", forbidden)
                patch.setattr(Path, "open", forbidden)
                result = super().fit(*args, **kwargs)
            called.append(True)
            return result

    binding = dict(diagnostic=True, code_sha256="synthetic-fixture")
    post.diagnostic_units(
        list(units()), data, version, tmp_path, binding, model_factory=lambda u: InMemory()
    )
    assert len(called) == len(journals) == len(units())
    paths = list((tmp_path / "synthetic").glob("unit-*.json"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert all(json.loads(p.read_bytes())["payload"]["diagnostic"] is True for p in paths)

    def forbidden_factory(unit):
        pytest.fail("A cached diagnostic was fitted again")

    post.diagnostic_units(
        list(units()), data, version, tmp_path, binding, model_factory=forbidden_factory
    )
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())
    with pytest.raises(ValueError, match="outdated"):
        post.diagnostic_units(
            list(units()),
            data,
            version,
            tmp_path,
            binding | {"code_sha256": "changed"},
            model_factory=forbidden_factory,
        )
