"""Synthetic human-review overrides, provisional dependencies and immutable revisions."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.goals_experiment import load_goal_experiment
from ligat.evaluate.goals_ladder import (
    GoalLadder,
    LadderConfiguration,
    LadderReviewRequired,
    SavedGoalScores,
    exp005_full_status,
)
from ligat.models.m2_glm import GLMGoalConfig
from ligat.models.m2_xgb import XGBGoalConfig
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_goals_ladder import small_ladder
from tests.test_goals_runner import LoggedMean, provenance


def reviewed(engine, provider, root, mode, choices=None):
    spec = replace(
        engine.spec, raw=engine.spec.raw | {"rule6_mode": mode, "review_decisions": choices or {}}
    )
    return GoalLadder(spec, provider, root, provenance={"source": "synthetic"})


def candidates():
    return [
        LadderConfiguration("xgb", asdict(XGBGoalConfig(d, 1, 8, 0.1, 0.8, 0.8, 1)))
        for d in (2, 3, 4)
    ]


def test_stop_requires_decision_and_accepts_only_an_existing_candidate(monkeypatch, tmp_path):
    engine, provider = small_ladder(monkeypatch, tmp_path, recent_flip=True)
    engine = reviewed(engine, provider, tmp_path, "stop")
    choices = candidates()
    with pytest.raises(LadderReviewRequired):
        engine._select(choices, ("is_home",), "choice")
    accepted = reviewed(engine, provider, tmp_path, "stop", {"choice": choices[1].key})
    selected, _ = accepted._select(choices, ("is_home",), "choice")
    assert selected.key == choices[1].key
    decision = accepted.decisions[-1]
    assert decision["amit_choice"] == choices[1].key and decision["rule6"]
    assert decision["provisional_on"] == []
    assert decision["previous_decision"]["path"]
    invalid = reviewed(engine, provider, tmp_path, "stop", {"choice": "not-a-candidate"})
    count = len(provider.calls)
    with pytest.raises(ValueError, match="not a candidate"):
        invalid._select(choices, ("is_home",), "choice")
    assert len(provider.calls) == count


@pytest.mark.slow
def test_provisional_marks_actual_dependencies_and_exposes_readable_reviews(monkeypatch, tmp_path):
    engine, provider = small_ladder(monkeypatch, tmp_path, recent_flip=True)
    result = reviewed(engine, provider, tmp_path, "provisional").run()
    assert result["open_reviews"]
    for review in result["open_reviews"]:
        assert review["deep_leaders"][0]["label"] and review["recent_leaders"][0]["label"]
        assert review["dependents"]
    audit = next(d for d in result["decisions"] if d["kind"] == "entry")
    assert "S2-A" in audit["provisional_on"] and "S1-A" not in audit["provisional_on"]
    s2 = next(d for d in result["decisions"] if d["name"] == "S2-AB")
    assert "S2-A" in s2["provisional_on"]


@pytest.mark.slow
@pytest.mark.parametrize("choice", ["F1-A", "none"])
def test_f1_explicit_choice_overrides_larger_mean_improvement(monkeypatch, tmp_path, choice):
    engine, provider = small_ladder(monkeypatch, tmp_path, family="F1-R")
    result = reviewed(engine, provider, tmp_path, "stop", {"F1": choice}).run()
    family = next(d for d in result["decisions"] if d["kind"] == "family")
    assert family["proposal"] == "F1-R" and family["amit_choice"] == choice
    assert family["selected_family"] == (None if choice == "none" else choice)


def test_pending_blocks_full_before_inputs():
    spec = load_goal_experiment()
    spec = replace(
        spec,
        raw=spec.raw
        | {
            "full_run_approved": True,
            "fast_path": "short",
            "correlation_drops": [],
            "rule6_mode": "pending",
        },
    )
    with pytest.raises(ValueError, match="rule6_mode"):
        spec.guard_full()


@pytest.mark.slow
def test_review_reuses_signed_units_and_preserves_previous_decision(
    goal_sources, tmp_path, monkeypatch
):
    data = prepare(goal_sources, ("is_home",))
    spec = load_goal_experiment()
    spec = replace(
        spec,
        raw=spec.raw
        | {
            "output": str(tmp_path),
            "n_jobs": 1,
            "full_run_approved": True,
            "fast_path": "short",
            "correlation_drops": [],
            "rule6_mode": "stop",
        },
    )
    calls = []

    class Away(LoggedMean):
        def predict(self, X, *, context):
            return pd.Series(np.where(context.is_home, 0.2, 5.0), index=X.index, name="lambda")

    def factory(*args):
        calls.append(1)
        return Away()

    monkeypatch.setattr("ligat.evaluate.goals_runner._model", factory)
    configs = [LadderConfiguration("glm", asdict(GLMGoalConfig("r0")), hl) for hl in (None, 1)]
    first = GoalLadder(
        spec,
        SavedGoalScores(spec, data, provenance(data)),
        tmp_path / "full",
        provenance={"data": "synthetic"},
    )
    first._select(configs, ("is_home",), "choice")
    assert len(calls) == 30
    files = {p: p.read_bytes() for p in (tmp_path / "full" / "decisions").glob("*.json")}
    picked = next(c for c in configs if c.key != first.decisions[-1]["selected"])
    spec = replace(spec, raw=spec.raw | {"review_decisions": {"choice": picked.key}})
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner._model", lambda *a: pytest.fail("Refitted reviewed unit")
    )
    second = GoalLadder(
        spec,
        SavedGoalScores(spec, data, provenance(data)),
        tmp_path / "full",
        provenance={"data": "synthetic"},
    )
    selected, _ = second._select(configs, ("is_home",), "choice")
    assert selected.key == picked.key
    assert all(p.read_bytes() == raw for p, raw in files.items())
    assert list((tmp_path / "full" / "decisions").glob("*.r0001.json"))
    assert "previous_decision" in second.decisions[-1]
    monkeypatch.setattr("ligat.evaluate.goals_ladder.load_goal_inputs", lambda _: data)
    monkeypatch.setattr(
        "ligat.evaluate.goals_ladder.goal_provenance", lambda *a, **k: provenance(data)
    )
    assert exp005_full_status(spec)["counts"] == {"complete": 30}
    for path in (tmp_path / "full" / "decisions").glob("*.json"):
        assert json.loads(path.read_text())["sha256"]
