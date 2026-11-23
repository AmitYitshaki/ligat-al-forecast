"""Target-free correlation screening and synthetic full-ladder contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.goals_ladder import (
    GoalLadder,
    LadderConfiguration,
    LadderReviewRequired,
    candidate_correlations,
)
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_m2_identity import identity_rows as identity_rows


def test_correlation_screening_excludes_nominal_ids_and_keeps_all_features():
    frame = pd.DataFrame(
        {
            "elo_pre": np.arange(20),
            "elo_diff": np.arange(20) * 2,
            "is_home": [True, False] * 10,
            "stage": ["regular"] * 20,
            "team_id": np.arange(20) + 1,
            "opponent_id": np.arange(20)[::-1] + 1,
        }
    )
    original = frame.copy(deep=True)
    pairs, nominal = candidate_correlations(frame, seasons=pd.Series([8] * 20))
    assert nominal == ("stage", "team_id", "opponent_id")
    assert pairs[["first", "second"]].values.tolist() == [["elo_pre", "elo_diff"]]
    assert pairs.iloc[0].r == pytest.approx(1)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="development-only"):
        candidate_correlations(frame, seasons=pd.Series([26] * 20))
    with pytest.raises(ValueError, match="development-only"):
        candidate_correlations(frame.assign(goals_scored=1), seasons=pd.Series([8] * 20))


@pytest.mark.real_data
def test_real_correlations_read_features_only(real_tables):
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from ligat.evaluate.goals_runner import load_goal_inputs

    if not Path("data/derived/prematch/prematch_v2.parquet").exists():
        pytest.skip("Frozen M0 is local only")
    spec = load_goal_experiment()
    data = load_goal_inputs(spec, tables=real_tables)
    pairs, nominal = candidate_correlations(data.X, seasons=data.context.season_id)
    root = Path("data/reports/T-014")
    root.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(root / "D1b_correlations.csv", index=False)
    (root / "D1b_correlations.json").write_text(
        json.dumps(
            {
                "pairs": pairs.to_dict(orient="records"),
                "not_applicable": nominal,
                "correlation_drops": spec.raw["correlation_drops"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert spec.raw["correlation_drops"] == "pending"
    assert not {"goals_scored", "poisson_nll"} & set(data.X)
    print(pairs.to_string(index=False))


@pytest.mark.parametrize("backend", ["xgb", "catboost"])
@pytest.mark.parametrize("column", ["team_id", "opponent_id"])
def test_internal_identity_ablation_projects_encoded_columns_and_roundtrips(
    identity_rows, tmp_path, backend, column
):
    from ligat.models.m2_catboost import CatBoostGoalModel
    from ligat.models.m2_identity import TeamIdentity
    from ligat.models.m2_xgb import XGBGoalModel
    from tests.test_m2_catboost import config as cat_config
    from tests.test_m2_xgb import config as xgb_config

    X, y, context, vocabulary = identity_rows
    partial = X.loc[:, [column, "elo_diff"]]
    identity = TeamIdentity.from_context(context, team_ids=vocabulary)
    projected = identity.transform(partial, context=context, columns=(column,))
    pd.testing.assert_frame_equal(
        projected, identity.transform(X, context=context).loc[:, partial.columns]
    )
    model = (
        XGBGoalModel(xgb_config(), team_ids=vocabulary, identity_columns=(column,))
        if backend == "xgb"
        else CatBoostGoalModel(cat_config(), team_ids=vocabulary, identity_columns=(column,))
    )
    model.fit(partial, y, context=context)
    native_columns = (
        model._estimator.get_booster().feature_names
        if backend == "xgb"
        else model._estimator.feature_names_
    )
    assert native_columns == partial.columns.tolist()
    path = tmp_path / ("model.json" if backend == "xgb" else "model.cbm")
    model.save(path)
    loaded = type(model).load(path)
    np.testing.assert_array_equal(
        model.predict(partial, context=context), loaded.predict(partial, context=context)
    )


class FamilyGoalSpy:
    """Fixed synthetic predictions improve exactly one family, without fitted trees."""

    def __init__(self, family="F2", *, recent_flip=False, leakage=False):
        self.family, self.recent_flip, self.leakage = family, recent_flip, leakage
        self.calls = []

    def __call__(self, config, columns, role):
        from ligat.evaluate.goal_diagnostics import goal_outcome_metrics
        from ligat.evaluate.goal_metrics import poisson_nll
        from ligat.evaluate.team_goal_backtest import TeamGoalEvaluation
        from ligat.features import registry

        self.calls.append((config, columns, role))
        depth = config.parameters.get("max_depth", config.parameters.get("depth", 0))
        depth = 6 if depth is None else depth
        trees = (
            config.parameters.get("n_estimators", config.parameters.get("iterations", 0)) or 1000
        )
        rows = []
        for season in range(11, 26):
            rate = 2.5 + depth * 0.01 + trees * 0.00001
            if self.family in ("F1-R", "F1-A") and "gf_all_w5" in columns:
                replacement = "gf_venue_ewma5" not in columns
                rate -= 0.35 if replacement == (self.family == "F1-R") else 0.2
            elif self.family in registry.M2_FAMILY_COLUMNS and set(
                registry.M2_FAMILY_COLUMNS[self.family]
            ) <= set(columns):
                rate -= 0.3
            if self.recent_flip:
                rate = (3.0 if season >= 21 else 1.5) if depth == 2 else 2.5
            if self.leakage:
                rate = 0.1
            for team in (1, 2):
                rows.append(
                    {
                        "game_id": season * 100,
                        "team_id": team,
                        "season_id": season,
                        "fold": season - 10,
                        "deep_fold": season
                        >= (
                            13
                            if config.backend == "glm" and config.parameters["training_start"] == 9
                            else 12
                        ),
                        "is_home": team == 1,
                        "goals_scored": 1,
                        "lambda": rate,
                    }
                )
        frame = pd.DataFrame(rows).set_index(["game_id", "team_id"])
        frame["poisson_nll"] = poisson_nll(frame.goals_scored, frame["lambda"])
        folds = (
            frame.reset_index()
            .groupby(["fold", "season_id"], as_index=False)
            .agg(
                games=("game_id", "nunique"),
                deep_fold=("deep_fold", "first"),
                poisson_nll=("poisson_nll", "mean"),
            )
        )
        evaluation = TeamGoalEvaluation(
            frame, folds, pd.Series({"actual_depth": depth, "actual_iterations": trees}), "per_fold"
        )
        outcomes = goal_outcome_metrics(evaluation)
        for score in ("rps", "log_loss"):
            frame[score] = frame.index.get_level_values("game_id").map(outcomes.per_game[score])
            folds[score] = outcomes.per_fold[score]
        return evaluation


def small_ladder(monkeypatch, tmp_path, *, family="F2", recent_flip=False, leakage=False):
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from ligat.models.m2_catboost import CatBoostGoalConfig
    from ligat.models.m2_xgb import XGBGoalConfig

    configs = [XGBGoalConfig(d, 1, n, 0.1, 0.8, 0.8, 1) for d, n in ((2, 8), (3, 12), (4, 16))]
    monkeypatch.setattr("ligat.evaluate.goals_ladder.xgb_stage_a", lambda _: configs)
    monkeypatch.setattr(
        "ligat.evaluate.goals_ladder.xgb_stage_b",
        lambda spec, anchor: [
            (replace(anchor, reg_lambda=10), 1),
            (replace(anchor, reg_lambda=0), 2),
        ],
    )
    monkeypatch.setattr(
        "ligat.evaluate.goals_ladder.catboost_grid",
        lambda _: [CatBoostGoalConfig(4, 3, 16, 0.1), CatBoostGoalConfig.library_default()],
    )
    spec = load_goal_experiment()
    spec = replace(spec, raw=spec.raw | {"correlation_drops": [], "fast_path": "short"})
    provider = FamilyGoalSpy(family, recent_flip=recent_flip, leakage=leakage)
    return GoalLadder(spec, provider, tmp_path, provenance={"source": "synthetic"}), provider


@pytest.mark.slow
@pytest.mark.parametrize("family", ["F2", "F1-R", "F1-A", "F6"])
def test_complete_synthetic_ladder_enters_only_helpful_family(monkeypatch, tmp_path, family):
    engine, provider = small_ladder(monkeypatch, tmp_path, family=family)
    result = engine.run()
    assert result["entered"] == [family]
    cat = [columns for config, columns, _ in provider.calls if config.backend == "catboost"]
    assert any("team_id" in columns for columns in cat) == (family == "F6")
    assert len(result["top5_s2"]) == 5
    assert len({LadderConfiguration(**c).key for c in result["top5_s2"]}) == 5
    for path in (tmp_path / "decisions").glob("*.json"):
        from ligat.evaluate.goals_runner import _payload_digest

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["sha256"] == _payload_digest(saved["payload"])
    for decision in result["decisions"]:
        if decision["kind"] == "selection" and decision["name"].endswith("-short"):
            assert set(decision["configurations"]) == {
                LadderConfiguration(**c).key for c in result["top5_s2"]
            }


def test_rule6_stops_before_dependent_stages(monkeypatch, tmp_path):
    engine, _ = small_ladder(monkeypatch, tmp_path, recent_flip=True)
    with pytest.raises(LadderReviewRequired, match="Rule 6"):
        engine.run()


@pytest.mark.slow
def test_explicit_catboost_f6_request_runs_without_xgb_entry(monkeypatch, tmp_path):
    engine, provider = small_ladder(monkeypatch, tmp_path)
    engine.spec = replace(engine.spec, raw=engine.spec.raw | {"catboost_f6_requested": True})
    result = engine.run()
    assert "F6" not in result["entered"]
    assert any(
        config.backend == "catboost" and "team_id" in columns
        for config, columns, _ in provider.calls
    )


def test_leakage_threshold_stops_synthetic_ladder(monkeypatch, tmp_path):
    from ligat.evaluate.selection_rules import LeakageSuspectedError

    engine, _ = small_ladder(monkeypatch, tmp_path, leakage=True)
    with pytest.raises(LeakageSuspectedError):
        engine.run()


def test_pilot_cli_does_not_invoke_any_ladder_rule(monkeypatch, tmp_path, goal_sources):
    from typer.testing import CliRunner

    from ligat.cli import app
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from tests.test_goal_inputs import prepare
    from tests.test_goals_runner import LoggedMean, provenance, units

    data = prepare(goal_sources, ("is_home",))
    spec = load_goal_experiment()
    spec = replace(spec, raw=spec.raw | {"output": str(tmp_path), "n_jobs": 1})
    monkeypatch.setattr("ligat.evaluate.goals_experiment.load_goal_experiment", lambda _: spec)
    monkeypatch.setattr("ligat.evaluate.goals_runner.load_goal_inputs", lambda _: data)
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner.goal_provenance", lambda *a, **k: provenance(data)
    )
    monkeypatch.setattr("ligat.evaluate.goals_runner.pilot_units", lambda _: units()[:1])
    monkeypatch.setattr("ligat.evaluate.goals_runner._model", lambda *a: LoggedMean())
    for name in (
        "rule8",
        "close_configurations",
        "recent_leader_flag",
        "leakage_check",
        "metric_disagreement_entry",
        "metric_disagreement_selection",
    ):
        monkeypatch.setattr(
            "ligat.evaluate.goals_ladder." + name,
            lambda *a, **k: pytest.fail("Pilot invoked a decision rule"),
        )
    result = CliRunner().invoke(app, ["run", "goals", "--stage", "pilot"])
    assert result.exit_code == 0, result.output


def test_full_score_provider_uses_c5_status_and_resumes_units(monkeypatch, tmp_path, goal_sources):
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from ligat.evaluate.goals_ladder import SavedGoalScores
    from ligat.models.m2_glm import GLMGoalConfig
    from tests.test_goal_inputs import prepare
    from tests.test_goals_runner import LoggedMean, provenance

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

    class AwayBiasedSpy(LoggedMean):
        def predict(self, X, *, context):
            # Deliberately poor synthetic predictions keep the leakage gate clear.
            return pd.Series(np.where(context.is_home, 0.2, 5.0), index=X.index, name="lambda")

    monkeypatch.setattr("ligat.evaluate.goals_runner._model", lambda *a: AwayBiasedSpy())
    provider = SavedGoalScores(spec, data, provenance(data))
    config = LadderConfiguration("glm", asdict(GLMGoalConfig("r0")))
    first = provider(config, ("is_home",), "synthetic-full")
    assert first.per_fold.fold.nunique() == 15
    assert set(first.per_team.season_id) == set(range(11, 26))
    assert (tmp_path / "full" / "index.csv").exists()
    monkeypatch.setattr(
        "ligat.evaluate.goals_runner._model",
        lambda *a: pytest.fail("Refitted completed full units"),
    )
    second = SavedGoalScores(spec, data, provenance(data))(config, ("is_home",), "resume")
    pd.testing.assert_frame_equal(first.per_team, second.per_team)
    pd.testing.assert_frame_equal(first.per_fold, second.per_fold)


def test_full_entry_guard_precedes_inputs_and_scores(monkeypatch):
    from ligat.evaluate.goals_experiment import load_goal_experiment
    from ligat.evaluate.goals_ladder import run_goal_full

    monkeypatch.setattr(
        "ligat.evaluate.goals_ladder.load_goal_inputs",
        lambda *a: pytest.fail("Unapproved full loaded inputs"),
    )
    with pytest.raises(ValueError, match="not approved"):
        run_goal_full(load_goal_experiment())


@pytest.mark.slow
def test_synthetic_full_report_contains_all_comparators_and_confidence_intervals(
    monkeypatch, tmp_path
):
    from ligat.evaluate.goals_ladder import run_goal_full
    from ligat.evaluate.goals_runner import GoalProvenance

    engine, provider = small_ladder(monkeypatch, tmp_path)
    spec = replace(
        engine.spec,
        raw=engine.spec.raw
        | {"output": str(tmp_path), "full_run_approved": True, "rule6_mode": "stop"},
    )
    monkeypatch.setattr("ligat.evaluate.goals_ladder.load_goal_inputs", lambda _: object())
    monkeypatch.setattr(
        "ligat.evaluate.goals_ladder.goal_provenance",
        lambda *a, **k: GoalProvenance("0" * 64, "0" * 64, "synthetic"),
    )
    monkeypatch.setattr("ligat.evaluate.goals_ladder.SavedGoalScores", lambda *a: provider)
    result = run_goal_full(spec)
    assert {"glm", "maher", "catboost-base", "catboost-selected"} <= set(result["readable_metrics"])
    assert {"R0", "R1", "EXP-002", "EXP-003"} <= set(result["readable_metrics"])
    assert result["readable_metrics"]["glm"]["paired"]["n"] == 13
    assert result["readable_metrics"]["S2"]["paired"]["n"] == 14
    assert "ci_lower" in result["readable_metrics"]["glm"]["paired"]
    saved = json.loads((tmp_path / "full" / "ladder_report.json").read_text(encoding="utf-8"))
    assert saved["payload"] == result
