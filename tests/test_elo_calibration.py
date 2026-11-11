"""Prepared/direct equivalence, judge integration, predeclared selection rules and saves."""

import hashlib
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.prepare import AnalysisTables
from ligat.evaluate.backtest import EvaluationResult, evaluate, summarize
from ligat.evaluate.calibration_artifacts import (
    CalibrationProvenance,
    save_calibration,
    verify_calibration,
)
from ligat.evaluate.comparison import compare
from ligat.evaluate.elo_calibration import (
    CalibrationBatch,
    CalibrationConfig,
    CalibrationRun,
    build_refinement_grid,
    combine_stages,
    hyperparameter_summary,
    load_experiment,
    prepare_calibration,
    refinement_applicable,
    run_configuration,
    run_grid,
    simplicity_proposals,
    stage2_configs,
    structural_additions,
)
from ligat.features.elo import EloParams, compute_elo, compute_prepared_elo, prepare_elo
from ligat.models.ordered_logit import OrderedEloPredictor


def history() -> tuple[AnalysisTables, pd.Series]:
    rng = np.random.default_rng(305)
    records = []
    strengths = {1: 1.3, 2: 0.4, 3: -0.4, 4: -1.3}
    for season in range(8, 26):
        for n in range(36):
            home, away = rng.choice([1, 2, 3, 4], size=2, replace=False)
            x = strengths[int(home)] - strengths[int(away)]
            cumulative = 1 / (1 + np.exp(-np.array([-0.9 - x, 0.6 - x])))
            outcome = int((rng.random() > cumulative).sum())
            gf, ga = [(0, 1), (1, 1), (1, 0)][outcome]
            date = pd.Timestamp(season + 1998, 8, 1, tz="Asia/Jerusalem") + pd.Timedelta(
                days=n // 2
            )
            game = (season - 8) * 36 + n + 1
            for team, opponent, scored, conceded, is_home in (
                (home, away, gf, ga, True),
                (away, home, ga, gf, False),
            ):
                records.append(
                    dict(
                        game_id=game,
                        team_id=int(team),
                        opponent_id=int(opponent),
                        season_id=season,
                        date=date,
                        round=n + 1,
                        stage="regular",
                        is_home=is_home,
                        goals_scored=scored,
                        goals_conceded=conceded,
                        rejected_postmatch=999,
                    )
                )
    rows = pd.DataFrame(records).set_index(["team_id", "game_id"])
    empty = pd.DataFrame()
    tables = AnalysisTables(empty, rows, empty, empty, empty, empty)
    flags = pd.Series(
        np.arange(len(rows)) % 17 == 0,
        index=rows.reset_index().set_index(["game_id", "team_id"]).index,
        name="nothing_to_play",
    )
    return tables, flags


@pytest.fixture(scope="module")
def pilot_batch() -> CalibrationBatch:
    tables, flags = history()
    spec = load_experiment()
    return run_grid(prepare_calibration(tables, flags), spec.pilot[:2], leakage_threshold=0.01)


def test_grid_fixed_before_scoring_and_pilot_exact_corners() -> None:
    spec = load_experiment()
    assert spec.experiment_id == "EXP-001"
    assert len(spec.grid) == 1280 and len({c.config_id for c in spec.grid}) == 1280
    assert len(spec.pilot) == 8
    assert {(c.elo.k, c.elo.home_advantage, c.elo.dead_game_discount) for c in spec.pilot} == set(
        product((10, 40), (30, 90), (0, 1))
    )
    assert all(c.elo.season_regression == 0.33 and c.elo.use_goal_multiplier for c in spec.pilot)
    assert all(c.elo.promoted_initialization == "previous_promoted_mean" for c in spec.pilot)
    assert spec.half_lives == (None, 1, 2, 4) and spec.stage2_top_n == 10
    assert spec.leakage_threshold == 0.98
    assert spec.sha256 == hashlib.sha256(Path("config/experiment.yaml").read_bytes()).hexdigest()
    assert (
        CalibrationConfig(EloParams(), 2).config_id == CalibrationConfig(EloParams(), 2.0).config_id
    )


def test_prepared_engine_exact_direct_and_causal_and_detached_from_source() -> None:
    tables, flags = history()
    original = tables.team_matches.copy(deep=True)
    prepared = prepare_elo(tables, flags)
    for config in load_experiment().pilot:
        direct = compute_elo(tables, flags, config.elo)
        with patch("ligat.features.elo._games", side_effect=AssertionError("reprojection")):
            fast = compute_prepared_elo(prepared, config.elo)
        assert_frame_equal(direct.prematch, fast.prematch, check_exact=True)
        assert_frame_equal(direct.season_snapshots, fast.season_snapshots, check_exact=True)
    assert_frame_equal(original, tables.team_matches, check_exact=True)
    tables.team_matches.loc[:, "goals_scored"] = 99
    flags[:] = True
    again = compute_prepared_elo(prepared, config.elo)
    assert_frame_equal(again.prematch, fast.prematch, check_exact=True)


def test_fast_path_exact_direct_judge_probabilities_scores_and_parameters() -> None:
    tables, flags = history()
    config = load_experiment().pilot[0]
    prepared = prepare_calibration(tables, flags)
    run = run_configuration(prepared, config, leakage_threshold=0.01)
    direct = compute_elo(tables, flags, config.elo)
    games = prepared.games.copy()
    idx = pd.MultiIndex.from_frame(games[["game_id", "home_team_id"]])
    games["elo_diff"] = direct.prematch.elo_diff.reindex(idx).to_numpy()
    models = []

    def factory():
        model = OrderedEloPredictor()
        models.append(model)
        return model

    expected = evaluate(factory, games, feature_columns=("elo_diff",))
    assert_frame_equal(run.evaluation.per_game, expected.per_game, check_exact=True)
    assert_frame_equal(run.evaluation.per_fold, expected.per_fold, check_exact=True)
    for model, row in zip(models, run.fits.itertuples(), strict=True):
        assert asdict(model.fit_result) == {
            key: getattr(row, key) for key in asdict(model.fit_result)
        }
        assert max(model.training_seasons) < row.season_id
    assert len(run.fits) == 15 and run.fits.converged.all() and run.fits.beta.gt(0).all()


def test_time_stage_uses_same_elo_and_does_not_change_judge_or_features() -> None:
    tables, flags = history()
    prepared = prepare_calibration(tables, flags)
    base = load_experiment().pilot[0]
    configs = (base, replace(base, half_life=2))
    with patch(
        "ligat.evaluate.elo_calibration.compute_prepared_elo", wraps=compute_prepared_elo
    ) as engine:
        batch = run_grid(prepared, configs, leakage_threshold=0.01)
    assert engine.call_count == 1
    assert_frame_equal(batch.runs[0].snapshots, batch.runs[1].snapshots, check_exact=True)
    assert len(batch.runs[1].fits) == 15
    assert not batch.runs[0].fits.beta.equals(batch.runs[1].fits.beta)
    with pytest.raises(ValueError, match="chronological"):
        run_configuration(prepared, configs[1], leakage_threshold=0.01, mode="loso")


def test_validation_result_cannot_change_its_own_probability_or_any_fit() -> None:
    tables, flags = history()
    config = load_experiment().pilot[0]
    before = run_configuration(prepare_calibration(tables, flags), config, leakage_threshold=0.01)
    rows = tables.team_matches
    game = rows.loc[rows.season_id.eq(25)].reset_index().game_id.iloc[0]
    mask = rows.index.get_level_values("game_id") == game
    rows.loc[mask, "goals_scored"] = [3, 0]
    rows.loc[mask, "goals_conceded"] = [0, 3]
    after = run_configuration(prepare_calibration(tables, flags), config, leakage_threshold=0.01)
    assert_frame_equal(before.fits, after.fits, check_exact=True)
    columns = ["p_home", "p_draw", "p_away"]
    assert_frame_equal(
        before.evaluation.per_game.set_index("game_id").loc[[game], columns],
        after.evaluation.per_game.set_index("game_id").loc[[game], columns],
        check_exact=True,
    )


def test_leakage_gate_stops_before_next_configuration() -> None:
    tables, flags = history()
    with patch(
        "ligat.evaluate.elo_calibration.compute_prepared_elo", wraps=compute_prepared_elo
    ) as engine:
        with pytest.raises(RuntimeError, match="Leakage threshold"):
            run_grid(
                prepare_calibration(tables, flags),
                load_experiment().pilot[:2],
                leakage_threshold=10,
            )
    assert engine.call_count == 1


@pytest.mark.parametrize("season", [26, 27, 28])
def test_no_holdout_accepted_before_engine_or_judge(season: int) -> None:
    tables, flags = history()
    tables.team_matches.loc[tables.team_matches.season_id.eq(25), "season_id"] = season
    with pytest.raises(ValueError, match="development"):
        prepare_calibration(tables, flags)


def synthetic_run(config: CalibrationConfig, losses: np.ndarray) -> CalibrationRun:
    scored = pd.DataFrame(
        {
            "fold": range(1, 16),
            "season_id": range(11, 26),
            "game_id": range(15),
            "outcome": 0,
            "log_loss": losses,
            "rps": losses / 5,
        }
    )
    folds, summary = summarize(scored)
    result = EvaluationResult(scored, folds, summary, "walk_forward", "synthetic")
    return CalibrationRun(config, result, pd.DataFrame(), pd.DataFrame(), 0)


def batch_of(runs: tuple[CalibrationRun, ...]) -> CalibrationBatch:
    return CalibrationBatch(runs, compare({r.config.config_id: r.evaluation for r in runs}))


def test_simplicity_only_structural_and_proposal_not_approval() -> None:
    simple = CalibrationConfig(
        EloParams(
            dead_game_discount=1, use_goal_multiplier=False, promoted_initialization="fixed_1450"
        )
    )
    assert structural_additions(simple) == 0
    assert (
        structural_additions(
            replace(simple, elo=replace(simple.elo, k=40, home_advantage=90, season_regression=0.5))
        )
        == 0
    )
    complex_config = CalibrationConfig(EloParams(dead_game_discount=0.25), 2)
    assert structural_additions(complex_config) == 4
    best = synthetic_run(complex_config, np.ones(15))
    close = synthetic_run(simple, 1.0001 + np.linspace(0.01, -0.01, 15))
    batch = batch_of((best, close))
    assert simplicity_proposals(batch) == (simple.config_id,)
    assert batch.comparison.requires_decision  # All folds and last five disagree.
    assert not hasattr(batch, "chosen")


def test_stage2_builder_and_exact_cutoff_tie_needs_decision() -> None:
    spec = replace(load_experiment(), stage2_top_n=2)
    configs = spec.pilot[:3]
    runs = tuple(synthetic_run(c, np.full(15, 1 + i * 0.01)) for i, c in enumerate(configs))
    candidates = stage2_configs(batch_of(runs), spec)
    assert len(candidates) == 8
    assert {c.elo for c in candidates} == {c.elo for c in configs[:2]}
    assert {c.half_life for c in candidates} == {None, 1, 2, 4}
    tied = (*runs[:2], synthetic_run(configs[2], np.full(15, 1.01)))
    with pytest.raises(ValueError, match="cutoff"):
        stage2_configs(batch_of(tied), spec)


def test_combined_stages_reuse_controls_and_report_marginals() -> None:
    coarse = load_experiment().pilot[:2]
    first = tuple(synthetic_run(c, np.full(15, 1 + i * 0.01)) for i, c in enumerate(coarse))
    weighted = synthetic_run(replace(coarse[0], half_life=2), np.full(15, 0.99))
    combined = combine_stages(batch_of(first), batch_of((first[0], weighted)))
    assert len(combined.runs) == 3
    assert combined.comparison.all_winners == (weighted.config.config_id,)
    summary = hyperparameter_summary(combined)
    row = summary.loc[summary.parameter.eq("half_life") & summary.value.isna()].iloc[0]
    assert row.configs == 2 and row.mean_log_loss == pytest.approx(1.005)
    with pytest.raises(ValueError, match="different scores"):
        combine_stages(batch_of(first), batch_of((synthetic_run(coarse[0], np.full(15, 1.1)),)))


def test_refinement_requires_paired_improvement_and_identical_control() -> None:
    proposed = CalibrationConfig(EloParams(dead_game_discount=0.25))
    control = replace(proposed, elo=replace(proposed.elo, dead_game_discount=1))
    best = synthetic_run(proposed, np.ones(15))
    assert refinement_applicable(best, synthetic_run(control, np.full(15, 1.001)))
    assert not refinement_applicable(
        best, synthetic_run(control, 1 + np.linspace(-0.02, 0.021, 15))
    )
    assert not refinement_applicable(best, synthetic_run(control, np.ones(15)))
    assert not refinement_applicable(best, synthetic_run(control, np.full(15, 0.99)))
    with pytest.raises(ValueError, match="only"):
        refinement_applicable(best, synthetic_run(replace(control, half_life=1), np.ones(15)))


@pytest.mark.parametrize(
    "discount,expected",
    [
        (0, [0, 0.1, 0.2]),
        (0.25, [0.1, 0.2, 0.3, 0.4]),
        (0.5, [0.3, 0.4, 0.5, 0.6, 0.7]),
        (0.75, [0.6, 0.7, 0.8, 0.9]),
        (1, []),
    ],
)
def test_refinement_grid_five_approved_rows_and_reused_references(discount, expected) -> None:
    spec = load_experiment()
    proposed = CalibrationConfig(EloParams(k=30, home_advantage=70, dead_game_discount=discount), 2)
    with patch(
        "ligat.evaluate.elo_calibration.run_configuration", side_effect=AssertionError("no run")
    ):
        plan = build_refinement_grid(proposed, spec)
    assert [c.elo.dead_game_discount for c in plan.candidates] == expected
    if discount == 1:
        assert not plan.to_run and not plan.references
        return
    assert plan.references == (
        proposed,
        replace(proposed, elo=replace(proposed.elo, dead_game_discount=1)),
    )
    assert not set(plan.to_run) & set(plan.references)
    for config in (*plan.candidates, *plan.references):
        assert config.half_life == proposed.half_life
        assert replace(config.elo, dead_game_discount=discount) == proposed.elo


def test_save_roundtrip_exact_metadata_checksums_and_predictions(pilot_batch, tmp_path) -> None:
    spec = load_experiment()
    provenance = CalibrationProvenance("a" * 40, "b" * 64, "c" * 64, {"source.py": "d" * 64})
    directory = tmp_path / "pilot"
    manifest = save_calibration(
        pilot_batch,
        spec,
        provenance,
        directory=directory,
        prediction_ids=(pilot_batch.runs[0].config.config_id,),
        stage="pilot",
    )
    metadata = verify_calibration(manifest)
    assert metadata["input_sha256"] == provenance.input_sha256
    assert metadata["experiment_sha256"] == spec.sha256 and metadata["fold_rows"] == 30
    assert metadata["stage"] == "pilot"
    assert (directory / "experiment.yaml").read_bytes() == Path(
        "config/experiment.yaml"
    ).read_bytes()
    predictions = pd.read_parquet(directory / "predictions.parquet")
    assert_frame_equal(
        predictions.drop(columns="config_id"),
        pilot_batch.runs[0].evaluation.per_game,
        check_exact=True,
    )
    scores = pd.read_csv(directory / "fold_scores.csv", float_precision="round_trip")
    assert scores.beta.gt(0).all() and scores.converged.all()
    snapshots = pd.read_parquet(directory / "snapshots.parquet")
    assert len(snapshots) == sum(len(run.snapshots) for run in pilot_batch.runs)
    assert not any(Path(name).is_absolute() for name in metadata["files"])
    with pytest.raises(FileExistsError):
        save_calibration(pilot_batch, spec, provenance, directory=directory, stage="pilot")
    (directory / "fold_scores.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_calibration(manifest)
    assert json.loads(manifest.read_text())["completed"]
