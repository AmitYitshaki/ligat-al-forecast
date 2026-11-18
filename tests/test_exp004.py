"""EXP-004 declaration, joins, ordinal adapter, paired rules and blind persistence."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ligat.analysis.prepare import AnalysisTables
from ligat.evaluate import motivation_experiment as me
from ligat.evaluate.backtest import DevelopmentPolicy, EvaluationResult, evaluate, summarize
from ligat.evaluate.calibration_artifacts import verify_calibration
from ligat.evaluate.motivation_experiment_artifacts import (
    MotivationProvenance,
    blind_pilot_report,
    save_motivation_experiment,
)
from ligat.features.table_state import game_rows
from ligat.models.ordered_logit import OrderedEloPredictor, OrderedFit
from tests.test_form import season_fixtures, tables_from


def synthetic_inputs():
    tables = tables_from(season_fixtures(25))
    rows = game_rows(tables)
    index = pd.MultiIndex.from_frame(rows[["game_id", "team_id"]].astype("int64"))
    identity = rows.set_index(["game_id", "team_id"]).reindex(index)
    motivation = identity[["opponent_id", "season_id", "is_home", "as_of"]].copy()
    motivation["nothing_to_play"] = False
    motivation["leverage_max"] = 0.3
    motivation["title_lev"] = 0.3
    motivation["cut_lev"] = np.nan
    motivation["survival_lev"] = 0.1
    motivation.iloc[0, motivation.columns.get_loc("nothing_to_play")] = True
    elo = pd.DataFrame({"elo_diff": np.where(identity.is_home, 100.0, -100.0)}, index=index)
    return tables, me.MotivationInputs(motivation, elo)


def evaluation(pattern=None, *, rps_pattern=None, seasons=me.VALIDATION_SEASONS):
    pattern = {} if pattern is None else pattern
    rps_pattern = pattern if rps_pattern is None else rps_pattern
    frame = pd.DataFrame(
        [
            dict(
                fold=s - 10,
                season_id=s,
                game_id=s * 100 + i,
                outcome=i % 3,
                log_loss=1.1 - pattern.get(s, 0.0),
                rps=0.24 - rps_pattern.get(s, 0.0),
            )
            for s in seasons
            for i in range(1 + s % 3)
        ]
    )
    folds, summary = summarize(frame)
    return EvaluationResult(frame, folds, summary, "walk_forward", "synthetic")


def toy_run(config, *, pattern=None, stage="full"):
    seasons = me.PILOT_SEASONS if stage == "pilot" else me.VALIDATION_SEASONS
    result = evaluation(pattern, seasons=seasons)
    fits = pd.DataFrame(
        [
            dict(
                fold=s - 10,
                season_id=s,
                training_seasons=tuple(range(8, s)),
                training_games=100,
                validation_games=int(result.per_fold.set_index("season_id").loc[s, "games"]),
                beta=1.0,
                theta1=-1.0,
                delta=0.2,
                theta2=-1 + np.exp(0.2),
                additional_betas=tuple(-0.2 for _ in config.columns),
                log_likelihood=-10.0,
                converged=True,
                iterations=10,
                accepted_by_gradient_check=False,
                max_abs_gradient=1e-9,
            )
            for s in seasons
        ]
    )
    return me.MotivationRun(config, result, fits, 0.2, stage)


def test_exact_grid_order_pairs_and_commented_yaml():
    spec = me.load_motivation_experiment()
    configs = me.motivation_configurations()
    assert len(configs) == len({c.config_id for c in configs}) == 20
    assert [c.config_id for c in configs] == [r["config_id"] for r in spec.raw["configurations"]]
    assert all("#" in line for line in spec.source_text.splitlines() if line.strip())
    assert me.MotivationConfiguration("Comp", 4).columns == (
        "Comp_title_diff",
        "Comp_cut_diff",
        "Comp_survival_diff",
    )
    assert me.MotivationConfiguration("L2-diff", 4).reference_id == "C0-hl4"
    assert me.MotivationConfiguration("C0", 4).reference_id == "C0"
    with pytest.raises(ValueError, match="outside"):
        me.MotivationConfiguration("unknown")


@pytest.mark.parametrize("field", ["hash", "folds", "decay", "pilot", "extra"])
def test_yaml_drift_is_rejected(tmp_path, field):
    spec = me.load_motivation_experiment()
    raw = yaml.safe_load(spec.source_text)
    if field == "hash":
        raw["inputs"]["motivation"]["sha256"] = "0" * 64
    elif field == "folds":
        raw["evaluation"]["training_start"] = 9
    elif field == "decay":
        raw["configurations"][3]["half_life"] = 5
    elif field == "pilot":
        raw["pilot"]["configurations"][-1] = "Comp"
    else:
        raw["undeclared"] = True
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="predeclared"):
        me.load_motivation_experiment(path)


def test_shared_representations_design_and_dead_zero(monkeypatch):
    tables, inputs = synthetic_inputs()
    calls = []
    original = me.representations.game_contrast

    def spy(frame, level, form="diff"):
        calls.append((level, form))
        return original(frame, level, form)

    monkeypatch.setattr(me.representations, "game_contrast", spy)
    prepared = me.prepare_motivation_experiment(tables, inputs)
    assert len(calls) == 10 and ("Comp", "diff") in calls
    for config in me.motivation_configurations()[::2]:
        design = me.MotivationOrderedPredictor(config).design(prepared.games)
        np.testing.assert_array_equal(design[:, 0], prepared.games.elo_diff / 400)
        if config.columns:
            np.testing.assert_array_equal(design[:, 1:], prepared.games[list(config.columns)])
    dead = inputs.motivation.iloc[[0]].copy()
    for fn in (
        me.representations.level_l2,
        me.representations.level_l3,
        me.representations.level_l5,
        me.representations.level_lc,
        me.representations.components,
    ):
        assert np.asarray(fn(dead)).sum() == 0
    contrast = original(inputs.motivation, "L2", "sides")
    np.testing.assert_array_equal(
        prepared.games[["L2_m_home", "L2_m_away"]], contrast.reindex(prepared.games.game_id)
    )
    np.testing.assert_array_equal(
        prepared.games.L2_m_diff, prepared.games.L2_m_home - prepared.games.L2_m_away
    )


@pytest.mark.parametrize("damage", ["missing", "duplicate", "venue", "opponent", "date", "holdout"])
def test_bad_team_fixture_join_stops(damage):
    tables, inputs = synthetic_inputs()
    m = inputs.motivation.copy()
    if damage == "missing":
        m = m.iloc[1:]
    elif damage == "duplicate":
        m = pd.concat([m, m.iloc[[0]]])
    elif damage == "venue":
        m.iloc[0, m.columns.get_loc("is_home")] = not bool(m.iloc[0].is_home)
    elif damage == "opponent":
        m.iloc[0, m.columns.get_loc("opponent_id")] = 999
    elif damage == "date":
        m.iloc[0, m.columns.get_loc("as_of")] += pd.Timedelta(days=1)
    else:
        m.iloc[0, m.columns.get_loc("season_id")] = 26
    with pytest.raises(ValueError):
        me.prepare_motivation_experiment(tables, replace(inputs, motivation=m))


def test_nullable_identity_keys_match_frozen_numpy_keys(monkeypatch):
    tables, inputs = synthetic_inputs()
    rows = game_rows(tables).astype({"game_id": "Int64", "team_id": "Int64"})
    assert inputs.motivation.index.get_level_values("team_id").dtype == np.dtype("int64")
    monkeypatch.setattr(me, "game_rows", lambda _: rows)
    prepared = me.prepare_motivation_experiment(tables, inputs)
    assert len(prepared.games) * 2 == len(rows)
    assert prepared.games.home_team_id.dtype == np.dtype("int64")


@pytest.mark.parametrize("key", ["game_id", "team_id"])
def test_na_identity_keys_are_rejected_before_normalization(monkeypatch, key):
    tables, inputs = synthetic_inputs()
    rows = game_rows(tables).astype({"game_id": "Int64", "team_id": "Int64"})
    rows.loc[0, key] = pd.NA
    monkeypatch.setattr(me, "game_rows", lambda _: rows)
    with pytest.raises(ValueError, match="NA in team-game identity"):
        me.prepare_motivation_experiment(tables, inputs)


def test_holdout_table_is_rejected_before_projection(monkeypatch):
    tables, inputs = synthetic_inputs()
    table = tables.team_matches.copy(deep=True)
    table.iloc[0, table.columns.get_loc("season_id")] = 26
    monkeypatch.setattr(me, "game_rows", lambda _: pytest.fail("Holdout reached projection"))
    with pytest.raises(ValueError, match="outside development 8-25 before preparation"):
        me.prepare_motivation_experiment(replace(tables, team_matches=table), inputs)


def test_weights_and_elo_only_sign_guard_are_passed(monkeypatch):
    observed = {}

    def fit(x, y, weights, *, nonnegative_first):
        observed.update(x=x, y=y, weights=weights, guard=nonnegative_first)
        return OrderedFit(1, -1, 0.5, -10, True, 3, False, 1e-9, (-0.4,))

    monkeypatch.setattr(me, "fit_ordered_logit", fit)
    train = pd.DataFrame(
        dict(
            elo_diff=[100, 0, -100], L2_m_diff=[1, 0, -1], season_id=[11, 8, 10], outcome=[0, 1, 2]
        )
    )
    model = me.MotivationOrderedPredictor(me.MotivationConfiguration("L2-diff", 4))
    model.fit(train)
    np.testing.assert_allclose(observed["weights"], [1, 0.5 ** (3 / 4), 0.5 ** (1 / 4)])
    assert observed["guard"] is True and model.fit_result.additional_betas == (-0.4,)
    # Half-life four: the weight halves four seasons behind the preceding-season anchor.
    np.testing.assert_allclose(me.time_weights([11, 7], 12, 4), [1, 0.5])


def test_c0_one_column_path_matches_ordered_elo_exactly():
    rng = np.random.default_rng(1711)
    n = 1200
    strength = rng.normal(size=n) * 100
    probabilities = me.ordered_probabilities(
        OrderedFit(2.5, -1, 0.5, 0, True, 0, False, 0), strength / 400
    )
    train = pd.DataFrame(
        dict(
            elo_diff=strength,
            season_id=np.repeat([8, 9, 10], n // 3),
            outcome=[rng.choice(3, p=p) for p in probabilities],
        )
    )
    for half_life in (None, 4):
        a = me.MotivationOrderedPredictor(me.MotivationConfiguration("C0", half_life))
        b = OrderedEloPredictor(half_life)
        a.fit(train)
        b.fit(train)
        assert a.fit_result == b.fit_result
        games = train[["elo_diff", "season_id"]].copy()
        games["season_id"] = 11
        np.testing.assert_array_equal(a.predict(games), b.predict(games))


def test_paired_statistics_weighting_deep_se_and_four_of_five():
    pattern = {s: 0.001 * (s - 18) for s in me.VALIDATION_SEASONS}
    pattern[11] = 0.8
    pattern[21] = -0.001
    candidate = evaluation(pattern)
    reference = evaluation()
    result = me.paired_statistics(candidate, reference, "log_loss", me.DEEP_SEASONS)
    d = np.array([pattern[s] for s in me.DEEP_SEASONS])
    counts = np.array([1 + s % 3 for s in me.DEEP_SEASONS])
    assert result["mean_improvement"] == pytest.approx(np.average(d, weights=counts))
    assert result["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(14))
    assert result["folds_better"] == sum(d > 0)
    assert result["last5_folds_better"] == 4 and result["folds"] == 14
    assert (
        me.paired_statistics(candidate, reference, "log_loss", me.VALIDATION_SEASONS)["folds"] == 15
    )
    bad = reference.per_game.copy()
    bad.loc[0, "game_id"] = -1
    with pytest.raises(ValueError, match="same games"):
        me.paired_statistics(candidate, replace(reference, per_game=bad), "rps", me.DEEP_SEASONS)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(0.002, -0.002, True), (0.0001, -0.0001, False), (0.002, 0.003, False)],
)
def test_rule10_entry_both_directions_and_tiny_noise(left, right, expected):
    a = dict(mean_improvement=left, se=0.001)
    b = dict(mean_improvement=right, se=0.001)
    assert me.entry_disagreement(a, b) == expected
    assert me.entry_disagreement(b, a) == expected


def test_rule10_selection_is_bidirectional():
    a = dict(winners=("a",), close=("a", "b"))
    b = dict(winners=("b",), close=("b",))
    assert me.selection_disagreement(a, b) and me.selection_disagreement(b, a)
    assert not me.selection_disagreement(a, dict(winners=("b",), close=("a", "b")))


def test_full_rules_eight_four_six_and_no_automatic_selection():
    runs = []
    for config in me.motivation_configurations():
        pattern = None
        if config.name == "L2-diff":
            pattern = {s: 0.001 for s in me.VALIDATION_SEASONS}
        if config.name == "Lc-sides":
            pattern = {s: (0.0015 if s >= 21 else -0.001) for s in me.VALIDATION_SEASONS}
        runs.append(toy_run(config, pattern=pattern))
    comparison, review = me.motivation_comparisons(runs)
    row = comparison.loc[
        comparison.config_id.eq("L2-diff")
        & comparison.scope.eq("deep")
        & comparison.metric.eq("log_loss")
    ].iloc[0]
    assert row.passes_rule8 and row.folds == 14 and row.last5_folds_better == 5
    assert review["rule4_proposals"]["log_loss"] == "L2-diff"
    assert review["rule6"] and review["selected_configuration"] is None
    assert review["status"] == "מחכה לעמית"
    assert not review["rule10_selection"]
    assert comparison.loc[comparison.config_id.eq("L2-diff-hl4"), "reference"].eq("C0-hl4").all()


def test_absolute_leakage_stops_without_printing_score(capsys):
    result = evaluation({s: 0.13 for s in me.VALIDATION_SEASONS})
    with pytest.raises(RuntimeError, match="without displaying") as error:
        me.check_leakage(result, evaluation())
    assert "0.97" not in str(error.value) and "0.006" not in str(error.value)
    assert capsys.readouterr().out == ""


def test_paired_improvement_flags_without_stopping_or_disclosing_score(caplog, capsys):
    result = evaluation({s: 0.006 for s in me.VALIDATION_SEASONS})
    assert me.check_leakage(result, evaluation()) is True
    assert "automatic audit" in caplog.text
    assert "0.006" not in caplog.text and "1.094" not in caplog.text
    assert capsys.readouterr().out == ""


def test_zero_column_discrepancy_stops_without_scores():
    base = evaluation()
    frame = base.per_game.copy()
    frame[["p_home", "p_draw", "p_away"]] = [0.4, 0.3, 0.3]
    base = replace(base, per_game=frame)
    altered = frame.copy()
    altered[["p_home", "p_away"]] += [0.01, -0.01]
    with pytest.raises(RuntimeError, match="zero-column audit failed"):
        me.zero_column_check(replace(base, per_game=altered), base, 1e-8)
    assert me.zero_column_check(base, base, 1e-8) == 0


def test_joint_placebo_is_seeded_and_preserves_season_stage_groups():
    tables, inputs = synthetic_inputs()
    m = inputs.motivation.copy()
    m["leverage_max"] = np.arange(len(m)) / len(m)
    m["title_lev"] = m.leverage_max * 0.5
    m["cut_lev"] = m.leverage_max * 0.25
    prepared = me.prepare_motivation_experiment(tables, replace(inputs, motivation=m))
    a = me.shuffled_motivation(m, prepared.participants, np.random.default_rng(14112026))
    b = me.shuffled_motivation(m, prepared.participants, np.random.default_rng(14112026))
    pd.testing.assert_frame_equal(a, b)
    keys = prepared.participants.set_index(["game_id", "team_id"]).sort_index()
    cols = ["nothing_to_play", "leverage_max", "title_lev", "cut_lev", "survival_lev"]
    for _, part in keys.groupby(["season_id", "stage"], observed=True):
        original = m.loc[part.index, cols].sort_values("leverage_max").reset_index(drop=True)
        shuffled = a.loc[part.index, cols].sort_values("leverage_max").reset_index(drop=True)
        pd.testing.assert_frame_equal(original, shuffled)
    pd.testing.assert_frame_equal(a.drop(columns=cols), m.sort_index().drop(columns=cols))


def test_decomposition_sums_to_overall_improvement():
    tables, inputs = synthetic_inputs()
    m = inputs.motivation.copy()
    rows = game_rows(tables).loc[lambda f: f.season_id.eq(11)]
    ids = rows.game_id.unique()
    first = rows.loc[rows.game_id.eq(ids[0]) & rows.is_home].iloc[0]
    m.loc[(first.game_id, first.team_id), "nothing_to_play"] = True
    m.loc[m.index.get_level_values("game_id") == ids[1], "nothing_to_play"] = True
    prepared = me.prepare_motivation_experiment(tables, replace(inputs, motivation=m))
    frame = prepared.games.loc[
        prepared.games.season_id.isin(me.VALIDATION_SEASONS), ["game_id", "season_id"]
    ].copy()
    frame["fold"] = frame.season_id - 10
    frame["outcome"], frame["log_loss"], frame["rps"] = 0, 1.1, 0.24
    for side, value in [("home", 0.4), ("draw", 0.3), ("away", 0.3)]:
        frame["p_" + side] = value
    folds, summary = summarize(frame)
    ref_eval = EvaluationResult(frame, folds, summary, "walk_forward", "synthetic")
    changed = frame.copy()
    changed["log_loss"] -= np.arange(len(frame)) * 1e-6
    changed["rps"] -= 1e-5
    folds, summary = summarize(changed)
    ref = toy_run(me.MotivationConfiguration("C0"))
    candidate = toy_run(me.MotivationConfiguration("L2-diff"))
    decomposition, top = me.audit_decomposition(
        prepared,
        replace(
            candidate,
            evaluation=replace(ref_eval, per_game=changed, per_fold=folds, summary=summary),
        ),
        replace(ref, evaluation=ref_eval),
    )
    all_rows = decomposition.loc[decomposition.scope.eq("all")]
    assert set(all_rows.group) == {"one_dead", "both_dead", "neither_dead"}
    assert all_rows.log_loss_contribution.sum() == pytest.approx(
        (frame.log_loss - changed.log_loss).mean()
    )
    assert all_rows.rps_contribution.sum() == pytest.approx(1e-5)
    assert len(top) == 10


def test_failed_timing_audit_stops(monkeypatch):
    run = toy_run(me.MotivationConfiguration("L2-diff"))
    ref = toy_run(me.MotivationConfiguration("C0"))
    prepared = me.MotivationExperimentInput(
        pd.DataFrame(), "a" * 64, pd.DataFrame(), pd.DataFrame()
    )
    monkeypatch.setattr(me, "audit_decomposition", lambda *a: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(me, "_audit_evaluation", lambda *a: run.evaluation)
    monkeypatch.setattr(me, "zero_column_check", lambda *a: 0.0)
    monkeypatch.setattr(me, "audit_motivation_timing", lambda *a: {"passed": False})
    with pytest.raises(RuntimeError, match="timing audit failed"):
        me.audit_motivation_configuration(prepared, run, ref)


def test_run_scope_and_target_projection(monkeypatch):
    tables, inputs = synthetic_inputs()
    prepared = me.prepare_motivation_experiment(tables, inputs)
    spec = me.load_motivation_experiment()
    with pytest.raises(ValueError, match="scope"):
        me.run_motivation_configuration(
            prepared, spec, me.MotivationConfiguration("L5-sides"), stage="pilot"
        )
    with pytest.raises(ValueError, match="matching decay"):
        me.run_motivation_configuration(
            prepared, spec, me.MotivationConfiguration("Comp", 4), stage="pilot"
        )

    class Spy:
        def fit(self, train):
            assert "outcome" in train

        def predict(self, games):
            assert not {"outcome", "home_goals", "away_goals", "l3_involved"} & set(games)
            return np.full((len(games), 3), 1 / 3)

    evaluate(
        Spy,
        prepared.games,
        feature_columns=("elo_diff",),
        policy=DevelopmentPolicy(training_start=8),
    )


def test_blind_persistence_and_manifest_leave_derived_untouched(tmp_path, monkeypatch, capsys):
    spec = me.load_motivation_experiment()
    configs = {c.config_id: c for c in me.motivation_configurations()}
    runs = [toy_run(configs[name], stage="pilot") for name in me.PILOT_IDS]
    safe = blind_pilot_report(runs)
    assert not {"beta", "additional_betas", "log_loss", "rps", "theta1", "delta"} & set(safe)
    assert len(safe) == 15 and safe.converged.all()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/derived").mkdir(parents=True)
    protected = tmp_path / "data/derived/keep"
    protected.write_bytes(b"frozen")
    args = (
        runs,
        spec,
        me.MotivationExperimentInput(pd.DataFrame(), "a" * 64),
        MotivationProvenance("a" * 40, "b" * 64, {"code.py": "c" * 64}),
    )
    manifest = save_motivation_experiment(
        *args,
        stage="pilot",
        baseline_verification=dict(passed=True, controls={"C0": {}, "C0-hl4": {}}),
    )
    meta = verify_calibration(manifest)
    assert meta["pilot_blind"] and meta["fold_rows"] == 15
    assert (
        list((tmp_path / "data/derived").iterdir()) == [protected]
        and protected.read_bytes() == b"frozen"
    )
    assert capsys.readouterr().out == ""
    assert set(pd.read_csv(manifest.parent / "comparison.csv").status) == {
        "not_evaluated_blind_pilot"
    }
    with pytest.raises(FileExistsError):
        save_motivation_experiment(
            *args,
            stage="pilot",
            baseline_verification=dict(passed=True, controls={"C0": {}, "C0-hl4": {}}),
        )
    (manifest.parent / "blind_report.csv").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_calibration(manifest)


def test_new_loader_reads_development_only(monkeypatch):
    spec = me.load_motivation_experiment()
    original = pd.read_parquet
    calls = []

    def guarded(path, **kwargs):
        calls.append(kwargs.get("filters"))
        assert kwargs["filters"] == [("season_id", "in", list(range(8, 26)))]
        return pd.DataFrame()

    # Hash/sidecar verification is synthetic here; no real data are read by this test.
    contents = {Path(v["path"]): v for v in spec.raw["inputs"].values()}
    original_bytes = Path.read_bytes
    original_text = Path.read_text

    class Digest:
        def __init__(self, value):
            self.value = value

        def hexdigest(self):
            return self.value

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: contents[p]["sha256"].encode() if p in contents else original_bytes(p),
    )
    monkeypatch.setattr(me.hashlib, "sha256", lambda b: Digest(b.decode()))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, **kwargs: (
            json.dumps(
                dict(
                    seasons=list(range(8, 26)),
                    artifact_sha256=me.INPUTS["motivation" if "motivation" in str(p) else "elo"][
                        "sha256"
                    ],
                )
            )
            if p in contents
            else original_text(p, **kwargs)
        ),
    )
    monkeypatch.setattr(pd, "read_parquet", guarded)
    me.load_motivation_inputs(spec)
    assert len(calls) == 2
    assert original is not guarded


@pytest.fixture(scope="module")
def real_prepared(real_tables: AnalysisTables):
    spec = me.load_motivation_experiment()
    if not all(Path(v["path"]).exists() for v in spec.raw["inputs"].values()):
        pytest.skip("Frozen EXP-004 inputs are local only")
    return me.prepare_motivation_experiment(real_tables, me.load_motivation_inputs(spec)), spec


@pytest.mark.real_data
def test_both_c0_controls_reproduce_exp001(real_prepared):
    prepared, spec = real_prepared
    c0 = me.run_motivation_configuration(
        prepared, spec, me.MotivationConfiguration("C0"), stage="baseline"
    )
    weighted = me.run_motivation_configuration(
        prepared, spec, me.MotivationConfiguration("C0", 4), stage="baseline", reference=c0
    )
    result = me.verify_motivation_baselines(c0, weighted)
    assert result["passed"]
    assert result["controls"]["C0"]["max_probability_error"] <= 1e-9
    assert result["controls"]["C0"]["max_fit_error"] <= 1e-9
