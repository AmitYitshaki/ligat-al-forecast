"""GLM recipe, rank, ruler reconstruction and blind real-data engineering checks."""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.goal_reporting import readable_goal_metrics
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.features.registry import M2_BASE_COLUMNS
from ligat.models.form_models import FormPoissonPredictor
from ligat.models.m2_glm import GLMGoalConfig, GLMGoalModel, glm_design
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare


def sample(n=2000):
    rng = np.random.default_rng(14303)
    X = pd.DataFrame(
        {
            "is_home": rng.integers(0, 2, n).astype(bool),
            "elo_diff": rng.normal(0, 150, n),
            "elo_pre": rng.normal(1500, 100, n),
            "gf_venue_ewma5": rng.uniform(0.3, 2.8, n),
            "opp_ga_venue_ewma5": rng.uniform(0.3, 2.8, n),
            "value_avg_log_ratio": rng.normal(size=n),
            "stage": np.resize(["regular", "upper", "middle", "lower"], n),
            "rounds_to_stage_end": rng.integers(1, 30, n),
            "promoted_in": rng.random(n) < 0.2,
            "opp_promoted_in": rng.random(n) < 0.2,
        }
    )
    design = glm_design(X, GLMGoalConfig("base", 9))
    beta = np.zeros(design.shape[1])
    beta[0:3] = [-0.1, 0.05, 0.3]
    y = pd.Series(rng.poisson(np.exp(design.to_numpy() @ beta)), index=X.index, name="goals_scored")
    context = pd.DataFrame({"season_id": rng.integers(8, 12, n)}, index=X.index)
    return X, y, context


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("recipe", ["r0", "r1", "ruler", "base"])
def test_coefficients_agree_with_statsmodels(recipe, weighted):
    X, y, context = sample()
    cfg = GLMGoalConfig(recipe, 9 if recipe == "base" else 8)
    weights = pd.Series(np.linspace(0.3, 2, len(X)), index=X.index) if weighted else None
    model = GLMGoalModel(cfg).fit(X, y, weights, context=context)
    selected = context.season_id.ge(cfg.training_start)
    design = glm_design(X.loc[selected], cfg)
    native = sm.GLM(
        y.loc[selected],
        design,
        family=sm.families.Poisson(),
        freq_weights=None if weights is None else weights.loc[selected],
    ).fit()
    np.testing.assert_allclose(model.coefficients, native.params, rtol=0, atol=1e-6)


def test_absent_dummy_is_recorded_as_zero_and_predicts_regular():
    X, y, context = sample()
    # Randomize season/stage alignment, then remove middle explicitly.
    context["season_id"] = 9
    train = X.stage.ne("middle")
    model = GLMGoalModel(GLMGoalConfig("base", 9)).fit(
        X.loc[train], y.loc[train], context=context.loc[train]
    )
    assert model.dropped_columns == ("stage_middle",)
    assert model.coefficients.stage_middle == 0
    middle = X.loc[~train]
    regular = middle.assign(stage="regular")
    pd.testing.assert_series_equal(
        model.predict(middle, context=context.loc[~train]),
        model.predict(regular, context=context.loc[~train]),
    )


def test_rank_deficiency_names_columns():
    X, y, context = sample()
    X["elo_diff"] = X.is_home.astype(float)
    with pytest.raises(ValueError, match="dependent columns.*(elo_diff|is_home)"):
        GLMGoalModel(GLMGoalConfig("r1")).fit(X, y, context=context)


def test_training_start_and_unit_weights_and_persistence(tmp_path):
    X, y, context = sample()
    cfg = GLMGoalConfig("r1", 9)
    a = GLMGoalModel(cfg).fit(X, y, context=context)
    b = GLMGoalModel(cfg).fit(X, y, pd.Series(1.0, index=X.index), context=context)
    pd.testing.assert_series_equal(a.coefficients, b.coefficients)
    changed = y.copy()
    changed.loc[context.season_id.eq(8)] += 100
    c = GLMGoalModel(cfg).fit(X, changed, context=context)
    pd.testing.assert_series_equal(a.coefficients, c.coefficients)
    assert a.training_seasons == (9, 10, 11)
    d = GLMGoalModel(cfg).fit(
        X, y, pd.Series(np.where(y > 1, 5.0, 0.2), index=X.index), context=context
    )
    assert not np.array_equal(a.coefficients, d.coefficients)
    path = tmp_path / "glm.json"
    a.save(path)
    pd.testing.assert_series_equal(
        a.predict(X, context=context), GLMGoalModel.load(path).predict(X, context=context)
    )
    saved = json.loads(path.read_text())
    saved["payload"]["coefficients"][0] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="signature"):
        GLMGoalModel.load(path)
    with pytest.raises(ValueError, match="no training rows"):
        GLMGoalModel(cfg).fit(
            X.loc[context.season_id.eq(8)],
            y.loc[context.season_id.eq(8)],
            context=context.loc[context.season_id.eq(8)],
        )


def test_nonfinite_design_and_configuration_and_labels():
    X, y, context = sample()
    X.loc[0, "elo_diff"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        GLMGoalModel(GLMGoalConfig("r1")).fit(X, y, context=context)
    with pytest.raises(ValueError):
        GLMGoalConfig("base", 8)
    with pytest.raises(ValueError):
        GLMGoalConfig("other")
    cfg = GLMGoalConfig("base", 9)
    assert cfg.config_id != replace(cfg, recipe="r1").config_id
    assert "בסיס" in cfg.label


def test_glm_works_through_judge_with_half_life(goal_sources):
    cfg = GLMGoalConfig("r0")
    result = evaluate_team_goals(
        lambda: GLMGoalModel(cfg),
        prepare(goal_sources, ("is_home",)),
        feature_columns=cfg.feature_columns,
        half_life=2,
    )
    assert result.per_team["lambda"].gt(0).all()
    assert result.per_team.half_life.eq(2).all()


@pytest.fixture(scope="module")
def real_goal_data(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Frozen prematch is local only")
    seasons = tuple(range(8, 26))
    frame = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    return prepare_goal_data(
        frame, real_tables.match_table, feature_columns=M2_BASE_COLUMNS, seasons=seasons
    )


@pytest.mark.real_data
def test_real_reference_rulers_reproduce_closed_experiments(real_goal_data):
    data = real_goal_data
    evaluations = {}
    for recipe, start in (("r0", 8), ("r1", 8), ("ruler", 8), ("ruler", 9)):
        cfg = GLMGoalConfig(recipe, start)
        evaluations[(recipe, start)] = evaluate_team_goals(
            lambda c=cfg: GLMGoalModel(c),
            data,
            feature_columns=cfg.feature_columns,
            policy=DevelopmentPolicy(training_start=8),
        )
    r0, r1 = evaluations[("r0", 8)], evaluations[("r1", 8)]
    expected = {
        ("r0", 8): (1.457037, 1e-6),
        ("r1", 8): (1.410383, 1e-6),
        ("ruler", 8): (1.409455, 1e-6),
        ("ruler", 9): (1.41010, 1e-5),
    }
    report = []
    for key, evaluation in evaluations.items():
        readable = readable_goal_metrics(evaluation, evaluation, r0=r0, r1=r1)
        actual = float(readable.summary.loc["all", "poisson_nll"])
        target, tolerance = expected[key]
        report.append(
            {"recipe": key, "expected": target, "actual": actual, "difference": actual - target}
        )
        assert actual == pytest.approx(target, abs=tolerance), report
    train = data.context.season_id.lt(15)
    cfg = GLMGoalConfig("r1")
    model = GLMGoalModel(cfg).fit(
        data.X.loc[train], data.y.loc[train], context=data.context.loc[train]
    )
    home = train & data.context.is_home
    away = train & ~data.context.is_home
    home_context = data.context.loc[home]
    game_index = pd.Index(home_context.game_id.to_numpy(), name="game_id")
    original_features = pd.DataFrame(
        {
            "season_id": home_context.season_id.to_numpy(),
            "elo_diff": data.X.loc[home, "elo_diff"].to_numpy(),
        },
        index=game_index,
    )
    away_goals = pd.Series(data.y.loc[away].to_numpy(), index=data.context.loc[away, "game_id"])
    original_targets = pd.DataFrame(
        {
            "home_goals": data.y.loc[home].to_numpy(),
            "away_goals": away_goals.reindex(game_index).to_numpy(),
        },
        index=game_index,
    )
    original = FormPoissonPredictor(with_elo=True, with_form=False, log_offset=0.1)
    original.fit(original_features, original_targets)
    legacy = original.fit_result
    np.testing.assert_allclose(model.coefficients, legacy.coefficients, rtol=0, atol=1e-8)
    print(
        json.dumps(
            {
                "reference_reconstruction": report,
                "r1_coefficient_max_difference": float(
                    np.max(np.abs(model.coefficients.to_numpy() - legacy.coefficients))
                ),
            }
        )
    )


@pytest.mark.real_data
def test_real_base_fold_eleven_absent_middle_and_fold_fifteen_blind_diagnostics(real_goal_data):
    data = real_goal_data
    cfg = GLMGoalConfig("base", 9)
    for season in (11, 12):
        train = data.context.season_id.lt(season)
        model = GLMGoalModel(cfg).fit(
            data.X.loc[train], data.y.loc[train], context=data.context.loc[train]
        )
        assert ("stage_middle" in model.dropped_columns) == (season == 11)
        if season == 11:
            middle = data.context.season_id.eq(11) & data.X.stage.eq("middle")
            assert middle.sum() == 12
            pd.testing.assert_series_equal(
                model.predict(data.X.loc[middle], context=data.context.loc[middle]),
                model.predict(
                    data.X.loc[middle].assign(stage="regular"), context=data.context.loc[middle]
                ),
            )
    train = data.context.season_id.between(9, 14)
    val = data.context.season_id.eq(15)
    started = perf_counter()
    model = GLMGoalModel(cfg).fit(
        data.X.loc[train], data.y.loc[train], context=data.context.loc[train]
    )
    elapsed = perf_counter() - started
    rates = model.predict(data.X.loc[val], context=data.context.loc[val])
    home = data.context.loc[val, "is_home"]
    assert rates.gt(0).all()
    print(
        json.dumps(
            {
                "base_engineering": {
                    "fit_seconds": elapsed,
                    "lambda_min": float(rates.min()),
                    "lambda_max": float(rates.max()),
                    "lambda_mean": float(rates.mean()),
                    "training_goal_mean": float(data.y.loc[train].mean()),
                    "home_mean": float(rates.loc[home].mean()),
                    "away_mean": float(rates.loc[~home].mean()),
                    "coefficients": model.coefficients.to_dict(),
                    "fold11_dropped": ["stage_middle"],
                }
            }
        )
    )
