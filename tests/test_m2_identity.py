"""Q29 historical identity encoding, explicit opt-in, persistence and real counts."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from ligat.evaluate.backtest import DevelopmentPolicy
from ligat.evaluate.goal_inputs import prepare_goal_data
from ligat.evaluate.team_goal_backtest import evaluate_team_goals
from ligat.features.prematch_v2 import validate_prematch_v2_frame
from ligat.models.m2_identity import IDENTITY_COLUMNS, TeamIdentity
from ligat.models.m2_xgb import STAGE_DTYPE, XGBGoalConfig, XGBGoalModel
from tests.test_goal_inputs import goal_sources as goal_sources
from tests.test_goal_inputs import prepare
from tests.test_m2_xgb import config, sample


@pytest.fixture
def identity_rows():
    rows = []
    seasons = {
        8: [(1, 2)],
        9: [(1, 3)],
        10: [(1, 4), (2, 4)],
        11: [(4, 1)],
        12: [(1, 2)],
        13: [(3, 1)],
    }
    for season, games in seasons.items():
        for number, (home, away) in enumerate(games):
            for team, opponent in ((home, away), (away, home)):
                rows.append(
                    {
                        "game_id": season * 100 + number,
                        "team_id": team,
                        "opponent_id": opponent,
                        "season_id": season,
                    }
                )
    context = pd.DataFrame(rows).set_index(["game_id", "team_id"], drop=False)
    X = context.loc[:, list(IDENTITY_COLUMNS)].copy()
    X["elo_diff"] = np.linspace(-200, 200, len(X))
    y = pd.Series(np.resize([2, 0, 1], len(X)), index=X.index, name="goals_scored")
    return X, y, context, (1, 2, 3, 4, 5)


def test_first_season_unknown_rule_and_season_eight_exception_apply_to_both_sides(identity_rows):
    X, _, context, vocabulary = identity_rows
    state = TeamIdentity.from_context(context, team_ids=vocabulary)
    encoded = state.transform(X, context=context)
    assert encoded.loc[context.season_id.eq(8), "team_id"].tolist() == ["team:1", "team:2"]
    assert encoded.loc[context.season_id.eq(8), "opponent_id"].tolist() == ["team:2", "team:1"]
    first = context.season_id.eq(10) & context.team_id.eq(4)
    assert encoded.loc[first, "team_id"].tolist() == ["unknown", "unknown"]
    assert (
        encoded.loc[context.season_id.eq(10) & X.opponent_id.eq(4), "opponent_id"]
        .eq("unknown")
        .all()
    )
    assert (
        encoded.loc[context.season_id.eq(11) & context.team_id.eq(4), "team_id"].eq("team:4").all()
    )


@pytest.mark.parametrize("season", [9, 10, 11, 13])
def test_training_and_prediction_use_the_same_earlier_season_rule(identity_rows, season):
    X, _, context, vocabulary = identity_rows
    full = TeamIdentity.from_context(context, team_ids=vocabulary).transform(X, context=context)
    train, predict = context.season_id.lt(season), context.season_id.eq(season)
    prefix = TeamIdentity.from_context(context.loc[train], team_ids=vocabulary)
    actual = prefix.transform(X.loc[predict], context=context.loc[predict])
    pd.testing.assert_frame_equal(actual, full.loc[predict])


def test_returning_team_is_known_after_several_absent_seasons(identity_rows):
    X, _, context, vocabulary = identity_rows
    train = context.season_id.lt(13)
    state = TeamIdentity.from_context(context.loc[train], team_ids=vocabulary)
    assert dict(state.first_seen)[3] == 9
    actual = state.transform(X.loc[~train], context=context.loc[~train])
    assert set(actual.team_id) == {"team:1", "team:3"}
    assert set(actual.opponent_id) == {"team:1", "team:3"}


def test_fixed_string_vocabulary_is_not_evidence_of_familiarity(identity_rows):
    X, _, context, vocabulary = identity_rows
    original = X.copy(deep=True)
    state = TeamIdentity.from_context(context.loc[context.season_id.lt(10)], team_ids=vocabulary)
    result = state.transform(X, context=context)
    for column in IDENTITY_COLUMNS:
        assert result[column].dtype == state.dtype
        assert all(isinstance(label, str) for label in result[column].cat.categories)
        assert "team:5" in result[column].cat.categories
        assert not result[column].isna().any()
    assert result.loc[context.team_id.eq(4), "team_id"].eq("unknown").all()
    pd.testing.assert_frame_equal(X, original)
    with pytest.raises(FrozenInstanceError):
        state.first_seen = ()


def test_identity_metadata_roundtrip_preserves_original_history_and_unused_categories(
    identity_rows,
):
    X, _, context, vocabulary = identity_rows
    state = TeamIdentity.from_context(context, team_ids=vocabulary)
    loaded = TeamIdentity.from_metadata(json.loads(json.dumps(state.to_metadata())))
    assert loaded == state and loaded.dtype == state.dtype
    pd.testing.assert_frame_equal(
        loaded.transform(X, context=context), state.transform(X, context=context)
    )


def test_new_team_outside_fixed_vocabulary_is_unknown_and_cannot_update_history(identity_rows):
    X, _, context, vocabulary = identity_rows
    train = context.season_id.lt(13)
    state = TeamIdentity.from_context(context.loc[train], team_ids=vocabulary)
    before = state.to_metadata()
    future = context.loc[~train].rename(index={3: 99}, level="team_id").copy()
    future["team_id"] = future.team_id.replace({3: 99})
    future["season_id"] = 14
    features = X.loc[~train].rename(index={3: 99}, level="team_id").copy()
    features[list(IDENTITY_COLUMNS)] = features[list(IDENTITY_COLUMNS)].replace({3: 99})
    result = state.transform(features, context=future)
    assert result.loc[future.team_id.eq(99), "team_id"].eq("unknown").all()
    assert result.loc[features.opponent_id.eq(99), "opponent_id"].eq("unknown").all()
    assert state.to_metadata() == before
    assert "team:99" not in result.team_id.cat.categories


@pytest.mark.parametrize(
    "damage", ["team", "season", "missing_id", "fractional_id", "missing_context", "vocabulary"]
)
def test_identity_rejects_invalid_keys_seasons_and_incomplete_requests(identity_rows, damage):
    X, _, context, vocabulary = identity_rows
    if damage == "team":
        context = context.assign(team_id=999)
    elif damage == "season":
        context = context.assign(season_id=7)
    elif damage == "missing_id":
        X = X.drop(columns="opponent_id")
    elif damage == "fractional_id":
        X = X.assign(opponent_id=0.5)
    elif damage == "missing_context":
        context = context.drop(columns="season_id")
    else:
        vocabulary = (1, 2)
    with pytest.raises(ValueError):
        state = TeamIdentity.from_context(context, team_ids=vocabulary)
        state.transform(X, context=context)


def test_xgboost_identity_requires_both_explicit_columns_and_opt_in(identity_rows):
    X, y, context, vocabulary = identity_rows
    with pytest.raises(ValueError, match="opt-in"):
        XGBGoalModel(config()).fit(X, y, context=context)
    with pytest.raises(ValueError, match="both explicitly"):
        XGBGoalModel(config(), team_ids=vocabulary).fit(
            X.drop(columns="team_id"), y, context=context
        )
    model = XGBGoalModel(config(), team_ids=vocabulary).fit(X, y, context=context)
    assert model.identity is not None
    assert model.predict(X, context=context).gt(0).all()


def test_identity_and_predictions_are_exact_after_reload(identity_rows, tmp_path):
    X, y, context, vocabulary = identity_rows
    train = context.season_id.lt(11)
    model = XGBGoalModel(config(), team_ids=vocabulary).fit(
        X.loc[train], y.loc[train], context=context.loc[train]
    )
    path = tmp_path / "identities.json"
    model.save(path)
    loaded = XGBGoalModel.load(path)
    assert loaded.identity == model.identity
    assert dict(loaded.identity.first_seen)[4] == 10
    for mask in (train, ~train):
        pd.testing.assert_frame_equal(
            loaded.identity.transform(X.loc[mask], context=context.loc[mask]),
            model.identity.transform(X.loc[mask], context=context.loc[mask]),
        )
        pd.testing.assert_series_equal(
            loaded.predict(X.loc[mask], context=context.loc[mask]),
            model.predict(X.loc[mask], context=context.loc[mask]),
            check_exact=True,
        )
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    assert meta["identity"] == model.identity.to_metadata()


@pytest.mark.parametrize("damage", ["history", "categories", "removed"])
def test_mismatched_saved_identity_is_rejected(identity_rows, tmp_path, damage):
    X, y, context, vocabulary = identity_rows
    path = tmp_path / "model.json"
    XGBGoalModel(config(), team_ids=vocabulary).fit(X, y, context=context).save(path)
    sidecar = path.with_suffix(".metadata.json")
    meta = json.loads(sidecar.read_text())
    if damage == "history":
        meta["identity"]["first_seen"][-1][1] = 8
    elif damage == "categories":
        meta["identity"]["categories"].reverse()
    else:
        del meta["identity"]
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        XGBGoalModel.load(path)


def test_no_ids_path_remains_exactly_the_c1a_native_reference(tmp_path):
    X, y, context = sample()
    model = XGBGoalModel(config()).fit(X, y, context=context)
    parameters = config()
    native = XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        enable_categorical=True,
        max_depth=parameters.max_depth,
        min_child_weight=parameters.min_child_weight,
        n_estimators=parameters.n_estimators,
        learning_rate=parameters.learning_rate,
        subsample=parameters.subsample,
        colsample_bytree=parameters.colsample_bytree,
        reg_lambda=parameters.reg_lambda,
        max_delta_step=parameters.max_delta_step,
        random_state=parameters.seed,
    )
    prepared = X.assign(is_home=X.is_home.astype(float), stage=X.stage.astype(STAGE_DTYPE))
    native.fit(prepared, y)
    np.testing.assert_array_equal(
        model.predict(X, context=context).to_numpy(), native.predict(prepared)
    )
    assert model.identity is None
    path = tmp_path / "no_ids.json"
    model.save(path)
    assert "identity" not in json.loads(path.with_suffix(".metadata.json").read_text())


def test_labels_are_readable_distinguish_seeds_presets_and_close_float_values():
    cfg = config(
        max_depth=4, n_estimators=400, min_child_weight=10, learning_rate=0.05, subsample=0.8
    )
    assert cfg.label == "d4 · n400 · lr0.05 · mcw10 · ss0.8 · cs0.8 · rl1 · seed0"
    assert XGBGoalConfig.library_default().label == "ברירת המחדל של XGBoost"
    variants = [
        cfg,
        replace(cfg, seed=1),
        replace(cfg, learning_rate=np.nextafter(0.05, 1)),
        XGBGoalConfig.library_default(),
        XGBGoalConfig.library_default(seed=1),
    ]
    assert len({v.label for v in variants}) == len(variants)
    assert "seed1" in XGBGoalConfig.library_default(seed=1).label


def test_f6_flows_through_existing_goal_adapter_and_judge(goal_sources):
    data = prepare(goal_sources, ("elo_diff", "team_id", "opponent_id"))
    result = evaluate_team_goals(
        lambda: XGBGoalModel(config(n_estimators=8), team_ids=(1, 2, 3, 4)),
        data,
        feature_columns=("elo_diff", "team_id", "opponent_id"),
        policy=DevelopmentPolicy(training_start=8, validation_seasons=(15,)),
        half_life=2,
    )
    assert result.per_team["lambda"].gt(0).all()
    assert result.per_team.season_id.eq(15).all()


@pytest.mark.real_data
def test_real_unknown_team_seasons_and_rows_match_claude_exactly(real_tables):
    path = Path("data/derived/prematch/prematch_v2.parquet")
    if not path.exists():
        pytest.skip("Published prematch v2 is local only")
    seasons = tuple(range(8, 26))
    prematch = pd.read_parquet(path, filters=[("season_id", "in", list(seasons))])
    validate_prematch_v2_frame(prematch)
    data = prepare_goal_data(
        prematch, real_tables.match_table, feature_columns=IDENTITY_COLUMNS, seasons=seasons
    )
    vocabulary = tuple(sorted(int(i) for i in prematch.index.get_level_values("team_id").unique()))
    assert len(vocabulary) == 28
    identity = TeamIdentity.from_context(data.context, team_ids=vocabulary)
    encoded = identity.transform(data.X, context=data.context)
    unknown = encoded.team_id.eq("unknown")
    expected = {9: 2, 11: 6, 12: 1, 13: 2, 20: 1, 21: 1, 23: 2, 24: 1}
    report = []
    for season in seasons:
        mask = data.context.season_id.eq(season)
        unknown_rows = data.context.loc[mask & unknown]
        count = int(unknown_rows.team_id.nunique())
        assert count == expected.get(season, 0), season
        report.append(
            {
                "season_id": season,
                "unknown_teams": count,
                "unknown_rows": len(unknown_rows),
                "all_rows": int(mask.sum()),
                "claude_teams": expected.get(season, 0),
            }
        )
    assert sum(r["unknown_teams"] for r in report) == 16
    assert int(unknown.sum()) == 550 and len(unknown) == 8630
    assert round(100 * float(unknown.mean()), 2) == 6.37
    print(
        json.dumps(
            {
                "seasons": report,
                "unknown_team_seasons": 16,
                "unknown_rows": int(unknown.sum()),
                "all_rows": len(unknown),
                "unknown_percent": 100 * float(unknown.mean()),
            }
        )
    )
