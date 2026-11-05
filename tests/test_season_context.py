"""Season-fixed context, prior-only imputations and immutable derived publication."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.analysis.season_context_review import season_position_comparison
from ligat.data.repository import Repository
from ligat.features.base import FeatureBlock
from ligat.features.form import form_opening_values
from ligat.features.form_artifacts import FormProvenance
from ligat.features.registry import FEATURES, SEASON_CONTEXT_COLUMNS, validate_feature_columns
from ligat.features.season_context import (
    PROMOTED_POSITION_PCT,
    SeasonContextBlock,
    completed_sporting_positions,
    compute_season_context,
)
from ligat.features.season_context_artifacts import (
    save_season_context,
    season_context_input_fingerprint,
)
from ligat.features.table_state import game_rows, historical_groups
from ligat.simulate.rules import PlayoffGroup, sporting_table
from tests.test_form import change_score, rules, season_fixtures, tables_from


def context_tables(last=10, fixtures=None):
    tables = tables_from(season_fixtures(last) if fixtures is None else fixtures)
    rows = game_rows(tables)
    values = []
    for season, part in rows.groupby("season_id"):
        for order, team in enumerate(sorted(part.team_id.unique()), 1):
            values.append(
                dict(
                    season_id=season,
                    team_id=team,
                    avg_value_eur=order,
                    total_value_eur=10 * order,
                    promoted_in=True,  # deliberately wrong official-derived flag
                    points_official=-999,
                    final_position=99,
                )
            )
    return replace(tables, team_seasons=pd.DataFrame(values).set_index(["season_id", "team_id"]))


def select_season(features, tables, season):
    identities = game_rows(tables).set_index(["game_id", "team_id"])
    return features.loc[identities.index[identities.season_id.eq(season)]].sort_index()


@pytest.mark.parametrize("season", [8, 9, 10])
def test_changed_season_result_cannot_change_own_context_but_changes_next(season):
    tables = context_tables(11)
    fixture = next(f for f in season_fixtures(11) if f[0] == season and (season != 10 or f[3] == 7))
    before = compute_season_context(tables, rules()).features
    # Keep the synthetic retention coefficients within the existing form guard.
    changed = change_score(tables, fixture[1], fixture[5] + (-1 if season == 10 else 1))
    after = compute_season_context(changed, rules()).features
    assert_frame_equal(select_season(before, tables, season), select_season(after, tables, season))
    next_before = select_season(before, tables, season + 1).xs(1, level="team_id")
    next_after = select_season(after, tables, season + 1).xs(1, level="team_id")
    assert not next_before.prev_gd_per_game.equals(next_after.prev_gd_per_game)


def test_future_season_result_does_not_change_any_earlier_season():
    tables = context_tables(10)
    before = compute_season_context(tables, rules()).features
    fixture = next(f for f in season_fixtures(10) if f[0] == 10)
    after = compute_season_context(
        change_score(tables, fixture[1], fixture[5] + 2), rules()
    ).features
    for season in (8, 9):
        assert_frame_equal(
            select_season(before, tables, season), select_season(after, tables, season)
        )


def test_context_is_constant_per_team_season_and_matches_protocol_and_index():
    tables = context_tables()
    result = compute_season_context(tables, rules())
    block: FeatureBlock = SeasonContextBlock(rules())
    assert_frame_equal(block.compute(tables), result.features)
    identities = game_rows(tables).set_index(["game_id", "team_id"])
    joined = identities[["season_id"]].join(result.features)
    assert (
        joined.groupby(["season_id", "team_id"])[list(SEASON_CONTEXT_COLUMNS)]
        .nunique(dropna=False)
        .eq(1)
        .all()
        .all()
    )
    assert len(result.features) == len(identities)
    assert result.features.index.is_unique
    assert tuple(result.features.columns) == SEASON_CONTEXT_COLUMNS
    assert "promoted_position_mean" not in result.features
    assert result.features.league_size.eq(4).all()


def test_promoted_and_returning_club_receive_fixed_position_and_form_prior():
    tables = context_tables(11)
    result = compute_season_context(tables, rules())
    first9 = select_season(result.features, tables, 9).xs(5, level="team_id").iloc[0]
    return11 = select_season(result.features, tables, 11).xs(3, level="team_id").iloc[0]
    assert first9.promoted_in and return11.promoted_in
    assert first9.prev_position_pct == return11.prev_position_pct == PROMOTED_POSITION_PCT == 0.75
    openings = (
        form_opening_values(tables, rules())
        .query("season_id == 9 and team_id == 5 and stratum == 'all'")
        .set_index("metric")
    )
    assert first9.prev_ppm == openings.at["pts", "promoted_value"]
    assert first9.prev_gd_per_game == (
        openings.at["gf", "promoted_value"] - openings.at["ga", "promoted_value"]
    )
    assert first9.squad_value_imputed and first9.prev_season_known


def test_season_8_stays_empty_without_neutral_values():
    tables = context_tables(8)
    features = compute_season_context(tables, rules()).features
    assert features.loc[:, SEASON_CONTEXT_COLUMNS[:5]].isna().all().all()
    assert not features.prev_season_known.any()
    assert not features.promoted_in.any()
    assert not features.squad_value_imputed.any()


def test_nullable_source_identifiers_produce_the_same_canonical_form_index():
    tables = context_tables(8)
    expected = compute_season_context(tables, rules()).features
    rows = tables.team_matches.reset_index()
    rows[["game_id", "team_id"]] = rows[["game_id", "team_id"]].astype("Int64")
    nullable = replace(tables, team_matches=rows.set_index(["team_id", "game_id"]))
    result = compute_season_context(nullable, rules()).features
    assert result.index.equals(expected.index)
    assert all(str(level.dtype) == "int64" for level in result.index.levels)
    assert_frame_equal(result, expected)


def test_missing_veteran_value_is_generic_flagged_q30_without_renormalizing():
    tables = context_tables(9)
    values = tables.team_seasons.copy()
    values.loc[(8, 1), ["avg_value_eur", "total_value_eur"]] = np.nan
    result = compute_season_context(replace(tables, team_seasons=values), rules())
    season9 = select_season(result.features, tables, 9)
    veteran = season9.xs(1, level="team_id").iloc[0]
    assert not veteran.promoted_in and veteran.squad_value_imputed
    # Known prior totals: 20,30,40; lower floor(3/2)=1 => share 20/90.
    assert veteran.prev_value_share_total == pytest.approx(20 / 90)
    assert veteran.prev_value_share_avg == pytest.approx(2 / 9)
    assert result.audit.set_index("season_id").at[9, "value_imputed_teams"] == 3
    assert season9.reset_index().drop_duplicates("team_id").prev_value_share_total.sum() != 1


def test_total_and_average_imputations_are_computed_independently():
    tables = context_tables(9)
    values = tables.team_seasons.copy()
    values.loc[(8, 1), "avg_value_eur"] = np.nan
    result = compute_season_context(replace(tables, team_seasons=values), rules())
    row = select_season(result.features, tables, 9).xs(1, level="team_id").iloc[0]
    assert row.prev_value_share_avg == pytest.approx(2 / 9)
    assert row.prev_value_share_total == pytest.approx(0.1)
    assert row.squad_value_imputed and not row.promoted_in


def test_halved_table_points_do_not_enter_ppm_or_goal_difference():
    tables = context_tables(12)
    games = tables.team_matches.copy()
    game_ids = games.index.get_level_values("game_id")
    upper = (
        games.season_id.eq(11)
        & games.index.get_level_values("team_id").isin([1, 2])
        & games.opponent_id.isin([1, 2])
    )
    lower = (
        games.season_id.eq(11)
        & games.index.get_level_values("team_id").isin([3, 4])
        & games.opponent_id.isin([3, 4])
    )
    games.loc[upper, "stage"] = "upper"
    games.loc[lower, "stage"] = "lower"
    for game in (1102, 1105, 1112, 1115):
        games.loc[game_ids == game, "date"] = pd.Timestamp(
            "2009-09-01" if game in (1102, 1112) else "2009-09-02", tz="Asia/Jerusalem"
        )
    tables = replace(tables, team_matches=games)
    halved = rules()
    halved[11] = replace(
        halved[11],
        regular_rounds=4,
        groups=(PlayoffGroup("upper", 2, 2), PlayoffGroup("lower", 2, 2)),
        points_halved=True,
    )
    unhalved = {**halved, 11: replace(halved[11], points_halved=False)}
    before = compute_season_context(tables, halved).features
    after = compute_season_context(tables, unhalved).features
    for column in ("prev_ppm", "prev_gd_per_game"):
        assert select_season(before, tables, 12)[column].equals(
            select_season(after, tables, 12)[column]
        )
    rows = game_rows(tables).query("season_id == 11")
    table = sporting_table(rows, [1, 2, 3, 4], halved[11], groups=historical_groups(rows))
    assert table.points.lt(3 * table.wins + table.draws).any()
    expected = np.where(
        rows.goals_scored.gt(rows.goals_conceded),
        3,
        np.where(rows.goals_scored.eq(rows.goals_conceded), 1, 0),
    )
    means = rows.assign(earned=expected).groupby("team_id").earned.mean()
    assert select_season(before, tables, 12).xs(1, level="team_id").prev_ppm.eq(means.loc[1]).all()


def test_official_deductions_positions_and_flags_are_never_read(monkeypatch):
    tables = context_tables()
    before = compute_season_context(tables, rules()).features
    values = tables.team_seasons.assign(
        points_official=12345,
        points_model=-12345,
        deduction=-100,
        final_position=-1,
        promoted_in=False,
    )
    monkeypatch.setattr(
        Repository, "read", lambda *a, **k: pytest.fail("unexpected repository read")
    )
    monkeypatch.setattr(
        Repository, "read_filtered", lambda *a, **k: pytest.fail("unexpected repository read")
    )
    after = compute_season_context(replace(tables, team_seasons=values), rules()).features
    assert_frame_equal(before, after)


def test_current_season_squad_values_cannot_change_current_context():
    tables = context_tables()
    before = compute_season_context(tables, rules()).features
    values = tables.team_seasons.copy()
    values.loc[10, "avg_value_eur"] = 10000
    values.loc[10, "total_value_eur"] = 100000
    after = compute_season_context(replace(tables, team_seasons=values), rules()).features
    assert_frame_equal(before, after)


def test_expanding_promoted_position_mean_uses_past_only_and_stays_outside_features():
    tables = context_tables(12)
    result = compute_season_context(tables, rules())
    audit = result.audit.set_index("season_id")
    assert audit.at[11, "previous_promoted_events"] == 4
    positions = completed_sporting_positions(tables, rules())
    first_four = [(9, 5), (9, 6), (10, 7), (10, 8)]
    assert audit.at[11, "promoted_position_mean"] == pytest.approx(
        np.mean([positions.at[key, "position_pct"] for key in first_four])
    )
    assert audit.at[11, "promoted_position_mean"] != 0.75
    prior = [(9, 5), (9, 6), (10, 7), (10, 8), (11, 3), (11, 4)]
    assert audit.at[12, "promoted_position_mean"] == pytest.approx(
        np.mean([positions.at[key, "position_pct"] for key in prior])
    )
    fixture = next(f for f in season_fixtures(12) if f[0] == 12)
    changed = compute_season_context(change_score(tables, fixture[1], fixture[5] + 1), rules())
    assert_frame_equal(result.audit, changed.audit)


@pytest.mark.parametrize("season", [7, 26, 28])
def test_season_outside_development_is_rejected(season):
    tables = context_tables(8)
    changed = replace(tables, team_matches=tables.team_matches.assign(season_id=season))
    with pytest.raises(ValueError, match="8–25"):
        compute_season_context(changed, rules())


def test_held_out_value_rows_are_rejected_even_with_development_matches():
    tables = context_tables(8)
    values = tables.team_seasons.reset_index().assign(season_id=26)
    with pytest.raises(ValueError, match="8–25"):
        compute_season_context(replace(tables, team_seasons=values), rules())


def test_registry_has_exact_context_statuses_and_rejects_unregistered_column():
    features = compute_season_context(context_tables(8), rules()).features
    validate_feature_columns(features, "season_context")
    for column in SEASON_CONTEXT_COLUMNS:
        spec = FEATURES[column]
        assert spec.models == ("M2", "M4", "M5") and spec.block == "season_context"
        assert spec.status == ("🔬" if column in SEASON_CONTEXT_COLUMNS[:5] else "✅")
        assert spec.availability == ("K" if column == "league_size" else "S")
    with pytest.raises(ValueError, match="Unregistered"):
        validate_feature_columns(features.assign(promoted_position_mean=0.75), "season_context")


def test_context_artifact_round_trip_hash_metadata_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tables = context_tables(9)
    result = compute_season_context(tables, rules())
    provenance = FormProvenance("revision", "a" * 64, {"context.py": "b" * 64})
    target, sidecar = save_season_context(result, provenance)
    assert_frame_equal(pd.read_parquet(target), result.features)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["artifact_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert metadata["training_end"] is None and metadata["version"] == "1"
    assert metadata["audit"][0]["imputed_ppm"] is None
    assert "no neutral replacement" in metadata["season_8_exception"]
    assert metadata["input_sha256"] == "a" * 64
    assert metadata["source_sha256"] == dict(provenance.source_sha256)
    before = target.read_bytes(), sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        save_season_context(result, provenance)
    assert (target.read_bytes(), sidecar.read_bytes()) == before
    with pytest.raises(ValueError, match="inside data/derived"):
        save_season_context(result, provenance, directory=Path("data/raw"))


def test_context_hash_covers_values_results_rules_and_excludes_official_data():
    tables = context_tables(9)
    baseline = season_context_input_fingerprint(tables, rules())
    assert baseline == season_context_input_fingerprint(
        replace(tables, team_seasons=tables.team_seasons.assign(deduction=-10)), rules()
    )
    values = tables.team_seasons.copy()
    values.loc[(8, 1), "total_value_eur"] += 1
    assert baseline != season_context_input_fingerprint(
        replace(tables, team_seasons=values), rules()
    )
    assert baseline != season_context_input_fingerprint(change_score(tables, 802, 10), rules())
    assert baseline != season_context_input_fingerprint(
        tables, {**rules(), 8: replace(rules()[8], points_halved=True)}
    )


def test_position_review_compares_every_season_and_rejects_held_out_officials():
    tables = context_tables(9)
    finals = completed_sporting_positions(tables, rules()).reset_index()
    finals.loc[0, "position"] += 1
    result = season_position_comparison(tables, rules(), finals).set_index("season_id")
    assert result.teams.tolist() == [4, 4]
    assert result.different_positions.tolist() == [1, 0]
    with pytest.raises(ValueError, match="8–25"):
        season_position_comparison(tables, rules(), finals.assign(season_id=26))
