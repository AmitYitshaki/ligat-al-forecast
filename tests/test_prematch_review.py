"""Notebook diagnostics keep paired-game denominators and development boundaries."""

import pandas as pd
import pytest

from ligat.analysis.prematch_review import (
    dead_game_summary,
    place_line_contradictions,
    stage_end_distribution,
)
from ligat.analysis.prepare import AnalysisTables
from ligat.features.table_state import TableStateBlock
from ligat.simulate.rules import load_season_rules


def test_dead_game_denominator_is_matches_not_team_rows() -> None:
    frame = pd.DataFrame(
        dict(
            game_id=[1, 1, 2, 2, 3, 3],
            team_id=[1, 2] * 3,
            season_id=14,
            stage="lower",
            nothing_to_play=[False, False, True, False, True, True],
            rounds_to_stage_end=[2, 2, 1, 1, 0, 0],
        )
    ).set_index(["game_id", "team_id"])
    summary, matches = dead_game_summary(frame)
    row = summary.iloc[0]
    assert row.games == 3 and row.neither == row.one == row.both == 1
    assert row.at_least_one_share == pytest.approx(2 / 3)
    assert row.both_share == pytest.approx(1 / 3)
    assert len(matches) == 3
    assert stage_end_distribution(frame).team_games.sum() == 6
    with pytest.raises(ValueError, match="both team rows"):
        dead_game_summary(frame.iloc[:-1])
    with pytest.raises(ValueError, match="development"):
        dead_game_summary(frame.assign(season_id=26))


@pytest.mark.slow
def test_real_place_decisions_never_cross_final_sporting_lines(real_tables: AnalysisTables) -> None:
    rules = load_season_rules()
    features = TableStateBlock(rules).compute(real_tables)
    assert features.places_decided.any()
    errors = place_line_contradictions(real_tables, features, rules)
    assert errors.empty, errors.to_string(index=False)
    # A false opening-day claim on eventual champion Beitar must be detected.
    games = real_tables.team_matches.reset_index()
    champion = games.loc[games.season_id.eq(8) & games.team_id.eq(3595)].sort_values("date").iloc[0]
    key = (champion.game_id, champion.team_id)
    assert features.loc[key, "group_position"] > 4
    assert not features.loc[key, "places_decided"]
    altered = features.copy()
    altered.loc[key, "places_decided"] = True
    assert not place_line_contradictions(real_tables, altered, rules).empty
    with pytest.raises(ValueError, match="development"):
        place_line_contradictions(real_tables, features.assign(season_id=26), rules)
