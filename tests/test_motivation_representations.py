"""Exact EXP-004 boundaries, dead-row overrides and fixture-side orientation."""

import numpy as np
import pandas as pd
import pytest

from ligat.motivation.representations import (
    components,
    game_contrast,
    level_l2,
    level_l3,
    level_l5,
    level_lc,
)


@pytest.mark.parametrize(
    "score,l3,l5",
    [
        (0.0, 0.5, 0.25),
        (0.01, 0.5, 0.25),
        (np.nextafter(0.01, 1), 1.0, 0.5),
        (0.05, 1.0, 0.5),
        (np.nextafter(0.05, 1), 1.0, 0.75),
        (0.2, 1.0, 0.75),
        (np.nextafter(0.2, 1), 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_exact_boundaries_and_dead_override(score, l3, l5):
    data = pd.DataFrame(dict(leverage_max=[score, 0.9], nothing_to_play=[False, True]))
    assert level_l3(data).tolist() == [l3, 0.0]
    assert level_l5(data).tolist() == [l5, 0.0]
    assert level_l2(data).tolist() == [1.0, 0.0]
    assert level_lc(data).tolist() == [score, 0.0]


def paired():
    data = pd.DataFrame(
        dict(
            game_id=[1, 1, 2, 2],
            team_id=[10, 20, 20, 30],
            opponent_id=[20, 10, 30, 20],
            is_home=[True, False, True, False],
            leverage_max=[0.005, 0.3, 0.1, 0.8],
            nothing_to_play=[False, False, True, False],
            title_lev=[0.005, 0.3, 0.1, 0.8],
            cut_lev=[np.nan] * 4,
            survival_lev=[0.0] * 4,
            survival_direct_lev=[1.0] * 4,
        )
    )
    return data.set_index(["game_id", "team_id"])


@pytest.mark.parametrize("level", ["L2", "L3", "L5", "Lc", "Comp"])
def test_diff_is_antisymmetric_under_swapping_venue_sides(level):
    data = paired()
    swapped = data.copy()
    swapped["is_home"] = ~swapped.is_home
    first = game_contrast(data, level)
    reverse = game_contrast(swapped, level)
    np.testing.assert_array_equal(first.to_numpy(), -reverse.to_numpy())


@pytest.mark.parametrize("level", ["L2", "L3", "L5", "Lc"])
def test_sides_are_explicit_home_then_away(level):
    data = paired()
    fn = {"L2": level_l2, "L3": level_l3, "L5": level_l5, "Lc": level_lc}[level]
    result = game_contrast(data, level, "sides")
    np.testing.assert_array_equal(result.m_home, fn(data.loc[data.is_home]))
    np.testing.assert_array_equal(result.m_away, fn(data.loc[~data.is_home]))


def test_components_use_only_three_targets_with_zero_for_structural_cut_absence():
    result = components(paired())
    assert list(result) == ["title", "cut", "survival"]
    assert result.cut.eq(0).all()
    assert result.iloc[2].eq(0).all()
    with pytest.raises(ValueError, match="Comp"):
        game_contrast(paired(), "Comp", "sides")


def test_wrong_opponent_fails_instead_of_mispairing_rows():
    data = paired()
    data.loc[(1, 10), "opponent_id"] = 30
    with pytest.raises(ValueError, match="opponent"):
        game_contrast(data, "L2")
