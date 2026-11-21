"""Small synthetic checks for the fifteen pure prematch v2 additions in T-014 A1."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ligat.features.prematch_v2 import build_prematch_v2_additions

COPIES = (
    "gf_all_w5",
    "gf_venue_w5",
    "gf_all_std_m6",
    "gf_all_prev",
    "gf_all_hist",
)
FORM_MIRRORS = (
    ("opp_ga_all_w5", "ga_all_w5"),
    ("opp_ga_venue_w5", "ga_venue_w5"),
    ("opp_ga_all_std_m6", "ga_all_std_m6"),
    ("opp_ga_all_prev", "ga_all_prev"),
    ("opp_ga_all_hist", "ga_all_hist"),
    ("opp_pts_venue_ewma5", "pts_venue_ewma5"),
)
TABLE_MIRRORS = (
    ("opp_points_per_game", "points_per_game"),
    ("opp_position_pct", "position_pct"),
    ("opp_promoted_in", "promoted_in"),
    ("opp_nothing_to_play", "nothing_to_play"),
)


@pytest.fixture
def sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two fixtures reverse the home side, exposing joins that ignore game or venue."""
    index = pd.MultiIndex.from_tuples(
        [(101, 1), (101, 2), (202, 1), (202, 2)], names=["game_id", "team_id"]
    )
    dates = pd.to_datetime(["2020-08-01"] * 2 + ["2020-08-08"] * 2).tz_localize("Asia/Jerusalem")
    prematch = pd.DataFrame(
        {
            "season_id": pd.array([22] * 4, dtype="Int64"),
            "date": dates,
            "as_of": dates.normalize(),
            "is_home": pd.array([True, False, False, True], dtype="boolean"),
            "opponent_id": pd.array([2, 1, 2, 1], dtype="Int64"),
            "points_per_game": pd.array([2.5, 0.75, 0.0, 1.0], dtype="Float64"),
            "position_pct": [0.1, 0.9, 0.7, 0.2],
            "promoted_in": pd.array([False, True, False, True], dtype="boolean"),
            "nothing_to_play": pd.array([False, True, True, False], dtype="boolean"),
        },
        index=index,
    )
    columns = (*COPIES, *(source for _, source in FORM_MIRRORS))
    form = pd.DataFrame(
        {
            column: pd.array(
                [i + 0.1, i + 0.2, i + 0.3, i + 0.4],
                dtype="Float64" if i % 2 else "float32",
            )
            for i, column in enumerate(columns)
        },
        index=index,
    )
    form["ga_venue_w5"] = [0.7, 1.6, 1.1, 0.9]
    return prematch, form


@pytest.mark.parametrize("column", COPIES)
def test_form_copy_is_exact(sources, column):
    """Every own-team copy preserves source values, dtype and index exactly."""
    prematch, form = sources
    result = build_prematch_v2_additions(prematch, form)
    assert_series_equal(result[column], form[column], check_exact=True)


@pytest.mark.parametrize("target,source", (*FORM_MIRRORS, *TABLE_MIRRORS))
def test_opponent_mirrors_in_both_directions(sources, target, source):
    """Each mirror uses the other row of that fixture, with its original dtype."""
    prematch, form = sources
    result = build_prematch_v2_additions(prematch, form)
    original = form if (target, source) in FORM_MIRRORS else prematch
    expected = original.loc[[(101, 2), (101, 1), (202, 2), (202, 1)], source].copy()
    expected.index = prematch.index
    assert_series_equal(result[target], expected.rename(target), check_exact=True)
    # Haifa at home sees Sakhnin's away defense; the reverse sees Haifa's home defense.
    if target == "opp_ga_venue_w5":
        assert result.loc[(101, 1), target] == 1.6
        assert result.loc[(101, 2), target] == 0.7


def test_output_is_only_the_fifteen_additions(sources):
    """A1 exposes no partial prematch artifact, ratio, target or unrequested column."""
    result = build_prematch_v2_additions(*sources)
    assert tuple(result.columns) == (*COPIES, *(c for c, _ in (*FORM_MIRRORS, *TABLE_MIRRORS)))
    assert result.shape == (4, 15)


@pytest.mark.parametrize(
    "shuffle_prematch,shuffle_form", [(True, False), (False, True), (True, True)]
)
def test_input_row_order_does_not_change_output(sources, shuffle_prematch, shuffle_form):
    """Independent input permutations produce the same canonical keyed result."""
    prematch, form = sources
    expected = build_prematch_v2_additions(prematch, form)
    actual = build_prematch_v2_additions(
        prematch.sample(frac=1, random_state=3) if shuffle_prematch else prematch,
        form.sample(frac=1, random_state=7) if shuffle_form else form,
    )
    assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("which", ["prematch", "form"])
@pytest.mark.parametrize("bad", ["duplicate", "missing_key", "wrong_index"])
def test_invalid_keys_fail_clearly(sources, which, bad):
    """Malformed identities cannot turn into silent row loss or a many-to-many join."""
    prematch, form = sources
    frame = prematch if which == "prematch" else form
    if bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
        message = "duplicate.*keys"
    elif bad == "missing_key":
        index = frame.index.to_frame(index=False)
        index.loc[0, "team_id"] = np.nan
        frame.index = pd.MultiIndex.from_frame(index)
        message = "keys.*missing"
    else:
        frame = frame.reset_index()
        message = "MultiIndex"
    with pytest.raises(ValueError, match=message):
        build_prematch_v2_additions(
            frame if which == "prematch" else prematch, frame if which == "form" else form
        )


def test_missing_form_row_fails_instead_of_creating_nulls(sources):
    """A missing source key must fail before reindexing can manufacture missing features."""
    prematch, form = sources
    with pytest.raises(ValueError, match="keys.*match"):
        build_prematch_v2_additions(prematch, form.iloc[1:])


def test_one_row_fixture_is_rejected(sources):
    """Matching but incomplete inputs still cannot represent half of a fixture."""
    prematch, form = sources
    with pytest.raises(ValueError, match="two team rows"):
        build_prematch_v2_additions(prematch.iloc[1:], form.iloc[1:])


@pytest.mark.parametrize("opponent,message", [(1, "own ID"), (99, "other row"), (pd.NA, "missing")])
def test_invalid_opponent_is_rejected(sources, opponent, message):
    """Self-opponents, missing opponents and nonexistent paired keys fail explicitly."""
    prematch, form = sources
    prematch.loc[(101, 1), "opponent_id"] = opponent
    with pytest.raises(ValueError, match=message):
        build_prematch_v2_additions(prematch, form)


@pytest.mark.parametrize("bad", ["venue", "season", "date"])
def test_paired_identity_mismatch_uses_v1_validation(sources, bad):
    """Reuse v1's home/away and same-fixture checks rather than weaken pairing."""
    prematch, form = sources
    if bad == "venue":
        prematch.loc[(101, 2), "is_home"] = True
    elif bad == "season":
        prematch.loc[(101, 2), "season_id"] = 21
    else:
        prematch.loc[(101, 2), "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="one home|paired rows"):
        build_prematch_v2_additions(prematch, form)


@pytest.mark.parametrize("which,column", [("form", "gf_all_w5"), ("prematch", "promoted_in")])
def test_missing_source_column_is_rejected(sources, which, column):
    """Absent required source columns produce a descriptive validation error."""
    prematch, form = sources
    with pytest.raises(ValueError, match=f"missing source columns.*{column}"):
        build_prematch_v2_additions(
            prematch.drop(columns=column) if which == "prematch" else prematch,
            form.drop(columns=column) if which == "form" else form,
        )


@pytest.mark.parametrize("column", ["gf_all_w5", "ga_venue_w5", "pts_venue_ewma5"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_form_sources_must_remain_complete_and_finite(sources, column, bad):
    """Both copies and mirrors retain v1's prohibition on missing or infinite form values."""
    prematch, form = sources
    form.loc[(101, 1), column] = bad
    with pytest.raises(ValueError, match=f"Source {column}.*nonfinite"):
        build_prematch_v2_additions(prematch, form)


def test_missing_promoted_flag_is_rejected(sources):
    """A season-context flag cannot gain a missing value that v1 forbids."""
    prematch, form = sources
    prematch.loc[(101, 1), "promoted_in"] = pd.NA
    with pytest.raises(ValueError, match="promoted_in.*missing"):
        build_prematch_v2_additions(prematch, form)


@pytest.mark.parametrize("column", ["points_per_game", "position_pct", "nothing_to_play"])
def test_permitted_table_state_nulls_are_mirrored_unchanged(sources, column):
    """Nullable table-state values stay missing instead of being silently filled or cast."""
    prematch, form = sources
    prematch[column] = prematch[column].convert_dtypes()
    prematch.loc[(101, 1), column] = pd.NA
    result = build_prematch_v2_additions(prematch, form)
    assert pd.isna(result.loc[(101, 2), "opp_" + column])
    assert result["opp_" + column].dtype == prematch[column].dtype


def test_assembly_never_recomputes_or_performs_io(sources, monkeypatch):
    """Failing calculators and Parquet APIs demonstrate assembly uses only supplied frames."""

    def forbidden(*args, **kwargs):
        pytest.fail("Pure additions must not recompute features or read/write data")

    for name in (
        "ligat.features.table_state.TableStateBlock.compute",
        "ligat.features.elo.compute_elo",
        "ligat.features.form.compute_form",
        "ligat.features.season_context.compute_season_context",
        "pandas.read_parquet",
        "pandas.DataFrame.to_parquet",
    ):
        monkeypatch.setattr(name, forbidden)
    assert build_prematch_v2_additions(*sources).shape == (4, 15)


def test_inputs_and_output_are_independent(sources):
    """Building and then editing the result leaves both caller-owned inputs unchanged."""
    prematch, form = sources
    before = prematch.copy(deep=True), form.copy(deep=True)
    result = build_prematch_v2_additions(prematch, form)
    result.iloc[0, 0] = 999
    result.loc[(101, 1), "opp_promoted_in"] = True
    assert_frame_equal(prematch, before[0], check_exact=True)
    assert_frame_equal(form, before[1], check_exact=True)
