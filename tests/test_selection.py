"""Hand-computed paired evidence, rule 8 boundaries, and frozen EXP-004 reproduction."""

import json
from dataclasses import replace
from math import fsum, sqrt
from pathlib import Path
from statistics import stdev

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ligat.evaluate.selection import paired_statistics, rule8


def fold_tables(differences=None, *, seasons=tuple(range(11, 26)), games=None):
    """Synthetic fold scores with independently chosen game counts and improvements."""
    n = len(seasons)
    d = np.linspace(-0.003, 0.009, n) if differences is None else np.asarray(differences, float)
    reference = pd.DataFrame(
        {
            "fold": [s - 10 for s in seasons],
            "season_id": seasons,
            "games": np.arange(1, n + 1) if games is None else games,
            "deep_fold": [s != 11 for s in seasons],
            "loss": np.zeros(n),
        }
    )
    candidate = reference.copy(deep=True)
    candidate["loss"] = -d
    return candidate, reference


def test_fourteen_deep_folds_match_hand_mean_se_and_t_interval():
    candidate, reference = fold_tables()
    candidate.loc[0, "loss"] = -999
    stats = paired_statistics(candidate, reference, score="loss")
    d = [-float(v) for v in candidate.loss.iloc[1:]]
    counts = candidate.games.iloc[1:].tolist()
    mean = fsum(w * v for w, v in zip(counts, d, strict=True)) / sum(counts)
    se = stdev(d) / sqrt(14)
    critical = 2.1603686564610127  # Student t(0.975, df=13), independently tabulated.
    assert stats.n == 14
    assert stats.mean_improvement == pytest.approx(mean, abs=1e-12)
    assert stats.se == pytest.approx(se, abs=1e-12)
    assert stats.ci_lower == pytest.approx(mean - critical * se, abs=1e-12)
    assert stats.ci_upper == pytest.approx(mean + critical * se, abs=1e-12)
    recent_d, recent_w = d[-5:], counts[-5:]
    recent_mean = fsum(w * v for w, v in zip(recent_w, recent_d, strict=True)) / sum(recent_w)
    assert stats.last5_improvement == pytest.approx(recent_mean, abs=1e-12)
    assert stats.last5_folds == 5 and stats.last5_folds_better == sum(v > 0 for v in recent_d)
    assert stats.mean_improvement != pytest.approx(fsum(d) / 14, abs=1e-8)
    decision = rule8(stats, version=2)
    assert decision.stats is stats and decision.version == 2
    assert (
        decision.mean_exceeds_se and decision.recent_count_passes and decision.recent_mean_positive
    )


def test_sign_reversal_shallow_fold_and_row_order_do_not_change_the_definition():
    candidate, reference = fold_tables()
    before = [f.copy(deep=True) for f in (candidate, reference)]
    positive = paired_statistics(candidate, reference, score="loss")
    negative = paired_statistics(reference, candidate, score="loss")
    assert positive.mean_improvement > 0 and negative.mean_improvement < 0
    assert positive.mean_improvement == -negative.mean_improvement
    assert positive.se == negative.se
    candidate.loc[0, "loss"] = 1e6
    candidate = candidate.sample(frac=1, random_state=14)
    reference = reference.sample(frac=1, random_state=15)
    shuffled = paired_statistics(candidate, reference, score="loss")
    assert positive.mean_improvement == shuffled.mean_improvement
    assert positive.se == shuffled.se and positive.n == shuffled.n
    assert_frame_equal(positive.differences, shuffled.differences)
    # The first calculation did not mutate either source table.
    assert_frame_equal(before[1], reference.sort_values("season_id").reset_index(drop=True))


@pytest.mark.parametrize(
    "recent,expected1,expected2",
    [
        ([1, 1, 1, -0.1, -0.1], False, True),
        ([1, 1, 1, -10, -10], False, False),
        ([1, 1, 1, 1, -0.1], True, True),
        ([1, 1, 1, 1, -10], True, False),
    ],
)
def test_rule_versions_keep_their_distinct_recent_conditions(recent, expected1, expected2):
    candidate, reference = fold_tables(
        [20] * 9 + recent, seasons=tuple(range(12, 26)), games=[1] * 14
    )
    stats = paired_statistics(candidate, reference, score="loss")
    assert stats.mean_improvement > stats.se
    one, two = rule8(stats, version=1), rule8(stats, version=2)
    assert one.passes is expected1 and two.passes is expected2
    assert one.recent_mean_positive is None
    assert two.recent_mean_positive == (sum(recent) > 0)


def test_strict_mean_equals_se_and_zero_fold_difference_boundaries():
    candidate, reference = fold_tables([1] * 13 + [0], seasons=tuple(range(12, 26)), games=[1] * 14)
    stats = paired_statistics(candidate, reference, score="loss")
    assert stats.last5_folds_better == 4
    for version in (1, 2):
        decision = rule8(replace(stats, mean_improvement=stats.se), version=version)
        assert decision.mean_exceeds_se is False and decision.passes is False
    zero = paired_statistics(reference, reference, score="loss")
    assert zero.se == 0 and zero.mean_improvement == 0 and zero.last5_folds_better == 0
    assert rule8(zero, version=2).passes is False


def test_zero_recent_mean_fails_version_two_even_with_three_improved_seasons():
    a, b = fold_tables([20] * 9 + [1, 1, 1, -1, -2], seasons=tuple(range(12, 26)), games=[1] * 14)
    stats = paired_statistics(a, b, score="loss")
    assert stats.last5_folds_better == 3 and stats.last5_improvement == 0
    decision = rule8(stats, version=2)
    assert decision.mean_exceeds_se and decision.recent_count_passes
    assert decision.recent_mean_positive is False and decision.passes is False


def test_recent_mean_uses_game_weights_instead_of_season_vote_counts():
    a, b = fold_tables(
        [20] * 9 + [1, 1, 1, -1, -1], seasons=tuple(range(12, 26)), games=[1] * 12 + [100, 100]
    )
    stats = paired_statistics(a, b, score="loss")
    assert stats.last5_folds_better == 3
    assert stats.last5_improvement == pytest.approx(-197 / 203)
    assert rule8(stats, version=2).recent_mean_positive is False


def test_ten_deep_folds_use_sqrt_ten_and_nine_degrees_of_freedom():
    a, b = fold_tables(seasons=tuple(range(16, 26)))
    stats = paired_statistics(a, b, score="loss")
    expected = stdev((-a.loss).tolist()) / sqrt(10)
    assert stats.n == 10 and stats.se == pytest.approx(expected, abs=1e-12)
    assert stats.ci_upper == pytest.approx(
        stats.mean_improvement + 2.2621571628540993 * expected, abs=1e-12
    )


def test_descriptive_95_percent_interval_is_not_an_extra_entry_condition():
    d = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1] + [0.5] * 5) + 0.3
    candidate, reference = fold_tables(d, seasons=tuple(range(12, 26)), games=[1] * 14)
    stats = paired_statistics(candidate, reference, score="loss")
    assert stats.ci_lower < 0 < stats.ci_upper
    assert rule8(stats, version=2).passes is True


@pytest.mark.parametrize(
    "seasons,deep", [((21,), True), ((12, 13, 14, 23, 24, 25), True), ((11,), False)]
)
def test_insufficient_evidence_never_becomes_an_automatic_failure(seasons, deep):
    a, b = fold_tables(seasons=seasons)
    a["deep_fold"] = b["deep_fold"] = deep
    stats = paired_statistics(a, b, score="loss")
    for version in (1, 2):
        decision = rule8(stats, version=version)
        assert decision.status == "insufficient_evidence" and decision.passes is None
    if stats.n < 2:
        assert stats.se is None and stats.ci_lower is None and stats.ci_upper is None
    else:
        assert stats.se is not None


@pytest.mark.parametrize(
    "damage",
    [
        "fold",
        "season",
        "games",
        "depth",
        "missing",
        "duplicate",
        "null_fold",
        "nan",
        "infinite",
        "zero_games",
        "fractional_games",
        "nonbool_depth",
    ],
)
def test_bad_or_mismatched_fold_tables_are_rejected(damage):
    a, b = fold_tables()
    if damage in ("fold", "season", "games", "depth"):
        column, value = {
            "fold": ("fold", 999),
            "season": ("season_id", 999),
            "games": ("games", 99),
            "depth": ("deep_fold", True),
        }[damage]
        a.loc[0, column] = value
    elif damage == "missing":
        a = a.iloc[1:]
    elif damage == "duplicate":
        a = pd.concat([a, a.iloc[[0]]])
    elif damage == "null_fold":
        a.loc[0, "fold"] = np.nan
    elif damage in ("nan", "infinite"):
        a.loc[0, "loss"] = np.nan if damage == "nan" else np.inf
    elif damage == "zero_games":
        a.loc[0, "games"] = 0
    elif damage == "fractional_games":
        a = a.astype({"games": float})
        a.loc[0, "games"] = 0.5
    else:
        a["deep_fold"] = a.deep_fold.astype(int)
    with pytest.raises(ValueError):
        paired_statistics(a, b, score="loss")


def test_score_and_version_must_be_explicit_and_valid():
    a, b = fold_tables()
    with pytest.raises(TypeError):
        paired_statistics(a, b)
    with pytest.raises(ValueError):
        paired_statistics(a, b, score="missing")
    stats = paired_statistics(a, b, score="loss")
    with pytest.raises(TypeError):
        rule8(stats)
    for invalid in (0, 3, True, 1.0, "2"):
        with pytest.raises(ValueError, match="version 1 or 2"):
            rule8(stats, version=invalid)


@pytest.mark.real_data
def test_rule8_v1_reproduces_all_nineteen_exp004_summary_rows(real_tables):
    """Frozen fold scores reproduce the old summary without refitting or rewriting anything."""
    root = Path("reports/experiments/EXP-004/full")
    if not (root / "fold_scores.parquet").exists():
        pytest.skip("Local EXP-004 folds unavailable")
    folds = pd.read_parquet(
        root / "fold_scores.parquet", filters=[("season_id", "in", list(range(8, 26)))]
    )
    summary = pd.read_csv(root / "summary.csv", float_precision="round_trip").set_index("config_id")
    configs = json.loads((root / "configurations.json").read_text(encoding="utf-8"))
    assert set(folds.season_id) == set(range(11, 26))
    actual_counts = real_tables.match_table.groupby("season_id").size()
    report = []
    for config in configs:
        cid = config["config_id"]
        if cid == "C0":
            continue
        ref = "C0-hl4" if config["half_life"] and config["name"] != "C0" else "C0"
        frames = []
        for name in (cid, ref):
            frame = folds.loc[folds.config_id.eq(name)].copy()
            assert frame.games.to_list() == frame.season_id.map(actual_counts).to_list()
            frame["deep_fold"] = frame.season_id.ge(12)
            frames.append(frame)
        result = rule8(paired_statistics(*frames, score="log_loss"), version=1)
        expected = summary.loc[cid]
        delta_mean = abs(result.stats.mean_improvement - float(expected.log_loss_mean_improvement))
        delta_se = abs(result.stats.se - float(expected.log_loss_se))
        delta_count = abs(
            result.stats.last5_folds_better - int(expected.log_loss_last5_folds_better)
        )
        equal = result.passes == bool(expected.log_loss_passes_rule8)
        assert max(delta_mean, delta_se, delta_count) <= 1e-12 and equal, cid
        report.append(
            {
                "config_id": cid,
                "reference": ref,
                "mean_error": delta_mean,
                "se_error": delta_se,
                "count_error": delta_count,
                "passes_error": int(not equal),
                "passes_equal": equal,
            }
        )
    assert len(report) == 19
    print(json.dumps(report))
