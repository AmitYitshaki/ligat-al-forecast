"""Hand examples and frozen EXP-004 regression for the generic review rules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligat.evaluate.selection import paired_statistics
from ligat.evaluate.selection_rules import (
    LeakageSuspectedError,
    close_configurations,
    leakage_check,
    metric_disagreement_entry,
    metric_disagreement_selection,
    recent_leader_flag,
)


def folds(values, *, seasons=None, games=None, shallow=()):
    seasons = list(range(21, 21 + len(values))) if seasons is None else list(seasons)
    return pd.DataFrame(
        {
            "fold": [f"season-{s}" for s in seasons],
            "season_id": seasons,
            "games": [1] * len(values) if games is None else games,
            "deep_fold": [s not in shallow for s in seasons],
            "loss": values,
        }
    )


def example():
    return {
        "winner": folds([2.0, 2.0, 2.0]),
        "simple": folds([1.0, 3.0, 3.0]),
        "far": folds([3.0, 3.0, 3.0]),
    }


def test_rule4_proposes_simplest_close_configuration_and_keeps_penalty_sign():
    frames = example()
    result = close_configurations(
        frames, score="loss", simplicity_order=["far", "simple", "winner"]
    )
    assert result.ranking.winners == ("winner",)
    assert result.ranking.order == ("winner", "simple", "far")
    assert result.close == ("winner", "simple")
    assert result.proposal == "simple"
    evidence = result.comparisons.set_index("config_id")
    assert evidence.loc["simple", "mean_difference"] == pytest.approx(1 / 3)
    assert evidence.loc["simple", "se"] == pytest.approx(2 / 3)
    assert evidence.loc["winner", ["mean_difference", "se"]].tolist() == [0, 0]
    assert evidence.loc["far", "mean_difference"] == 1


def test_rule4_includes_exact_one_se_boundary():
    frames = {"winner": folds([2, 2, 2]), "boundary": folds([2, 2, 5])}
    result = close_configurations(frames, score="loss", simplicity_order=["boundary", "winner"])
    row = result.comparisons.set_index("config_id").loc["boundary"]
    assert row.mean_difference == row.se == 1
    assert result.close == ("winner", "boundary")
    assert result.proposal == "boundary"


def test_rule4_uses_game_weights_ignores_shallow_and_preserves_inputs():
    a = folds([100, 1, 4], seasons=[11, 21, 22], games=[1000, 10, 1], shallow=[11])
    b = folds([0, 2, 2], seasons=[11, 21, 22], games=[1000, 10, 1], shallow=[11])
    original = a.copy(deep=True)
    result = close_configurations(
        {"a": a.sample(frac=1, random_state=4), "b": b}, score="loss", simplicity_order=["a", "b"]
    )
    assert result.ranking.winners == ("a",)
    assert result.ranking.scores["a"] == pytest.approx(14 / 11)
    pd.testing.assert_frame_equal(a, original)


def test_rule4_keeps_every_exact_tied_winner_as_a_reference():
    frames = {"a": folds([1, 3, 2]), "b": folds([3, 1, 2]), "c": folds([2, 2, 3])}
    result = close_configurations(frames, score="loss", simplicity_order=["c", "b", "a"])
    assert result.ranking.winners == ("a", "b")
    assert result.close == ("a", "b", "c")
    assert result.proposal == "c"
    assert len(result.comparisons) == 6
    assert result.comparisons.loc[result.comparisons.config_id.isin(["a", "b"]), "close"].all()


def test_rule4_requires_closeness_to_every_tied_leader():
    frames = {"a": folds([0, 2, 1]), "b": folds([2, 0, 1]), "c": folds([0.2, 2.2, 1.2])}
    result = close_configurations(frames, score="loss", simplicity_order=["c", "b", "a"])
    assert result.ranking.winners == ("a", "b")
    assert result.close == ("a", "b")
    evidence = result.comparisons.loc[result.comparisons.config_id.eq("c")].set_index("winner")
    assert not evidence.loc["a", "close"]
    assert evidence.loc["b", "close"]
    assert result.proposal == "b"


@pytest.mark.parametrize("order", [["winner"], ["winner", "simple", "far", "far"], "winner"])
def test_rule4_requires_complete_unique_explicit_simplicity_order(order):
    with pytest.raises(ValueError, match="simplicity_order"):
        close_configurations(example(), score="loss", simplicity_order=order)
    with pytest.raises(TypeError):
        close_configurations(example(), score="loss")


@pytest.mark.parametrize("different", [False, True])
def test_rule6_compares_recent_with_deep_and_reports_all_separately(different):
    seasons = list(range(11, 26))
    a = folds([1000] + [1] * 14, seasons=seasons, shallow=[11])
    b = folds([0] + [2] * 9 + ([0] * 5 if different else [2] * 5), seasons=seasons, shallow=[11])
    result = recent_leader_flag({"a": a, "b": b}, score="loss")
    assert result.flag is different
    assert result.deep_ranking.winners == ("a",)
    assert result.last5_ranking.winners == (("b",) if different else ("a",))
    assert result.all_ranking.winners == ("b",)
    a.loc[a.season_id.eq(11), "loss"] = -1000
    changed = recent_leader_flag({"a": a, "b": b}, score="loss")
    assert changed.flag is result.flag
    assert changed.deep_ranking.scores == result.deep_ranking.scores
    assert changed.last5_ranking.scores == result.last5_ranking.scores
    assert changed.all_ranking.winners == ("a",)


def test_rule6_uses_weighted_recent_scores_and_compares_all_tied_leaders():
    a = folds([0, 2, 2, 2, 2], games=[10, 1, 1, 1, 1])
    b = folds([1, 1, 1, 1, 1], games=[10, 1, 1, 1, 1])
    result = recent_leader_flag({"a": a, "b": b}, score="loss")
    assert result.last5_ranking.winners == ("a",)
    assert result.last5_ranking.scores["a"] == pytest.approx(8 / 14)
    tied = folds([1] * 14, seasons=range(12, 26))
    changed = tied.copy()
    changed.loc[changed.season_id.le(20), "loss"] = 2
    result = recent_leader_flag({"a": tied, "b": changed}, score="loss")
    assert result.deep_ranking.winners == ("a",)
    assert result.last5_ranking.winners == ("a", "b")
    assert result.flag


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([-1, 0, 1.03], [1, 0, -1.03], False),
        ([0.2] * 3, [1, 0, -1.03], True),
        ([-1, 0, 1.03], [-0.2] * 3, True),
        ([0, 0, 3], [0, 0, -3], False),
        ([0] * 3, [-0.2] * 3, False),
        ([0.2] * 3, [0.3] * 3, False),
    ],
)
def test_rule10_entry_requires_opposite_signs_and_strict_mean_over_se(a, b, expected):
    reference = folds([0] * 3)
    stats = [paired_statistics(folds(-np.array(d)), reference, score="loss") for d in (a, b)]
    assert metric_disagreement_entry(*stats) is expected
    assert metric_disagreement_entry(*reversed(stats)) is expected


@pytest.mark.parametrize("far_metric", [None, "primary", "secondary"])
def test_rule10_selection_checks_closeness_in_both_directions(far_metric):
    primary = {"a": folds([2, 2, 2]), "b": folds([1, 3, 3])}
    secondary = {"a": folds([1, 3, 3]), "b": folds([2, 2, 2])}
    if far_metric == "primary":
        primary["b"]["loss"] = 3
    if far_metric == "secondary":
        secondary["a"]["loss"] = 3
    ranks = [
        close_configurations(f, score="loss", simplicity_order=["a", "b"])
        for f in (primary, secondary)
    ]
    assert ranks[0].ranking.winners == ("a",)
    assert ranks[1].ranking.winners == ("b",)
    assert metric_disagreement_selection(*ranks) is (far_metric is not None)
    assert metric_disagreement_selection(*reversed(ranks)) is (far_metric is not None)


def test_rule10_selection_checks_every_tied_winner():
    primary = {"a": folds([1, 1, 1]), "b": folds([2, 2, 2])}
    secondary = {"a": folds([1, 1, 1]), "b": folds([1, 1, 1])}
    ranks = [
        close_configurations(f, score="loss", simplicity_order=["a", "b"])
        for f in (primary, secondary)
    ]
    assert metric_disagreement_selection(*ranks)


def test_leakage_configuration_mean_stops_without_exposing_scores(capsys):
    with pytest.raises(LeakageSuspectedError, match="stop for review") as exc:
        leakage_check(folds([0.979] * 3), score="loss")
    assert "0.979" not in str(exc.value)
    assert capsys.readouterr() == ("", "")


def test_leakage_flags_single_low_fold_without_returning_any_scores(capsys):
    result = leakage_check(folds([0.978, 1.002]), score="loss")
    assert result.flagged_folds.fold.tolist() == ["season-21"]
    assert result.flagged_folds.season_id.tolist() == [21]
    assert "loss" not in result.flagged_folds
    assert not hasattr(result, "mean")
    assert capsys.readouterr() == ("", "")


def test_leakage_exact_threshold_is_allowed_and_custom_threshold_is_respected():
    assert leakage_check(folds([0.98] * 3), score="loss").flagged_folds.empty
    assert leakage_check(folds([0.5] * 3), score="loss", threshold=0.5).flagged_folds.empty
    with pytest.raises(LeakageSuspectedError):
        leakage_check(folds([0.5] * 3), score="loss", threshold=0.51)


def test_leakage_weights_every_fold_including_shallow():
    safe = folds([0.96, 0.99], seasons=[11, 21], games=[1, 10], shallow=[11])
    result = leakage_check(safe, score="loss")
    assert result.flagged_folds.season_id.tolist() == [11]
    unsafe = safe.copy()
    unsafe["games"] = [10, 1]
    with pytest.raises(LeakageSuspectedError):
        leakage_check(unsafe, score="loss")


@pytest.mark.parametrize("threshold", [np.nan, np.inf, True, "0.98"])
def test_leakage_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="threshold"):
        leakage_check(folds([1] * 3), score="loss", threshold=threshold)


@pytest.mark.parametrize("column", ["fold", "season_id", "games", "deep_fold"])
def test_review_rules_reject_mismatched_fold_populations(column):
    frames = example()
    replacement = {"fold": "other", "season_id": 20, "games": 2, "deep_fold": False}
    frames["far"].loc[0, column] = replacement[column]
    with pytest.raises(ValueError, match="identical"):
        close_configurations(frames, score="loss", simplicity_order=list(frames))
    with pytest.raises(ValueError, match="identical"):
        recent_leader_flag(frames, score="loss")


def test_review_rules_reject_empty_or_insufficient_evidence():
    with pytest.raises(ValueError, match="nonempty"):
        close_configurations({}, score="loss", simplicity_order=[])
    with pytest.raises(ValueError, match="two deep"):
        close_configurations({"a": folds([1])}, score="loss", simplicity_order=["a"])
    with pytest.raises(ValueError, match="five deep recent"):
        recent_leader_flag({"a": folds([1] * 3)}, score="loss")
    stats = paired_statistics(folds([1]), folds([2]), score="loss")
    with pytest.raises(ValueError, match="Insufficient evidence"):
        metric_disagreement_entry(stats, stats)


def test_rule10_rejects_different_populations_and_configuration_sets():
    a, b = folds([1] * 3), folds([2] * 3)
    stats = paired_statistics(a, b, score="loss")
    changed = paired_statistics(a.assign(games=2), b.assign(games=2), score="loss")
    with pytest.raises(ValueError, match="population"):
        metric_disagreement_entry(stats, changed)
    ranks = [
        close_configurations(f, score="loss", simplicity_order=list(f))
        for f in ({"a": a}, {"b": b}, {"a": a.assign(games=2)})
    ]
    for other in ranks[1:]:
        with pytest.raises(ValueError, match="same configurations and deep folds"):
            metric_disagreement_selection(ranks[0], other)


@pytest.mark.real_data
def test_review_rules_reproduce_exp004_flags_and_every_ranking(real_tables):
    root = Path("reports/experiments/EXP-004/full")
    if not (root / "fold_scores.parquet").exists():
        pytest.skip("Local EXP-004 folds unavailable")
    data = pd.read_parquet(
        root / "fold_scores.parquet", filters=[("season_id", "in", list(range(8, 26)))]
    )
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    configs = json.loads((root / "configurations.json").read_text(encoding="utf-8"))
    counts = real_tables.match_table.groupby("season_id").size()
    frames = {}
    for config in configs:
        cid = config["config_id"]
        frame = data.loc[data.config_id.eq(cid)].copy()
        assert set(frame.season_id) == set(range(11, 26))
        assert frame.games.tolist() == frame.season_id.map(counts).tolist()
        frames[cid] = frame.assign(deep_fold=frame.season_id.ge(12))
    rankings = {
        metric: close_configurations(frames, score=metric, simplicity_order=list(frames))
        for metric in ("log_loss", "rps")
    }
    report = {}
    for metric, result in rankings.items():
        expected = review["rankings"][metric]
        assert list(result.ranking.winners) == expected["winners"]
        assert list(result.close) == expected["close"]
        assert result.ranking.order == tuple(sorted(expected["scores"], key=expected["scores"].get))
        assert result.ranking.scores == pytest.approx(expected["scores"], abs=1e-12, rel=0)
        report[metric] = {
            "winners": result.ranking.winners,
            "close": result.close,
            "order": result.ranking.order,
            "max_score_error": max(
                abs(result.ranking.scores[c] - value) for c, value in expected["scores"].items()
            ),
        }
    recent = recent_leader_flag(frames, score="log_loss")
    expected = review["last5_ranking"]
    assert recent.flag is review["rule6"] is True
    assert recent.deep_ranking.scores == rankings["log_loss"].ranking.scores
    assert list(recent.last5_ranking.winners) == expected["winners"]
    assert recent.last5_ranking.scores == pytest.approx(expected["scores"], abs=1e-12, rel=0)
    assert recent.last5_ranking.order == tuple(
        sorted(expected["scores"], key=expected["scores"].get)
    )
    assert expected["close"] == []  # No one-SE close set on five seasons, per rule 4.
    entry = {}
    for config in configs:
        cid = config["config_id"]
        if cid == "C0":
            continue
        ref = "C0-hl4" if config["half_life"] and config["name"] != "C0" else "C0"
        stats = [paired_statistics(frames[cid], frames[ref], score=m) for m in rankings]
        entry[cid] = metric_disagreement_entry(*stats)
    assert entry == review["rule10_entry"]
    assert len(entry) == 19 and not any(entry.values())
    selection = metric_disagreement_selection(*rankings.values())
    assert selection is review["rule10_selection"] is False
    report["recent"] = {
        "winners": recent.last5_ranking.winners,
        "order": recent.last5_ranking.order,
        "max_score_error": max(
            abs(recent.last5_ranking.scores[c] - value) for c, value in expected["scores"].items()
        ),
    }
    report.update(rule6=recent.flag, rule10_entry=entry, rule10_selection=selection)
    print(json.dumps(report))
