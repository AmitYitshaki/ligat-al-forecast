"""Metric-independent review rules; proposals never choose a model for Amit."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.selection import (
    _FOLD_COLUMNS,
    PairedStatistics,
    _numeric,
    _prepare_folds,
    paired_statistics,
)


@dataclass(frozen=True)
class ConfigurationRanking:
    """Game-weighted scores, ascending order and every exactly tied leader."""

    score: str
    scores: dict[str, float]
    order: tuple[str, ...]
    winners: tuple[str, ...]
    population: pd.DataFrame


@dataclass(frozen=True)
class CloseConfigurations:
    """Rule 4 evidence against every tied leader and a simplicity proposal only."""

    ranking: ConfigurationRanking
    close: tuple[str, ...]
    comparisons: pd.DataFrame
    proposal: str


@dataclass(frozen=True)
class RecentLeaderFlag:
    """Rule 6 compares recent versus deep leaders; all folds are report-only."""

    flag: bool
    deep_ranking: ConfigurationRanking
    last5_ranking: ConfigurationRanking
    all_ranking: ConfigurationRanking


@dataclass(frozen=True)
class LeakageCheck:
    """Safe configuration result: flagged fold metadata, without score values."""

    score: str
    threshold: float
    flagged_folds: pd.DataFrame


class LeakageSuspectedError(RuntimeError):
    """Hard stop requiring leakage review; the exception never includes scores."""


def _same_population(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in _FOLD_COLUMNS
    )


def _configurations(
    folds_by_config: Mapping[str, pd.DataFrame], score: str
) -> dict[str, pd.DataFrame]:
    if not folds_by_config or any(not isinstance(c, str) or not c for c in folds_by_config):
        raise ValueError("Review requires named, nonempty configurations")
    frames = {name: _prepare_folds(frame, score) for name, frame in folds_by_config.items()}
    first = next(iter(frames.values()))
    if any(not _same_population(first, frame) for frame in frames.values()):
        raise ValueError("Review requires identical folds, seasons, games and depth")
    return frames


def _ranking(frames: Mapping[str, pd.DataFrame], score: str) -> ConfigurationRanking:
    population = next(iter(frames.values())).loc[:, list(_FOLD_COLUMNS)].reset_index(drop=True)
    if population.empty:
        raise ValueError("Insufficient evidence: ranking has no folds")
    scores = {name: float(np.average(f[score], weights=f.games)) for name, f in frames.items()}
    order = tuple(sorted(scores, key=lambda name: scores[name]))
    winners = tuple(name for name in scores if scores[name] == scores[order[0]])
    return ConfigurationRanking(score, scores, order, winners, population)


def close_configurations(
    folds_by_config: Mapping[str, pd.DataFrame], *, score: str, simplicity_order: Sequence[str]
) -> CloseConfigurations:
    """Propose the simplest configuration within one paired SE of the deep winner.

    Rule 4 uses loss(configuration) minus loss(winner), the reverse of entry
    improvement. All exact tied winners remain references, as in EXP-004;
    other configurations must be close to every winner. Winners are always
    close. The caller supplies a complete, unique simplicity order in advance.
    Fewer than two deep folds cannot support an SE and raises ValueError.
    """
    frames = _configurations(folds_by_config, score)
    if (
        isinstance(simplicity_order, str)
        or len(set(simplicity_order)) != len(simplicity_order)
        or set(simplicity_order) != set(frames)
    ):
        raise ValueError("simplicity_order must list each configuration exactly once")
    deep = {name: f.loc[f.deep_fold] for name, f in frames.items()}
    if len(next(iter(deep.values()))) < 2:
        raise ValueError("Insufficient evidence: rule 4 requires at least two deep folds")
    ranking = _ranking(deep, score)
    records = []
    near = []
    for name, frame in frames.items():
        conditions = []
        for winner in ranking.winners:
            stats = paired_statistics(frames[winner], frame, score=score)
            assert stats.mean_improvement is not None and stats.se is not None
            close = name in ranking.winners or stats.mean_improvement <= stats.se
            conditions.append(close)
            records.append(
                {
                    "config_id": name,
                    "winner": winner,
                    "mean_difference": stats.mean_improvement,
                    "se": stats.se,
                    "close": close,
                }
            )
        if all(conditions):
            near.append(name)
    proposal = next(name for name in simplicity_order if name in near)
    return CloseConfigurations(ranking, tuple(near), pd.DataFrame(records), proposal)


def recent_leader_flag(
    folds_by_config: Mapping[str, pd.DataFrame], *, score: str
) -> RecentLeaderFlag:
    """Compare leaders on deep folds and deep seasons 21-25, never all vs recent.

    Rankings use game weights. Recent ranking has no one-SE close set (rule 4).
    Missing recent seasons raise ValueError rather than inventing a review flag.
    """
    frames = _configurations(folds_by_config, score)
    deep = {name: f.loc[f.deep_fold] for name, f in frames.items()}
    recent = {name: f.loc[f.deep_fold & f.season_id.between(21, 25)] for name, f in frames.items()}
    if len(next(iter(recent.values()))) != 5:
        raise ValueError("Insufficient evidence: rule 6 requires all five deep recent seasons")
    deep_rank, last_rank, all_rank = (_ranking(group, score) for group in (deep, recent, frames))
    return RecentLeaderFlag(
        set(deep_rank.winners) != set(last_rank.winners), deep_rank, last_rank, all_rank
    )


def metric_disagreement_entry(stats_a: PairedStatistics, stats_b: PairedStatistics) -> bool:
    """Rule 10: opposite mean signs and absolute mean > SE in at least one metric."""
    if not _same_population(stats_a.differences, stats_b.differences):
        raise ValueError("Metric entry evidence requires the same deep fold population")
    if (
        stats_a.n < 2
        or stats_b.n < 2
        or stats_a.mean_improvement is None
        or stats_b.mean_improvement is None
        or stats_a.se is None
        or stats_b.se is None
    ):
        raise ValueError("Insufficient evidence: metric disagreement requires paired SEs")
    a, b = stats_a.mean_improvement, stats_b.mean_improvement
    opposite = (a < 0 < b) or (b < 0 < a)
    return opposite and (abs(a) > stats_a.se or abs(b) > stats_b.se)


def metric_disagreement_selection(
    ranking_a: CloseConfigurations, ranking_b: CloseConfigurations
) -> bool:
    """Rule 10: any winner outside the other metric's close set raises a flag."""
    if set(ranking_a.ranking.scores) != set(ranking_b.ranking.scores) or not _same_population(
        ranking_a.ranking.population, ranking_b.ranking.population
    ):
        raise ValueError("Metric selection requires the same configurations and deep folds")
    return not set(ranking_a.ranking.winners) <= set(ranking_b.close) or not set(
        ranking_b.ranking.winners
    ) <= set(ranking_a.close)


def leakage_check(folds: pd.DataFrame, *, score: str, threshold: float = 0.98) -> LeakageCheck:
    """Stop below the game-weighted all-fold mean threshold; flag individual folds.

    Callers select the applicable score (H/D/A log loss for the 0.98 policy).
    Depth does not exclude folds from this check. Neither the result nor the
    exception reports the configuration mean or any fold score.
    """
    _numeric(pd.Series([threshold]), "threshold")
    frame = _prepare_folds(folds, score)
    if float(np.average(frame[score], weights=frame.games)) < threshold:
        raise LeakageSuspectedError("Configuration mean below leakage threshold; stop for review")
    flagged = frame.loc[frame[score].lt(threshold), list(_FOLD_COLUMNS)].reset_index(drop=True)
    return LeakageCheck(score, threshold, flagged)
