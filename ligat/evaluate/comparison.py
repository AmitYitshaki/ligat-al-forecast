"""Rank configurations on identical games; flag uncertainty, never select one."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.evaluate.backtest import (
    RETROSPECTIVE_LABEL,
    EvaluationResult,
    summarize,
)
from ligat.evaluate.goal_backtest import GoalEvaluationResult, summarize_goals


@dataclass(frozen=True)
class Comparison:
    """Rankings and review flags; choosing simplicity or a model belongs to B3."""

    rankings: pd.DataFrame
    all_winners: tuple[str, ...]
    last5_winners: tuple[str, ...]
    disagreement: bool
    requires_decision: bool
    label: str


def compare(
    results: Mapping[str, EvaluationResult | GoalEvaluationResult],
    *,
    for_selection: bool = True,
    metric: Literal["log_loss", "poisson_nll"] = "log_loss",
) -> Comparison:
    """Compare log loss on paired games, using Amit's paired one-SE rule.

    LOSO is accepted only when explicitly requested as a retrospective stability
    table. No one-SE flags are computed for LOSO or for the last five folds.
    Exact ties in the overall leader require review: all leaders are marked close,
    while other close flags remain undetermined rather than inventing a reference.
    """
    if not results:
        raise ValueError("At least one evaluated configuration is required")
    if metric not in ("log_loss", "poisson_nll"):
        raise ValueError("Unknown selection metric")
    if metric == "poisson_nll" and not all(
        isinstance(r, GoalEvaluationResult) for r in results.values()
    ):
        raise ValueError("Poisson selection requires goal evaluations")
    modes = {r.mode for r in results.values()}
    if len(modes) != 1 or not modes <= {"walk_forward", "loso"}:
        raise ValueError("Configurations must share the same evaluation mode")
    mode = next(iter(modes))
    if mode == "loso" and for_selection:
        raise ValueError("LOSO is retrospective, not for selection")
    reference = None
    losses, counts, records = {}, {}, []
    for name in sorted(results):
        result = results[name]
        frame = result.per_game.set_index(["fold", "season_id", "game_id"]).sort_index()
        if not frame.index.is_unique:
            raise ValueError("Duplicate game keys in comparison")
        if reference is None:
            reference = frame
        elif not frame.index.equals(reference.index) or not np.array_equal(
            frame.outcome.to_numpy(), reference.outcome.to_numpy()
        ):
            raise ValueError("Configurations must score exactly the same games and outcomes")
        if metric == "poisson_nll" and reference is not None:
            if not np.array_equal(
                frame[["home_goals", "away_goals"]], reference[["home_goals", "away_goals"]]
            ):
                raise ValueError("Configurations must score identical observed goals")
        folds, summary = (
            summarize_goals(frame.reset_index())
            if metric == "poisson_nll"
            else summarize(frame.reset_index())
        )
        if mode == "walk_forward" and folds.season_id.tolist() != list(range(11, 26)):
            raise ValueError("Selection requires all 15 approved development folds")
        losses[name] = folds.set_index("season_id")[metric]
        counts[name] = folds.set_index("season_id")["rows" if metric == "poisson_nll" else "games"]
        records.append(
            {
                "config_id": name,
                "games": int(summary.games),
                "log_loss": summary.log_loss,
                "rps": summary.rps,
                "log_loss_std": summary.log_loss_std,
                "last5_log_loss": summary.last5_log_loss,
                "last5_rps": summary.last5_rps,
            }
        )
        if metric == "poisson_nll":
            records[-1].update(
                {
                    "rows": int(summary.rows),
                    "poisson_nll": summary.poisson_nll,
                    "poisson_nll_std": summary.poisson_nll_std,
                    "last5_poisson_nll": summary.last5_poisson_nll,
                }
            )
    loss_table = pd.DataFrame(losses)
    ranked = pd.DataFrame(records).set_index("config_id")
    ranked["rank_all"] = ranked[metric].rank(method="min").astype(int)
    ranked["rank_last5"] = ranked[f"last5_{metric}"].rank(method="min").astype(int)
    wins = loss_table.eq(loss_table.min(axis=1), axis=0).sum()
    ranked["folds_best"] = wins
    all_winners = tuple(ranked.index[ranked.rank_all.eq(1)])
    last_winners = tuple(ranked.index[ranked.rank_last5.eq(1)])
    disagreement = mode == "walk_forward" and set(all_winners) != set(last_winners)
    tied = len(all_winners) != 1 or len(last_winners) != 1
    if mode == "walk_forward":
        ranked["paired_mean_difference"] = np.nan
        ranked["paired_se"] = np.nan
        close: dict[str, bool | None] = {name: None for name in ranked.index}
        if len(all_winners) == 1:
            best = all_winners[0]
            for name in ranked.index:
                differences = loss_table[name] - loss_table[best]
                mean_difference = float(np.average(differences, weights=counts[best]))
                se = float(differences.std(ddof=1) / np.sqrt(15))
                ranked.loc[name, "paired_mean_difference"] = mean_difference
                ranked.loc[name, "paired_se"] = se
                close[name] = name == best or mean_difference <= se
        else:
            for name in all_winners:
                close[name] = True
        ranked["within_one_se"] = pd.Series(close, dtype="boolean")
    ranked = ranked.sort_values(["rank_all", "config_id"])
    return Comparison(
        ranked,
        all_winners,
        last_winners,
        disagreement,
        mode == "walk_forward" and (disagreement or tied),
        "15 development folds" if mode == "walk_forward" else RETROSPECTIVE_LABEL,
    )
