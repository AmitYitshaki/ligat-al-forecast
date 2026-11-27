"""Read-only M3 candidate evidence from the exact as-evaluated native units."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import t  # type: ignore[import-untyped]

from ligat.evaluate.goal_diagnostics import outcome_probabilities, poisson_score_matrix
from ligat.evaluate.goal_metrics import poisson_nll
from ligat.evaluate.goals_experiment import GoalUnit, load_goal_experiment
from ligat.evaluate.goals_post_hoc import load_post_hoc
from ligat.evaluate.goals_runner import _payload_digest
from ligat.evaluate.metrics import log_loss, rps
from ligat.viz import goals_walk_evidence as evidence
from ligat.viz import goals_walk_review as review

ROOT = Path("data/derived/exp005")
SCOPES = review.SCOPES
OUTCOMES = ["בית", "תיקו", "חוץ"]


def source_bundle() -> dict[str, Any]:
    """Read metadata only; never load historical/test-season feature tables or fit."""
    report = json.loads((ROOT / "full/ladder_report.json").read_bytes())
    if report["sha256"] != _payload_digest(report["payload"]):
        raise ValueError("Invalid signed ladder report")
    return dict(
        report=report["payload"],
        validation=pd.read_parquet(ROOT / "full/validation_views.parquet").assign(
            season_id=lambda p: p.fold + 10
        ),
        post=load_post_hoc(load_goal_experiment()),
    )


def load_candidates(b: dict[str, Any] | None = None) -> dict[str, Any]:
    """λ comes directly from the same native units as the candidate-vs-F5 table."""
    b = source_bundle() if b is None else b
    folds = evidence.candidate_folds(b)
    folds = {review.LABELS["F5"]: review.folds(b, "F5"), **folds}
    order = [review.LABELS[x] for x in ["F5", "glm", "EXP-002", "maher"]]
    order += [n for n in folds if "בדיעבד" in n] + [review.LABELS["R1"]]
    filters = [("season_id", ">=", 11), ("season_id", "<=", 25)]
    matches = pd.read_parquet("data/clean/matches.parquet", filters=filters).set_index("game_id")
    motivation = pd.read_parquet("data/derived/motivation/motivation_v1.parquet", filters=filters)
    active = motivation.groupby(level="game_id").nothing_to_play.max()
    frames = {}
    sources = []
    for name in order:
        rows = []
        for fold in folds[name].itertuples():
            if hasattr(fold, "unit_id"):
                path = ROOT / "full" / (fold.unit_id + ".json")
            else:
                check = "P4" if "CatBoost" in name else "P3"
                record = next(
                    r
                    for r in b["post"]["records"]
                    if r["check"] == check
                    and (
                        (
                            check == "P4"
                            and r["parameters"].get("iterations") == 150
                            and r["parameters"].get("l2_leaf_reg") == 10
                        )
                        or (
                            check == "P3"
                            and r["parameters"].get("max_depth") == 1
                            and r["parameters"].get("n_estimators") == 400
                            and r["parameters"].get("learning_rate") == 0.05
                        )
                    )
                )
                unit = GoalUnit(
                    "catboost" if check == "P4" else "xgb",
                    record["parameters"],
                    tuple(record["features"]),
                    int(fold.season_id),
                    record["half_life"],
                )
                path = ROOT / "diagnostics/post_hoc/synthetic" / (unit.unit_id + ".json")
            saved = json.loads(path.read_bytes())
            payload = saved["payload"]
            if saved["sha256"] != _payload_digest(payload):
                raise ValueError(f"Invalid unit {path}")
            if payload["unit"].get("prior_seasons_only", False):
                raise ValueError("Not as_evaluated")
            part = pd.DataFrame(payload["per_team"])
            if not part.season_id.between(11, 25).all():
                raise ValueError("Non-development rows")
            rows.append(part)
            sources.append(dict(candidate=name, path=str(path), season=int(fold.season_id)))
        rates = pd.concat(rows, ignore_index=True)
        if rates.duplicated(["game_id", "team_id"]).any():
            raise ValueError("Duplicate native rates")
        home = rates.loc[rates.is_home].set_index("game_id").sort_index()
        away = rates.loc[~rates.is_home].set_index("game_id").reindex(home.index)
        if away["lambda"].isna().any():
            raise ValueError("Unpaired game")
        p = matches.reindex(home.index).copy()
        assert p.season_id.equals(home.season_id)
        assert p.home_team_id.equals(home.team_id) and p.away_team_id.equals(away.team_id)
        p["lh"] = home["lambda"]
        p["la"] = away["lambda"]
        p["deep"] = home.deep_fold & away.deep_fold
        p["dead"] = active.reindex(p.index)
        assert p.dead.notna().all() and p[["lh", "la"]].gt(0).all().all()
        matrix = poisson_score_matrix(p.lh, p.la)
        prob = outcome_probabilities(matrix)
        p[["ph", "pd", "pa"]] = prob
        p["outcome"] = np.where(
            p.home_goals > p.away_goals, 0, np.where(p.home_goals == p.away_goals, 1, 2)
        )
        p["nll"] = (poisson_nll(p.home_goals, p.lh) + poisson_nll(p.away_goals, p.la)) / 2
        p["rps"] = rps(p.outcome, prob)
        p["log_loss"] = log_loss(p.outcome, prob)
        p["pearson"] = ((p.home_goals - p.lh) ** 2 / p.lh + (p.away_goals - p.la) ** 2 / p.la) / 2
        for h, a in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            p[f"p{h}{a}"] = matrix[:, h, a]
        total = np.add.outer(np.arange(6), np.arange(6))
        for goals in range(6):
            p[f"total{goals}"] = (
                matrix[:, total == goals].sum(axis=1)
                if goals < 5
                else matrix[:, total >= 5].sum(axis=1)
            )
        expected = folds[name].set_index("season_id").poisson_nll
        actual = p.groupby("season_id").nll.mean()
        if not np.allclose(actual.sort_index(), expected.sort_index(), rtol=0, atol=1e-10):
            raise ValueError("Rates differ from signed candidate scores")
        if np.average(p.log_loss) < 0.98:
            raise ValueError("Stop: configuration-average HDA below .98")
        frames[name] = p
    common = set.intersection(*(set(p.index) for p in frames.values()))
    if any(set(p.index) != common for p in frames.values()):
        raise ValueError("Unequal candidate coverage")
    deep = set.intersection(*(set(p.loc[p.deep].season_id) for p in frames.values()))
    return dict(frames=frames, deep_seasons=sorted(deep), sources=sources, policy="as_evaluated")


def scoped(data: dict[str, Any], scope: str) -> dict[str, Any]:
    frames = {}
    for name, p in data["frames"].items():
        mask = (
            p.season_id.isin(data["deep_seasons"])
            if scope == "deep"
            else p.season_id.ge(21)
            if scope == "last5"
            else p.season_id.ge(11)
        )
        frames[name] = p.loc[mask].copy()
    reference = frames[review.LABELS["R1"]]
    ranked = (reference.ph - reference.pa).abs().sort_values(kind="stable").index
    groups = pd.Series(np.minimum(np.arange(len(ranked)) * 3 // len(ranked), 2), index=ranked)
    for p in frames.values():
        p["balance"] = groups.reindex(p.index).astype(int)
    return frames


def _counts(p: Any) -> dict[str, int]:
    return dict(seasons=int(p.season_id.nunique()), games=len(p), rows=2 * len(p))


def dossier(data: dict[str, Any]) -> dict[str, Any]:
    """D1–D6, pooled games and season-level uncertainty, without model selection."""
    tables: dict[str, list[Any]] = {
        key: []
        for key in ["draw", "scores", "totals", "calibration", "segments", "dispersion", "summary"]
    }
    matrices = {}
    for scope in SCOPES:
        frames = scoped(data, scope)
        reference = frames[review.LABELS["F5"]]
        for name, p in frames.items():
            base = dict(candidate=name, scope=scope, post_hoc="בדיעבד" in name)
            for group in [-1, 0, 1, 2]:
                q = p if group == -1 else p.loc[p.balance.eq(group)]
                tables["draw"].append(
                    dict(
                        **base,
                        group=group,
                        predicted=q.pd.mean(),
                        observed=q.outcome.eq(1).mean(),
                        **_counts(q),
                    )
                )
            for h, a in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                observed = (p.home_goals.eq(h) & p.away_goals.eq(a)).mean()
                predicted = p[f"p{h}{a}"].mean()
                tables["scores"].append(
                    dict(
                        **base,
                        score=f"{h}-{a}",
                        predicted=predicted,
                        observed=observed,
                        ratio=predicted / observed if observed else np.nan,
                        **_counts(p),
                    )
                )
            for goals in range(6):
                observed = ((p.home_goals + p.away_goals).clip(upper=5) == goals).mean()
                predicted = p[f"total{goals}"].mean()
                tables["totals"].append(
                    dict(
                        **base,
                        goals=str(goals) if goals < 5 else "5+",
                        predicted=predicted,
                        observed=observed,
                        ratio=predicted / observed if observed else np.nan,
                        **_counts(p),
                    )
                )
            eces = []
            for outcome, field in enumerate(["ph", "pd", "pa"]):
                bins = np.minimum((p[field] * 10).astype(int), 9)
                ece = 0.0
                for bin_id in range(10):
                    q = p.loc[bins.eq(bin_id)]
                    predicted = q[field].mean()
                    observed = q.outcome.eq(outcome).mean()
                    if len(q):
                        ece += len(q) / len(p) * abs(predicted - observed)
                    tables["calibration"].append(
                        dict(
                            **base,
                            outcome=OUTCOMES[outcome],
                            bin=bin_id,
                            predicted=predicted,
                            observed=observed,
                            **_counts(q),
                        )
                    )
                eces.append(ece)
                for row in tables["calibration"][-10:]:
                    row["ece"] = ece
            seasons = p.groupby("season_id").pearson.mean()
            tables["dispersion"].append(
                dict(
                    **base,
                    pearson=p.pearson.mean(),
                    se=seasons.std(ddof=1) / np.sqrt(len(seasons)),
                    **_counts(p),
                )
            )
            segments = {f"איזון {group + 1}": p.balance.eq(group) for group in range(3)}
            segments.update({str(stage): p.stage.eq(stage) for stage in sorted(p.stage.unique())})
            segments.update({"דגל פעיל": p.dead.astype(bool), "ללא דגל": ~p.dead.astype(bool)})
            for label, mask in segments.items():
                q = p.loc[mask]
                r = reference.reindex(q.index)
                for metric in ["nll", "rps"]:
                    differences = (r[metric] - q[metric]).groupby(q.season_id).mean()
                    mean = float((r[metric] - q[metric]).mean())
                    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
                    radius = float(t.ppf(0.975, len(differences) - 1)) * se
                    tables["segments"].append(
                        dict(
                            **base,
                            segment=label,
                            metric=metric,
                            mean=q[metric].mean(),
                            d=mean,
                            se=se,
                            lower=mean - radius,
                            upper=mean + radius,
                            includes_zero=mean - radius <= 0 <= mean + radius,
                            **_counts(q),
                        )
                    )
            tables["summary"].append(
                dict(
                    **base,
                    nll=p.nll.mean(),
                    draw_gap=p.pd.mean() - p.outcome.eq(1).mean(),
                    zero_gap=p.p00.mean() - (p.home_goals.eq(0) & p.away_goals.eq(0)).mean(),
                    one_gap=p.p11.mean() - (p.home_goals.eq(1) & p.away_goals.eq(1)).mean(),
                    ece_draw=eces[1],
                    pearson=p.pearson.mean(),
                    **_counts(p),
                )
            )
        lambdas = pd.DataFrame(
            {name: p[["lh", "la"]].to_numpy().ravel() for name, p in frames.items()}
        )
        losses = pd.DataFrame({name: p.nll for name, p in frames.items()})
        matrices[scope] = dict(
            lambda_correlation=lambdas.corr(),
            nll_correlation=losses.corr(),
            lambda_difference=pd.DataFrame(
                {name: (lambdas.sub(lambdas[name], axis=0)).abs().mean() for name in lambdas}
            ),
        )
    result = {key: pd.DataFrame(rows) for key, rows in tables.items()}
    result["summary"]["maxcorr"] = [
        matrices[row.scope]["lambda_correlation"].loc[row.candidate].drop(row.candidate).max()
        for row in result["summary"].itertuples()
    ]
    result["matrices"] = matrices
    return result
