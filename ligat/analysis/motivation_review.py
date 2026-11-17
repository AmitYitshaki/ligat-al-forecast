"""Read-only, descriptive notebook-04 analysis; no feature selection or model fitting."""

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import yaml

from ligat.models.ordered_logit import OrderedFit
from ligat.motivation.artifacts import match_participants
from ligat.motivation.representations import game_contrast, level_l2, level_l3, level_l5, level_lc
from ligat.simulate.engine import LeagueEngine
from ligat.simulate.leverage import (
    E0050,
    SCORE_TARGETS,
    SeasonSimulator,
    conversion_fits,
    empirical_scores,
)
from ligat.simulate.rules import SeasonRules, load_season_rules
from ligat.simulate.snapshot import SeasonSnapshot, season_snapshots, snapshot_as_of

VALIDATION_SEASONS = tuple(range(11, 26))
LEVELS = {"L2": level_l2, "L3": level_l3, "L5": level_l5, "Lc": level_lc}
METRICS = ("res_pts", "res_gf", "res_ga")
STAGES = ("regular", "upper", "middle", "lower")
ZERO_GROUPS = ("places_only", "direct_only", "practical_zero", "other_mc")


@dataclass(frozen=True)
class MotivationReview:
    """Verified inputs plus an independently joined team-game descriptive frame."""

    frame: pd.DataFrame
    participants: pd.DataFrame
    prematch: pd.DataFrame
    table_state: pd.DataFrame
    motivation: pd.DataFrame
    elo: pd.DataFrame
    fits: Mapping[int, OrderedFit]
    rules: Mapping[int, SeasonRules]
    names: Mapping[int, str]
    metadata: Mapping[str, Any]
    verification: pd.DataFrame
    fingerprints: Mapping[str, str]
    seed: int


def _read_dev(path: Path, *, extra: list[tuple[str, str, Any]] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("season_id", "in", list(range(8, 26))), *(extra or [])])
    if frame.empty or not set(frame.season_id) <= set(range(8, 26)):
        raise ValueError("Notebook 04 permits only development seasons 8-25")
    return frame


def load_motivation_review(root: Path) -> MotivationReview:
    """Verify every derived input against its sidecar or associated experiment manifest.

    EXP-001/002 predictions have experiment manifests rather than adjacent sidecars;
    their exact manifest entries are verified before any selected rows are loaded.
    """
    fingerprints: dict[str, str] = {}
    records = []

    def checked(path: Path, manifest: Path, key: str | None = None) -> pd.DataFrame:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        expected = metadata["artifact_sha256"] if key is None else metadata["files"][key]["sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Checksum mismatch: {path}")
        fingerprints[path.relative_to(root).as_posix()] = actual
        fingerprints[manifest.relative_to(root).as_posix()] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        extra = (
            [("config_id", "==", E0050)]
            if "exp001_predictions" in path.name
            else (
                [("config_id", "==", "P_ewma5_venue_elo")] if path.parent.name == "exp002" else []
            )
        )
        frame = _read_dev(path, extra=extra)
        game_ids = (
            frame.index.get_level_values("game_id")
            if "game_id" in frame.index.names
            else frame.game_id
        )
        records.append(
            dict(
                file=path.relative_to(root).as_posix(),
                sha256=actual,
                manifest=manifest.relative_to(root).as_posix(),
                verified=True,
                n=len(frame),
                n_games=game_ids.nunique(),
                n_seasons=frame.season_id.nunique(),
            )
        )
        return frame

    base = root / "data/derived"
    motivation = checked(
        base / "motivation/motivation_v1.parquet", base / "motivation/motivation_v1.metadata.json"
    )
    prematch = checked(
        base / "prematch/prematch_v1.parquet", base / "prematch/prematch_v1.metadata.json"
    )
    table_state = checked(
        base / "prematch/table_state_v2.parquet", base / "prematch/table_state_v2.metadata.json"
    )
    converter_manifest = root / "reports/experiments/EXP-001/combined/metadata.json"
    ep = checked(
        base / "elo/exp001_predictions_combined.parquet",
        converter_manifest,
        "../../../../data/derived/elo/exp001_predictions_combined.parquet",
    )
    gp = checked(
        base / "exp002/predictions.parquet",
        root / "reports/experiments/EXP-002/full/metadata.json",
        "predictions.parquet",
    )
    matches = _read_dev(root / "data/clean/matches.parquet")
    fingerprints["data/clean/matches.parquet"] = hashlib.sha256(
        (root / "data/clean/matches.parquet").read_bytes()
    ).hexdigest()
    for relative in (
        "config/teams.yaml",
        "config/seasons.yaml",
        "config/motivation.yaml",
        "reports/experiments/EXP-001/combined/fold_scores.csv",
    ):
        fingerprints[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    cm = json.loads(converter_manifest.read_text(encoding="utf-8"))
    if (
        fingerprints["reports/experiments/EXP-001/combined/fold_scores.csv"]
        != cm["files"]["fold_scores.csv"]["sha256"]
    ):
        raise ValueError("Conversion parameters differ from the frozen manifest")
    metadata = json.loads(
        (base / "motivation/motivation_v1.metadata.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load((root / "config/motivation.yaml").read_text(encoding="utf-8"))
    names = {
        int(t["team_id"]): str(t["name_he"])
        for t in yaml.safe_load((root / "config/teams.yaml").read_text(encoding="utf-8"))["teams"]
    }
    participants = match_participants(matches)
    # Match the approved game_rows projection's nullable sporting goal columns.
    for column in ("goals_scored", "goals_conceded"):
        participants[column] = participants[column].astype("Int64")
    frame = join_review(prematch, motivation, participants, ep, gp)
    records.append(
        dict(
            file="data/clean/matches.parquet",
            sha256=fingerprints["data/clean/matches.parquet"],
            manifest="clean input; protected baseline",
            verified=True,
            n=len(matches),
            n_games=matches.game_id.nunique(),
            n_seasons=matches.season_id.nunique(),
        )
    )
    return MotivationReview(
        frame,
        participants,
        prematch,
        table_state,
        motivation,
        prematch[["season_id", "as_of", "elo_pre"]],
        conversion_fits(
            pd.read_csv(
                root / "reports/experiments/EXP-001/combined/fold_scores.csv",
                float_precision="round_trip",
            )
        ),
        load_season_rules(root / "config/seasons.yaml"),
        names,
        metadata,
        pd.DataFrame(records),
        fingerprints,
        int(config["seed"]),
    )


def join_review(
    prematch: pd.DataFrame,
    motivation: pd.DataFrame,
    participants: pd.DataFrame,
    elo_predictions: pd.DataFrame,
    goal_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join frozen before-game states with outcomes and strictly OOF validation forecasts."""
    if not prematch.index.equals(motivation.index):
        raise ValueError("M0 and M1 do not have identical ordered keys")
    shared = motivation.columns.intersection(prematch.columns)
    for col in shared:
        pd.testing.assert_series_equal(
            prematch[col], motivation[col], check_exact=True, check_dtype=False
        )
    data = prematch.join(motivation.drop(columns=list(shared)), validate="one_to_one")
    actual = participants.set_index(["game_id", "team_id"])[["goals_scored", "goals_conceded"]]
    data = data.join(actual, validate="one_to_one")
    if not elo_predictions.game_id.is_unique or not goal_predictions.game_id.is_unique:
        raise ValueError("OOF forecasts must identify each game once")
    data = data.join(
        elo_predictions.set_index("game_id")[["p_home", "p_draw", "p_away", "fold"]],
        on="game_id",
        validate="many_to_one",
    )
    data = data.join(
        goal_predictions.set_index("game_id")[["lambda_home", "lambda_away"]],
        on="game_id",
        validate="many_to_one",
    )
    data["gf"], data["ga"] = data.goals_scored.astype(float), data.goals_conceded.astype(float)
    data["pts"] = np.select([data.gf.gt(data.ga), data.gf.eq(data.ga)], [3.0, 1.0], default=0.0)
    data["p_win"] = np.where(data.is_home, data.p_home, data.p_away)
    data["xpts"] = 3 * data.p_win + data.p_draw
    data["res_pts"] = data.pts - data.xpts
    data["xgf"] = np.where(data.is_home, data.lambda_home, data.lambda_away)
    data["xga"] = np.where(data.is_home, data.lambda_away, data.lambda_home)
    data["res_gf"], data["res_ga"] = data.gf - data.xgf, data.ga - data.xga
    data["title_close"] = (
        ~data.title_decided & data.gap_to_leader.abs().le(3) & data.games_remaining_stage.le(8)
    )
    data["cut_close"] = (
        ~data.cut_decided
        & data.stage.eq("regular")
        & data.gap_to_cut.abs().le(3)
        & data.games_remaining_stage.le(8)
    )
    data["releg_close"] = (
        ~data.relegation_decided & data.gap_to_safety.abs().le(3) & data.games_remaining_stage.le(8)
    )
    bands = pd.cut(
        data.games_remaining_stage, [-np.inf, 1, 3, 6, np.inf], labels=["<=1", "2-3", "4-6", "7+"]
    )
    data["stratum"] = data.stage.astype(str) + " / " + bands.astype(str)
    for key, fn in LEVELS.items():
        data[key] = fn(data)
    opponent = data[["nothing_to_play"]].rename(columns={"nothing_to_play": "opp_dead"})
    keys = pd.MultiIndex.from_arrays(
        [data.index.get_level_values("game_id"), data.opponent_id], names=data.index.names
    )
    data["opp_dead"] = opponent.reindex(keys).opp_dead.to_numpy(bool)
    val = data.loc[data.season_id.ge(11)]
    if val[[*METRICS, "p_home", "p_draw", "p_away"]].isna().any().any():
        raise ValueError("Missing OOF validation forecasts")
    return data


def season_bootstrap_weights(years: Sequence[int], repeats: int, seed: int) -> pd.DataFrame:
    """Resample whole seasons; every row of a season, including both opponents, shares weight."""
    if not years or len(set(years)) != len(years) or repeats < 2:
        raise ValueError("Need distinct seasons and at least two bootstrap replicates")
    draws = np.random.default_rng(seed).integers(0, len(years), size=(repeats, len(years)))
    weights = np.stack([np.bincount(row, minlength=len(years)) for row in draws])
    return pd.DataFrame(weights, columns=list(years))


def _finite_std(values: np.ndarray[Any, Any]) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if len(finite) > 1 else np.nan


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    games = (
        frame.index.get_level_values("game_id") if "game_id" in frame.index.names else frame.game_id
    )
    return dict(
        n=len(frame), n_games=int(games.nunique()), n_seasons=int(frame.season_id.nunique())
    )


def within_strata_contrast(
    frame: pd.DataFrame,
    mask: pd.Series,
    metric: str,
    *,
    repeats: int = 2000,
    seed: int = 14112026,
    years: Sequence[int] | None = None,
    include_selected_in_reference: bool = False,
) -> dict[str, Any]:
    """Category-minus-complement means, weighted by usable category counts per stratum."""
    data = frame.loc[frame[metric].notna()]
    selected = mask.reindex(data.index).to_numpy(bool)
    reference = np.ones_like(selected) if include_selected_in_reference else ~selected
    seasons = sorted(int(s) for s in frame.season_id.unique()) if years is None else list(years)
    strata = pd.Categorical(data.stratum)
    width = len(strata.categories)
    year_codes = pd.Categorical(data.season_id, categories=seasons).codes.astype(np.int64)
    if (year_codes < 0).any():
        raise ValueError("Requested season universe does not cover the data")
    values = data[metric].to_numpy(float)
    index = year_codes * width + strata.codes
    if not width:
        return dict(estimate=np.nan, se=np.nan, **_counts(data), n_used=0, n_excluded=0)
    arrays = np.stack(
        [
            np.bincount(index, weights=w, minlength=len(seasons) * width).reshape(
                len(seasons), width
            )
            for w in (
                selected.astype(float),
                values * selected,
                reference.astype(float),
                values * reference,
            )
        ],
        axis=2,
    )

    def estimate(moment: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nb, sb, nr, sr = np.moveaxis(moment, -1, 0)
        good = (nb > 0) & (nr > 0)
        delta = np.divide(sb, nb, out=np.zeros_like(sb), where=nb > 0) - np.divide(
            sr, nr, out=np.zeros_like(sr), where=nr > 0
        )
        used = np.where(good, nb, 0)
        den = used.sum(axis=-1)
        return np.asarray(
            np.divide(
                (delta * used).sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 0
            )
        )

    total = arrays.sum(axis=0)
    weights = season_bootstrap_weights(seasons, repeats, seed).to_numpy()
    samples = estimate(np.einsum("bs,skm->bkm", weights, arrays))
    valid = (total[:, 0] > 0) & (total[:, 2] > 0)
    used = int(total[valid, 0].sum())
    group = data.loc[mask.reindex(data.index)]
    return dict(
        estimate=float(estimate(total)),
        se=_finite_std(samples) if group.season_id.nunique() > 1 else np.nan,
        **_counts(group),
        n_used=used,
        n_excluded=len(group) - used,
    )


def stratified_decomposition(
    frame: pd.DataFrame,
    mask: pd.Series,
    metric: str,
    *,
    minimum_comparison: int = 1,
    repeats: int = 2000,
    seed: int = 14112026,
    years: Sequence[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Decompose the existing contrast with identical weights and season draws.

    Sensitivity eligibility is fixed in the observed sample, not reselected in each
    bootstrap draw. Single-season uncertainty is unidentified and reported as NA.
    """
    if minimum_comparison not in (1, 10):
        raise ValueError("Only the original estimator and the preregistered threshold 10")
    data = frame.loc[frame[metric].notna()]
    selected = mask.reindex(data.index).to_numpy(bool)
    seasons = sorted(int(s) for s in frame.season_id.unique()) if years is None else list(years)
    strata = pd.Categorical(data.stratum)
    width = len(strata.categories)
    codes = pd.Categorical(data.season_id, categories=seasons).codes.astype(np.int64)
    if not width or (codes < 0).any():
        raise ValueError("Need nonempty strata and a season universe covering the data")
    values = data[metric].to_numpy(float)
    index = codes * width + strata.codes
    arrays = np.stack(
        [
            np.bincount(index, weights=w, minlength=len(seasons) * width).reshape(
                len(seasons), width
            )
            for w in (
                selected.astype(float),
                values * selected,
                (~selected).astype(float),
                values * (~selected),
            )
        ],
        axis=2,
    )
    total = arrays.sum(axis=0)
    admissible = (total[:, 0] > 0) & (total[:, 2] >= minimum_comparison)

    def parts(moment: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        nb, sb, nc, sc = np.moveaxis(moment, -1, 0)
        good = admissible & (nb > 0) & (nc > 0)
        used = np.where(good, nb, 0)
        den = used.sum(axis=-1)
        means = []
        for sums, counts in ((sb, nb), (sc, nc)):
            mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            means.append(
                np.divide(
                    (mean * used).sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 0
                )
            )
        return np.stack([means[0], means[1], means[0] - means[1]], axis=-1)

    point = parts(total)
    weights = season_bootstrap_weights(seasons, repeats, seed).to_numpy()
    draws = parts(np.einsum("bs,skm->bkm", weights, arrays))
    group = data.loc[selected]
    used = int(total[admissible, 0].sum())
    names = ("dead_mean", "control_mean", "estimate")
    result: dict[str, Any] = dict(
        **_counts(group),
        n_used=used,
        n_excluded=len(group) - used,
        minimum_comparison=minimum_comparison,
        dead_raw_mean=float(group[metric].mean()),
    )
    for j, name in enumerate(names):
        result[name] = float(point[j])
        result["se" if name == "estimate" else f"{name}_se"] = (
            _finite_std(draws[:, j]) if group.season_id.nunique() > 1 else np.nan
        )
    layers = pd.DataFrame(
        dict(
            stratum=strata.categories,
            n_dead=total[:, 0].astype(int),
            n_control=total[:, 2].astype(int),
            included=admissible,
            dead_mean=np.divide(
                total[:, 1], total[:, 0], out=np.full(width, np.nan), where=total[:, 0] > 0
            ),
            control_mean=np.divide(
                total[:, 3], total[:, 2], out=np.full(width, np.nan), where=total[:, 2] > 0
            ),
        )
    )
    layers["weight"] = np.where(admissible, total[:, 0] / used if used else 0, 0)
    layers["contribution"] = np.where(
        admissible, (layers.dead_mean - layers.control_mean) * layers.weight, 0
    )
    if used and not np.isclose(layers.contribution.sum(), result["estimate"], atol=1e-12):
        raise AssertionError("Stratum contributions do not reconstruct the contrast")
    return result, layers


def flag_decomposition_review(
    frame: pd.DataFrame, *, seed: int, repeats: int = 2000
) -> dict[str, pd.DataFrame]:
    """All 15 seasons, individual years and the three fixed five-season blocks."""
    data = validation_rows(frame)
    scopes = {
        "all15": VALIDATION_SEASONS,
        **SEASON_BLOCKS,
        **{f"S{s}": (s,) for s in VALIDATION_SEASONS},
    }
    original, sensitive, layers = [], [], []
    for scope, years in scopes.items():
        part = data.loc[data.season_id.isin(years)]
        for metric in ("res_pts", "res_ga"):
            result, detail = stratified_decomposition(
                part, part.nothing_to_play, metric, seed=seed, repeats=repeats, years=years
            )
            original.append(dict(scope=scope, metric=metric, **result))
            if scope in SEASON_BLOCKS and metric == "res_pts":
                layers.append(detail.assign(scope=scope, metric=metric))
            if not scope.startswith("S"):
                sensitivity, _ = stratified_decomposition(
                    part,
                    part.nothing_to_play,
                    metric,
                    minimum_comparison=10,
                    seed=seed,
                    repeats=repeats,
                    years=years,
                )
                sensitive.append(dict(scope=scope, metric=metric, **sensitivity))
    return dict(
        original=pd.DataFrame(original),
        sensitivity=pd.DataFrame(sensitive),
        layers=pd.concat(layers, ignore_index=True),
    )


def closed_game_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One home row per game; exclusive types in preregistered priority order."""
    data = validation_rows(frame)
    game_contrast(data, "L2")  # Validate paired opponents before keeping the home row.
    games = data.loc[data.is_home].copy()
    ids = data.index.get_level_values("game_id")
    dead = data.nothing_to_play.groupby(ids).sum()
    close = (data.releg_close & ~data.nothing_to_play).groupby(ids).any()
    home_ids = games.index.get_level_values("game_id")
    games["one_dead"] = dead.reindex(home_ids).to_numpy() == 1
    games["game_type"] = np.select(
        [
            close.reindex(home_ids).to_numpy(),
            dead.reindex(home_ids).to_numpy() == 2,
            games.one_dead.to_numpy(),
        ],
        ["relegation_alive", "both_dead", "one_dead"],
        default="other",
    )
    games["actual_goals"] = games.gf + games.ga
    games["expected_goals"] = games.lambda_home + games.lambda_away
    games["res_total_goals"] = games.actual_goals - games.expected_goals
    games["actual_draw"] = games.gf.eq(games.ga).astype(float)
    games["expected_draw"] = games.p_draw
    games["res_draw"] = games.actual_draw - games.expected_draw
    return games


def closed_game_review(
    frame: pd.DataFrame, *, seed: int, repeats: int = 2000
) -> dict[str, pd.DataFrame]:
    """Raw calibration and home-stratum-adjusted comparisons; no predictive fitting."""
    games = closed_game_rows(frame)
    weights = season_bootstrap_weights(VALIDATION_SEASONS, repeats, seed).to_numpy()
    raw, adjusted, adjusted_other = [], [], []
    for kind in ("relegation_alive", "both_dead", "one_dead", "other"):
        mask = games.game_type.eq(kind)
        group = games.loc[mask]
        row: dict[str, Any] = dict(game_type=kind, **_counts(group))
        for column in ("actual_goals", "expected_goals", "actual_draw", "expected_draw"):
            row[column] = float(group[column].mean())
        for metric in ("res_total_goals", "res_draw"):
            moments = (
                group.groupby("season_id")[metric]
                .agg(["sum", "count"])
                .reindex(VALIDATION_SEASONS, fill_value=0)
            )
            counts = weights @ moments["count"].to_numpy()
            draws = np.divide(
                weights @ moments["sum"].to_numpy(),
                counts,
                out=np.full(repeats, np.nan),
                where=counts > 0,
            )
            row[metric] = float(group[metric].mean())
            row[f"{metric}_se"] = _finite_std(draws) if group.season_id.nunique() > 1 else np.nan
            if kind != "other":
                adjusted.append(
                    dict(
                        game_type=kind,
                        metric=metric,
                        **within_strata_contrast(
                            games,
                            mask,
                            metric,
                            years=VALIDATION_SEASONS,
                            seed=seed,
                            repeats=repeats,
                        ),
                    )
                )
                adjusted_other.append(
                    dict(
                        game_type=kind,
                        metric=metric,
                        **other_only_contrast(games, kind, metric, seed=seed, repeats=repeats),
                    )
                )
        raw.append(row)
    annual = games.groupby("season_id").agg(
        n_games=("one_dead", "size"), n_one_dead=("one_dead", "sum")
    )
    annual["one_dead_pct"] = 100 * annual.n_one_dead / annual.n_games
    return dict(
        games=games,
        raw=pd.DataFrame(raw),
        adjusted=pd.DataFrame(adjusted),
        adjusted_other=pd.DataFrame(adjusted_other),
        annual=annual.reset_index(),
    )


def other_only_contrast(
    games: pd.DataFrame, kind: str, metric: str, *, seed: int, repeats: int = 2000
) -> dict[str, Any]:
    """Compare the declared type only to `other`, excluding every other type entirely."""
    if kind not in ("relegation_alive", "both_dead", "one_dead"):
        raise ValueError("The reference category other cannot be compared to itself")
    universe = games.loc[games.game_type.isin([kind, "other"])]
    return within_strata_contrast(
        universe,
        universe.game_type.eq(kind),
        metric,
        years=VALIDATION_SEASONS,
        seed=seed,
        repeats=repeats,
    )


def season_residual_mean(
    games: pd.DataFrame, metric: str, *, years: Sequence[int], seed: int, repeats: int = 2000
) -> dict[str, Any]:
    """Mean residual with whole-season cluster uncertainty, including an explicit empty cell."""
    data = games.loc[games[metric].notna()]
    if not set(data.season_id).issubset(years):
        raise ValueError("Requested season universe does not cover the data")
    moments = data.groupby("season_id")[metric].agg(["sum", "count"]).reindex(years, fill_value=0)
    weights = season_bootstrap_weights(years, repeats, seed).to_numpy()
    counts = weights @ moments["count"].to_numpy()
    draws = np.divide(
        weights @ moments["sum"].to_numpy(), counts, out=np.full(repeats, np.nan), where=counts > 0
    )
    return dict(
        estimate=float(data[metric].mean()),
        se=_finite_std(draws) if data.season_id.nunique() > 1 else np.nan,
        **_counts(data),
        n_used=len(data),
        n_excluded=len(games) - len(data),
    )


def closed_game_period_review(
    frame: pd.DataFrame, *, seed: int, repeats: int = 2000
) -> pd.DataFrame:
    """Raw OOF goals/draw calibration by game type and the two approved season periods.

    Each bootstrap samples only whole seasons of its own period. The existing
    all-fifteen-season table remains a separate, unchanged calculation.
    """
    games = closed_game_rows(frame)
    rows = []
    for period, years in (("first10", tuple(range(11, 21))), ("last5", SEASON_BLOCKS["B3"])):
        part = games.loc[games.season_id.isin(years)]
        for kind in ("relegation_alive", "both_dead", "one_dead", "other"):
            group = part.loc[part.game_type.eq(kind)]
            row: dict[str, Any] = dict(period=period, **_counts(group), game_type=kind)
            for column in ("actual_goals", "expected_goals", "actual_draw", "expected_draw"):
                row[column] = float(group[column].mean())
            for metric in ("res_total_goals", "res_draw"):
                result = season_residual_mean(
                    group, metric, years=years, seed=seed, repeats=repeats
                )
                row[metric] = result["estimate"]
                row[f"{metric}_se"] = result["se"]
            rows.append(row)
    return pd.DataFrame(rows)


def stage_residual_review(frame: pd.DataFrame, *, seed: int, repeats: int = 2000) -> pd.DataFrame:
    """Frozen OOF goals/draw calibration by stage, with clean/all populations and recent years."""
    games = closed_game_rows(frame)
    rows = []
    for stage in ("regular", "upper", "lower", "middle"):
        for population in ("clean", "all_games"):
            group = games.loc[games.stage.eq(stage)]
            if population == "clean":
                group = group.loc[group.game_type.eq("other")]
            recent = group.loc[group.season_id.isin(SEASON_BLOCKS["B3"])]
            row: dict[str, Any] = dict(
                stage=stage,
                population=population,
                **_counts(group),
                last5_n_games=len(recent),
                last5_n_seasons=int(recent.season_id.nunique()),
            )
            for metric, label in (("res_total_goals", "goals"), ("res_draw", "draw")):
                for prefix, part, years in (
                    ("", group, VALIDATION_SEASONS),
                    ("last5_", recent, SEASON_BLOCKS["B3"]),
                ):
                    result = season_residual_mean(
                        part, metric, years=years, seed=seed, repeats=repeats
                    )
                    row[f"{prefix}{label}_residual"] = result["estimate"]
                    row[f"{prefix}{label}_se"] = result["se"]
            rows.append(row)
    return pd.DataFrame(rows)


def verify_rounded_claims(
    claims: Mapping[str, str], calculated: Mapping[str, tuple[float, int]]
) -> pd.DataFrame:
    """Compare independently authored display numbers to computed values at display precision."""
    if claims.keys() != calculated.keys():
        raise AssertionError("Missing or unexpected numerical claim")
    rows = []
    for key, text in claims.items():
        value, precision = calculated[key]
        expected = f"{value:.{precision}f}"
        matches = text.replace("\N{MINUS SIGN}", "-") == expected
        rows.append(dict(claim=key, displayed=text, calculated=value, matches=matches))
    result = pd.DataFrame(rows)
    if not result.matches.all():
        raise AssertionError(f"Bottom-line numerical mismatch: {result.to_dict('records')}")
    return result


def within_strata_slope(
    frame: pd.DataFrame,
    predictor: str,
    metric: str = "res_pts",
    *,
    repeats: int = 2000,
    seed: int = 14112026,
    years: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Slope after centering x and y within strata; bootstrap recenters every resample."""
    data = frame.dropna(subset=[predictor, metric])
    seasons = sorted(int(s) for s in frame.season_id.unique()) if years is None else list(years)
    strata = pd.Categorical(data.stratum)
    width = len(strata.categories)
    if not width:
        return dict(estimate=np.nan, se=np.nan, **_counts(data), n_used=0, n_excluded=0)
    year_codes = pd.Categorical(data.season_id, categories=seasons).codes.astype(np.int64)
    index = year_codes * width + strata.codes
    x, y = data[predictor].to_numpy(float), data[metric].to_numpy(float)
    arrays = np.stack(
        [
            np.bincount(index, weights=w, minlength=len(seasons) * width).reshape(
                len(seasons), width
            )
            for w in (np.ones(len(data)), x, y, x * x, x * y)
        ],
        axis=2,
    )

    def estimate(moment: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        n, sx, sy, sxx, sxy = np.moveaxis(moment, -1, 0)
        xx = sxx - np.divide(sx * sx, n, out=np.zeros_like(sx), where=n > 0)
        xy = sxy - np.divide(sx * sy, n, out=np.zeros_like(sx), where=n > 0)
        den = xx.sum(axis=-1)
        return np.asarray(
            np.divide(xy.sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 1e-12)
        )

    samples = estimate(
        np.einsum(
            "bs,skm->bkm", season_bootstrap_weights(seasons, repeats, seed).to_numpy(), arrays
        )
    )
    return dict(
        estimate=float(estimate(arrays.sum(axis=0))),
        se=_finite_std(samples) if len(seasons) > 1 else np.nan,
        **_counts(data),
        n_used=len(data),
        n_excluded=0,
    )


def naive_paired_se_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare independent-binomial SE to the saved paired Monte Carlo SE by objective."""
    records = []
    for target in SCORE_TARGETS:
        part = frame.loc[frame[f"{target}_p_win"].notna()]
        pw, pl = part[f"{target}_p_win"], part[f"{target}_p_loss"]
        naive = np.sqrt((pw * (1 - pw) + pl * (1 - pl)) / part.N)
        paired = part[f"{target}_mc_se"]
        ratio = naive.loc[paired.gt(0)] / paired.loc[paired.gt(0)]
        infinite = paired.eq(0) & naive.gt(0)
        ratio = pd.concat([ratio, pd.Series(np.inf, index=part.index[infinite])])
        records.append(
            dict(
                target=target,
                **_counts(part),
                n_ratio=len(ratio),
                paired_zero=int(paired.eq(0).sum()),
                infinite_ratio=int(infinite.sum()),
                undefined_ratio=int((paired.eq(0) & naive.eq(0)).sum()),
                q25=ratio.quantile(0.25, interpolation="nearest"),
                median=ratio.quantile(0.5, interpolation="nearest"),
                q75=ratio.quantile(0.75, interpolation="nearest"),
            )
        )
    return pd.DataFrame(records)


def zero_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Partition alive zero-score rows by places, diagnostic safety, paired zero, or MC residue."""
    zero = ~frame.nothing_to_play & frame.leverage_max.eq(0)
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    places = (
        zero
        & frame[["title_decided", "cut_decided", "relegation_decided"]].all(axis=1)
        & ~frame.places_decided
    )
    direct = zero & ~places & frame.survival_direct_lev.gt(0)
    exact = frame[[f"{t}_lev_raw" for t in SCORE_TARGETS]].fillna(0).eq(0).all(axis=1)
    exact &= frame[[f"{t}_mc_se" for t in SCORE_TARGETS]].fillna(0).eq(0).all(axis=1)
    practical = zero & ~places & ~direct & exact
    result.loc[places], result.loc[direct], result.loc[practical] = (
        "places_only",
        "direct_only",
        "practical_zero",
    )
    result.loc[zero & result.isna()] = "other_mc"
    data = frame.loc[zero].copy()
    data["zero_group"] = result.loc[zero]
    if data.zero_group.isna().any() or len(data) != int(zero.sum()):
        raise AssertionError("Zero decomposition is not exhaustive")
    return data


def validation_rows(frame: pd.DataFrame, *, last_five: bool = False) -> pd.DataFrame:
    """Keep OOF seasons 11-25, or 21-25; never use warm-up outcomes for residual inference."""
    return frame.loc[frame.season_id.between(21 if last_five else 11, 25)].copy()


def level_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count ordinal levels by stage, retaining both stage and full-population denominators."""
    records = []
    for level, fn in (("L3", level_l3), ("L5", level_l5)):
        values = fn(frame)
        for stage, part in frame.groupby("stage", observed=True, sort=False):
            for value in sorted(values.unique()):
                group = part.loc[values.reindex(part.index).eq(value)]
                records.append(
                    dict(
                        level=level,
                        stage=stage,
                        value=value,
                        **_counts(group),
                        fraction_stage=len(group) / len(part),
                        fraction_all=len(group) / len(frame),
                    )
                )
    return pd.DataFrame(records)


def distribution_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize frozen importance; tied leading targets follow the pilot's target order."""
    score = frame.leverage_max
    overall = dict(
        **_counts(frame),
        mean=score.mean(),
        zero_fraction=score.eq(0).mean(),
        **{f"q{q}": score.quantile(q / 100) for q in (25, 50, 75, 90, 99)},
    )
    season = []
    distance = []
    dominant = []
    for sid, part in frame.groupby("season_id", sort=True):
        season.append(
            dict(
                season_id=int(sid),
                **_counts(part),
                mean=part.leverage_max.mean(),
                median=part.leverage_max.median(),
                above_02=part.leverage_max.gt(0.2).mean(),
            )
        )
    for (stage, remaining), part in frame.groupby(
        ["stage", "games_remaining_stage"], observed=True, sort=True
    ):
        distance.append(
            dict(
                stage=stage,
                remaining=int(remaining),
                mean=part.leverage_max.mean(),
                **_counts(part),
            )
        )
    leading = (
        frame[[f"{t}_lev" for t in SCORE_TARGETS]]
        .fillna(-np.inf)
        .idxmax(axis=1)
        .str.removesuffix("_lev")
    )
    leading = leading.where(score.gt(0), "zero")
    for stage, part in frame.groupby("stage", observed=True, sort=False):
        for target in (*SCORE_TARGETS, "zero"):
            group = part.loc[leading.reindex(part.index).eq(target)]
            dominant.append(
                dict(stage=stage, target=target, **_counts(group), fraction=len(group) / len(part))
            )
    return dict(
        overall=pd.DataFrame([overall]),
        seasons=pd.DataFrame(season),
        distance=pd.DataFrame(distance),
        dominant=pd.DataFrame(dominant),
    )


def gap_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise-complete Spearman correlations among alive rows, pooled and by stage."""
    alive = frame.loc[~frame.nothing_to_play]
    records = []
    for stage, part in [("all", alive), *list(alive.groupby("stage", observed=True, sort=False))]:
        for feature in ("gap_to_safety", "gap_to_leader", "gap_to_cut", "games_remaining_stage"):
            pair = part.dropna(subset=[feature, "leverage_max"])
            x = pair[feature].abs() if feature in ("gap_to_safety", "gap_to_cut") else pair[feature]
            rho = (
                pair.leverage_max.corr(x, method="spearman")
                if len(pair) > 1 and x.nunique() > 1 and pair.leverage_max.nunique() > 1
                else np.nan
            )
            records.append(dict(stage=stage, feature=feature, rho=rho, **_counts(pair)))
    return pd.DataFrame(records)


def close_importance_crosstab(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross transparent close-race definitions with the preregistered L5 bins."""
    data = frame.loc[~frame.nothing_to_play]
    records = []
    for close in ("title_close", "cut_close", "releg_close"):
        for state in (False, True):
            part = data.loc[data[close].eq(state)]
            for level in (0.25, 0.5, 0.75, 1.0):
                group = part.loc[part.L5.eq(level)]
                records.append(
                    dict(
                        close=close,
                        is_close=state,
                        level=level,
                        **_counts(group),
                        fraction=len(group) / len(part) if len(part) else np.nan,
                    )
                )
    return pd.DataFrame(records)


def _category_specs(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    specs = [
        ("L2", "dead", frame.nothing_to_play),
        ("L2", "dead_vs_alive_opponent", frame.nothing_to_play & ~frame.opp_dead),
        ("L2", "alive_vs_dead_opponent", ~frame.nothing_to_play & frame.opp_dead),
    ]
    for level, values in (("L3", (0.0, 0.5, 1.0)), ("L5", (0.0, 0.25, 0.5, 0.75, 1.0))):
        for value in values:
            specs.append((level, f"{value:g}", frame[level].eq(value)))
    leading = frame[[f"{t}_lev" for t in SCORE_TARGETS]].fillna(-np.inf).idxmax(axis=1)
    for target in SCORE_TARGETS:
        specs.append(("Comp", target, leading.eq(f"{target}_lev") & frame[f"{target}_lev"].gt(0.2)))
    for target in ("title_close", "cut_close", "releg_close"):
        specs.append(("transparent", target, frame[target]))
    return specs


def residual_comparisons(
    frame: pd.DataFrame, *, seed: int, repeats: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All requested descriptive contrasts, last-five checks and per-season sign stability."""
    data = validation_rows(frame)
    overall = []
    seasonal = []
    for representation, category, mask in _category_specs(data):
        for metric in METRICS:
            common = dict(
                representation=representation, category=category, metric=metric, method="contrast"
            )
            result = within_strata_contrast(
                data, mask, metric, seed=seed, repeats=repeats, years=VALIDATION_SEASONS
            )
            recent = validation_rows(data, last_five=True)
            last = within_strata_contrast(
                recent,
                mask.reindex(recent.index),
                metric,
                seed=seed,
                repeats=repeats,
                years=tuple(range(21, 26)),
            )
            signs = []
            for sid, part in data.groupby("season_id", sort=True):
                one = within_strata_contrast(
                    part, mask.reindex(part.index), metric, seed=seed, repeats=2, years=[int(sid)]
                )
                seasonal.append(dict(**common, season_id=int(sid), **one))
                if one["n_used"] >= 30 and np.isfinite(one["estimate"]):
                    signs.append(one["estimate"])
            same = sum(np.sign(v) == np.sign(result["estimate"]) for v in signs)
            overall.append(
                dict(
                    **common,
                    **result,
                    last5_estimate=last["estimate"],
                    last5_se=last["se"],
                    last5_n=last["n"],
                    last5_n_used=last["n_used"],
                    sign_same=same,
                    sign_eligible=len(signs),
                )
            )
    home = data.loc[data.is_home].copy()
    for level in LEVELS:
        games = home.join(game_contrast(data, level), on="game_id", validate="one_to_one")
        common = dict(
            representation=level, category="home_minus_away", metric="res_pts", method="slope"
        )
        result = within_strata_slope(
            games, "m_diff", seed=seed, repeats=repeats, years=VALIDATION_SEASONS
        )
        recent = validation_rows(games, last_five=True)
        last = within_strata_slope(
            recent, "m_diff", seed=seed, repeats=repeats, years=tuple(range(21, 26))
        )
        signs = []
        for sid, part in games.groupby("season_id", sort=True):
            one = within_strata_slope(part, "m_diff", seed=seed, repeats=2, years=[int(sid)])
            seasonal.append(dict(**common, season_id=int(sid), **one))
            if one["n_used"] >= 30 and np.isfinite(one["estimate"]):
                signs.append(one["estimate"])
        overall.append(
            dict(
                **common,
                **result,
                last5_estimate=last["estimate"],
                last5_se=last["se"],
                last5_n=last["n"],
                last5_n_used=last["n_used"],
                sign_same=sum(np.sign(v) == np.sign(result["estimate"]) for v in signs),
                sign_eligible=len(signs),
            )
        )
    return pd.DataFrame(overall), pd.DataFrame(seasonal)


def verify_pilot_estimates(frame: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Hard gate against the pilot's original comparison universes, not newly defined levels."""
    data = validation_rows(frame)
    baseline = float(data.res_pts.mean())
    if abs(baseline - 0.00105122) > 1e-8:
        raise AssertionError(f"Pilot mean mismatch: current={baseline:.17g}; reference=0.00105122")
    ref = reference.loc[
        reference.method.eq("within_strata_vs_rest")
        & reference.family.isin(["flag_baseline", "all_importance", "alive_only_importance"])
    ]
    errors = []
    checked = 0
    for row in ref.itertuples():
        part = data.loc[~data.nothing_to_play] if row.family == "alive_only_importance" else data
        value = part.leverage_max
        masks = {
            "negative": value.lt(0),
            "zero": value.eq(0),
            "0_to_005": value.gt(0) & value.le(0.05),
            "005_to_02": value.gt(0.05) & value.le(0.2),
            "above_02": value.gt(0.2),
            "up_to_001": value.le(0.01),
            "001_to_005": value.gt(0.01) & value.le(0.05),
            "dead": part.nothing_to_play,
            "alive": ~part.nothing_to_play,
        }
        current = within_strata_contrast(part, masks[row.category], row.metric, repeats=2)[
            "estimate"
        ]
        if np.isnan(current) and pd.isna(row.estimate):
            continue
        error = abs(current - row.estimate)
        if not np.isfinite(error) or error > 1e-9:
            raise AssertionError(
                f"Pilot contrast mismatch {row.family}/{row.category}/{row.metric}: "
                f"current={current}; pilot={row.estimate}"
            )
        errors.append(error)
        checked += 1
    rho = float(data.leverage_max.corr(data.pts, method="spearman"))
    if abs(rho) > 0.2:
        raise AssertionError(f"Leakage alarm: importance/outcome Spearman={rho}")
    zero = zero_partition(frame)
    if len(zero) != 778:
        raise AssertionError(f"Zero partition: expected 778, got {len(zero)}")
    return dict(
        mean_res_pts=baseline,
        reference_mean=0.00105122,
        pilot_max_error=max(errors, default=0.0),
        pilot_contrasts=checked,
        spearman_outcome=rho,
        zero_rows=len(zero),
        zero_counts={k: int(zero.zero_group.eq(k).sum()) for k in ZERO_GROUPS},
    )


def example_views(review: MotivationReview) -> dict[str, pd.DataFrame]:
    """Select the requested examples by season/date/round, with explicit one-row counts."""
    frame = review.frame
    maccabi = [t for t, n in review.names.items() if n in ("מכבי תל אביב", 'מכבי ת"א')]
    if len(maccabi) != 1:
        raise ValueError("Cannot uniquely resolve Maccabi Tel Aviv from teams.yaml")
    views = {
        "split_2011": frame.loc[
            frame.season_id.eq(12) & frame.stage.eq("regular") & frame["round"].eq(30)
        ],
        "maccabi_2019": frame.loc[
            frame.season_id.eq(20) & (frame.index.get_level_values("team_id") == maccabi[0])
        ],
        "ones_2020": frame.loc[
            frame.season_id.eq(21)
            & frame.stage.eq("regular")
            & frame["round"].eq(review.rules[21].regular_rounds)
            & frame.leverage_max.eq(1)
        ],
        "july_2020": frame.loc[
            frame.season_id.eq(21)
            & frame.stage.eq("lower")
            & frame.as_of.dt.year.eq(2020)
            & frame.as_of.dt.month.eq(7)
        ],
        "test_match_limit": frame.loc[
            frame.as_of.eq(pd.Timestamp("2010-05-15", tz="Asia/Jerusalem"))
            & frame.survival_direct_lev.gt(0.3)
        ],
    }
    return {name: part.assign(n=1, n_games=1, n_seasons=1) for name, part in views.items()}


def probability_example(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose both focal teams and every target's conditional probabilities without reduction."""
    records = []
    for (game, team), row in frame.iterrows():
        for target in SCORE_TARGETS:
            records.append(
                dict(
                    game_id=int(game),
                    team_id=int(team),
                    opponent_id=int(row.opponent_id),
                    target=target,
                    n=1,
                    n_games=1,
                    n_seasons=1,
                    N=int(row.N),
                    **{
                        suffix: row[f"{target}_{suffix}"]
                        for suffix in ("p_win", "p_draw", "p_loss", "lev_raw", "lev", "mc_se")
                    },
                )
            )
    return pd.DataFrame(records)


def representation_example(frame: pd.DataFrame, game_id: int) -> pd.DataFrame:
    """Show scalar home/away levels and all three component contrasts for the same match."""
    game = frame.loc[[game_id]]
    sid = int(game.season_id.iloc[0])
    home = game.loc[game.is_home].iloc[0]
    away = game.loc[~game.is_home].iloc[0]
    records = []
    for level in LEVELS:
        sides = game_contrast(game, level, "sides").iloc[0]
        diff = game_contrast(game, level).iloc[0]
        records.append(
            dict(
                representation=level,
                m_home=sides.m_home,
                m_away=sides.m_away,
                m_diff=diff.m_diff,
                n=2,
                n_games=1,
                n_seasons=1,
                season_id=sid,
            )
        )
    for target in SCORE_TARGETS:
        h = home[f"{target}_lev"]
        a = away[f"{target}_lev"]
        h = 0.0 if pd.isna(h) else float(h)
        a = 0.0 if pd.isna(a) else float(a)
        records.append(
            dict(
                representation=f"Comp / {target}",
                m_home=h,
                m_away=a,
                m_diff=h - a,
                n=2,
                n_games=1,
                n_seasons=1,
                season_id=sid,
            )
        )
    return pd.DataFrame(records)


def zero_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return full decomposition counts, stage counts, examples, and every unexplained MC row."""
    data = zero_partition(frame)
    summary = []
    stages = []
    examples = []
    for group in ZERO_GROUPS:
        part = data.loc[data.zero_group.eq(group)]
        summary.append(dict(zero_group=group, **_counts(part)))
        if not part.empty:
            examples.append(part.sort_values("as_of").head(1))
        for stage in STAGES:
            subset = part.loc[part.stage.eq(stage)]
            stages.append(dict(zero_group=group, stage=stage, **_counts(subset)))
    return dict(
        summary=pd.DataFrame(summary),
        stages=pd.DataFrame(stages),
        examples=pd.concat(examples).assign(n=1, n_games=1, n_seasons=1),
        other=data.loc[data.zero_group.eq("other_mc")].assign(n=1, n_games=1, n_seasons=1),
    )


def validation_plan(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the fifteen preregistered future folds; do not fit or score a model."""
    records = []
    for sid in VALIDATION_SEASONS:
        part = frame.loc[frame.season_id.eq(sid)]
        records.append(
            dict(
                fold=sid - 10,
                season_id=sid,
                train_first=8,
                train_last=sid - 1,
                train_seasons=sid - 8,
                deep=sid - 8 >= 4,
                **_counts(part),
            )
        )
    return pd.DataFrame(records)


def candidate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Enumerate the fixed twenty EXP-004 candidates, without training or selection."""
    records = []
    counts = _counts(validation_rows(frame))
    models = [
        ("C0", "none"),
        *[(level, form) for level in ("L2", "L3", "L5", "Lc") for form in ("diff", "sides")],
        ("Comp", "diff"),
    ]
    for level, form in models:
        for decay in ("none", "4 seasons"):
            records.append(
                dict(
                    candidate=f"{level}-{form}-{decay}",
                    representation=level,
                    form=form,
                    decay=decay,
                    **counts,
                )
            )
    assert len(records) == 20
    return pd.DataFrame(records)


def lottery_table(metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Read full path counts; these denominators repeat paths across focal scenarios."""
    records = []
    for item in metadata["season_audits"]:
        split = int(item.get("split_rank_paths", 0))
        final = int(item.get("final_rank_paths", 0))
        sl = int(item.get("split_lottery_paths", 0))
        fl = int(item.get("final_lottery_paths", 0))
        records.append(
            dict(
                season_id=int(item["season_id"]),
                n=int(item["rows"]),
                n_games=int(item["rows"]) // 2,
                n_seasons=1,
                split_rank_paths=split,
                split_lottery_paths=sl,
                split_rate=sl / split if split else np.nan,
                final_rank_paths=final,
                final_lottery_paths=fl,
                final_rate=fl / final,
            )
        )
    return pd.DataFrame(records)


def direct_component_impact(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover A4b's changed-row count from final components only; no old artifact is read."""
    data = frame.loc[frame.survival_direct_lev.gt(frame.leverage_max)]
    return pd.DataFrame(
        [dict(season_id=int(sid), **_counts(part)) for sid, part in data.groupby("season_id")]
    )


def headline_results(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Select declared examples for the short summary; do not choose categories by effect size."""
    chosen = [
        ("transparent", "releg_close", "contrast"),
        ("L2", "dead", "contrast"),
        ("L3", "0.5", "contrast"),
        ("L5", "1", "contrast"),
        ("Lc", "home_minus_away", "slope"),
        ("Comp", "survival", "contrast"),
    ]
    parts = []
    for rep, category, method in chosen:
        parts.append(
            comparisons.loc[
                comparisons.representation.eq(rep)
                & comparisons.category.eq(category)
                & comparisons.method.eq(method)
                & comparisons.metric.eq("res_pts")
            ]
        )
    return pd.concat(parts, ignore_index=True)


def cached_diagnostic(
    directory: Path,
    label: str,
    fingerprints: Mapping[str, str],
    compute: Callable[[], Mapping[str, Any]],
    *,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Cache a diagnostic only for matching inputs, relevant source bytes and parameters."""
    if not directory.resolve().is_relative_to(Path("data/reports/T-013/nb04_cache").resolve()):
        raise ValueError("Notebook diagnostics cache must stay in data/reports/T-013/nb04_cache")
    root = Path(__file__).resolve().parents[2]
    sources = (
        "ligat/analysis/motivation_review.py",
        "ligat/analysis/motivation_checks.py",
        "ligat/motivation/representations.py",
        "ligat/simulate/snapshot.py",
        "ligat/simulate/engine.py",
        "ligat/simulate/leverage.py",
        "ligat/simulate/rules.py",
        "ligat/features/elo.py",
        "ligat/features/table_state.py",
    )
    payload = dict(
        inputs=dict(fingerprints),
        parameters=dict(parameters),
        sources={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
    )
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = directory / f"{label}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return {**saved, "cache_hit": True}
    started = time.perf_counter()
    result = dict(compute())
    saved = dict(
        key=key,
        computed_at=datetime.now(UTC).isoformat(),
        seconds=time.perf_counter() - started,
        result=result,
        cache_hit=False,
        provenance=payload,
    )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return saved


def compare_tie_rules(
    state: Mapping[str, Any],
    teams: Sequence[int],
    old_rule: SeasonRules,
    new_rule: SeasonRules,
    lottery: Any,
) -> dict[str, int]:
    """Rank the identical terminal states under both canonical rules, without changing a score."""
    counts = []
    for rule in (old_rule, new_rule):
        engine = LeagueEngine(rule, list(teams))
        engine.rank(dict(state), lottery=lottery)
        counts.append(dict(engine.tie_counts))
    old, new = counts
    return dict(
        sampled_paths=old["final_rank_paths"],
        old_tied_paths=old["final_lottery_paths"],
        new_tied_paths=new["final_lottery_paths"],
        resolved_paths=old["final_lottery_paths"] - new["final_lottery_paths"],
        old_tied_pairs=old["final_lottery_pairs"],
        new_tied_pairs=new["final_lottery_pairs"],
        resolved_pairs=old["final_lottery_pairs"] - new["final_lottery_pairs"],
    )


class _TerminalTieAudit(LeagueEngine):
    """Observe canonical simulated terminal states; the simulator itself remains untouched."""

    def __init__(self, snapshot: SeasonSnapshot, new_rule: SeasonRules) -> None:
        super().__init__(snapshot.rules, list(snapshot.table.index), snapshot.previous_positions)
        self.new_rule = new_rule
        self.audit: dict[str, int] = {}

    def terminal(self, state: Any, groups: Any, lottery: Any) -> dict[str, Any]:
        targets = super().terminal(state, groups, lottery)
        counts = compare_tie_rules(state, self.teams.tolist(), self.rules, self.new_rule, lottery)
        for key, value in counts.items():
            self.audit[key] = self.audit.get(key, 0) + value
        return targets


def lottery_rule_diagnostic(review: MotivationReview, paths: int = 2000) -> dict[str, Any]:
    """Audit all focal scenarios on the first morning of each old-rule season, at fixed seed."""
    records = []
    for sid in (8, 9):
        snapshot = next(season_snapshots(review.participants, review.elo, review.rules, sid))
        new_rule = replace(snapshot.rules, tiebreak_rules=review.rules[10].tiebreak_rules)
        simulator = SeasonSimulator(
            snapshot, review.fits[sid], empirical_scores(review.participants, sid)
        )
        observer = _TerminalTieAudit(snapshot, new_rule)
        np.testing.assert_array_equal(observer.teams, simulator.engine.teams)
        simulator.engine = observer
        simulator.simulate(paths, review.seed)
        record = dict(season_id=sid, as_of=snapshot.as_of.isoformat(), N=paths, **observer.audit)
        if record["old_tied_pairs"] <= 0 or record["resolved_pairs"] <= 0:
            raise AssertionError(f"Old-rule lottery hypothesis not corroborated: {record}")
        record["resolved_fraction"] = record["resolved_pairs"] / record["old_tied_pairs"]
        records.append(record)
    return dict(passed=True, rows=records, N=paths, seed=review.seed)


SEASON_BLOCKS = {"B1": tuple(range(11, 16)), "B2": tuple(range(16, 21)), "B3": tuple(range(21, 26))}


def block_rows(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    """The three disjoint, fixed five-season development blocks; never fit across their boundary."""
    return frame.loc[frame.season_id.isin(SEASON_BLOCKS[block])].copy()


def alive_contrast(
    frame: pd.DataFrame, mask: pd.Series, metric: str, **kwargs: Any
) -> dict[str, Any]:
    """Exclude dead rows from both sides of an alive-only contrast."""
    alive = frame.loc[~frame.nothing_to_play]
    return within_strata_contrast(alive, mask.reindex(alive.index), metric, **kwargs)


def near_zero_dead_contrast(frame: pd.DataFrame, metric: str, **kwargs: Any) -> dict[str, Any]:
    """Compare only alive importance <=.01 against dead rows, excluding every other alive row."""
    near = level_l3(frame).eq(0.5)
    part = frame.loc[near | frame.nothing_to_play]
    return within_strata_contrast(part, near.reindex(part.index), metric, **kwargs)


def live_game_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a fixture only when both of its paired team rows are alive."""
    if "game_id" in frame.index.names:
        dead_game = frame.nothing_to_play.groupby(level="game_id").transform("any")
    else:
        dead_game = frame.nothing_to_play.groupby(frame.game_id).transform("any")
    return frame.loc[~dead_game].copy()


def game_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """One home row per game, with the four preregistered home-minus-away predictors."""
    home = frame.loc[frame.is_home].copy()
    for level in LEVELS:
        diff = game_contrast(frame, level).rename(columns={"m_diff": f"{level}_diff"})
        home = home.join(diff, on="game_id", validate="one_to_one")
    return home


def joint_strata_slopes(
    frame: pd.DataFrame,
    predictors: Sequence[str] = ("L2_diff", "Lc_diff"),
    metric: str = "res_pts",
    *,
    seed: int = 14112026,
    repeats: int = 2000,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """OLS without intercept after within-stratum centering; recenter every season bootstrap.

    A Moore-Penrose solution supports a zero flag column: its coefficient is zero
    but not identifiable (SE missing), while the identifiable continuous slope is retained.
    """
    data = frame.dropna(subset=[*predictors, metric])
    seasons = sorted(int(s) for s in frame.season_id.unique()) if years is None else list(years)
    strata = pd.Categorical(data.stratum)
    width, p = len(strata.categories), len(predictors)
    if data.empty or not p:
        raise ValueError("Joint slopes need observed predictors and a nonempty population")
    year = pd.Categorical(data.season_id, categories=seasons).codes.astype(np.int64)
    if (year < 0).any():
        raise ValueError("Bootstrap seasons do not cover the regression rows")
    index = year * width + strata.codes
    x, y = data[list(predictors)].to_numpy(float), data[metric].to_numpy(float)
    terms = [np.ones(len(data)), *x.T, y]
    terms += [x[:, i] * x[:, j] for i in range(p) for j in range(p)]
    terms += [x[:, i] * y for i in range(p)]
    moments = np.stack(
        [
            np.bincount(index, weights=t, minlength=len(seasons) * width).reshape(
                len(seasons), width
            )
            for t in terms
        ],
        axis=-1,
    )

    def centered(moment: Any) -> tuple[Any, Any]:
        n = moment[..., 0]
        sx, sy = moment[..., 1 : 1 + p], moment[..., 1 + p]
        xx = moment[..., 2 + p : 2 + p + p * p].reshape((*n.shape, p, p))
        xy = moment[..., 2 + p + p * p :]
        xx = xx - np.divide(
            sx[..., :, None] * sx[..., None, :],
            n[..., None, None],
            out=np.zeros_like(xx),
            where=n[..., None, None] > 0,
        )
        xy = xy - np.divide(
            sx * sy[..., None], n[..., None], out=np.zeros_like(xy), where=n[..., None] > 0
        )
        return xx.sum(axis=-3), xy.sum(axis=-2)

    xx, xy = centered(moments.sum(axis=0))
    coefficients = np.linalg.pinv(xx, rcond=1e-12) @ xy
    boot_xx, boot_xy = centered(
        np.einsum(
            "bs,skm->bkm", season_bootstrap_weights(seasons, repeats, seed).to_numpy(), moments
        )
    )
    draws = np.einsum("bij,bj->bi", np.linalg.pinv(boot_xx, rcond=1e-12), boot_xy)
    correlation = (
        float(xx[0, 1] / np.sqrt(xx[0, 0] * xx[1, 1]))
        if p == 2 and np.all(np.diag(xx) > 1e-12)
        else np.nan
    )
    records = []
    for i, predictor in enumerate(predictors):
        identifiable = bool(xx[i, i] > 1e-12)
        valid = boot_xx[:, i, i] > 1e-12
        records.append(
            dict(
                predictor=predictor,
                estimate=float(coefficients[i]),
                se=_finite_std(draws[valid, i]) if identifiable and len(seasons) > 1 else np.nan,
                centered_corr=correlation,
                identifiable=identifiable,
                **_counts(data),
                n_used=len(data),
                n_excluded=0,
            )
        )
    return pd.DataFrame(records)


def typical_difference(
    frame: pd.DataFrame, predictor: str, estimate: float, se: float
) -> dict[str, Any]:
    """Median/P90 among nonzero absolute differences in this regression's population, in points."""
    nonzero = frame[predictor].dropna().abs()
    nonzero = nonzero.loc[nonzero.ne(0)]
    median = float(nonzero.quantile(0.5)) if len(nonzero) else np.nan
    p90 = float(nonzero.quantile(0.9)) if len(nonzero) else np.nan
    return dict(
        n_nonzero=len(nonzero),
        median_diff=median,
        p90_diff=p90,
        median_effect=estimate * median,
        p90_effect=estimate * p90,
        median_effect_se=se * median,
        p90_effect_se=se * p90,
    )


def _conditional_specs(frame: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame, pd.Series]]:
    alive = frame.loc[~frame.nothing_to_play]
    specs = [
        ("dead", "L2", "dead", frame, frame.nothing_to_play),
        (
            "dead_vs_alive_opponent",
            "L2",
            "dead_vs_alive_opponent",
            frame,
            frame.nothing_to_play & ~frame.opp_dead,
        ),
        (
            "alive_vs_dead_opponent",
            "L2",
            "alive_vs_dead_opponent",
            frame,
            ~frame.nothing_to_play & frame.opp_dead,
        ),
    ]
    # L3=.5 and L5=.25 identify exactly the same rows and comparison universe.
    # Estimate once and show the explicit alias instead of doubling the evidence.
    for value in (0.25, 0.5, 0.75, 1.0):
        specs.append(
            (
                f"alive_L5_{value:g}",
                "L3/L5" if value == 0.25 else "L5",
                "near_zero_alive" if value == 0.25 else f"{value:g}",
                alive,
                level_l5(alive).eq(value),
            )
        )
    near = level_l3(frame).eq(0.5)
    near_dead = frame.loc[near | frame.nothing_to_play]
    specs.append(
        ("near_zero_vs_dead", "L3", "near_zero_vs_dead", near_dead, near.reindex(near_dead.index))
    )
    leading = alive[[f"{t}_lev" for t in SCORE_TARGETS]].fillna(-np.inf).idxmax(axis=1)
    for target in SCORE_TARGETS:
        specs.append(
            (
                f"alive_Comp_{target}",
                "Comp",
                target,
                alive,
                leading.eq(f"{target}_lev") & alive[f"{target}_lev"].gt(0.2),
            )
        )
    for target in ("title_close", "cut_close", "releg_close"):
        specs.append((f"alive_{target}", "transparent", target, alive, alive[target]))
    return specs


def _conditional_contrasts(
    frame: pd.DataFrame, years: Sequence[int], seed: int, repeats: int
) -> pd.DataFrame:
    rows = []
    for key, representation, category, universe, mask in _conditional_specs(frame):
        for metric in METRICS:
            result = within_strata_contrast(
                universe, mask, metric, years=years, seed=seed, repeats=repeats
            )
            rows.append(
                dict(
                    comparison_id=key,
                    representation=representation,
                    category=category,
                    metric=metric,
                    method="contrast",
                    **result,
                )
            )
    return pd.DataFrame(rows)


def _conditional_slopes(
    frame: pd.DataFrame, years: Sequence[int], seed: int, repeats: int
) -> pd.DataFrame:
    games = game_predictors(frame)
    alive_games = game_predictors(live_game_rows(frame))
    rows = []
    for level in LEVELS:
        predictor = f"{level}_diff"
        result = within_strata_slope(games, predictor, seed=seed, repeats=repeats, years=years)
        rows.append(
            dict(
                comparison_id=f"all_{level}",
                representation=level,
                category="all_games",
                metric="res_pts",
                method="slope",
                predictor=predictor,
                **result,
                **typical_difference(games, predictor, result["estimate"], result["se"]),
            )
        )
    result = within_strata_slope(alive_games, "Lc_diff", seed=seed, repeats=repeats, years=years)
    rows.append(
        dict(
            comparison_id="live_Lc",
            representation="Lc",
            category="live_games",
            metric="res_pts",
            method="slope",
            predictor="Lc_diff",
            **result,
            **typical_difference(alive_games, "Lc_diff", result["estimate"], result["se"]),
        )
    )
    joint = joint_strata_slopes(games, years=years, seed=seed, repeats=repeats)
    for record in joint.to_dict("records"):
        predictor = record["predictor"]
        rows.append(
            dict(
                comparison_id=f"joint_{predictor[:-5]}",
                representation=predictor[:-5],
                category="joint",
                method="joint",
                metric="res_pts",
                **record,
                **typical_difference(games, predictor, record["estimate"], record["se"]),
            )
        )
    return pd.DataFrame(rows)


def conditional_review(
    frame: pd.DataFrame, *, seed: int, repeats: int = 2000
) -> dict[str, pd.DataFrame]:
    """The A5b preregistered contrasts, conditional slopes and five-season stability blocks."""
    data = validation_rows(frame)
    all_contrasts = _conditional_contrasts(data, VALIDATION_SEASONS, seed, repeats)
    slopes = _conditional_slopes(data, VALIDATION_SEASONS, seed, repeats)
    recent = validation_rows(data, last_five=True)
    recent_all = pd.concat(
        [
            _conditional_contrasts(recent, SEASON_BLOCKS["B3"], seed, repeats),
            _conditional_slopes(recent, SEASON_BLOCKS["B3"], seed, repeats),
        ],
        ignore_index=True,
    )
    overall = pd.concat([all_contrasts, slopes], ignore_index=True)
    keys = ["comparison_id", "metric"]
    stability = overall.merge(
        recent_all[keys + ["estimate", "se", "n", "n_used", "n_games", "n_seasons"]].rename(
            columns={
                c: f"last5_{c}" for c in ("estimate", "se", "n", "n_used", "n_games", "n_seasons")
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    block_tables = []
    for block, years in SEASON_BLOCKS.items():
        part = block_rows(data, block)
        contrasts = _conditional_contrasts(part, years, seed, repeats)
        wanted = (
            (
                contrasts.comparison_id.isin(["dead", "dead_vs_alive_opponent"])
                & contrasts.metric.isin(["res_pts", "res_ga"])
            )
            | contrasts.comparison_id.str.startswith("alive_L5_")
            | contrasts.comparison_id.isin(
                ["near_zero_vs_dead", "alive_releg_close", "alive_Comp_survival"]
            )
        )
        regressions = _conditional_slopes(part, years, seed, repeats)
        regressions = regressions.loc[
            regressions.comparison_id.isin(["all_Lc", "live_Lc", "joint_L2", "joint_Lc"])
        ]
        block_tables.append(
            pd.concat([contrasts.loc[wanted], regressions], ignore_index=True).assign(
                block=block, block_first=min(years), block_last=max(years)
            )
        )
    blocks = pd.concat(block_tables, ignore_index=True)
    signs = []
    for row in overall.itertuples():
        group = blocks.loc[
            blocks.comparison_id.eq(row.comparison_id) & blocks.metric.eq(row.metric)
        ]
        eligible = group.loc[group.n_used.ge(30) & group.estimate.notna()]
        signs.append(
            dict(
                comparison_id=row.comparison_id,
                metric=row.metric,
                blocks_same=int(np.sign(eligible.estimate).eq(np.sign(row.estimate)).sum()),
                blocks_eligible=len(eligible),
                blocks_small=int(group.n_used.lt(30).sum()),
            )
        )
    stability = stability.merge(pd.DataFrame(signs), on=keys, validate="one_to_one")
    yearly = []
    for sid, part in data.groupby("season_id", sort=True):
        games = game_predictors(part)
        yearly.append(
            dict(
                representation="Lc",
                category="home_minus_away",
                metric="res_pts",
                season_id=int(sid),
                **within_strata_slope(games, "Lc_diff", years=[int(sid)], repeats=2, seed=seed),
            )
        )
    flags = all_contrasts.loc[all_contrasts.representation.eq("L2")].copy()
    alive = all_contrasts.loc[~all_contrasts.representation.eq("L2")].copy()
    for table in (flags, alive):
        if table.duplicated(["estimate", "n", "metric"]).any():
            raise AssertionError("Duplicate contrast in chapter 4")
    eligible = overall.loc[overall.se.notna() & overall.se.gt(0) & overall.n_used.ge(30)]
    multiplicity = pd.DataFrame(
        [
            dict(
                comparisons=len(overall),
                estimable=len(eligible),
                beyond_two_se=int(eligible.estimate.abs().gt(2 * eligible.se).sum()),
            )
        ]
    )
    return dict(
        flags=flags,
        alive=alive,
        slopes=slopes,
        stability=stability,
        blocks=blocks,
        annual=pd.DataFrame(yearly),
        summary=conditional_summary(stability),
        multiplicity=multiplicity,
    )


def conditional_summary(stability: pd.DataFrame) -> pd.DataFrame:
    """One declared example per representation; a descriptive rule, not EXP-004 selection."""
    chosen = [
        ("L2", "dead"),
        ("L3", "alive_L5_0.25"),
        ("L5", "alive_L5_1"),
        ("Lc", "joint_Lc"),
        ("Comp", "alive_Comp_survival"),
        ("transparent", "alive_releg_close"),
    ]
    rows = []
    for representation, key in chosen:
        row = (
            stability.loc[stability.comparison_id.eq(key) & stability.metric.eq("res_pts")]
            .iloc[0]
            .to_dict()
        )
        value, se = (
            (row["median_effect"], row["median_effect_se"])
            if key == "joint_Lc"
            else (row["estimate"], row["se"])
        )
        beyond = "yes" if abs(value) > 2 * se else "unclear" if abs(value) > se else "no"
        rows.append(
            dict(
                row,
                representation=representation,
                central_effect=value,
                central_se=se,
                beyond_flag="no" if representation == "L2" else beyond,
            )
        )
    return pd.DataFrame(rows)


def extended_summary(
    frame: pd.DataFrame, evidence: Mapping[str, pd.DataFrame], *, seed: int, repeats: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add defensive evidence using the same descriptive SE rule, including joint Lc."""
    games = game_predictors(validation_rows(frame))
    joint_ga = joint_strata_slopes(
        games, metric="res_ga", seed=seed, repeats=repeats, years=VALIDATION_SEASONS
    )
    result = evidence["summary"].copy()
    for idx, row in result.iterrows():
        if row.representation == "L2":
            flag = (
                evidence["stability"]
                .loc[
                    evidence["stability"].comparison_id.eq("dead_vs_alive_opponent")
                    & evidence["stability"].metric.eq("res_pts")
                ]
                .iloc[0]
            )
            for field in ("central_effect", "central_se", "n", "n_used", "n_games", "n_seasons"):
                source = {"central_effect": "estimate", "central_se": "se"}.get(field, field)
                result.loc[idx, field] = flag[source]
            for field in (
                "comparison_id",
                "category",
                "estimate",
                "se",
                "n_excluded",
                "last5_estimate",
                "last5_se",
                "last5_n",
                "last5_n_used",
                "last5_n_games",
                "last5_n_seasons",
                "blocks_same",
                "blocks_eligible",
                "blocks_small",
            ):
                result.loc[idx, field] = flag[field]
            result.loc[idx, "beyond_flag"] = "baseline"
            result.loc[idx, "beyond_flag_ga"] = "baseline"
            continue
        if row.representation == "Lc":
            ga = joint_ga.loc[joint_ga.predictor.eq("Lc_diff")].iloc[0]
            scale = typical_difference(games, "Lc_diff", ga.estimate, ga.se)
            value, se = scale["median_effect"], scale["median_effect_se"]
        else:
            ga = (
                evidence["stability"]
                .loc[
                    evidence["stability"].comparison_id.eq(row.comparison_id)
                    & evidence["stability"].metric.eq("res_ga")
                ]
                .iloc[0]
            )
            value, se = ga.estimate, ga.se
        result.loc[idx, "ga_effect"] = value
        result.loc[idx, "ga_se"] = se
        result.loc[idx, "beyond_flag_ga"] = (
            "yes" if abs(value) > 2 * se else "unclear" if abs(value) > se else "no"
        )
    return result, joint_ga


def expanded_multiplicity(
    evidence: Mapping[str, pd.DataFrame],
    decomposition: Mapping[str, pd.DataFrame],
    closed: Mapping[str, pd.DataFrame],
    joint_ga: pd.DataFrame,
    stage_review: pd.DataFrame | None = None,
    period_review: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Count central all-15 estimates once; exclude block/year repeats and rescalings."""
    tables = [
        evidence["stability"][["estimate", "se", "n_used"]],
        decomposition["sensitivity"].loc[
            lambda x: x.scope.eq("all15"), ["estimate", "se", "n_used"]
        ],
        closed["adjusted"][["estimate", "se", "n_used"]],
        joint_ga[["estimate", "se", "n_used"]],
    ]
    if "adjusted_other" in closed:
        tables.append(closed["adjusted_other"][["estimate", "se", "n_used"]])
    extra = []
    for row in decomposition["original"].loc[lambda x: x.scope.eq("all15")].itertuples():
        for name in ("dead_mean", "control_mean"):
            extra.append(
                dict(estimate=getattr(row, name), se=getattr(row, f"{name}_se"), n_used=row.n_used)
            )
    for row in closed["raw"].itertuples():
        for metric in ("res_total_goals", "res_draw"):
            extra.append(
                dict(estimate=getattr(row, metric), se=getattr(row, f"{metric}_se"), n_used=row.n)
            )
    if stage_review is not None:
        for row in stage_review.itertuples():
            for metric in ("goals", "draw"):
                extra.append(
                    dict(
                        estimate=getattr(row, f"{metric}_residual"),
                        se=getattr(row, f"{metric}_se"),
                        n_used=row.n_games,
                    )
                )
    if period_review is not None:
        for row in period_review.itertuples():
            for metric in ("res_total_goals", "res_draw"):
                extra.append(
                    dict(
                        estimate=getattr(row, metric),
                        se=getattr(row, f"{metric}_se"),
                        n_used=row.n_games,
                    )
                )
    central = pd.concat([*tables, pd.DataFrame(extra)], ignore_index=True)
    eligible = central.loc[central.se.notna() & central.se.gt(0) & central.n_used.ge(30)]
    return pd.DataFrame(
        [
            dict(
                comparisons=len(central),
                estimated=int(central.estimate.notna().sum()),
                estimable=len(eligible),
                beyond_two_se=int(eligible.estimate.abs().gt(2 * eligible.se).sum()),
            )
        ]
    )


def select_mc_example(frame: pd.DataFrame) -> tuple[int, int]:
    """Select only from prematch quantities; outcomes and residuals cannot affect selection."""
    fields = [
        "season_id",
        "as_of",
        "leverage_max",
        "survival_mc_se",
        *[f"{t}_lev" for t in SCORE_TARGETS],
        *[f"survival_p_{o}" for o in ("win", "draw", "loss")],
    ]
    data = frame[fields].copy()
    leading = data[[f"{t}_lev" for t in SCORE_TARGETS]].fillna(-np.inf).idxmax(axis=1)
    valid = data.season_id.between(11, 25) & data.leverage_max.between(0.25, 0.35)
    valid &= leading.eq("survival_lev") & data.survival_mc_se.gt(0)
    for outcome in ("win", "draw", "loss"):
        valid &= data[f"survival_p_{outcome}"].between(0.02, 0.98)
    choices = data.loc[valid].reset_index().assign(distance=lambda d: (d.leverage_max - 0.30).abs())
    if choices.empty:
        raise ValueError("No interior Monte Carlo example satisfies the fixed selection rule")
    row = choices.sort_values(["distance", "as_of", "team_id", "game_id"]).iloc[0]
    return int(row.game_id), int(row.team_id)


def second_live_example(review: MotivationReview) -> dict[str, Any]:
    """Reproduce the selected nondegenerate morning at the artifact's exact N and seed."""
    game, team = select_mc_example(review.frame)
    row = review.frame.loc[(game, team)]
    sid = int(row.season_id)
    snapshot = snapshot_as_of(review.participants, review.elo, review.rules, sid, row.as_of)
    started = time.perf_counter()
    features = (
        SeasonSimulator(snapshot, review.fits[sid], empirical_scores(review.participants, sid))
        .simulate(2000, review.seed)
        .features
    )
    pd.testing.assert_frame_equal(
        features, review.motivation.reindex(features.index)[features.columns], check_exact=True
    )
    return dict(
        game_id=game,
        team_id=team,
        season_id=sid,
        snapshot=snapshot,
        features=features.loc[[game]],
        artifact_error=0.0,
        seconds=time.perf_counter() - started,
        n=len(features),
        n_games=features.index.get_level_values("game_id").nunique(),
        n_seasons=1,
    )


def probability_intervals(features: pd.DataFrame) -> pd.DataFrame:
    """Unclipped ±2 paired-MC-SE around raw leverage; excludes model uncertainty."""
    result = probability_example(features)
    result["mc_lower"] = result.lev_raw - 2 * result.mc_se
    result["mc_upper"] = result.lev_raw + 2 * result.mc_se
    return result


def past_elo_endpoints(
    participants: pd.DataFrame, prematch: pd.DataFrame, as_of: pd.Timestamp, season_id: int
) -> pd.Series:
    """Replay M0 only on strict past fixtures; no next appearance or future result is consulted."""
    from ligat.analysis.prepare import AnalysisTables
    from ligat.features.elo import EloParams, replay_elo_season

    past = participants.loc[
        participants.season_id.eq(season_id) & participants.as_of.lt(as_of)
    ].copy()
    if past.empty:
        return pd.Series(dtype=float, name="elo_pre")
    keys = pd.MultiIndex.from_frame(past[["game_id", "team_id"]])
    known = prematch.reindex(keys).reset_index().sort_values(["as_of", "game_id"])
    assert known.as_of.lt(as_of).all()
    initial = known.drop_duplicates("team_id").set_index("team_id").elo_pre
    tables = AnalysisTables(
        pd.DataFrame(),
        past.set_index(["team_id", "game_id"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    flags = prematch.nothing_to_play.reindex(keys)
    flags.index.names = ["game_id", "team_id"]
    replay = replay_elo_season(
        tables,
        flags,
        season_id=season_id,
        elo_start=initial,
        params=EloParams(k=10, home_advantage=30, promoted_initialization="fixed_1450"),
    )
    return replay.season_snapshots.xs(season_id).elo_end.rename("elo_pre")


def morning_example(review: MotivationReview, snapshot: SeasonSnapshot) -> dict[str, pd.DataFrame]:
    """Presentation views of a snapshot, including past-only end ratings and two upcoming rounds."""
    roster = snapshot.table.copy()
    roster["elo_pre"] = snapshot.elo
    home = snapshot.remaining.home_team_id.value_counts()
    away = snapshot.remaining.away_team_id.value_counts()
    remaining = home.add(away, fill_value=0).reindex(roster.index).fillna(0).astype(int)
    roster["remaining_fixtures"] = remaining
    roster["finished"] = remaining.eq(0)
    missing = roster.elo_pre.isna() & roster.finished
    if missing.any():
        ends = past_elo_endpoints(
            review.participants, review.prematch, snapshot.as_of, snapshot.season_id
        )
        roster.loc[missing, "elo_pre"] = ends.reindex(roster.index[missing])
    # Snapshot keeps only the simulation contract. Round is a display field from
    # the already-visible fixture identities, without joining scores or hidden playoffs.
    rounds = review.participants[["game_id", "round"]].drop_duplicates()
    fixtures = snapshot.remaining.merge(rounds, on="game_id", how="left", validate="one_to_one")
    next_rounds = (
        fixtures[["stage", "round"]].drop_duplicates().groupby("stage", sort=False).head(2)
    )
    shown = fixtures.merge(next_rounds, on=["stage", "round"], how="inner", validate="many_to_one")
    remainder = fixtures.loc[~fixtures.game_id.isin(shown.game_id)]
    return dict(
        table=roster.sort_values("position"),
        schedule=shown,
        other_schedule=pd.DataFrame([dict(other_games=len(remainder))]),
    )


def verify_conditional_pilot(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the pilot's alive-only, near/dead and alive-team universes."""
    data = validation_rows(frame)
    alive = data.loc[~data.nothing_to_play]
    target = reference.loc[
        (
            reference.family.eq("alive_only_importance")
            & reference.method.eq("within_strata_vs_rest")
        )
        | (reference.family.eq("near_zero_vs_dead") & reference.method.eq("within_strata_vs_dead"))
        | (
            reference.family.eq("alive_importance_difference")
            & reference.method.eq("within_strata_slope")
        )
    ]
    rows = []
    for row in target.itertuples():
        if row.family == "near_zero_vs_dead":
            result = near_zero_dead_contrast(data, row.metric, repeats=2)
        elif row.family == "alive_importance_difference":
            pairs = data.leverage_max
            opponent_keys = pd.MultiIndex.from_arrays(
                [alive.index.get_level_values("game_id"), alive.opponent_id],
                names=alive.index.names,
            )
            part = alive.assign(
                importance_diff=alive.leverage_max.to_numpy()
                - pairs.reindex(opponent_keys).to_numpy()
            )
            result = within_strata_slope(part, "importance_diff", metric=row.metric, repeats=2)
        else:
            bins = {
                "up_to_001": alive.leverage_max.le(0.01),
                "001_to_005": alive.leverage_max.gt(0.01) & alive.leverage_max.le(0.05),
                "005_to_02": alive.leverage_max.gt(0.05) & alive.leverage_max.le(0.2),
                "above_02": alive.leverage_max.gt(0.2),
            }
            result = within_strata_contrast(alive, bins[row.category], row.metric, repeats=2)
        difference = abs(result["estimate"] - row.estimate)
        if not np.isfinite(difference) or difference > 1e-9:
            raise AssertionError(
                f"Pilot mismatch {row.family}/{row.category}/{row.metric}: "
                f"current={result['estimate']}; pilot={row.estimate}; difference={difference}"
            )
        rows.append(
            dict(
                family=row.family,
                category=row.category,
                metric=row.metric,
                current=result["estimate"],
                reference=row.estimate,
                max_error=difference,
                **{k: result[k] for k in ("n", "n_used", "n_excluded", "n_games", "n_seasons")},
            )
        )
    return pd.DataFrame(rows)
