"""Frozen-Elo, paired competitive leverage; all inputs are already loaded by callers."""

import ast
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ligat.models.ordered_logit import OrderedFit, ordered_probabilities
from ligat.simulate.engine import Array, LeagueEngine
from ligat.simulate.snapshot import DEVELOPMENT_SEASONS, SeasonSnapshot

TARGETS = ("title", "cut", "survival", "survival_direct")
# Decision H-M3: the score uses these three objectives; direct survival is diagnostic.
SCORE_TARGETS = ("title", "cut", "survival")
E0050 = "elo-b24989debf11c0ba4af09aa247bd264ad754349b8d0db4ba88e9c440fd00e950"
SIMULATOR_VERSION = "1"
DEFAULT_SEED = 14112026


def conversion_fits(fold_scores: pd.DataFrame) -> dict[int, OrderedFit]:
    """Select frozen E0050 folds; warm-up seasons 8-10 share the season-11 fit."""
    selected = fold_scores.loc[fold_scores.config_id.eq(E0050)]
    if set(selected.season_id) != set(range(11, 26)) or selected.season_id.duplicated().any():
        raise ValueError("Expected exactly the fifteen E0050 development folds")
    fits = {}
    for row in selected.itertuples():
        if list(ast.literal_eval(row.training_seasons)) != list(range(8, int(row.season_id))):
            raise ValueError("Conversion training seasons are not strictly historical")
        fit = OrderedFit(
            float(row.beta),
            float(row.theta1),
            float(row.delta),
            float(row.log_likelihood),
            bool(row.converged),
            int(row.iterations),
            bool(row.accepted_by_gradient_check),
            float(row.max_abs_gradient),
        )
        if abs(fit.theta2 - float(row.theta2)) > 1e-12:
            raise ValueError("Stored ordered thresholds disagree")
        fits[int(row.season_id)] = fit
    for sid in (8, 9, 10):
        fits[sid] = fits[11]
    return fits


def empirical_scores(rows: pd.DataFrame, season_id: int) -> dict[int, Array]:
    """Use earlier seasons only, conditional on H/D/A; warm-up is training-only.

    For seasons 8-10 the pool deliberately includes all of 8-10, just like their
    conversion fit. These in-sample rows are only for training EXP-004, never for
    validation. No validation-season or future score enters a later season's pool.
    """
    if season_id not in DEVELOPMENT_SEASONS or not set(rows.season_id) <= DEVELOPMENT_SEASONS:
        raise ValueError("Empirical inputs require development seasons 8-25")
    train = rows.loc[rows.season_id.le(10) if season_id <= 10 else rows.season_id.lt(season_id)]
    train = train.loc[train.is_home.eq(True)].sort_values(["season_id", "date", "game_id"])
    gf, ga = train.goals_scored, train.goals_conceded
    pools = {
        i: train.loc[mask, ["goals_scored", "goals_conceded"]].to_numpy(dtype=np.int16)
        for i, mask in enumerate((gf.gt(ga), gf.eq(ga), gf.lt(ga)))
    }
    if any(not len(pool) for pool in pools.values()):
        raise ValueError("Every outcome needs an empirical exact-score pool")
    return pools


@dataclass(frozen=True)
class LeverageResult:
    """Paired target estimates, the three-objective score and auditable rank counters."""

    features: pd.DataFrame
    tie_counts: Mapping[str, int]


class SeasonSimulator:
    """Simulate only a snapshot's known schedule, with frozen ratings and shared draws."""

    def __init__(
        self, snapshot: SeasonSnapshot, fit: OrderedFit, score_pools: Mapping[int, Array]
    ) -> None:
        if snapshot.season_id not in DEVELOPMENT_SEASONS:
            raise ValueError("Simulator requires a development season 8-25")
        self.snapshot = snapshot
        self.engine = LeagueEngine(
            snapshot.rules, list(snapshot.table.index), snapshot.previous_positions
        )
        self.fit = fit
        self.pools = {k: np.asarray(v, dtype=np.int16).copy() for k, v in score_pools.items()}
        if set(self.pools) != {0, 1, 2} or any(
            p.ndim != 2 or p.shape[1] != 2 or not len(p) for p in self.pools.values()
        ):
            raise ValueError("Expected nonempty home/draw/away exact-score pairs")
        for outcome, pool in self.pools.items():
            if (pool < 0).any() or not (
                np.sign(pool[:, 0] - pool[:, 1]) == (1, 0, -1)[outcome]
            ).all():
                raise ValueError("Empirical scores must match their outcome")
        ratings = snapshot.elo.reindex(self.engine.teams).to_numpy(float)
        # Frozen ratings permit one canonical probability computation per pair/day.
        valid = np.isfinite(ratings)
        h, a = np.nonzero(valid[:, None] & valid[None, :])
        self.pair_probabilities = np.full((self.engine.n, self.engine.n, 3), np.nan)
        if len(h):
            self.pair_probabilities[h, a] = ordered_probabilities(
                fit, (ratings[h] - ratings[a]) / 400
            )

    def exact_scores(self, outcomes: Array, uniforms: Array) -> tuple[Array, Array]:
        """Sample a score uniformly from the training rows of each chosen outcome."""
        hg = np.zeros(outcomes.shape, dtype=np.int16)
        ag = hg.copy()
        for outcome in (0, 1, 2):
            mask = outcomes == outcome
            pool = self.pools[outcome]
            picks = np.minimum((uniforms[mask] * len(pool)).astype(int), len(pool) - 1)
            hg[mask], ag[mask] = pool[picks, 0], pool[picks, 1]
        return hg, ag

    def probabilities(self, home: int | Array, away: int | Array) -> Array:
        """Return canonical H/D/A probabilities with no extra home-rating offset."""
        result = np.asarray(self.pair_probabilities[home, away])
        if not np.isfinite(result).all():
            raise ValueError("A remaining participant has no finite morning Elo")
        return result

    def _scores(
        self, home: int | Array, away: int | Array, outcome_u: Array, score_u: Array
    ) -> tuple[Array, Array]:
        probabilities = self.probabilities(home, away)
        ph, pd_ = probabilities[..., 0], probabilities[..., 1]
        outcomes = np.where(outcome_u < ph, 0, np.where(outcome_u < ph + pd_, 1, 2))
        return self.exact_scores(outcomes, score_u)

    def playoff_slots(self) -> list[tuple[int, int]]:
        """Keep the pilot's rank-pair slot order; higher rank hosts a single leg."""
        slots = []
        offset = 0
        for group in self.snapshot.rules.groups:
            legs = group.rounds // (group.teams - 1)
            if legs not in (1, 2) or group.rounds % (group.teams - 1):
                raise ValueError("Unsupported playoff legs")
            for leg in range(legs):
                for i in range(group.teams):
                    for j in range(i + 1, group.teams):
                        slots.append(
                            (
                                offset + (i if leg % 2 == 0 else j),
                                offset + (j if leg % 2 == 0 else i),
                            )
                        )
            offset += group.teams
        return slots

    def simulate(self, paths: int = 2000, seed: int = DEFAULT_SEED) -> LeverageResult:
        """Force focal H/D/A with CRN; score title/cut/survival, retaining diagnostics.

        The score is the maximum nonnegative effect among available SCORE_TARGETS.
        Direct survival remains a component and never enters that maximum.
        """
        if paths < 2 or seed < 0:
            raise ValueError("At least two paths and a nonnegative seed are required")
        snapshot, engine = self.snapshot, self.engine
        engine.tie_counts.clear()
        entropy = [seed, snapshot.season_id, int(snapshot.as_of.timestamp()) & 0xFFFFFFFF]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        lottery = np.random.default_rng(np.random.SeedSequence([*entropy, 621])).random(
            (paths, engine.n)
        )
        future = snapshot.remaining.sort_values(["as_of", "date", "game_id"])
        current = future.loc[future.as_of.eq(snapshot.as_of)]
        if current.empty:
            raise ValueError("No visible focal fixture on this morning")
        before_split = bool(snapshot.rules.groups) and snapshot.phase == "regular"
        state = engine.from_played(snapshot.played, paths)
        uout, uscore = rng.random((len(future), paths)), rng.random((len(future), paths))
        drawn: dict[int, tuple[Array, Array, Array]] = {}
        for slot, row in enumerate(future.itertuples()):
            home, away = engine.lookup[int(row.home_team_id)], engine.lookup[int(row.away_team_id)]
            hg, ag = self._scores(home, away, uout[slot], uscore[slot])
            drawn[int(row.game_id)] = (hg, ag, uscore[slot])
            engine.add(state, home, away, hg, ag, regular=row.stage == "regular")
        slots = self.playoff_slots()
        puout, puscore = rng.random((len(slots), paths)), rng.random((len(slots), paths))
        known_groups = np.zeros((paths, engine.n), dtype=np.int16)
        if snapshot.groups is not None:
            known_groups[:] = [
                next(
                    i
                    for i, g in enumerate(snapshot.rules.groups)
                    if g.name == snapshot.groups[int(t)]
                )
                for t in engine.teams
            ]
        records = []
        for row in current.itertuples():
            home, away = engine.lookup[int(row.home_team_id)], engine.lookup[int(row.away_team_id)]
            oldh, olda, force_u = drawn[int(row.game_id)]
            scenarios = []
            for outcome in (0, 1, 2):
                altered = engine.clone(state)
                engine.add(altered, home, away, oldh, olda, regular=row.stage == "regular", sign=-1)
                forcedh, forceda = self.exact_scores(np.full(paths, outcome), force_u)
                engine.add(altered, home, away, forcedh, forceda, regular=row.stage == "regular")
                if before_split:
                    order, _ = engine.rank(altered, at_split=True, lottery=lottery)
                    group_ids = engine.membership(order)
                    for slot, (hrank, arank) in enumerate(slots):
                        hi, ai = order[:, hrank], order[:, arank]
                        hg, ag = self._scores(hi, ai, puout[slot], puscore[slot])
                        engine.add(altered, hi, ai, hg, ag)
                else:
                    group_ids = known_groups
                scenarios.append(engine.terminal(altered, group_ids, lottery))
            for index, team, opponent, is_home in (
                (home, row.home_team_id, row.away_team_id, True),
                (away, row.away_team_id, row.home_team_id, False),
            ):
                win, draw, loss = (0, 1, 2) if is_home else (2, 1, 0)
                record = dict(
                    game_id=int(row.game_id),
                    team_id=int(team),
                    opponent_id=int(opponent),
                    season_id=snapshot.season_id,
                    as_of=snapshot.as_of,
                    is_home=is_home,
                    N=paths,
                    before_split=before_split,
                )
                for target in TARGETS:
                    available = not (
                        (target == "cut" and not before_split)
                        or (target == "survival_direct" and snapshot.season_id not in (11, 12))
                    )
                    if available:
                        arrays = [
                            scenarios[k][target][:, index].astype(float) for k in (win, draw, loss)
                        ]
                        pw, pd_, pl = (float(x.mean()) for x in arrays)
                        raw = pw - pl
                        se = float((arrays[0] - arrays[2]).std(ddof=1) / np.sqrt(paths))
                    else:
                        pw = pd_ = pl = raw = se = np.nan
                    record.update(
                        {
                            f"{target}_p_win": pw,
                            f"{target}_p_draw": pd_,
                            f"{target}_p_loss": pl,
                            f"{target}_lev_raw": raw,
                            f"{target}_lev": max(raw, 0) if available else np.nan,
                            f"{target}_mc_se": se,
                        }
                    )
                scores = [
                    float(record[f"{t}_lev"])
                    for t in SCORE_TARGETS
                    if np.isfinite(record[f"{t}_lev"])
                ]
                if not scores:
                    raise ValueError("No available scoring objective for a focal participant")
                record["leverage_max"] = max(scores)
                records.append(record)
        features = pd.DataFrame(records).set_index(["game_id", "team_id"]).sort_index()
        if not features.index.is_unique:
            raise AssertionError("Duplicate focal team-game")
        return LeverageResult(features, dict(engine.tie_counts))
