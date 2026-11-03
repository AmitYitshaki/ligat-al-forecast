"""Result-only Elo with calendar-day isolation and reproducible season snapshots.

Elo is a deterministic rating formula. Hyperparameter selection and conversion to
home/draw/away probabilities belong to later work, not to this engine.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Literal, TypeAlias

import pandas as pd  # type: ignore[import-untyped]

from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables
from ligat.features.registry import validate_feature_columns
from ligat.features.table_state import game_rows

# Fixed definitions from T-010 §8.1 and ה-ת11, not calibration parameters.
BASE_RATING = 1500.0
RATING_SCALE = 400.0
PROMOTED_FALLBACK = 1450.0
MIN_PREVIOUS_PROMOTED = 4
PromotedInitialization = Literal["previous_promoted_mean", "fixed_1450"]


@dataclass(frozen=True)
class EloParams:
    """One explicit Elo configuration; defaults are the specified sanity example."""

    k: float = 20.0
    home_advantage: float = 50.0
    season_regression: float = 0.33
    use_goal_multiplier: bool = True
    promoted_initialization: PromotedInitialization = "previous_promoted_mean"
    dead_game_discount: float = 1.0

    def __post_init__(self) -> None:
        for name in ("k", "home_advantage", "season_regression", "dead_game_discount"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.k < 0:
            raise ValueError("k must be nonnegative")
        if not 0 <= self.season_regression <= 1 or not 0 <= self.dead_game_discount <= 1:
            raise ValueError("Regression and discount must be in [0, 1]")
        if not isinstance(self.use_goal_multiplier, bool):
            raise ValueError("use_goal_multiplier must be boolean")
        if self.promoted_initialization not in ("previous_promoted_mean", "fixed_1450"):
            raise ValueError("Unknown promoted initialization")

    @property
    def config_id(self) -> str:
        """Identify all six settings stably, including equivalent integer/float inputs."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "elo-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EloResult:
    """Prematch features and season endpoints; elo_end is metadata, never a feature."""

    prematch: pd.DataFrame
    season_snapshots: pd.DataFrame
    config_id: str


Game: TypeAlias = tuple[int, int, pd.Timestamp, int, int, int, int, bool]


@dataclass(frozen=True)
class EloDay:
    """Immutable fixtures sharing one morning rating state."""

    date: pd.Timestamp
    games: tuple[Game, ...]


@dataclass(frozen=True)
class EloSeason:
    """A season's schedule and roster, projected once for a calibration grid."""

    season_id: int
    roster: tuple[int, ...]
    days: tuple[EloDay, ...]


@dataclass(frozen=True)
class PreparedElo:
    """Validated result/flag/day projection; independent of every Elo parameter."""

    seasons: tuple[EloSeason, ...]


def expected_home_score(home: float, away: float, home_advantage: float) -> float:
    """Expected 1/0.5/0 score, including HFA; this is not a home-win probability."""
    gap = (home + home_advantage - away) / RATING_SCALE
    if gap >= 0:
        return 1.0 / (1.0 + math.pow(10.0, -gap))
    odds = math.pow(10.0, gap)
    return odds / (1.0 + odds)


def goal_difference_multiplier(difference: int, enabled: bool = True) -> float:
    """Give larger score margins the fixed, reviewed T-010 multiplier."""
    margin = abs(difference)
    if not enabled or margin <= 1:
        return 1.0
    return 1.5 if margin == 2 else (11 + margin) / 8


def _games(tables: AnalysisTables, nothing_to_play: pd.Series) -> pd.DataFrame:
    rows = game_rows(tables)
    if rows.empty or not set(rows.season_id.unique()) <= set(DEVELOPMENT_SEASONS):
        raise ValueError("Elo requires nonempty development seasons 8–25")
    if rows.as_of.isna().any():
        raise ValueError("Elo requires calendar dates")
    if (
        nothing_to_play.index.names != ["game_id", "team_id"]
        or not nothing_to_play.index.is_unique
        or not pd.api.types.is_bool_dtype(nothing_to_play.dtype)
    ):
        raise ValueError("Flags require a unique (game_id, team_id) boolean Series")
    flags = nothing_to_play.reindex(pd.MultiIndex.from_frame(rows[["game_id", "team_id"]]))
    if flags.isna().any():
        raise ValueError("Missing nothing_to_play flag for a team-game")
    rows["dead"] = flags.to_numpy(dtype=bool)
    home = rows.loc[rows.is_home.eq(True)].set_index("game_id")
    away = rows.loc[rows.is_home.eq(False)].set_index("game_id")
    if not home.index.is_unique or not away.index.is_unique or set(home.index) != set(away.index):
        raise ValueError("Expected exactly one home and one away row per game")
    away = away.reindex(home.index)
    if (
        len(home) * 2 != len(rows)
        or not (
            home.team_id.eq(away.opponent_id)
            & home.opponent_id.eq(away.team_id)
            & home.team_id.ne(home.opponent_id)
            & home.date.eq(away.date)
            & home.season_id.eq(away.season_id)
            & home.goals_scored.eq(away.goals_conceded)
            & home.goals_conceded.eq(away.goals_scored)
        )
        .fillna(False)
        .all()
    ):
        raise ValueError("Inconsistent paired result rows")
    goals = home[["goals_scored", "goals_conceded"]]
    if goals.isna().any().any() or goals.lt(0).any().any() or goals.mod(1).ne(0).any().any():
        raise ValueError("Goals must be nonnegative integers")
    games = home[["season_id", "as_of", "team_id", "opponent_id", *goals.columns]].copy()
    games["dead"] = home.dead | away.dead
    return games.reset_index().sort_values(["season_id", "as_of", "game_id"])


def _prepare_season(games: pd.DataFrame, season: int) -> EloSeason:
    days = tuple(
        EloDay(day, tuple(batch.itertuples(index=False, name=None)))
        for day, batch in games.groupby("as_of", sort=False)
    )
    roster = tuple(
        sorted(set(int(t) for t in games.team_id) | set(int(t) for t in games.opponent_id))
    )
    return EloSeason(season, roster, days)


def prepare_elo(tables: AnalysisTables, nothing_to_play: pd.Series) -> PreparedElo:
    """Project and validate result rows, flags and calendar batches once, without I/O."""
    games = _games(tables, nothing_to_play)
    seasons = sorted(int(s) for s in games.season_id.unique())
    if seasons != list(range(8, seasons[-1] + 1)):
        raise ValueError("Full Elo history must start at season 8 without gaps; use replay")
    prepared = tuple(
        _prepare_season(fixtures, int(season))
        for season, fixtures in games.groupby("season_id", sort=True)
    )
    for previous, current in zip(prepared, prepared[1:], strict=False):
        if current.days[0].date <= previous.days[-1].date:
            raise ValueError("Season calendars overlap; cannot use future or same-day results")
    return PreparedElo(prepared)


def _run_prepared_season(
    season: EloSeason, initial: dict[int, float], params: EloParams
) -> tuple[pd.DataFrame, dict[int, float]]:
    ratings = initial.copy()
    rows = []
    for batch in season.days:
        day = batch.date
        changes: dict[int, float] = {}
        for game, season_id, _, home, away, gf, ga, dead in batch.games:
            home, away = int(home), int(away)
            home_pre, away_pre = ratings[home], ratings[away]
            expected = expected_home_score(home_pre, away_pre, params.home_advantage)
            rows.append(
                (game, home, season_id, day, home_pre, away_pre, home_pre - away_pre, expected)
            )
            rows.append(
                (game, away, season_id, day, away_pre, home_pre, away_pre - home_pre, 1 - expected)
            )
            difference = int(gf - ga)
            score = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
            discount = params.dead_game_discount if dead else 1.0
            multiplier = goal_difference_multiplier(difference, params.use_goal_multiplier)
            delta = params.k * multiplier * discount * (score - expected)
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        # Every fixture on this date used the same morning state, even if a team
        # occurs twice. Applying the paired increments only now prevents leakage.
        for team, change in changes.items():
            ratings[team] += change
    frame = (
        pd.DataFrame(
            rows,
            columns=[
                "game_id",
                "team_id",
                "season_id",
                "as_of",
                "elo_pre",
                "elo_opponent_pre",
                "elo_diff",
                "elo_expected",
            ],
        )
        .set_index(["game_id", "team_id"])
        .sort_index()
    )
    validate_feature_columns(frame, "elo")
    return frame, ratings


def compute_elo(
    tables: AnalysisTables, nothing_to_play: pd.Series, params: EloParams | None = None
) -> EloResult:
    """Run consecutive development history from season 8, without I/O or fitting.

    Newcomers are inferred only from consecutive season rosters. Returning clubs
    are newcomers again; their old ratings are discarded. Historical promoted
    endpoints enter the initialization pool only after that season has finished.
    """
    return compute_prepared_elo(prepare_elo(tables, nothing_to_play), params)


def compute_prepared_elo(prepared: PreparedElo, params: EloParams | None = None) -> EloResult:
    """Run the identical rating arithmetic on reusable, immutable calendar batches."""
    params = params or EloParams()
    previous: dict[int, float] = {}
    promoted_endings: list[float] = []
    features: list[pd.DataFrame] = []
    snapshots: list[tuple[int, int, float, float]] = []
    for fixtures in prepared.seasons:
        season, roster = fixtures.season_id, fixtures.roster
        promoted = set(roster) - previous.keys() if season != 8 else set()
        newcomer_rating = PROMOTED_FALLBACK
        if (
            params.promoted_initialization == "previous_promoted_mean"
            and len(promoted_endings) >= MIN_PREVIOUS_PROMOTED
        ):
            newcomer_rating = mean(promoted_endings)
        initial = {
            team: BASE_RATING
            if season == 8
            else (
                previous[team] + params.season_regression * (BASE_RATING - previous[team])
                if team in previous
                else newcomer_rating
            )
            for team in sorted(roster)
        }
        prematch, previous = _run_prepared_season(fixtures, initial, params)
        features.append(prematch)
        snapshots.extend((int(season), t, initial[t], previous[t]) for t in sorted(roster))
        promoted_endings.extend(previous[t] for t in sorted(promoted))
    return EloResult(
        pd.concat(features).sort_index(),
        pd.DataFrame(snapshots, columns=["season_id", "team_id", "elo_start", "elo_end"]).set_index(
            ["season_id", "team_id"]
        ),
        params.config_id,
    )


def replay_elo_season(
    tables: AnalysisTables,
    nothing_to_play: pd.Series,
    *,
    season_id: int,
    elo_start: pd.Series,
    params: EloParams | None = None,
) -> EloResult:
    """Reconstruct one season from its opening ratings, results and prematch flags.

    No earlier ratings or promoted-team history are needed. The original EloParams
    and the same nothing_to_play flags must accompany the opening snapshot.
    """
    params = params or EloParams()
    games = _games(tables, nothing_to_play)
    games = games.loc[games.season_id.eq(season_id)]
    roster = set(games.team_id) | set(games.opponent_id)
    if games.empty or not elo_start.index.is_unique or set(elo_start.index) != roster:
        raise ValueError("Opening snapshot must match the entire requested season roster")
    initial = {int(t): float(r) for t, r in elo_start.items()}
    if not all(math.isfinite(r) for r in initial.values()):
        raise ValueError("Opening ratings must be finite")
    prematch, final = _run_prepared_season(_prepare_season(games, season_id), initial, params)
    snapshot = pd.DataFrame(
        [(season_id, t, initial[t], final[t]) for t in sorted(initial)],
        columns=["season_id", "team_id", "elo_start", "elo_end"],
    ).set_index(["season_id", "team_id"])
    return EloResult(prematch, snapshot, params.config_id)


@dataclass(frozen=True)
class EloBlock:
    """FeatureBlock adapter; the only extra input is table_state.nothing_to_play."""

    nothing_to_play: pd.Series
    params: EloParams = EloParams()
    name: str = "elo"
    football_rationale: str = "Past results summarize team strength before the next game."

    def compute(self, tables: AnalysisTables) -> pd.DataFrame:
        """Return registered prematch features; season endpoints remain separate."""
        return compute_elo(tables, self.nothing_to_play, self.params).prematch
