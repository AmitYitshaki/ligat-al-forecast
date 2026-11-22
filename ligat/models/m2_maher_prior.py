"""Target-free season rosters and frozen promoted-team Maher prior estimation."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from ligat.evaluate.goal_inputs import GoalData
from ligat.models.m2_xgb import _digest


@dataclass(frozen=True)
class PromotedPrior:
    """League-relative log attack/defense averages, frozen from development seasons."""

    a_star: float
    b_star: float
    source_sha256: str
    team_seasons: int

    def __post_init__(self) -> None:
        if not np.isfinite([self.a_star, self.b_star]).all() or self.team_seasons < 1:
            raise ValueError("Invalid promoted-team prior")
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_sha256
        ):
            raise ValueError("Promoted prior requires its input SHA-256")


@dataclass(frozen=True)
class SeasonRoster:
    """Only season, team, promoted flag and season-start date; no match outcomes."""

    rows: tuple[tuple[int, int, bool, str], ...]

    def teams(self, season: int) -> tuple[int, ...]:
        """Every current-season club, including entrants without a training fixture."""
        return tuple(team for s, team, _, _ in self.rows if s == season)

    def promoted(self, season: int) -> tuple[int, ...]:
        return tuple(team for s, team, flag, _ in self.rows if s == season and flag)

    def season_for(self, as_of: pd.Timestamp) -> int:
        starts = {s: pd.Timestamp(start) for s, _, _, start in self.rows}
        eligible = [s for s, start in starts.items() if start <= as_of]
        if not eligible:
            raise ValueError("No season roster at the requested as_of")
        return max(eligible)

    def start(self, season: int) -> pd.Timestamp:
        dates = {date for s, _, _, date in self.rows if s == season}
        if len(dates) != 1:
            raise ValueError("A roster must declare exactly one start per season")
        return pd.Timestamp(dates.pop())


def season_roster(data: GoalData) -> SeasonRoster:
    """Read promoted flags and fixture timing from M0, never from outcomes."""
    if "promoted_in" not in data.X:
        raise ValueError("Season roster requires M0 promoted_in")
    frame = data.context.loc[:, ["season_id", "team_id", "date"]].copy()
    frame["promoted_in"] = data.X.promoted_in
    frame = frame.reset_index(drop=True)
    if frame.promoted_in.isna().any() or not pd.api.types.is_bool_dtype(frame.promoted_in.dtype):
        raise ValueError("Roster promoted flags must be complete booleans")
    rows = []
    for season, group in frame.groupby("season_id", sort=True):
        start = group.date.min().normalize().isoformat()
        for team, part in group.groupby("team_id", sort=True):
            if part.promoted_in.nunique() != 1:
                raise ValueError("M0 promoted flag must be constant within a team-season")
            rows.append((int(season), int(team), bool(part.promoted_in.iloc[0]), start))
    return SeasonRoster(tuple(rows))


def estimate_promoted_prior(data: GoalData) -> dict[str, object]:
    """Fit each complete development season separately, without decay or penalties.

    Season 8 is the initialization season: promotion cannot be established from
    earlier development seasons, so it contributes no promoted observations.
    """
    from ligat.models.m2_maher import MAHER_COLUMNS, MaherGoalConfig, MaherGoalModel

    if set(data.context.season_id) != set(range(8, 26)):
        raise ValueError("Frozen prior requires exactly development seasons 8-25")
    roster = season_roster(data)
    source = data.X.loc[:, [*MAHER_COLUMNS, "promoted_in"]].copy()
    source["season_id"] = data.context.season_id
    source["date"] = data.context.date
    source["goals_scored"] = data.y
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(source, index=True).to_numpy(dtype="<u8").tobytes()
    ).hexdigest()
    source_hash = _digest({"columns": list(source.columns), "rows_sha256": digest})
    attacks, defenses, table = [], [], []
    for season in range(8, 26):
        mask = data.context.season_id.eq(season)
        context = data.context.loc[mask]
        model = MaherGoalModel(MaherGoalConfig()).fit(
            data.X.loc[mask],
            data.y.loc[mask],
            context=context,
            as_of=context.date.max().normalize() + pd.Timedelta(days=1),
        )
        params = model.parameters
        promoted = () if season == 8 else roster.promoted(season)
        mean_beta = float(params.log_beta.mean())
        rows = []
        for team in promoted:
            attack = float(params.loc[team, "log_alpha"])
            defense = float(params.loc[team, "log_beta"] - mean_beta)
            attacks.append(attack)
            defenses.append(defense)
            rows.append({"team_id": team, "log_alpha": attack, "relative_log_beta": defense})
        table.append(
            {
                "season_id": season,
                "teams": len(params),
                "mean_log_beta": mean_beta,
                "promoted": rows,
                "initialization_excluded": season == 8,
            }
        )
    prior = PromotedPrior(
        float(np.mean(attacks)), float(np.mean(defenses)), source_hash, len(attacks)
    )
    return {
        "format_version": 1,
        "prior": asdict(prior),
        "seasons": table,
        "method": "identified unpenalized per-season Maher; season 8 initialization excluded",
    }


def load_promoted_prior(path: Path = Path("config/m2_maher_prior.yaml")) -> PromotedPrior:
    """Load the frozen input-signed prior without refitting or reading outcomes."""
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    if saved["format_version"] != 1:
        raise ValueError("Unsupported promoted prior format")
    return PromotedPrior(**saved["prior"])


def verify_promoted_prior(
    data: GoalData, path: Path = Path("config/m2_maher_prior.yaml")
) -> dict[str, object]:
    """Recompute the complete prior table and reject any difference from the frozen file."""
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = estimate_promoted_prior(data)
    if actual != expected:
        raise ValueError(
            "Recomputed promoted prior or its input signature differs from the frozen file"
        )
    return actual
