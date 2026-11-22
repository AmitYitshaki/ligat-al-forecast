"""Cold-start Maher Poisson model with identification enforced inside optimization."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from ligat.models.m2_maher_prior import PromotedPrior, SeasonRoster
from ligat.models.m2_xgb import _alignment, _digest
from ligat.models.training_log import ObjectiveJournal, TrainingJournal

MAHER_COLUMNS = ("team_id", "opponent_id", "is_home")
_FORMAT_VERSION = 4
_MAX_ITERATIONS = 20000


@dataclass(frozen=True)
class MaherGoalConfig:
    """Day half-life and optional frozen prior; None prior retains the C4a baseline."""

    half_life: float | None = None
    prior: PromotedPrior | None = None

    @property
    def config_id(self) -> str:
        return "maher-" + _digest({"format_version": _FORMAT_VERSION, "config": asdict(self)})

    @property
    def label(self) -> str:
        label = (
            "Maher · בלי דעיכה"
            if self.half_life is None
            else f"Maher · דעיכה {self.half_life:g} עונה"
        )
        return label if self.prior is None else label + " · עולה 6 משחקים"

    def __post_init__(self) -> None:
        if isinstance(self.half_life, (bool, np.bool_)) or self.half_life not in (None, 0.5, 1, 2):
            raise ValueError("Maher half_life must be None, 0.5, 1 or 2 seasons")
        if self.half_life is not None:
            object.__setattr__(self, "half_life", float(self.half_life))
        if self.prior is not None and not isinstance(self.prior, PromotedPrior):
            raise ValueError("Maher prior must be a frozen PromotedPrior")


def day_weights(
    dates: pd.Series, as_of: pd.Timestamp, half_life: float | None
) -> NDArray[np.float64]:
    """Continuous calendar age: one season is exactly 365.25 days (Q36)."""
    ages = (as_of - dates).dt.total_seconds().to_numpy(float) / 86400
    if not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("Decay dates must not follow as_of")
    return np.ones(len(dates)) if half_life is None else np.power(0.5, ages / (365.25 * half_life))


def unpack_parameters(
    theta: NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Last attack is minus all others, so sum(log alpha) is always zero."""
    a = np.r_[theta[: n - 1], -theta[: n - 1].sum()]
    return a, theta[n - 1 : 2 * n - 1], float(theta[-1])


def maher_objective(
    theta: NDArray[np.float64],
    team: NDArray[np.int64],
    opponent: NDArray[np.int64],
    home: NDArray[np.float64],
    goals: NDArray[np.float64],
    weights: NDArray[np.float64],
    n: int,
    promoted: NDArray[np.bool_] | None = None,
    a_star: float = 0.0,
    b_star: float = 0.0,
    penalty_strength: float = 0.0,
    reference: NDArray[np.bool_] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Weighted training likelihood and analytic gradient in identified coordinates."""
    a, b, gamma = unpack_parameters(theta, n)
    eta = a[team] + b[opponent] + gamma * home
    rates = np.exp(eta)
    residual = weights * (rates - goals)
    ga = np.bincount(team, weights=residual, minlength=n)
    gb = np.bincount(opponent, weights=residual, minlength=n)
    value = float(weights @ (rates - goals * eta))
    if promoted is not None and penalty_strength > 0:
        if reference is None or not reference.any():
            raise ValueError("Promoted penalty requires the current-season reference clubs")
        size = int(reference.sum())
        da = promoted * (a - a[reference].mean() - a_star)
        db = promoted * (b - b[reference].mean() - b_star)
        value += 0.5 * penalty_strength * float(da @ da + db @ db)
        ga += penalty_strength * (da - reference * da.sum() / size)
        gb += penalty_strength * (db - reference * db.sum() / size)
    gradient = np.r_[ga[:-1] - ga[-1], gb, residual @ home]
    mass = weights.sum()
    return value / mass, gradient / mass


def _input(X: pd.DataFrame) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    if not isinstance(X, pd.DataFrame) or not set(MAHER_COLUMNS) <= set(X) or X.empty:
        raise ValueError("Maher requires explicit team_id, opponent_id and is_home")
    for name in ("team_id", "opponent_id"):
        values = X[name].to_numpy(float, na_value=np.nan)
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError("Maher identities must be complete positive integers")
    home = X.is_home.to_numpy(float, na_value=np.nan)
    if not np.isin(home, [0.0, 1.0]).all():
        raise ValueError("Maher is_home must be boolean")
    return X.team_id.to_numpy(np.int64), X.opponent_id.to_numpy(np.int64), home


class MaherGoalModel(TrainingJournal):
    """Daily cold fits on judge-cut history, with an explicit prediction as_of."""

    refit_policy: ClassVar[Literal["per_matchday"]] = "per_matchday"
    supports_sample_weight: ClassVar[bool] = False
    requires_as_of: ClassVar[bool] = True

    def __init__(
        self,
        config: MaherGoalConfig,
        *,
        roster: SeasonRoster | None = None,
        record_training_log: bool = True,
    ) -> None:
        self.record_training_log = record_training_log
        self._training_log = None
        self._config = config
        self.roster = roster
        self._parameters: pd.DataFrame | None = None
        self.log_gamma: float | None = None
        self.as_of: pd.Timestamp | None = None
        self.unseen_team_rows = 0
        self.unseen_teams: tuple[int, ...] = ()
        self.iterations = 0
        self.penalty_teams: tuple[int, ...] = ()
        self.penalty_strength = 0.0

    @property
    def config(self) -> MaherGoalConfig:
        return self._config

    @property
    def parameters(self) -> pd.DataFrame:
        if self._parameters is None:
            raise RuntimeError("Fit or load Maher before reading parameters")
        return self._parameters.copy(deep=True)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        context: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        prediction_features: pd.DataFrame | None = None,
    ) -> Self:
        started = perf_counter()
        self._training_log = None
        self._parameters, self.log_gamma, self.as_of = None, None, None
        self.unseen_team_rows, self.unseen_teams = 0, ()
        if sample_weight is not None:
            raise ValueError(
                "Maher rejects external weights; day decay belongs to its configuration"
            )
        _alignment(X, context)
        if as_of is None or as_of.tzinfo is None or str(as_of.tz) != "Asia/Jerusalem":
            raise ValueError("Maher requires an explicit Asia/Jerusalem as_of")
        if "date" not in context or not context.date.lt(as_of).all():
            raise ValueError(f"Maher history must be strictly earlier than as_of={as_of}")
        if not isinstance(y, pd.Series) or not y.index.equals(X.index):
            raise ValueError("Maher targets must align with features")
        goals = y.to_numpy(float, na_value=np.nan)
        if (
            not np.isfinite(goals).all()
            or (goals < 0).any()
            or (goals % 1 != 0).any()
            or goals.sum() <= 0
        ):
            raise ValueError("Maher requires nonnegative integer goals with positive mass")
        own, opp, home = _input(X)
        weights = day_weights(context.date, as_of, self.config.half_life)
        promoted_teams: tuple[int, ...] = ()
        reference_teams: tuple[int, ...] = ()
        strength = 0.0
        prior = self.config.prior
        if prior is not None:
            if self.roster is None:
                raise ValueError("Maher promoted prior requires target-free M0 season roster")
            season = self.roster.season_for(as_of)
            reference_teams = self.roster.teams(season)
            promoted_teams = self.roster.promoted(season) if season > 8 else ()
            w0 = day_weights(pd.Series([self.roster.start(season)]), as_of, self.config.half_life)[
                0
            ]
            strength = 6 * float(np.average(goals, weights=weights)) * w0
        teams = np.union1d(np.union1d(own, opp), np.asarray(reference_teams, dtype=np.int64))
        n = len(teams)
        ti, oi = np.searchsorted(teams, own), np.searchsorted(teams, opp)
        initial = np.zeros(2 * n)
        initial[n - 1 : 2 * n - 1] = np.log(np.average(goals, weights=weights))
        promoted_mask = np.isin(teams, promoted_teams)
        reference_mask = np.isin(teams, reference_teams)
        journal = ObjectiveJournal(maher_objective)
        with threadpool_limits(limits=1):
            result = minimize(
                journal if self.record_training_log else maher_objective,
                initial,
                args=(
                    ti,
                    oi,
                    home,
                    goals,
                    weights,
                    n,
                    promoted_mask,
                    0.0 if prior is None else prior.a_star,
                    0.0 if prior is None else prior.b_star,
                    strength,
                    reference_mask,
                ),
                jac=True,
                callback=journal.callback if self.record_training_log else None,
                method="L-BFGS-B",
                options={"maxiter": _MAX_ITERATIONS, "gtol": 1e-8, "ftol": 1e-13},
            )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Maher did not converge at as_of={as_of}: {result.message}")
        self._training_log = {
            "model": "maher",
            "config_id": self.config.config_id,
            "label": self.config.label,
            "actual_params": {
                "method": "L-BFGS-B",
                "gtol": 1e-8,
                "ftol": 1e-13,
                "maxiter": _MAX_ITERATIONS,
                **asdict(self.config),
            },
            "n_rows": len(X),
            "n_features": 2 * n,
            "fit_seconds": perf_counter() - started,
            "dataset": "training",
            "metric": "weighted_poisson_without_log_factorial_plus_prior",
            "train_loss": journal.losses,
            "iterations": int(result.nit),
            "converged": bool(result.success),
            "message": str(result.message),
            "final_loss": float(result.fun),
            "as_of": as_of.isoformat(),
            "penalty_strength": strength,
            "promoted_teams": list(promoted_teams),
            "recorded": self.record_training_log,
        }
        a, b, gamma = unpack_parameters(result.x, n)
        self._parameters = pd.DataFrame(
            {"log_alpha": a, "log_beta": b}, index=pd.Index(teams, name="team_id")
        )
        self.log_gamma, self.as_of, self.iterations = gamma, as_of, int(result.nit)
        self.penalty_teams, self.penalty_strength = promoted_teams, strength
        return self

    def predict(self, X: pd.DataFrame, *, context: pd.DataFrame) -> pd.Series:
        _alignment(X, context)
        parameters = self.parameters
        own, opp, home = _input(X)
        assert self.log_gamma is not None
        missing = ~np.isin(own, parameters.index)
        self.unseen_team_rows += int(missing.sum())
        self.unseen_teams = tuple(
            sorted(set(self.unseen_teams) | set(int(v) for v in own[missing]))
        )
        a = parameters.log_alpha.reindex(own).fillna(0).to_numpy()
        b = parameters.log_beta.reindex(opp).fillna(float(parameters.log_beta.mean())).to_numpy()
        rates = np.exp(a + b + self.log_gamma * home)
        if not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("Maher must return positive finite lambda")
        return pd.Series(rates, index=X.index.copy(), name="lambda")

    def save(self, path: Path) -> None:
        params = self.parameters
        payload = {
            "training_log": self.training_log,
            "format_version": _FORMAT_VERSION,
            "config": asdict(self.config),
            "config_id": self.config.config_id,
            "team_ids": params.index.tolist(),
            "log_alpha": params.log_alpha.tolist(),
            "log_beta": params.log_beta.tolist(),
            "log_gamma": self.log_gamma,
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
            "unseen_team_rows": self.unseen_team_rows,
            "unseen_teams": self.unseen_teams,
            "iterations": self.iterations,
            "roster": None if self.roster is None else self.roster.rows,
            "penalty_teams": self.penalty_teams,
            "penalty_strength": self.penalty_strength,
        }
        path.write_text(
            json.dumps({"payload": payload, "sha256": _digest(payload)}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = saved["payload"]
        if payload["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Maher format_version {payload['format_version']}; "
                f"expected {_FORMAT_VERSION}"
            )
        if saved["sha256"] != _digest(payload):
            raise ValueError("Invalid Maher signature or format version")
        config_values = dict(payload["config"])
        if config_values.get("prior") is not None:
            config_values["prior"] = PromotedPrior(**config_values["prior"])
        config = MaherGoalConfig(**config_values)
        if payload["config_id"] != config.config_id:
            raise ValueError("Invalid Maher configuration signature")
        roster = (
            None
            if payload["roster"] is None
            else SeasonRoster(tuple(tuple(row) for row in payload["roster"]))
        )
        result = cls(config, roster=roster)
        result._training_log = payload["training_log"]
        result._parameters = pd.DataFrame(
            {"log_alpha": payload["log_alpha"], "log_beta": payload["log_beta"]},
            index=pd.Index(payload["team_ids"], name="team_id"),
        )
        if (
            not result._parameters.index.is_unique
            or not np.isfinite(result._parameters.to_numpy()).all()
            or abs(result._parameters.log_alpha.sum()) > 1e-10
        ):
            raise ValueError("Invalid Maher parameters or identification")
        result.log_gamma = float(payload["log_gamma"])
        result.as_of = pd.Timestamp(payload["as_of"]).tz_convert("Asia/Jerusalem")
        result.unseen_team_rows = int(payload["unseen_team_rows"])
        result.unseen_teams = tuple(payload["unseen_teams"])
        result.iterations = int(payload["iterations"])
        result.penalty_teams = tuple(payload["penalty_teams"])
        result.penalty_strength = float(payload["penalty_strength"])
        return result
