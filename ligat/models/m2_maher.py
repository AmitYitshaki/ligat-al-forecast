"""Cold-start Maher Poisson model with identification enforced inside optimization."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from ligat.models.m2_xgb import _alignment, _digest

MAHER_COLUMNS = ("team_id", "opponent_id", "is_home")
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MaherGoalConfig:
    """No decay or promoted-team penalty in the C4a baseline."""

    half_life: None = None

    @property
    def config_id(self) -> str:
        return "maher-" + _digest({"format_version": _FORMAT_VERSION, "config": asdict(self)})

    @property
    def label(self) -> str:
        return "Maher · בלי דעיכה"

    def __post_init__(self) -> None:
        if self.half_life is not None:
            raise ValueError("C4a supports no decay only")


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
) -> tuple[float, NDArray[np.float64]]:
    """Weighted training likelihood and analytic gradient in identified coordinates."""
    a, b, gamma = unpack_parameters(theta, n)
    eta = a[team] + b[opponent] + gamma * home
    rates = np.exp(eta)
    residual = weights * (rates - goals)
    ga = np.bincount(team, weights=residual, minlength=n)
    gb = np.bincount(opponent, weights=residual, minlength=n)
    gradient = np.r_[ga[:-1] - ga[-1], gb, residual @ home]
    mass = weights.sum()
    return float(weights @ (rates - goals * eta) / mass), gradient / mass


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


class MaherGoalModel:
    """Daily cold fits on judge-cut history, with an explicit prediction as_of."""

    refit_policy: ClassVar[Literal["per_matchday"]] = "per_matchday"
    supports_sample_weight: ClassVar[bool] = False
    requires_as_of: ClassVar[bool] = True

    def __init__(self, config: MaherGoalConfig) -> None:
        self._config = config
        self._parameters: pd.DataFrame | None = None
        self.log_gamma: float | None = None
        self.as_of: pd.Timestamp | None = None
        self.unseen_team_rows = 0
        self.unseen_teams: tuple[int, ...] = ()
        self.iterations = 0

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
        teams = np.union1d(own, opp)
        n = len(teams)
        ti, oi = np.searchsorted(teams, own), np.searchsorted(teams, opp)
        initial = np.zeros(2 * n)
        initial[n - 1 : 2 * n - 1] = np.log(goals.mean())
        with threadpool_limits(limits=1):
            result = minimize(
                maher_objective,
                initial,
                args=(ti, oi, home, goals, np.ones(len(y)), n),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-13},
            )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Maher did not converge at as_of={as_of}: {result.message}")
        a, b, gamma = unpack_parameters(result.x, n)
        self._parameters = pd.DataFrame(
            {"log_alpha": a, "log_beta": b}, index=pd.Index(teams, name="team_id")
        )
        self.log_gamma, self.as_of, self.iterations = gamma, as_of, int(result.nit)
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
        }
        path.write_text(
            json.dumps({"payload": payload, "sha256": _digest(payload)}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        saved = json.loads(path.read_text(encoding="utf-8"))
        payload = saved["payload"]
        if saved["sha256"] != _digest(payload) or payload["format_version"] != _FORMAT_VERSION:
            raise ValueError("Invalid Maher signature or format version")
        config = MaherGoalConfig(**payload["config"])
        if payload["config_id"] != config.config_id:
            raise ValueError("Invalid Maher configuration signature")
        result = cls(config)
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
        return result
