"""EXP-002 Poisson and ordinal adapters; no candidate selection occurs here."""

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from ligat.evaluate.goal_metrics import poisson_nll
from ligat.models.ordered_logit import (
    PRECISION_LOSS_GRADIENT_TOL,
    OrderedFit,
    fit_ordered_logit,
    ordered_probabilities,
)


@dataclass(frozen=True)
class PoissonFit:
    """Unweighted log-link MLE diagnostics, including recomputed convergence gradient."""

    coefficients: tuple[float, ...]
    converged: bool
    iterations: int
    accepted_by_gradient_check: bool
    max_abs_gradient: float


def fit_poisson(x: ArrayLike, goals: ArrayLike) -> PoissonFit:
    """Fit an explicit design matrix (including its intercept) with no observation weights."""
    design, y = np.asarray(x, float), np.asarray(goals, float)
    if (
        design.ndim != 2
        or not len(design)
        or not design.shape[1]
        or not np.isfinite(design).all()
        or y.shape != (len(design),)
    ):
        raise ValueError("Expected aligned finite Poisson design and target")
    poisson_nll(y, np.ones(len(y)))
    if y.sum() <= 0 or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Poisson MLE requires positive goal mass and full-rank design")
    initial = np.zeros(design.shape[1])
    if np.array_equal(design[:, 0], np.ones(len(design))):
        initial[0] = np.log(y.mean())

    def objective(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        eta = design @ beta
        with np.errstate(over="ignore", invalid="ignore"):
            lam = np.exp(eta)
            value = float(np.mean(lam - y * eta))
            gradient = design.T @ (lam - y) / len(y)
        return value, gradient

    result = minimize(
        objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000}
    )
    _, gradient = objective(result.x)
    maximum = float(np.max(np.abs(gradient)))
    accepted = bool(
        not result.success and result.status == 2 and maximum <= PRECISION_LOSS_GRADIENT_TOL
    )
    if not np.isfinite(result.x).all() or not (result.success or accepted):
        raise RuntimeError(f"Poisson optimizer did not converge: status={result.status}")
    return PoissonFit(tuple(float(v) for v in result.x), True, int(result.nit), accepted, maximum)


class FormPoissonPredictor:
    """Cross own attack with opponent defense, respecting each team's current venue."""

    def __init__(self, *, with_elo: bool, with_form: bool, log_offset: float) -> None:
        if not np.isfinite(log_offset) or log_offset <= 0:
            raise ValueError("Positive predeclared log offset required")
        self.with_elo, self.with_form, self.log_offset = with_elo, with_form, log_offset
        self.fit_result: PoissonFit | None = None
        self.training_seasons: tuple[int, ...] = ()

    def design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Interleave home/away team rows; apply log(x+offset) inside this route only."""
        n = len(features)
        columns = [np.ones(2 * n), np.tile([1.0, 0.0], n)]
        if self.with_form:
            gf = features[["home_gf", "away_gf"]].to_numpy(float).ravel()
            ga = features[["away_ga", "home_ga"]].to_numpy(float).ravel()
            if (
                not np.isfinite(gf).all()
                or not np.isfinite(ga).all()
                or (gf < 0).any()
                or (ga < 0).any()
            ):
                raise ValueError("Attack and defense must be finite and nonnegative")
            columns += [np.log(gf + self.log_offset), np.log(ga + self.log_offset)]
        if self.with_elo:
            diff = features.elo_diff.to_numpy(float) / 400
            columns.append(np.column_stack((diff, -diff)).ravel())
        result = np.column_stack(columns)
        if not n or not np.isfinite(result).all():
            raise ValueError("Nonempty finite Poisson features required")
        return result

    def fit(self, features: pd.DataFrame, goals: pd.DataFrame) -> None:
        """Fit from training features and separately supplied home/away goal targets."""
        self.fit_result = None
        if not features.index.equals(goals.index):
            raise ValueError("Poisson targets must align with training features")
        self.training_seasons = tuple(sorted(int(s) for s in features.season_id.unique()))
        self.fit_result = fit_poisson(
            self.design(features), goals[["home_goals", "away_goals"]].to_numpy().ravel()
        )

    def predict_goals(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Return n×2 λ values without receiving observed match goals."""
        if self.fit_result is None:
            raise RuntimeError("Fit Poisson before prediction")
        return np.asarray(
            np.exp(self.design(features) @ np.array(self.fit_result.coefficients)).reshape(-1, 2),
            dtype=float,
        )


class FormOrderedPredictor:
    """Elo first and optional home-minus-away points form, with no time weights."""

    def __init__(self, *, with_form: bool = True) -> None:
        self.with_form = with_form
        self.fit_result: OrderedFit | None = None
        self.training_seasons: tuple[int, ...] = ()

    def design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Always use a matrix; the single-column path reuses the original arithmetic."""
        columns = [features.elo_diff.to_numpy(float) / 400]
        if self.with_form:
            columns.append(features.points_difference.to_numpy(float))
        return np.column_stack(columns)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit Elo and optional form using only the current fold's training rows."""
        self.fit_result = None
        self.training_seasons = tuple(sorted(int(s) for s in train.season_id.unique()))
        self.fit_result = fit_ordered_logit(self.design(train), train.outcome.to_numpy())

    def predict(self, games: pd.DataFrame) -> NDArray[np.float64]:
        """Use only explicit prematch features in the original supplied row order."""
        if self.fit_result is None:
            raise RuntimeError("Fit ordered form before prediction")
        return ordered_probabilities(self.fit_result, self.design(games))
