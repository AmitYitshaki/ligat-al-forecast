"""Weighted, three-parameter ordinal conversion of raw prematch Elo differences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import expit, logit  # type: ignore[import-untyped]

from ligat.evaluate.metrics import validate_outcomes, validate_probabilities
from ligat.features.elo import RATING_SCALE

PRECISION_LOSS_GRADIENT_TOL = 1e-6
"""Maximum absolute normalized-objective gradient accepted for SciPy status 2 only."""


@dataclass(frozen=True)
class OrderedFit:
    """Unpenalized MLE, with thresholds in away/draw/home order."""

    beta: float
    theta1: float
    delta: float
    log_likelihood: float
    converged: bool
    iterations: int
    accepted_by_gradient_check: bool
    max_abs_gradient: float
    additional_betas: tuple[float, ...] = ()

    @property
    def theta2(self) -> float:
        """Keep the draw interval positive by construction."""
        return float(self.theta1 + np.exp(self.delta))


def _x(values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a finite, nonempty vector of Elo differences / 400")
    return x


def _log_probabilities(
    parameters: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    beta, theta1, delta = parameters
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gap = np.exp(delta)
        a = theta1 - beta * x
        b = a + gap
        # This form avoids cancellation in sigmoid(b) - sigmoid(a).
        log_draw = -np.logaddexp(0, -b) - np.logaddexp(0, a) + np.log(-np.expm1(-gap))
        logs = np.column_stack((-np.logaddexp(0, b), log_draw, -np.logaddexp(0, -a)))
        sa, sb = expit(a), expit(b)
        score_threshold = np.column_stack((-sb, 1 - sb - sa, 1 - sa))
        score_delta = np.column_stack(
            (-sb * gap, (1 - sb + np.exp(-gap) / -np.expm1(-gap)) * gap, np.zeros(len(x)))
        )
        scores = np.stack((-x[:, None] * score_threshold, score_threshold, score_delta), axis=2)
    return logs, scores


def fit_ordered_logit(
    x: ArrayLike,
    outcomes: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    nonnegative_first: bool = True,
) -> OrderedFit:
    """Fit beta/theta1/delta by weighted MLE; x contains no home-advantage offset.

    Outcome codes use the judge's H/D/A order (0/1/2). Zero weights are allowed;
    each outcome must still have positive training mass for finite thresholds.
    Precision loss (status 2) is accepted only with a recomputed normalized
    gradient within the fixed tolerance. Other failures or reversed strength stop.
    The optional EXP-003 alone route sets nonnegative_first=False because its
    first column is a context candidate, rather than Elo. Defaults are unchanged.
    """
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 2:
        if matrix.shape[1] == 1:
            return fit_ordered_logit(
                matrix[:, 0], outcomes, sample_weight, nonnegative_first=nonnegative_first
            )
        return _fit_matrix(matrix, outcomes, sample_weight, nonnegative_first=nonnegative_first)
    values = _x(x)
    y = validate_outcomes(outcomes, len(values))
    weights = np.ones(len(values)) if sample_weight is None else np.asarray(sample_weight, float)
    if (
        weights.shape != values.shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Weights must be finite, nonnegative, aligned, and have positive sum")
    mass = np.bincount(y, weights=weights, minlength=3)
    if (mass <= 0).any():
        raise ValueError("All three outcomes need positive training weight")
    theta1 = float(logit(mass[2] / mass.sum()))
    theta2 = float(logit((mass[2] + mass[1]) / mass.sum()))
    initial = np.array([1.0, theta1, np.log(theta2 - theta1)])
    indices = np.arange(len(values))
    normalized = weights / weights.sum()

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logs, scores = _log_probabilities(parameters, values)
        return (
            -float(normalized @ logs[indices, y]),
            -np.sum(normalized[:, None] * scores[indices, y], axis=0),
        )

    result = minimize(
        objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000}
    )
    failure_message = (
        f"Ordered-logit optimizer did not converge: status={result.status}; "
        f"message={result.message}"
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError(failure_message)
    _, gradient = objective(result.x)
    max_abs_gradient = float(np.max(np.abs(gradient)))
    accepted_by_gradient_check = bool(
        not result.success
        and result.status == 2
        and max_abs_gradient <= PRECISION_LOSS_GRADIENT_TOL
    )
    if not (result.success or accepted_by_gradient_check):
        raise RuntimeError(failure_message)
    beta, theta1, delta = (float(p) for p in result.x)
    if nonnegative_first and beta < 0:
        raise RuntimeError("Negative ordered-logit beta: stop and investigate")
    logs, _ = _log_probabilities(result.x, values)
    likelihood = float(weights @ logs[indices, y])
    if not np.isfinite(likelihood) or not np.isfinite(theta1 + np.exp(delta)):
        raise RuntimeError("Nonfinite ordered-logit fit")
    return OrderedFit(
        beta,
        theta1,
        delta,
        likelihood,
        True,
        int(result.nit),
        accepted_by_gradient_check,
        max_abs_gradient,
    )


def ordered_probabilities(fit: OrderedFit, x: ArrayLike) -> NDArray[np.float64]:
    """Return H/D/A probabilities; larger raw Elo differences favor the home team."""
    values = np.asarray(x, float)
    if fit.additional_betas:
        if values.ndim != 2 or values.shape[1] != 1 + len(fit.additional_betas):
            raise ValueError("Ordered feature matrix does not match fitted coefficients")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Ordered feature matrix must be finite and nonempty")
        parameters = np.array([fit.beta, *fit.additional_betas, fit.theta1, fit.delta])
        logs, _ = _matrix_logs(parameters, values)
    else:
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        logs, _ = _log_probabilities(np.array([fit.beta, fit.theta1, fit.delta]), _x(values))
    return validate_probabilities(np.exp(logs))


def _matrix_logs(
    parameters: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    beta, theta1, delta = parameters[:-2], parameters[-2], parameters[-1]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gap = np.exp(delta)
        a = theta1 - x @ beta
        b = a + gap
        logs = np.column_stack(
            (
                -np.logaddexp(0, b),
                -np.logaddexp(0, -b) - np.logaddexp(0, a) + np.log(-np.expm1(-gap)),
                -np.logaddexp(0, -a),
            )
        )
        sa, sb = expit(a), expit(b)
        threshold = np.column_stack((-sb, 1 - sb - sa, 1 - sa))
        d_delta = np.column_stack(
            (-sb * gap, (1 - sb + np.exp(-gap) / -np.expm1(-gap)) * gap, np.zeros(len(x)))
        )
        scores = np.concatenate(
            (-x[:, None, :] * threshold[:, :, None], threshold[:, :, None], d_delta[:, :, None]),
            axis=2,
        )
    return logs, scores


def _fit_matrix(
    x: NDArray[np.float64],
    outcomes: ArrayLike,
    sample_weight: ArrayLike | None,
    *,
    nonnegative_first: bool = True,
) -> OrderedFit:
    if not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite nonempty feature matrix with Elo first")
    y = validate_outcomes(outcomes, len(x))
    weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, float)
    if (
        weights.shape != (len(x),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Weights must be finite, nonnegative, aligned, and have positive sum")
    mass = np.bincount(y, weights=weights, minlength=3)
    if (mass <= 0).any():
        raise ValueError("All three outcomes need positive training weight")
    theta1 = float(logit(mass[2] / mass.sum()))
    theta2 = float(logit((mass[2] + mass[1]) / mass.sum()))
    initial = np.array([1.0, *([0.0] * (x.shape[1] - 1)), theta1, np.log(theta2 - theta1)])
    normalized = weights / weights.sum()
    indices = np.arange(len(x))

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logs, scores = _matrix_logs(parameters, x)
        return -float(normalized @ logs[indices, y]), -np.sum(
            normalized[:, None] * scores[indices, y], axis=0
        )

    result = minimize(
        objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000}
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Nonfinite ordered-matrix fit")
    _, gradient = objective(result.x)
    max_gradient = float(np.max(np.abs(gradient)))
    accepted = bool(
        not result.success and result.status == 2 and max_gradient <= PRECISION_LOSS_GRADIENT_TOL
    )
    if not (result.success or accepted):
        raise RuntimeError(f"Ordered-matrix optimizer did not converge: status={result.status}")
    if nonnegative_first and result.x[0] < 0:
        raise RuntimeError("Negative ordered-logit Elo beta: stop and investigate")
    logs, _ = _matrix_logs(result.x, x)
    likelihood = float(weights @ logs[indices, y])
    if not np.isfinite(likelihood) or not np.isfinite(result.x[-2] + np.exp(result.x[-1])):
        raise RuntimeError("Nonfinite ordered-matrix fit")
    return OrderedFit(
        float(result.x[0]),
        float(result.x[-2]),
        float(result.x[-1]),
        likelihood,
        True,
        int(result.nit),
        accepted,
        max_gradient,
        tuple(float(v) for v in result.x[1:-2]),
    )


def time_weights(
    seasons: ArrayLike, validation_season: int, half_life: float | None
) -> NDArray[np.float64]:
    """Weight training games by season age; the immediately preceding season has weight 1."""
    years = np.asarray(seasons, dtype=float)
    if years.ndim != 1 or not len(years) or not np.isfinite(years).all():
        raise ValueError("Training seasons must be a finite nonempty vector")
    if (years >= validation_season).any():
        raise ValueError("Time-weighted calibration requires strictly earlier training seasons")
    if half_life is None:
        return np.ones(len(years))
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("Half-life must be positive or None")
    return np.asarray(0.5 ** ((validation_season - 1 - years) / half_life), dtype=float)


class OrderedEloPredictor:
    """Judge adapter: thresholds learn home advantage on training seasons only."""

    def __init__(self, half_life: float | None = None) -> None:
        self.half_life = half_life
        self.fit_result: OrderedFit | None = None
        self.training_seasons: tuple[int, ...] = ()

    def fit(self, train: pd.DataFrame) -> None:
        """Fit only elo_diff/400 and outcomes; no HFA parameter enters this layer."""
        self.fit_result = None
        self.training_seasons = tuple(sorted(int(s) for s in train.season_id.unique()))
        weights = None
        if self.half_life is not None:
            weights = time_weights(train.season_id, max(self.training_seasons) + 1, self.half_life)
        self.fit_result = fit_ordered_logit(
            train.elo_diff.to_numpy(float) / RATING_SCALE, train.outcome.to_numpy(), weights
        )

    def predict(self, games: pd.DataFrame) -> NDArray[np.float64]:
        """Use only prematch raw Elo differences, preserving the supplied row order."""
        if self.fit_result is None:
            raise RuntimeError("Fit the converter before prediction")
        if (
            self.half_life is not None
            and not games.season_id.eq(max(self.training_seasons) + 1).all()
        ):
            raise ValueError("Time weights are for chronological folds, not LOSO")
        return ordered_probabilities(self.fit_result, games.elo_diff.to_numpy(float) / RATING_SCALE)
