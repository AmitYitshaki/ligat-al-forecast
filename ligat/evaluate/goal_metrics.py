"""Poisson likelihood and independent-goal H/D/A conversion for EXP-002."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]
from scipy.stats import poisson  # type: ignore[import-untyped]

from ligat.evaluate.metrics import validate_probabilities


def poisson_nll(goals: ArrayLike, rates: ArrayLike) -> NDArray[np.float64]:
    """Return λ − y log λ + log(y!), one real Poisson NLL per team-game row."""
    y, lam = np.asarray(goals, float), np.asarray(rates, float)
    if (
        y.shape != lam.shape
        or y.ndim != 1
        or not len(y)
        or not np.isfinite(y).all()
        or (y < 0).any()
        or (y % 1 != 0).any()
        or not np.isfinite(lam).all()
        or (lam <= 0).any()
    ):
        raise ValueError("Poisson scores require aligned integer goals and positive finite rates")
    return np.asarray(lam - y * np.log(lam) + gammaln(y + 1), float)


def poisson_outcomes(
    rates: ArrayLike, *, max_goals: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize the 0..max_goals square; return H/D/A and omitted joint mass per game."""
    lam = np.asarray(rates, float)
    if (
        lam.ndim != 2
        or lam.shape[1] != 2
        or not len(lam)
        or not np.isfinite(lam).all()
        or (lam <= 0).any()
        or isinstance(max_goals, bool)
        or not isinstance(max_goals, int)
        or max_goals < 1
    ):
        raise ValueError("Expected positive n×2 home/away rates and a positive integer grid limit")
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lam[:, :1])
    pa = poisson.pmf(goals[None, :], lam[:, 1:])
    matrix = ph[:, :, None] * pa[:, None, :]
    mass = matrix.sum(axis=(1, 2))
    if not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("Poisson grid has no representable retained probability")
    matrix /= mass[:, None, None]
    p = np.column_stack(
        (
            np.tril(matrix, -1).sum(axis=(1, 2)),
            matrix.trace(axis1=1, axis2=2),
            np.triu(matrix, 1).sum(axis=(1, 2)),
        )
    )
    return validate_probabilities(p), np.maximum(0, 1 - mass)
