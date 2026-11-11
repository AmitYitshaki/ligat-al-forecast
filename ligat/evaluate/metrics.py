"""Per-game H/D/A scores: home=0, draw=1, away=2; lower is better."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

EPS = float(np.finfo(float).eps)
PROBABILITY_TOLERANCE = 1e-9


def validate_probabilities(
    probabilities: ArrayLike, n_matches: int | None = None
) -> NDArray[np.float64]:
    """Require finite, nonnegative n×3 probabilities summing to 1 within 1e-9.

    Validation never normalizes, clips or otherwise repairs a predictor's output.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("Expected a nonempty n×3 H/D/A probability matrix")
    if n_matches is not None and len(p) != n_matches:
        raise ValueError("Probability rows do not match the number of games")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(p.sum(axis=1), 1, atol=PROBABILITY_TOLERANCE, rtol=0):
        raise ValueError("Probability rows must sum to 1 within 1e-9")
    return p


def validate_outcomes(outcomes: ArrayLike, n_matches: int) -> NDArray[np.int64]:
    """Validate one integer target per match in the ordered H/D/A coding 0/1/2."""
    values = np.asarray(outcomes)
    if values.ndim != 1 or len(values) != n_matches or not n_matches:
        raise ValueError("Expected one outcome per game")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Outcomes must be finite numeric codes 0, 1 or 2")
    if not np.isin(values, (0, 1, 2)).all():
        raise ValueError("Outcomes must be integer codes 0, 1 or 2")
    return values.astype(np.int64)


def log_loss(
    outcomes: ArrayLike, probabilities: ArrayLike, *, eps: float = EPS
) -> NDArray[np.float64]:
    """Return -log(max(p_observed, eps)) per match, matching notebook 02 exactly.

    Only the observed probability is lower-clipped; rows are not renormalized.
    """
    p = validate_probabilities(probabilities)
    y = validate_outcomes(outcomes, len(p))
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("eps must be finite and strictly between 0 and 1")
    return -np.log(np.maximum(p[np.arange(len(y)), y], eps))


def rps(outcomes: ArrayLike, probabilities: ArrayLike) -> NDArray[np.float64]:
    """Return ordered H/D/A ranked probability scores, divided by K−1=2."""
    p = validate_probabilities(probabilities)
    y = validate_outcomes(outcomes, len(p))
    cumulative = p.cumsum(axis=1)[:, :-1]
    observed = np.eye(3)[y].cumsum(axis=1)[:, :-1]
    return ((cumulative - observed) ** 2).sum(axis=1) / 2
