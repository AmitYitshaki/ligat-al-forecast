"""Content-addressed, atomic JSON cache for the expensive development-only diagnostic."""

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]

from ligat.analysis.league_stats.core import _selection, _subset
from ligat.analysis.league_stats.poisson_check import DIAGNOSTIC_SEED, FIT_ITERATIONS, PSEUDOCOUNT
from ligat.analysis.league_stats.score_review import (
    ALGORITHM_VERSION,
    NULL_REPLICATES,
    NULL_SEED,
    calibrated_poisson_check,
)
from ligat.analysis.prepare import DEVELOPMENT_SEASONS, AnalysisTables


def cached_poisson_check(
    tables: AnalysisTables,
    cache_path: Path,
    seasons: Iterable[int] = DEVELOPMENT_SEASONS,
    *,
    replicates: int = NULL_REPLICATES,
    seed: int = NULL_SEED,
    observed_seed: int = DIAGNOSTIC_SEED,
    iterations: int = FIT_ITERATIONS,
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse only identical data, parameters, dependency versions and algorithm sources.

    JSON avoids executable pickle caches. The caller chooses an ignored derived path;
    neither clean data nor production model artifacts are touched. Invalid caches are
    recomputed. A temporary sibling and atomic replace prevent partially written caches.
    """
    selected = _selection(seasons)
    games = _subset(tables.match_table, selected).sort_index()[
        ["season_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    ]
    source_root = Path(__file__).parent / "league_stats"
    digest = hashlib.sha256(pd.util.hash_pandas_object(games, index=True).to_numpy().tobytes())
    for filename in ("poisson_check.py", "score_review.py"):
        digest.update((source_root / filename).read_bytes())
    parameters = dict(
        seasons=selected,
        replicates=replicates,
        seed=seed,
        observed_seed=observed_seed,
        iterations=iterations,
        pseudocount=PSEUDOCOUNT,
        version=ALGORITHM_VERSION,
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
    )
    digest.update(json.dumps(parameters, sort_keys=True).encode())
    key = digest.hexdigest()
    try:
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved["key"] == key:
            summary = pd.DataFrame(**saved["summary"])
            samples = pd.DataFrame(**saved["samples"])
            summary.attrs.update(saved["attrs"], cache_hit=True)
            return summary, samples
    except (OSError, ValueError, KeyError, TypeError):
        pass
    started = time.monotonic()
    summary, samples = calibrated_poisson_check(
        tables,
        selected,
        replicates=replicates,
        seed=seed,
        observed_seed=observed_seed,
        iterations=iterations,
        progress=progress,
    )
    summary.attrs.update(
        cache_key=key, simulation_seconds=time.monotonic() - started, cache_hit=False
    )
    payload = dict(
        key=key,
        parameters=parameters,
        attrs=summary.attrs,
        summary=summary.to_dict(orient="split"),
        samples=samples.to_dict(orient="split"),
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary, samples
