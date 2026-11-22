"""Training-only journals and explicitly requested atomic journal publication."""

import gzip
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


class TrainingJournal:
    """In-memory journal; fit never writes files and callers receive defensive copies."""

    record_training_log: bool = True
    _training_log: dict[str, Any] | None = None

    @property
    def training_log(self) -> dict[str, Any]:
        if self._training_log is None:
            raise RuntimeError("Fit or load before reading training_log")
        return deepcopy(self._training_log)


class ObjectiveJournal:
    """Observe accepted SciPy iterations without altering objective arithmetic."""

    def __init__(self, objective: Any) -> None:
        self.objective = objective
        self.losses: list[float] = []
        self.x: Any = None
        self.value = 0.0

    def __call__(self, x: Any, *args: Any) -> Any:
        result = self.objective(x, *args)
        self.x, self.value = x.copy(), float(result[0])
        return result

    def callback(self, x: Any) -> None:
        if self.x is None or not np.array_equal(self.x, x):
            raise RuntimeError("Optimizer callback did not match the evaluated training iterate")
        self.losses.append(self.value)


def atomic_bytes(path: Path, contents: bytes) -> None:
    """Publish a complete artifact after flushing its unique temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def journal_bytes(value: Any, *, compressed: bool = False) -> bytes:
    """Canonical JSON, optionally gzip with a deterministic header."""
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return gzip.compress(raw, mtime=0) if compressed else raw


def write_training_log(model: TrainingJournal, path: Path) -> None:
    """Explicit manual/notebook entry point; .gz selects gzip compression."""
    atomic_bytes(path, journal_bytes(model.training_log, compressed=path.suffix == ".gz"))
