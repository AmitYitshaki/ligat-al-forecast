"""Validated, atomic storage for the clean Parquet datasets."""

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]

from ligat.data.schemas import SCHEMAS


class Repository:
    """Keep Parquet access and schema enforcement behind one boundary."""

    def __init__(self, root: Path = Path("data/clean")) -> None:
        self.root = root
        if "raw" in (part.casefold() for part in root.resolve().parts):
            raise ValueError("Clean repository cannot write into raw storage")

    def read(self, name: str) -> pd.DataFrame:
        """Read and validate a known dataset, preserving timezone/category types."""
        schema = SCHEMAS[name]
        frame = pd.read_parquet(self.root / f"{name}.parquet")
        schema.validate(frame, lazy=True)
        return frame

    def write(self, name: str, frame: pd.DataFrame) -> None:
        """Validate before publishing; failed writes leave existing data intact."""
        SCHEMAS[name].validate(frame, lazy=True)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{name}.parquet"
        temporary = self.root / f".{name}.{uuid4().hex}.parquet"
        try:
            frame.to_parquet(temporary, index=False, engine="pyarrow")
            # Validate the actual serialized artifact before replacing its predecessor.
            SCHEMAS[name].validate(pd.read_parquet(temporary), lazy=True)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def read_filtered(self, name: str, filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        """Push filters into Parquet loading so analysis never loads held-out rows."""
        if not filters:
            raise ValueError("Filtered repository reads require an explicit selection")
        schema = SCHEMAS[name]
        frame = pd.read_parquet(self.root / f"{name}.parquet", filters=filters, engine="pyarrow")
        schema.validate(frame, lazy=True)
        return frame
