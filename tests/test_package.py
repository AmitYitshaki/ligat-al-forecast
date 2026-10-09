"""Verify that the installed project exposes its package skeleton."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "name",
    ["ligat", "scrape", "data", "features", "models", "simulate", "analysis", "viz", "evaluate"],
)
def test_package_import(name: str) -> None:
    package = name if name == "ligat" else f"ligat.{name}"
    assert import_module(package).__name__ == package
