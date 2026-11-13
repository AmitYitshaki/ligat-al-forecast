"""List tests affected by main...HEAD, index and working-tree changes; never run pytest.

Run from the repository root. PowerShell:

    $files = & .venv/Scripts/python.exe scripts/focused_tests.py
    if ($files) { & .venv/Scripts/python.exe -m pytest -m "not slow" $files }

Bash:

    mapfile -t files < <(python scripts/focused_tests.py)
    if ((${#files[@]})); then python -m pytest -m "not slow" "${files[@]}"; fi

Deleted files are excluded. With no affected tests, stdout is empty and exit is 0.
"""

import ast
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


def _path(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).as_posix()


def _imports(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in modules if name == "ligat" or name.startswith("ligat.")}


def select_focused_tests(
    changed_files: Iterable[str], test_sources: Mapping[str, str]
) -> list[str]:
    """Select existing tests using only changed paths and their source, without Git or I/O."""
    changed = {_path(name) for name in changed_files}
    if not changed:
        return []
    tests = {
        _path(name): source
        for name, source in test_sources.items()
        if _path(name).startswith("tests/")
        and PurePosixPath(_path(name)).name.startswith("test_")
        and PurePosixPath(_path(name)).suffix == ".py"
    }
    if changed & {"tests/conftest.py", "pyproject.toml"}:
        return sorted(tests)
    modules: set[str] = set()
    packages: set[str] = set()
    for name in changed:
        path = PurePosixPath(name)
        if path.parts[0] != "ligat" or path.suffix != ".py":
            continue
        if path.name == "__init__.py":
            packages.add(".".join(path.parts[:-1]))
        else:
            modules.add(".".join(path.with_suffix("").parts))
    selected = changed & tests.keys()
    if not modules and not packages:
        return sorted(selected)
    for name, source in tests.items():
        imports = _imports(source)
        if modules & imports or any(
            imported == package or imported.startswith(package + ".")
            for package in packages
            for imported in imports
        ):
            selected.add(name)
    return sorted(selected)


def _git_paths(root: Path, *args: str) -> list[str]:
    output = subprocess.check_output(["git", *args], cwd=root)
    return [name for name in output.decode("utf-8").split("\0") if name]


def main() -> None:
    """Print affected test paths, combining committed, staged and untracked work."""
    root = Path(__file__).resolve().parents[1]
    changed: set[str] = set()
    for args in (
        ("diff", "--name-only", "-z", "main...HEAD"),
        ("diff", "--name-only", "-z"),
        ("diff", "--cached", "--name-only", "-z"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
    ):
        changed.update(name for name in _git_paths(root, *args) if (root / name).is_file())
    if not changed:
        return
    candidates = _git_paths(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "tests"
    )
    sources = {
        name: (root / name).read_text(encoding="utf-8-sig")
        for name in candidates
        if PurePosixPath(name).name.startswith("test_")
        and PurePosixPath(name).suffix == ".py"
        and (root / name).is_file()
    }
    for name in select_focused_tests(changed, sources):
        print(name)


if __name__ == "__main__":
    main()
