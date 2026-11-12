"""Focused selection is pure; these tests never invoke Git or read project data."""

import pytest

from scripts.focused_tests import select_focused_tests


@pytest.mark.parametrize(
    "statement",
    [
        "import ligat.features.form",
        "import ligat.features.form as form",
        "from ligat.features import form",
        "from ligat.features.form import FormBlock",
    ],
)
def test_changed_module_import_forms(statement: str) -> None:
    sources = {"tests/test_form.py": statement, "tests/test_other.py": "import ligat.features.elo"}
    assert select_focused_tests(["ligat/features/form.py"], sources) == ["tests/test_form.py"]


@pytest.mark.parametrize(
    "statement",
    [
        "import ligat.features",
        "import ligat.features.form",
        "from ligat import features",
        "from ligat.features.form import FormBlock",
    ],
)
def test_changed_package_includes_package_and_descendants(statement: str) -> None:
    sources = {
        "tests/test_form.py": statement,
        "tests/test_other.py": "import ligat.features_extra",
    }
    assert select_focused_tests(["ligat/features/__init__.py"], sources) == ["tests/test_form.py"]


@pytest.mark.parametrize("changed", ["tests/conftest.py", "pyproject.toml"])
def test_shared_configuration_selects_all_tests(changed: str) -> None:
    sources = {"tests/test_z.py": "pass", "tests/test_a.py": "pass", "tests/helper.py": "pass"}
    assert select_focused_tests([changed], sources) == ["tests/test_a.py", "tests/test_z.py"]


def test_changed_tests_are_sorted_and_deduplicated() -> None:
    sources = {"tests/test_z.py": "pass", "tests/test_a.py": "pass"}
    assert select_focused_tests(
        ["tests/test_z.py", "tests/test_a.py", "tests/test_z.py"], sources
    ) == ["tests/test_a.py", "tests/test_z.py"]


@pytest.mark.parametrize("changed", ["ligat/features/form_extra.py", "docs/README.md"])
def test_unrelated_change_selects_nothing(changed: str) -> None:
    sources = {"tests/test_form.py": "from ligat.features.form import FormBlock"}
    assert select_focused_tests([changed], sources) == []


def test_text_mentions_do_not_count_as_imports() -> None:
    sources = {
        "tests/test_form.py": (
            '"from ligat.features.form import FormBlock"\n'  # Keep the source lines distinct.
            "# import ligat.features.form"
        )
    }
    assert select_focused_tests(["ligat/features/form.py"], sources) == []


def test_no_changes_selects_nothing() -> None:
    assert select_focused_tests([], {"tests/test_form.py": "import ligat.features.form"}) == []


def test_deleted_test_without_source_is_excluded() -> None:
    assert select_focused_tests(["tests/test_deleted.py"], {"tests/test_kept.py": "pass"}) == []


def test_windows_paths_are_normalized() -> None:
    assert select_focused_tests(
        [r"ligat\features\form.py"], {r"tests\test_form.py": "import ligat.features.form"}
    ) == ["tests/test_form.py"]
