"""Shared unit-test fixtures.

Every unit test starts with no release open: the registry `open_release` keeps of the forms
opened in this process (`ogquery.format.record_opened`) is process-wide, so one test's
releases would otherwise decide whether the next one's open warns `MixedCpraNormalForms`.

A 0.4.0 fixture release is built once per session, from the bundle release it names
(`format_0_4.as_0_4`), under a directory no test writes to (`built_0_4`). A test that mutates
a release copies it first (`copy_built_0_4`).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from contract_reader import RELEASES
from format_0_4 import as_0_4

from ogquery.format import _forget_opened_forms


@pytest.fixture(autouse=True)
def _no_release_open() -> Iterator[None]:
    _forget_opened_forms()
    yield
    _forget_opened_forms()


@pytest.fixture(scope="session")
def built_0_4(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """`built_0_4(name)`: the 0.4.0 copy of bundle release `name`, built on first use and
    shared by every test of the session. **Read only**: never edit it; copy it."""
    root = tmp_path_factory.mktemp("built_0_4")
    built: dict[str, Path] = {}

    def build(name: str) -> Path:
        if name not in built:
            built[name] = as_0_4(Path(shutil.copytree(RELEASES / name, root / name)))
        return built[name]

    return build


@pytest.fixture
def copy_built_0_4(
    built_0_4: Callable[[str], Path], tmp_path: Path
) -> Callable[[str], Path]:
    """`copy_built_0_4(name)`: a private copy of `built_0_4(name)`, for a test that edits it."""

    def copy(name: str) -> Path:
        return Path(shutil.copytree(built_0_4(name), tmp_path / f"{name}-copy"))

    return copy
