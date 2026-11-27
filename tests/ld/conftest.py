"""The LD tests' settings: the helper directories on `sys.path`, and the Hypothesis profiles.

The LD tests came from opengwasdb's `tests/ld/` at `c8ebe4ca` (ogquery#2). They import their
helpers by bare name (`import testdata`, `from support import …`), so this directory and
`contract/` go at the end of `sys.path`. `ci` is what CI loads (`HYPOTHESIS_PROFILE=ci`): about
50 examples a property, derandomised so a red run reproduces; `dev`, the local default, is
random and keeps exploring.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import settings

for _directory in (".", "contract"):
    _path = str((Path(__file__).resolve().parent / _directory).resolve())
    if _path not in sys.path:
        sys.path.append(_path)

settings.register_profile(
    "ci", max_examples=50, deadline=timedelta(milliseconds=200), derandomize=True
)
settings.register_profile("dev", max_examples=100, deadline=timedelta(milliseconds=200))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

#: The fixtures that serve a store over HTTP: a test that takes one reads through the `http`
#: extra, so it carries the `http` mark, as the tests that serve a store themselves do by hand.
HTTP_FIXTURES = frozenset({"served_store", "served"})
#: The fixture that places the Genotype Store: a test that takes one reads it through the
#: `genotype` extra, so it carries the `genotype` mark.
GENOTYPE_FIXTURES = frozenset({"genotype_store"})
#: Why a `genotype` test does not run on Windows: the extra does not install there.
NO_GENOTYPE_EXTRA = (
    "pgenlib publishes no Windows wheel, so the `genotype` extra, and a Genotype Store read, "
    "is not tested on Windows (ADR 0002, amended)"
)


@pytest.fixture(scope="session")
def genotype_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Test Panel's Genotype Store, placed once for the session beside the inputs it pins,
    whose mtimes it records (`genotype_answers.place`): an open refuses a changed one, and a
    checkout keeps none. No test reads `data/` in place: it may not change a byte there."""
    import genotype_answers

    return genotype_answers.place(tmp_path_factory.mktemp("genotype-store"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """The hook sees every collected item, the other directories' too, so it marks only ours."""
    here = Path(__file__).resolve().parent
    for item in items:
        ours = here in Path(item.path).resolve().parents
        if ours and HTTP_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.http)
        if ours and GENOTYPE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.genotype)
        if ours and sys.platform == "win32" and item.get_closest_marker("genotype"):
            item.add_marker(pytest.mark.skip(reason=NO_GENOTYPE_EXTRA))
