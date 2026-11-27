"""Fixtures the contract tests share."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from serving import serve
from support import TEST_STORE


@pytest.fixture
def served_store() -> Iterator[str]:
    """The Test Store's directory served over HTTP from this machine, as a URL."""
    with serve(TEST_STORE) as url:
        yield url
