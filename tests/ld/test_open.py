"""`ogquery.ld.open` reads LD Stores, and dispatches a Genotype Store to its own reader (ogquery#2,
opengwas/ogquery#11).

A Genotype Store's answers are computed from genotypes with pgenlib, which only the `genotype`
extra installs (ADR 0002, amended): its root names its kind, and `open` hands it to
`ogquery.ld.genotypes`'s reader, never reading it as an LD Store. That reader's answers are
`test_genotype_answers.py`'s; a missing extra is `test_genotype_extra.py`'s.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import zarr
from serving import serve
from store_copies import TEST_STORE, copy_store

import ogquery.ld
from ogquery.ld.query.kind import GENOTYPE, open_kind
from ogquery.ld.store import location
from ogquery.opening import GenotypeExtraMissing


def test_a_genotype_store_goes_to_the_genotype_reader_not_the_ld_store_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without pgenlib the dispatch names the extra: it took the Genotype Store's branch.
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    root = zarr.open_group(str(tmp_path / "sidecar"), mode="w")
    root.attrs["genotype_store"] = {"format_version": "0.1.0"}
    with pytest.raises(GenotypeExtraMissing, match=r"Genotype Store.*ogquery\[genotype\]"):
        ogquery.ld.open(tmp_path / "sidecar")


GENOTYPE_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "genotype-store"


@pytest.fixture
def served_genotype_store() -> Iterator[str]:
    """The committed Genotype Store served over HTTP from this machine, as a URL. Only its root's
    metadata is read: the open is refused before anything else."""
    with serve(GENOTYPE_STORE) as url:
        yield url


class _Located:
    """An `os.PathLike[str]` that is not a `Path`: what `__fspath__` returns is the location."""

    def __init__(self, location: str) -> None:
        self._location = location

    def __fspath__(self) -> str:
        return self._location

    def __repr__(self) -> str:
        return "<_Located>"


@pytest.mark.http
@pytest.mark.parametrize("form", ["str", "pathlike"])
@pytest.mark.parametrize("scheme", ["http", "HTTP"])
def test_a_genotype_store_over_http_is_refused_as_local_only(
    served_genotype_store: str, monkeypatch: pytest.MonkeyPatch, scheme: str, form: str
) -> None:
    # Before the `genotype` extra is checked: installing it would not make a URL readable. A
    # PathLike is the location its `__fspath__` names, as the opener reads it.
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    url = scheme + served_genotype_store.removeprefix("http")
    location: str | os.PathLike[str] = url if form == "str" else _Located(url)
    with pytest.raises(ogquery.ld.InputError) as refused:
        ogquery.ld.open(location)
    message = str(refused.value)
    assert message.startswith(f"{url} is a Genotype Store, which is read from a local path only")
    assert "pgens" in message and "arrays" in message
    assert "ogquery[genotype]" not in message


def test_the_kind_seam_records_a_location_as_the_opener_reads_it(tmp_path: Path) -> None:
    # A `Path` is recorded as it always was, `str(path)`; any PathLike as its `__fspath__`.
    assert open_kind(TEST_STORE).location == str(TEST_STORE)
    assert open_kind(_Located(str(TEST_STORE))).location == str(TEST_STORE)
    assert open_kind(str(TEST_STORE)).location == str(TEST_STORE)


def test_the_kind_seam_still_names_a_genotype_store(tmp_path: Path) -> None:
    # `ogquery.ld.open` dispatches on it, as opengwasdb's `open` did.
    root = zarr.open_group(str(tmp_path / "sidecar"), mode="w")
    root.attrs["genotype_store"] = {"format_version": "0.1.0"}
    kinded = open_kind(tmp_path / "sidecar")
    assert kinded.kind == GENOTYPE
    assert kinded.record() == {"format_version": "0.1.0"}


def test_the_reader_protocol_is_public() -> None:
    reader = ogquery.ld.open(TEST_STORE)
    assert isinstance(reader._reader, ogquery.ld.Reader)


def test_an_open_reads_an_ld_store_s_root_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # One read of the root's metadata, whichever kind it holds (opengwas/ogld#91): from
    # opengwasdb's `tests/ld/test_reader_kind.py`, whose validate and Genotype Store halves are
    # opengwasdb's. Counted where zarr is first asked for the group.
    reads: list[Path] = []
    original = location._opened

    def counted(path: str | Path) -> object:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(location, "_opened", counted)
    with ogquery.ld.open(TEST_STORE) as store:
        assert store.provenance["format_version"]
    assert len(reads) == 1, f"an LD Store open reads its root metadata {len(reads)} times"


def test_a_format_version_this_reader_does_not_implement_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    copy = copy_store(tmp_path / "store")
    root = zarr.open_group(str(copy), mode="r+")
    recorded = dict(root.attrs["ld_store"])
    recorded["format_version"] = "9.0.0"
    root.attrs["ld_store"] = recorded
    with pytest.raises(ogquery.ld.InputError, match=r"'9\.0\.0'; this ogquery\.ld reads"):
        ogquery.ld.open(copy)
