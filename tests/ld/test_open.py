"""`ogquery.ld.open` reads LD Stores, and refuses a Genotype Store naming opengwasdb (ogquery#2).

A Genotype Store's answers are computed from genotypes with pgenlib, which ogquery never imports
(ADR 0002): its root names its kind, and `open` refuses it rather than read it as an LD Store.
opengwasdb's own `open` reads it, through the same `ogquery.ld.Reader` protocol.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zarr
from store_copies import TEST_STORE, copy_store

import ogquery.ld
from ogquery.ld.query.kind import GENOTYPE, open_kind
from ogquery.ld.store import location


def test_a_genotype_store_is_refused_naming_opengwasdb(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "sidecar"), mode="w")
    root.attrs["genotype_store"] = {"format_version": "0.1.0"}
    with pytest.raises(ogquery.ld.InputError, match=r"Genotype Store.*opengwasdb\.ld\.open"):
        ogquery.ld.open(tmp_path / "sidecar")


def test_the_kind_seam_still_names_a_genotype_store(tmp_path: Path) -> None:
    # opengwasdb dispatches on it to its own reader.
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
