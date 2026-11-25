"""`ogquery.ld.open` reads LD Stores, and refuses a Genotype Store naming opengwasdb (ogquery#2).

A Genotype Store's answers are computed from genotypes with pgenlib, which ogquery never imports
(ADR 0002): its root names its kind, and `open` refuses it rather than read it as an LD Store.
opengwasdb's own `open` reads it, through the same `ogquery.ld.Reader` protocol.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zarr

import ogquery.ld
from ogquery.ld.query.kind import GENOTYPE, open_kind


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
    reader = ogquery.ld.open(Path(__file__).resolve().parent / "data" / "test-panel" / "store")
    assert isinstance(reader._reader, ogquery.ld.Reader)
