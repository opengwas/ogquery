"""One root read for the kind dispatch, and a genotype reader that composes, not inherits
(opengwas/ogld#91).

The read parts of opengwasdb `964ac1b2`'s `tests/ld/test_reader_kind.py` (opengwas/ogquery#11),
on the committed Genotype Store. Validation's root read is opengwasdb's; an LD Store's open is
`test_open.py`'s.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
import support
from support import TEST_STORE

import ogquery.ld
from ogquery.ld.query.reader import Reader
from ogquery.ld.store import StoreReader
from ogquery.ld.store import location as open_module
from ogquery.ld.store.variants import VariantTable

pytestmark = pytest.mark.genotype


def count_root_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The root openings `open_group` makes, counted where zarr is first asked for the group."""
    reads: list[Path] = []
    original = open_module._opened

    def counted(path: str | Path) -> object:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(open_module, "_opened", counted)
    return reads


def test_an_open_reads_a_genotype_store_s_root_once(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = count_root_reads(monkeypatch)
    ogquery.ld.open(genotype_store)
    assert len(reads) == 1, f"a Genotype Store open reads its root metadata {len(reads)} times"


def test_the_genotype_reader_composes_the_variant_table_and_store(genotype_store: Path) -> None:
    from ogquery.ld.genotypes.reader import GenotypeReader

    reader = ogquery.ld.open(genotype_store)._reader
    assert isinstance(reader, GenotypeReader)
    assert isinstance(reader, Reader)
    assert not isinstance(reader, StoreReader), "a genotype reader is not an LD Store reader"
    assert isinstance(reader._variant_table, VariantTable)
    assert reader._opened.root is reader._variant_table._opened.root
    # The inherited members that meant nothing here are gone, not silently wrong.
    assert not hasattr(reader, "r_of_pairs")
    assert not hasattr(reader, "frame_cache_bytes")
    assert not hasattr(reader, "has_partner_rows")
    # And an argument that had no meaning on this kind is refused, not ignored.
    with pytest.raises(TypeError):
        GenotypeReader.open(genotype_store, frame_cache_bytes=1)  # type: ignore[call-arg]


def test_a_store_closes_its_reader_and_its_pgen_handles(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ogquery.ld.genotypes.pgen import Genotypes
    from ogquery.ld.genotypes.reader import GenotypeReader

    closed: list[object] = []
    original_close = Genotypes.close

    def counted_close(self: Genotypes) -> None:
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(Genotypes, "close", counted_close)
    names = support.alids(support.passing_rows()[:10])
    store = ogquery.ld.open(genotype_store)
    store.matrix(names)
    reader = store._reader
    assert isinstance(reader, GenotypeReader)
    handles = len(reader._handles._idle)
    assert handles >= 1, "the fixture must open at least one pgen handle"
    store.close()
    assert reader._closed
    assert reader._handles._idle == [], "a closed store keeps no idle pgen handle"
    assert len(closed) == handles, "close() closes every handle the reader kept"
    with pytest.raises(ogquery.ld.InputError):
        store.matrix(names)
    with closing(ogquery.ld.open(TEST_STORE)) as ld:
        assert ld.provenance["format_version"]
