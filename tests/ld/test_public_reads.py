"""What a store holds, read before any r (ogld#79; ogquery#2): `variants_in`, `count_in` and
`has`.

ogdb-api reached these through `LDStore._reader.rows_in` and `rows_of`. Expected values are the
Test Panel's: its passing rows, their positions, and their ALIDs as the tests spell them
(`spelling.py`), never what the reader computes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import support
from spelling import sorted_identity
from store_copies import copy_store, store_reads
from support import alids, panel_data, passing_rows

import ogquery.ld
from ogquery.alid import ReferenceFirstIndel


def expected_rows(chrom: str, start: int, stop: int) -> np.ndarray:
    """The Test Panel's passing rows on `chrom` with start ≤ position ≤ stop, in panel order,
    which is the Test Store's genomic order."""
    v = panel_data().variants()
    rows = passing_rows()
    keep = (v.chrom[rows] == chrom) & (v.pos[rows] >= start) & (v.pos[rows] <= stop)
    return rows[keep]


@pytest.mark.parametrize(
    ("chrom", "start", "stop"),
    [
        ("22", 16_000_000, 17_000_000),
        ("21", 1, 3_000_000_000),
        ("22", 16_190_010, 16_190_010),
        ("22", 1, 10),
    ],
)
def test_variants_in_is_the_panel_s_passing_rows_in_the_range(
    chrom: str, start: int, stop: int
) -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    found = store.variants_in(chrom, start, stop)
    rows = expected_rows(chrom, start, stop)
    v = panel_data().variants()
    assert found.alid.tolist() == alids(rows)
    assert found.position.tolist() == v.pos[rows].tolist()
    assert set(found.chrom.tolist()) <= {chrom}
    assert store.count_in(chrom, start, stop) == len(rows) == len(found)


def test_the_ranges_hold_something_and_their_bounds_are_inclusive() -> None:
    # The fixture's meaning first: a range of one position finds the variant at it.
    assert len(expected_rows("21", 1, 3_000_000_000)) > 400
    assert len(expected_rows("22", 16_000_000, 17_000_000)) > 5
    assert len(expected_rows("22", 16_190_010, 16_190_010)) == 1


def test_variants_in_carries_each_variant_s_alleles_as_the_store_spells_them() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    found = store.variants_in("22", 1, 3_000_000_000)
    v = panel_data().variants()
    rows = expected_rows("22", 1, 3_000_000_000)
    formed = [sorted_identity("22", int(v.pos[i]), str(v.ref[i]), str(v.alt[i])) for i in rows]
    assert found.a1.tolist() == [f.a1 for f in formed]
    assert found.a2.tolist() == [f.a2 for f in formed]
    assert np.isfinite(found.a1_frequency).all()


def test_variants_in_takes_the_chr_prefix_region_takes() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    plain = store.variants_in("22", 16_000_000, 20_000_000)
    assert store.variants_in("chr22", 16_000_000, 20_000_000).alid.tolist() == plain.alid.tolist()
    assert store.count_in("chr22", 16_000_000, 20_000_000) == len(plain)


def test_a_chromosome_the_store_does_not_hold_has_no_variants() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    assert len(store.variants_in("1", 1, 3_000_000_000)) == 0
    assert store.count_in("X", 1, 3_000_000_000) == 0


@pytest.mark.parametrize(
    ("chrom", "start", "stop", "message"),
    [
        ("22", 20_000_000, 19_999_999, "start is at most its stop"),
        ("22", 1.5, 10, "whole-number position"),
        ("22", 1, "10", "whole-number position"),
        ("22", True, 10, "whole-number position"),
        ("HLA-A", 1, 10, "not a canonical chromosome"),
    ],
)
def test_a_range_that_is_not_one_is_an_input_error(
    chrom: str, start: object, stop: object, message: str
) -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    with pytest.raises(ogquery.ld.InputError, match=message):
        store.variants_in(chrom, start, stop)  # type: ignore[arg-type]
    with pytest.raises(ogquery.ld.InputError, match=message):
        store.count_in(chrom, start, stop)  # type: ignore[arg-type]


#: The LD components a store holds beside its variant table (ld-store §1; partner-rows §5): the
#: Near Band, the Far Pairs, the Partner Rows, and a 0.1.0 store's Proxy Index.
COMPONENTS = ("near/", "far/", "partners/", "index/")


def test_the_variant_reads_open_no_ld_component() -> None:
    # They are what ogdb-api asks before a region's r is read (ogld#79): the variant table alone,
    # not one key of a component, its metadata included. Each read is a cold store's first.
    reads = {
        "count_in": lambda store: store.count_in("22", 1, 3_000_000_000),
        "variants_in": lambda store: len(store.variants_in("21", 1, 3_000_000_000)),
        "has": lambda store: int(store.has(alids(passing_rows()[:50])).sum()),
    }
    for name, read in reads.items():
        store = ogquery.ld.open(support.TEST_STORE)
        with store_reads() as keys:
            assert read(store) > 0, name
        assert [k for k in keys if k.startswith(COMPONENTS)] == [], name
        assert any(k.startswith("variants/") for k in keys), f"{name} reads the variant table"


def test_the_variant_reads_answer_with_the_near_band_s_metadata_gone(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    gone = sorted((copy / "near").rglob("zarr.json"))
    assert len(gone) >= 6, "both chromosomes' Near Band groups and arrays"
    for path in gone:
        path.unlink()
    whole, broken = ogquery.ld.open(support.TEST_STORE), ogquery.ld.open(copy)
    for chrom in ("21", "22"):
        assert broken.variants_in(chrom, 1, 3_000_000_000).columns().keys() == {
            "chrom", "position", "alid", "a1", "a2", "a1_frequency"
        }
        for column, value in broken.variants_in(chrom, 1, 3_000_000_000).columns().items():
            np.testing.assert_array_equal(
                value, whole.variants_in(chrom, 1, 3_000_000_000).columns()[column]
            )
        assert broken.count_in(chrom, 1, 3_000_000_000) == whole.count_in(chrom, 1, 3_000_000_000)
    asked = alids(np.arange(0, 1061, 5))
    assert broken.has(asked).tolist() == whole.has(asked).tolist()
    with pytest.raises(ValueError, match="near"):
        broken.region("22", 30_000_000, 1_000_000)  # the LD read still refuses, loudly


def test_has_is_true_exactly_for_the_variant_set() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    v = panel_data().variants()
    dropped = np.flatnonzero(~v.passes_filters)
    assert len(dropped) > 0, "the Test Panel must have rows the build drops"
    present = alids(passing_rows()[:5])
    absent = alids(dropped[:3])
    asked = [*present, *absent, "22:1:A:C", "not an alid", present[0]]
    assert store.has(asked).tolist() == [True] * 5 + [False] * 5 + [True]
    assert store.has([]).tolist() == []


def test_has_agrees_with_what_the_queries_report_absent() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    v = panel_data().variants()
    asked = alids(np.arange(0, len(v), 7))
    absent = set(store.matrix(asked).absent)
    assert absent, "some of the rows asked for must be absent"
    assert store.has(asked).tolist() == [a not in absent for a in asked]


def test_has_refuses_an_indel_spelt_reference_first() -> None:
    # As matrix, proxies and clump refuse it: the store holds that variant under its sorted ALID.
    store = ogquery.ld.open(support.TEST_STORE)
    v = panel_data().variants()
    rows = passing_rows()
    indel = next(int(i) for i in rows if len(v.ref[i]) != len(v.alt[i]))
    chrom, pos, a1, a2 = alids([indel])[0].split(":")
    with pytest.raises(ReferenceFirstIndel, match="reference first"):
        store.has([f"{chrom}:{pos}:{a2}:{a1}"])
