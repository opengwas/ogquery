"""What a store holds, read before any r (ogld#79; ogquery#2): `variants_in`, `count_in` and
`has`.

ogdb-api reached these through `LDStore._reader.rows_in` and `rows_of`. Expected values are the
Test Panel's: its passing rows, their positions, and their ALIDs as the tests spell them
(`spelling.py`), never what the reader computes.
"""

from __future__ import annotations

import numpy as np
import pytest
import support
from spelling import sorted_identity
from store_copies import store_reads
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


def test_variants_in_reads_no_r() -> None:
    # It is what ogdb-api asks before a region's r is read (ogld#79): no chunk of any r array.
    store = ogquery.ld.open(support.TEST_STORE)
    with store_reads() as keys:
        assert store.count_in("22", 1, 3_000_000_000) > 0
        assert len(store.variants_in("21", 1, 3_000_000_000)) > 0
    assert [k for k in keys if "/r/c" in k or k.startswith("partners/")] == []
    assert any(k.startswith("variants/position") for k in keys), "the positions are read"


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
