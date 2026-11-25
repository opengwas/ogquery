"""ogld#4's contract: read LD Stores; LD matrices with NaN and fill.

Against the Test Store, which holds the Test Panel's reference r at k = 10 (support.py).
Expected values are the reference r signed to A1, never what the reader computes. Every
test calls ogld first; the whole-store matrices are read once per worker and cached.

From opengwasdb's `tests/ld/contract/test_store.py` at `c8ebe4ca` (ogquery#2): its reads. Its
tests of writing and validating a store, and of `ogld validate` and `info`, are build code's
and stay there (ADR 0002).
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import support
import zarr
from support import alids, panel_data, passing_rows, reference, reference_matrix

import ogquery.ld
from ogquery.ld.format import RefusalReason


def two(store: ogquery.ld.LDStore, i: int, j: int) -> tuple[float, str]:
    """The store's r and answer kind for Test Panel rows i and j."""
    m = store.matrix(alids([i, j]))
    assert m.absent == ()
    return float(m.r[0, 1]), str(m.kinds()[0, 1])


@cache
def opened() -> ogquery.ld.LDStore:
    return ogquery.ld.open(support.TEST_STORE)


@cache
def whole_store(fill: float | None = None) -> ogquery.ld.LDMatrix:
    """Every Test Store variant's matrix, both chromosomes, in Test Panel order."""
    store = opened()
    return store.matrix(alids(passing_rows()), fill=fill)


def expected() -> np.ndarray:
    return reference_matrix(support.K, tuple(int(row) for row in passing_rows()))


def with_provenance(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of the Test Store whose provenance `edit` has changed."""
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    edit(doc["attributes"]["ld_store"])
    meta.write_text(json.dumps(doc), encoding="utf-8")
    return copy


# ------------------------------------------------------------------ opening


def test_the_test_store_records_its_panel_and_its_distances() -> None:
    p = opened().provenance
    assert p["format_version"] == "0.2.0"
    assert p["panel"]["n"] == support.N_SAMPLES
    assert p["adjustment"]["k"] == support.K
    assert p["adjustment"]["covariates"] == [f"PC{i}" for i in range(1, 11)]
    assert p["t_far"] == support.T_FAR
    assert p["far_code_floor"] == support.FLOOR
    assert p["t_partner"] == support.T_PARTNER
    assert p["partner_code_floor"] == support.PARTNER_FLOOR
    assert p["counts"]["partner_entries"] > 0
    assert p["encoding"] == {
        "r_dtype": "int16",
        "r_scale": support.R_SCALE,
        "r_reserved_code": -32768,
        "rounding": "half_to_even",
    }
    assert (p["near_bp"], p["far_bp"]) == (3_000_000, 10_000_000)
    assert p["source"]["kind"] == "genotypes"
    assert p["n_variants"] == len(passing_rows())


def test_a_reader_refuses_a_format_version_it_does_not_implement(tmp_path: Path) -> None:
    opened()

    def future(p: dict[str, Any]) -> None:
        p["format_version"] = "9.0.0"

    with pytest.raises(ValueError, match=r"9\.0\.0"):
        ogquery.ld.open(with_provenance(tmp_path, future))


def test_a_0_1_0_store_is_read_as_it_was_before_the_partner_rows(tmp_path: Path) -> None:
    # ld-store §13: the bump lands with the writer, so a store published before the Partner Rows is
    # still read, from its Near Band, its Proxy Index and its Far Pairs.
    def earlier(p: dict[str, Any]) -> None:
        p["format_version"] = "0.1.0"

    location = with_provenance(tmp_path, earlier)
    shutil.rmtree(location / "partners")
    opened_early = ogquery.ld.open(location)
    assert opened_early.provenance["format_version"] == "0.1.0"
    rows = alids(passing_rows()[:20])
    assert opened_early.matrix(rows).r.shape == (20, 20)


def test_the_test_store_reads_the_same_over_http(served_store: str) -> None:
    # ld-store §1: a local path or an HTTP(S) URL. A static Range-serving server, as opengwas/ogld#9
    # may use.
    remote = ogquery.ld.open(served_store)
    assert remote.provenance == opened().provenance
    some = alids(passing_rows()[:40])
    np.testing.assert_array_equal(remote.matrix(some).r, opened().matrix(some).r)


def test_every_array_decodes_as_zarr_over_http(served_store: str) -> None:
    from ogquery.ld.store.location import ChunkCache, open_group

    cache = ChunkCache()
    remote = open_group(served_store, cache=cache)
    local = zarr.open_group(support.TEST_STORE, mode="r")
    arrays = [(p, a) for p, a in local.members(max_depth=None) if isinstance(a, zarr.Array)]
    assert len(arrays) > 20
    for path, array in arrays:
        served = remote[path]
        assert isinstance(served, zarr.Array)
        np.testing.assert_array_equal(cache.take(served, np.arange(array.shape[0])), array[:])


# ------------------------------------------------------------------ pair answers (ld-store §8)


def test_a_pair_exactly_3_mb_apart_is_the_near_band_s_exact_r() -> None:
    store = opened()
    pair = reference().where(reference().distance == 3_000_000, "a pair exactly 3 Mb apart")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


def test_a_pair_beyond_3_mb_below_t_far_is_nan_and_the_bound() -> None:
    store = opened()
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    pair = ref.where(far & (np.abs(ref.code) < support.FLOOR), "a 3–10 Mb pair below t_far")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert math.isnan(r), "an unstored pair is NaN, never a number"
    assert kind == "bound"


def test_a_far_pair_is_stored_and_reads_back_its_r() -> None:
    store = opened()
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    pair = ref.where(far & (np.abs(ref.code) >= support.FLOOR), "a real Far Pair at k = 10")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


def a_pair_apart(distance: int) -> tuple[int, int] | None:
    """Two passing Test Panel rows on one chromosome exactly `distance` bp apart, if any."""
    v = panel_data().variants()
    rows = passing_rows()
    pos, chrom = v.pos[rows].astype(np.int64), v.chrom[rows]
    gap = (pos[None, :] - pos[:, None] == distance) & (chrom[None, :] == chrom[:, None])
    found = np.argwhere(gap)
    return (int(rows[found[0][0]]), int(rows[found[0][1]])) if len(found) else None


def test_exactly_10_mb_is_measured_and_one_base_more_is_not() -> None:
    store = opened()
    at = reference().where(reference().distance == 10_000_000, "a pair exactly 10 Mb apart")
    assert two(store, int(at.row_i[0]), int(at.row_j[0]))[1] in {"stored", "bound"}
    beyond = a_pair_apart(10_000_001)
    assert beyond is not None, "the Test Panel must hold a pair 10,000,001 bp apart"
    r, kind = two(store, *beyond)
    assert math.isnan(r)
    assert kind == "not_measured"


def test_another_chromosome_is_not_measured() -> None:
    store = opened()
    v = panel_data().variants()
    rows = passing_rows()
    chr21 = rows[v.chrom[rows] == "21"]
    chr22 = rows[v.chrom[rows] == "22"]
    assert len(chr21), "the Test Panel must hold chr21"
    assert len(chr22), "the Test Panel must hold chr22"
    r, kind = two(store, int(chr21[0]), int(chr22[0]))
    assert math.isnan(r)
    assert kind == "not_measured"


def test_two_records_at_one_position_are_a_near_band_pair() -> None:
    store = opened()
    pair = reference().where(reference().distance == 0, "two records at one position")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


# ------------------------------------------------------------------ matrices, NaN and fill


def test_the_whole_store_reads_back_within_half_a_step() -> None:
    # The round trip: every pair of both chromosomes against the reference r, signed to A1.
    m = whole_store()
    ref = expected()
    stored = m.kinds() == "stored"
    assert stored.sum() > 1000, "the store must hold many pairs for this to mean anything"
    assert not np.isnan(ref[stored]).any(), "every stored pair is in the reference"
    assert np.all(np.abs(m.r[stored] - ref[stored]) <= support.HALF_STEP + support.SLACK)
    assert np.isnan(m.r[~stored]).all(), "every pair the store does not hold is NaN"


def test_every_stored_cell_is_exactly_what_the_codes_say() -> None:
    m = whole_store()
    ref = expected()
    stored = ~np.isnan(m.r)
    np.fill_diagonal(stored, False)
    assert stored.any()
    np.testing.assert_array_equal(m.r[stored], support.stored_at(ref[stored]))


def test_unstored_cells_are_nan_and_a_fill_replaces_exactly_them() -> None:
    plain = whole_store()
    kinds = plain.kinds()
    bound, nowhere = kinds == "bound", kinds == "not_measured"
    assert bound.any(), "the matrix must hold a Bound cell for this to mean anything"
    assert nowhere.any(), "the matrix must hold a Not Measured cell for this to mean anything"
    unstored = bound | nowhere
    assert np.isnan(plain.r[unstored]).all()
    assert not np.isnan(plain.r[~unstored]).any()
    # A non-zero fill, so a matrix that writes 0 for any fill cannot pass.
    filled = whole_store(fill=-9.0)
    assert filled.fill == -9.0
    assert (filled.r[unstored] == -9.0).all()
    np.testing.assert_array_equal(filled.r[~unstored], plain.r[~unstored])
    with pytest.raises(ValueError, match="fill"):
        filled.kinds()


def test_absent_alids_are_listed_and_are_not_rows() -> None:
    store = opened()
    v = panel_data().variants()
    failing = np.flatnonzero(~v.passes_filters)
    assert len(failing), "the Test Panel must hold variants the filters remove"
    present = alids(passing_rows()[:3])
    asked = [present[0], *alids(failing[:1]), "not-an-alid", *present[1:]]
    m = store.matrix(asked)
    assert m.absent == (asked[1], "not-an-alid")
    assert list(m.variants.alid) == present
    assert m.r.shape == (3, 3)


# ------------------------------------------------------------------ regions


def test_a_region_is_the_matrix_of_its_variants_in_genomic_order() -> None:
    store = opened()
    v = panel_data().variants()
    rows = passing_rows()
    middle = rows[len(rows) // 2]
    centre, chrom = int(v.pos[middle]), str(v.chrom[middle])
    region = store.region(chrom, centre, 2_000_000)
    offset = np.abs(v.pos[rows].astype(np.int64) - centre)
    inside = rows[(v.chrom[rows] == chrom) & (offset <= 2_000_000)]
    assert len(inside) > 1, "the region must hold variants"
    assert list(region.variants.alid) == alids(inside)
    np.testing.assert_array_equal(region.r, store.matrix(alids(inside)).r)


def test_a_region_is_at_most_10_mb_wide() -> None:
    store = opened()
    store.region("22", 30_000_000, 5_000_000)
    with pytest.raises(ogquery.ld.Refusal) as wide:
        store.region("22", 30_000_000, 5_000_001)
    assert wide.value.reason is RefusalReason.REGION_OVER_FAR
    with pytest.raises(ogquery.ld.Refusal) as negative:
        store.region("22", 30_000_000, -1)
    assert negative.value.reason is RefusalReason.INVALID


# ------------------------------------------------------------------ writing


def test_a_lost_chunk_is_refused_never_read_as_zero(tmp_path: Path) -> None:
    opened()
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    shards = sorted((copy / "near" / "22" / "r" / "c").iterdir())
    assert shards, "the Test Store's chr22 Near Band must have a shard to lose"
    shards[0].unlink()
    v = panel_data().variants()
    rows = passing_rows()[v.chrom[passing_rows()] == "22"]
    with pytest.raises(ValueError, match=r"-32768|reserved"):
        ogquery.ld.open(copy).matrix(alids(rows[:50]))
