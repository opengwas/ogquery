"""The walking skeleton's contract: the thinnest real path, and the Test Store it writes.

pgen → r at k = 0 → the Near Band → write → read → one pair answered. The skeleton's script,
`scripts/ld/make_test_store.py`, writes the Test Store from the Test Panel's `reference_r(10)`
into `tests/ld/data/test-panel/store/`; the ticket runs it, and no test writes there. Its
`write_test_store(k, out)` also gives ogld#6 a k = 0 store. From then on ogld#4–opengwas/ogld#6
build against real data in the format.

These tests go through the store seam, `StoreReader`, not the queries, which are ogld#4's
and ogld#6's. From opengwasdb's `tests/ld/contract/test_walking_skeleton.py` at `c8ebe4ca`
(ogquery#2); the two that run the script to write a store are build code's and stay there.
"""

from __future__ import annotations

import numpy as np
import support
import zarr
from support import alids, panel_data, passing_rows, reference

import ogquery.ld
from ogquery.ld.store import StoreReader


def near_pair_r(reader: StoreReader, k: int) -> tuple[float, float]:
    """A Near Band pair's r as the store reads it back, and as the reference has it."""
    ref = reference(k)
    pair = ref.where((ref.distance > 0) & (ref.distance <= 3_000_000), "a Near Band pair")
    rows = reader.rows_of(alids([pair.row_i[0], pair.row_j[0]]))
    assert (rows >= 0).all(), "both variants must be in the Test Store"
    return float(reader.pair_r(rows)[0, 1]), float(pair.r[0])


def test_the_test_store_opens_through_ogld_open() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    assert (store.t_far, store.near_bp, store.far_bp) == (support.T_FAR, 3_000_000, 10_000_000)


def test_one_near_band_pair_reads_back_its_reference_r() -> None:
    got, expected = near_pair_r(StoreReader.open(support.TEST_STORE), 10)
    assert abs(got - expected) <= support.HALF_STEP + support.SLACK


def test_pairs_the_store_does_not_hold_read_as_nan() -> None:
    reader = StoreReader.open(support.TEST_STORE)
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    bound = ref.where(far & (np.abs(ref.code) < support.FLOOR), "a 3–10 Mb pair below t_far")
    v = panel_data().variants()
    rows = passing_rows()
    pos, chrom = v.pos[rows].astype(np.int64), v.chrom[rows]
    apart = (pos[None, :] - pos[:, None] > 10_000_000) & (chrom[None, :] == chrom[:, None])
    assert apart.any(), "the Test Panel must hold a pair more than 10 Mb apart"
    a, b = np.argwhere(apart)[0]
    for i, j in ((int(bound.row_i[0]), int(bound.row_j[0])), (int(rows[a]), int(rows[b]))):
        found = reader.rows_of(alids([i, j]))
        assert (found >= 0).all(), "both variants must be in the Test Store"
        assert np.isnan(reader.pair_r(found)[0, 1]), f"rows {i} and {j}: unstored, so NaN"


def arrays(group: zarr.Group, prefix: str = "") -> dict[str, np.ndarray]:
    """Every array under a Zarr group, decoded, by path."""
    out: dict[str, np.ndarray] = {}
    for name, member in group.members():
        if isinstance(member, zarr.Array):
            out[prefix + name] = np.asarray(member[:])
        else:
            out |= arrays(member, f"{prefix}{name}/")
    return out


