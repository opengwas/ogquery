"""The Far Pairs beside the Proxy Index (ogld#31; ld-store §5, §8, §10; ADR 0093).

Each chromosome's `far/<chrom>/` holds its pairs more than near_bp and at most far_bp apart whose
code reaches the far code floor, once each, in the earlier variant's row: the pairs a Proxy Index
held beyond near_bp, in their earlier rows. `matrix` and `region` read those pairs from the Far
Pairs alone; from opengwas/ogld#72 no store holds an index at all, and a store that still carries a
leftover one answers the same (ld-store §12).

Every expected value here is read from the store's own arrays with zarr, from the Test Panel's
reference r, or is the answer the committed Test Store gives; never the reader's far part.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from store_copies import TEST_STORE, copy_store, resharded

import ogquery.ld
from ogquery.ld.store import StoreReader

#: Inner chunks of 512 elements, 4 to a shard, as `test_store_reads.py` re-shards the Test Store.
INNER, PER_SHARD = 512, 4

_Pairs = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]


def provenance(store: Path) -> dict[str, object]:
    found: dict[str, object] = json.loads((store / "zarr.json").read_text())["attributes"][
        "ld_store"
    ]
    return found


def chromosomes(store: Path) -> list[tuple[str, int, int]]:
    ranges = zarr.open_group(store / "variants", mode="r").attrs["chromosomes"]
    return [(c["name"], c["row_start"], c["row_stop"]) for c in ranges]


def rows_of(store: Path, kind: str, chrom: str) -> _Pairs:
    """`kind/<chrom>/`'s entries as (row, partner, code), local rows, in stored order."""
    group = zarr.open_group(store / kind / chrom, mode="r")
    indptr = np.asarray(group["indptr"][:]).astype(np.int64)
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    partner = np.asarray(group["partner"][:]).astype(np.int64)
    return owner, partner, np.asarray(group["r"][:]).astype(np.int64)


# ------------------------------------------------------------------ what the store holds


def test_the_test_store_holds_its_far_pairs_once_each_and_at_the_floor() -> None:
    counts = provenance(TEST_STORE)["counts"]
    assert isinstance(counts, dict)
    assert counts["far_pairs"] > 0, "the Test Store must hold Far Pairs for this to mean anything"
    held = 0
    for chrom, start, _ in chromosomes(TEST_STORE):
        owner, partner, code = rows_of(TEST_STORE, "far", chrom)
        position = np.asarray(zarr.open_array(TEST_STORE / "variants" / "position", mode="r"))
        near_bp = int(str(provenance(TEST_STORE)["near_bp"]))
        gap = position[start + partner].astype(np.int64) - position[start + owner]
        assert (np.abs(gap) > near_bp).all(), "a Far Pair is beyond near_bp"
        assert (np.abs(code) >= int(str(provenance(TEST_STORE)["far_code_floor"]))).all()
        assert (partner > owner).all(), "each in its earlier row"
        held += len(owner)
    assert held == counts["far_pairs"], "each Far Pair once, as provenance counts them"


# ------------------------------------------------------------------ what a matrix reads


@pytest.fixture(scope="module")
def small_chunks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return resharded(tmp_path_factory.mktemp("resharded") / "store", INNER, PER_SHARD)


def every_alid() -> list[str]:
    reader = StoreReader.open(TEST_STORE)
    return list(reader.variants(np.arange(reader.provenance["n_variants"])).alid)


#: Regions 10 Mb wide whose variants hold Far Pairs among them: chr21's two at 26.0 and 33.8 Mb,
#: and chr22's three at 33.5–42.3 Mb.
WIDE_REGIONS = (("21", 29_000_000, 5_000_000), ("22", 38_000_000, 5_000_000))


def far_stored(m: ogquery.ld.LDMatrix) -> int:
    """The pairs of `m` more than 3 Mb apart that hold a Stored r."""
    gap = np.abs(m.variants.position[:, None] - m.variants.position[None, :])
    return int(((m.kinds() == "stored") & (gap > 3_000_000)).sum()) // 2


def test_a_store_without_far_pairs_is_refused_naming_them(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    shutil.rmtree(copy / "far")
    with pytest.raises(ValueError, match=r"far/2[12]/indptr is missing"):
        ogquery.ld.open(copy).matrix(every_alid())


# ------------------------------------------------------------------ stores written before them


def files(store: Path, under: str = "") -> dict[str, bytes]:
    """Every file of `store` under `under`, by path, with its bytes."""
    root = store / under
    return {str(p.relative_to(store)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


