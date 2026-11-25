"""The Proxy Index is gone, and a store that still carries one reads alike (opengwas/ogld#72).

From opengwasdb's `tests/ld/test_proxy_index_removed.py` at `c8ebe4ca` (ogquery#2): its reads.

- **A leftover `index/`.** A 0.2.0 store derived before opengwas/ogld#72 without `--drop-index`
  holds the component. It opens and answers exactly as the store without one does, reading no key
  of it -- even when every entry of it is gone, which a read of it would have to refuse.
- **A 0.1.0 store** still opens and answers `matrix` and `region`, which read the Near Band and
  the Far Pairs; `clump` and `proxies` are refused, naming the Partner Rows.

The leftover index is written into a copy of the committed Test Store with zarr alone. Its source
scans, the store a build writes, and validation's view of the index stay in opengwasdb.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from store_copies import FLOOR, TEST_STORE, copy_store, store_reads
from zarr.codecs import ZstdCodec

import ogquery.ld
from ogquery.ld.format import InputError, Refusal
from ogquery.ld.store import StoreReader


def write_a_leftover_index(store: Path) -> None:
    """Give `store` a Proxy Index a store derived before opengwas/ogld#72 still holds: one entry a
    chromosome, valid enough that reading it would answer, and never read by this package."""
    root = zarr.open_group(str(store), mode="r+")
    for entry in root["variants"].attrs["chromosomes"]:
        rows = entry["row_stop"] - entry["row_start"]
        group = root.create_group(f"index/{entry['name']}")
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indptr[1:] = 1 if rows else 0
        partner = np.minimum(np.arange(rows, dtype=np.int64) + 1, max(0, rows - 1))
        for name, data, fill in (
            ("indptr", indptr, -1),
            ("partner", partner.astype(np.int32), -1),
            ("r", np.full(rows, FLOOR, dtype=np.int16), -32768),
        ):
            group.create_array(
                name,
                shape=data.shape,
                dtype=data.dtype,
                chunks=(max(1, len(data)),),
                shards=(max(1, len(data)),),
                compressors=[ZstdCodec(level=3)],
                fill_value=fill,
            )[:] = data


def answers(store: Path) -> dict[str, np.ndarray]:
    reader = StoreReader.open(store)
    names = list(reader.variants(np.arange(reader.provenance["n_variants"])).alid)
    found: dict[str, np.ndarray] = {"matrix": ogquery.ld.open(store).matrix(names).r}
    for chrom in ("21", "22"):
        found[f"region-{chrom}"] = ogquery.ld.open(store).region(chrom, 30_000_000, 5_000_000).r
    return found


def test_a_leftover_index_answers_exactly_as_a_store_without_one(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    assert (copy / "index" / "21" / "r" / "zarr.json").is_file(), "the copy holds the component"
    plain, leftover = answers(TEST_STORE), answers(copy)
    assert leftover["matrix"].shape == (1056, 1056), "a matrix of every variant"
    for name, answer in plain.items():
        np.testing.assert_array_equal(leftover[name], answer, err_msg=name)


def every_query(store_path: Path) -> None:
    """Every read the package makes of a store: one of each query family, and the pair reads
    behind them. A read under `index/` anywhere here is a read of the component."""
    store = ogquery.ld.open(store_path)
    reader = StoreReader.open(store_path)
    rows = np.arange(reader.provenance["n_variants"])
    names = list(reader.variants(rows).alid)
    small = names[:40]
    store.matrix(small)
    store.region("22", 30_000_000, 4_000_000)
    reader.region_r(468, 1056)
    reader.pair_r(rows[::7])
    reader.r_of_pairs(rows[::13], np.minimum(rows[::13] + 1, len(rows) - 1))
    scores = np.linspace(5.0, 1.0, len(small))
    store.clump(small, scores, r2=0.01, kb=3000)
    store.proxies(small, r2=0.5, kb=3000)
    store.variants_in("21", 1, 3_000_000_000)
    store.has(small)


def test_no_query_reads_a_key_of_a_leftover_index(tmp_path: Path) -> None:
    # On a 0.2.0 copy that still holds the component, a read of any array of it -- or of its own
    # `zarr.json` -- would be a store key here (opengwas/ogld#71's method).
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    assert list(copy.rglob("index/*/*/zarr.json")), "the copy holds the component's metadata"
    with store_reads() as keys:
        every_query(copy)
    offending = sorted({key for key in keys if key.startswith("index/")})
    assert offending == [], f"a query read the component: {offending}"
    assert [key for key in keys if key.endswith("zarr.json")], "metadata reads are recorded"


def test_a_leftover_index_whose_entries_are_lost_is_still_not_read(tmp_path: Path) -> None:
    # A read of any of its arrays would refuse them (ld-store §6): a matrix and a region answer on.
    copy = copy_store(tmp_path / "store")
    write_a_leftover_index(copy)
    chunks = sorted(copy.glob("index/*/*/c/*"))
    assert len(chunks) >= 4, "both chromosomes' arrays have a chunk to lose"
    for chunk in chunks:
        chunk.unlink()
    plain, lost = answers(TEST_STORE), answers(copy)
    for name, answer in plain.items():
        np.testing.assert_array_equal(lost[name], answer, err_msg=name)


def a_0_1_0_store(to: Path) -> Path:
    """The Test Store as a 0.1.0 store records itself: no Partner Rows claimed, no `t_partner`,
    and the Proxy Index a 0.1.0 store holds with its `counts.index_entries`."""
    copy = copy_store(to)
    write_a_leftover_index(copy)
    root = zarr.open_group(str(copy), mode="r+")
    recorded = dict(root.attrs["ld_store"])
    recorded["format_version"] = "0.1.0"
    for key in ("t_partner", "partner_code_floor"):
        recorded.pop(key, None)
    recorded["counts"] = {
        key: value for key, value in recorded["counts"].items() if key != "partner_entries"
    }
    recorded["counts"]["index_entries"] = 2 * 1056
    recorded["software"] = {
        key: value for key, value in recorded["software"].items() if key != "zstd"
    }
    root.attrs.update({"ld_store": recorded})
    return copy


def test_a_0_1_0_store_opens_and_answers_matrix_and_region(tmp_path: Path) -> None:
    copy = a_0_1_0_store(tmp_path / "store")
    assert ogquery.ld.open(copy).provenance["format_version"] == "0.1.0"
    plain, older = answers(TEST_STORE), answers(copy)
    for name, answer in plain.items():
        np.testing.assert_array_equal(older[name], answer, err_msg=name)


def _clump_or_proxies(query: str, store: Path, names: list[str], scores: np.ndarray) -> None:
    if query == "clump":
        ogquery.ld.open(store).clump(names, scores, r2=0.01, kb=3000)
    else:
        ogquery.ld.open(store).proxies(names, r2=0.5, kb=3000)


@pytest.mark.parametrize("query", ["clump", "proxies"])
def test_a_0_1_0_store_is_refused_for_clump_and_proxies_naming_the_partner_rows(
    query: str, tmp_path: Path
) -> None:
    copy = a_0_1_0_store(tmp_path / "store")
    names = list(StoreReader.open(copy).variants(np.arange(20)).alid)
    scores = np.linspace(5.0, 1.0, len(names))
    ask = partial(_clump_or_proxies, query, copy, names, scores)
    with pytest.raises((InputError, Refusal), match="Partner Rows"):
        ask()
