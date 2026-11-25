"""A take of a string array across chunks returns every string, long ones included (ogquery#2).

numpy 2.0 and 2.1 scatter a StringDType string longer than 15 bytes, which numpy keeps outside
the array, into a fresh string array as an empty string (fixed in numpy 2.2). numpy 2.0 is
ogquery's floor, and the chunk cache once scattered a multi-chunk take back into request order
that way, so at the floor a long ALID read back as "" -- the fill value, which the reader
refused (`location._inverse`). CI's `floors` job runs this at numpy 2.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

from ogquery.ld.store import location

#: Long and short strings alike: an indel's ALID is often more than 15 bytes.
VALUES = np.array([f"22:{i}:" + "A" * (i % 40) + ":C" for i in range(1_000)], dtype=object)


@pytest.fixture
def strings(tmp_path: Path) -> zarr.Array:
    group = zarr.open_group(str(tmp_path / "s"), mode="w")
    made = group.create_array(
        "alid",
        shape=VALUES.shape,
        dtype=str,
        chunks=(64,),
        shards=(256,),
        compressors=[ZstdCodec(level=3)],
        fill_value="",
    )
    made[:] = VALUES
    member = location.open_group(tmp_path / "s", cache=location.ChunkCache())["alid"]
    assert isinstance(member, zarr.Array)
    return member


def test_a_take_across_chunks_returns_each_string_in_request_order(strings: zarr.Array) -> None:
    at = np.random.default_rng(7).permutation(len(VALUES))[:600]
    assert len(np.unique(at // 64)) > 1, "the take must span chunks for this to mean anything"
    assert max(len(v) for v in VALUES[at]) > 15, "and hold strings numpy keeps outside the array"
    taken = location.ChunkCache().take(strings, at)
    assert taken.tolist() == VALUES[at].tolist()


def test_a_bounded_take_returns_each_string_in_request_order(
    strings: zarr.Array, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read whose chunks are more than its bytes in flight is taken a batch of chunks at a time.
    cache = location.ChunkCache()
    cache.take(strings, np.arange(64))  # counts a string chunk, so the bound applies to strings
    monkeypatch.setattr(location, "_FETCH_BYTES", 1)
    at = np.random.default_rng(11).permutation(len(VALUES))
    assert cache.take(strings, at).tolist() == VALUES[at].tolist()
