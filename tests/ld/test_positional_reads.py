"""A local frame is one positional read, on every platform (ogquery#2).

A Partner Rows frame of a local store is read from its chunk file at an offset
(`location.PreadSource`, partner-rows §7): `os.pread`, which Windows does not have. There
`location._read_at` seeks and reads under one lock instead. These tests run that path on any
platform, by removing `os.pread`, and hold it to `os.pread`'s bytes and to the same answers.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from store_copies import TEST_STORE
from support import alids, passing_rows

import ogquery.ld
from ogquery.ld.store import location


@pytest.fixture
def a_file(tmp_path: Path) -> Path:
    path = tmp_path / "frames.bin"
    # Every byte value, \r\n and ^Z among them, which a text-mode descriptor would change.
    path.write_bytes(bytes(range(256)) * 64 + b"\r\n\x1a\r\n")
    return path


@pytest.mark.parametrize(
    ("start", "stop"), [(0, 16), (250, 270), (16_380, 16_390), (17_000, 17_010)]
)
def test_a_seek_and_a_read_give_the_bytes_at_the_offset(
    a_file: Path, start: int, stop: int
) -> None:
    data = a_file.read_bytes()
    source = location.PreadSource(a_file)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.delattr(os, "pread", raising=False)
            assert source.read(start, stop) == data[start:stop]
    finally:
        source.close()


def test_reads_on_threads_without_pread_each_get_their_own_bytes(a_file: Path) -> None:
    data = a_file.read_bytes()
    source = location.PreadSource(a_file)
    found: dict[int, bytes] = {}
    starts = list(range(0, 16_000, 97))

    def read(start: int) -> None:
        found[start] = source.read(start, start + 61)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.delattr(os, "pread", raising=False)
            threads = [threading.Thread(target=read, args=(s,)) for s in starts]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        source.close()
    assert found == {s: data[s : s + 61] for s in starts}


def test_the_queries_answer_alike_without_pread(monkeypatch: pytest.MonkeyPatch) -> None:
    names = alids(passing_rows()[::4])
    score = np.linspace(6.0, 1.0, len(names))

    def answers() -> dict[str, object]:
        store = ogquery.ld.open(TEST_STORE)
        try:
            return {
                "proxies": store.proxies(names[:40], r2=0.2, kb=3000).columns(),
                "clump": store.clump(names, score, r2=0.1, kb=3000).columns(),
            }
        finally:
            store.close()

    with_pread = answers()
    monkeypatch.delattr(os, "pread", raising=False)
    without = answers()
    for query, columns in with_pread.items():
        assert isinstance(columns, dict)
        for name, value in columns.items():
            np.testing.assert_array_equal(value, without[query][name], err_msg=f"{query}.{name}")
    assert len(with_pread["clump"]["alid"]) > 200
