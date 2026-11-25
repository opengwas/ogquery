"""The local ALID companion (ADR 0098) is read wherever the Store was written (ogquery#2).

A companion pins the canonical files it was derived from by their paths relative to the Store,
and a reader compares those pins with the files it finds. opengwasdb derives companions on Linux,
so a published manifest's keys are spelt with `/`. A reader on Windows forms the same relative
paths with `\\`, and compared as strings every pin differed: a valid companion was refused there.
The pins are compared as POSIX paths (`layout.alid_mmap_sources`).

The companion here is written by the test from the Test Store's own ALID index, in the schema
the reader checks (`layout.alid_mmap_manifest`): the payloads are the index's bytes, so no build
code is needed. Windows is reproduced on any platform by making `relative_to` give Windows paths.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
import zarr
from store_copies import copy_store
from support import TEST_STORE_K0, alids, panel_data, passing_rows

import ogquery.ld
from ogquery.ld.store import layout


def with_companion(store: Path) -> Path:
    """Publish a companion of `store`'s ALID index, as a Linux derivation publishes one."""
    root = zarr.open_group(str(store), mode="r")
    keys = np.asarray(root[layout.ALID_INDEX_ALID][:], dtype=np.uint8)
    rows = np.asarray(root[layout.ALID_INDEX_ROW][:]).astype(layout.ALID_MMAP_ROW_DTYPE)
    count = len(rows)
    assert len(keys) == count * layout.ALID_INDEX_WIDTH
    sources = layout.alid_mmap_sources(store)
    partial = store / layout.ALID_MMAP / "partial"
    partial.mkdir(parents=True)
    (partial / layout.ALID_MMAP_KEYS).write_bytes(keys.tobytes())
    (partial / layout.ALID_MMAP_ROWS).write_bytes(rows.tobytes())
    payloads = {name: layout.file_sha256(partial / name) for name in (
        layout.ALID_MMAP_KEYS, layout.ALID_MMAP_ROWS
    )}
    raw = layout.json_bytes(layout.alid_mmap_manifest(count, sources, payloads))
    (partial / layout.ALID_MMAP_MANIFEST).write_bytes(raw)
    pointer = layout.alid_mmap_pointer(raw)
    partial.rename(store / layout.ALID_MMAP / layout.alid_mmap_generation(pointer))
    document = json.loads((store / "zarr.json").read_bytes())
    document["attributes"][layout.ALID_MMAP_ATTR] = pointer
    (store / "zarr.json").write_bytes(layout.json_bytes(document))
    return store


@pytest.fixture
def companion(tmp_path: Path) -> Path:
    store = with_companion(copy_store(tmp_path / "store"))
    manifest = next((store / layout.ALID_MMAP).glob(f"*/{layout.ALID_MMAP_MANIFEST}"))
    pins = json.loads(manifest.read_bytes())["sources"]
    nested = [name for name in pins if "/" in name]
    assert nested and not any("\\" in name for name in pins), "pins spelt as Linux spells them"
    return store


def asked() -> list[str]:
    v = panel_data().variants()
    return alids(np.arange(0, len(v), 3)) + ["22:1:A:C", "x" * 40]


def test_a_companion_answers_as_the_index_does(companion: Path) -> None:
    whole = ogquery.ld.open(companion)
    assert whole._reader._variant_table._has_alid_mmap, "the companion is published"
    found = whole.has(asked())
    assert found.tolist() == ogquery.ld.open(copy_store(companion.parent / "plain")).has(
        asked()
    ).tolist()
    assert whole._reader._variant_table._alid_mmap is not None, "and it answered"
    assert found.sum() == len(set(passing_rows()) & set(range(0, len(panel_data().variants()), 3)))


def test_a_companion_written_with_slashes_is_read_where_paths_use_backslashes(
    companion: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pathlib.Path.relative_to

    def windows(self: Path, *other: object, **kw: object) -> PureWindowsPath:
        return PureWindowsPath(*original(self, *other, **kw).parts)  # type: ignore[arg-type]

    monkeypatch.setattr(pathlib.Path, "relative_to", windows)
    assert str(Path("/a/b/c").relative_to("/a")) == "b\\c", "relative paths as Windows spells them"
    store = ogquery.ld.open(companion)
    found = store.has(asked())
    assert store._reader._variant_table._alid_mmap is not None, "the companion answered"
    assert found.any()


def test_a_companion_is_read_where_a_path_s_stat_and_a_handle_s_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On Windows with Python 3.14 (CI run 45664832290), a valid companion was refused "changed
    # before mapping": its payloads' identity was recorded from a path's stat and compared with
    # the opened file's fstat, and there the two disagree. Here a path's stat reports another
    # ctime; the reader records and compares descriptors' identities alone (opengwas/ogquery#2).
    real = pathlib.Path.stat

    class Shifted:
        def __init__(self, stat: object) -> None:
            self._stat = stat

        def __getattr__(self, name: str) -> object:
            return getattr(self._stat, name)

        @property
        def st_ctime_ns(self) -> int:
            return int(self._stat.st_ctime_ns) + 1  # type: ignore[attr-defined]

    def path_stat(self: Path, *args: object, **kwargs: object) -> object:
        found = real(self, *args, **kwargs)  # type: ignore[arg-type]
        return Shifted(found) if self.parent.parent.name == "alid_mmap" else found

    monkeypatch.setattr(pathlib.Path, "stat", path_stat)
    store = ogquery.ld.open(TEST_STORE_K0)
    assert store.has(asked()).any()
    assert store._reader._variant_table._alid_mmap is not None, "the companion answered"
