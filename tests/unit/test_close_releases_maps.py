"""Closing a reader lets go of the release's files (ogquery#1, `npy_sidecar.release_maps`).

The Variant Axis memory-maps its sidecars. On Windows a file that a live map holds can be
neither replaced nor deleted, so a user who closed a facade could not delete the release
or rewrite it in place. `close()` -- or leaving the facade's `with` block -- releases
every map, and a read through a closed reader raises `ReaderClosed` rather than reading
as empty.

On Windows, deleting and rewriting the files is the proof itself. Everywhere, no live
`np.memmap` may name a file of the release once it is closed, which Linux and macOS can
check although they would let the file go.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import pytest
from contract_reader import RELEASES

from ogquery.facade import query_release
from ogquery.npy_sidecar import ReaderClosed, is_memory_mapped, release_maps
from ogquery.release import open_release

LAYOUTS = ["dense_observed", "ragged_completed", "hybrid_completed"]


def _live_maps_under(root: Path) -> list[str]:
    gc.collect()
    held = []
    for obj in gc.get_objects():
        if isinstance(obj, np.memmap) and obj.filename is not None:
            if Path(obj.filename).resolve().is_relative_to(root.resolve()):
                held.append(Path(obj.filename).name)
    return sorted(held)


def _copy(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(RELEASES / name, tmp_path / name))


def _ask(q) -> str:  # type: ignore[no-untyped-def]
    rows = q.variants_table()
    alid = rows[min(rows)]["alid"]
    assert q.phewas(alid)["z"].size
    return str(alid)


@pytest.mark.parametrize("name", LAYOUTS)
def test_a_closed_facade_holds_no_map_and_its_release_can_be_deleted(
    tmp_path: Path, name: str
):
    release = _copy(tmp_path, name)
    q = query_release(open_release(release))
    _ask(q)
    assert _live_maps_under(release), "the facade mapped nothing: the test proves nothing"
    q.close()  # the facade is still referenced: close alone must let go
    assert _live_maps_under(release) == []
    shutil.rmtree(release)  # on Windows, refused while a map is live
    assert not release.exists()


@pytest.mark.parametrize("name", LAYOUTS)
def test_a_release_rewritten_after_its_with_block_reopens_with_the_new_files(
    tmp_path: Path, name: str
):
    """Reopening a release after a rewrite, the case a Windows user meets: every sidecar
    is replaced in place once the facade's block is left, and a new facade answers."""
    release = _copy(tmp_path, name)
    with query_release(open_release(release)) as q:
        alid = _ask(q)
        before = q.phewas(alid)
    for path in sorted(release.glob("*.npy")):
        np.save(path, np.load(path))  # replace each file, same content
    with query_release(open_release(release)) as again:
        after = again.phewas(alid)
    assert {k: np.asarray(v).tolist() for k, v in before.items()} == {
        k: np.asarray(v).tolist() for k, v in after.items()
    }
    assert _live_maps_under(release) == []


@pytest.mark.parametrize("name", LAYOUTS)
def test_a_read_through_a_closed_facade_is_refused(name: str):
    q = query_release(open_release(RELEASES / name))
    alid = _ask(q)
    q.close()
    with pytest.raises(ReaderClosed, match="was released when its reader was closed"):
        q.phewas(alid)


def test_a_hybrid_dense_components_axis_lets_go_of_dense_to_shared(tmp_path: Path):
    release = _copy(tmp_path, "hybrid_completed")
    axis = open_release(release).dense_component().variant_axis(aliases=False)
    assert axis.n_variants
    assert "dense_to_shared.npy" in _live_maps_under(release)
    axis.close()
    assert _live_maps_under(release) == []


def test_release_maps_walks_ogquery_objects_and_leaves_other_things_alone(tmp_path: Path):
    path = tmp_path / "a.npy"
    np.save(path, np.arange(4))

    class Foreign:  # a caller's object: not ogquery's, so not walked
        def __init__(self) -> None:
            self.mapped = np.load(path, mmap_mode="r")

    foreign = Foreign()
    axis = open_release(RELEASES / "dense_ranges").variant_axis(aliases=False)
    axis.foreign = foreign  # type: ignore[attr-defined]
    axis.copy = np.load(path, mmap_mode="r")[[0, 1]]  # type: ignore[attr-defined]
    view = np.load(path, mmap_mode="r")[1:]
    assert is_memory_mapped(view) and is_memory_mapped(memoryview(view))
    assert not is_memory_mapped(axis.copy)  # a copy keeps no file open
    release_maps(axis)
    assert is_memory_mapped(foreign.mapped)  # not ogquery's to release
    assert isinstance(axis.copy, np.ndarray)
    with pytest.raises(ReaderClosed):
        len(axis.alid_index.keys)  # type: ignore[attr-defined]
