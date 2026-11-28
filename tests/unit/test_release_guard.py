"""The release_id guard (opengwasdb ADR 0109 §2, ogquery#13).

A splice renumbers `analysis_index` and `variant_index`, so an index cached from one release
still names an Analysis or a variant in another, and fails no check. Every entry point an
integer can reach takes the `release_id` it came from, `expect_release_id=`, and refuses a
mismatch, naming both ids. A query result carries the `release_id` it was read from, and
`resolve` refuses one from another release.

The contract releases used here have distinct ids: `dense_completed` is ``comp``,
`dense_rho` ``v1`` and `hybrid_completed` ``v1-completed``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ogquery import digests
from ogquery.associations import Associations
from ogquery.guard import ReleaseMismatchError
from ogquery.indexed import open_indexed_subset

RELEASES = Path(__file__).parents[1] / "contract" / "releases"
HERE, ELSEWHERE = "comp", "v1-completed"


def release_id(name: str) -> str:
    text = (RELEASES / name / "manifest.json").read_text(encoding="utf-8")
    return str(json.loads(text)["release_id"])


def test_the_contract_releases_have_the_ids_this_file_names() -> None:
    assert release_id("dense_completed") == HERE
    assert release_id("hybrid_completed") == ELSEWHERE


def entry_points() -> dict[str, Callable[[str | None], Any]]:
    """Each entry point an integer reaches, as a call taking the id it expects."""
    dense = digests.open_query(RELEASES / "dense_completed")
    rho = digests.open_query(RELEASES / "dense_rho")
    hybrid = digests.open_query(RELEASES / "hybrid_completed")
    ragged = digests.open_query(RELEASES / "ragged_completed")
    subset = open_indexed_subset(RELEASES / "dense_completed", "contract")
    rows = np.array([0, 1], dtype=np.int64)
    reader = rho.rho_reader()
    assert reader is not None
    return {
        "VariantAxis.by_index": lambda e: ragged._variant_axis.by_index(
            0, expect_release_id=e
        ),
        "VariantAxis.identity_by_indices": lambda e: ragged._variant_axis.identity_by_indices(
            rows, expect_release_id=e
        ),
        "VariantAxis.rsids_by_indices": lambda e: ragged._variant_axis.rsids_by_indices(
            rows, expect_release_id=e
        ),
        "VariantAxis.rows_by_indices": lambda e: ragged._variant_axis.rows_by_indices(
            rows, expect_release_id=e
        ),
        "PanelVariantAxis.by_index": lambda e: dense._variant_axis.by_index(
            0, expect_release_id=e
        ),
        "PanelVariantAxis.identity_by_indices": lambda e: hybrid._variant_axis.identity_by_indices(
            rows, expect_release_id=e
        ),
        "DenseRhoReader.row": lambda e: reader.row(0, expect_release_id=e),
        "DenseRhoReader.pair": lambda e: reader.pair(0, 1, expect_release_id=e),
        "DenseRhoReader.matrix": lambda e: reader.matrix([0, 1], expect_release_id=e),
        "TopHitTier.clumps": lambda e: dense._top_hits.require(5e-8).clumps(
            0, expect_release_id=e
        ),
        "IndexedSubset.decode_analysis": lambda e: subset.decode_analysis(
            0, expect_release_id=e
        ),
    }


READERS = {
    "VariantAxis.by_index": "comp",
    "VariantAxis.identity_by_indices": "comp",
    "VariantAxis.rsids_by_indices": "comp",
    "VariantAxis.rows_by_indices": "comp",
    "PanelVariantAxis.by_index": "comp",
    "PanelVariantAxis.identity_by_indices": "v1-completed",
    "DenseRhoReader.row": "v1",
    "DenseRhoReader.pair": "v1",
    "DenseRhoReader.matrix": "v1",
    "TopHitTier.clumps": "comp",
    "IndexedSubset.decode_analysis": "comp",
}


@pytest.fixture(scope="module")
def points() -> dict[str, Callable[[str | None], Any]]:
    return entry_points()


@pytest.mark.parametrize("name", sorted(READERS))
def test_an_index_cached_from_another_release_is_refused(
    points: dict[str, Callable[[str | None], Any]], name: str
) -> None:
    mine = READERS[name]
    other = "elsewhere" if mine != "elsewhere" else "somewhere"
    with pytest.raises(ReleaseMismatchError) as refused:
        points[name](other)
    assert mine in str(refused.value) and other in str(refused.value), "names both ids"
    assert isinstance(refused.value, ValueError)


@pytest.mark.parametrize("name", sorted(READERS))
def test_an_index_with_its_own_releases_id_is_read(
    points: dict[str, Callable[[str | None], Any]], name: str
) -> None:
    points[name](READERS[name])


def test_the_id_is_required_not_defaulted() -> None:
    """A keyword-only parameter with no default: omitting it is a `TypeError`, never an
    unchecked read."""
    import inspect

    from ogquery.indexed import IndexedSubset
    from ogquery.panel_axis import PanelVariantAxis
    from ogquery.rho import DenseRhoReader
    from ogquery.top_hits import TopHitTier
    from ogquery.variant_axis import VariantAxis

    owners = {
        VariantAxis: ("by_index", "identity_by_indices", "rsids_by_indices", "rows_by_indices"),
        PanelVariantAxis: ("by_index", "identity_by_indices", "rsids_by_indices",
                           "rows_by_indices"),
        DenseRhoReader: ("row", "pair", "matrix"),
        TopHitTier: ("clumps",),
        IndexedSubset: ("decode_analysis",),
    }
    for owner, names in owners.items():
        for name in names:
            parameter = inspect.signature(getattr(owner, name)).parameters["expect_release_id"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (owner, name)
            assert parameter.default is inspect.Parameter.empty, (owner, name)


# ── a result carries its release ─────────────────────────────────────────────


def test_a_result_carries_the_release_it_was_read_from_outside_its_mapping() -> None:
    with digests.open_query(RELEASES / "dense_completed") as q:
        found = q.analysis("a")
        assert found.release_id == HERE == q.release_id
        assert "release_id" not in found, "it is not a key of the mapping"
        assert found.head(1).release_id == HERE
        assert found.where(found.z == found.z).release_id == HERE
        assert replace(found, z=found.z).release_id == HERE


def test_a_result_read_from_another_release_is_not_resolved() -> None:
    with (
        digests.open_query(RELEASES / "dense_completed") as ours,
        digests.open_query(RELEASES / "hybrid_completed") as theirs,
    ):
        found = ours.analysis("a")
        assert list(ours.resolve(found))
        with pytest.raises(ReleaseMismatchError) as refused:
            list(theirs.resolve(found))
        assert HERE in str(refused.value) and ELSEWHERE in str(refused.value)


def test_a_result_with_no_release_is_not_resolved() -> None:
    with digests.open_query(RELEASES / "dense_completed") as q:
        found = q.analysis("a")
        bare = replace(found, release_id=None)
        with pytest.raises(ReleaseMismatchError, match="no release_id"):
            list(q.resolve(bare))
        with pytest.raises(ReleaseMismatchError, match="no release_id"):
            list(q.resolve(dict(found)))


def test_an_empty_answer_carries_its_release_too() -> None:
    with digests.open_query(RELEASES / "dense_completed") as q:
        empty = q.phewas("rs0")
        assert isinstance(empty, Associations) and empty.release_id == HERE


# ── the Reference serialises ─────────────────────────────────────────────────


def test_a_reference_round_trips_through_json() -> None:
    for name in ("dense_completed", "ragged_completed", "hybrid_completed"):
        with digests.open_query(RELEASES / name) as q:
            reference = digests.reference_of(q)
        text = json.dumps(reference.to_json())
        again = digests.Reference.from_json(json.loads(text))
        assert again == reference
        assert not digests.differences(
            digests.release_digests(RELEASES / name, reference=reference),
            digests.release_digests(RELEASES / name, reference=again),
        )


# ── a reader learns its release_id from the release it opens ─────────────────


def test_readers_constructed_the_old_way_carry_their_releases_id() -> None:
    from ogquery.panel_axis import PanelVariantAxis
    from ogquery.release import open_release, release_arrays
    from ogquery.rho import DenseRhoReader
    from ogquery.top_hits import TopHitTiers
    from ogquery.variant_axis import NoVariantTable, VariantAxis

    ragged = RELEASES / "ragged_completed"
    version = open_release(ragged).manifest.format_version
    axis = VariantAxis(ragged, NoVariantTable(ragged), format_version=version)
    assert axis.release_id == "comp"
    tiers = TopHitTiers(release_arrays(RELEASES / "dense_completed"), clumps=None)
    assert tiers.require(5e-8).release_id == "comp"
    group = release_arrays(RELEASES / "dense_rho")
    assert DenseRhoReader(group["rho"], 3).release_id == "v1"
    shared = open_release(RELEASES / "hybrid_completed").own_variant_axis(aliases=False)
    assert PanelVariantAxis(shared, np.arange(2)).release_id == shared.release_id == ELSEWHERE


def _without_manifest(tmp_path: Path, name: str) -> Path:
    import shutil

    copy = tmp_path / name
    shutil.copytree(RELEASES / name, copy)
    (copy / "manifest.json").unlink()
    return copy


def test_a_reader_whose_release_cannot_be_determined_refuses_naming_the_path(
    tmp_path: Path,
) -> None:
    import zarr

    from ogquery.guard import ReleaseIdUnknown
    from ogquery.release import release_arrays
    from ogquery.rho import DenseRhoReader
    from ogquery.top_hits import TopHitTiers
    from ogquery.variant_axis import NoVariantTable, VariantAxis

    bare = _without_manifest(tmp_path, "ragged_completed")
    with pytest.raises(ReleaseIdUnknown, match="ragged_completed"):
        VariantAxis(bare, NoVariantTable(bare), format_version="0.3.0")
    with pytest.raises(ReleaseIdUnknown, match="ragged_completed"):
        TopHitTiers(release_arrays(bare))
    rho = _without_manifest(tmp_path, "dense_rho")
    with pytest.raises(ReleaseIdUnknown, match="dense_rho"):
        DenseRhoReader(release_arrays(rho)["rho"], 3)
    memory = zarr.group()
    with pytest.raises(ReleaseIdUnknown):
        TopHitTiers(memory)
    assert TopHitTiers(memory, release_id="given").release_id == "given"
