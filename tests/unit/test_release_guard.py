"""The index-space guard (opengwasdb ADR 0109 §2, ogquery#13).

A splice renumbers `analysis_index` and `variant_index`, so an index cached from one release
still names an Analysis or variant in another, and fails no check. What an index is scoped to
is an `IndexSpace`: the Store and release it belongs to and the axis it numbers. Every entry
point an integer can reach takes the space it came from, `expect_space=` (or, for cells,
`expect_spaces=`), and refuses a mismatch naming both. A query result carries the spaces of
its two index columns, and `resolve` refuses a result from another.

The contract releases have these identities: `dense_completed` is store ``s`` release
``comp``; `ragged_completed` is store ``residual-rag``, also release ``comp``; `dense_rho` is
release ``v1``; `hybrid_completed` is store ``hyb`` release ``v1-completed`` (its Dense
Component is store ``hyb-dense`` of the same release and is scoped to the Hybrid).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ogquery import digests
from ogquery.associations import Associations
from ogquery.associations.cells import Cells
from ogquery.guard import (
    ANALYSIS,
    PANEL,
    VARIANT,
    IndexSpace,
    ReleaseIdentity,
    ReleaseIdUnknown,
    ReleaseMismatchError,
)
from ogquery.indexed import open_indexed_subset
from ogquery.manifest import StoreManifest

RELEASES = Path(__file__).parents[1] / "contract" / "releases"
DENSE = ReleaseIdentity("s", "comp")
RAGGED = ReleaseIdentity("residual-rag", "comp")
HYBRID = ReleaseIdentity("hyb", "v1-completed")
RHO = ReleaseIdentity("rho", "v1")


def query(name: str) -> Any:
    return digests.open_query(RELEASES / name)


def other_space(space: IndexSpace) -> IndexSpace:
    return IndexSpace(ReleaseIdentity("elsewhere", "other"), space.space)


# ── a result's spaces ────────────────────────────────────────────────────────


def test_the_contract_releases_have_the_identities_this_file_names() -> None:
    for name, identity in (
        ("dense_completed", DENSE),
        ("ragged_completed", RAGGED),
        ("hybrid_completed", HYBRID),
    ):
        assert StoreManifest.load(RELEASES / name).identity == identity


def test_a_result_carries_the_spaces_of_its_indices_outside_its_mapping() -> None:
    with query("dense_completed") as q:
        found = q.analysis("a")
        assert found.variant_space == IndexSpace(DENSE, PANEL) or found.variant_space == IndexSpace(
            DENSE, VARIANT
        )
        assert found.analysis_space == IndexSpace(DENSE, ANALYSIS)
        assert "variant_space" not in found and "release_id" not in found
        assert found.head(1).analysis_space == found.analysis_space
        assert replace(found, z=found.z).variant_space == found.variant_space
    with query("ragged_completed") as q:
        assert q.analysis("a").variant_space == IndexSpace(RAGGED, VARIANT)


def test_a_hybrids_answers_are_in_the_shared_space_on_both_sides_of_the_join() -> None:
    """Dense rows are remapped onto the shared axis and the Overflow's are stamped at their
    source, so the join is stamped, not vouched for."""
    with query("hybrid_completed") as q:
        found = q.analysis("trait_a")
        assert found.variant_space == IndexSpace(HYBRID, VARIANT)
        assert found.analysis_space == IndexSpace(HYBRID, ANALYSIS)
        dense = q._dense.analysis("trait_a")
        assert dense.variant_space == IndexSpace(HYBRID, PANEL), "the component's own numbering"
        assert len(found["z"]) > len(dense["z"]), "the fixture joins Overflow rows"


def test_a_dense_components_rows_are_not_resolved_through_the_shared_axis() -> None:
    """Row 200 of the panel is `1:300000:A:G`; row 200 of the shared axis another variant."""
    with query("hybrid_completed") as q:
        panel = q._dense.analysis("trait_a")
        assert len(panel["z"])
        with pytest.raises(ReleaseMismatchError, match="variant"):
            list(q.resolve(panel))
        assert list(q.resolve(q.analysis("trait_a")))


def test_two_stores_that_share_a_release_id_are_not_confused() -> None:
    with query("dense_completed") as dense, query("ragged_completed") as ragged:
        found = dense.analysis("a")
        assert found.variant_space.release_id == ragged.variant_space.release_id == "comp"
        with pytest.raises(ReleaseMismatchError):
            list(ragged.resolve(found))
        with pytest.raises(ReleaseMismatchError):
            ragged._variant_axis.by_index(0, expect_space=dense.variant_space)


def test_a_result_with_no_spaces_is_not_resolved() -> None:
    with query("dense_completed") as q:
        found = q.analysis("a")
        for bare in (replace(found, variant_space=None), replace(found, analysis_space=None)):
            with pytest.raises(ReleaseMismatchError, match="no index space"):
                list(q.resolve(bare))
        with pytest.raises(ReleaseMismatchError, match="no index space"):
            list(q.resolve(dict(found)))


def test_a_result_from_another_release_is_not_resolved() -> None:
    with query("dense_completed") as ours, query("hybrid_completed") as theirs:
        found = ours.analysis("a")
        assert list(ours.resolve(found))
        with pytest.raises(ReleaseMismatchError) as refused:
            list(theirs.resolve(found))
        assert "comp" in str(refused.value) and "v1-completed" in str(refused.value)


def test_records_of_different_spaces_or_of_none_cannot_be_joined() -> None:
    with query("dense_completed") as dense, query("ragged_completed") as ragged:
        a, b = dense.analysis("a"), ragged.analysis("a")
        with pytest.raises(ValueError, match="different spaces"):
            Associations.concat([a, b])
        with pytest.raises(ValueError, match="different spaces"):
            Associations.concat([a, replace(a, variant_space=None, analysis_space=None)])
        assert Associations.concat([a, a]).variant_space == a.variant_space
        assert Associations.concat([a, Associations.empty()]).variant_space == a.variant_space


# ── every entry point an integer reaches ─────────────────────────────────────


def _n_pairs(csr: Any, rows: Any, expected: Any) -> None:
    """`n_pairs` on a fixture with no `n` plane: the guard first, then the plane's refusal."""
    try:
        csr.n_pairs(rows, np.zeros(2, dtype=np.int64), expect_spaces=expected)
    except ReleaseMismatchError:
        raise
    except ValueError as error:
        assert "no n plane" in str(error)


def entry_points() -> dict[str, tuple[Any, Callable[[Any], Any]]]:
    """Each as ``(the space the integers belong to, a call taking what it expects)``."""
    dense, hybrid, ragged = (
        query("dense_completed"),
        query("hybrid_completed"),
        query("ragged_completed"),
    )
    rho = query("dense_rho")
    reader = rho.rho_reader()
    assert reader is not None
    subset = open_indexed_subset(RELEASES / "dense_completed", "contract")
    rows = np.array([0, 1], dtype=np.int64)
    tier = dense._top_hits.require(5e-8)
    csr = ragged._reader.structure
    by_variant = ragged._reader._by_variant
    if by_variant is None:
        ragged._reader.open_variant_index(ragged._variant_axis.n_variants)
        by_variant = ragged._reader._by_variant
    cells = Cells.analysis(0)
    return {
        "VariantAxis.by_index": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.by_index(0, expect_space=e),
        ),
        "VariantAxis.by_indices": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.by_indices([0], expect_space=e),
        ),
        "VariantAxis.identity_by_indices": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.identity_by_indices(rows, expect_space=e),
        ),
        "VariantAxis.rsids_by_indices": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.rsids_by_indices(rows, expect_space=e),
        ),
        "VariantAxis.source_alids_by_indices": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.source_alids_by_indices(rows, expect_space=e),
        ),
        "VariantAxis.rows_by_indices": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.rows_by_indices(rows, expect_space=e),
        ),
        "VariantAxis.sidecar_records": (
            ragged.variant_space,
            lambda e: ragged._variant_axis.sidecar_records(rows, expect_space=e),
        ),
        "PanelVariantAxis.by_index": (
            dense.variant_space,
            lambda e: dense._variant_axis.by_index(0, expect_space=e),
        ),
        "PanelVariantAxis.identity_by_indices": (
            hybrid._dense.variant_space,
            lambda e: hybrid._dense._variant_axis.identity_by_indices(rows, expect_space=e),
        ),
        "DenseRhoReader.row": (reader.index_space, lambda e: reader.row(0, expect_space=e)),
        "DenseRhoReader.pair": (reader.index_space, lambda e: reader.pair(0, 1, expect_space=e)),
        "DenseRhoReader.matrix": (
            reader.index_space,
            lambda e: reader.matrix([0, 1], expect_space=e),
        ),
        "TopHitTier.clumps": (tier.index_space, lambda e: tier.clumps(0, expect_space=e)),
        "TopHitTier.read": (tier.index_space, lambda e: tier.read(0, expect_space=e)),
        "IndexedSubset.decode_analysis": (
            subset.index_space,
            lambda e: subset.decode_analysis(0, expect_space=e),
        ),
        "ByVariantReader.rows_for_variant": (
            by_variant.index_space,
            lambda e: by_variant.rows_for_variant(0, expect_space=e),
        ),
        "RaggedCSRReader.span": (csr.spaces.analysis, lambda e: csr.span(0, expect_space=e)),
        "RaggedCSRReader.segment_positions": (
            csr.spaces,
            lambda e: csr.segment_positions(rows, analysis_index=0, expect_spaces=e),
        ),
        "RaggedCSRReader.eaf_pairs": (
            csr.spaces,
            lambda e: csr.eaf_pairs(rows, np.zeros(2, dtype=np.int64), expect_spaces=e),
        ),
        "RaggedCSRReader.n_pairs": (csr.spaces, lambda e: _n_pairs(csr, rows, e)),
        "RaggedCSRReader.pair_positions": (
            csr.spaces,
            lambda e: csr.pair_positions(rows, np.zeros(2, dtype=np.int64), expect_spaces=e),
        ),
        "DenseGridReader.read": (
            dense._grid.spaces,
            lambda e: dense._grid.read(Cells.analysis(0), expect_spaces=e),
        ),
        "DenseGridReader.inspect": (
            dense._grid.spaces,
            lambda e: dense._grid.inspect(Cells.analysis(0), expect_spaces=e),
        ),
        "CsrReader.read": (
            ragged._reader.spaces,
            lambda e: ragged._reader.read(cells, expect_spaces=e),
        ),
        "CsrReader.inspect": (
            ragged._reader.spaces,
            lambda e: ragged._reader.inspect(cells, expect_spaces=e),
        ),
        # Review round 2's finds (`test_index_space_sweep`).
        "ByVariantReader.rows_for_variant_range": (
            by_variant.index_space,
            lambda e: by_variant.rows_for_variant_range(0, 1, expect_space=e),
        ),
        "ByVariantReader.decode": (
            by_variant.index_space,
            lambda e: by_variant.decode(0, 1, 0, 0, expect_space=e),
        ),
        "RaggedCSRReader.se_slice": (
            csr.spaces.analysis,
            lambda e: csr.se_slice(0, 1, analysis_index=0, expect_space=e),
        ),
        "DenseGridReader.z": (
            dense._grid.spaces,
            lambda e: dense._grid.z(Cells.cross(rows, [0]), expect_spaces=e),
        ),
        "DenseGridReader.windows": (
            dense._grid.spaces,
            lambda e: dense._grid.windows(Cells.analysis(0), expect_spaces=e),
        ),
        "CsrReader.z": (
            ragged._reader.spaces,
            lambda e: ragged._reader.z(Cells.positions(rows), expect_spaces=e),
        ),
        "CsrReader.windows": (
            ragged._reader.spaces,
            lambda e: ragged._reader.windows(Cells.analysis(0), expect_spaces=e),
        ),
    }


@pytest.fixture(scope="module")
def points() -> dict[str, tuple[Any, Callable[[Any], Any]]]:
    return entry_points()


def _foreign(expected: Any) -> Any:
    from ogquery.guard import CellSpaces

    if isinstance(expected, CellSpaces):
        return CellSpaces(other_space(expected.variant), other_space(expected.analysis))
    return other_space(expected)


@pytest.mark.parametrize("name", sorted(entry_points()))
def test_an_index_from_another_space_is_refused_and_its_own_is_read(
    points: dict[str, tuple[Any, Callable[[Any], Any]]], name: str
) -> None:
    owner, call = points[name]
    call(owner)
    with pytest.raises(ReleaseMismatchError) as refused:
        call(_foreign(owner))
    assert "elsewhere" in str(refused.value)
    assert isinstance(refused.value, ValueError)


@pytest.mark.parametrize("name", ["VariantAxis.by_index", "DenseRhoReader.pair", "TopHitTier.read"])
def test_the_right_release_in_the_wrong_axis_is_refused(
    points: dict[str, tuple[Any, Callable[[Any], Any]]], name: str
) -> None:
    owner, call = points[name]
    wrong = ANALYSIS if owner.space != ANALYSIS else VARIANT
    with pytest.raises(ReleaseMismatchError):
        call(IndexSpace(owner.identity, wrong))


def test_a_cached_index_with_no_space_named_is_refused_where_one_is_given() -> None:
    with query("dense_completed") as q:
        tier = q._top_hits.require(5e-8)
        assert tier.read(None) is not None, "no index, nothing to scope"
        with pytest.raises(ReleaseMismatchError, match="no index space"):
            tier.read(0)


def test_the_space_is_required_not_defaulted() -> None:
    import inspect

    from ogquery.by_variant import ByVariantReader
    from ogquery.csr import RaggedCSRReader
    from ogquery.indexed import IndexedSubset
    from ogquery.panel_axis import PanelVariantAxis
    from ogquery.rho import DenseRhoReader
    from ogquery.variant_axis import VariantAxis

    owners: dict[Any, tuple[str, ...]] = {
        VariantAxis: (
            "by_index",
            "by_indices",
            "identity_by_indices",
            "rsids_by_indices",
            "source_alids_by_indices",
            "rows_by_indices",
            "sidecar_records",
        ),
        PanelVariantAxis: (
            "by_index",
            "by_indices",
            "identity_by_indices",
            "rsids_by_indices",
            "source_alids_by_indices",
            "rows_by_indices",
        ),
        DenseRhoReader: ("row", "pair", "matrix"),
        IndexedSubset: ("decode_analysis",),
        ByVariantReader: ("rows_for_variant", "rows_for_variant_range", "decode"),
    }
    for owner, names in owners.items():
        for name in names:
            parameter = inspect.signature(getattr(owner, name)).parameters["expect_space"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (owner, name)
            assert parameter.default is inspect.Parameter.empty, (owner, name)
    for name in ("segment_positions", "eaf_pairs", "n_pairs", "pair_positions"):
        parameter = inspect.signature(getattr(RaggedCSRReader, name)).parameters["expect_spaces"]
        assert parameter.default is inspect.Parameter.empty, name


# ── no unverifiable ids; the one parser; no guessing ─────────────────────────


@pytest.mark.parametrize("bad", [None, "", "  ", "None"])
def test_an_id_that_names_nothing_is_refused(bad: Any) -> None:
    with pytest.raises(ReleaseIdUnknown):
        ReleaseIdentity("store", bad)
    with pytest.raises(ReleaseIdUnknown):
        ReleaseIdentity(bad, "release")


def test_the_manifest_does_not_coerce_a_null_id_to_the_text_none(tmp_path: Path) -> None:
    text = (RELEASES / "dense_completed" / "manifest.json").read_text(encoding="utf-8")
    data = json.loads(text)
    for field in ("release_id", "store_id"):
        broken = {**data, field: None}
        with pytest.raises(ReleaseIdUnknown, match=field):
            StoreManifest.from_dict(broken)


def _copy(tmp_path: Path, name: str, *, manifest: bool = True) -> Path:
    copy = tmp_path / name
    shutil.copytree(RELEASES / name, copy)
    if not manifest:
        (copy / "manifest.json").unlink()
    return copy


def test_a_reader_learns_its_identity_from_the_release_it_opens() -> None:
    from ogquery.manifest import group_identity
    from ogquery.release import open_release, release_arrays
    from ogquery.rho import DenseRhoReader
    from ogquery.top_hits import TopHitTiers
    from ogquery.variant_axis import NoVariantTable, VariantAxis

    ragged = RELEASES / "ragged_completed"
    version = open_release(ragged).manifest.format_version
    axis = VariantAxis(ragged, NoVariantTable(ragged), format_version=version)
    assert axis.index_space == IndexSpace(RAGGED, VARIANT)
    tiers = TopHitTiers(release_arrays(RELEASES / "dense_completed"), clumps=None)
    assert tiers.require(5e-8).index_space == IndexSpace(DENSE, ANALYSIS)
    group = release_arrays(RELEASES / "dense_rho")
    assert DenseRhoReader(group["rho"], 3).index_space.release_id == "v1"
    component = release_arrays(RELEASES / "hybrid_completed" / "dense")
    assert group_identity(component) == HYBRID, "a Dense Component is scoped to its Hybrid"


def test_an_explicit_identity_is_refused_where_a_manifest_says_otherwise() -> None:
    from ogquery.release import release_arrays
    from ogquery.top_hits import TopHitTiers

    group = release_arrays(RELEASES / "dense_completed")
    assert TopHitTiers(group, identity=DENSE).identity == DENSE
    with pytest.raises(ReleaseIdUnknown, match="reachable decides"):
        TopHitTiers(group, identity=ReleaseIdentity("s", "invented"))


def test_a_reader_whose_release_cannot_be_determined_refuses_naming_the_path(
    tmp_path: Path,
) -> None:
    import zarr

    from ogquery.release import release_arrays
    from ogquery.rho import DenseRhoReader
    from ogquery.top_hits import TopHitTiers
    from ogquery.variant_axis import NoVariantTable, VariantAxis

    bare = _copy(tmp_path, "ragged_completed", manifest=False)
    with pytest.raises(ReleaseIdUnknown, match="ragged_completed"):
        VariantAxis(bare, NoVariantTable(bare), format_version="0.3.0")
    with pytest.raises(ReleaseIdUnknown, match="ragged_completed"):
        TopHitTiers(release_arrays(bare))
    rho = _copy(tmp_path, "dense_rho", manifest=False)
    with pytest.raises(ReleaseIdUnknown, match="dense_rho"):
        DenseRhoReader(release_arrays(rho)["rho"], 3)
    with pytest.raises(ReleaseIdUnknown):
        TopHitTiers(zarr.group())
    assert TopHitTiers(zarr.group(), identity=HYBRID).identity == HYBRID


def test_a_zarr_only_copy_does_not_take_a_parents_unrelated_id(tmp_path: Path) -> None:
    """A copied ``data.zarr`` beside a manifest that does not describe it is refused, not
    read as that manifest's release; so is one whose parent holds none."""
    from ogquery.release import release_arrays
    from ogquery.top_hits import TopHitTiers

    parent = tmp_path / "unrelated"
    parent.mkdir()
    shutil.copy(RELEASES / "ragged_completed" / "manifest.json", parent / "manifest.json")
    shutil.copytree(RELEASES / "dense_completed" / "data.zarr", parent / "data.zarr")
    with pytest.raises(ReleaseIdUnknown, match="not this store's manifest"):
        TopHitTiers(release_arrays(parent))
    stray = tmp_path / "stray" / "data.zarr"
    shutil.copytree(RELEASES / "dense_completed" / "data.zarr", stray)
    with pytest.raises(ReleaseIdUnknown):
        TopHitTiers(release_arrays(stray.parent))


def test_a_symlinked_store_is_the_release_it_points_at(tmp_path: Path) -> None:
    from ogquery.release import release_arrays
    from ogquery.top_hits import TopHitTiers

    link = tmp_path / "linked"
    link.mkdir()
    (link / "data.zarr").symlink_to((RELEASES / "dense_completed" / "data.zarr").resolve())
    shutil.copy(RELEASES / "ragged_completed" / "manifest.json", link / "manifest.json")
    tiers = TopHitTiers(release_arrays(link))
    assert tiers.identity == DENSE, "the target's own manifest, not the link's neighbour"


# ── the Reference serialises ─────────────────────────────────────────────────


def test_a_reference_round_trips_through_json_in_every_layout() -> None:
    for name in ("dense_completed", "ragged_completed", "hybrid_completed"):
        with query(name) as q:
            reference = digests.reference_of(q)
        again = digests.Reference.from_json(json.loads(json.dumps(reference.to_json())))
        assert again == reference
        assert again.variant_space == reference.variant_space is not None


def test_a_restored_reference_asks_the_same_questions() -> None:
    path = RELEASES / "ragged_completed"
    with query("ragged_completed") as q:
        reference = digests.reference_of(q)
    again = digests.Reference.from_json(json.loads(json.dumps(reference.to_json())))
    assert not digests.differences(
        digests.release_digests(path, reference=reference),
        digests.release_digests(path, reference=again),
    )
