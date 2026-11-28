"""Where a reader learns the index space it reads in (opengwasdb ADR 0109 §2, ogquery#13).

A reader learns its space from the release it opens, never from a caller: a Hybrid's Dense
Component is known as one by its place (a ``dense/`` directory under a Hybrid manifest of
the same release), and its rows are the Hybrid's panel rows (`PANEL`), whoever opens it.
The manifest it learns from must describe the Zarr store it reads: the store's own records
(layout, completion state) and dimensions (Analyses, variants) are held against it.

The contract bundle is read, never written: every changed release here is a copy.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ogquery import arrays as store_arrays
from ogquery import digests
from ogquery.associations import CsrReader, DenseGridReader, SubsetReader
from ogquery.associations.cells import Cells
from ogquery.csr import RaggedCSRReader
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
from ogquery.manifest import StoreManifest, group_identity, release_identity
from ogquery.release import open_release

RELEASES = Path(__file__).parents[1] / "contract" / "releases"
HYBRID = ReleaseIdentity("hyb", "v1-completed")
RAGGED = ReleaseIdentity("residual-rag", "comp")


def query(path: Path) -> Any:
    return digests.open_query(path)


def copy(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(RELEASES / name, target)
    return target


def component_grid(path: Path) -> DenseGridReader:
    component = open_release(path / "dense")
    return DenseGridReader.open(component.arrays(), component.manifest.encoding)


# ── 1. a Dense Component's space comes from where it is, in the reader ────────


def test_a_dense_components_grid_opened_directly_reads_the_hybrids_panel_rows() -> None:
    """The reviewers' case: row 200 of the panel is `1:350000:A:G`, row 200 of the shared
    axis `1:300000:A:G`. A grid opened on the component, with no facade to say so, reads
    panel rows, and the Hybrid refuses to resolve them through its shared axis."""
    path = RELEASES / "hybrid_completed"
    grid = component_grid(path)
    assert grid.spaces.variant == IndexSpace(HYBRID, PANEL)
    assert grid.spaces.analysis == IndexSpace(HYBRID, ANALYSIS)
    found = grid.read(Cells.variants([200]), expect_spaces=grid.spaces)
    assert len(found["z"]) and set(found.variant_index.tolist()) == {200}
    with query(path) as q:
        with pytest.raises(ReleaseMismatchError):
            list(q.resolve(found))
        rows = list(q._dense.resolve(found))
        assert {row["alid"] for row in rows} == {"1:350000:A:G"}


def test_a_dense_components_subset_opened_directly_is_in_the_hybrids_spaces() -> None:
    """An Indexed Variant Subset records its Dense Component's own Store id, but its rows
    are the Hybrid's panel rows and its columns the Hybrid's Analyses."""
    path = RELEASES / "hybrid_completed"
    for opened in (path, path / "dense"):
        subset = SubsetReader.open(opened, "contract")
        assert subset.spaces.variant == IndexSpace(HYBRID, PANEL), opened
        assert subset.spaces.analysis == IndexSpace(HYBRID, ANALYSIS), opened
        assert open_indexed_subset(opened, "contract").index_space == IndexSpace(
            HYBRID, ANALYSIS
        )


def test_no_caller_says_what_space_a_grid_or_subset_reads() -> None:
    import inspect

    for opener in (DenseGridReader.open, DenseGridReader, SubsetReader.open, SubsetReader):
        assert "variant_space" not in inspect.signature(opener).parameters, opener


def test_a_dense_component_under_an_unreadable_hybrid_manifest_is_refused(
    tmp_path: Path,
) -> None:
    """Whether it is a component cannot be told, so it is not read as a standalone Dense
    release: that would put its panel rows in a variant space of their own."""
    path = copy(tmp_path, "hybrid_completed")
    (path / "manifest.json").write_text("{not json", encoding="utf-8")
    group = store_arrays.open_group(path / "dense" / "data.zarr")
    with pytest.raises(ValueError, match="manifest"):
        group_identity(group)
    with pytest.raises(ValueError, match="manifest"):
        release_identity(path / "dense")


def test_a_dense_component_of_another_release_is_refused(tmp_path: Path) -> None:
    """A ``dense/`` under a Hybrid manifest of another release is no component of it, and
    no release of its own: its rows would be read in one space or the other by guess."""
    path = copy(tmp_path, "hybrid_completed")
    manifest = json.loads((path / "dense" / "manifest.json").read_text(encoding="utf-8"))
    manifest["release_id"] = "another"
    (path / "dense" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReleaseIdUnknown, match="another"):
        open_release(path / "dense")
    with pytest.raises(ReleaseIdUnknown, match="another"):
        release_identity(path / "dense")


def test_a_symlinked_dense_component_shares_the_axis_of_the_hybrid_it_points_into(
    tmp_path: Path,
) -> None:
    """`StoreRelease` asks the real path, as the readers do: a link named ``dense`` in a
    directory with no Hybrid manifest is still that Hybrid's component."""
    path = copy(tmp_path, "hybrid_completed")
    (tmp_path / "elsewhere").mkdir()
    link = tmp_path / "elsewhere" / "dense"
    link.symlink_to(path / "dense", target_is_directory=True)
    release = open_release(link)
    assert release.shared_root is not None
    assert release.shared_root.resolve() == path.resolve()
    assert release_identity(link) == HYBRID


def test_a_hybrids_tiers_refuse_its_dense_components_arrays() -> None:
    """Both are the Hybrid's Analyses, but the component's tiers hold panel rows: physical
    ownership is checked, not identity alone."""
    hybrid = open_release(RELEASES / "hybrid_completed")
    with pytest.raises(ReleaseIdUnknown, match="dense"):
        hybrid.top_hit_tiers(hybrid.dense_component().arrays())
    assert hybrid.top_hit_tiers(hybrid.arrays()).identity == HYBRID


def test_a_facade_verifies_a_readers_stamp_and_never_overwrites_it() -> None:
    """An answer a reader stamped with another space is refused, not relabelled."""
    with query(RELEASES / "dense_completed") as q:
        grid = q._grid
        found = grid.read(Cells.analysis(0), expect_spaces=grid.spaces)
        foreign = IndexSpace(ReleaseIdentity("elsewhere", "other"), VARIANT)

        class Misstamping:
            spaces = grid.spaces

            def read(self, *args: Any, **kwargs: Any) -> Any:
                return replace(found, variant_space=foreign)

        q._grid = Misstamping()
        with pytest.raises(ReleaseMismatchError, match="elsewhere"):
            q.analysis("a")


# ── 2. the manifest found must describe the Zarr store ───────────────────────


def _beside(tmp_path: Path, data: str, manifest: str) -> Path:
    """`data`'s `data.zarr`, copied beside `manifest`'s release files but its data."""
    target = tmp_path / f"{data}_beside_{manifest}"
    shutil.copytree(RELEASES / manifest, target, ignore=shutil.ignore_patterns("data.zarr"))
    shutil.copytree(RELEASES / data / "data.zarr", target / "data.zarr")
    return target


def test_a_dense_store_beside_another_dense_releases_manifest_is_refused(
    tmp_path: Path,
) -> None:
    """The reviewers' case: `dense_completed`'s store beside `dense_rho`'s manifest. Both
    are Dense; the store records another completion state, and other dimensions."""
    path = _beside(tmp_path, "dense_completed", "dense_rho")
    with pytest.raises(ReleaseIdUnknown, match="not this store's manifest"):
        group_identity(store_arrays.open_group(path / "data.zarr"))


def test_a_dense_store_of_other_dimensions_is_refused_where_its_records_agree(
    tmp_path: Path,
) -> None:
    """`dense_observed` and `dense_rho` record the same layout and completion state; the
    number of Analyses and variants tell them apart."""
    data = StoreManifest.load(RELEASES / "dense_observed")
    manifest = StoreManifest.load(RELEASES / "dense_rho")
    assert (data.primary_layout, data.completion_state) == (
        manifest.primary_layout,
        manifest.completion_state,
    )
    path = _beside(tmp_path, "dense_observed", "dense_rho")
    with pytest.raises(ReleaseIdUnknown, match="not this store's manifest"):
        group_identity(store_arrays.open_group(path / "data.zarr"))


def test_a_ragged_store_beside_another_ragged_releases_manifest_is_refused(
    tmp_path: Path,
) -> None:
    path = _beside(tmp_path, "ragged_observed", "ragged_completed")
    with pytest.raises(ReleaseIdUnknown, match="not this store's manifest"):
        RaggedCSRReader(path)


def test_a_csr_whose_data_zarr_points_into_another_release_is_refused(tmp_path: Path) -> None:
    """A release directory whose ``data.zarr`` is a link into another release reads that
    release's associations; its own manifest does not describe them."""
    path = copy(tmp_path, "ragged_completed")
    shutil.rmtree(path / "data.zarr")
    other = copy(tmp_path, "ragged_observed")
    (path / "data.zarr").symlink_to(other / "data.zarr", target_is_directory=True)
    with pytest.raises(ReleaseIdUnknown, match="data.zarr"):
        RaggedCSRReader(path)
    with pytest.raises(ReleaseIdUnknown, match="data.zarr"):
        CsrReader.open(path)


def test_every_contract_release_binds_to_its_own_store() -> None:
    for name in sorted(p.name for p in RELEASES.iterdir() if p.is_dir()):
        release = open_release(RELEASES / name)
        assert group_identity(release.arrays()) == release.manifest.identity, name
        if (RELEASES / name / "dense").is_dir():
            component = release.dense_component()
            assert group_identity(component.arrays()) == release.manifest.identity, name


# ── 4. a CSR with no manifest is read with the identity it is given ──────────


def test_a_csr_with_no_manifest_reads_with_the_identity_it_is_given(tmp_path: Path) -> None:
    path = copy(tmp_path, "ragged_completed")
    encoding = StoreManifest.load(path).encoding
    (path / "manifest.json").unlink()
    with pytest.raises(ReleaseIdUnknown):
        CsrReader.open(path, encoding, requires_mask=True)
    reader = CsrReader.open(path, encoding, requires_mask=True, identity=RAGGED)
    assert reader.spaces.analysis == IndexSpace(RAGGED, ANALYSIS)
    found = reader.read(Cells.analysis(0), expect_spaces=reader.spaces)
    assert len(found["z"]) and np.all(found.analysis_index == 0)


# ── review round 3 ───────────────────────────────────────────────────────────


def test_an_analyses_table_read_from_elsewhere_is_not_stamped_with_the_directorys_identity(
) -> None:
    """The reviewers' case: `dense_rho`'s `analyses.tsv` given to `dense_completed` read its
    Analysis `a` from another column under `dense_completed`'s stamp."""
    from ogquery.analyses import AnalysesIndex
    from ogquery.facade import StoreQuery

    with pytest.raises(ReleaseIdUnknown, match="analyses.tsv"):
        AnalysesIndex(RELEASES / "dense_completed", table=RELEASES / "dense_rho" / "analyses.tsv")
    own = AnalysesIndex(
        RELEASES / "dense_completed", table=RELEASES / "dense_completed" / "analyses.tsv"
    )
    assert own.require_index("a") == 0
    component = open_release(RELEASES / "hybrid_completed" / "dense")
    shared = AnalysesIndex(component.path, table=component.analyses_path)
    assert shared.index_space == IndexSpace(HYBRID, ANALYSIS)
    with StoreQuery(open_release(RELEASES / "dense_completed")) as q:
        assert len(q.analysis("a")["z"])


def test_a_hybrid_store_swapped_for_another_hybrids_is_refused(tmp_path: Path) -> None:
    """The reviewers' case: `hybrid_observed`'s store in a copy of `hybrid_completed` agrees
    on layout, state and dimensions, but holds 200 Overflow associations, not 199."""
    path = copy(tmp_path, "hybrid_completed")
    shutil.rmtree(path / "data.zarr")
    shutil.copytree(RELEASES / "hybrid_observed" / "data.zarr", path / "data.zarr")
    with pytest.raises(ReleaseIdUnknown, match="Overflow associations"):
        group_identity(store_arrays.open_group(path / "data.zarr"))
    with pytest.raises(ReleaseIdUnknown):
        digests.open_query(path)


def test_unreadable_store_metadata_is_refused_not_skipped(tmp_path: Path) -> None:
    from ogquery.manifest import store_spaces

    path = copy(tmp_path, "dense_completed")
    (path / "data.zarr" / "z" / "zarr.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ReleaseIdUnknown, match="unreadable"):
        store_spaces(path / "data.zarr")
    path = copy(tmp_path, "ragged_completed")
    (path / "variant_alid_order.npy").write_bytes(b"not an array")
    with pytest.raises(ReleaseIdUnknown, match="unreadable"):
        store_spaces(path / "data.zarr")


def test_absent_optional_metadata_is_skipped(tmp_path: Path) -> None:
    """An archived 0.1.0 Overflow has no by-variant index: there are no rows to compare."""
    from ogquery.manifest import store_spaces

    path = copy(tmp_path, "ragged_completed")
    shutil.rmtree(path / "data.zarr" / "ragged" / "by_variant")
    assert store_spaces(path / "data.zarr").analysis.identity == RAGGED


def test_a_quoted_multiline_label_is_one_analysis(tmp_path: Path) -> None:
    import csv

    from ogquery.manifest import store_spaces

    path = copy(tmp_path, "dense_completed")
    with (path / "analyses.tsv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, rows = list(reader.fieldnames or ()), list(reader)
    rows[0]["analysis_label"] = "a label\nover two lines"
    with (path / "analyses.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assert (path / "analyses.tsv").read_text(encoding="utf-8").count("\n") == 4
    assert store_spaces(path / "data.zarr").analysis.identity == ReleaseIdentity("s", "comp")


def test_tiers_of_a_data_zarr_linked_into_another_release_are_refused(tmp_path: Path) -> None:
    """The reviewers' case: `dense_completed`'s `data.zarr` a link to `dense_rho`'s returned
    `dense_rho`'s tiers, with `root=` or without."""
    path = copy(tmp_path, "dense_completed")
    other = copy(tmp_path, "dense_rho")
    shutil.rmtree(path / "data.zarr")
    (path / "data.zarr").symlink_to(other / "data.zarr", target_is_directory=True)
    release = open_release(path)
    with pytest.raises(ReleaseIdUnknown, match="data.zarr"):
        release.top_hit_tiers()
    with pytest.raises(ReleaseIdUnknown, match="data.zarr"):
        release.top_hit_tiers(release.arrays())


def test_an_invented_identity_is_refused_where_the_linked_store_has_a_manifest(
    tmp_path: Path,
) -> None:
    """The reviewers' case: a directory with no manifest, its `data.zarr` a link into a
    release that has one, read under an `identity=` the caller made up."""
    other = copy(tmp_path, "ragged_observed")
    path = tmp_path / "bare"
    path.mkdir()
    (path / "data.zarr").symlink_to(other / "data.zarr", target_is_directory=True)
    encoding = StoreManifest.load(other).encoding
    invented = ReleaseIdentity("made", "up")
    with pytest.raises(ReleaseIdUnknown, match="manifest"):
        RaggedCSRReader(path, encoding, identity=invented)
    with pytest.raises(ReleaseIdUnknown, match="manifest"):
        CsrReader.open(path, encoding, requires_mask=False, identity=invented)


def test_a_csr_reader_loads_its_manifest_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[Path] = []
    original = StoreManifest.load.__func__  # type: ignore[attr-defined]

    def counting(cls: Any, path: Any) -> StoreManifest:
        loads.append(Path(path))
        return original(cls, path)  # type: ignore[no-any-return]

    monkeypatch.setattr(StoreManifest, "load", classmethod(counting))
    CsrReader.open(RELEASES / "ragged_completed")
    assert len(loads) == 1, loads
