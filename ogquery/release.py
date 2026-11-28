"""A Store Release, opened read-only: its paths, its manifest and its arrays.

`open_release` reads a release's manifest and refuses a `format_version` this
build cannot interpret before anything else is read (ADR 0038 §2).
`StoreRelease` is the opened release every reader holds: where its artifacts
live, its Zarr data group opened read-only, its Hybrid Dense Component, its
Variant Axis, and the query facade over it.

From format 0.3.0 a Hybrid's Dense Component carries no Variant Axis and no
`analyses.tsv` of its own (#472): opened as a release, its `variant_axis` is the
shared one seen through `dense_to_shared.npy` (`panel_axis.PanelVariantAxis`), and
its `analyses_path` is the root's. Every reader of the component -- the query
facade, completion, the validator, the Indexed Variant Subset builder -- gets
them from here, so none reads a second copy.

Writing to a release -- staging, publishing, amending its provenance, opening
its arrays for writing or its `index.sqlite` -- is `opengwasdb.store.open`'s,
whose `OpenGWASDBStore` is this class with those added (#464). So is the Store
Variant Table's 0.2.x access (`opengwasdb.variants.legacy_table`): a 0.2 release
opened with no `legacy` opener has no Variant Axis, and `variant_axis` refuses it
(`VariantTableReaderRequired`), naming its `format_version` and how to make it a 0.3.0
one: `ogdb rewrite-release` for a 0.2.3 release, a rebuild from source for any earlier
one. A 0.3.0 release needs none: its axis reads the sidecars, and its
table refuses every read (`NoVariantTable`, #474).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ogquery.arrays import open_group
from ogquery.enums import PrimaryStorageLayout
from ogquery.format import (
    LAST_0_2_FORMAT_VERSION,
    CpraNormalForm,
    carries_cpra,
    carries_variant_offsets,
    check_format_version,
    cpra_normal_form,
    dense_component_shares_axis,
    record_opened,
    require_equal_declarations,
)
from ogquery.guard import ReleaseIdUnknown
from ogquery.hybrid import DENSE_TO_SHARED_FILENAME
from ogquery.manifest import StoreManifest, release_scope_of
from ogquery.panel_axis import AnyVariantAxis, PanelVariantAxis
from ogquery.top_hits import TopHitTiers
from ogquery.variant_axis import (
    LegacyTableOpener,
    LegacyVariantTable,
    NoVariantTable,
    VariantAxis,
)

if TYPE_CHECKING:
    from ogquery.facade import HybridStoreQuery, RaggedStoreQuery, StoreQuery


#: opengwasdb's command that rewrites a release of `REWRITABLE_FORMAT_VERSION` as 0.3.0,
#: which the Store Reader reads alone. It refuses every other version.
REWRITE_COMMAND = "ogdb rewrite-release"

#: The one version `REWRITE_COMMAND` rewrites: 0.2.3, the last of the 0.2 series, whose
#: features 0.3.0 keeps (`format.LAST_0_2_FORMAT_VERSION`). A 0.1.0 or 0.2.0-0.2.2 release
#: lacks one of them -- reference-first indels, the SE count -- and is rebuilt from source.
REWRITABLE_FORMAT_VERSION = LAST_0_2_FORMAT_VERSION


class VariantTableReaderRequired(ValueError):
    """A 0.2.x or 0.1.0 release opened without a reader for its Store Variant Table.

    Those formats answer a query's variants from `variants.tsv.gz` (BGZF and tabix), which
    needs pysam, and from `index.sqlite`; ogquery reads neither. The caller either passes
    a reader, `open_release(path, legacy=...)`, as opengwasdb's `open_store` does, or makes
    the release a 0.3.0 one: a 0.2.3 release is rewritten with `REWRITE_COMMAND`, and any
    earlier one is rebuilt from source, since the rewrite refuses it (opengwasdb's
    `store.rewrite`). The message says which applies. A `ValueError`, as the refusal was
    before it had a name.
    """

    def __init__(self, path: Path, format_version: str) -> None:
        self.path = path
        self.format_version = format_version
        if format_version == REWRITABLE_FORMAT_VERSION:
            remedy = (
                f"Rewrite it as 0.3.0 with `{REWRITE_COMMAND}` (opengwasdb), which ogquery "
                "reads alone"
            )
        else:
            remedy = (
                f"Rebuild it from source at format 0.3.0, which ogquery reads alone; "
                f"`{REWRITE_COMMAND}` rewrites only a {REWRITABLE_FORMAT_VERSION} release"
            )
        super().__init__(
            f"{path} is format_version {format_version}, whose queries read its Store "
            "Variant Table (a BGZF/tabix file), and it was opened without a reader for its "
            f"Store Variant Table. {remedy}, or pass open_release(..., legacy=...), as "
            "opengwasdb's open_store does"
        )


def release_paths(path: Path) -> dict[str, Path]:
    return {
        "manifest": path / "manifest.json",
        "data": path / "data.zarr",
        "index": path / "index.sqlite",
        "analyses": path / "analyses.tsv",
        "variant_table": path / "variants.tsv.gz",
        "variant_tabix": path / "variants.tsv.gz.tbi",
        "variant_offsets": path / "variant_offsets.npy",
    }


def release_arrays(release: str | Path) -> Any:
    """The `data.zarr` group of the release at `release`, read-only.

    The one read-only open of a release's arrays: `StoreRelease.arrays` is this, for
    an opened release, and a reader holding only a path -- a staged release not yet
    published, a Hybrid's Dense Component, a builder adding to a release -- calls it
    directly (#473).
    """
    return open_group(release_paths(Path(release))["data"])


class ReleasePaths:
    """Artifact-path properties shared by an opened release and a staged one.

    Both ``StoreRelease`` (and so ``OpenGWASDBStore``) and ``StagedRelease`` set ``_paths`` (via
    ``release_paths()``) in ``__post_init__`` and inherit these instead of
    each re-deriving the same suffixes off ``self.path``.
    """

    _paths: dict[str, Path]

    @property
    def manifest_path(self) -> Path:
        return self._paths["manifest"]

    @property
    def data_path(self) -> Path:
        return self._paths["data"]

    @property
    def index_path(self) -> Path:
        return self._paths["index"]

    @property
    def analyses_path(self) -> Path:
        return self._paths["analyses"]

    @property
    def variant_table_path(self) -> Path:
        return self._paths["variant_table"]

    @property
    def variant_tabix_path(self) -> Path:
        return self._paths["variant_tabix"]

    @property
    def variant_offsets_path(self) -> Path:
        return self._paths["variant_offsets"]


@dataclass(frozen=True)
class StoreRelease(ReleasePaths):
    """A local Store Release opened from an explicit path, read-only.

    Frozen: the manifest a caller holds never changes under them. `legacy` opens
    the release's 0.2.x Store Variant Table (`LegacyTableOpener`); it is carried
    to the Hybrid Dense Component, and it is what `variant_axis` reads through.
    """

    path: Path
    manifest: StoreManifest
    legacy: LegacyTableOpener | None = field(default=None, repr=False, compare=False)
    _paths: dict[str, Path] = field(init=False, repr=False)
    _shared_root: Path | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", release_paths(self.path))
        # Whether it is a Dense Component is asked as every reader asks it, of the real path,
        # requiring the Hybrid to be of its own release (`release_scope_of`).
        hybrid = release_scope_of(self.path, manifest=self.manifest).hybrid_root
        root = None
        if hybrid is not None and dense_component_shares_axis(self.manifest.format_version):
            given = self.path.parent
            root = given if given.resolve() == hybrid else hybrid
        object.__setattr__(self, "_shared_root", root)
        if root is not None:
            self._paths["analyses"] = release_paths(root)["analyses"]

    @property
    def shared_root(self) -> Path | None:
        """The Hybrid release whose Variant Axis and `analyses.tsv` this one reads: set
        for a Dense Component from format 0.3.0 (#472), None for every other release."""
        return self._shared_root

    @property
    def release_id(self) -> str:
        return self.manifest.release_id

    @property
    def cpra_normal_form(self) -> CpraNormalForm:
        """The form this release's keys and signs take (spec §21.2): its manifest entry from
        0.4.0, derived from its stamp before."""
        return cpra_normal_form(self.manifest)

    def dense_component(self) -> StoreRelease:
        """Open a Hybrid release's nested Dense Component release.

        The Dense Component (ADR 0026) is a Dense Store Release at
        ``<store>/dense``, built and completed by the unchanged dense machinery --
        it is opened the same way any release is. From 0.3.0 its Variant Axis and
        `analyses_path` are its Hybrid's (`shared_root`, #472).
        """
        self._require_hybrid()
        return open_release(self.path / "dense", legacy=self.legacy)

    def _require_hybrid(self) -> None:
        if self.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
            raise ValueError(
                "dense_component() is only valid for a Hybrid release "
                f"(primary_layout={self.manifest.primary_layout})"
            )

    def arrays(self) -> Any:
        """Open this release's ``data.zarr`` group read-only (`release_arrays`)."""
        return release_arrays(self.path)

    def top_hit_tiers(self, root: Any = None) -> TopHitTiers:
        """This release's Top-Hit tiers, read from `root` (its `data.zarr`, opened here if
        None), knowing which tier's Clumps its manifest declares (spec §2, §18.1).

        `root` must be this release's own `data.zarr`, by its real path: a Hybrid and its
        Dense Component share an identity, but the component's tiers hold panel rows, so
        identity alone cannot tell one's arrays from the other's. A `data.zarr` that is a
        link into another release is not its own either, with `root` or without."""
        own = self._own_data()
        if root is None:
            root = self.arrays()
        self._require_own_arrays(root, own)
        return TopHitTiers(root, clumps=self.manifest.clumps)

    def _own_data(self) -> Path:
        """The real path of this release's `data.zarr`, refused where it resolves out of
        the release's own directory (a link into another release)."""
        own = (self.path / "data.zarr").resolve()
        if own.parent != self.path.resolve():
            raise ReleaseIdUnknown(
                f"{self.path / 'data.zarr'} resolves to {own}, outside the release at "
                f"{self.path}: its arrays are another release's"
            )
        return own

    def _require_own_arrays(self, root: Any, own: Path) -> None:
        local = getattr(root.store, "root", None)
        found = None if local is None else Path(local).resolve()
        if found != own or str(getattr(root, "path", "")).strip("/"):
            where = repr(root.store) if found is None else f"{found}/{root.path}".rstrip("/")
            raise ReleaseIdUnknown(
                f"the arrays at {where} are not the data.zarr of the release at {self.path} "
                f"({own}): its tiers are read from its own arrays"
            )

    def variant_axis(self, *, aliases: bool) -> AnyVariantAxis:
        """This release's Variant Axis; `aliases` also consults `variant_aliases`. For a
        0.3.0 Hybrid's Dense Component, the shared axis in its panel rows (#472)."""
        if self._shared_root is not None:
            # The Dense Component's rows, named by the shared axis: one ALID, position
            # and rsID index, the root's (#472). A 0.3.0 release has no `variant_aliases`.
            root = open_release(self._shared_root, legacy=self.legacy)
            return PanelVariantAxis(
                root.own_variant_axis(aliases=False),
                np.load(self.path / DENSE_TO_SHARED_FILENAME, mmap_mode="r"),
            )
        return self.own_variant_axis(aliases=aliases)

    def own_variant_axis(self, *, aliases: bool) -> VariantAxis:
        """The Variant Axis this release's own files hold: every release's but a 0.3.0
        Dense Component's, which holds none and is refused (`DenseComponentSharesAxis`)."""
        if self.legacy is not None:
            table: LegacyVariantTable = self.legacy(self.path, aliases=aliases)
        elif carries_variant_offsets(self.manifest.format_version):
            raise VariantTableReaderRequired(self.path, self.manifest.format_version)
        else:
            # 0.3.0 reads its sidecars, never the table (ADR 0105 §7, §8, #474).
            table = NoVariantTable(self.path)
        return VariantAxis(
            self.path,
            table,
            format_version=self.manifest.format_version,
        )

    def query(self) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
        """The query facade over this release, by its layout (ADR 0033)."""
        from ogquery.facade import query_release

        return query_release(self)


def open_release(path: str | Path, *, legacy: LegacyTableOpener | None = None) -> StoreRelease:
    """Open a local Store Release directory, read-only.

    The function intentionally opens exactly the path supplied by the
    caller. Release discovery and default selection belong to a higher-level
    catalogue. Raises ``UnsupportedFormatVersion`` if the release declares a
    format_version this build cannot interpret (ADR 0038 §2).
    """
    store_path = Path(path)
    manifest = StoreManifest.load(store_path)
    check_format_version(manifest.format_version, source=f"release at {store_path}")
    release = StoreRelease(path=store_path, manifest=manifest, legacy=legacy)
    _require_hybrid_declarations_equal(release)
    # Refused here when the entry is absent, present where it is forbidden, or not 0.4.0's.
    record_opened(release.cpra_normal_form, manifest.release_id)
    return release


def _require_hybrid_declarations_equal(release: StoreRelease) -> None:
    """A 0.4.0 Hybrid declares its form, reference sequence and admission in both manifests,
    its root's and its Dense Component's, each equal in the two (spec §2): refused here, from
    either side, before either is read for its own form."""
    if not carries_cpra(release.manifest.format_version):
        return
    if release.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        component = release.path / "dense"
        if (component / "manifest.json").exists():
            require_equal_declarations(
                release.manifest, str(release.path / "manifest.json"),
                StoreManifest.load(component), str(component / "manifest.json"),
            )
        return
    root = release_scope_of(release.path, manifest=release.manifest).hybrid_root
    if root is not None:
        require_equal_declarations(
            release.manifest, str(release.path / "manifest.json"),
            StoreManifest.load(root), str(root / "manifest.json"),
        )
