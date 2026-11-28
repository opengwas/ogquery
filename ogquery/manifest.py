"""A Store Release's `manifest.json`, as the model every reader opens it into.

Reading it checks the release's `format_version` first and parses its declared
encoding (spec §2, §6a). Rewriting a manifest's coverage to the written spelling
is a writer's job, `opengwasdb.model.manifest.normalise_coverage`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ogquery.encoding.plan import StoreEncoding, UnsupportedEncoding
from ogquery.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from ogquery.format import check_format_version
from ogquery.guard import (
    ANALYSIS,
    PANEL,
    VARIANT,
    CellSpaces,
    IndexSpace,
    ReleaseIdentity,
    ReleaseIdUnknown,
    identifier,
)
from ogquery.hybrid import DENSE_TO_SHARED_FILENAME, UnreadableHybridManifest, hybrid_manifest_of


def _encoding_from_dict(data: dict[str, Any]) -> StoreEncoding:
    """The release's declared plan. Never inferred, and never defaulted.

    The version is parsed first, because a release this build cannot read must
    be refused by version rather than by whatever its `encoding` block happens
    to say (spec §6a). Past that, every readable release declares a block:
    there is one format, so there is no version at which an absent block means
    something (issue #143).

    Falling back to a plan the release did not state would decode an `int16`
    plane as `float16` and hand back plausible, wrong z-scores -- the exact
    failure an explicit declaration exists to prevent -- so a missing block is
    refused rather than guessed at.

    Issue #157's per-kind version gate is gone with the versions it gated: it
    stopped a release stamped 1.0 or 2.0 declaring a kind only a later format
    admitted, and this build reads exactly one format, whose block may declare
    any kind the parser implements.
    """
    version = str(data["format_version"])
    check_format_version(version, source="release")
    declared = data.get("encoding")
    if declared is None:
        raise UnsupportedEncoding(
            f"release declares format_version={version!r} but no `encoding` block; "
            "the encoding is required (spec §6a), and a release that does not "
            "declare one cannot be decoded"
        )
    return StoreEncoding.from_manifest(declared)


@dataclass(frozen=True)
class RegionsEntry:
    """The manifest's `regions` entry: the region table a Ragged Store Release carries
    (`regions.tsv`, spec §17), by its path in the release and the SHA-256 of its bytes.

    It is how a reader finds the table (spec §21.2's manifest-based feature discovery),
    as `encoding` declares the planes: a release carrying no table has no entry, and its
    manifest is the one it would have been without this field.
    """

    path: str
    sha256: str

    def to_manifest(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_manifest(cls, data: Any) -> RegionsEntry:
        """Read a declared entry, refusing one that is not `{path, sha256}`."""
        if (
            not isinstance(data, dict)
            or set(data) != {"path", "sha256"}
            or not all(isinstance(value, str) for value in data.values())
        ):
            raise ValueError(
                f"manifest.json's regions entry {data!r} is not {{path, sha256}} (spec §17)"
            )
        return cls(path=data["path"], sha256=data["sha256"])


@dataclass(frozen=True)
class StoreManifest:
    """Minimal manifest required to identify and open a Store Release."""

    store_id: str
    release_id: str
    format_version: str
    primary_layout: PrimaryStorageLayout
    association_coverage: AssociationCoverage
    completion_state: CompletionState
    reference_assembly: str
    #: How this release's statistic planes are encoded (ADR 0037, issue #119).
    #: Required, with no default: a plan nobody stated is a plan nobody can be
    #: held to, and every release this build reads or writes declares one
    #: (issue #143). The plan is read, never re-derived -- re-running the
    #: decision tree on read would mean a later threshold change silently
    #: altered how existing stores decode.
    encoding: StoreEncoding
    created_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    #: The region table the release carries (`RegionsEntry`); `None` when it carries
    #: none, and then omitted from the manifest entirely, as an absent `encoding.n` is.
    regions: RegionsEntry | None = None
    #: The `clumps` entry (spec §2): which tier's Clumps the release declares, None when it
    #: declares none. Read, never written: `to_dict` leaves it out, so a manifest written
    #: from this object -- a completion's, which rewrites the tier the Clumps were parallel
    #: to -- never declares Clumps it does not carry (ADR 0106 §10). Only the step that
    #: writes them declares them, in place (`OpenGWASDBStore.set_manifest_entry`).
    clumps: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoreManifest:
        return cls(
            encoding=_encoding_from_dict(data),
            store_id=identifier(data["store_id"], "manifest store_id"),
            release_id=identifier(data["release_id"], "manifest release_id"),
            format_version=str(data["format_version"]),
            primary_layout=PrimaryStorageLayout(data["primary_layout"]),
            association_coverage=AssociationCoverage(data["association_coverage"]),
            completion_state=CompletionState(data["completion_state"]),
            reference_assembly=str(data["reference_assembly"]),
            created_at=data.get("created_at"),
            provenance=dict(data.get("provenance", {})),
            regions=(
                RegionsEntry.from_manifest(data["regions"]) if "regions" in data else None
            ),
            clumps=data.get("clumps"),
        )

    @property
    def identity(self) -> ReleaseIdentity:
        """The Store and release this manifest describes."""
        return ReleaseIdentity(self.store_id, self.release_id)

    @classmethod
    def load(cls, path: str | Path) -> StoreManifest:
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "store_id": self.store_id,
            "release_id": self.release_id,
            "format_version": self.format_version,
            "primary_layout": self.primary_layout.value,
            "association_coverage": self.association_coverage.value,
            "completion_state": self.completion_state.value,
            "reference_assembly": self.reference_assembly,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "encoding": self.encoding.to_manifest(),
        }
        if self.regions is not None:
            data["regions"] = self.regions.to_manifest()
        return data


@dataclass(frozen=True)
class ReleaseScope:
    """What the release in a directory numbers its indices in (`ogquery.guard`).

    Its own manifest, and the identity and variant axis its indices are scoped to: its
    own, or, for a Hybrid's Dense Component, the Hybrid's identity and the Hybrid's panel
    rows (`PANEL`). Found by `release_scope_of`, the one place a reader learns its space.
    """

    #: The release directory's real path, symlinks followed.
    path: Path
    manifest: StoreManifest
    identity: ReleaseIdentity
    #: `VARIANT` for a release's own variant axis, `PANEL` for a Dense Component's rows.
    variant: str
    #: The real path of the Hybrid this is the Dense Component of, else None.
    hybrid_root: Path | None = None

    @property
    def spaces(self) -> CellSpaces:
        """Its cells' numbering: its rows (variants) and its columns (Analyses)."""
        return CellSpaces(
            IndexSpace(self.identity, self.variant), IndexSpace(self.identity, ANALYSIS)
        )

    @property
    def analyses_path(self) -> Path:
        """The `analyses.tsv` its columns are numbered by: a Dense Component's Hybrid's."""
        return (self.path if self.hybrid_root is None else self.hybrid_root) / "analyses.tsv"


def release_scope_of(
    directory: str | Path, *, manifest: StoreManifest | None = None
) -> ReleaseScope:
    """The scope of the release in `directory`, from its `manifest.json` (passed as
    `manifest` where the caller has already loaded it, so each is read once).

    A Dense Component is told from its place, on its real path: a ``dense/`` directory under
    a Hybrid manifest (`hybrid.hybrid_manifest_of`, the one structural test, which
    `hybrid_root_of` asks too). It is one only if it is a Dense release of the Hybrid's own
    release; a ``dense/`` under a Hybrid of another release is refused, and so is one under a
    manifest that cannot be read, rather than taken for a standalone release."""
    path = Path(directory).resolve()
    if manifest is None:
        manifest = _load(path / "manifest.json")
    try:
        found = hybrid_manifest_of(path)
    except UnreadableHybridManifest as error:
        raise ReleaseIdUnknown(
            f"cannot tell whether {path} is a Dense Component: {error}"
        ) from error
    if found is None:
        return ReleaseScope(path, manifest, manifest.identity, VARIANT)
    root, recorded = found
    try:
        hybrid = StoreManifest.from_dict(recorded)
    except (ValueError, KeyError, TypeError) as error:
        raise ReleaseIdUnknown(
            f"cannot tell whether {path} is a Dense Component: {root / 'manifest.json'} is "
            f"unreadable ({type(error).__name__}: {error})"
        ) from error
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise ReleaseIdUnknown(
            f"{path} sits where a Hybrid's Dense Component does but is a "
            f"{manifest.primary_layout.value} release"
        )
    if hybrid.release_id != manifest.release_id:
        raise ReleaseIdUnknown(
            f"{path} is release {manifest.release_id!r} but sits where the Dense Component of "
            f"the Hybrid release {hybrid.release_id!r} at {root} does: it is neither that "
            "Hybrid's component nor a release of its own"
        )
    return ReleaseScope(path, manifest, hybrid.identity, PANEL, hybrid_root=root)


def _load(manifest_path: Path) -> StoreManifest:
    try:
        return StoreManifest.load(manifest_path)
    except ReleaseIdUnknown as error:
        raise ReleaseIdUnknown(f"{manifest_path}: {error}") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReleaseIdUnknown(
            f"cannot learn which release {str(manifest_path.parent)!r} is: {manifest_path} is "
            f"unreadable ({type(error).__name__}: {error})"
        ) from error


def release_identity(
    directory: str | Path, *, explicit: ReleaseIdentity | None = None
) -> ReleaseIdentity:
    """The identity of the release in `directory` (`release_spaces`)."""
    return release_spaces(directory, explicit=explicit).analysis.identity


def release_spaces(directory: str | Path, *, explicit: ReleaseIdentity | None = None) -> CellSpaces:
    """The spaces of the release in `directory`, from its `manifest.json` (`release_scope_of`).
    An `explicit` identity is accepted only where no manifest is there, and refused where one
    is and says otherwise. With neither, the reader refuses."""
    directory = Path(directory).resolve()
    if not (directory / "manifest.json").is_file():
        return _explicit(
            explicit, f"cannot learn which release {str(directory)!r} is: it has no manifest.json"
        )
    found = release_scope_of(directory).spaces
    _agrees(found, explicit, directory / "manifest.json")
    return found


def group_identity(group: Any, *, explicit: ReleaseIdentity | None = None) -> ReleaseIdentity:
    """The identity of the release whose `data.zarr` holds the Zarr `group` (`group_spaces`)."""
    return group_spaces(group, explicit=explicit).analysis.identity


def group_spaces(group: Any, *, explicit: ReleaseIdentity | None = None) -> CellSpaces:
    """The spaces of the release whose `data.zarr` holds the Zarr `group`.

    The store's real path (symlinks followed) must be a directory named ``data.zarr`` whose
    parent holds a `manifest.json`, and that manifest must describe this store
    (`store_spaces`). A store that is not a local ``data.zarr`` has no manifest to ask, and
    needs `explicit`; its rows are then that release's own variant axis."""
    root = getattr(group.store, "root", None)
    if root is None:
        return _explicit(
            explicit,
            f"cannot learn which release the Zarr store {group.store!r} belongs to: it has no "
            "local directory",
        )
    return store_spaces(Path(root), explicit=explicit)


def store_spaces(store: str | Path, *, explicit: ReleaseIdentity | None = None) -> CellSpaces:
    """The spaces of the release whose ``data.zarr`` is at `store` (a directory, or a link
    to one), bound to it.

    Never guessed, and never more than the store's own records can tell. The manifest beside
    the store's real path must describe it, by the cheap discriminators the store records:

    - its layout: a Dense store has no ``ragged/`` component and a Ragged or Hybrid one does,
      and a store that records its ``layout`` records the manifest's;
    - its completion state, where the store records one (a Dense grid's, or a Ragged
      release's sequence; a Hybrid's Overflow is observed by construction and says nothing
      of the release's);
    - its dimensions: the Analyses its grid or sequence holds against the release's
      `analyses.tsv` (a Dense Component's, its Hybrid's), and the rows of its grid (or of a
      by-variant index) against the release's variant axis (a Dense Component's,
      ``dense_to_shared.npy``).

    **The limit.** A store that agrees with another release's manifest on all of these -- an
    exact structural twin, two releases built alike from inputs of the same shape -- cannot
    be told from its own: nothing in a store names the release that wrote it. Telling them
    apart needs an identity the writer stamps into the store (opengwasdb's to add)."""
    path = Path(store).resolve()
    manifest_path = path.parent / "manifest.json"
    if path.name != "data.zarr" or not manifest_path.is_file():
        return _explicit(
            explicit,
            f"cannot learn which release the Zarr store at {str(path)!r} belongs to: it is not "
            "the data.zarr of a release with a manifest.json",
        )
    scope = release_scope_of(path.parent)
    faults = _store_faults(path, scope)
    if faults:
        raise ReleaseIdUnknown(
            f"{manifest_path} is not this store's manifest: {'; '.join(faults)} ({path})"
        )
    _agrees(scope.spaces, explicit, manifest_path)
    return scope.spaces


def data_spaces(release: str | Path, *, explicit: ReleaseIdentity | None = None) -> CellSpaces:
    """The spaces of what a reader opening ``<release>/data.zarr`` reads (`store_spaces`).
    The store must be that release's own: a ``data.zarr`` that is a link into another
    release is refused, since the manifest at `release` does not describe it. A release
    directory with no manifest takes `explicit`."""
    directory = Path(release).resolve()
    store = Path(release) / "data.zarr"
    if not (directory / "manifest.json").is_file():
        return _explicit(
            explicit, f"cannot learn which release {str(directory)!r} is: it has no manifest.json"
        )
    if store.exists() and store.resolve().parent != directory:
        raise ReleaseIdUnknown(
            f"{store} resolves to {store.resolve()}, outside the release at {directory}: its "
            f"manifest does not describe the data.zarr it would read"
        )
    return store_spaces(store, explicit=explicit)


def _explicit(explicit: ReleaseIdentity | None, cannot: str) -> CellSpaces:
    if explicit is None:
        raise ReleaseIdUnknown(f"{cannot}; pass identity= where no manifest is reachable")
    return CellSpaces(IndexSpace(explicit, VARIANT), IndexSpace(explicit, ANALYSIS))


def _agrees(found: CellSpaces, explicit: ReleaseIdentity | None, manifest_path: Path) -> None:
    if explicit is not None and explicit != found.analysis.identity:
        raise ReleaseIdUnknown(
            f"{manifest_path} says {found.analysis.identity} but {explicit} was given; a "
            "manifest that is reachable decides"
        )


def _store_faults(store: Path, scope: ReleaseScope) -> list[str]:
    """How the Zarr store at `store` disagrees with the release `scope` describes."""
    manifest = scope.manifest
    dense = manifest.primary_layout is PrimaryStorageLayout.DENSE
    ragged = (store / "ragged").is_dir()
    if ragged == dense:
        return [
            f"it describes a {manifest.primary_layout.value} release but the store "
            f"{'has' if ragged else 'has no'} ragged/ component"
        ]
    if dense:
        records, shape = _zarr_attributes(store), _zarr_shape(store / "z")
        recorded_analyses: Any = None if shape is None else shape[1]
        recorded_rows = None if shape is None else shape[0]
    else:
        records = _zarr_attributes(store / "ragged")
        offsets = _zarr_shape(store / "ragged" / "offsets")
        recorded_analyses = records.get("n_analyses", None if offsets is None else offsets[0] - 1)
        by_variant = _zarr_shape(store / "ragged" / "by_variant" / "offsets")
        recorded_rows = None if by_variant is None else by_variant[0] - 1
    n_rows = _n_rows(scope)
    faults = []
    layout = records.get("layout")
    if layout is not None and layout != ("dense" if dense else "ragged"):
        faults.append(f"the store records layout {layout!r}")
    state = records.get("completion_state")
    hybrid = manifest.primary_layout is PrimaryStorageLayout.HYBRID
    if state is not None and not hybrid and state != manifest.completion_state.value:
        faults.append(
            f"the store records completion state {state!r}, the manifest "
            f"{manifest.completion_state.value!r}"
        )
    n_analyses = _n_analyses(scope.analyses_path)
    if recorded_analyses is not None and n_analyses is not None:
        if int(recorded_analyses) != n_analyses:
            faults.append(
                f"the store holds {int(recorded_analyses)} Analyses, the release {n_analyses}"
            )
    if recorded_rows is not None and n_rows is not None and recorded_rows != n_rows:
        faults.append(f"the store holds {recorded_rows} variant rows, the release {n_rows}")
    return faults


def _zarr_attributes(group: Path) -> dict[str, Any]:
    """A Zarr group's recorded attributes, read from its metadata alone (v3, then v2)."""
    for name, key in (("zarr.json", "attributes"), (".zattrs", None)):
        document = _json(group / name)
        if isinstance(document, dict):
            found = document if key is None else document.get(key, {})
            return dict(found) if isinstance(found, dict) else {}
    return {}


def _zarr_shape(array: Path) -> tuple[int, ...] | None:
    """A Zarr array's recorded shape, from its metadata alone (v3, then v2); None if absent."""
    for name in ("zarr.json", ".zarray"):
        document = _json(array / name)
        if isinstance(document, dict) and isinstance(document.get("shape"), list):
            return tuple(int(n) for n in document["shape"])
    return None


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _n_analyses(path: Path) -> int | None:
    """The Analyses `analyses.tsv` lists, by its rows; None where there is no table."""
    try:
        with path.open("rb") as handle:
            return max(sum(1 for line in handle if line.strip()) - 1, 0)
    except OSError:
        return None


#: The variant axis's row count, by the first of these the release carries: format 0.3.0's
#: ALID sidecar, or the 0.2 series' offsets.
_AXIS_ROW_FILES = ("variant_alid_order.npy", "variant_offsets.npy")


def _n_rows(scope: ReleaseScope) -> int | None:
    """How many variant rows the release's grid is numbered over: a Dense Component's panel
    (`dense_to_shared.npy`), or the release's own axis; None where nothing records it."""
    if scope.hybrid_root is not None:
        return _npy_length(scope.path / DENSE_TO_SHARED_FILENAME)
    for name in _AXIS_ROW_FILES:
        found = _npy_length(scope.path / name)
        if found is not None:
            return found
    return None


def _npy_length(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except (OSError, ValueError):
        return None
