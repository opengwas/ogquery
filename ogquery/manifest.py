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
    apart needs an identity the writer stamps into the store (opengwasdb's to add).

    It also assumes a `(store_id, release_id)` pair names one release, which every real
    release does: a build, a completion and a splice each mint a new `release_id`
    (`ogquery.guard`). Two directories that claim the same pair are not told apart."""
    return _bound(Path(store), explicit)[0]


def _bound(
    store: Path, explicit: ReleaseIdentity | None
) -> tuple[CellSpaces, StoreManifest | None]:
    """`store_spaces`, with the manifest it read (None where it read none)."""
    path = store.resolve()
    manifest_path = path.parent / "manifest.json"
    if path.name != "data.zarr" or not manifest_path.is_file():
        return _explicit(
            explicit,
            f"cannot learn which release the Zarr store at {str(path)!r} belongs to: it is not "
            "the data.zarr of a release with a manifest.json",
        ), None
    scope = release_scope_of(path.parent)
    faults = _store_faults(path, scope)
    if faults:
        raise ReleaseIdUnknown(
            f"{manifest_path} is not this store's manifest: {'; '.join(faults)} ({path})"
        )
    _agrees(scope.spaces, explicit, manifest_path)
    return scope.spaces, scope.manifest


def data_spaces(release: str | Path, *, explicit: ReleaseIdentity | None = None) -> CellSpaces:
    """The spaces of what a reader opening ``<release>/data.zarr`` reads (`data_scope`)."""
    return data_scope(release, explicit=explicit)[0]


def data_scope(
    release: str | Path, *, explicit: ReleaseIdentity | None = None
) -> tuple[CellSpaces, StoreManifest | None]:
    """The spaces of what a reader opening ``<release>/data.zarr`` reads (`store_spaces`),
    and the release's manifest, read once (None where the directory has none).

    The store must be that release's own. A ``data.zarr`` that is a link out of `release` is
    refused where either side has a manifest: the one at `release` does not describe what
    it would read, and the one beside the link's target decides that store's identity, so
    no `explicit` identity may stand in for it. A directory with no manifest, whose store
    is its own or a link to a store with none, takes `explicit`."""
    directory = Path(release).resolve()
    store = Path(release) / "data.zarr"
    own = (directory / "manifest.json").is_file()
    target = store.resolve() if store.exists() else None
    if target is not None and target.parent != directory:
        if own or (target.parent / "manifest.json").is_file():
            raise ReleaseIdUnknown(
                f"{store} resolves to {target}, outside the release at {directory}: "
                + (
                    "its manifest does not describe the data.zarr it would read"
                    if own
                    else f"the manifest at {target.parent} decides that store's identity, "
                    "and no identity= may stand in for it"
                )
            )
    if not own:
        return _explicit(
            explicit, f"cannot learn which release {str(directory)!r} is: it has no manifest.json"
        ), None
    return _bound(store, explicit)


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
    """How the Zarr store at `store` disagrees with the release `scope` describes. Metadata
    the store or release does not carry is skipped; metadata it carries that cannot be read
    is refused (`_unreadable`), never taken for absent."""
    manifest = scope.manifest
    dense = manifest.primary_layout is PrimaryStorageLayout.DENSE
    hybrid = manifest.primary_layout is PrimaryStorageLayout.HYBRID
    ragged = (store / "ragged").is_dir()
    if ragged == dense:
        return [
            f"it describes a {manifest.primary_layout.value} release but the store "
            f"{'has' if ragged else 'has no'} ragged/ component"
        ]
    associations: int | None = None
    if dense:
        records, shape = _zarr_attributes(store), _zarr_shape(store / "z")
        recorded_analyses = None if shape is None else shape[1]
        recorded_rows = None if shape is None else shape[0]
    else:
        records = _zarr_attributes(store / "ragged")
        offsets = _zarr_shape(store / "ragged" / "offsets")
        recorded_analyses = _recorded_int(
            records, "n_analyses", store / "ragged", None if offsets is None else offsets[0] - 1
        )
        sequence = _zarr_shape(store / "ragged" / "variant_index")
        associations = _recorded_int(
            records, "n_associations", store / "ragged", None if sequence is None else sequence[0]
        )
        by_variant = _zarr_shape(store / "ragged" / "by_variant" / "offsets")
        recorded_rows = None if by_variant is None else by_variant[0] - 1
    faults = []
    layout = records.get("layout")
    if layout is not None and layout != ("dense" if dense else "ragged"):
        faults.append(f"the store records layout {layout!r}")
    state = records.get("completion_state")
    if state is not None and not hybrid and state != manifest.completion_state.value:
        faults.append(
            f"the store records completion state {state!r}, the manifest "
            f"{manifest.completion_state.value!r}"
        )
    n_analyses = _n_analyses(scope.analyses_path)
    if recorded_analyses is not None and n_analyses is not None:
        if recorded_analyses != n_analyses:
            faults.append(f"the store holds {recorded_analyses} Analyses, the release {n_analyses}")
    n_rows = _n_rows(scope)
    if recorded_rows is not None and n_rows is not None and recorded_rows != n_rows:
        faults.append(f"the store holds {recorded_rows} variant rows, the release {n_rows}")
    overflow = _overflow_associations(manifest) if hybrid else None
    if overflow is not None and associations is not None and overflow != associations:
        faults.append(
            f"the store holds {associations} Overflow associations, the manifest records "
            f"{overflow} (provenance.hybrid.n_overflow_associations)"
        )
    return faults


def _overflow_associations(manifest: StoreManifest) -> int | None:
    """The Overflow association count a Hybrid manifest records, if it records one."""
    recorded = manifest.provenance.get("hybrid")
    if not isinstance(recorded, dict) or "n_overflow_associations" not in recorded:
        return None
    value = recorded["n_overflow_associations"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _unreadable(
            Path("manifest.json"),
            f"provenance.hybrid.n_overflow_associations is {value!r}, not a count",
        )
    return value


def _recorded_int(records: dict[str, Any], key: str, where: Path, fallback: int | None) -> Any:
    """`records[key]` as a count, `fallback` where the store does not record it."""
    if key not in records:
        return fallback
    value = records[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _unreadable(where, f"its {key!r} attribute is {value!r}, not a count")
    return value


def _unreadable(path: Path, why: str) -> ReleaseIdUnknown:
    return ReleaseIdUnknown(
        f"{path} is unreadable ({why}), so whether its manifest describes the store cannot be "
        "told: it is refused, not skipped"
    )


def _zarr_attributes(group: Path) -> dict[str, Any]:
    """A Zarr group's recorded attributes, read from its metadata alone (v3, then v2): none
    where it has no metadata, refused where its metadata cannot be read."""
    for name, key in (("zarr.json", "attributes"), (".zattrs", None)):
        document = _json(group / name)
        if document is None:
            continue
        found = document.get(key, {}) if isinstance(document, dict) and key else document
        if not isinstance(document, dict) or not isinstance(found, dict):
            raise _unreadable(group / name, "not a JSON object of attributes")
        return dict(found)
    return {}


def _zarr_shape(array: Path) -> tuple[int, ...] | None:
    """A Zarr array's recorded shape, from its metadata alone (v3, then v2); None where the
    array is absent, refused where its metadata cannot be read."""
    for name in ("zarr.json", ".zarray"):
        document = _json(array / name)
        if document is None:
            continue
        shape = document.get("shape") if isinstance(document, dict) else None
        if not isinstance(shape, list) or not all(
            isinstance(n, int) and not isinstance(n, bool) for n in shape
        ):
            raise _unreadable(array / name, "it records no shape")
        return tuple(shape)
    return None


def _json(path: Path) -> Any:
    """The JSON document at `path`; None where there is none, refused where it cannot be
    read."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _unreadable(path, f"{type(error).__name__}: {error}") from error


def _n_analyses(path: Path) -> int | None:
    """The Analyses `analyses.tsv` lists, read by its one reader (`analyses.read_analyses`,
    so a quoted label over two lines is one row); None where there is no table."""
    from ogquery.analyses import read_analyses  # it imports this module

    if not path.exists():
        return None
    try:
        return len(read_analyses(path).rows)
    except (OSError, ValueError) as error:
        raise _unreadable(path, f"{type(error).__name__}: {error}") from error


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
    """The length of the array at `path`; None where there is none, refused where it cannot
    be read."""
    if not path.exists():
        return None
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except (OSError, ValueError, IndexError) as error:
        raise _unreadable(path, f"{type(error).__name__}: {error}") from error
