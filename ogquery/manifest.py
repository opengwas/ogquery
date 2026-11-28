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

from ogquery.encoding.plan import StoreEncoding, UnsupportedEncoding
from ogquery.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from ogquery.format import check_format_version
from ogquery.guard import ReleaseIdentity, ReleaseIdUnknown, identifier


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


def release_identity(
    directory: str | Path, *, explicit: ReleaseIdentity | None = None
) -> ReleaseIdentity:
    """The identity of the release in `directory`, from its `manifest.json`: the one parser
    (`StoreManifest`). An `explicit` identity is accepted only where no manifest is there,
    and refused where one is and says otherwise. With neither, the reader refuses."""
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        if explicit is None:
            raise ReleaseIdUnknown(
                f"cannot learn which release {str(directory)!r} is: it has no manifest.json; "
                "pass identity= where no manifest is reachable"
            )
        return explicit
    try:
        found = _component_identity(directory, StoreManifest.load(manifest_path))
    except ReleaseIdUnknown as error:
        raise ReleaseIdUnknown(f"{manifest_path}: {error}") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReleaseIdUnknown(
            f"cannot learn which release {str(directory)!r} is: {manifest_path} is unreadable "
            f"({type(error).__name__}: {error})"
        ) from error
    if explicit is not None and explicit != found:
        raise ReleaseIdUnknown(
            f"{manifest_path} says {found} but {explicit} was given; a manifest that is "
            "reachable decides"
        )
    return found


def group_identity(group: Any, *, explicit: ReleaseIdentity | None = None) -> ReleaseIdentity:
    """The identity of the release whose `data.zarr` holds the Zarr `group`.

    Never guessed. The store's real path (symlinks followed) must be a directory named
    ``data.zarr`` whose parent holds a `manifest.json`, and that manifest must describe this
    store: a release with a Ragged component (`ragged/`) is not Dense, and a Dense one has
    none. A Zarr-only copy, or a store that is not a local ``data.zarr``, has no manifest to
    ask, and needs `explicit`."""
    root = getattr(group.store, "root", None)
    if root is None:
        if explicit is None:
            raise ReleaseIdUnknown(
                f"cannot learn which release the Zarr store {group.store!r} belongs to: it has "
                "no local directory; pass identity="
            )
        return explicit
    store = Path(root).resolve()
    if store.name != "data.zarr" or not (store.parent / "manifest.json").is_file():
        if explicit is None:
            raise ReleaseIdUnknown(
                f"cannot learn which release the Zarr store at {str(store)!r} belongs to: it is "
                "not the data.zarr of a release with a manifest.json; pass identity="
            )
        return explicit
    manifest = StoreManifest.load(store.parent / "manifest.json")
    ragged = (store / "ragged").is_dir()
    if ragged == (manifest.primary_layout is PrimaryStorageLayout.DENSE):
        raise ReleaseIdUnknown(
            f"{store.parent / 'manifest.json'} describes a {manifest.primary_layout.value} "
            f"release but {store} {'has' if ragged else 'has no'} ragged/ component: it is not "
            "this store's manifest"
        )
    return release_identity(store.parent, explicit=explicit)


def release_scope(path: str | Path, manifest: StoreManifest) -> ReleaseIdentity:
    """The identity an opened release's indices are scoped to: its manifest's, or its
    Hybrid's where it is a Hybrid's Dense Component (`release_identity`)."""
    return _component_identity(Path(path).resolve(), manifest)


def _component_identity(directory: Path, manifest: StoreManifest) -> ReleaseIdentity:
    """A release's identity, or its Hybrid's where it is a Hybrid's Dense Component.

    A Dense Component is the ``dense/`` directory of a Hybrid release and has a manifest of
    its own, with a Store id of its own, but its Analyses and panel rows are the Hybrid's:
    an index into them is scoped to the Hybrid. It is a component only if the parent really
    is a Hybrid's manifest and names the same release; nothing else is taken for one."""
    parent = directory.parent / "manifest.json"
    if directory.name != "dense" or not parent.is_file():
        return manifest.identity
    try:
        hybrid = StoreManifest.load(parent)
    except (OSError, ValueError, KeyError, TypeError):
        return manifest.identity
    if (
        hybrid.primary_layout is PrimaryStorageLayout.HYBRID
        and manifest.primary_layout is PrimaryStorageLayout.DENSE
        and hybrid.release_id == manifest.release_id
    ):
        return hybrid.identity
    return manifest.identity
