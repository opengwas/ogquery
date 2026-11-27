"""One read of a store's root metadata, and which kind of store it says it is (opengwas/ogld#91).

`ogquery.ld.open`, and opengwasdb's own `open` and `validate`, come through here: the root is
opened once, its kind is read from the attribute it records, and the opened group is handed to
that kind's reader or validator. One read is enough: opening the root again in the reader would
add a metadata read — an HTTP round trip for a remote store — to every open.

`ogquery.ld.open` dispatches an `LD` root to `StoreReader` and a `GENOTYPE` root to
`ogquery.ld.genotypes`'s `GenotypeReader`, which needs the `genotype` extra (opengwas/ogquery#11;
ADR 0002, amended).

This reads the kind and nothing more: an envelope a reader refuses (a Zarr v2 hierarchy, a root
with no provenance) is still validation's to report as findings, not this seam's to raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

import zarr

from ogquery.ld.store.location import ChunkCache, open_group

LD: Final = "ld_store"
GENOTYPE: Final = "genotype_store"
#: The root attribute a Genotype Store records its provenance under, which is this kind's name:
#: `ogquery.ld.genotypes.layout.KEY` is this one.
KEY: Final = GENOTYPE


@dataclass(frozen=True)
class Kinded:
    """The root's kind, the one opening of its metadata, and what it was opened through."""

    kind: Literal["ld_store", "genotype_store"]
    root: zarr.Group
    location: str
    cache: ChunkCache

    def record(self) -> object:
        """The kind's recorded provenance, or None where the root records none."""
        return self.root.attrs.get(KEY if self.kind == GENOTYPE else LD)


def open_kind(location: str | os.PathLike[str], *, cache_bytes: int | None = None) -> Kinded:
    """Open `location` once and say which kind it is: a root with the Genotype Store key is one,
    and every other root is read as an LD Store, whose envelope its reader or validation judges.

    The location is read once, by `os.fspath`, so what is opened and what is recorded -- and so
    what a reader names and judges, such as whether it is a URL -- are the same string; a `Path`
    is recorded as `str(path)`."""
    where = os.fspath(location)
    cache = ChunkCache(cache_bytes)
    root = open_group(where, cache=cache)
    kind: Literal["ld_store", "genotype_store"] = GENOTYPE if KEY in root.attrs else LD
    return Kinded(kind, root, where, cache)
