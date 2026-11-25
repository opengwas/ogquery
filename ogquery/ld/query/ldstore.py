"""`LDStore`: the object a caller queries (ld-store §8, §9).

The four queries are methods of one opened store, because each needs the same state:
the reader, t_far and the distances the store measured. That keeps the interface to
`ogquery.ld.open` and four calls, while the reader, the codes and the refusal rules stay
behind it. Each method delegates to a module of its own, so the tickets that implement
them (ogld#4: matrices; ogld#6: proxies and clumping) edit different files.

An `LDStore` reads through the `Reader` protocol, so a reader of another kind plugs into it:
opengwasdb's Genotype Store reader, which needs pgenlib and so is not ogquery's (ADR 0002).
`open` here opens LD Stores and refuses a Genotype Store, naming opengwasdb.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy.typing as npt

from ogquery.ld.format import InputError, Provenance
from ogquery.ld.query import clumping, matrices
from ogquery.ld.query import proxies as proxies_query
from ogquery.ld.query.kind import GENOTYPE, open_kind
from ogquery.ld.query.reader import Reader
from ogquery.ld.query.results import Clumps, LDMatrix, Proxies
from ogquery.ld.store.reader import StoreReader, opened_ld


class LDStore:
    """An opened LD Store. Get one from `ogquery.ld.open`."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader

    @property
    def provenance(self) -> Provenance:
        """The store's provenance, as recorded (ld-store §7)."""
        return self._reader.provenance

    def close(self) -> None:
        """Close the store's reader: its chunk cache and companion, or a Genotype Store's pgen
        handles and ALID mappings. Closing twice is closing once, as a reader's `close` is."""
        self._reader.close()

    def __enter__(self) -> LDStore:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        self.close()

    @property
    def t_far(self) -> float:
        """The Far Threshold this store recorded, the one every answer uses (ld-store §3)."""
        return self._reader.provenance["t_far"]

    @property
    def near_bp(self) -> int:
        """The distance up to which the store holds every pair (ld-store §7)."""
        return self._reader.provenance["near_bp"]

    @property
    def far_bp(self) -> int:
        """The distance up to which the store measured pairs (ld-store §7)."""
        return self._reader.provenance["far_bp"]

    def matrix(self, alids: Sequence[str], *, fill: float | None = None) -> LDMatrix:
        """The signed-r matrix of `alids`, in request order; NaN where unstored unless `fill`."""
        with self._reader.request():
            return matrices.matrix(self._reader, alids, fill=fill)

    def region(
        self, chrom: str, centre: int, radius: int, *, fill: float | None = None
    ) -> LDMatrix:
        """The signed-r matrix within `radius` bp of `centre`; refused if radius > far_bp / 2."""
        with self._reader.request():
            return matrices.region(self._reader, chrom, centre, radius, fill=fill)

    def proxies(self, alids: Sequence[str], *, r2: float = 0.8, kb: float = 10_000) -> Proxies:
        """Each ALID's proxies at r̂² ≥ r2 within kb × 1,000 bp, with signed r."""
        with self._reader.request():
            return proxies_query.proxies(self._reader, alids, r2=r2, kb=kb)

    def clump(self, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float) -> Clumps:
        """Clump `alids` by `score`: join at r̂² > r2, within kb × 1,000 bp, as plink2 does."""
        with self._reader.request():
            return clumping.clump(self._reader, alids, score, r2=r2, kb=kb)


def open(location: str | os.PathLike[str], *, cache_bytes: int | None = None) -> LDStore:
    """Open the LD Store at a local path or an HTTP(S) URL (ld-store §1; ogld#4).

    The store keeps at most `cache_bytes` of the chunks and shard indexes its reads fetch, 4 GiB
    if None, and none if 0; anything but an int ≥ 0 is an `InputError` (ld-store §10; ogld#32).
    A URL needs the `http` extra (`pip install ogquery[http]`). A Genotype Store is an
    `InputError` naming opengwasdb, whose reader computes its answers from genotypes."""
    kinded = open_kind(location, cache_bytes=cache_bytes)
    if kinded.kind == GENOTYPE:
        raise InputError(
            f"{location} is a Genotype Store, which ogquery does not read: its answers are "
            "computed from genotypes with pgenlib. Open it with opengwasdb.ld.open "
            "(pip install 'opengwasdb[ld-build]')"
        )
    return LDStore(StoreReader.from_opened(opened_ld(kinded.root, kinded.location, kinded.cache)))
