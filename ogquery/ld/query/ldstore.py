"""`LDStore`: the object a caller queries (ld-store §8, §9).

The four queries are methods of one opened store, because each needs the same state:
the reader, t_far and the distances the store measured. That keeps the interface to
`ogquery.ld.open` and four calls, while the reader, the codes and the refusal rules stay
behind it. Each method delegates to a module of its own, so the tickets that implement
them (ogld#4: matrices; ogld#6: proxies and clumping) edit different files.

Two reads answer what a store holds before any r is read: `variants_in` (and its count,
`count_in`) for a region, and `has` for a list of ALIDs (ogld#79; ogquery#2).

An `LDStore` reads through the `Reader` protocol, so a reader of another kind plugs into it:
opengwasdb's Genotype Store reader, which needs pgenlib and so is not ogquery's (ADR 0002).
`open` here opens LD Stores and refuses a Genotype Store, naming opengwasdb.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogquery.identity import normalise_chromosome
from ogquery.ld.format import InputError, Provenance
from ogquery.ld.query import clumping, matrices
from ogquery.ld.query import proxies as proxies_query
from ogquery.ld.query.kind import GENOTYPE, open_kind
from ogquery.ld.query.reader import Reader
from ogquery.ld.query.results import Clumps, LDMatrix, Proxies
from ogquery.ld.store.reader import StoreReader, opened_ld
from ogquery.ld.store.tables import Variants


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

    def variants_in(self, chrom: str, start: int, stop: int) -> Variants:
        """The Variant Set's variants on `chrom` with start ≤ position ≤ stop, in genomic order.

        What a `region` would hold, read from the variant table alone, so a caller can see how
        large a region is before it asks for its r (ogld#79). It judges no radius: unlike
        `region`, nothing here reads a pair, so no far_bp limit applies. `chrom` is normalised
        as `region` normalises it (`chr22` is `22`); a range with start > stop is an
        `InputError`, never an empty answer.
        """
        chrom, start, stop = _range(chrom, start, stop)
        with self._reader.request():
            return self._reader.variants(self._reader.rows_in(chrom, start, stop))

    def count_in(self, chrom: str, start: int, stop: int) -> int:
        """How many variants `variants_in` would return: their positions read, no string column."""
        chrom, start, stop = _range(chrom, start, stop)
        with self._reader.request():
            return len(self._reader.rows_in(chrom, start, stop))

    def has(self, alids: Sequence[str]) -> npt.NDArray[np.bool_]:
        """Whether the Variant Set holds each ALID, in request order (ogld#79).

        True exactly where `matrix`, `proxies` and `clump` find the ALID present, and False where
        they report it absent, a malformed ALID included. An indel spelt reference first is
        refused (`ogquery.alid.ReferenceFirstIndel`), as those queries refuse it, rather than
        answered False: the store would hold that variant under its sorted ALID.
        """
        with self._reader.request():
            found: npt.NDArray[np.bool_] = self._reader.rows_of(alids) >= 0
            return found


def _range(chrom: str, start: int, stop: int) -> tuple[str, int, int]:
    """A region's chromosome, normalised, and its whole-number bounds, start ≤ stop; else an
    `InputError` naming what is wrong."""
    for name, value in (("start", start), ("stop", stop)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise InputError(f"{name} is a whole-number position, not {value!r}")
    if start > stop:
        raise InputError(f"a region's start is at most its stop: start {start}, stop {stop}")
    return normalise_chromosome(chrom), int(start), int(stop)


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
