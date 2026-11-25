"""LD Stores: per-ancestry LD for clumping, proxies and LD matrices, read side (ogquery#2).

The read path of opengwasdb's `opengwasdb.ld`, moved here from opengwasdb `c8ebe4ca`
(ADR 0002). The public surface (opengwasdb's `docs/spec/ld-store.md` §8, §9):

    store = ogquery.ld.open(path_or_url, cache_bytes=None)  # LDStore; keeps 4 GiB of chunks
    store.matrix(alids, fill=None)                  # LDMatrix: signed r, NaN where unstored
    store.region(chrom, centre, radius, fill=None)  # LDMatrix; radius ≤ far_bp / 2
    store.proxies(alids, r2=0.8, kb=10_000)         # Proxies
    store.clump(alids, score, r2=..., kb=...)       # Clumps

A request the store cannot answer exactly raises `Refusal`, naming its reason. An http(s)
URL needs the `http` extra (`pip install ogquery[http]`). Building, validating and the
Genotype Store reader are opengwasdb's.
"""

from ogquery.ld.format import AnswerKind, InputError, Provenance, Refusal, RefusalReason
from ogquery.ld.query import Clumps, LDMatrix, LDStore, Proxies, Reader, open
from ogquery.ld.store import Variants

__all__ = [
    "AnswerKind",
    "Clumps",
    "InputError",
    "LDMatrix",
    "LDStore",
    "Provenance",
    "Proxies",
    "Reader",
    "Refusal",
    "RefusalReason",
    "Variants",
    "open",
]
