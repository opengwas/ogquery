"""Queries on an LD Store: LD matrices, regions, proxies and clumping (layer 2; ogld#4, ogld#6).

It reads through the `ogquery.ld.query.reader.Reader` protocol, which `StoreReader` satisfies
and opengwasdb's Genotype Store reader does too (opengwas/ogld#91), and judges every request
with `ogquery.ld.format`. An unstored pair is NaN, and only a caller's `fill` turns it into a
number.
"""

from ogquery.ld.query.ldstore import LDStore, open
from ogquery.ld.query.reader import Reader
from ogquery.ld.query.results import Clumps, LDMatrix, Proxies

__all__ = ["Clumps", "LDMatrix", "LDStore", "Proxies", "Reader", "open"]
