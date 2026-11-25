"""ogquery: the Store Reader, the read-only path every reader of a Store Release uses.

It came out of opengwasdb, where it was `opengwasdb.store_reader` (opengwasdb#464),
with its history (README, docs/adr/0001). Everything the query facade reads -- the
release envelope and its format version, the encoding plan and the decoded planes, the
association readers, the Ragged CSR and By-Variant Index readers, Indexed Variant
Subsets, the Top-Hit Index, the Analyses index, the Variant Axis, and the facade and
result resolution -- lives here, and imports nothing but numpy, zarr, numcodecs, the
standard library (not `sqlite3`) and this package
(`tests/unit/test_store_reader_imports.py`).

Open a release and ask it questions:

    from ogquery.facade import query_release
    from ogquery.release import open_release

    with query_release(open_release(path)) as q:
        q.phewas("1:1000:A:G")

The 0.2.x reads that need pysam or SQLite -- the Store Variant Table's tabix range
fetches and BGZF seeks, and the `variant_aliases` fallback -- are not here: the Variant
Axis calls them through `LegacyVariantTable`, which opengwasdb implements
(`opengwasdb.variants.axis`) and passes as `open_release(path, legacy=...)`. A 0.2.x
release opened without one is refused, naming its `format_version` and
`ogdb rewrite-release`, the command that rewrites it as 0.3.0.

This file imports nothing, so importing any module of the package loads only what that
module needs; `__version__` is read on first use (`ogquery.version`).
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from ogquery.version import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
