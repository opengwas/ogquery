# ogquery Domain Glossary

ogquery is opengwasdb's Store Reader as its own package, so it shares opengwasdb's
vocabulary. **Shared terms are defined in
[opengwasdb's CONTEXT.md](https://github.com/opengwas/opengwasdb/blob/main/CONTEXT.md)**,
and are not repeated here: Store Release, Format Version, Primary Storage Layout (Dense,
Ragged, Hybrid), Dense Component, Analysis, Variant Index, Variant Axis, ALID Search
Index, Store Variant Table, Top-Hit Index, Indexed Variant Subset, Association Status,
Observed-Only Query, Store Reader and Store Reader Contract Bundle among them. Use them as
defined there (Analysis, not phenotype; Store Release, not dataset). A change to a shared
term is made there.

This file defines only what is ogquery's own.

## Language

**ogquery**:
The distribution of the **Store Reader** (opengwasdb's term): the package `ogquery`,
whose only runtime dependencies are numpy, zarr and numcodecs. Its modules are the ones
`opengwasdb.store_reader` had, by the same names (ADR 0001).
_Avoid_: query engine, client (opengwaspy and opengwasr are the clients).

**Variant Table Reader**:
What a caller passes as `open_release(path, legacy=...)` so ogquery can query a 0.2.x or
0.1.0 Store Release: an implementation of `ogquery.variant_axis.LegacyVariantTable`,
which reads that release's BGZF/tabix Store Variant Table and its `variant_aliases`.
ogquery defines the protocol and implements none that needs pysam; opengwasdb implements
it. Without one, querying such a release raises `VariantTableReaderRequired`, naming its
format version and how it becomes 0.3.0: `ogdb rewrite-release` for a 0.2.3 release, a
rebuild from source for an earlier one. A 0.3.0 release needs none.
_Avoid_: legacy reader, legacy provider.

**Bundle Source**:
The opengwasdb commit the committed contract bundle came from, named in
`tests/contract/SOURCE`. The bundle's data (`releases/`, `expected/`, `VERSION`,
`.gitattributes`) is that commit's byte for byte; ogquery never edits it, and takes a
newer one by refreshing from a later commit.

**Clean Install**:
A fresh virtual environment into which `pip install ogquery` was run with source builds
refused: it must hold ogquery, numpy, zarr, numcodecs and their own dependencies, and
nothing else, every one from a wheel (`scripts/check_installed_dependencies.py`). CI
checks one on Linux, macOS and Windows.
