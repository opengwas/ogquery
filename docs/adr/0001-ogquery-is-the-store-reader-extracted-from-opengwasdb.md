# ogquery is the Store Reader, extracted from opengwasdb

## Status

Accepted with [ogquery#1](https://github.com/opengwas/ogquery/issues/1), the first ticket
of the second part of epic
[opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450). The owner
planned the extraction in that epic; this records what moved and the choices the move
made.

## Context

Every reader of a Store Release -- opengwasdb's builders, validator and completion, the
Data API, opengwaspy and opengwasr -- needs the same read path: open the release and
check its format version, decode its encoding, read its associations under one presence
rule, its Variant Axis, Top-Hit Index, Indexed Variant Subsets and Analyses, and answer
the query shapes. Inside opengwasdb that path pulled in what opengwasdb needs for
everything else -- pysam, SQLite, scipy, pandas, scikit-learn, bcftools -- so a
researcher on Windows who only wanted to query a downloaded release could not install it.

opengwasdb#464 gathered the read path into one subpackage, `opengwasdb.store_reader`,
that imports nothing but numpy, zarr, numcodecs, the standard library and itself, and
held it there with a test. Format 0.3.0 (opengwasdb ADR 0105, #467-#472) then made a
release answerable without its Store Variant Table: a 0.3.0 query reads sidecars, never
`variants.tsv.gz`, so it needs no pysam. opengwasdb#474 committed the contract bundle,
nine small 0.3.0 releases and every answer the Store Reader gives on them. With those, the
subpackage could become its own distribution.

## Decision

**ogquery is that subpackage, moved with its history.** `opengwasdb/store_reader/`
became `ogquery/`, and `opengwasdb/store/query_digests.py`, the questions the contract and
opengwasdb's rewrite proof ask, `ogquery/digests.py`. The contract bundle came as
`tests/contract/` and the Store Reader's own unit tests that need no opengwasdb builder as
`tests/unit/`. `git filter-repo` kept those paths, renamed, from opengwasdb `69c29452`;
history starts where the files entered `store_reader/` (#464), and the earlier history is
opengwasdb's (README, "Where it came from").

**The public API is `opengwasdb.store_reader`'s, by the same names.** Only the package
prefix changes: `opengwasdb.store_reader.facade.query_release` is
`ogquery.facade.query_release`. So opengwasdb can re-export ogquery (#475) and every
caller keeps working. Two additions: `ogquery.__version__`, and
`ogquery.release.VariantTableReaderRequired` with `REWRITE_COMMAND` (below).
`ogquery.digests` opens a release with the Store Reader alone by default
(`digests.open_query`), where it used opengwasdb's `query_store`.

**Runtime dependencies are numpy, zarr and numcodecs, at opengwasdb's bounds** (`numpy>=1.26,<2.5`,
`zarr>=3.4,<4`, `numcodecs>=0.17`), and nothing else, so `pip install ogquery` compiles
nothing on Linux, macOS or Windows. `tests/unit/test_store_reader_imports.py` holds the
package's imports to them, the declared dependencies to exactly them, and a fresh
interpreter that imports every module to loading none of pysam, sqlite3, scipy, pandas or
threadpoolctl; the contract test asks every question in a fresh interpreter and holds it
to the same. CI's clean install holds what pip installs to their closure, all wheels.

**Python 3.12 or later**, not the 3.11 the ticket names. zarr 3.4, opengwasdb's floor,
requires 3.12 (zarr 3.2 raised it). Keeping 3.11 would mean allowing zarr 3.1 for it,
a version the Store Reader was never tested on: this choice keeps the reader on the zarr
it was proved with. Lowering the floor is a later decision, with the contract run on 3.11
and zarr 3.1 first.

**It reads 0.3.0 alone, and 0.2.x with a reader for its Store Variant Table.** A 0.2.x
(and 0.1.0) release answers its variants from a BGZF/tabix `variants.tsv.gz` and, for a
Dense or Hybrid one, `index.sqlite`'s `variant_aliases`. ogquery defines the reader's
protocol (`variant_axis.LegacyVariantTable`) and opengwasdb implements it; a caller passes
it as `open_release(path, legacy=...)`. Without one, a query of such a release raises
`VariantTableReaderRequired`, a `ValueError` naming the release's `format_version` and
`ogdb rewrite-release`, which rewrites a 0.2.3 release as 0.3.0.

**What ogquery owns: the read path.** The format gates a reader applies, decoding,
the association, CSR, By-Variant, Indexed Variant Subset, Top-Hit and Rho readers, the
Variant Axis and its sidecars' readers, the query facade and result resolution, the
contract questions (`ogquery.digests`), and the contract test.

**What it does not own** stays in opengwasdb: building, writing and staging a release,
validation and its Validation Rules, Reference Completion, `ogdb rewrite-release`, the
0.2.x Variant Table's pysam and SQLite reads, and generating the contract bundle. Some
writers' halves live in modules ogquery carries -- the sidecars' `write_*` functions, and
`format.format_version_for` and `WRITTEN_FORMAT_VERSIONS` -- because the Store Reader held
them when it moved; they import nothing more, and moving them is opengwasdb's call.

**The contract bundle is opengwasdb's, read only here.** Only opengwasdb has the
builders, so it regenerates the bundle. `tests/contract/SOURCE` names the opengwasdb commit
the copy came from, `scripts/contract_bundle.py check` (CI) holds the data to it byte for
byte, and `refresh` takes a newer commit's.

**Releases follow opengwasdb ADR 0083.** Every merge to `main` is tagged from its
changelog fragments (`release.yml`), and `publish.yml` publishes every tag from
`PYPI_FIRST_RELEASE` on to PyPI by trusted publishing, environment `pypi` (`testpypi` for
a rehearsal). The first release is `v0.1.0`; the repository's first commit, which
predates the package, is never released.

## Consequences

- Until #475, the Store Reader exists twice: here and in opengwasdb. #475 makes
  `opengwasdb.store_reader` (and `opengwasdb.query`) re-export ogquery and deletes the
  copy and the unit tests that moved; after it, a change to the read path is made here
  and released before opengwasdb takes it.
- A change to what the Store Reader answers is made in opengwasdb's generator first, and
  reaches ogquery by a bundle refresh.
- Issue and ADR numbers in code that came from opengwasdb are opengwasdb's; new
  references name the repository.
- Renaming `publish.yml` or the `pypi`/`testpypi` environments breaks publication until
  the trusted publishers on PyPI and TestPyPI are changed to match.
