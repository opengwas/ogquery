# ogquery

The light, read-only query package for OpenGWAS **GWAS Stores**. It is the
**Store Reader**: the one read path that the OpenGWASDB builder and validator,
the Data API (ogdb-api), opengwaspy and opengwasr all use.

- **Dependencies:** numpy, zarr and numcodecs only. No pysam, no SQLite, no
  scipy or pandas, nothing to compile. It runs on Linux, macOS and Windows, on
  Python 3.12 to 3.14.
- **Reads** Store Releases of format `0.3.0` on its own, and `0.2.x` ones when
  given a reader for their Store Variant Table (below).
- **Contract:** the Store Reader contract bundle (`tests/contract/`): nine
  small releases and what the Store Reader must answer on each, which any
  reader, including a future Rust core shipped inside the same wheels, must
  reproduce.

Most users never install ogquery by name. They install
[opengwaspy](https://github.com/opengwas/opengwaspy), whose Local Source reads
through it, or [opengwasr](https://github.com/opengwas/opengwasr), which reaches
it through opengwaspy.

## Install

```bash
pip install ogquery
```

It installs numpy, zarr, numcodecs and their own dependencies, all as wheels.

## Open a release

```python
from ogquery.facade import query_release
from ogquery.release import open_release

release = open_release("tests/contract/releases/dense_ranges")
print(release.manifest.format_version, release.manifest.primary_layout)  # 0.3.0 dense

with query_release(release) as q:
    analyses = q.analyses_table()                      # {analysis_index: row}
    phewas = q.phewas("1:1000:A:G")                    # every Analysis, by ALID or rsID
    hits = q.top_hits(threshold=5e-8)                  # in genomic order
    in_range = q.variants_in_range("1", 1, 10_000)     # {variant_index: row}
```

An association answer is a mapping of numpy arrays (`z`, `se`, `beta`, `pval`,
`eaf`, `association_status`, `variant_index`, `analysis_index`); `lookup`,
`analysis` and `range_phewas` are the other shapes (`ogquery.facade`).

**A 0.2.x release** answers its variants from `variants.tsv.gz`, a BGZF/tabix
file that needs pysam. ogquery refuses to query one alone, with an error
(`ogquery.release.VariantTableReaderRequired`) naming the release's
`format_version` and `ogdb rewrite-release`, the opengwasdb command that
rewrites it as 0.3.0. opengwasdb passes a reader for the table,
`open_release(path, legacy=...)`, and reads both.

## Develop

```bash
pixi run -e test test              # the whole suite
pixi run -e test contract          # the contract bundle alone
pixi run -e dev lint               # ruff; `typecheck` is mypy --strict
pixi run -e dev contract-check     # tests/contract is opengwasdb's bundle
```

Or, without pixi: `pip install -e ".[test]"` and `pytest`.

[CONTRIBUTING.md](./CONTRIBUTING.md) is how to contribute,
[AGENTS.md](./AGENTS.md) how work moves, [CONTEXT.md](./CONTEXT.md) the
vocabulary, and [docs/adr/](./docs/adr/) the decisions.

## Where it came from

ogquery is opengwasdb's Store Reader, `opengwasdb.store_reader`, moved into its
own package by [ogquery#1](https://github.com/opengwas/ogquery/issues/1)
([ADR 0001](./docs/adr/0001-ogquery-is-the-store-reader-extracted-from-opengwasdb.md)).
It came with its history, extracted from opengwasdb
[`69c29452`](https://github.com/opengwas/opengwasdb/commit/69c2945229070ede4ccf8ced0520b281a87a0d68)
with `git filter-repo`:

| Here | In opengwasdb |
|---|---|
| `ogquery/` | `opengwasdb/store_reader/` |
| `ogquery/digests.py` | `opengwasdb/store/query_digests.py` |
| `tests/contract/` | `tests/store_reader_contract/` (less `generate.py`) |
| `tests/unit/` | nine of `tests/test_*.py`, the Store Reader's own |

**The history starts at opengwasdb#464**, where these files entered
`store_reader/`. Their earlier history -- when the same code lived across
`opengwasdb/query/`, `opengwasdb/store/`, `opengwasdb/encoding/` and the rest --
is opengwasdb's, at
[`c542111c`](https://github.com/opengwas/opengwasdb/commit/c542111c74699a934b5ae5ae9163cce69b814570), the
commit before #464's first. A bare issue number (`#464`) or ADR number in code
that came from opengwasdb is opengwasdb's.

## Licence

GPL-3.0-only. See [LICENSE](./LICENSE). Copyright © University of Bristol.
Contributions from outside the University of Bristol need a signed
[Contributor License Agreement](./CLA.md).
