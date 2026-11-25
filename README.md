# ogquery

The light, read-only query package for OpenGWAS **GWAS Stores** and **LD
Stores**. It is the **Store Reader**: the one read path that the OpenGWASDB
builder and validator, the Data API (ogdb-api), opengwaspy and opengwasr all use.

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
pip install ogquery            # local Stores
pip install 'ogquery[http]'    # and Stores served over HTTP(S)
```

It installs numpy, zarr, numcodecs and their own dependencies, all as wheels.
The `http` extra adds fsspec and aiohttp, which zarr reads a URL with; without
it, opening an `http(s)://` Store raises `ogquery.arrays.HttpExtraMissing`,
naming the extra.

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

**A 0.2.x or 0.1.0 release** answers its variants from `variants.tsv.gz`, a
BGZF/tabix file that needs pysam. ogquery refuses to query one alone, with an
error (`ogquery.release.VariantTableReaderRequired`) naming the release's
`format_version` and how to make it a 0.3.0 release: a **0.2.3** release is
rewritten with `ogdb rewrite-release` (opengwasdb); an earlier one must be
**rebuilt from source**, since the rewrite accepts 0.2.3 alone. opengwasdb
passes a reader for the table, `open_release(path, legacy=...)`, and reads both.

## Query an LD Store

An LD Store holds one ancestry's LD for clumping, proxies and LD matrices
(opengwasdb builds them; `ogquery.ld` reads them, [ADR 0002](./docs/adr/0002-the-ld-read-path-is-ogquery-s.md)).

```python
import numpy as np
import ogquery.ld

with ogquery.ld.open("tests/ld/data/test-panel/store") as store:   # or an https:// URL
    v = store.variants_in("22", 16_000_000, 17_000_000)  # Variants in a range; no r is read
    n = store.count_in("22", 16_000_000, 17_000_000)     # how many, from positions alone
    present = store.has(["22:16021005:C:G", "22:1:A:C"])  # [True, False]
    m = store.matrix(v.alid.tolist())                     # LDMatrix: signed r, NaN unstored
    r = store.region("22", 30_000_000, 1_000_000)         # radius ≤ far_bp / 2
    p = store.proxies(["22:16021005:C:G"], r2=0.2, kb=3_000)       # Proxies
    score = np.linspace(6, 1, len(v))                     # |z| or −log10 p, one per ALID
    c = store.clump(v.alid.tolist(), score, r2=0.01, kb=3_000)     # Clumps, as plink2 --clump
```

A request the store cannot answer exactly raises `ogquery.ld.Refusal`, naming
its reason (`RefusalReason`), and a malformed one `ogquery.ld.InputError`.
Every unstored pair is NaN; only a caller's `fill` replaces it, and the result
records the fill. `variants_in`, `count_in` and `has` read the variant table
alone, so a caller can size a region or check membership before any r is
read; `has` is true exactly where the queries find an ALID present, and an
indel spelt reference first is refused (`ogquery.alid.ReferenceFirstIndel`), as
the queries refuse it. A Genotype Store is opengwasdb's to read
(`opengwasdb.ld.open`); `ogquery.ld.open` refuses one, naming it.

## Develop

```bash
pixi run -e test test              # the whole suite, with the `http` extra
pixi run -e test-nohttp test -m "not http"   # and without it
pixi run -e test contract          # the contract bundle alone
pixi run -e dev lint               # ruff; `typecheck` is mypy --strict
pixi run -e dev contract-check     # tests/contract is opengwasdb's bundle
```

Or, without pixi: `pip install -e ".[test,http]"` and `pytest`.

[CONTRIBUTING.md](./CONTRIBUTING.md) is how to contribute,
[AGENTS.md](./AGENTS.md) how work moves, [CONTEXT.md](./CONTEXT.md) the
vocabulary, and [docs/adr/](./docs/adr/) the decisions.

## Releases

Every pull request merged into `main` is a version: `release.yml` tags it
from its changelog fragments, and `publish.yml` publishes it to PyPI by
trusted publishing ([ADR 0001](./docs/adr/0001-ogquery-is-the-store-reader-extracted-from-opengwasdb.md);
opengwasdb ADR 0083).

**Merge with "Create a merge commit"**, never squash or rebase. A squash loses
the history this package carries. A rebase puts each of the branch's commits on
`main`, about 65 for the first merge, which is more than one release run
publishes (`MAX_PLANNED_RELEASES`), so nothing is tagged. Only a merge commit
keeps the history and makes the merge one version.

**The first release.** The first merge skips the repository's first commit
(`a8822a8`, never released) and becomes `v0.1.0`, with a tag and a GitHub
Release. It then dispatches `publish.yml` with `index=pypi`, which publishes
straight to PyPI with no TestPyPI rehearsal (the owner's ruling). If the
trusted publisher is not registered yet, the upload fails at OIDC before
anything reaches PyPI, and the daily run (06:17 UTC) retries with the same
archives. The GitHub environments `pypi` and `testpypi`, limited to `main`, are
the owner's to create; `pypi` must exist before the upload.

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

The LD read path came from opengwasdb
[`c8ebe4ca`](https://github.com/opengwas/opengwasdb/commit/c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4)
as a plain copy, split from its build code by name, so its history there is
opengwasdb's ([ADR 0002](./docs/adr/0002-the-ld-read-path-is-ogquery-s.md)):
`opengwasdb/ld/{format,store,query}/` is `ogquery/ld/{format,store,query}/`,
and its tests and their data are `tests/ld/`.

## Licence

GPL-3.0-only. See [LICENSE](./LICENSE). Copyright © University of Bristol.
Contributions from outside the University of Bristol need a signed
[Contributor License Agreement](./CLA.md).
