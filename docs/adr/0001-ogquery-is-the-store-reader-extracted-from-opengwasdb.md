# ogquery is the Store Reader, extracted from opengwasdb

## Status

Accepted with [ogquery#1](https://github.com/opengwas/ogquery/issues/1), the first ticket
of the second part of epic
[opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450). The owner
planned the extraction in that epic; this records what moved and the choices the move
made. Two of them are the owner's rulings of 25 Nov 2026, marked below: the Python floor,
and a first release published to PyPI with no rehearsal.

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
`ogquery.release.VariantTableReaderRequired` with `REWRITE_COMMAND` (below). The package
carries `py.typed`, so a caller type-checked with `mypy --strict`, as opengwasdb is,
sees its types (CI checks a strict caller against the installed wheel).
`ogquery.digests` opens a release with the Store Reader alone by default
(`digests.open_query`), where it used opengwasdb's `query_store`.

**Runtime dependencies are numpy, zarr and numcodecs**, at opengwasdb's bounds raised to
what zarr 3.4 allows (`numpy>=2.0,<2.5`, `zarr>=3.4,<4`, `numcodecs>=0.17`; opengwasdb's
`numpy>=1.26` cannot resolve beside zarr 3.4), and nothing else, so `pip install ogquery` compiles
nothing on Linux, macOS or Windows. `tests/unit/test_store_reader_imports.py` holds the
package's imports to them, the declared dependencies to exactly them, and a fresh
interpreter that imports every module to loading none of pysam, sqlite3, scipy, pandas or
threadpoolctl; the contract test asks every question in a fresh interpreter and holds it
to the same. CI's clean install holds what pip installs to their closure, all wheels, and
CI's `floors` job runs the suite at the three floors together (`.github/floors.txt`).

**Python 3.12 or later** (owner's ruling, 25 Nov 2026), replacing the 3.11 the ticket
names. zarr 3.4, opengwasdb's floor, requires 3.12 (zarr 3.2 raised it). Keeping 3.11 would
mean allowing zarr 3.1 for it, a version the Store Reader was never tested on. Lowering the
floor is a later decision, with the contract run on 3.11 and zarr 3.1 first.

**It reads 0.3.0 alone, and 0.2.x with a reader for its Store Variant Table.** A 0.2.x
(and 0.1.0) release answers its variants from a BGZF/tabix `variants.tsv.gz` and, for a
Dense or Hybrid one, `index.sqlite`'s `variant_aliases`. ogquery defines the reader's
protocol (`variant_axis.LegacyVariantTable`) and opengwasdb implements it; a caller passes
it as `open_release(path, legacy=...)`. Without one, a query of such a release raises
`VariantTableReaderRequired`, a `ValueError` naming the release's `format_version` and how
it becomes a 0.3.0 release: a 0.2.3 one by `ogdb rewrite-release`; a 0.1.0 or 0.2.0-0.2.2
one by a rebuild from source, since the rewrite accepts 0.2.3 alone (opengwasdb's
`store.rewrite`). Advice to run a command that would refuse the release is a wrong answer
that looks like a right one, so the message never gives it.

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

**Releases follow opengwasdb ADR 0083, with one departure.** Every merge to `main` is
tagged from its changelog fragments (`release.yml`), and `publish.yml` publishes every tag
from `PYPI_FIRST_RELEASE` on to PyPI by trusted publishing, environment `pypi` (`testpypi`
for a rehearsal). **The departure (owner's ruling, 25 Nov 2026):** ADR 0083's boundary
starts closed and opens after a TestPyPI rehearsal; ogquery's starts open,
`PYPI_FIRST_RELEASE = "v0.1.0"`, so the first merge is published to PyPI directly, with no
rehearsal. The repository's first commit, which predates the package, is never released
(`release.NEVER_RELEASED`).

**A pull request is merged with a merge commit**, never squashed or rebased. A squash
drops the history this package carries; a rebase puts each of its commits on `main`'s
first-parent line, and the first merge (about 65 of them) is over `MAX_PLANNED_RELEASES`,
so the release run stops and nothing is tagged. Only a merge commit keeps the history and
makes the merge one version, `v0.1.0`.

### The first release

What the first merge to `main` does:

1. `release.yml` plans the first-parent commits past the newest tag. There is none, so it
   skips `a8822a8` (`NEVER_RELEASED`) and plans the merge commit; its `added` fragment makes
   it `v0.1.0`. It pushes the tag, creates the GitHub Release with the fragment as notes,
   and dispatches `publish.yml` with `index=pypi`.
2. `publish.yml` builds and checks the archives from the tag, attaches them to the Release,
   then uploads them from the `pypi` environment by OIDC.
3. If the trusted publisher is not yet registered, the upload fails at the OIDC exchange,
   before anything reaches PyPI: nothing partial is published. The verified archives stay
   on the Release, and the daily run (06:17 UTC), or a manual dispatch, retries with the
   same bytes.

The GitHub environments `pypi` and `testpypi`, each limited to deployments from `main`
(ADR 0083), are the owner's to create, and `pypi` must exist before the upload. A run
that names an environment the repository lacks has GitHub create it without that
policy.

### Changing what the Store Reader answers

The contract bundle is generated in opengwasdb, which builds Stores, and read here. A
change that alters an answer therefore moves in one order:

> **The reader that answers changes first; opengwasdb's generator then runs that reader to
> regenerate the bundle; ogquery commits that bundle with `SOURCE` naming the opengwasdb
> commit that generated it.**

- **Before #475** opengwasdb's generator runs opengwasdb's own in-tree reader. opengwasdb
  changes its reader and regenerates the bundle, and merges. ogquery then makes the same
  reader change and runs `scripts/contract_bundle.py refresh --commit <that opengwasdb
  commit>` in one pull request.
- **After #475** opengwasdb's generator runs ogquery. The ogquery change merges first. Its
  pull request carries the refreshed bundle, generated on an opengwasdb branch that
  installs that ogquery commit; the branch is pushed, not merged, and `SOURCE` names its
  commit. ogquery merges and releases. Then opengwasdb moves its `ogquery` pin to that
  release, regenerates (byte for byte the same bundle) and merges with a merge commit, so
  the commit `SOURCE` names stays on opengwasdb's history.

Neither side ever waits for the other to merge or release first: ogquery's pull request
needs only an opengwasdb branch commit, which exists before either merges, and opengwasdb's
merge needs only the ogquery release, which exists by then. A change that alters no answer
needs no bundle change and moves on either side alone.

## Consequences

- Until #475, the Store Reader exists twice: here and in opengwasdb. #475 makes
  `opengwasdb.store_reader` (and `opengwasdb.query`) re-export ogquery, pinned
  `ogquery>=0.1,<0.2` (below 1.0 a minor carries breaking changes), and deletes the copy
  and the unit tests that moved; after it, a change to the read path is made here and
  released before opengwasdb takes it, in the order above.
- Issue and ADR numbers in code that came from opengwasdb are opengwasdb's; new
  references name the repository.
- Renaming `publish.yml` or the `pypi`/`testpypi` environments breaks publication until
  the trusted publishers on PyPI and TestPyPI are changed to match.
