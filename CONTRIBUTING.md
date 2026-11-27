# Contributing to ogquery

**Sign the Contributor License Agreement first.** ogquery is released under
GPL-3.0-only ([LICENSE](./LICENSE)), and the University of Bristol holds the
copyright. Every contribution from outside the University needs a signed
Harmony agreement; [CLA.md](./CLA.md) says which one and how to sign it. The CLA
bot asks on your first pull request into `main`, and a pull request is not
merged until it passes.

## Development

**Environment.** [pixi](https://pixi.sh) builds one from the package alone, on Linux,
macOS and Windows: ogquery and its dependencies from PyPI, and the tools beside them.

```bash
pixi run -e test test              # the whole suite: contract, LD, unit and tooling tests
pixi run -e test-py314 test        # the same on the highest Python supported
pixi run -e test-nohttp test -m "not http and not genotype"   # without either extra
pixi run -e test-nogenotype test -m "not genotype"            # without the `genotype` extra
pixi run -e dev lint               # ruff
pixi run -e dev typecheck          # mypy --strict on ogquery/ and scripts/
pixi run -e dev contract-check     # tests/contract is opengwasdb's bundle, byte for byte
```

Without pixi, `pip install -e ".[test,http,genotype]"` and `pytest` run the same tests. CI runs
them on Linux, macOS and Windows, on Python 3.12 and 3.14, and once at the dependency floors.
pgenlib, which the `genotype` extra installs, publishes no Windows wheel: there the extra is left
out and the tests marked `genotype`, which read a Genotype Store, are skipped, naming the
platform.

**What a change must keep.** ogquery imports numpy, zarr, numcodecs, the standard
library (not `sqlite3`) and itself, nothing else
(`tests/unit/test_store_reader_imports.py`), but for `ogquery/ld/genotypes/`, which alone
imports pgenlib and threadpoolctl, the `genotype` extra (ADR 0002, amended). The contract bundle in `tests/contract/` is
opengwasdb's and is never edited here, and so is the LD test data in `tests/ld/data/`
(`tests/ld/README.md`); a change that alters what the Store Reader answers
follows the order in [ADR 0001](./docs/adr/0001-ogquery-is-the-store-reader-extracted-from-opengwasdb.md)
("Changing what the Store Reader answers"), and `tests/contract/README.md` says how to
refresh the bundle. New tests must be seen to fail against the unfixed code.

**The changelog.** A pull request that changes `ogquery/` adds
`changelog.d/<slug>.<type>.md`; [changelog.d/README.md](./changelog.d/README.md) says
what to write and how the type sets the version. CI fails a pull request without one,
unless it carries the `no-changelog` label.

**Merging.** Every pull request merged into `main` is a tagged release (README,
"Releases"). Merge with **"Create a merge commit"**, never "Squash and merge" or "Rebase
and merge": a squash loses the history the package carries, and a rebase puts each
commit on `main`, which the release run refuses to publish as one version per commit.

### Licence

By contributing, you agree that your contribution is licensed as described in
the CLA you signed. ogquery is released under GPL-3.0-only ([LICENSE](./LICENSE)).
