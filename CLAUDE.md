# Working in this repository

Short by design: it loads into every session. [CONTRIBUTING.md](./CONTRIBUTING.md)
and [AGENTS.md](./AGENTS.md) hold the rest.

## The principle

**A wrong answer that looks like a right answer is the worst thing this package can
produce.** ogquery is the one read path every OpenGWAS reader uses, so fail loudly rather
than degrade quietly, and never substitute a default for missing data.

## The two boundaries

- **Dependencies.** ogquery imports numpy, zarr, numcodecs, the standard library (not
  `sqlite3`) and itself, and nothing else, ever: no pysam, scipy, pandas or SQLite, not
  even inside a function. `tests/unit/test_store_reader_imports.py` fails otherwise. A
  test may use scipy as a reference; the package may not.
- **The contract bundle** (`tests/contract/`) is opengwasdb's and read only. Never edit
  `releases/` or `expected/` to make a test pass: a failing contract is a wrong answer.
  `tests/contract/README.md` says how to refresh it from opengwasdb.

## Environment

```bash
pixi run -e test test              # the whole suite, about two minutes
pixi run -e dev lint               # ruff
pixi run -e dev typecheck          # mypy --strict
pixi run -e dev python …           # anything ad hoc
```

`pixi` may not be on PATH: `export PATH="$HOME/.pixi/bin:$PATH"`.

## Before you finish a change

- [ ] A `changelog.d/<slug>.<type>.md` fragment for any change to `ogquery/`
      (`changelog.d/README.md`); CI fails the pull request without one.
- [ ] No ruff or mypy finding.
- [ ] New tests observed to fail against the unfixed code.
- [ ] The Store Reader's public names unchanged unless the change says so: opengwasdb
      re-exports them.
- [ ] Merged with a merge commit, never squashed or rebased (CONTRIBUTING.md, Development).

## Where the authority lives

| | |
|---|---|
| [CONTEXT.md](./CONTEXT.md) | ogquery's terms; shared terms are opengwasdb's glossary |
| [docs/adr/](./docs/adr/) | ogquery's decisions; opengwasdb's ADRs govern the Store format |
| [opengwasdb's spec](https://github.com/opengwas/opengwasdb/blob/main/docs/spec/store-format.md) | the Store format the reader reads |

A bare `#NNN` or `ADR NNNN` in code that came from opengwasdb is opengwasdb's.
