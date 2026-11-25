# Changelog fragments

Every pull request that changes `ogquery/` adds **one or more fragments**
here, and every merge to `main` becomes a version whose release notes are
exactly the fragments that merge added. There is no `CHANGELOG.md` and no
`Unreleased` section to edit: the history is the GitHub Releases. The rule is
opengwasdb's (its `changelog.d/README.md`, opengwas/opengwasdb#333), ported for
opengwas/ogquery#1.

A pull request with no user-visible effect is the exception, and it says so
with the **`no-changelog`** label on the pull request, not with an empty
fragment. `scripts/check_changelog.py` enforces both, in CI and locally:

    python scripts/check_changelog.py --base-ref origin/main

## The file

`changelog.d/<slug>.<type>.md`, directly in this directory.

- **`<slug>`** names the change; `1-store-reader` and `12-http-reads` are the
  shape (the issue number, then a few words). It only has to be unique among
  the fragments a pull request adds: the file is never edited or deleted.
- **`<type>`** is one of the five below. It is not decoration — it is the
  version bump (see "What the type decides").
- The **body** is Markdown: lead with the change and its consequence, name the
  issue and ADR, and give the measured numbers when there are numbers. It
  becomes the bullet in the release notes as written, so there is no need to
  repeat it anywhere else.

```markdown
- **`open_release` refuses a 0.2.x release opened without a Variant Table
  reader by naming its `format_version` and `ogdb rewrite-release`
  (opengwas/ogquery#1).**
```

A fragment that is empty, or whose name the release cannot read, is refused
rather than skipped: a silently omitted note is a release that does not say
what it changed.

## The five types

| type | write it when | bump |
|---|---|---|
| `breaking` | the API, or how a Store Release is read, changes incompatibly | minor |
| `added` | a new capability, backwards compatible | minor |
| `changed` | an existing behaviour changes | minor |
| `fixed` | a defect is fixed, and nothing else changes | patch |
| `internal` | the change is real but invisible to a user of the package | patch |

## What the type decides

The package follows [Semantic Versioning](https://semver.org); below `1.0.0`
the minor number carries breaking changes. `scripts/release.py` reads the
fragment types a pull request added and picks the strongest bump:

- any `breaking`, `added` or `changed` → the minor number moves (`0.1.1` →
  `0.2.0`);
- only `fixed` or `internal`, or no fragment at all → the patch number moves
  (`0.1.0` → `0.1.1`).

The first release is `0.1.0`: the merge that adds the package, whose fragment
is `added`. The repository's first commit, which holds only the licence and the
contributor agreements, is never released (`release.NEVER_RELEASED`).

`.github/workflows/release.yml` runs that script on a push to `main`, tags the
merge commit `vX.Y.Z` and publishes a GitHub Release whose notes are the
fragments, with a link to the pull request. It never commits to `main`, it never
moves a tag, and every run also finishes merges an earlier run could not publish
(a failed `gh release create` is retried, not skipped). It then dispatches
`.github/workflows/publish.yml`, which uploads every tag from
`PYPI_FIRST_RELEASE` on (`scripts/pypi_release.py`) to PyPI by trusted
publishing, as opengwasdb ADR 0083 designs it.
