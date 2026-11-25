# Working in this repository

Read [CONTRIBUTING.md](./CONTRIBUTING.md) before making changes, and
[CLAUDE.md](./CLAUDE.md) for the short form of the rules: the two boundaries, the
environment, and the checks before finishing. This file exists so that a session looking
for `AGENTS.md` finds the same instruction rather than concluding there is none.

## How work moves

Work on a feature branch and open a pull request against **`main`**. **Every pull
request merged into `main` is a tagged version**: `.github/workflows/release.yml` tags the
merge `vX.Y.Z` and publishes a GitHub Release whose notes are that pull request's
changelog fragments, and `.github/workflows/publish.yml` puts it on PyPI (ADR 0001;
opengwasdb ADR 0083).

A pull request that changes `ogquery/` adds `changelog.d/<slug>.<type>.md`, where the
type is `breaking`, `added`, `changed`, `fixed` or `internal`. The type sets the bump --
`breaking`, `added` or `changed` is a minor, `fixed` or `internal` a patch -- so there is
no version to bump by hand. [`changelog.d/README.md`](./changelog.d/README.md) says what a
fragment should say; the `no-changelog` label says there is nothing a user would notice.

CI must pass on Linux, macOS and Windows.

## Agent skills

**Issue tracker.** GitHub issues, here and in `opengwas/opengwasdb`. Status lives only on
the [OpenGWASDB project board](https://github.com/orgs/opengwas/projects/1). Its README
holds the conventions; read it with
`gh project view 1 --owner opengwas --format json --jq .readme` and follow it. Don't copy
it here.

**Domain docs.** [CONTEXT.md](./CONTEXT.md), which points to opengwasdb's glossary for
shared terms, and [docs/adr/](./docs/adr/).
