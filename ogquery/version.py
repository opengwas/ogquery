"""The package version, read from the installed distribution (as opengwasdb#333).

Every merge to `main` is tagged `vX.Y.Z` by `.github/workflows/release.yml`, and
`hatch-vcs` turns that tag into the distribution's version at build time (the comment
in `pyproject.toml`). Nothing here computes a version: what this package reports is
whatever the installed distribution says. A release checkout says `0.1.0`; a checkout
past the tag says `0.1.1.dev4+g1a2b3c4`; a run with no installed metadata at all says
`FALLBACK_VERSION`, which is a valid PEP 440 version, sorts below every real one, and
cannot be mistaken for a release.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

FALLBACK_VERSION = "0.0.0+unknown"


def installed_version() -> str:
    """The installed distribution's version, or `FALLBACK_VERSION`.

    Only "there is no installed distribution" is answered with the fallback. A
    distribution whose metadata cannot be read is a different failure, and raises.
    """
    try:
        return version("ogquery")
    except PackageNotFoundError:
        return FALLBACK_VERSION


__version__ = installed_version()
