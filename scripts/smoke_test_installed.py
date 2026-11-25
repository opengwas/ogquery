#!/usr/bin/env python3
"""Use an installed ogquery the way a user would, outside any checkout.

The test suite runs against an editable install of the working tree, so it
cannot see what a `pip install ogquery` user gets: a wheel that left out a
module, a dependency only the developer environment supplies, or a version
that is not the one the archive was built as. This runs with the interpreter
of a clean environment the wheel was installed into (ported from opengwasdb's
`scripts/smoke_test_installed.py`, opengwas/opengwasdb#438), and checks:

* `ogquery` is imported from that environment's site-packages, never from a
  checkout;
* it reports the version the wheel was built as;
* one release of the Store Reader contract bundle, copied out of the checkout,
  opens with `open_release` and `query_release` alone and gives every readable
  answer its `expected/<release>.json` holds (`tests/contract/README.md`), with
  `pval` and `log10_p` compared to the bundle's cross-machine tolerance;
* none of pandas, scipy, pysam or sqlite3 was loaded on the way.

Run it isolated (`-I`), so neither the current directory nor the script's own
directory can put a checkout on the import path, and from a scratch directory:

    venv/bin/python -I scripts/smoke_test_installed.py \
        --fixture tests/contract/releases/dense_ranges \
        --expected tests/contract/expected/dense_ranges.json --version 0.1.0

The release is copied, so it is read from files outside the checkout.
Standard library and ogquery only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import sysconfig
import tempfile
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

#: What neither importing ogquery nor asking it every question may load.
HEAVY = ("pandas", "scipy", "pysam", "sqlite3")

DEFAULT_FIXTURE = Path("tests/contract/releases/dense_ranges")
DEFAULT_EXPECTED = Path("tests/contract/expected/dense_ranges.json")


class SmokeTestFailure(Exception):
    """The installed package does not behave as a user is promised."""


def check_import_location() -> Path:
    """The package directory, which must be in this environment's site-packages."""
    import ogquery

    package = Path(ogquery.__file__).resolve().parent
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not package.is_relative_to(site_packages):
        raise SmokeTestFailure(
            f"ogquery was imported from {package}, not from this environment's "
            f"site-packages {site_packages}: a checkout is on the import path"
        )
    return package


def check_version(required: str) -> None:
    import ogquery

    installed = distribution_version("ogquery")
    if (ogquery.__version__, installed) != (required, required):
        raise SmokeTestFailure(
            f"ogquery.__version__ is {ogquery.__version__} and the installed "
            f"metadata says {installed}; the wheel was built as {required}"
        )


def _open_query(path: Path) -> Any:
    """The query facade over a release, opened by the Store Reader alone."""
    from ogquery.facade import query_release
    from ogquery.release import open_release

    return query_release(open_release(path))


def check_release(release: Path, expected: dict[str, Any]) -> int:
    """The copied release answers as `expected` says; how many answers were compared."""
    from ogquery.digests import CROSS_MACHINE_RTOL, readable_differences, release_answers

    with _open_query(release) as q:
        if not q.variants_table():
            raise SmokeTestFailure(f"{release.name} opened with no variants")
    found = release_answers(release, query=_open_query)["readable"]
    want = expected["readable"]
    problems = readable_differences(want, found, rtol=CROSS_MACHINE_RTOL)
    if problems:
        shown = ", ".join(problems[:5])
        raise SmokeTestFailure(
            f"{release.name} answers other than its expected file at {len(problems)} "
            f"place(s): {shown}"
        )
    return sum(len(section) for section in want.values() if isinstance(section, dict))


def check_no_heavy_import() -> None:
    loaded = [name for name in HEAVY if name in sys.modules]
    if loaded:
        raise SmokeTestFailure(f"importing and querying ogquery loaded {loaded}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help="a contract-bundle release directory")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED,
                        help="that release's expected answers")
    parser.add_argument("--version", required=True, help="the version the wheel was built as")
    args = parser.parse_args(argv)
    try:
        package = check_import_location()
        check_version(args.version)
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as scratch:
            release = Path(shutil.copytree(args.fixture, Path(scratch) / args.fixture.name))
            compared = check_release(release, expected)
        check_no_heavy_import()
    except (SmokeTestFailure, OSError) as failure:
        print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"ok: ogquery {args.version} from {package}; {args.fixture.name} gave all "
        f"{compared} readable answers it is expected to, loading none of {', '.join(HEAVY)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
