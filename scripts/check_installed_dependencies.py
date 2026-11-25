#!/usr/bin/env python3
"""A clean `pip install ogquery` installs numpy, zarr, numcodecs and their own
dependencies, every one a wheel: nothing else, and nothing compiled (ogquery#1).

Run it with the interpreter of a fresh virtual environment the wheel was installed into,
isolated, after an install that wrote pip's report:

    venv/bin/python -m pip install --only-binary=:all: --report report.json dist/ogquery-*.whl
    venv/bin/python -I scripts/check_installed_dependencies.py --report report.json

It checks:

* ogquery's own runtime requirements are exactly numpy, zarr and numcodecs;
* every distribution in the environment is ogquery, one of the requirements ogquery's
  metadata reaches, recursively, with each requirement's extras and markers evaluated
  for this interpreter, or pip itself -- no other distribution;
* every distribution the install brought came as a wheel (pip's report), so nothing was
  built from source. `--only-binary=:all:` makes pip refuse a source build outright; the
  report shows it held.

It prints what was installed, which is the evidence a reviewer reads. It needs
`packaging`, which zarr requires, so it is there whenever ogquery is.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import Distribution, distribution, distributions
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

PACKAGE = "ogquery"
#: ogquery's runtime dependencies, and all of them (pyproject.toml).
DIRECT = frozenset({"numpy", "zarr", "numcodecs"})
#: What a fresh virtual environment holds before anything is installed into it.
BOOTSTRAP = frozenset({"pip"})


def canonical(name: str) -> str:
    """PEP 503's normalised name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(dist: Distribution, extras: frozenset[str]) -> list[Requirement]:
    """`dist`'s requirements that apply here, with `extras` asked for."""
    wanted = []
    for spec in dist.requires or []:
        requirement = Requirement(spec)
        marker = requirement.marker
        if marker is None:
            wanted.append(requirement)
        elif any(marker.evaluate({"extra": extra}) for extra in (extras or {""})):
            wanted.append(requirement)
    return wanted


def closure(root: str) -> dict[str, set[str]]:
    """Every distribution `root`'s metadata reaches, each with the extras asked of it."""
    reached: dict[str, set[str]] = {canonical(root): set()}
    pending: list[tuple[str, frozenset[str]]] = [(root, frozenset())]
    while pending:
        name, extras = pending.pop()
        for requirement in _requirements(distribution(name), extras):
            key = canonical(requirement.name)
            new = set(requirement.extras) - reached.get(key, set())
            if key not in reached or new:
                reached.setdefault(key, set()).update(requirement.extras)
                pending.append((requirement.name, frozenset(reached[key])))
    return reached


def installed() -> dict[str, str]:
    return {canonical(d.metadata["Name"]): d.version for d in distributions()}


def problems(report: dict[str, Any]) -> list[str]:
    out = []
    direct = {canonical(r.name) for r in _requirements(distribution(PACKAGE), frozenset())}
    if direct != DIRECT:
        out.append(f"{PACKAGE} requires {sorted(direct)}, not exactly {sorted(DIRECT)}")
    allowed = set(closure(PACKAGE)) | BOOTSTRAP
    stray = sorted(set(installed()) - allowed)
    if stray:
        out.append(f"installed beyond ogquery's dependencies: {stray}")
    for item in report["install"]:
        url = item["download_info"]["url"]
        name = item["metadata"]["name"]
        if not url.split("#")[0].endswith(".whl"):
            out.append(f"{name} was not installed from a wheel: {url}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="pip install --report")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reached = closure(PACKAGE)
    print(f"Python {sys.version.split()[0]} on {sys.platform}; installed:")
    for name, version in sorted(installed().items()):
        why = "bootstrap" if name in BOOTSTRAP else "ogquery" if name == PACKAGE else (
            "direct" if name in DIRECT else "dependency" if name in reached else "STRAY"
        )
        print(f"  {name:24} {version:14} {why}")
    wheels = [item["download_info"]["url"].rsplit("/", 1)[-1] for item in report["install"]]
    print(f"from {len(wheels)} wheel(s): " + ", ".join(sorted(wheels)))
    found = problems(report)
    for problem in found:
        print(f"FAIL: {problem}", file=sys.stderr)
    if found:
        return 1
    print("ok: only numpy, zarr, numcodecs and their dependencies, all from wheels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
