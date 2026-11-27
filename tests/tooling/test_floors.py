"""`.github/floors.txt` pins each runtime dependency at pyproject.toml's floor, and
`.github/floors-http.txt` and `floors-genotype.txt` each dependency of the `http` and `genotype`
extras at its floor.

CI's `floors` job installs those pins and runs the suite, so a floor pyproject.toml
promises is one the Store Reader was tested on. Here, the files agree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _floors_declared(extra: str | None = None) -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    specs = project["dependencies"] if extra is None else project["optional-dependencies"][extra]
    out = {}
    for spec in specs:
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)", spec)
        assert match, f"{spec!r} declares no floor"
        out[match[1]] = match[2]
    return out


def _floors_pinned(name: str = "floors.txt") -> dict[str, str]:
    out = {}
    for line in (ROOT / ".github" / name).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, version = line.partition("==")
            out[name.strip()] = version.strip()
    return out


def _release(version: str) -> tuple[int, ...]:
    parts = [int(p) for p in version.split(".")]
    return tuple(parts + [0] * (3 - len(parts)))


def test_every_floor_is_pinned_exactly_at_the_declared_floor():
    declared, pinned = _floors_declared(), _floors_pinned()
    assert set(pinned) == set(declared) == {"numpy", "zarr", "numcodecs"}
    for name, floor in declared.items():
        assert _release(pinned[name]) == _release(floor), (name, floor, pinned[name])


def test_every_http_floor_is_pinned_exactly_at_the_declared_floor():
    declared, pinned = _floors_declared("http"), _floors_pinned("floors-http.txt")
    assert set(pinned) == set(declared) == {"fsspec", "aiohttp"}
    for name, floor in declared.items():
        assert _release(pinned[name]) == _release(floor), (name, floor, pinned[name])


def test_every_genotype_floor_is_pinned_exactly_at_the_declared_floor():
    declared, pinned = _floors_declared("genotype"), _floors_pinned("floors-genotype.txt")
    assert set(pinned) == set(declared) == {"pgenlib", "threadpoolctl"}
    for name, floor in declared.items():
        assert _release(pinned[name]) == _release(floor), (name, floor, pinned[name])


def test_dev_carries_the_genotype_extra_where_pgenlib_has_wheels():
    # `dev` names the two modules itself, under a marker, rather than `ogquery[genotype]`: pixi
    # drops a marker on a self-reference, and a Windows dev environment would compile pgenlib.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    marked = {f"{spec}; sys_platform != 'win32'" for spec in extras["genotype"]}
    assert marked <= set(extras["dev"]), extras["dev"]
