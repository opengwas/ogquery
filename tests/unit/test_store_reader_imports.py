"""ogquery imports numpy, zarr, numcodecs, the standard library and itself (#464, ogquery#1),
and `ogquery.ld.genotypes` the `genotype` extra besides (opengwas/ogquery#11).

ogquery is the read path every reader of a Store Release uses, and installs on Linux,
macOS and Windows with nothing to compile only while nothing in it reaches for a heavy
dependency. Four checks hold that:

- the static one parses every module of the package and refuses any import -- at
  module level, inside a function or under `TYPE_CHECKING`, which all couple a file to
  what it names -- outside numpy, zarr, numcodecs, the standard library less `sqlite3`,
  and `ogquery` itself; under `ogquery/ld/genotypes/` alone it also allows pgenlib and
  threadpoolctl, the `genotype` extra, which a Genotype Store's reader needs (ADR 0002,
  amended);
- the declared one requires `pyproject.toml`'s runtime dependencies to be exactly
  numpy, zarr and numcodecs, and the `genotype` extra exactly what that package
  imports, so the static check and what pip installs agree;
- the run-time one imports every module but the genotype package's in a fresh
  interpreter, and queries a GWAS release, and requires that pysam, sqlite3, scipy,
  pandas, pgenlib and threadpoolctl were never loaded, which the static check cannot
  see through a dependency's own imports;
- and the probe is shown to see a heavy import when one happens.

Asking every query shape of a release in a fresh interpreter is the contract's
`test_asking_every_question_loads_no_heavy_dependency` (tests/contract).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

#: CI runs the import boundary in its own step (`-m boundary`), and every other step leaves it
#: out (`-m "not boundary"`), so each of these runs once a leg.
pytestmark = pytest.mark.boundary

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ogquery"
OWN = "ogquery"
THIRD_PARTY = frozenset({"numpy", "zarr", "numcodecs"})
REFUSED_STDLIB = frozenset({"sqlite3"})
#: The `genotype` extra, and the one package that may import it: a Genotype Store's reader.
GENOTYPE_EXTRA = frozenset({"pgenlib", "threadpoolctl"})
GENOTYPE_PACKAGE = f"{PACKAGE}/ld/genotypes/"
HEAVY = ("pandas", "scipy", "pysam", "sqlite3", "pgenlib", "threadpoolctl")


def imports_in_source(path: str, source: str) -> set[str]:
    """Every dotted name `source` imports, wherever the import sits, relative imports
    resolved against `path`'s package (opengwasdb's `tests/architecture_model.py`)."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(path, node)
            targets.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return targets


def _import_base(path: str, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if not node.level:
        return module
    parts = path.removesuffix(".py").split("/")
    parts = parts[:-1] if parts[-1] != "__init__" else parts[:-1]
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, module] if module else parts)


def foreign_imports(path: str, source: str) -> list[str]:
    """Every name `source` imports that ogquery may not."""
    refused = []
    for name in sorted(imports_in_source(path, source)):
        top = name.split(".")[0]
        if name == OWN or name.startswith(OWN + "."):
            continue
        if top in THIRD_PARTY:
            continue
        if top in GENOTYPE_EXTRA and path.startswith(GENOTYPE_PACKAGE):
            continue
        if top in sys.stdlib_module_names and top not in REFUSED_STDLIB:
            continue
        refused.append(name)
    return refused


def package_sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / PACKAGE).rglob("*.py"))
    }


def test_the_check_refuses_what_it_should():
    path = f"{PACKAGE}/example.py"
    source = "\n".join(
        [
            "import numpy as np",
            "import sqlite3",
            "from typing import TYPE_CHECKING",
            "from . import facade",
            "from .associations import record",
            "from ogquery.alid import parse_canonical_alid",
            "if TYPE_CHECKING:",
            "    from opengwasdb.encoding.association_codec import Encoded",
            "def f():",
            "    import scipy.stats",
            "    import pandas",
            "    from pysam import TabixFile",
        ]
    )
    assert foreign_imports(path, source) == [
        "opengwasdb.encoding.association_codec.Encoded",
        "pandas",
        "pysam.TabixFile",
        "scipy.stats",
        "sqlite3",
    ]


def test_the_check_allows_the_genotype_extra_only_in_the_genotype_package():
    source = "import pgenlib\nfrom threadpoolctl import threadpool_limits\n"
    assert foreign_imports(f"{GENOTYPE_PACKAGE}reader.py", source) == []
    assert foreign_imports(f"{PACKAGE}/ld/query/ldstore.py", source) == [
        "pgenlib",
        "threadpoolctl.threadpool_limits",
    ]
    assert foreign_imports(f"{PACKAGE}/ld/genotypes.py", source) != [], "a module beside it"


def test_ogquery_imports_only_numpy_zarr_numcodecs_stdlib_and_itself():
    sources = package_sources()
    # The Store Reader is some 11,000 lines in 41 modules; a check over a handful of
    # files would pass a read path that had not reached the package.
    assert len(sources) >= 40, sorted(sources)
    assert sum(len(text.splitlines()) for text in sources.values()) > 10_000
    refused = {
        path: names for path, source in sources.items() if (names := foreign_imports(path, source))
    }
    assert refused == {}


def test_the_package_declares_exactly_the_three_dependencies_it_imports():
    """What pip installs is what the static check allows: no fourth runtime dependency,
    and no extra that the package itself imports."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0] for spec in project["dependencies"]}
    assert declared == THIRD_PARTY


def test_the_genotype_extra_is_exactly_what_the_genotype_package_imports():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extra = project["optional-dependencies"]["genotype"]
    declared = {re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0] for spec in extra}
    assert declared == GENOTYPE_EXTRA
    assert not any(";" in spec for spec in extra), "no platform marker: pip decides (ADR 0002)"
    imported = {
        name.split(".")[0]
        for path, source in package_sources().items()
        if path.startswith(GENOTYPE_PACKAGE)
        for name in imports_in_source(path, source)
    }
    assert GENOTYPE_EXTRA <= imported, "each module the extra installs is one the package reads"


def _loaded_after(statements: str) -> list[str]:
    probe = (
        "import importlib, json, sys\n"
        f"{statements}\n"
        f"print(json.dumps([name for name in {HEAVY!r} if name in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return list(json.loads(out.stdout.strip().splitlines()[-1]))


def test_importing_ogquery_loads_no_heavy_dependency():
    """Every module of the package, the facade and the digests included, imported in one
    fresh interpreter: the package `__init__` alone imports nothing, so importing only it
    would prove nothing about the modules a query loads."""
    modules = sorted(
        path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
        for path in package_sources()
        if not path.startswith(GENOTYPE_PACKAGE)
    )
    assert {"ogquery.facade", "ogquery.digests", "ogquery.release", "ogquery.ld"} <= set(modules)
    statements = f"for name in {modules!r}:\n    importlib.import_module(name)\n"
    statements += "import ogquery\nogquery.__version__\n"
    assert _loaded_after(statements) == []


def test_querying_a_gwas_release_loads_no_heavy_dependency():
    """A GWAS release opened and asked a question loads neither the `genotype` extra nor any
    other heavy module (the contract asks every question; its list predates pgenlib)."""
    release = ROOT / "tests" / "contract" / "releases" / "dense_ranges"
    statements = (
        "from ogquery.facade import query_release\n"
        "from ogquery.release import open_release\n"
        f"with query_release(open_release({str(release)!r})) as q:\n"
        "    assert len(q.variants_table())\n"
    )
    assert _loaded_after(statements) == []


def test_the_run_time_probe_sees_a_heavy_import():
    """The probe's check, on a statement that does load one, finds it: `sqlite3` is in
    every Python's standard library, so the check never depends on what is installed."""
    assert _loaded_after("import ogquery.facade\nimport sqlite3") == ["sqlite3"]
