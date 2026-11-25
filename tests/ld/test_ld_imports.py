"""Importing the LD read path, and querying the Test Store with it, loads no build code and no
heavy dependency (ogquery#2; ADR 0002).

opengwasdb's `opengwasdb.ld` imported its builder and validator when imported, and its Genotype
Store reader imported threadpoolctl. ogquery holds the read side alone, so a fresh interpreter
that imports `ogquery.ld` and asks every LD question of a local Test Store must never have
loaded opengwasdb, pgenlib, threadpoolctl, fsspec, aiohttp, scipy, pysam, pandas or sqlite3.
The static boundary over every module is `tests/unit/test_store_reader_imports.py`'s.

One is measured against a baseline: `import zarr` itself imports fsspec whenever fsspec is
installed (zarr 3.4's storage module), whatever imports zarr. So what ogquery's LD read path may
not add is measured over what numpy, zarr and numcodecs load alone; without the `http` extra
that baseline is empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"
#: Nothing here, and nothing it imports, may load these.
FORBIDDEN = (
    "opengwasdb",
    "pgenlib",
    "threadpoolctl",
    "fsspec",
    "aiohttp",
    "scipy",
    "pysam",
    "pandas",
    "sqlite3",
)

EVERY_QUESTION = f"""
import numpy as np
import ogquery.ld
store = ogquery.ld.open({str(TEST_STORE)!r})
rows = store.variants_in("22", 1, 3_000_000_000).alid.tolist()
assert store.count_in("22", 1, 3_000_000_000) == len(rows)
assert store.has(rows[:3]).all()
store.matrix(rows[:50]).kinds()
store.region("22", 30_000_000, 1_000_000)
store.proxies(rows[:20], r2=0.5, kb=3000)
store.clump(rows[:200], np.arange(200.0), r2=0.1, kb=3000).leads()
store.close()
"""


def _loaded_after(statements: str) -> list[str]:
    probe = (
        "import json, sys\n"
        f"{statements}\n"
        "tops = {name.split('.')[0] for name in sys.modules}\n"
        f"print(json.dumps(sorted(tops & {set(FORBIDDEN)!r})))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    loaded: list[str] = json.loads(done.stdout)
    return loaded


#: What ogquery's three dependencies load by themselves.
BASELINE = "import numcodecs, numpy, zarr"


#: The one allowance, an accepted exception to the import boundary (ADR 0002): with the `http`
#: extra installed, zarr's own import loads fsspec, whatever imports zarr, so a local LD query
#: loads it too. ogquery adds nothing to it; without the extra it is not there to load.
ALLOWED_IN_THE_BASELINE = frozenset({"fsspec"})


def test_the_baseline_loads_no_build_code_and_at_most_fsspec() -> None:
    assert set(_loaded_after(BASELINE)) <= ALLOWED_IN_THE_BASELINE


def test_asking_every_ld_question_loads_no_build_code_and_no_heavy_dependency() -> None:
    assert _loaded_after(EVERY_QUESTION) == _loaded_after(BASELINE)


def test_importing_every_ld_module_loads_none_of_them() -> None:
    package = Path(__file__).resolve().parents[2] / "ogquery" / "ld"
    modules = sorted(
        ".".join(("ogquery", "ld", *path.relative_to(package).with_suffix("").parts)).removesuffix(
            ".__init__"
        )
        for path in package.rglob("*.py")
    )
    assert len(modules) >= 20, modules
    imports = "\n".join(f"import {module}" for module in modules)
    assert _loaded_after(imports) == _loaded_after(BASELINE)


def test_the_probe_sees_a_forbidden_import() -> None:
    # scipy is a test dependency (pyproject.toml), so the probe can be shown to see one load.
    baseline = _loaded_after(BASELINE)
    assert _loaded_after("import ogquery.ld\nimport scipy.special") == sorted([*baseline, "scipy"])
