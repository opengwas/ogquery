"""The public surface, pinned signature by signature.

What a caller of `ogquery.ld` sees is pinned here: `ogquery.ld.open`, the queries and their
results, and the identity functions an LD Store's ALIDs are spelt by. Changing an entry is an
API change, made on purpose, with a changelog fragment that says so; opengwasdb re-exports these
names.

From opengwasdb's `tests/ld/test_surface.py` at `c8ebe4ca` (ogquery#2), less what is
opengwasdb's: `build`, `validate`, `BuildInputs`, `Finding`, the CLI and the Genotype Store's
pgen reads. `StoreReader` is a seam, not pinned (owner, 10 November 2026).
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ogquery.ld
from ogquery import alid, identity

#: The committed Test Store (tests/ld/contract/support.py), written by the walking skeleton.
TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"

SURFACE: dict[str, tuple[Callable[..., Any], str]] = {
    "ogquery.ld.open": (
        ogquery.ld.open,
        "(location: 'str | os.PathLike[str]', *, cache_bytes: 'int | None' = None) -> 'LDStore'",
    ),
    "ogquery.ld.LDStore.matrix": (
        ogquery.ld.LDStore.matrix,
        "(self, alids: 'Sequence[str]', *, fill: 'float | None' = None) -> 'LDMatrix'",
    ),
    "ogquery.ld.LDStore.region": (
        ogquery.ld.LDStore.region,
        "(self, chrom: 'str', centre: 'int', radius: 'int', *, fill: 'float | None' = None)"
        " -> 'LDMatrix'",
    ),
    "ogquery.ld.LDStore.proxies": (
        ogquery.ld.LDStore.proxies,
        "(self, alids: 'Sequence[str]', *, r2: 'float' = 0.8, kb: 'float' = 10000) -> 'Proxies'",
    ),
    "ogquery.ld.LDStore.clump": (
        ogquery.ld.LDStore.clump,
        "(self, alids: 'Sequence[str]', score: 'npt.ArrayLike', *, r2: 'float', kb: 'float')"
        " -> 'Clumps'",
    ),
    "ogquery.ld.LDMatrix": (
        ogquery.ld.LDMatrix,
        "(r: 'npt.NDArray[np.float64]', variants: 'Variants', absent: 'tuple[str, ...]', "
        "t_far: 'float', near_bp: 'int', far_bp: 'int', fill: 'float | None') -> None",
    ),
    "ogquery.ld.LDMatrix.kinds": (
        ogquery.ld.LDMatrix.kinds,
        "(self) -> 'npt.NDArray[np.str_]'",
    ),
    "ogquery.ld.Proxies": (
        ogquery.ld.Proxies,
        "(query: 'npt.NDArray[np.str_]', proxy: 'npt.NDArray[np.str_]', "
        "r: 'npt.NDArray[np.float64]', r2: 'npt.NDArray[np.float64]', "
        "distance: 'npt.NDArray[np.int64]', "
        "absent: 'tuple[str, ...]', r2_min: 'float', window_bp: 'float', t_far: 'float') -> None",
    ),
    "ogquery.ld.Clumps": (
        ogquery.ld.Clumps,
        "(alid: 'npt.NDArray[np.str_]', chrom: 'npt.NDArray[np.str_]', "
        "position: 'npt.NDArray[np.int64]', score: 'npt.NDArray[np.float64]', "
        "lead: 'npt.NDArray[np.str_]', r: 'npt.NDArray[np.float64]', absent: 'tuple[str, ...]', "
        "r2: 'float', window_bp: 'float') -> None",
    ),
    "ogquery.ld.Clumps.leads": (ogquery.ld.Clumps.leads, "(self) -> 'npt.NDArray[np.str_]'"),
    "ogquery.ld.Variants": (
        ogquery.ld.Variants,
        "(chrom: 'npt.NDArray[np.str_]', position: 'npt.NDArray[np.int64]', "
        "alid: 'npt.NDArray[np.str_]', a1: 'npt.NDArray[np.str_]', a2: 'npt.NDArray[np.str_]', "
        "a1_frequency: 'npt.NDArray[np.float64]') -> None",
    ),
    "ogquery.ld.Refusal": (
        ogquery.ld.Refusal,
        "(reason: 'RefusalReason', detail: 'str') -> 'None'",
    ),
    "ogquery.identity.normalise_chromosome": (
        identity.normalise_chromosome,
        "(chromosome: 'str') -> 'str'",
    ),
    "ogquery.alid.sorted_alid_locus": (
        alid.sorted_alid_locus,
        "(alid: 'str') -> 'tuple[str, int] | None'",
    ),
}
@pytest.mark.parametrize("name", sorted(SURFACE))
def test_the_signature_is_the_recorded_one(name: str) -> None:
    function, signature = SURFACE[name]
    assert str(inspect.signature(function)) == signature


def test_the_top_level_names_are_the_public_surface() -> None:
    expected = (
        "AnswerKind Clumps InputError LDMatrix LDStore Provenance Proxies Reader Refusal "
        "RefusalReason Variants open"
    )
    assert " ".join(sorted(ogquery.ld.__all__)) == expected
    for name in ogquery.ld.__all__:
        assert hasattr(ogquery.ld, name), name


def test_no_body_in_ogquery_ld_is_a_stub() -> None:
    # Epic opengwas/ogld#1 built every public body (opengwas/ogld#3–opengwas/ogld#6,
    # opengwas/ogld#12–opengwas/ogld#17); a stub raising NotImplementedError(…) would be a call that
    # looks wired and is not. An abstract method's bare `raise NotImplementedError` is not one.
    package = Path(ogquery.ld.__file__).parent
    stubs = [
        f"{path.relative_to(package)}:{n}"
        for path in sorted(package.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "raise NotImplementedError(" in line
    ]
    assert stubs == [], f"stubs in ogquery/ld/: {stubs}"


def test_ogquery_ld_imports_and_reads_the_test_store_without_pgenlib() -> None:
    # pgenlib is the builder's pgen reader. A reader-only install lacks it, so nothing here may
    # import it: the committed Test Store is opened and read with pgenlib unimportable, and any
    # error fails the test.
    code = (
        "import sys\n"
        "sys.modules['pgenlib'] = None\n"
        "import ogquery.ld\n"
        f"store = ogquery.ld.open({str(TEST_STORE)!r})\n"
        "store.region('22', 30_000_000, 100_000)\n"
        "print('read', store.provenance['n_variants'])\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"read \d+", done.stdout.strip()), done.stdout
