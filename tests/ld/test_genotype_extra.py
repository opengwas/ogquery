"""Reading a Genotype Store needs the `genotype` extra, and a missing one says so
(opengwas/ogquery#11; ADR 0002, amended).

A Genotype Store's reader computes r from the pgens it pins, with pgenlib, and holds BLAS to one
thread a request with threadpoolctl. Neither is one of ogquery's three dependencies, so
`ogquery.ld.open` checks for both before it imports the reader, and raises `GenotypeExtraMissing`
naming `ogquery[genotype]`; an LD Store and a GWAS Store still open and answer. These tests block
the imports in a fresh interpreter, so they hold with the extra installed or not, on every
platform.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from blocked import without

from ogquery import opening

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
#: Opened in place: the open is refused before anything past the root's metadata is read.
GENOTYPE_STORE = HERE / "data" / "test-panel" / "genotype-store"
TEST_STORE = HERE / "data" / "test-panel" / "store"
RELEASE = ROOT / "tests" / "contract" / "releases" / "dense_ranges"
EXPECTED = ROOT / "tests" / "contract" / "expected" / "dense_ranges.json"


#: Without the extra: a Genotype Store's open, and an LD Store's reads.
OPEN_EACH_KIND = f"""
import numpy as np
import ogquery.ld
from ogquery.opening import GenotypeExtraMissing
try:
    ogquery.ld.open({str(GENOTYPE_STORE)!r})
    out = {{'raised': None}}
except Exception as error:
    out = {{'raised': type(error).__name__, 'message': str(error),
           'import_error': isinstance(error, ImportError),
           'missing_extra': isinstance(error, GenotypeExtraMissing)}}
store = ogquery.ld.open({str(TEST_STORE)!r})
rows = store.variants_in('22', 1, 3_000_000_000).alid.tolist()[:40]
out['ld'] = [list(store.matrix(rows).r.shape), len(store.clump(rows, np.arange(40.0), r2=0.1,
             kb=1000)), len(store.proxies(rows, r2=0.5, kb=1000)) > 0]
out['loaded'] = [n for n in ('pgenlib', 'threadpoolctl') if sys.modules.get(n) is not None]
print(json.dumps(out))
"""

#: Without the extra: a GWAS release's every readable answer, as the contract bundle expects.
ANSWER_A_RELEASE = f"""
from ogquery.digests import CROSS_MACHINE_RTOL, readable_differences, release_answers
expected = json.load(open({str(EXPECTED)!r}, encoding='utf-8'))['readable']
found = release_answers({str(RELEASE)!r})['readable']
out = {{'differences': readable_differences(expected, found, rtol=CROSS_MACHINE_RTOL),
       'answers': sum(len(s) for s in found.values() if isinstance(s, dict)),
       'loaded': [n for n in ('pgenlib', 'threadpoolctl') if sys.modules.get(n) is not None]}}
print(json.dumps(out))
"""


@pytest.mark.parametrize(
    "blocked",
    [("pgenlib", "threadpoolctl"), ("pgenlib",), ("threadpoolctl",)],
    ids=["both", "pgenlib", "threadpoolctl"],
)
def test_without_the_extra_a_genotype_store_names_it_and_an_ld_store_answers(
    blocked: tuple[str, ...],
) -> None:
    out = without(blocked, OPEN_EACH_KIND)
    assert out["raised"] == "GenotypeExtraMissing", out
    assert out["import_error"] and out["missing_extra"]
    message = str(out["message"])
    assert str(GENOTYPE_STORE) in message
    assert "pip install 'ogquery[genotype]'" in message
    named = message.split("not importable:")[1].strip().removesuffix(")").split(", ")
    # What it lacks: the modules blocked, and any the extra did not install here.
    absent = {name for name in opening.GENOTYPE_MODULES if importlib.util.find_spec(name) is None}
    assert sorted(named) == sorted({*blocked, *absent})
    assert out["ld"] == [[40, 40], 40, True]
    assert out["loaded"] == []


def test_without_the_extra_a_gwas_release_answers_as_the_contract_expects() -> None:
    out = without(opening.GENOTYPE_MODULES, ANSWER_A_RELEASE)
    assert out["differences"] == []
    assert isinstance(out["answers"], int) and out["answers"] > 0
    assert out["loaded"] == []


def test_the_check_finds_the_modules_and_imports_neither() -> None:
    statements = (
        "from ogquery import opening\n"
        "try:\n"
        "    opening.require_genotype('a Genotype Store')\n"
        "    raised = None\n"
        "except opening.GenotypeExtraMissing as error:\n"
        "    raised = str(error)\n"
        "out = {'raised': raised,\n"
        "       'loaded': [n for n in ('pgenlib', 'threadpoolctl') if n in sys.modules]}\n"
        "print(json.dumps(out))"
    )
    out = without((), statements)
    assert out["loaded"] == [], "require_genotype finds the modules; it never imports them"
    assert opening.GENOTYPE_MODULES == ("pgenlib", "threadpoolctl")
    assert opening.GENOTYPE_EXTRA == "ogquery[genotype]"
    assert issubclass(opening.GenotypeExtraMissing, ImportError)
