"""Kind dispatch, immutable inputs and query refusals at ogquery.ld.open (opengwas/ogld#91).

The read parts of opengwasdb `964ac1b2`'s `tests/ld/test_genotype_refusals.py`
(opengwas/ogquery#11), on the committed Genotype Store. What needs a build -- validation of an
edited sidecar, a sidecar built and moved with its inputs -- stays there; a missing pgenlib is
`test_genotype_extra.py`'s, which names this package's extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from support import TEST_STORE

import ogquery.ld
from ogquery.ld.store import StoreReader

pytestmark = pytest.mark.genotype


def test_kind_dispatch_and_explicit_wrong_kind_refusals(genotype_store: Path) -> None:
    from ogquery.ld.genotypes.reader import GenotypeReader

    assert isinstance(ogquery.ld.open(genotype_store)._reader, GenotypeReader)
    assert type(ogquery.ld.open(TEST_STORE)._reader) is StoreReader
    with pytest.raises(ogquery.ld.InputError, match="LD Store"):
        StoreReader.open(genotype_store)
    with pytest.raises(ogquery.ld.InputError, match="Genotype Store"):
        GenotypeReader.open(TEST_STORE)


@pytest.mark.parametrize("fault", ["missing", "size", "mtime_ns"])
def test_changed_or_missing_pgen_is_named_on_open(
    genotype_store: Path, tmp_path: Path, fault: str
) -> None:
    import genotype_answers

    copy = genotype_answers.place(tmp_path)
    genotype_answers.edit_pin(copy, fault)
    with pytest.raises(ogquery.ld.InputError, match=rf"pgen.*{fault}"):
        ogquery.ld.open(copy)


@pytest.mark.parametrize("query", ["clump", "proxies"])
@pytest.mark.parametrize(
    ("r2", "kb"),
    [
        (-1, 10),
        (float("nan"), 10),
        (0.1, -1),
        (0.1, float("inf")),
        (0.1, 10001),
        (0.001, 3000),
        (0.01, 5000),
    ],
)
def test_threshold_refusals_equal_the_test_store(
    genotype_store: Path, query: str, r2: float, kb: float
) -> None:
    reasons = []
    for path in (TEST_STORE, genotype_store):
        store = ogquery.ld.open(path)
        call = getattr(store, query)
        args = (["absent"], [1]) if query == "clump" else (["absent"],)
        with pytest.raises(ogquery.ld.Refusal) as refused:
            call(*args, r2=r2, kb=kb)
        reasons.append(refused.value.reason)
    assert reasons[0] == reasons[1]


@pytest.mark.parametrize("radius", [-1, 5000001])
def test_region_refusal_equals_the_test_store(genotype_store: Path, radius: int) -> None:
    reasons = []
    for path in (TEST_STORE, genotype_store):
        with pytest.raises(ogquery.ld.Refusal) as refused:
            ogquery.ld.open(path).region("22", 25000000, radius)
        reasons.append(refused.value.reason)
    assert reasons[0] == reasons[1]


@pytest.mark.parametrize("score", [[-1], [np.inf], [np.nan], [1, 2]])
def test_invalid_scores_equal_the_test_store(genotype_store: Path, score: list[float]) -> None:
    for path in (TEST_STORE, genotype_store):
        with pytest.raises(ogquery.ld.Refusal, match="score"):
            ogquery.ld.open(path).clump(["absent"], score, r2=0.1, kb=3000)
