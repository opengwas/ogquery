"""The engine's variant order, as an LD build sorts its Variant Set by it
(`opengwasdb.variants.axis.alid_sort_key` over sorted-spelling ALIDs), and a chromosome's place in
it, `ogquery.identity.chromosome_rank` (ld-store §2).

From opengwasdb's `tests/ld/test_variant_order.py` at `c8ebe4ca` (ogquery#2). The tests that
sort by `alid_sort_key`, the build's order, stay there."""

from __future__ import annotations

import numpy as np
import pytest
from spelling import sorted_identity

from ogquery.identity import AUTOSOMES, VariantNormalisationError, chromosome_rank
from ogquery.ld.query.results import Clumps

RECORDS = [
    ("X", 5, "A", "C"),
    ("22", 9, "A", "G"),
    ("2", 90, "A", "G"),
    ("22", 9, "A", "C"),
    ("MT", 1, "A", "G"),
    ("2", 100, "C", "G"),
    ("Y", 5, "A", "C"),
    ("22", 9, "A", "AT"),
]
#: The engine's order of RECORDS, as (chromosome, position, ALT).
ENGINE_ORDER = [
    ("2", 90, "G"),
    ("2", 100, "G"),
    ("22", 9, "AT"),  # 22:9:A:AT before 22:9:A:C: `AT` sorts before `C`
    ("22", 9, "C"),  # 22:9:A:C before 22:9:A:G: ALID ascending at one position
    ("22", 9, "G"),
    ("X", 5, "C"),
    ("Y", 5, "C"),
    ("MT", 1, "G"),
]


def test_a_clumping_s_leads_take_the_order_of_their_fields_on_a_tie() -> None:
    # What a clumping's result holds, with no alleles to re-form an identity from (ogld#14).
    found = [sorted_identity(*r) for r in RECORDS]
    clumps = Clumps(
        alid=np.array([o.alid for o in found]),
        chrom=np.array([o.chrom for o in found]),
        position=np.array([o.position for o in found], dtype=np.int64),
        score=np.ones(len(found)),
        lead=np.array([o.alid for o in found]),
        r=np.ones(len(found)),
        absent=(),
        r2=0.1,
        window_bp=1e6,
    )
    by_alid = {o.alid: r for o, r in zip(found, RECORDS, strict=True)}
    assert [(by_alid[a][0], by_alid[a][1], by_alid[a][3]) for a in clumps.leads()] == ENGINE_ORDER


def test_a_chromosome_the_engine_does_not_rank_is_refused() -> None:
    for label in ("HLA-A", "chr 22", "-1"):
        with pytest.raises(VariantNormalisationError, match="no place in the variant order"):
            chromosome_rank(label)
    with pytest.raises(VariantNormalisationError, match="unsupported chromosome"):
        chromosome_rank("27")


def test_the_rank_reads_a_chr_prefix_as_the_engine_does() -> None:
    assert chromosome_rank("chr1") == chromosome_rank("1")
    assert chromosome_rank("chrX") > chromosome_rank("chr22")


def test_the_autosomes_are_1_to_22_in_the_engine_s_order() -> None:
    assert tuple(str(n) for n in range(1, 23)) == AUTOSOMES
    assert sorted(AUTOSOMES, key=chromosome_rank) == list(AUTOSOMES)
