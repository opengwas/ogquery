"""ogld#3's contract: the engine's ALID and A1, so r is signed as the engine signs a statistic.

An LD Store spells ALIDs sorted until #462 (`ogquery.alid.sorted_alid_locus`): ALID is
`chr:pos:A1:A2`, A1 the alphabetically first allele after upper-casing, an indel's too. These are
literals of that rule. Nothing is trimmed or left-aligned (ADR 0089).

From opengwasdb's `tests/ld/contract/test_identity.py` at `c8ebe4ca` (ogquery#2). The tests of a
record's sorted identity as a build forms it (`sorted_identity`) are build code's and stay there.
"""

from __future__ import annotations

import pytest

from ogquery.alid import sorted_alid_locus
from ogquery.identity import normalise_chromosome


@pytest.mark.parametrize(
    ("label", "canonical"),
    [
        ("22", "22"),
        ("chr22", "22"),
        ("X", "X"),
        ("chrX", "X"),
        ("23", "X"),
        ("M", "MT"),
        ("chrM", "MT"),
    ],
)
def test_chromosome_labels_are_the_engine_s(label: str, canonical: str) -> None:
    assert normalise_chromosome(label) == canonical


def test_an_unsupported_chromosome_is_refused() -> None:
    with pytest.raises(ValueError, match="chr"):
        normalise_chromosome("27")


@pytest.mark.parametrize(
    ("alid", "locus"),
    [
        ("22:16050075:A:G", ("22", 16050075)),
        ("X:100:A:ACGT", ("X", 100)),
        ("rs123", None),
        ("22:abc:A:G", None),
        ("", None),
    ],
)
def test_an_alid_names_its_locus(alid: str, locus: tuple[str, int] | None) -> None:
    assert sorted_alid_locus(alid) == locus


