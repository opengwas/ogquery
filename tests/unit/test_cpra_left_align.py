"""A variant's VCF normal form against a reference sequence (opengwasdb ADR 0108 §2).

Ported from opengwasdb `0ba413e7`'s `tests/test_left_align.py` with the code it pins
(opengwas/ogquery#14), its expectations unchanged. Every expectation below is written by
hand from the fixture sequence, and was checked once against `bcftools norm -f -c e`
(1.24) on the same FASTA there. opengwasdb read the fixture through its `FastaSequence`
(pysam), which ogquery may not import; here the same sequence is read through
`_Sequence`, a `ReferenceSequence` held in memory with `FastaSequence`'s answers. The
tests of `FastaSequence` itself (its index, its digest's wording) stay in opengwasdb.
"""

from __future__ import annotations

import pytest

from ogquery.cpra import (
    ContigNotHeld,
    NotAnIndel,
    PositionOutsideContig,
    ReferenceAlleleMismatch,
    ReferenceSequence,
    normalise_indel,
    trim_alleles,
)
from ogquery.identity import VariantNormalisationError, normalise_chromosome

#: Contig 1, by 1-based position:
#:   1 G  2 A  3 T  4 C  5 A  6 A  7 A  8 A  9 G 10 T 11 C 12 A 13 C 14 A
#:  15 C 16 A 17 G 18 G 19 T 20 T 21 A 22 C 23 G 24 T 25 A 26 C 27 G 28 T
#: `CAAAA` at 4-8, `(CA)x3` at 11-16 after the `T` at 10.
CONTIG_1 = "GATCAAAAGTCACACAGGTTACGTACGT"
#: Contig 2 starts inside a run of `A`, so no base lies left of the repeat.
CONTIG_2 = "AAACGTTGCA"


class _Sequence:
    """Contigs held in memory, answering as `FastaSequence` does: by label, upper-cased."""

    def __init__(self, contigs: dict[str, str]) -> None:
        self._contigs = contigs

    def fetch(self, chrom: str, start: int, end: int) -> str:
        try:
            contig = self._contigs.get(normalise_chromosome(chrom))
        except VariantNormalisationError:
            contig = None
        if contig is None:
            raise ContigNotHeld(f"no primary contig {chrom!r}")
        if not 0 <= start <= end <= len(contig):
            raise PositionOutsideContig(f"{chrom}:{start + 1}-{end}: outside contig {chrom}")
        return contig[start:end].upper()

    def digest(self) -> str:
        raise NotImplementedError


@pytest.fixture
def sequence() -> ReferenceSequence:
    # Soft-masked (lower case), as hg38 is: it may not change a result.
    held = _Sequence({"1": CONTIG_1.lower(), "2": CONTIG_2})
    assert held.fetch("1", 0, len(CONTIG_1)) == CONTIG_1
    assert held.fetch("2", 0, len(CONTIG_2)) == CONTIG_2
    return held


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        # The ADR's trim: shared trailing bases go, the anchor stays.
        (("1", 4, "CA", "CAA"), (4, "C", "CA")),
        # A deletion and an insertion in the `CAAAA` repeat, spelt at its right end.
        (("1", 7, "AA", "A"), (4, "CA", "C")),
        (("1", 8, "A", "AA"), (4, "C", "CA")),
        # A deletion and an insertion of a whole `CA` unit in `(CA)x3`.
        (("1", 14, "ACA", "A"), (10, "TCA", "T")),
        (("1", 16, "A", "ACA"), (10, "T", "TCA")),
        # A delins sharing its first base: trimmed to an unanchored delins, not shifted.
        (("1", 17, "GGTT", "GA"), (18, "GTT", "A")),
        (("1", 18, "GTT", "A"), (18, "GTT", "A")),
        # At a contig's start no base lies to the left: the first base stays the anchor.
        (("2", 2, "AA", "A"), (1, "AA", "A")),
        (("2", 3, "A", "AA"), (1, "A", "AA")),
        # Already in normal form.
        (("1", 4, "CA", "C"), (4, "CA", "C")),
    ],
)
def test_an_indel_is_trimmed_and_left_shifted(
    sequence: ReferenceSequence, given: tuple[str, int, str, str], expected: tuple[int, str, str]
) -> None:
    chrom, pos, ref, alt = given
    assert normalise_indel(chrom, pos, ref, alt, sequence=sequence) == expected


def test_the_declared_reference_must_be_the_sequences(sequence: ReferenceSequence) -> None:
    # Position 4 is `C`: an indel declaring `G` there is not this sequence's.
    with pytest.raises(ReferenceAlleleMismatch, match="1:4"):
        normalise_indel("1", 4, "GA", "G", sequence=sequence)


def test_alleles_of_one_length_are_not_an_indel(sequence: ReferenceSequence) -> None:
    with pytest.raises(NotAnIndel):
        normalise_indel("1", 4, "C", "T", sequence=sequence)
    with pytest.raises(NotAnIndel):
        normalise_indel("1", 4, "CA", "TA", sequence=sequence)


def test_a_contig_the_sequence_does_not_hold_is_refused(sequence: ReferenceSequence) -> None:
    with pytest.raises(ContigNotHeld, match="22"):
        sequence.fetch("22", 0, 1)
    with pytest.raises(ContigNotHeld):
        normalise_indel("22", 10, "AA", "A", sequence=sequence)


def test_a_position_past_a_contigs_end_is_refused(sequence: ReferenceSequence) -> None:
    assert sequence.fetch("1", 27, 28) == "T"
    with pytest.raises(PositionOutsideContig, match="28"):
        sequence.fetch("1", 27, 29)
    with pytest.raises(PositionOutsideContig):
        sequence.fetch("1", -1, 2)
    # A deletion whose span runs off the end of contig 2 (10 bases).
    with pytest.raises(PositionOutsideContig):
        normalise_indel("2", 10, "AC", "A", sequence=sequence)


@pytest.mark.parametrize(
    ("given", "keep", "expected"),
    [
        # #517's substitution normal form (ruled 28 Nov): trailing, then leading.
        ((100, "CC", "TC"), 1, (100, "C", "T")),
        ((100, "GCA", "GTA"), 1, (101, "C", "T")),
        ((100, "AC", "GT"), 1, (100, "AC", "GT")),  # no shared base: a substitution stays
        ((100, "A", "G"), 1, (100, "A", "G")),
        # An anchored indel keeps its padding base, and is not shifted (no sequence).
        ((100, "CA", "C"), 1, (100, "CA", "C")),
        ((100, "CAG", "CG"), 1, (100, "CA", "C")),
        # keep=0: stripped to the bases that differ, an indel's shorter allele to nothing.
        ((100, "CA", "C"), 0, (101, "A", "")),
        ((100, "AT", "T"), 0, (100, "A", "")),
        ((100, "CC", "TC"), 0, (100, "C", "T")),
    ],
)
def test_alleles_are_trimmed_trailing_bases_first(
    given: tuple[int, str, str], keep: int, expected: tuple[int, str, str]
) -> None:
    assert trim_alleles(*given, keep=keep) == expected
