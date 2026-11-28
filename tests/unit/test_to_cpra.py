"""`to_cpra`: one variant's CPRA against the byte genome, or the status that says why it has
none (opengwas/ogquery#14; opengwasdb ADR 0108 §2-§5, #518's statuses).

Every expectation is written by hand from the fixture sequence below.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from ogquery.cpra import ContigNotHeld, CpraResult, CpraStatus, ShortReferenceRead, to_cpra
from ogquery.genome import ByteGenome, write_byte_genome
from ogquery.identity import InvalidAllele, MissingChromosome, VariantNormalisationError

#: Contig 1, by 1-based position (as in test_cpra_left_align.py):
#:   1 G  2 A  3 T  4 C  5 A  6 A  7 A  8 A  9 G 10 T 11 C 12 A 13 C 14 A
#:  15 C 16 A 17 G 18 G 19 T 20 T 21 A 22 C 23 G 24 T 25 A 26 C 27 G 28 T
CONTIG_1 = "GATCAAAAGTCACACAGGTTACGTACGT"
#: Contig 3: `CC` at 3-4, a gap of `N` at 7-10.
#:   1 T  2 T  3 C  4 C  5 G  6 A  7 N  8 N  9 N 10 N 11 A 12 T
CONTIG_3 = "TTCCGANNNNAT"


@pytest.fixture(scope="module")
def genome(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ByteGenome]:
    path = write_byte_genome(
        tmp_path_factory.mktemp("genome") / "t.bytes",
        [("1", [CONTIG_1.lower()]), ("3", [CONTIG_3])],
        assembly="test",
        provenance={"source": "test_to_cpra"},
    )
    with ByteGenome.open(path) as held:
        yield held


def _ok(cpra: str, a1_is_alt: bool) -> CpraResult:
    chrom, pos, ref, alt = cpra.split(":")
    return CpraResult(CpraStatus.OK, cpra, chrom, int(pos), ref, alt, a1_is_alt)


def _status(status: CpraStatus) -> CpraResult:
    return CpraResult(status, None, None, None, None, None, None)


# -- ok ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        # An SNV takes REF from the genome (G at 1:1), whichever allele the source names first.
        (("1", 1, "G", "A"), _ok("1:1:G:A", a1_is_alt=False)),
        (("1", 1, "A", "G"), _ok("1:1:G:A", a1_is_alt=True)),
        # Spelt as an identity is: a `chr` prefix and lower-case bases do not matter.
        (("chr1", 1, "a", "g"), _ok("1:1:G:A", a1_is_alt=True)),
        # A palindromic SNV whose alleles include the genome base (T at 3:1).
        (("3", 1, "A", "T"), _ok("3:1:T:A", a1_is_alt=True)),
        (("3", 1, "T", "A"), _ok("3:1:T:A", a1_is_alt=False)),
        # An equal-length substitution is trimmed first (ruling 9): `CC/TC` is `C/T`.
        (("3", 3, "CC", "TC"), _ok("3:3:C:T", a1_is_alt=False)),
        (("3", 3, "TC", "CC"), _ok("3:3:C:T", a1_is_alt=True)),
        # ... trailing, then leading bases: `GTA/GCA` at 1:9 is `T/C` at 1:10 (T).
        (("1", 9, "GTA", "GCA"), _ok("1:10:T:C", a1_is_alt=False)),
        # A multi-base substitution that stays one: REF is the genome's span (CC at 3:3).
        (("3", 3, "GT", "CC"), _ok("3:3:CC:GT", a1_is_alt=True)),
        # The base at a contig's last position.
        (("1", 28, "T", "C"), _ok("1:28:T:C", a1_is_alt=False)),
    ],
)
def test_an_snv_or_substitution_takes_ref_from_the_genome(
    genome: ByteGenome, given: tuple[str, int, str, str], expected: CpraResult
) -> None:
    assert to_cpra(*given, sequence=genome) == expected


@pytest.mark.parametrize(
    ("given", "declared", "expected"),
    [
        # A declared SNV REF the genome carries.
        (("1", 1, "A", "G"), "a2", _ok("1:1:G:A", a1_is_alt=True)),
        (("1", 1, "G", "A"), "a1", _ok("1:1:G:A", a1_is_alt=False)),
        (("3", 3, "TC", "CC"), "a2", _ok("3:3:C:T", a1_is_alt=True)),
        # A declared substitution is checked whole, then trimmed: `CA/TA` at 1:4 (CA).
        (("1", 4, "TA", "CA"), "a2", _ok("1:4:C:T", a1_is_alt=True)),
        # A declared indel is trimmed and left-shifted: `AA/A` at 1:7 is `CA/C` at 1:4.
        (("1", 7, "AA", "A"), "a1", _ok("1:4:CA:C", a1_is_alt=False)),
        # An insertion whose effect allele is ALT: `A/AA` at 1:8 is `C/CA` at 1:4.
        (("1", 8, "AA", "A"), "a2", _ok("1:4:C:CA", a1_is_alt=True)),
        # A whole `CA` unit of `(CA)x3`.
        (("1", 16, "ACA", "A"), "a2", _ok("1:10:T:TCA", a1_is_alt=True)),
    ],
)
def test_a_declared_reference_is_checked_and_normalised(
    genome: ByteGenome,
    given: tuple[str, int, str, str],
    declared: str,
    expected: CpraResult,
) -> None:
    assert to_cpra(*given, sequence=genome, declared_ref=declared) == expected  # type: ignore[arg-type]


# -- one test per status -------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "declared"),
    [
        (("1", 1, "A", "C"), "a1"),  # an SNV declaring A at a G
        (("3", 3, "TC", "CC"), "a1"),  # a substitution, trimmed: T declared at a C
        (("1", 4, "GA", "G"), "a1"),  # an indel declaring G at a C
        # A declared substitution whose trimmed REF the genome carries, but not the whole of
        # it: `CC` at 1:4, where the genome reads `CA` (bcftools: REF_MISMATCH).
        (("1", 4, "CC", "TC"), "a1"),
        (("1", 4, "TC", "CC"), "a2"),
        (("1", 4, "C", "CAG"), "a2"),  # a deletion declaring CAG where the genome has CAA
    ],
)
def test_ref_mismatch(genome: ByteGenome, given: tuple[str, int, str, str], declared: str) -> None:
    result = to_cpra(*given, sequence=genome, declared_ref=declared)  # type: ignore[arg-type]
    assert result == _status(CpraStatus.REF_MISMATCH)


@pytest.mark.parametrize(
    "given",
    [
        ("1", 1, "A", "T"),  # a palindromic pair at a G: neither allele is the genome's
        ("1", 1, "C", "T"),  # C's complement is G, the genome base: never complemented
        ("3", 3, "GC", "TC"),  # a substitution, trimmed G/T at a C
        ("3", 3, "GT", "AA"),  # a multi-base substitution at CC
    ],
)
def test_allele_not_reference(genome: ByteGenome, given: tuple[str, int, str, str]) -> None:
    assert to_cpra(*given, sequence=genome) == _status(CpraStatus.ALLELE_NOT_REFERENCE)


@pytest.mark.parametrize(
    "given",
    [
        ("1", 7, "AA", "A"),  # a deletion's reading fits the genome, an insertion's does too
        ("1", 4, "CAG", "C"),  # one the genome could not orient either: never tried
        ("1", 900, "AT", "A"),  # not even read: undeclared is settled before the genome
    ],
)
def test_indel_unresolved(genome: ByteGenome, given: tuple[str, int, str, str]) -> None:
    assert to_cpra(*given, sequence=genome) == _status(CpraStatus.INDEL_UNRESOLVED)


@pytest.mark.parametrize(
    ("given", "declared"),
    [
        (("3", 7, "A", "C"), None),  # an SNV at an N
        (("3", 7, "A", "C"), "a1"),
        (("3", 6, "AC", "GT"), None),  # a substitution whose span reaches the gap
        (("1", 29, "A", "C"), None),  # past contig 1's 28 bases
        (("1", 0, "A", "C"), None),  # positions are 1-based
        # ... judged before a trim could move the variant onto the contig (`G` at 1:1).
        (("1", 0, "AG", "AA"), None),
        (("1", 0, "AG", "AA"), "a1"),
        (("1", -3, "AAAAG", "AAAAA"), None),
        (("1", 28, "TA", "GC"), None),  # a substitution whose span leaves the contig
        (("1", 28, "TA", "T"), "a1"),  # a declared deletion that leaves the contig
        (("3", 10, "AA", "A"), "a1"),  # a declared indel whose REF span holds an N
        (("3", 11, "AA", "A"), "a2"),  # an insertion whose left shift reaches the gap
    ],
)
def test_off_sequence(
    genome: ByteGenome, given: tuple[str, int, str, str], declared: str | None
) -> None:
    result = to_cpra(*given, sequence=genome, declared_ref=declared)  # type: ignore[arg-type]
    assert result == _status(CpraStatus.OFF_SEQUENCE)


# -- malformed input is the caller's bug, and raises ---------------------------------------


@pytest.mark.parametrize(
    ("given", "error"),
    [
        (("1", 1, "", "A"), InvalidAllele),
        (("1", 1, "A", "AN"), InvalidAllele),
        (("1", 1, "G", "-"), InvalidAllele),
        (("1", 1, "G", "G"), VariantNormalisationError),
        (("1", 1, "g", "G"), VariantNormalisationError),
        (("", 1, "G", "A"), MissingChromosome),
        (("22", 1, "G", "A"), ContigNotHeld),
    ],
)
def test_malformed_input_raises(
    genome: ByteGenome, given: tuple[str, int, str, str], error: type[Exception]
) -> None:
    with pytest.raises(error):
        to_cpra(*given, sequence=genome)


class _Short:
    """A broken `ReferenceSequence`: it answers with fewer bases than it was asked for."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self._answer

    def digest(self) -> str:
        raise NotImplementedError


@pytest.mark.parametrize(
    ("answer", "given", "declared"),
    [
        ("", ("1", 1, "G", "A"), None),
        ("", ("1", 1, "G", "A"), "a1"),
        ("G", ("1", 1, "GC", "AT"), None),
        ("G", ("1", 1, "GA", "G"), "a1"),
        ("GAT", ("1", 1, "GA", "TT"), "a1"),
    ],
)
def test_a_sequence_that_answers_short_or_long_raises(
    answer: str, given: tuple[str, int, str, str], declared: str | None
) -> None:
    with pytest.raises(ShortReferenceRead, match="asked"):
        to_cpra(*given, sequence=_Short(answer), declared_ref=declared)  # type: ignore[arg-type]


def test_a_declaration_names_a1_or_a2(genome: ByteGenome) -> None:
    with pytest.raises(ValueError, match="declared_ref"):
        to_cpra("1", 1, "G", "A", sequence=genome, declared_ref="G")  # type: ignore[arg-type]


def test_the_statuses_are_exactly_518s() -> None:
    assert {s.value for s in CpraStatus} == {
        "ok",
        "ref_mismatch",
        "allele_not_reference",
        "indel_unresolved",
        "off_sequence",
    }


def test_a_result_is_frozen(genome: ByteGenome) -> None:
    result = to_cpra("1", 1, "G", "A", sequence=genome)
    assert result.status == "ok"
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.cpra = "1:1:A:G"  # type: ignore[misc]
