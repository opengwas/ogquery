"""A variant's CPRA: its VCF normal form against the release assembly (opengwasdb ADR 0108).

ogquery is CPRA formation's one owner (opengwasdb ADR 0108 §7, opengwas/ogquery#14): the
engine's builders, completion, the LD import and the alignment tool (opengwasdb #518)
all form a CPRA here, so a CPRA a tool prints is the CPRA a Store holds.

`trim_alleles` drops the bases two alleles share at their end, then at their start:
VCF's parsimony, which makes the substitution `CC`/`TC` the SNV `C`/`T` and needs no
sequence. `normalise_indel` trims an indel's shared bases and shifts it left while the
base before it equals its moved sequence's last base, keeping VCF's padding base: what
`bcftools norm -f` does, by the published algorithm (Tan, Abecasis and Kang 2015,
"Unified representation of genetic variants"). An unanchored delins, whose alleles
share no first base once trimmed, is trimmed and not shifted. At a contig's first
base nothing lies to the left, so the first base stays the anchor.

`to_cpra` forms one variant's CPRA, `chrom:pos:REF:ALT`, from its chromosome, position
and two alleles, by ADR 0108's rules: REF is the genome's (§3), an equal-length
substitution is trimmed first (§2), an indel is left-shifted from its declared reference
(§2), and a row it cannot spell gets a `CpraStatus` and no CPRA: it never guesses, and
never complements (§4, §5).

The sequence comes through `ReferenceSequence`, which `ogquery.genome.ByteGenome`
implements over the memory-mapped byte genome, and opengwasdb's `FastaSequence` over an
indexed FASTA. The protocol, the normal form and its errors moved here unchanged from
opengwasdb `0ba413e7` (`opengwasdb/variants/left_align.py`, #479).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from ogquery.identity import (
    VALID_BASES,
    InvalidAllele,
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

#: How many bases a left shift reads at a time: one fetch covers every repeat the
#: #479 measurement met (none moved an indel further than 1,999 bases), in a few reads.
_LEFT_WINDOW = 256


class NotAnIndel(ValueError):
    """Alleles of one length: an SNV or a substitution, which no reference moves."""


class ReferenceAlleleMismatch(ValueError):
    """An indel's declared reference is not what the sequence carries at its position.

    ADR 0108 §3: such an indel is dropped and counted (`indel_reference_mismatch`),
    never kept unnormalised and never respelt to fit."""


class ContigNotHeld(ValueError):
    """The sequence holds no primary contig of that name."""


class PositionOutsideContig(ValueError):
    """A span that does not lie wholly inside its contig."""


class UnreadableReferenceBase(ValueError):
    """A shift reached a base that is not A, C, G or T (an `N`), which no ALID can spell."""


class ShortReferenceRead(ValueError):
    """A `ReferenceSequence` answered a span with a different number of bases than it was
    asked for: the sequence is broken, and nothing it says can be judged."""


class ReferenceSequence(Protocol):
    """The release assembly's sequence, as `normalise_indel` reads it."""

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases ``[start, end)`` (0-based, half-open) of ``chrom``, upper-cased.

        ``chrom`` is an ALID's chromosome label (spec §4). Raises `ContigNotHeld` for a
        contig the sequence does not hold and `PositionOutsideContig` for a span that
        leaves it."""
        ...

    def digest(self) -> str:
        """The sequence's identity, as `FastaSequence.digest` defines it (ADR 0108 §9)."""
        ...


def trim_alleles(pos: int, ref: str, alt: str, *, keep: int = 1) -> tuple[int, str, str]:
    """``(pos, ref, alt)`` without the bases the alleles share at their end, then at their
    start, each allele keeping at least ``keep`` bases.

    With ``keep=1``, VCF's parsimony: `1:100:CC:TC` is the SNV `1:100:C:T` and
    `1:100:GCA:GTA` is `1:101:C:T`; an anchored indel keeps its padding base. With
    ``keep=0`` the alleles are stripped to the bases that differ, an indel's shorter
    allele to nothing, and ``pos`` is the first of them. No sequence is read, so an indel
    is trimmed but not shifted: that is `normalise_indel`.
    """
    while len(ref) > keep and len(alt) > keep and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > keep and len(alt) > keep and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


def normalise_indel(
    chrom: str, pos: int, ref: str, alt: str, *, sequence: ReferenceSequence
) -> tuple[int, str, str]:
    """The indel ``chrom:pos:ref:alt`` in VCF normal form: ``(pos, ref, alt)``.

    ``ref`` is the declared reference allele and must be what ``sequence`` carries at
    ``pos`` (`ReferenceAlleleMismatch` otherwise); ``ref`` and ``alt`` are spelt as an
    identity is (upper-case A/C/G/T, `identity_allele`). Shared trailing bases are
    dropped, and the indel shifts left while its last bases agree, taking the base
    before it as the new padding base; then shared leading bases are dropped down to
    one. ``ref`` stays the reference allele throughout.
    """
    _require_indel(ref, alt)
    if pos < 1:
        raise PositionOutsideContig(f"{chrom}:{pos}:{ref}:{alt}: a position is 1-based")
    carried = sequence.fetch(chrom, pos - 1, pos - 1 + len(ref))
    if carried != ref:
        raise ReferenceAlleleMismatch(
            f"{chrom}:{pos}:{ref}:{alt}: the reference sequence carries {carried!r} there, "
            f"not the declared reference {ref!r} (ADR 0108 §3)"
        )
    pos, ref, alt = _shift_left(_LeftBases(sequence, chrom), pos, ref, alt)
    return trim_alleles(pos, ref, alt)


def _require_indel(ref: str, alt: str) -> None:
    for allele in (ref, alt):
        if not allele or not set(allele) <= VALID_BASES:
            raise InvalidAllele(f"unsupported allele {allele!r}: an indel's alleles are A/C/G/T")
    if len(ref) == len(alt):
        raise NotAnIndel(f"{ref}/{alt} are one length: not an indel")


def _shift_left(left: _LeftBases, pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Drop shared trailing bases, padding from the left whenever an allele empties."""
    while ref[-1] == alt[-1] and (min(len(ref), len(alt)) > 1 or pos > 1):
        ref, alt = ref[:-1], alt[:-1]
        if not ref or not alt:
            pos -= 1
            base = left.base(pos)
            ref, alt = base + ref, base + alt
    return pos, ref, alt


class _LeftBases:
    """The bases to an indel's left, read a window at a time as a shift needs them."""

    def __init__(self, sequence: ReferenceSequence, chrom: str) -> None:
        self._sequence = sequence
        self._chrom = chrom
        self._start = 0  # 1-based position of self._bases[0]
        self._bases = ""

    def base(self, pos: int) -> str:
        if not self._start <= pos < self._start + len(self._bases):
            start = max(1, pos - _LEFT_WINDOW + 1)
            self._bases = self._sequence.fetch(self._chrom, start - 1, pos)
            self._start = start
        base = self._bases[pos - self._start]
        if base not in VALID_BASES:
            raise UnreadableReferenceBase(
                f"{self._chrom}:{pos}: a left shift reached {base!r}, which no ALID can spell"
            )
        return base


class CpraStatus(StrEnum):
    """Whether `to_cpra` formed a CPRA, and if not, why (opengwasdb #518's statuses)."""

    #: Formed: the CPRA's REF is the genome's.
    OK = "ok"
    #: The declared reference is not what the genome carries (ADR 0108 §3).
    REF_MISMATCH = "ref_mismatch"
    #: Neither allele is the genome's, so the row is dropped, never complemented (§4).
    ALLELE_NOT_REFERENCE = "allele_not_reference"
    #: An indel whose source declares no reference, which stays dropped (§5, ruling 3).
    INDEL_UNRESOLVED = "indel_unresolved"
    #: The span holds a base that is not A, C, G or T (a gap, `N`), or leaves its contig.
    OFF_SEQUENCE = "off_sequence"


@dataclass(frozen=True, slots=True)
class CpraResult:
    """One variant's CPRA, or the status saying why it has none.

    On `CpraStatus.OK` every field is set: ``cpra`` is ``chrom:pos:REF:ALT``, and
    ``a1_is_alt`` says whether the caller's ``a1`` (its effect allele, where it has one;
    #518's ``effect_is_alt``) is ALT, so its statistics are signed for ALT as they stand.
    On any other status every field but ``status`` is None.
    """

    status: CpraStatus
    cpra: str | None
    chrom: str | None
    pos: int | None
    ref: str | None
    alt: str | None
    a1_is_alt: bool | None


def to_cpra(
    chrom: str,
    pos: int,
    a1: str,
    a2: str,
    *,
    sequence: ReferenceSequence,
    declared_ref: Literal["a1", "a2"] | None = None,
) -> CpraResult:
    """The CPRA of the variant ``chrom:pos`` with alleles ``a1`` and ``a2`` on ``sequence``.

    ``pos`` is 1-based on the sequence's assembly (GRCh38). ``declared_ref`` names the
    allele the source declares as the reference (a GWAS-VCF's REF, a harmonised
    GWAS-SSF's), or is None where it declares none. The rules are opengwasdb ADR 0108's:

    - Alleles are spelt as an identity is (`normalise_allele`) and the chromosome by its
      canonical label (`normalise_chromosome`). An empty allele, a base other than A, C, G
      or T, or two alleles spelt alike raise the `ogquery.identity` errors: they are the
      caller's bug, not a status. So does a contig ``sequence`` does not hold
      (`ContigNotHeld`).
    - A position before 1 is `OFF_SEQUENCE`, judged before any trim could move the
      variant onto the contig. The genome's span is read at the position: one holding a
      base other than A, C, G or T, or leaving the contig, is `OFF_SEQUENCE`. A sequence
      that answers with more or fewer bases than asked is broken, and raises
      `ShortReferenceRead`.
    - **Alleles of one length, with a declared reference:** the whole declared allele is
      compared with the genome's span, `REF_MISMATCH` where they differ (§3: nothing is
      kept unchecked), and then the pair is trimmed (`trim_alleles`, §2).
    - **Alleles of one length, without one:** the pair is trimmed first (§2, ruling 9), the
      span at the trimmed position read, and the allele equal to it is REF. Where neither
      is, the row is `ALLELE_NOT_REFERENCE`; a complement is never tried (§4, ruling 8).
    - **An indel** without a declared reference is `INDEL_UNRESOLVED` (§5, ruling 3),
      wherever it lies: the genome is not asked to orient it. With one, the declared span
      is read as above for `OFF_SEQUENCE`, and `normalise_indel` checks the declared
      reference (`REF_MISMATCH`), trims it and shifts it left; a shift that reaches a gap
      is `OFF_SEQUENCE`.
    """
    chrom = normalise_chromosome(chrom)
    a1, a2 = normalise_allele(a1), normalise_allele(a2)
    if a1 == a2:
        raise VariantNormalisationError("effect and other alleles are identical")
    if declared_ref not in (None, "a1", "a2"):
        raise ValueError(f"declared_ref is {declared_ref!r}: name 'a1', 'a2' or None")
    if len(a1) != len(a2) and declared_ref is None:
        return _NO_CPRA[CpraStatus.INDEL_UNRESOLVED]
    if pos < 1:
        return _NO_CPRA[CpraStatus.OFF_SEQUENCE]
    if declared_ref is None:
        start, first, second = trim_alleles(pos, a1, a2)
        carried = _span(sequence, chrom, start, len(first))
        if carried is None:
            return _NO_CPRA[CpraStatus.OFF_SEQUENCE]
        if carried == first:
            return _formed(chrom, start, first, second, a1_is_alt=False)
        if carried == second:
            return _formed(chrom, start, second, first, a1_is_alt=True)
        return _NO_CPRA[CpraStatus.ALLELE_NOT_REFERENCE]
    a1_is_alt = declared_ref == "a2"
    ref, alt = (a2, a1) if a1_is_alt else (a1, a2)
    carried = _span(sequence, chrom, pos, len(ref))
    if carried is None:
        return _NO_CPRA[CpraStatus.OFF_SEQUENCE]
    if len(ref) == len(alt):
        if carried != ref:
            return _NO_CPRA[CpraStatus.REF_MISMATCH]
        pos, ref, alt = trim_alleles(pos, ref, alt)
    else:
        try:
            pos, ref, alt = normalise_indel(chrom, pos, ref, alt, sequence=sequence)
        except ReferenceAlleleMismatch:
            return _NO_CPRA[CpraStatus.REF_MISMATCH]
        except UnreadableReferenceBase:
            return _NO_CPRA[CpraStatus.OFF_SEQUENCE]
    return _formed(chrom, pos, ref, alt, a1_is_alt=a1_is_alt)


_NO_CPRA = {status: CpraResult(status, None, None, None, None, None, None) for status in CpraStatus}


def _formed(chrom: str, pos: int, ref: str, alt: str, *, a1_is_alt: bool) -> CpraResult:
    return CpraResult(CpraStatus.OK, f"{chrom}:{pos}:{ref}:{alt}", chrom, pos, ref, alt, a1_is_alt)


def _span(sequence: ReferenceSequence, chrom: str, pos: int, length: int) -> str | None:
    """The genome's bases at the 1-based ``pos`` (at least 1) for ``length`` bases, or None
    where the span leaves the contig or holds a base other than A, C, G or T. A sequence
    that answers with another number of bases raises `ShortReferenceRead`."""
    try:
        carried = sequence.fetch(chrom, pos - 1, pos - 1 + length)
    except PositionOutsideContig:
        return None
    if len(carried) != length:
        raise ShortReferenceRead(
            f"{chrom}:{pos}: the reference sequence answered {carried!r}, {len(carried)} "
            f"bases, when {length} were asked for; it is broken"
        )
    return carried if set(carried) <= VALID_BASES else None
