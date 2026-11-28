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

The sequence comes through `ReferenceSequence`, which `ogquery.genome.ByteGenome`
implements over the memory-mapped byte genome, and opengwasdb's `FastaSequence` over an
indexed FASTA. The protocol, the normal form and its errors moved here unchanged from
opengwasdb `0ba413e7` (`opengwasdb/variants/left_align.py`, #479).
"""

from __future__ import annotations

from typing import Protocol

from ogquery.identity import VALID_BASES, InvalidAllele

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
