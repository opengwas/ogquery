"""Asking a 0.4.0 release for a variant by its CPRA: what is refused, and what a miss says.

The query rules of spec §4.0, which `VariantAxis` applies to a release of the 0.4 series:

- a CPRA that is **not normal by shape** is refused, `NotNormalCpra`, naming the string
  asked and its form after `trim_alleles` (the one owner of that rule, `ogquery.cpra`);
  one that is not well-formed is refused as a malformed ALID is (`MalformedIdentity`,
  `UnsupportedChromosome`);
- an identifier the release lacks gives **no row**, and two misses are never bare: a
  `VariantMiss` says an SNV or substitution was asked in the swapped spelling of a row
  the release holds (`swapped_alleles`), or that an indel is absent, naming the form the
  release spells indels in (`indel_absent`).

A `VariantMiss` rides `Associations.misses` and is issued by `variants()` as a
`VariantMissWarning`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ogquery.cpra import trim_alleles
from ogquery.format import CpraNormalForm
from ogquery.identity import (
    MalformedIdentity,
    UnsupportedChromosome,
    VariantNormalisationError,
    is_primary_contig,
    normalise_allele,
    normalise_chromosome,
)

#: The reasons a `VariantMiss` gives (spec §4.0).
SWAPPED_ALLELES: Final = "swapped_alleles"
INDEL_ABSENT: Final = "indel_absent"
MISS_REASONS: Final = (SWAPPED_ALLELES, INDEL_ABSENT)


class NotNormalCpra(VariantNormalisationError):
    """A CPRA asked of a 0.4.0 release that `trim_alleles` would change: shared bases the
    normal form trims (spec §4.0, ADR 0108 §9). `identifier` is the string asked and
    `trimmed` its form after `trim_alleles`; no row is looked for under either."""

    def __init__(self, identifier: str, trimmed: str) -> None:
        self.identifier = identifier
        self.trimmed = trimmed
        super().__init__(
            f"{identifier!r} is not a normal CPRA: its alleles share bases that the normal "
            f"form trims, which would name {trimmed!r}. Align the identifier to CPRA first "
            "(spec §4.0)"
        )

    @property
    def fields(self) -> dict[str, object]:
        return {"identifier": self.identifier, "trimmed": self.trimmed}


@dataclass(frozen=True)
class VariantMiss:
    """Why an identifier named no row, where the reason is not plain absence (spec §4.0).

    `identifier` is as asked; `reason` is `swapped_alleles` or `indel_absent`. For
    `swapped_alleles`, `cpra` is the row's CPRA that the swapped spelling names and
    `variant_index` its index; for `indel_absent` both are None. `cpra_normal_form` is the
    release's."""

    identifier: str
    release_id: str
    reason: str
    cpra: str | None
    variant_index: int | None
    cpra_normal_form: CpraNormalForm

    def __post_init__(self) -> None:
        if self.reason not in MISS_REASONS:
            raise ValueError(f"a VariantMiss reason is one of {MISS_REASONS}, not {self.reason!r}")

    def describe(self) -> str:
        """One line naming the six fields, as the CLI prints it (spec §4.0)."""
        return (
            f"identifier={self.identifier} release_id={self.release_id} reason={self.reason} "
            f"cpra={self.cpra} variant_index={self.variant_index} "
            f"cpra_normal_form=({self.cpra_normal_form.describe()})"
        )


class VariantMissWarning(UserWarning):
    """`variants()` issues each `VariantMiss` as one; the record is `miss`."""

    def __init__(self, miss: VariantMiss) -> None:
        self.miss = miss
        super().__init__(f"variant not found: {miss.describe()}")


@dataclass(frozen=True)
class ParsedCpra:
    chromosome: str
    position: int
    ref: str
    alt: str

    @property
    def cpra(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.ref}:{self.alt}"

    @property
    def is_indel(self) -> bool:
        return len(self.ref) != len(self.alt)

    def swapped(self) -> str:
        """The spelling with REF and ALT exchanged."""
        return f"{self.chromosome}:{self.position}:{self.alt}:{self.ref}"


#: An rsID, recognised positively: `rs` and decimal digits (dbSNP's). Anything else asked of a
#: 0.4.0 release is a CPRA, and refused as one if it is not well-formed (`parse_cpra`).
RSID: Final = re.compile(r"rs[0-9]+")


def is_rsid(identifier: str) -> bool:
    """Whether `identifier` is an rsID: `rs` and digits, nothing else."""
    return RSID.fullmatch(str(identifier)) is not None


def parse_cpra(identifier: str) -> ParsedCpra:
    """`identifier` as a normal CPRA (spec §4.0), or the refusal that names why.

    Well-formed: a chromosome of the 25 primary contigs (`chr1` is `1`), a position of
    decimal digits, at least 1, with no sign and no leading zero, and two A/C/G/T alleles
    that differ (upper-cased as an ALID's are). `MalformedIdentity` and
    `UnsupportedChromosome` otherwise. Then normal by shape: `trim_alleles` leaves it
    unchanged, or `NotNormalCpra`.
    """
    text = str(identifier)
    parts = text.split(":")
    if len(parts) != 4:
        raise MalformedIdentity(f"{text!r} is not a CPRA, chrom:pos:REF:ALT")
    chromosome_text, position_text, ref_text, alt_text = parts
    chromosome = normalise_chromosome(chromosome_text)
    if not is_primary_contig(chromosome):
        # A CPRA exists only on the 25 primary contigs (spec §4.0): `HLA-A` is no CPRA's.
        raise UnsupportedChromosome(chromosome_text)
    if not (position_text.isascii() and position_text.isdigit()) or position_text != str(
        int(position_text)
    ):
        raise MalformedIdentity(
            f"{text!r}: the position {position_text!r} is not a decimal number with no sign "
            "and no leading zero"
        )
    position = int(position_text)
    if position < 1:
        raise MalformedIdentity(f"{text!r}: the position must be at least 1")
    ref, alt = normalise_allele(ref_text), normalise_allele(alt_text)
    if ref == alt:
        raise MalformedIdentity(f"{text!r}: REF and ALT must differ")
    trimmed_position, trimmed_ref, trimmed_alt = trim_alleles(position, ref, alt)
    if (trimmed_position, trimmed_ref, trimmed_alt) != (position, ref, alt):
        raise NotNormalCpra(
            text, f"{chromosome}:{trimmed_position}:{trimmed_ref}:{trimmed_alt}"
        )
    return ParsedCpra(chromosome, position, ref, alt)

