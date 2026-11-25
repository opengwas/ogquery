"""ALIDs and rsids as a reader is asked for them: parsing one, and whether it is indexable.

An ALID is ``chromosome:position:a1:a2``. A Store Release's search indexes hold
ALIDs and rsids in fixed-width byte slots (`ALID_WIDTH`, `RSID_WIDTH`); an
identifier wider than its slot is left out of the index rather than truncated
into it, and `is_indexable_alid` / `is_indexable_rsid` are the one statement of
that rule, for the writer that builds the indexes and the Variant Axis that
searches them.

An LD Store spells its ALIDs sorted, an indel's too: `sorted_alid_locus` parses
one as such a Store spells it, and `refuse_reference_first_indel` refuses an
indel spelt reference first rather than call it absent (ogquery#2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ogquery.identity import (
    CANONICAL_CHROMOSOMES,
    VALID_BASES,
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

# Fixed byte-width for ALID encoding in the mmap'd search index. Fixed, because
# that is what lets `np.searchsorted` run over the array as an mmap without
# parsing it. Measured across the FinnGen, GWAS Catalog and metabolome pilots,
# ALID length is mean 14.9 and p99 21; the slack is for indels.
#
# An ALID wider than this is *left out* of the index, never cast into it:
# `np.array(..., dtype="|S32")` truncates silently, and two indels at one
# position agreeing over the slot width then collapse to a single key, so a
# lookup by either full ALID returned whichever row sorted first (issue #127 --
# on the published finngen-r13 release, 342 variants answering to another's
# name). Excluded ALIDs stay reachable: `by_alid` resolves them by exact scan
# over the position instead, which is slower and right.
ALID_WIDTH = 32
ALID_DTYPE = f"|S{ALID_WIDTH}"

# Fixed byte-width for rsid encoding in the mmap'd rsid search index. An rsid is
# "rs" + digits, far inside this; the width exists so a malformed or non-rs
# identifier in a source's rsid column cannot silently truncate into a *false
# match* for a different variant. Anything wider is left out of the index (and
# counted in a warning) rather than stored truncated -- see `_write_rsid_index`.
RSID_WIDTH = 24
RSID_DTYPE = f"|S{RSID_WIDTH}"


def is_indexable_alid(alid: str) -> bool:
    """Whether `alid` fits the fixed-width ALID search index (issue #127).

    The one place the rule lives: `write_variant_axis` decides what to index by
    it, `VariantAxis.by_alid` decides whether a miss means "absent" or "ask the
    exact scan" by it, and `opengwasdb.validation.validate` checks the index
    holds no key shared by two variants -- which is what this rule prevents.
    """
    return len(alid.encode("utf-8")) <= ALID_WIDTH


def is_indexable_rsid(rsid: str | None) -> bool:
    """Whether `rsid` can be resolved by name in a store (issue #109).

    The one place the rule lives: `_write_rsid_index` decides what to index by
    it, `indices_by_alias` decides what is worth searching for by it, and
    `opengwasdb.validation.validate` counts what it should find by it. Three
    copies of a width check would be three chances for a correctly-built store
    to fail its own validator.
    """
    if not rsid or rsid == ".":
        return False
    return len(rsid.encode("utf-8")) <= RSID_WIDTH


@dataclass(frozen=True)
class ParsedAlid:
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str


def parse_canonical_alid(identifier: str) -> ParsedAlid | None:
    """Parse a canonical ALID-like identifier, returning None for aliases.

    The lenient parser, for the engine's reads of a GWAS Store, a variant list or a Variant
    Reference: the chromosome is normalised (`chr1`, `23`), the alleles are upper-cased and not
    required in either order, and the position has no width or int32 limit.
    `sorted_alid_locus` (below) is the strict one, for an LD Store's lookup, which must find
    nothing under a spelling the Store never wrote.
    """

    parts = str(identifier).split(":")
    if len(parts) != 4:
        return None
    chromosome, position_text, effect_allele, other_allele = parts
    try:
        position = int(position_text)
        if position <= 0:
            return None
        return ParsedAlid(
            chromosome=normalise_chromosome(chromosome),
            position=position,
            effect_allele=normalise_allele(effect_allele),
            other_allele=normalise_allele(other_allele),
        )
    except (TypeError, ValueError, VariantNormalisationError):
        return None

# ── The sorted spelling, as an LD Store spells an ALID ───────────────────────
#
# Every LD Store spells an ALID with both alleles sorted, an indel's too (opengwasdb#462;
# `ogquery.ld`). These are the read side of that spelling: the strict parser an LD lookup
# finds a locus by, and the refusal of an indel spelt reference first, which a sorted Store
# would hold under the swapped ALID. Their bodies are opengwasdb's
# `variants/sorted_spelling.py` at `c8ebe4ca` (ogquery#2); the build side -- a record's sorted
# identity, and which of a reversed pair a Store keeps -- stays there.


#: A position an LD Store's int32 column holds (ld-store spec §2), spelt with at most this many
#: digits.
_MAX_POSITION: Final = 2**31 - 1
_MAX_POSITION_DIGITS: Final = 10


class ReferenceFirstIndel(VariantNormalisationError):
    """An ALID that spells an indel reference first where ALIDs are spelt sorted.

    Its alleles differ in length and its A1 sorts after its A2: a deletion whose REF sorts after
    its ALT, which no sorted spelling forms. A sorted Store would hold that variant as
    `chr:pos:A2:A1`. That row is this deletion unless the panel also has the insertion at that
    site, and either way r there is signed against the other allele, A2 here (#462).
    """

    reason = "indel_spelt_reference_first"


def sorted_alid_locus(alid: str) -> tuple[str, int] | None:
    """The chromosome and position a sorted-spelling ALID names, or None when it is not one.

    The strict parser, for an LD Store's lookup: an ALID either is one exactly as such a Store
    spells it or names nothing there. `opengwasdb.variants.axis.parse_canonical_alid` is the
    lenient one, for the engine's GWAS Store and variant-list reads: it normalises a chromosome
    alias and an allele's case, takes the alleles in either order, and has no int32 or digit
    limit. Both exist because an LD lookup must find nothing under a spelling its Store never
    wrote, while the engine resolves an identifier as it is given.

    A reader finds an ALID by this locus, then compares whole ALID strings there, so a malformed
    one is Absent rather than an error, and this never raises. Such an ALID is
    `chrom:position:A1:A2`: a chromosome `1`–`22`, `X`, `Y` or `MT`, a position of at most 10
    ASCII digits, positive, unpadded and within int32, and two distinct alleles over A, C, G, T
    with A1 sorted before A2. A reference-first indel ALID is not one
    (`spells_indel_reference_first`).
    """
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos, a1, a2 = parts
    if not (chrom in CANONICAL_CHROMOSOMES and _is_canonical_position(pos)):
        return None
    if not (_is_allele(a1) and _is_allele(a2) and a1 < a2):
        return None
    return chrom, int(pos)


def spells_indel_reference_first(alid: str) -> bool:
    """Whether `alid` is a well-formed ALID that only the reference-first spelling forms: an
    indel whose A1 sorts after its A2, which `sorted_alid_locus` would accept with its alleles
    swapped."""
    parts = alid.split(":")
    if len(parts) != 4:
        return False
    chrom, pos, a1, a2 = parts
    if len(a1) == len(a2) or not a1 > a2:
        return False
    return sorted_alid_locus(f"{chrom}:{pos}:{a2}:{a1}") is not None


def refuse_reference_first_indel(alid: str, where: str) -> None:
    """Raise `ReferenceFirstIndel` when `alid` spells an indel reference first; `where` names
    the Store or reader that spells ALIDs sorted."""
    if spells_indel_reference_first(alid):
        chrom, pos, a1, a2 = alid.split(":")
        raise ReferenceFirstIndel(
            f"{alid} spells an indel reference first (ADR 0079), but {where} spells indels "
            f"sorted, so it would hold this variant as {chrom}:{pos}:{a2}:{a1}. That row is "
            "this deletion unless the panel also has the insertion at that site; either way r "
            f"there is signed against the other allele, {a2}, not {a1} (#462)"
        )


def _is_canonical_position(text: str) -> bool:
    if not (text.isascii() and text.isdigit()) or len(text) > _MAX_POSITION_DIGITS:
        return False
    return text == str(int(text)) and 0 < int(text) <= _MAX_POSITION


def _is_allele(text: str) -> bool:
    return bool(text) and VALID_BASES.issuperset(text)
