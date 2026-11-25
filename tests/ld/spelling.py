"""The sorted spelling of a Test Panel record's ALID, as the tests form it (ogquery#2).

opengwasdb forms it with `opengwasdb.variants.sorted_spelling.sorted_identity`, which judges a
source record by the engine's whole normalisation; that is build code and stays there
(ADR 0002). The tests only need the spelling of records the Test Panel already holds, every one
biallelic over A, C, G and T: both alleles sorted, an indel's too, A1 first. `test_spelling.py`
holds this to the Test Store's own ALIDs, every row of them, so it cannot drift from the store.
"""

from __future__ import annotations

from dataclasses import dataclass

from ogquery.identity import normalise_allele, normalise_chromosome


@dataclass(frozen=True)
class SortedIdentity:
    """A record's identity in the sorted spelling; `a1_is_alt` says r for A1 is r for ALT."""

    chrom: str
    position: int
    alid: str
    a1: str
    a2: str
    a1_is_alt: bool


def sorted_identity(chrom: str, position: int, ref: str, alt: str) -> SortedIdentity:
    """The sorted-spelling identity of the record `ref`/`alt` at `chrom`:`position`."""
    label = normalise_chromosome(chrom)
    a1, a2 = sorted((normalise_allele(ref), normalise_allele(alt)))
    if a1 == a2 or position <= 0:
        raise ValueError(f"not a biallelic record: {chrom}:{position}:{ref}:{alt}")
    return SortedIdentity(label, position, f"{label}:{position}:{a1}:{a2}", a1, a2, alt == a1)
