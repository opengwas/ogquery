"""The sorted spelling an LD Store's ALIDs use, held to ogld's committed answers (ADR 0089).

`tests/ld/data/identity/` holds the engine's answers at opengwasdb `1ab2b09`, before ADR 0079
made indels reference first, when every ALID was spelt sorted (the commit is on each file's
first line; the README there says how they were made). They are the independent source: no
expected value here comes from `opengwasdb.variants`. LD Stores keep that spelling until #462
(owner, 23 November 2026, on #456), so `opengwasdb.variants.sorted_spelling` must give every
answer they hold, and the engine's reference-first rule must differ from them only where the
two spellings differ.

From opengwasdb's `tests/ld/test_sorted_spelling_conformance.py` at `c8ebe4ca` (ogquery#2): the
tests of what ogquery owns, the locus an ALID names and a chromosome's label. Those of a record's
sorted identity and of a reversed indel pair, which a build forms, stay there.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest

from ogquery.alid import sorted_alid_locus, spells_indel_reference_first
from ogquery.identity import VariantNormalisationError, normalise_chromosome

DATA = Path(__file__).resolve().parent / "data" / "identity"


def _rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        header = handle.readline()
        assert header.startswith("# opengwasdb 1ab2b09:"), "the fixture is pinned before ADR 0079"
        return list(csv.DictReader(handle, delimiter="\t"))


ORIENT = _rows("orient.tsv")
CHROMOSOMES = _rows("chromosomes.tsv")
REVERSED = _rows("reversed.tsv")
CANONICAL = {*map(str, range(1, 23)), "X", "Y", "MT"}


def _position(text: str) -> Any:
    return int(text) if text.lstrip("-").isdigit() else text


def _alleles(row: dict[str, str]) -> tuple[str, str]:
    return row["ref"].strip().upper(), row["alt"].strip().upper()


def test_the_fixture_is_meaningful() -> None:
    # Asserted before anything is asserted about it: both outcomes, both signs, long ALIDs, and
    # indels on both sides of the line between the spellings.
    refused = [r for r in ORIENT if r["error"]]
    kept = [r for r in ORIENT if not r["error"]]
    assert (len(kept), len(refused)) == (49, 44)
    assert {r["a1_is_alt"] for r in kept} == {"true", "false"}
    assert any(len(r["alid"]) > 32 for r in kept)
    assert any(r["ref"] != r["ref"].upper() for r in kept)
    longer_ref = [_alleles(r) for r in kept if len(r["ref"].strip()) > len(r["alt"].strip())]
    assert sum(ref > alt for ref, alt in longer_ref) == 10
    assert sum(ref < alt for ref, alt in longer_ref) == 2  # 22:403 GAT>GT, 22:404 AGT>AT
    assert {r["reason"] for r in ORIENT} == {"", "allele_invalid", "chromosome_missing"}
    # Rows with two faults, whose message names the one judged first.
    messages = {(r["chrom"], r["position"], r["ref"], r["alt"]): r["error"] for r in refused}
    assert messages[("22", "0", "N", "G")] == "unsupported allele 'N'"
    assert messages[("22", "100", "N", "X")] == "unsupported allele 'X'"
    assert messages[("27", "100", "N", "G")] == "unsupported chromosome '27'"
    assert {r["canonical"] for r in CHROMOSOMES if "+22" in r["label"] or "-1" in r["label"]}
    assert sum(1 for r in CHROMOSOMES if r["error"]) > 10
    assert {r["canonical"] for r in CHROMOSOMES} >= {"1", "22", "X", "Y", "MT", ""}
    assert len(REVERSED) == 8
    assert {r["kept"] for r in REVERSED} == {"true", "false"}


@pytest.mark.parametrize("row", CHROMOSOMES, ids=lambda r: repr(r["label"]))
def test_the_engine_s_chromosome_label_is_the_committed_one(row: dict[str, str]) -> None:
    if row["error"]:
        with pytest.raises(
            VariantNormalisationError, match=rf"\A{re.escape(row['error'])}\Z"
        ) as fault:
            normalise_chromosome(row["label"])
        assert getattr(fault.value, "reason", "") == row["reason"]
    else:
        assert normalise_chromosome(row["label"]) == row["canonical"]


@pytest.mark.parametrize(
    "alid",
    [
        "22:16050075:A",
        "22:16050075:A:G:T",
        "chr22:100:A:G",
        "27:100:A:G",
        "22:0:A:G",
        "22:-1:A:G",
        "22:0100:A:G",
        "22:١٠٠:A:G",
        "22:100:A:",
        "22:100::G",
        "22:100:a:G",
        "22:100:G:A",
        "22:100:A:A",
        "foo:100:A:G",
        "22:2147483648:A:G",
        "22:" + "9" * 5000 + ":A:G",
        "22:" + "1" * 4301 + ":A:G",
        "22:100:A:N",
        ":100:A:G",
        "22: 100:A:G",
        "1:827587:GGGGCC:G",
    ],
)
def test_a_malformed_alid_has_no_locus(alid: str) -> None:
    assert sorted_alid_locus(alid) is None


@pytest.mark.parametrize(
    ("alid", "locus"),
    [("22:2147483647:A:G", ("22", 2147483647)), ("MT:1:C:T", ("MT", 1)), ("X:5:A:AC", ("X", 5))],
)
def test_an_alid_at_the_bounds_names_its_locus(alid: str, locus: tuple[str, int]) -> None:
    assert sorted_alid_locus(alid) == locus


def test_an_alid_over_32_bytes_names_its_locus() -> None:
    long = [r for r in ORIENT if r["alid"] and len(r["alid"]) > 32]
    assert long
    for row in long:
        chrom, pos = row["alid"].split(":")[:2]
        assert sorted_alid_locus(row["alid"]) == (chrom, int(pos))


@pytest.mark.parametrize(
    ("alid", "reference_first"),
    [
        ("1:827587:GGGGCC:G", True),
        ("22:400:CAA:CA", True),
        ("22:5:T:AT", True),  # an unanchored insertion whose REF sorts last
        ("1:827587:G:GGGGCC", False),
        ("22:403:GAT:GT", False),  # spelt alike by both
        ("22:100:G:A", False),  # an unsorted SNV: malformed in both, so Absent
        ("27:5:CAA:CA", False),  # no chromosome of the format: Absent
        ("22:5:CAA", False),
    ],
)
def test_only_an_indel_alid_no_sorted_spelling_forms_is_reference_first(
    alid: str, reference_first: bool
) -> None:
    assert spells_indel_reference_first(alid) is reference_first


