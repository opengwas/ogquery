"""`open_release(path)` with no Variant Table reader opens a 0.3.0 release (#474), and
refuses an earlier one by its `format_version` and how it becomes 0.3.0 (ogquery#1): a
0.2.3 release by `ogdb rewrite-release`, an earlier one by a rebuild from source, since
the rewrite refuses it.

A 0.3.0 release's queries read its sidecars, never `variants.tsv.gz` (ADR 0105 §7, §8),
so the Store Reader alone -- ogquery's whole read path -- must open it without the
pysam reader opengwasdb injects. Its table refuses every read instead
(`NoVariantTable`). A 0.2 release, whose queries do read the table, is refused unless a
reader is passed (`open_release(path, legacy=...)`); with one, it answers as the 0.3.0
release it is a copy of. The 0.2 releases are 0.2-shaped copies of bundle releases
(`format_0_2`), since only opengwasdb builds a real one.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pytest
from contract_reader import RELEASES, disagreements, expected
from format_0_2 import GzipVariantTable, copy_as_0_2

from ogquery.digests import VALUES, readable_value, release_answers
from ogquery.facade import query_release
from ogquery.release import (
    REWRITABLE_FORMAT_VERSION,
    REWRITE_COMMAND,
    VariantTableReaderRequired,
    open_release,
)
from ogquery.variant_axis import VariantTableUnread

BUNDLE = RELEASES


@pytest.mark.parametrize("name", ["dense_completed", "ragged_observed", "hybrid_completed"])
def test_a_0_3_0_release_opens_and_answers_without_a_variant_table_reader(name: str):
    with query_release(open_release(BUNDLE / name)) as q:
        table = q.variants_table()
        first = q.analyses_table()[0]["analysis_id"]
        assert len(table) > 100 and q.analysis(first)["z"].size
        row = table[min(table)]
        assert q.phewas(row["alid"])["z"].size
        assert q.variants_in_range(row["chromosome"], row["position"], row["position"])


def test_a_0_3_0_read_that_reaches_the_table_is_refused_by_name():
    release = open_release(BUNDLE / "dense_ranges")
    axis = release.variant_axis(aliases=False)
    with pytest.raises(VariantTableUnread, match="fetch"):
        axis.tabix_range("1", 1, 10_000)


def _refusal(tmp_path: Path, name: str, version: str) -> VariantTableReaderRequired:
    release = open_release(copy_as_0_2(BUNDLE / name, tmp_path, version))
    with pytest.raises(VariantTableReaderRequired) as refused:
        query_release(release)
    assert (refused.value.format_version, refused.value.path) == (version, release.path)
    return refused.value


@pytest.mark.parametrize("name", ["dense_ranges", "ragged_completed"])
def test_a_0_2_3_release_is_refused_naming_its_version_and_the_rewrite(
    tmp_path: Path, name: str
):
    refused = _refusal(tmp_path, name, "0.2.3")
    message = str(refused)
    assert REWRITE_COMMAND == "ogdb rewrite-release" and REWRITABLE_FORMAT_VERSION == "0.2.3"
    assert "format_version 0.2.3" in message, message
    assert "Rewrite it as 0.3.0 with `ogdb rewrite-release`" in message, message
    assert "Rebuild" not in message, message
    assert "open_release(..., legacy=...)" in message, message
    # The text opengwasdb's own test matches, and the type a caller already caught.
    assert "opened without a reader for its Store Variant Table" in message, message
    assert isinstance(refused, ValueError)


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.2.2"])
@pytest.mark.parametrize("name", ["dense_ranges", "ragged_completed"])
def test_an_earlier_release_is_told_to_rebuild_not_to_rewrite(
    tmp_path: Path, name: str, version: str
):
    """`ogdb rewrite-release` refuses anything but 0.2.3, so naming it as the remedy for an
    earlier release would send the user to a command that fails."""
    message = str(_refusal(tmp_path, name, version))
    assert f"format_version {version}" in message, message
    assert "Rebuild it from source at format 0.3.0" in message, message
    assert "Rewrite it" not in message, message
    assert "rewrites only a 0.2.3 release" in message, message
    assert "open_release(..., legacy=...)" in message, message
    assert "opened without a reader for its Store Variant Table" in message, message


def test_a_0_2_release_opens_without_a_reader_until_a_query_needs_the_table(tmp_path: Path):
    """The manifest and the arrays read without the table; the Variant Axis does not."""
    release = open_release(copy_as_0_2(BUNDLE / "dense_ranges", tmp_path))
    assert release.manifest.format_version == "0.2.3"
    assert release.arrays() is not None
    with pytest.raises(VariantTableReaderRequired, match="format_version 0.2.3"):
        release.variant_axis(aliases=True)


def _with_reader(path: Path, tables: list[GzipVariantTable]) -> Any:
    def legacy(release: Path, *, aliases: bool) -> GzipVariantTable:
        tables.append(GzipVariantTable(release, aliases=aliases))
        return tables[-1]

    return query_release(open_release(path, legacy=legacy))


@pytest.mark.parametrize("name", ["dense_observed", "ragged_completed", "dense_ranges"])
def test_a_0_2_release_with_a_variant_table_reader_answers_as_its_0_3_0_release(
    tmp_path: Path, name: str
):
    """Every question of the contract, asked of the 0.2.3 copy through a reader of its
    table, answers as the committed 0.3.0 release does, and the table was read."""
    source = copy_as_0_2(BUNDLE / name, tmp_path)
    tables: list[GzipVariantTable] = []
    found = release_answers(source, query=partial(_with_reader, tables=tables))
    for kind in VALUES:
        found["digests"][kind] = {
            shape: [readable_value(v) for v in values]
            for shape, values in found["digests"][kind].items()
        }
    assert disagreements(found, expected(name)) == []
    assert tables and any(table.reads for table in tables)


def test_the_bundle_release_is_untouched_by_a_copy(tmp_path: Path):
    before = sorted(p.name for p in (BUNDLE / "dense_ranges").iterdir())
    copy_as_0_2(BUNDLE / "dense_ranges", tmp_path)
    assert sorted(p.name for p in (BUNDLE / "dense_ranges").iterdir()) == before
