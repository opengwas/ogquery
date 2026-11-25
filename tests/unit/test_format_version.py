"""`format_version` semantics and the reader's obligations (issue #143, ADR 0041).

The format was reset to `0.1.0` at the end of Roadmap 1, and the shape of the
version string is the safety mechanism rather than a cosmetic choice: the
pre-release formats were stamped `0.1`, `1.0`, `2.0` and `3.0`, and a reset
that reused `0.1` would give two different formats one name. Three components
cannot be misread as two, so every pre-reset release is refused loudly instead
of being decoded under a contract it was not written against.

These tests pin what a reader owes a store it did not write: refuse a
pre-reset release with an instruction rather than a decode attempt, refuse an
unknown release series, read a known one, and say something when it meets a
compatible version from the future rather than quietly returning a subset of
the data.

In ogquery these are the reader's gates (`ogquery.format`), and the real release is a
copy of a contract-bundle release opened by `open_release`. What a writer refuses --
completion, the converter, `check_writable_format_version` -- is opengwasdb's, and its
tests stay there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from contract_reader import RELEASES

from ogquery import format as store_format
from ogquery.facade import query_release
from ogquery.format import CURRENT_FORMAT_VERSION, PRE_RESET_FORMAT_VERSIONS
from ogquery.release import open_release


@pytest.fixture
def dense_store_path(tmp_path: Path) -> Path:
    """A copy of the bundle's Dense release, format 0.3.0, to restamp."""
    return Path(shutil.copytree(RELEASES / "dense_observed", tmp_path / "dense_observed"))


def _set_version(store: Path, version: str) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# --- parsing ---------------------------------------------------------------


def test_version_parses_as_major_minor_patch():
    assert store_format.parse_format_version("0.1.0") == (0, 1, 0)
    assert store_format.parse_format_version("12.34.56") == (12, 34, 56)


@pytest.mark.parametrize("version", sorted(PRE_RESET_FORMAT_VERSIONS))
def test_a_pre_reset_version_is_refused_with_an_instruction(version):
    """`0.1`, `1.0`, `2.0` and `3.0` are the formats this project burned before
    it published anything. None is readable, and the message says what to do
    about it rather than complaining about punctuation: those stores are
    rebuilt, not migrated."""
    with pytest.raises(store_format.UnsupportedFormatVersion, match="Rebuild the release"):
        store_format.parse_format_version(version)


def test_the_pre_reset_versions_are_the_two_component_ones():
    """The reset is safe because no pre-reset stamp can be read as a current
    one. That holds only while every pre-reset version really is two-component
    -- a three-component one would collide with the new numbering."""
    for version in PRE_RESET_FORMAT_VERSIONS:
        assert len(version.split(".")) == 2, version
    assert len(CURRENT_FORMAT_VERSION.split(".")) == 3


@pytest.mark.parametrize(
    "version", ["", "1", "1.2", "1.", ".1", "0.1.0.0", "banana", "0.1.x", "v0.1.0"]
)
def test_a_version_that_is_not_major_minor_patch_is_refused(version):
    """Refused, not coerced. To a caller deciding whether it can read a
    release, an unparseable version and a future one are the same answer."""
    with pytest.raises(store_format.MalformedFormatVersion):
        store_format.parse_format_version(version)
    # And it stops a read, not merely a parse.
    assert issubclass(store_format.MalformedFormatVersion, store_format.UnsupportedFormatVersion)


# --- which component carries an incompatible change ------------------------


def test_the_leftmost_non_zero_component_is_the_breaking_one():
    """Semantic versioning's own rule (ADR 0041 §1). Below 1.0.0 a format
    declares itself unsettled, so MINOR carries a breaking change; from 1.0.0
    it is MAJOR, and MINOR joins the compatible remainder."""
    assert store_format.split_format_version("0.1.0") == ((0, 1), (0,))
    assert store_format.split_format_version("0.1.7") == ((0, 1), (7,))
    assert store_format.split_format_version("0.2.0") == ((0, 2), (0,))
    assert store_format.split_format_version("1.4.2") == ((1,), (4, 2))
    assert store_format.split_format_version("1.0.0") == ((1,), (0, 0))


# --- the reader contract ---------------------------------------------------


def test_the_version_this_build_writes_is_one_it_can_read():
    """Otherwise the build could not open what it had just produced."""
    store_format.check_format_version(CURRENT_FORMAT_VERSION)
    series, remainder = store_format.split_format_version(CURRENT_FORMAT_VERSION)
    assert store_format.SUPPORTED_FORMAT_VERSIONS[series] >= remainder


def test_the_readable_formats_are_the_two_the_epic_carries():
    """Two series: 0.1 (Zarr v2) and 0.2 (Zarr v3 with sharding, since #245). The 0.2
    series reads to 0.2.3: 0.2.1 adds a Ragged release's optional `regions.tsv` (#430),
    0.2.2 spells an indel's ALID reference first (#381, ADR 0079), and 0.2.3 records
    `n_missing_se_out_of_range` (#338, ADR 0068). The 0.3 series (#467, ADR 0105) is
    0.2.3 with the variable-length ALID sidecar. Another entry is a decision, not an
    accident."""
    assert dict(store_format.SUPPORTED_FORMAT_VERSIONS) == {
        (0, 1): (0,),
        (0, 2): (3,),
        (0, 3): (0,),
    }


def test_a_release_is_stamped_the_lowest_version_whose_features_it_uses():
    """The owner's decision of 2026-11-11 (spec §21.3, ADR 0038 §3): 0.2.1 differs from
    0.2.0 only by an optional, manifest-declared `regions.tsv`, and 0.2.2 from 0.2.1
    only by an indel's ALID spelt reference first, the region table still optional. A
    release is stamped the lowest of the three whose features it uses. 0.2.3 is 0.2.2
    whose `analyses.tsv` records `n_missing_se_out_of_range` (issue #338), so a release
    whose indels are sorted is stamped by them, whatever it counts."""
    stamp = store_format.format_version_for
    for counts in (False, True):
        assert stamp(
            carries_regions=False, reference_first_indels=False, records_se_out_of_range=counts,
            alid_sidecar=False,
        ) == "0.2.0"
        assert stamp(
            carries_regions=True, reference_first_indels=False, records_se_out_of_range=counts,
            alid_sidecar=False,
        ) == "0.2.1"
    for regions in (False, True):
        assert stamp(
            carries_regions=regions, reference_first_indels=True, records_se_out_of_range=False,
            alid_sidecar=False,
        ) == "0.2.2"
        assert stamp(
            carries_regions=regions, reference_first_indels=True, records_se_out_of_range=True,
            alid_sidecar=False,
        ) == "0.2.3"
        # 0.3.0 is 0.2.3 with the variable-length ALID sidecar (#467, ADR 0105).
        assert stamp(carries_regions=regions, reference_first_indels=True) == "0.3.0"
    assert store_format.WRITTEN_FORMAT_VERSIONS == ("0.2.0", "0.2.1", "0.2.2", "0.2.3", "0.3.0")


def test_a_sidecar_without_0_2_3_s_features_is_no_version():
    """0.3.0 carries every 0.2.3 feature; a sidecar on sorted indels, or without the SE
    count, is a release no version describes, and is refused rather than stamped."""
    stamp = store_format.format_version_for
    with pytest.raises(ValueError, match="reference first"):
        stamp(carries_regions=False, reference_first_indels=False)
    with pytest.raises(ValueError, match="counts the SEs"):
        stamp(carries_regions=False, reference_first_indels=True, records_se_out_of_range=False)


def test_the_current_version_is_the_stamp_of_a_release_using_no_optional_feature():
    """What every build of this package writes for a release without a region table,
    since every build spells an indel reference first, and what opengwasdb-stores'
    engine update probes (`ogstores.engine_update`) and compares with its
    `REQUIRED_FORMAT_VERSION`."""
    assert CURRENT_FORMAT_VERSION == store_format.format_version_for(
        carries_regions=False, reference_first_indels=True
    )
    assert CURRENT_FORMAT_VERSION == "0.3.0"


def test_each_readable_series_has_one_zarr_on_disk_format():
    """The version-to-layout rule lives beside the version tables, so a caller
    does not pattern-match a version string to guess it (#264 review)."""
    assert store_format.zarr_format_for_version("0.1.0") == 2
    assert store_format.zarr_format_for_version(CURRENT_FORMAT_VERSION) == 3
    assert dict(store_format.ZARR_FORMAT_BY_SERIES) == {(0, 1): 2, (0, 2): 3, (0, 3): 3}


def test_an_unreadable_version_has_no_zarr_layout():
    with pytest.raises(store_format.UnsupportedFormatVersion):
        store_format.zarr_format_for_version("9.9.9")


def test_an_unknown_series_is_rejected():
    """A breaking change moves the series, and this build reads three of them."""
    for version in ("0.4.0", "1.0.0", "9.9.9"):
        with pytest.raises(store_format.UnsupportedFormatVersion, match="release series"):
            store_format.check_format_version(version)


def test_a_newer_compatible_version_is_read_but_warned_about(caplog):
    """Reading it is the definition of a compatible change; the warning is how
    one misclassified as compatible becomes visible instead of silently losing
    data."""
    with caplog.at_level("WARNING"):
        store_format.check_format_version("0.1.7")

    assert "0.1.7" in caplog.text
    assert "not visible here" in caplog.text


# --- against a real release ------------------------------------------------


def test_open_release_refuses_a_pre_reset_release(dense_store_path):
    """The headline of the reset: a release stamped with a pre-release format is
    never decoded. Its bytes may or may not be readable -- a `3.0` store's are
    identical to a `0.1.0` store's, a `0.1` store's are not -- and this build
    does not try to tell the two apart."""
    _set_version(dense_store_path, "3.0")

    with pytest.raises(store_format.UnsupportedFormatVersion, match="Rebuild the release"):
        open_release(dense_store_path)


def test_open_release_rejects_an_unknown_series(dense_store_path):
    _set_version(dense_store_path, "1.0.0")

    with pytest.raises(store_format.UnsupportedFormatVersion):
        open_release(dense_store_path)


@pytest.mark.parametrize(
    ("version", "reference_first"),
    [("0.1.0", False), ("0.2.0", False), ("0.2.1", False), ("0.2.2", True), ("0.2.3", True)],
)
def test_a_store_spells_its_indels_reference_first_from_0_2_2(version, reference_first):
    """0.2.1 is package 0.9.0's region table, its indels still sorted; #381's reference-
    first spelling is 0.2.2 (ADR 0079)."""
    assert store_format.indels_reference_first(version) is reference_first


def test_each_0_3_0_feature_starts_at_0_3_0_and_each_dropped_file_stops_there():
    """0.3.0 is one format version (ADR 0105): every sidecar it adds is carried from it,
    and everything it drops is carried before it, so a reader asks one question per file
    and the answers never disagree within a release."""
    for version, is_0_3 in (("0.1.0", False), ("0.2.3", False), ("0.3.0", True)):
        assert store_format.carries_alid_sidecar(version) is is_0_3
        assert store_format.carries_position_sidecar(version) is is_0_3
        assert store_format.carries_source_alid_sidecar(version) is is_0_3
        assert store_format.dense_component_shares_axis(version) is is_0_3
        assert store_format.carries_variant_offsets(version) is not is_0_3
        assert store_format.carries_variant_aliases(version) is not is_0_3
        assert store_format.carries_index_metadata(version) is not is_0_3
        assert store_format.carries_stored_p(version) is not is_0_3


#: The version tables as they were before #467: the package that does not know 0.3.
_BEFORE_0_3 = {(0, 1): (0,), (0, 2): (3,)}


def test_a_package_without_the_0_3_series_refuses_a_0_3_0_release_by_name(
    dense_store_path, monkeypatch
):
    """ADR 0041: a series this build does not implement is refused, and says which
    version it was. A 0.3.0 release carries no fixed-width ALID index, so a 0.2 reader
    that opened it would find none and answer ALID lookups wrongly (ADR 0105)."""
    from types import MappingProxyType

    assert open_release(dense_store_path).manifest.format_version == "0.3.0"
    monkeypatch.setattr(store_format, "SUPPORTED_FORMAT_VERSIONS", MappingProxyType(_BEFORE_0_3))

    with pytest.raises(store_format.UnsupportedFormatVersion) as refused:
        query_release(open_release(dense_store_path))

    message = str(refused.value)
    assert "format_version='0.3.0'" in message, message
    assert "a release series this build does not implement" in message, message
    assert "readable series: ['0.1', '0.2']" in message, message
