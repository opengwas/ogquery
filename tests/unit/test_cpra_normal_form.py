"""The 0.4 series is read, and the join guard (issue opengwas/ogquery#19; spec §2, §21.2;
opengwasdb ADR 0108 §9).

A 0.4.0 release holds exactly 0.4.0's `cpra_normal_form`; a release of any earlier series
has a form derived from its stamp; `require_joinable` refuses two releases whose forms
differ, because a join on the key string would match an SNV whose REF sorts first and
mis-sign it; and opening two forms in one process warns.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
from contract_reader import RELEASES
from format_0_2 import copy_as_0_2
from format_0_4 import DIGEST, normal_form_entry

from ogquery import format as store_format
from ogquery.format import (
    CpraNormalForm,
    CpraNormalFormMismatch,
    CpraSequence,
    MixedCpraNormalForms,
    UnsupportedCpraNormalForm,
    UnsupportedFormatVersion,
    cpra_normal_form,
    require_joinable,
)
from ogquery.manifest import StoreManifest
from ogquery.release import open_release

# Opening releases of different forms is what several tests below do on purpose; the ones
# that assert on the warning catch it themselves (`pytest.warns`).
pytestmark = pytest.mark.filterwarnings("ignore::ogquery.format.MixedCpraNormalForms")

FORM_0_4 = CpraNormalForm("ref_alt", "alt", "left_aligned", CpraSequence("GRCh38", DIGEST))
FORM_ALID = CpraNormalForm("snv_sorted_indel_reference_first", "a1", "as_spelt", None)
FORM_SORTED = CpraNormalForm("sorted", "a1", "as_spelt", None)


@pytest.fixture
def dense_0_4(copy_built_0_4: Any) -> Path:
    return Path(copy_built_0_4("dense_observed"))


@pytest.fixture
def dense_0_3(tmp_path: Path) -> Path:
    return Path(shutil.copytree(RELEASES / "dense_observed", tmp_path / "dense_0_3"))


def _other_sequence(digest: str):
    """A change declaring another sequence digest, in both places §2 requires it equal."""

    def change(manifest):
        manifest["cpra_normal_form"]["sequence"]["digest"] = digest
        manifest["provenance"]["reference_sequence"]["digest"] = digest

    return change


def _edit(release: Path, change: Any) -> None:
    path = release / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# --- seam 1: format acceptance -------------------------------------------------------


def test_this_release_reads_a_0_4_0_release(dense_0_4):
    assert store_format.parse_format_version("0.4.0") == (0, 4, 0)
    release = open_release(dense_0_4)
    assert release.manifest.format_version == "0.4.0"
    assert release.cpra_normal_form == FORM_0_4
    assert store_format.zarr_format_for_version("0.4.0") == 3


def test_a_0_4_series_remainder_above_0_is_read_with_a_warning(dense_0_4, caplog):
    _edit(dense_0_4, lambda m: m.update(format_version="0.4.1"))
    with caplog.at_level("WARNING", logger="ogquery.format"):
        open_release(dense_0_4)
    assert "newer compatible version" in caplog.text


def test_the_0_4_series_is_written_and_has_a_stamp():
    stamp = store_format.format_version_for
    assert store_format.WRITTEN_FORMAT_VERSIONS[-1] == "0.4.0"
    assert stamp(carries_regions=False, reference_first_indels=True,
                 cpra_keys=True, signed_for_alt=True) == "0.4.0"
    # The default stamp is still 0.3.0: nothing changes for a build that asks for none.
    assert stamp(carries_regions=False, reference_first_indels=True) == "0.3.0"
    assert store_format.carries_cpra("0.4.0") and not store_format.carries_cpra("0.3.0")


@pytest.mark.parametrize("keys", [(True, False), (False, True)])
def test_a_release_with_cpra_keys_and_not_alt_signs_is_no_version(keys):
    cpra_keys, signed_for_alt = keys
    with pytest.raises(ValueError, match="exactly when"):
        store_format.format_version_for(
            carries_regions=False, reference_first_indels=True,
            cpra_keys=cpra_keys, signed_for_alt=signed_for_alt,
        )


def test_a_0_4_release_without_0_3_features_is_no_version():
    with pytest.raises(ValueError, match="keeps 0.3.0's features"):
        store_format.format_version_for(
            carries_regions=False, reference_first_indels=False,
            cpra_keys=True, signed_for_alt=True,
        )


def _drop(manifest):
    del manifest["cpra_normal_form"]


def _set(key, value):
    def change(manifest):
        manifest["cpra_normal_form"][key] = value
    return change


def _set_sequence(key, value):
    def change(manifest):
        manifest["cpra_normal_form"]["sequence"][key] = value
    return change


def _extra(manifest):
    manifest["cpra_normal_form"]["note"] = "x"


def _null(manifest):
    manifest["cpra_normal_form"] = None


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (_drop, "must declare cpra_normal_form"),
        (_null, "must declare cpra_normal_form"),
        (_extra, "exactly"),
        (_set("alleles", "snv_sorted_indel_reference_first"), r"alleles .* 'ref_alt'"),
        (_set("alleles", "sorted"), r"alleles .* 'ref_alt'"),
        (_set("signed_for", "a1"), r"signed_for .* 'alt'"),
        (_set("indels", "as_spelt"), r"indels .* 'left_aligned'"),
        (_set("alleles", ["ref_alt"]), r"alleles"),
        (_set("sequence", None), "sequence"),
        (_set_sequence("assembly", "GRCh37"), "reference_assembly"),
        (_set_sequence("digest", DIGEST.upper()), "64 lower-case hex"),
        (_set_sequence("digest", DIGEST[:-1]), "64 lower-case hex"),
        (_set_sequence("digest", None), "64 lower-case hex"),
        (_set_sequence("digest", 7), "64 lower-case hex"),
    ],
    ids=lambda v: getattr(v, "__name__", None) or v[:20],
)
def test_a_0_4_0_manifest_with_any_other_form_is_refused_at_open(dense_0_4, change, why):
    """Absent, `null`, extra or other keys, and any value outside spec §2's: refused."""
    _edit(dense_0_4, change)
    with pytest.raises(UnsupportedCpraNormalForm, match=why):
        open_release(dense_0_4)


def test_a_refusal_is_an_unsupported_format(dense_0_4):
    """To a caller deciding whether it can read a release, it is one more way it cannot."""
    _edit(dense_0_4, _drop)
    with pytest.raises(UnsupportedFormatVersion):
        open_release(dense_0_4)


def test_an_earlier_release_may_not_declare_a_form(dense_0_3):
    _edit(dense_0_3, lambda m: m.update(cpra_normal_form=normal_form_entry()))
    with pytest.raises(UnsupportedCpraNormalForm, match="only 0.4.0 and later"):
        open_release(dense_0_3)


# --- the derived forms ----------------------------------------------------------------


def test_a_0_3_release_has_the_form_its_stamp_gives(dense_0_3):
    assert open_release(dense_0_3).cpra_normal_form == FORM_ALID


@pytest.mark.parametrize(
    ("version", "form"),
    [
        ("0.2.3", FORM_ALID),
        ("0.2.2", FORM_ALID),
        ("0.2.1", FORM_SORTED),
        ("0.2.0", FORM_SORTED),
    ],
)
def test_a_0_2_release_has_the_form_its_stamp_gives(tmp_path, version, form):
    copy = copy_as_0_2(RELEASES / "dense_observed", tmp_path, version)
    assert StoreManifest.load(copy).format_version == version
    assert cpra_normal_form(StoreManifest.load(copy)) == form


def test_a_converted_release_that_sorts_its_indels_has_the_sorted_form(dense_0_3):
    """`provenance.converted_from.deviations` listing `variant_table.indel_order` (ADR 0108 §9)
    makes it `sorted` whatever its stamp."""
    manifest = StoreManifest.load(dense_0_3)
    assert cpra_normal_form(manifest) == FORM_ALID
    _edit(
        dense_0_3,
        lambda m: m["provenance"].update(
            converted_from={"deviations": ["variant_table.indel_order"]}
        ),
    )
    assert cpra_normal_form(StoreManifest.load(dense_0_3)) == FORM_SORTED
    _edit(dense_0_3, lambda m: m["provenance"].update(converted_from={"deviations": ["other"]}))
    assert cpra_normal_form(StoreManifest.load(dense_0_3)) == FORM_ALID


def test_the_manifest_keeps_its_entry_when_written_back(dense_0_4):
    manifest = StoreManifest.load(dense_0_4)
    assert manifest.to_dict()["cpra_normal_form"] == normal_form_entry()
    assert "cpra_normal_form" not in StoreManifest.load(RELEASES / "dense_observed").to_dict()


# --- seam 2: require_joinable ---------------------------------------------------------


def test_two_0_4_releases_join(built_0_4):
    a = open_release(built_0_4("dense_observed"))
    b = open_release(built_0_4("ragged_observed"))
    require_joinable(a, b)
    require_joinable(b, a)


def test_a_0_3_release_does_not_join_a_0_4_one(tmp_path, dense_0_3, dense_0_4):
    _edit(dense_0_4, lambda m: m.update(release_id="rebuilt-0-4"))
    old, new = open_release(dense_0_3), open_release(dense_0_4)
    assert old.release_id != new.release_id
    for first, second in ((old, new), (new, old)):
        with pytest.raises(CpraNormalFormMismatch) as raised:
            require_joinable(first, second)
        text = str(raised.value)
        assert first.release_id in text and second.release_id in text
        assert "signed_for=a1" in text and "signed_for=alt" in text
        assert "Rebuild" in text and "no override" in text


def test_two_0_3_releases_join(tmp_path, dense_0_3):
    require_joinable(open_release(dense_0_3), open_release(dense_0_3))


def test_releases_of_one_series_on_different_sequences_do_not_join(
    built_0_4, copy_built_0_4
):
    a = open_release(built_0_4("dense_observed"))
    other = copy_built_0_4("ragged_observed")
    _edit(other, _other_sequence("0" * 64))
    with pytest.raises(CpraNormalFormMismatch, match="sequence"):
        require_joinable(a, open_release(other))


def test_the_sorted_form_does_not_join_the_reference_first_form(tmp_path, dense_0_3):
    """The join that is silent today: a converted Store that sorts its indels against a
    reference-first one (ADR 0108 §9)."""
    copy = Path(shutil.copytree(dense_0_3, tmp_path / "converted"))
    _edit(copy, lambda m: m["provenance"].update(
        converted_from={"deviations": ["variant_table.indel_order"]}))
    with pytest.raises(CpraNormalFormMismatch):
        require_joinable(open_release(dense_0_3), open_release(copy))


def test_a_form_without_a_release_id_is_not_a_release():
    with pytest.raises(AttributeError):
        require_joinable(FORM_0_4, FORM_ALID)  # type: ignore[arg-type]


# --- opening two forms ------------------------------------------------------------------


def test_opening_a_second_form_warns_naming_both(tmp_path, dense_0_3, dense_0_4):
    open_release(dense_0_3)
    with pytest.warns(MixedCpraNormalForms) as caught:
        open_release(dense_0_4)
    text = str(caught[0].message)
    assert "signed_for=a1" in text and "signed_for=alt" in text
    assert issubclass(MixedCpraNormalForms, UserWarning)


def test_it_warns_once_per_pair_of_forms(tmp_path, dense_0_3, dense_0_4):
    open_release(dense_0_3)
    with pytest.warns(MixedCpraNormalForms):
        open_release(dense_0_4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        open_release(dense_0_4)
        open_release(dense_0_3)


def test_opening_one_form_never_warns(tmp_path, dense_0_3):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        open_release(dense_0_3)
        open_release(dense_0_3)


def test_a_caller_that_joins_makes_it_an_error(tmp_path, dense_0_3, dense_0_4):
    open_release(dense_0_3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MixedCpraNormalForms)
        with pytest.raises(MixedCpraNormalForms):
            open_release(dense_0_4)


def test_a_third_form_warns_for_its_own_pairs(copy_built_0_4, dense_0_3, dense_0_4):
    other = copy_built_0_4("ragged_observed")
    _edit(other, _other_sequence("1" * 64))
    open_release(dense_0_3)
    with pytest.warns(MixedCpraNormalForms):
        open_release(dense_0_4)
    with pytest.warns(MixedCpraNormalForms):
        open_release(other)


# --- review round 1 -----------------------------------------------------------------


def test_an_explicit_null_form_on_an_earlier_release_is_refused(dense_0_3):
    """§2: the key is absent, not `null`: presence is what is tested."""
    _edit(dense_0_3, lambda m: m.update(cpra_normal_form=None))
    written = json.loads((dense_0_3 / "manifest.json").read_text(encoding="utf-8"))
    assert "cpra_normal_form" in written
    with pytest.raises(UnsupportedCpraNormalForm, match="only 0.4.0 and later"):
        open_release(dense_0_3)


def _drop_provenance(key):
    def change(manifest):
        del manifest["provenance"][key]
    return change


def _provenance(key, value):
    def change(manifest):
        manifest["provenance"][key] = value
    return change


def _reference_sequence(key, value):
    def change(manifest):
        manifest["provenance"]["reference_sequence"][key] = value
    return change


def _drop_reference_sequence_key(key):
    def change(manifest):
        del manifest["provenance"]["reference_sequence"][key]
    return change


def _admission(change_to):
    def change(manifest):
        change_to(manifest["provenance"]["admission"])
    return change


def _dropped(value):
    def change(admission):
        admission["dropped"] = value
    return change


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (_drop_provenance("reference_sequence"), "provenance.reference_sequence"),
        (_drop_provenance("admission"), "provenance.admission"),
        (_provenance("reference_sequence", None), "provenance.reference_sequence"),
        (_provenance("reference_sequence", "x"), "provenance.reference_sequence"),
        (_drop_reference_sequence_key("sha256"), "reference_sequence"),
        (_reference_sequence("note", "x"), "reference_sequence"),
        (_reference_sequence("assembly", "GRCh37"), "assembly"),
        (_reference_sequence("digest", "0" * 64), "digest"),
        (_reference_sequence("sha256", "AB" * 32), "sha256"),
        (_reference_sequence("format_version", 1.0), "format_version"),
        (_reference_sequence("format", None), "format"),
        (_reference_sequence("path", 7), "path"),
        (_reference_sequence("source", "x"), "source"),
        (_provenance("admission", None), "provenance.admission"),
        (_provenance("admission", {"rules": "admission/17"}), "provenance.admission"),
        (_admission(lambda a: a.update(rules=17)), "rules"),
        (_admission(_dropped([])), "dropped"),
        (_admission(_dropped({"ref_mismatch": -1})), "dropped"),
        (_admission(_dropped({"ref_mismatch": 1.5})), "dropped"),
        (_admission(_dropped({"ref_mismatch": True})), "dropped"),
        (_admission(_dropped({"ref_mismatch": "3"})), "dropped"),
        (_admission(lambda a: a.update(extra=1)), "provenance.admission"),
    ],
)
def test_a_0_4_0_release_must_declare_what_it_was_formed_against(dense_0_4, change, why):
    """§2: a release MUST also declare `provenance.reference_sequence` and
    `provenance.admission`, each of the JSON types §2 gives; refused at open, naming it."""
    _edit(dense_0_4, change)
    with pytest.raises(UnsupportedCpraNormalForm, match=why):
        open_release(dense_0_4)


def test_a_complete_0_4_0_declaration_opens(dense_0_4):
    _edit(
        dense_0_4,
        lambda m: m["provenance"]["admission"].update(dropped={}),
    )
    assert open_release(dense_0_4).cpra_normal_form == FORM_0_4


def test_a_0_4_release_of_another_assembly_is_refused(dense_0_4):
    """§4.0: a 0.4.0 release's Reference Assembly is GRCh38, the one a Byte Genome exists for."""

    def change(m):
        m["reference_assembly"] = "GRCh37"
        m["cpra_normal_form"]["sequence"]["assembly"] = "GRCh37"
        m["provenance"]["reference_sequence"]["assembly"] = "GRCh37"

    _edit(dense_0_4, change)
    with pytest.raises(UnsupportedCpraNormalForm, match="GRCh38"):
        open_release(dense_0_4)


def _dense_edit(release: Path, change: Any) -> None:
    path = release / "dense" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["cpra_normal_form"]["sequence"].update(digest="1" * 64),
        lambda m: m["provenance"]["reference_sequence"].update(sha256="ef" * 32),
        lambda m: m["provenance"]["reference_sequence"].update(path="/elsewhere"),
        lambda m: m["provenance"]["admission"]["dropped"].update(ref_mismatch=9),
        lambda m: m["provenance"]["admission"].update(rules="admission/18"),
        lambda m: m["provenance"].pop("admission"),
    ],
    ids=["digest", "sha256", "path", "dropped", "rules", "missing"],
)
def test_a_hybrids_declarations_are_equal_in_both_manifests(copy_built_0_4, change):
    """§2: a Hybrid declares them in both manifests, "each equal in the two": refused, not
    warned, whether the root or the Dense Component is opened."""
    root = copy_built_0_4("hybrid_observed")
    assert open_release(root).dense_component().cpra_normal_form == FORM_0_4  # planted: equal
    _dense_edit(root, change)
    with pytest.raises(UnsupportedFormatVersion, match="equal"):
        open_release(root)
    with pytest.raises(UnsupportedFormatVersion, match="equal"):
        open_release(root / "dense")


def test_a_direct_dense_component_has_the_roots_form(copy_built_0_4):
    root = copy_built_0_4("hybrid_observed")
    component = open_release(root / "dense")
    assert component.cpra_normal_form == open_release(root).cpra_normal_form == FORM_0_4


def test_a_refused_open_is_refused_again_on_retry(dense_0_3, dense_0_4):
    """ADR 0108 §9: a warnings-as-errors refusal is not remembered as warned."""
    open_release(dense_0_3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MixedCpraNormalForms)
        for _ in range(3):
            with pytest.raises(MixedCpraNormalForms):
                open_release(dense_0_4)
    # And once the caller allows the warning, it is issued exactly once.
    with pytest.warns(MixedCpraNormalForms):
        open_release(dense_0_4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        open_release(dense_0_4)
