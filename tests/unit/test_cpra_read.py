"""A 0.4.0 release read back, and the query contract on it (issue opengwas/ogquery#19;
spec §4.0, §5, §19, §21.5).

The 0.4.0 releases are 0.3.0 contract-bundle releases rewritten to the spec's names by
`format_0_4` (its docstring says how and what it does not claim), for each layout the reader
supports: Dense, Ragged and Hybrid, observed and completed. Each test asserts the fixture's
planted cases before it asserts about them: the SNV and the indel it asks for are rows the
release holds, the spellings it expects to miss are rows it does not.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from contract_reader import RELEASES
from format_0_2 import GzipVariantTable
from format_0_4 import DIGEST, as_0_4, cpra_of, cpras

from ogquery.alid_index import MissingAlidIndex
from ogquery.alid_sidecar import AlidSidecar
from ogquery.cpra_query import (
    NotNormalCpra,
    VariantMiss,
    VariantMissWarning,
)
from ogquery.cpra_sidecar import CPRA_SIDECAR_FILENAMES, SOURCE_CPRA_SIDECAR_FILENAMES
from ogquery.format import CpraNormalForm, CpraSequence
from ogquery.identity import MalformedIdentity, UnsupportedChromosome
from ogquery.npy_sidecar import release_maps
from ogquery.release import open_release
from ogquery.resolve import resolve_rows
from ogquery.rsid_index import RsidIndexNotYetSupported
from ogquery.source_alid_sidecar import SourceAlidSidecar

pytestmark = pytest.mark.filterwarnings("ignore::ogquery.format.MixedCpraNormalForms")

FORM = CpraNormalForm("ref_alt", "alt", "left_aligned", CpraSequence("GRCh38", DIGEST))
LAYOUTS = [
    "dense_observed",
    "dense_completed",
    "ragged_observed",
    "ragged_completed",
    "hybrid_observed",
    "hybrid_completed",
]


def _alids(release: Path) -> list[str]:
    sidecar = AlidSidecar.open(release)
    assert sidecar is not None
    out = [str(a) for a in sidecar.alids(np.arange(sidecar.n_rows)).tolist()]
    release_maps(sidecar)
    return out


def _sources(release: Path) -> list[str | None]:
    alids, sidecar = AlidSidecar.open(release), SourceAlidSidecar.open(release)
    assert alids is not None and sidecar is not None
    rows = np.arange(alids.n_rows)
    out = sidecar.values(rows, alids.alids).tolist()
    release_maps(alids)
    release_maps(sidecar)
    return out


@pytest.fixture(params=LAYOUTS)
def pair(request: pytest.FixtureRequest, built_0_4: Any) -> tuple[Path, Path]:
    """The 0.3.0 release and its 0.4.0 copy (built once per session: read only)."""
    return RELEASES / request.param, built_0_4(request.param)


@pytest.fixture(params=["dense_observed", "ragged_observed", "hybrid_observed"])
def observed(request: pytest.FixtureRequest, built_0_4: Any) -> Path:
    """One 0.4.0 release per facade, for refusals made before any layout-specific read."""
    return Path(built_0_4(request.param))


class Planted:
    """The cases a layout's fixture plants, found and asserted to be there."""

    def __init__(self, old: Path, new: Path) -> None:
        self.alids = _alids(old)
        self.cpras = cpras(new)
        assert len(self.alids) == len(self.cpras) > 0
        snv = [a for a in self.alids if len(a.split(":")[2]) == len(a.split(":")[3]) == 1]
        indel = [a for a in self.alids if len(a.split(":")[2]) != len(a.split(":")[3])]
        assert snv and indel, "the fixture plants an SNV and an indel"
        self.snv_alid, self.indel_alid = snv[0], indel[0]
        self.snv = cpra_of(self.snv_alid)
        self.indel = cpra_of(self.indel_alid)
        chrom, pos, ref, alt = self.snv.split(":")
        assert ref != self.snv_alid.split(":")[2], "the SNV is respelt REF:ALT, ALT the old A1"
        assert self.snv in self.cpras
        #: The REF-second spelling of the SNV: the row the release holds, written ALID-style.
        self.swapped = f"{chrom}:{pos}:{alt}:{ref}"
        assert self.swapped not in self.cpras
        self.snv_row = self.cpras.index(self.snv)
        self.indel_row = self.cpras.index(self.indel)
        ichrom, ipos, iref, ialt = self.indel.split(":")
        #: The reverse indel: a different variant, absent here, never looked up swapped.
        self.reversed_indel = f"{ichrom}:{ipos}:{ialt}:{iref}"
        assert self.reversed_indel not in self.cpras
        #: An absent SNV whose swapped spelling is absent too.
        self.absent_snv = "22:49999999:A:G"
        assert self.absent_snv not in self.cpras
        assert "22:49999999:G:A" not in self.cpras
        #: An absent indel.
        self.absent_indel = "22:49999999:A:AT"
        assert self.absent_indel not in self.cpras


def _query(path: Path) -> Any:
    return open_release(path).query()


def _ids(query: Any) -> list[str]:
    return [row["analysis_id"] for row in query.analyses_table().values()]


# --- seam 3: read back ---------------------------------------------------------------


def test_every_row_reads_back_as_cpra_ref_alt(pair):
    old, new = pair
    planted = Planted(old, new)
    release = open_release(new)
    axis = release.variant_axis(aliases=False)
    assert axis.keys_are_cpra and axis.n_variants == len(planted.cpras)
    rows = np.arange(axis.n_variants, dtype=np.int64)
    identity = axis.identity_by_indices(rows, expect_space=axis.index_space)
    assert list(identity) == ["chromosome", "position", "ref", "alt", "cpra"]
    assert axis.identity_columns == tuple(identity)
    for row, cpra in enumerate(planted.cpras):
        chrom, pos, ref, alt = cpra.split(":")
        assert identity["cpra"][row] == cpra
        assert (identity["chromosome"][row], int(identity["position"][row])) == (chrom, int(pos))
        assert (identity["ref"][row], identity["alt"][row]) == (ref, alt)
        assert f"{chrom}:{pos}:{ref}:{alt}" == cpra
    assert "effect_allele" not in identity and "other_allele" not in identity


def test_a_0_3_release_keeps_its_names_and_columns(pair):
    old, _ = pair
    axis = open_release(old).variant_axis(aliases=False)
    assert not axis.keys_are_cpra
    assert axis.identity_columns == (
        "chromosome", "position", "alid", "effect_allele", "other_allele",
    )
    assert axis.source_column == "source_alid"
    identity = axis.identity_by_indices(np.arange(3), expect_space=axis.index_space)
    assert identity["alid"][0] == _alids(old)[0]
    assert axis.cpra_normal_form.signed_for == "a1"


def test_values_are_the_stored_ones_unchanged(pair):
    """Statistics are as stored, signed for ALT: the rewrite signs nothing, so every result
    carries exactly the 0.3.0 release's values for the same row."""
    old, new = pair
    planted = Planted(old, new)
    for alid, cpra in ((planted.snv_alid, planted.snv), (planted.indel_alid, planted.indel)):
        with _query(old) as before, _query(new) as after:
            ids = _ids(before)
            want = before.lookup([alid], ids)
            got = after.lookup([cpra], ids)
            assert want.count > 0, "the fixture's variant has associations"
            assert sorted(got.keys()) == sorted(want.keys())
            for key in want:
                np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(want[key]))


def test_resolved_rows_carry_one_column_order_and_no_effect_allele(pair):
    old, new = pair
    planted = Planted(old, new)
    release = open_release(new)
    with release.query() as q:
        found = q.lookup([planted.snv, planted.indel], _ids(q))
        assert found.count > 0
        rows = list(
            resolve_rows(q._analyses, q._variant_axis, found, include_variant_info=False)
        )
    assert {row["cpra"] for row in rows} == {planted.snv, planted.indel}
    for row in rows:
        names = list(row)
        assert names[:2] == ["analysis_id", "analysis_label"]
        assert names[2:7] == ["chromosome", "position", "ref", "alt", "cpra"]
        assert "effect_allele" not in row and "other_allele" not in row
        assert "alid" not in row
        chrom, pos, ref, alt = str(row["cpra"]).split(":")
        assert (row["chromosome"], row["position"], row["ref"], row["alt"]) == (
            chrom, int(pos), ref, alt,
        )


def test_an_rsid_is_not_yet_supported_on_0_4_0(pair):
    """#524's index has no layout: asking for an rsid, or for a row's, is a clear refusal and
    never an empty answer or a `.`."""
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        ids = _ids(q)
        with pytest.raises(RsidIndexNotYetSupported, match=r"#524"):
            q.lookup(["rs12345"], ids)
        with pytest.raises(RsidIndexNotYetSupported, match=r"#524"):
            q.variants(["rs12345"])
        with pytest.raises(RsidIndexNotYetSupported, match=r"#524"):
            q.variants_table()
        found = q.lookup([planted.snv], ids)
        with pytest.raises(RsidIndexNotYetSupported, match=r"#524"):
            list(resolve_rows(q._analyses, q._variant_axis, found, include_variant_info=True))


def _with_stand_in_rsids(query: Any) -> None:
    """The one thing a 0.4.0 reader cannot yet name, a row's rsID (#524), supplied as `None`:
    a stand-in for #524's index, so `variants()` and `variants_table()` can be read."""
    axis = query._variant_axis
    axis.rsids_by_indices = lambda rows, *, expect_space: np.full(
        len(np.atleast_1d(rows)), None, dtype=object
    )


def test_a_variants_table_row_is_the_tables_columns_without_variant_index(pair):
    old, new = pair
    planted = Planted(old, new)
    sources = _sources(old)
    renamed = dict(zip(planted.alids, planted.cpras, strict=True))
    with _query(new) as q:
        _with_stand_in_rsids(q)
        table = q.variants_table()
        assert sorted(table) == list(range(len(planted.cpras)))
        for row, cpra in enumerate(planted.cpras):
            chrom, pos, ref, alt = cpra.split(":")
            assert list(table[row]) == [
                "chromosome", "position", "ref", "alt", "cpra", "rsid", "source_cpra",
            ]
            want_source = None if sources[row] is None else renamed.get(sources[row], sources[row])
            assert table[row] == {
                "chromosome": chrom, "position": int(pos), "ref": ref, "alt": alt,
                "cpra": cpra, "rsid": None, "source_cpra": want_source,
            }
        # The variant-only reads equal `variants_table()` filtered to the rows asked.
        asked = q.variants([planted.snv, planted.indel])
        wanted = sorted((planted.snv_row, planted.indel_row))
        assert asked == {row: table[row] for row in wanted}
        assert list(asked) == wanted


def test_source_cpra_is_read_from_its_own_sidecar(pair):
    old, new = pair
    planted = Planted(old, new)
    axis = open_release(new).variant_axis(aliases=False)
    rows = np.arange(axis.n_variants)
    got = axis.source_alids_by_indices(rows, expect_space=axis.index_space).tolist()
    renamed = dict(zip(planted.alids, planted.cpras, strict=True))
    want = [None if s is None else renamed.get(s, s) for s in _sources(old)]
    assert got == want
    assert any(value is not None for value in got), "the fixture names some source_cpra"
    assert axis.source_column == "source_cpra"


def test_a_release_is_read_by_its_series_not_by_the_files_present(pair, tmp_path):
    """0.3 names beside a 0.4.0 release, and 0.4.0 names beside a 0.3 one, are never read."""
    old, new = pair
    planted = Planted(old, new)
    three = Path(shutil.copytree(old, tmp_path / "three"))
    four = Path(shutil.copytree(new, tmp_path / "four"))
    for name in AlidSidecar.FILENAMES + SourceAlidSidecar.FILENAMES:
        shutil.copy(old / name, four / name)  # 0.3 names beside the 0.4.0 release
    for name in CPRA_SIDECAR_FILENAMES + SOURCE_CPRA_SIDECAR_FILENAMES:
        shutil.copy(new / name, three / name)  # 0.4.0 names beside the 0.3 one
    axis4 = open_release(four).variant_axis(aliases=False)
    axis3 = open_release(three).variant_axis(aliases=False)
    rows = np.arange(axis4.n_variants)
    assert axis4.identity_by_indices(rows, expect_space=axis4.index_space)["cpra"].tolist() == (
        planted.cpras
    )
    assert axis3.identity_by_indices(rows, expect_space=axis3.index_space)["alid"].tolist() == (
        planted.alids
    )
    assert planted.cpras != planted.alids


def test_a_0_4_release_without_its_cpra_sidecar_is_refused_not_read_from_alids(pair, tmp_path):
    old, new = pair
    four = Path(shutil.copytree(new, tmp_path / "four"))
    for name in AlidSidecar.FILENAMES:
        shutil.copy(old / name, four / name)
    for name in CPRA_SIDECAR_FILENAMES:
        (four / name).unlink()
    with pytest.raises(MissingAlidIndex, match="CPRA index"):
        open_release(four).variant_axis(aliases=False)


# --- seam 4: the query contract, misses ------------------------------------------------


def test_a_cpra_that_is_not_normal_by_shape_is_refused(pair):
    old, new = pair
    planted = Planted(old, new)
    chrom, pos = planted.snv.split(":")[:2]
    cases = {
        f"{chrom}:{pos}:CC:TC": f"{chrom}:{pos}:C:T",
        f"{chrom}:{int(pos)}:GCA:GTA": f"{chrom}:{int(pos) + 1}:C:T",
        f"chr{chrom}:{pos}:AAC:AG": f"{chrom}:{int(pos) + 1}:AC:G",
    }
    with _query(new) as q:
        ids = _ids(q)
        for asked, trimmed in cases.items():
            for call in (
                lambda a=asked: q.lookup([a], ids),
                lambda a=asked: q.variants([a]),
                lambda a=asked: q._variant_axis.index_of(a),
            ):
                with pytest.raises(NotNormalCpra) as raised:
                    call()
                assert raised.value.identifier == asked
                assert raised.value.trimmed == trimmed
                assert raised.value.fields == {"identifier": asked, "trimmed": trimmed}
        # Nothing is looked up under either spelling, even when the trimmed one is held.
        with pytest.raises(NotNormalCpra):
            q.lookup([planted.snv, f"{chrom}:{pos}:CC:TC"], ids)


@pytest.mark.parametrize(
    ("asked", "error"),
    [
        ("1:0100:A:G", MalformedIdentity),
        ("1:+100:A:G", MalformedIdentity),
        ("1:-5:A:G", MalformedIdentity),
        ("1:0:A:G", MalformedIdentity),
        ("1:x:A:G", MalformedIdentity),
        ("1:100:A:A", MalformedIdentity),
        ("1:100:A:N", MalformedIdentity),
        ("1:100:A:", MalformedIdentity),
        ("1:100::G", MalformedIdentity),
        ("HLA-A:100:A:G", UnsupportedChromosome),
        ("27:100:A:G", UnsupportedChromosome),
    ],
)
def test_a_malformed_cpra_is_refused_as_a_malformed_alid_is(observed, asked, error):
    new = observed
    with _query(new) as q:
        with pytest.raises(error):
            q.lookup([asked], _ids(q))


def test_an_absent_cpra_gives_no_row_and_no_miss(pair):
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        found = q.lookup([planted.absent_snv], _ids(q))
        assert found.count == 0
        assert found.misses == ()
        assert q._variant_axis.index_of(planted.absent_snv) is None


def test_a_swapped_snv_spelling_is_a_miss_naming_the_row(pair):
    old, new = pair
    planted = Planted(old, new)
    release = open_release(new)
    with release.query() as q:
        found = q.lookup([planted.swapped], _ids(q))
        assert found.count == 0, "no row: the swapped spelling is not the variant's key"
        (miss,) = found.misses
        assert isinstance(miss, VariantMiss)
        assert miss.identifier == planted.swapped
        assert miss.release_id == release.release_id
        assert miss.reason == "swapped_alleles"
        assert miss.cpra == planted.snv
        assert miss.variant_index == planted.snv_row
        assert miss.cpra_normal_form == FORM
        assert found.cpra_normal_form == FORM
        # The row the miss names is the one the CPRA finds.
        assert q._variant_axis.index_of(planted.snv) == miss.variant_index


def test_an_absent_indel_is_a_miss_naming_the_form(pair):
    old, new = pair
    planted = Planted(old, new)
    release = open_release(new)
    with release.query() as q:
        for absent in (planted.absent_indel, planted.reversed_indel):
            found = q.lookup([absent], _ids(q))
            assert found.count == 0
            (miss,) = found.misses
            # An indel's swapped spelling is not looked up: it is the other indel.
            assert (miss.reason, miss.cpra, miss.variant_index) == ("indel_absent", None, None)
            assert miss.identifier == absent and miss.release_id == release.release_id
            assert miss.cpra_normal_form == FORM and miss.cpra_normal_form.indels == "left_aligned"


def test_a_present_indel_and_a_present_snv_give_rows_and_no_miss(pair):
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        found = q.lookup([planted.snv, planted.indel], _ids(q))
        assert found.count > 0 and found.misses == ()
        assert set(found.variant_index.tolist()) == {planted.snv_row, planted.indel_row}


def test_misses_come_in_the_order_asked(pair):
    old, new = pair
    planted = Planted(old, new)
    asked = [
        planted.absent_indel,
        planted.snv,
        planted.swapped,
        planted.absent_snv,
        planted.reversed_indel,
        planted.swapped,
    ]
    with _query(new) as q:
        found = q.lookup(asked, _ids(q))
        assert [(m.identifier, m.reason) for m in found.misses] == [
            (planted.absent_indel, "indel_absent"),
            (planted.swapped, "swapped_alleles"),
            (planted.reversed_indel, "indel_absent"),
            (planted.swapped, "swapped_alleles"),
        ]
        assert set(found.variant_index.tolist()) == {planted.snv_row}


def test_variants_warns_each_miss_and_keeps_its_dict_answer(pair, caplog):
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        _with_stand_in_rsids(q)
        with pytest.warns(VariantMissWarning) as caught:
            rows = q.variants([planted.swapped, planted.snv, planted.absent_indel])
        assert list(rows) == [planted.snv_row]
        assert rows[planted.snv_row]["cpra"] == planted.snv
        misses = [w.message.miss for w in caught if isinstance(w.message, VariantMissWarning)]
        assert [(m.identifier, m.reason) for m in misses] == [
            (planted.swapped, "swapped_alleles"),
            (planted.absent_indel, "indel_absent"),
        ]
        assert issubclass(VariantMissWarning, UserWarning)
        text = str(caught[0].message)
        for field in ("identifier=", "release_id=", "reason=", "cpra=", "variant_index="):
            assert field in text
        assert "cpra_normal_form=" in text
        # An absent SNV whose swap names no row, and a found row, warn nothing.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q.variants([planted.snv, planted.absent_snv])


def test_every_answer_carries_the_releases_form_and_keeps_it(pair):
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        ids = _ids(q)
        found = q.lookup([planted.snv, planted.indel], ids)
        assert found.cpra_normal_form == FORM
        assert found.take(slice(0, 1)).cpra_normal_form == FORM
        assert found.observed_only().cpra_normal_form == FORM
        again = type(found).concat([found, found])
        assert again.cpra_normal_form == FORM
        # `misses` and the form are not keys of the mapping view: no digest sees them.
        assert "misses" not in found and "cpra_normal_form" not in found
        empty = q.lookup([planted.absent_snv], ids)
        assert empty.cpra_normal_form == FORM


def test_a_0_3_release_never_misses_and_carries_its_derived_form(pair):
    old, _ = pair
    with _query(old) as q:
        ids = _ids(q)
        planted_alid = _alids(old)[0]
        found = q.lookup([planted_alid, "22:49999999:A:G", "22:49999999:A:AT"], ids)
        assert found.misses == ()
        assert found.cpra_normal_form is not None and found.cpra_normal_form.signed_for == "a1"


def test_dense_component_axis_resolves_misses_in_panel_rows(built_0_4):
    """A Hybrid's Dense Component reads the shared axis: a miss names the shared row."""
    new = built_0_4("hybrid_observed")
    old = RELEASES / "hybrid_observed"
    planted = Planted(old, new)
    component = open_release(new).dense_component()
    axis = component.variant_axis(aliases=False)
    found = axis.resolve_identifiers([planted.swapped, planted.snv])
    (miss,) = found.misses
    assert miss.cpra == planted.snv and miss.variant_index == planted.snv_row
    assert axis.cpra_normal_form == FORM and axis.keys_are_cpra
    assert axis.identity_columns[-1] == "cpra" and axis.source_column == "source_cpra"
    # The SNV's row is a panel row or off the panel; either way the answer is the panel's.
    shared = open_release(new).variant_axis(aliases=False)
    on_panel = shared.index_of(planted.snv)
    assert on_panel == planted.snv_row
    assert len(found.indices) in (0, 1)



# --- review round 1 -----------------------------------------------------------------


def test_phewas_keeps_the_misses_lookup_gives(pair):
    """§4.0: two misses are never bare, on every query that names a variant."""
    old, new = pair
    planted = Planted(old, new)
    with _query(new) as q:
        swapped = q.phewas(planted.swapped)
        assert swapped.count == 0
        (miss,) = swapped.misses
        assert (miss.reason, miss.cpra, miss.variant_index) == (
            "swapped_alleles", planted.snv, planted.snv_row,
        )
        for absent in (planted.absent_indel, planted.reversed_indel):
            (miss,) = q.phewas(absent).misses
            assert (miss.reason, miss.cpra, miss.variant_index) == ("indel_absent", None, None)
        # A plain absent SNV, and a held variant, are bare and give a row.
        assert q.phewas(planted.absent_snv).misses == ()
        held = q.phewas(planted.snv)
        assert held.count > 0 and held.misses == ()
        assert held.cpra_normal_form == FORM


@pytest.mark.parametrize(
    "asked",
    [
        "1:10:A", "1:10:A:G:T", "1:10:A:", "1:10::G", ":10:A:G", "1:10", "chr",
        "not-an-rsid", "rs", "rs12x", "RS12", "",
    ],
)
def test_an_identifier_that_is_neither_a_cpra_nor_an_rsid_is_refused(observed, asked):
    """§4.0: an rsid is `rs` and digits, recognised positively; anything else is a malformed
    CPRA and never reaches the rsid path."""
    with _query(observed) as q:
        with pytest.raises(MalformedIdentity):
            q.lookup([asked], _ids(q))
        with pytest.raises(MalformedIdentity):
            q.phewas(asked)


def test_an_rsid_still_reaches_the_rsid_path(pair):
    _, new = pair
    with _query(new) as q:
        for rsid in ("rs1", "rs12345678901"):
            with pytest.raises(RsidIndexNotYetSupported):
                q.lookup([rsid], _ids(q))


def _legacy_table(path: Path, *, aliases: bool) -> Any:
    return GzipVariantTable(path, aliases=aliases)


def test_the_table_rows_of_a_0_4_release_are_decoded_by_its_series(built_0_4):
    """§21.5: with a table reader injected, `written_table_records` and `tabix_range` give
    `ref`, `alt`, `cpra` and `source_cpra`, never the 0.3 names."""
    new = built_0_4("dense_ranges")
    old = RELEASES / "dense_ranges"
    release = open_release(new, legacy=_legacy_table)
    axis = release.variant_axis(aliases=False)
    written = list(axis.written_table_records())
    expected = cpras(new)
    assert [r.cpra for r in written] == expected
    assert [r.variant_index for r in written] == list(range(len(expected)))
    for record in written:
        assert list(record) == [
            "variant_index", "chromosome", "position", "ref", "alt", "cpra", "rsid", "source_cpra",
        ]
        for old_name in ("effect_allele", "other_allele", "alid", "source_alid"):
            assert old_name not in record
        assert record.cpra == f"{record.chromosome}:{record.position}:{record.ref}:{record.alt}"
    by_cpra = {r.cpra: r for r in written}
    assert by_cpra["1:5000:T:C"].source_cpra == "1:5000:T:C"
    assert by_cpra["1:5000:T:C"].rsid is None
    assert by_cpra["1:1000:G:A"].rsid == "rs1000"
    ranged = axis.tabix_range("1", 5000, 5000)
    assert [r.cpra for r in ranged] == [c for c in expected if c.startswith("1:5000:")]
    assert all(r.alt and r.ref for r in ranged)
    assert _alids(old)[0] == "1:1000:A:G"


def test_a_0_3_table_still_decodes_to_0_3_names():
    release = open_release(RELEASES / "dense_ranges", legacy=_legacy_table)
    axis = release.variant_axis(aliases=False)
    first = next(iter(axis.written_table_records()))
    assert first.alid == "1:1000:A:G" and first.effect_allele == "A"


def test_a_0_4_table_row_with_0_3_columns_is_refused(built_0_4, tmp_path):
    new = Path(shutil.copytree(built_0_4("dense_ranges"), tmp_path / "bad"))
    import gzip

    with gzip.open(new / "variants.tsv.gz", "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    lines[1] = "\t".join(lines[1].split("\t")[:7])
    with gzip.open(new / "variants.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    axis = open_release(new, legacy=_legacy_table).variant_axis(aliases=False)
    with pytest.raises(ValueError, match="8 fields"):
        list(axis.written_table_records())


def test_a_dense_component_names_its_own_rows(built_0_4):
    """A hit's or a miss's `variant_index` is the index in the axis asked. hybrid_observed's
    Dense Component holds 200 of the shared axis's 400 rows: the swapped spelling of shared
    row 200, which is off the panel, is no miss there and never reports index 200."""
    new = built_0_4("hybrid_observed")
    old = RELEASES / "hybrid_observed"
    planted = Planted(old, new)
    component = open_release(new).dense_component()
    axis = component.variant_axis(aliases=False)
    dense_to_shared = np.load(new / "dense" / "dense_to_shared.npy")
    assert dense_to_shared.size == 200 and 200 not in dense_to_shared.tolist()
    off = planted.cpras[200]
    chrom, pos, ref, alt = off.split(":")
    assert len(ref) == len(alt) == 1, "shared row 200 is an SNV"
    swapped = f"{chrom}:{pos}:{alt}:{ref}"
    assert swapped not in planted.cpras
    found = axis.resolve_identifiers([swapped, off])
    assert found.indices.tolist() == [] and found.misses == ()
    # An on-panel SNV: the hit is its panel row, the miss names it too.
    on = planted.cpras[7]
    assert dense_to_shared[7] == 7
    assert axis.resolve_identifiers([on]).indices.tolist() == [7]


def test_a_dense_component_panel_index_differs_from_the_shared_one(built_0_4):
    new = built_0_4("hybrid_completed")
    old = RELEASES / "hybrid_completed"
    planted = Planted(old, new)
    dense_to_shared = np.load(new / "dense" / "dense_to_shared.npy")
    assert dense_to_shared.size == 201 and int(dense_to_shared[200]) == 250
    shared_cpra = planted.cpras[250]
    chrom, pos, ref, alt = shared_cpra.split(":")
    assert len(ref) == len(alt) == 1
    swapped = f"{chrom}:{pos}:{alt}:{ref}"
    assert swapped not in planted.cpras
    axis = open_release(new).dense_component().variant_axis(aliases=False)
    found = axis.resolve_identifiers([swapped, shared_cpra])
    assert found.indices.tolist() == [200], "the hit is panel row 200, not shared row 250"
    (miss,) = found.misses
    assert (miss.reason, miss.cpra, miss.variant_index) == ("swapped_alleles", shared_cpra, 200)
    # The shared axis names the same variant by its own index.
    shared = open_release(new).variant_axis(aliases=False)
    assert shared.resolve_identifiers([swapped]).misses[0].variant_index == 250


# --- §5: statistics are signed for ALT ------------------------------------------------


#: dense_ranges' eight rows as 0.3.0 stores them (A1-signed): z -6 (SE infinite), 7, 8, -9, -6,
#: 7, 7, -8 and EAF .8 .3 .75 .65 .4 .2 .7 .55. Their 0.4.0 keys, and whether each is re-signed:
#: row 1 `1:5000:A:C` and row 2 `1:5000:C:G` keep their alleles but are now signed for ALT (C, G),
#: not for A1 (A, C); row 4 is a reference-first indel `1:9000:A:AT` signed for ALT, not REF; the
#: rest are SNVs spelt REF:ALT with ALT the old A1, whose values stand.
RESIGNED_KEYS = [
    "1:1000:G:A", "1:5000:A:C", "1:5000:C:G", "1:5000:T:C",
    "1:9000:A:AT", "2:300:T:C", "2:7000:C:A", "X:2500:T:G",
]
RESIGNED_ROWS = {1, 2, 4}
#: What a 0.4.0 reader gives back, worked by hand: z negated and EAF complemented on the
#: re-signed rows, every other value as 0.3.0 had it.
ALT_SIGNED_Z = [-6.0, -7.0, -8.0, -9.0, 6.0, 7.0, 7.0, -8.0]
ALT_SIGNED_EAF = [0.8, 0.7, 0.25, 0.65, 0.6, 0.2, 0.7, 0.55]


@pytest.fixture(scope="module")
def resigned(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("resigned")
    path = Path(shutil.copytree(RELEASES / "dense_ranges", root / "dense_ranges"))
    return as_0_4(path, keys=RESIGNED_KEYS, resign=RESIGNED_ROWS)


def test_the_re_signed_rows_are_planted(resigned):
    assert cpras(resigned) == RESIGNED_KEYS
    old_z = [-6.0, 7.0, 8.0, -9.0, -6.0, 7.0, 7.0, -8.0]
    for row in range(8):
        flipped = ALT_SIGNED_Z[row] == -old_z[row]
        assert flipped == (row in RESIGNED_ROWS), f"row {row}"
    # ALT differs from the old A1 on the re-signed SNVs (old A1: A, then C).
    assert RESIGNED_KEYS[1].split(":")[3] != "A" and RESIGNED_KEYS[2].split(":")[3] != "C"
    # Row 4 is a reference-first indel: REF is its old A1, and its ALT is what it is signed for.
    assert len(RESIGNED_KEYS[4].split(":")[2]) < len(RESIGNED_KEYS[4].split(":")[3])


def test_a_0_4_reader_gives_back_the_alt_signed_values(resigned):
    with open_release(resigned).query() as q:
        ids = _ids(q)
        found = q.lookup(RESIGNED_KEYS, ids)
        by_row = {int(v): i for i, v in enumerate(found["variant_index"])}
        assert sorted(by_row) == list(range(8))
        for row in range(8):
            i = by_row[row]
            assert float(found["z"][i]) == ALT_SIGNED_Z[row]
            assert float(found["eaf"][i]) == pytest.approx(ALT_SIGNED_EAF[row], abs=1e-6)
        # beta is z * se, so it is signed for ALT as z is: the re-signed rows' betas negate.
        assert float(found["beta"][by_row[1]]) == pytest.approx(-0.07001495, abs=1e-6)
        assert float(found["beta"][by_row[4]]) == pytest.approx(0.06001282, abs=1e-6)
        # The reference-first indel is signed for its ALT, the insertion.
        indel = q.phewas("1:9000:A:AT")
        assert indel.count == 1 and float(indel["z"][0]) == 6.0
        assert float(indel["eaf"][0]) == pytest.approx(0.6, abs=1e-6)
        assert float(q.phewas("1:5000:A:C")["z"][0]) == -7.0


def test_a_direct_dense_component_query_stamps_the_roots_form(built_0_4):
    """§2: the Dense Component's form equals the root's, so a query of it stamps that form."""
    root = built_0_4("hybrid_observed")
    with open_release(root / "dense").query() as q:
        found = q.lookup(["1:1000:G:A"], _ids(q))
        assert found.cpra_normal_form == open_release(root).cpra_normal_form == FORM
