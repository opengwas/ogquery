"""The Test Panel is what its manifest and README say it is (tests/ld/testdata.py; ADR 0088).

Expected values come from outside the loader: the SHA-256s from manifest.json, plink2's own r,
plink2's `--export A` dosages recomputed here by a different route (`numpy.linalg.lstsq`
residuals), and literal rows and positions copied from tests/ld/data/test-panel/README.md's
edge-case table. Nothing here recomputes an answer the way scripts/ld/make_test_panel.py does.

From opengwasdb's `tests/ld/test_testdata.py` at `c8ebe4ca` (ogquery#2). ogquery holds the Test
Panel's answers but not its genotypes, PC File or dosages (`data/test-panel/SOURCE`), so the
tests that read those stay there.
"""

from __future__ import annotations

import hashlib
import math
import pathlib

import numpy as np
import pytest
import testdata as td
from store_copies import t_far

HALF_STEP = 1 / 65534  # half of one int16 step: the quantisation tolerance (ld-store.md §3)
T_FAR_K10 = t_far(771, 10)  # t_far for 771 samples at k = 10 (ADR 0087, decision 3)
SCALE = 32767

# Literals from tests/ld/data/test-panel/README.md, "The edge cases, and where they are".
EXACT = {  # distance -> (row_i, row_j, pos_i, pos_j)
    3_000_000: (478, 491, 16_190_010, 19_190_010),
    3_000_001: (472, 490, 16_022_568, 19_022_569),
    10_000_000: (475, 821, 16_029_456, 26_029_456),
    10_000_001: (479, 823, 16_231_175, 26_231_176),
}
FAR_PAIRS_K10 = [(6, 39), (35, 216), (44, 221), (848, 1032), (854, 1020), (856, 1025)]
BOUND_PAIR = (0, 11)
BEYOND_10MB = (0, 36)
CROSS_CHROMOSOME = (0, 471)
MULTIALLELIC = (476, 477, 16_109_998)  # row_a, row_b, position
LONG_INDEL = (806, 183, 1)  # row, REF length, ALT length
FAILS_MAC = {468: 0, 469: 5, 470: 1, 473: 19, 474: 10}  # row -> MAC
N_VARIANTS = 1061
N_PASSING = 1056
N_PAIRS = 146_593
CAUSAL = [287, 353, 421, 537, 612, 693, 774, 798]
LEADS = {
    0.01: [612, 421, 774, 537, 353, 798, 287, 730],
    0.001: [612, 421, 774, 353, 287, 703],
}
CLUMP_P1 = 5e-8


def test_every_file_matches_its_manifest_hash() -> None:
    # The Test Stores under store/ and store-k0/ are not the Test Panel:
    # scripts/ld/make_test_store.py writes them (store-k0/: ogquery#2, data/SOURCE).
    files = td.manifest()["files"]
    on_disk = sorted(
        p.relative_to(td.TEST_PANEL).as_posix()
        for p in td.TEST_PANEL.rglob("*")
        if p.is_file()
        and p.name != "manifest.json"
        and p.relative_to(td.TEST_PANEL).parts[0] not in ("store", "store-k0")
    )
    # ogquery holds the files its tests read, not the genotypes, PCs and dosages a build or a
    # recomputation reads (tests/ld/data/test-panel/SOURCE): every one here is listed.
    assert len(on_disk) >= 8, "the fixture must hold the Test Panel's files"
    assert set(on_disk) <= set(files), "every file here is listed"
    for name in on_disk:
        data = (td.TEST_PANEL / name).read_bytes()
        assert len(data) == files[name]["bytes"], name
        assert hashlib.sha256(data).hexdigest() == files[name]["sha256"], name


def test_the_panel_is_within_its_size_budget() -> None:
    total = sum(p.stat().st_size for p in td.TEST_PANEL.rglob("*") if p.is_file())
    assert 2_000_000 < total <= 5_000_000, "the whole directory, manifest included"


def test_variants_are_in_pvar_order_with_their_filters() -> None:
    v = td.variants()
    assert len(v) == N_VARIANTS
    assert int(v.passes_filters.sum()) == N_PASSING
    assert set(v.chrom) == {"21", "22"}
    for c in ("21", "22"):
        assert (np.diff(v.pos[v.chrom == c]) >= 0).all(), "positions ascend within a chromosome"
    assert all(
        i == f"{c}:{p}:{r}:{a}"
        for i, c, p, r, a in zip(v.id, v.chrom, v.pos, v.ref, v.alt, strict=True)
    ), "IDs are chrom:pos:REF:ALT"
    assert (v.drop_reason[v.passes_filters] == "").all()
    assert (v.drop_reason[~v.passes_filters] != "").all()
    with pytest.raises(ValueError, match="read-only"):
        v.pos[0] = 1


def test_reference_pairs_are_every_close_pair_of_passing_rows() -> None:
    v = td.variants()
    for k in td.KS:
        ref = td.reference_r(k)
        assert len(ref) == N_PAIRS
        assert (ref.row_i < ref.row_j).all()
        assert v.passes_filters[ref.row_i].all()
        assert v.passes_filters[ref.row_j].all()
        assert (v.chrom[ref.row_i] == v.chrom[ref.row_j]).all()
        d = v.pos[ref.row_j] - v.pos[ref.row_i]
        assert (d >= 0).all()
        assert (d <= 10_000_000).all()
        # A perfectly correlated pair can come out a few ulps over 1 in float64.
        assert np.isfinite(ref.r).all()
        assert (np.abs(ref.r) <= 1 + 1e-12).all()
    with pytest.raises(ValueError, match="k = 5"):
        td.reference_r(5)


def test_reference_r_at_k0_equals_plink2_s() -> None:
    ref, plink2 = td.reference_r(0), td.plink2_r()
    assert np.array_equal(ref.row_i, plink2.row_i)
    assert np.array_equal(ref.row_j, plink2.row_j)
    strong = np.abs(plink2.r) > 0.9
    assert strong.sum() > 100, "the fixture must hold strong LD for this to mean anything"
    # Complete genotypes: numpy's Pearson r and plink2's agree to rounding (ADR 0088), far
    # inside the quantisation step the builder is held to.
    assert np.max(np.abs(ref.r - plink2.r)) <= HALF_STEP
    assert np.max(np.abs(ref.r - plink2.r)) < 1e-12


def _pair_index(i: int, j: int) -> int | None:
    ref = td.reference_r(0)
    hit = np.flatnonzero((ref.row_i == i) & (ref.row_j == j))
    return int(hit[0]) if len(hit) else None


@pytest.mark.parametrize("distance", sorted(EXACT))
def test_pairs_at_the_band_edges(distance: int) -> None:
    i, j, pi, pj = EXACT[distance]
    v = td.variants()
    assert (v.pos[i], v.pos[j]) == (pi, pj)
    assert pj - pi == distance
    assert v.passes_filters[i]
    assert v.passes_filters[j]
    assert v.chrom[i] == v.chrom[j]
    assert (_pair_index(i, j) is not None) == (distance <= 10_000_000)


def _stored_r2(r: float) -> float:
    return (round(r * SCALE) / SCALE) ** 2


def test_pairs_the_reference_does_not_hold() -> None:
    v = td.variants()
    i, j = BEYOND_10MB
    assert v.chrom[i] == v.chrom[j]
    assert v.pos[j] - v.pos[i] > 10_000_000
    a, b = CROSS_CHROMOSOME
    assert v.chrom[a] != v.chrom[b]
    for x, y in (BEYOND_10MB, CROSS_CHROMOSOME):
        assert v.passes_filters[x]
        assert v.passes_filters[y]
        assert _pair_index(x, y) is None


def test_a_split_multiallelic_site_is_two_rows_at_one_position() -> None:
    v = td.variants()
    a, b, pos = MULTIALLELIC
    assert v.pos[a] == v.pos[b] == pos
    assert v.ref[a] == v.ref[b]
    assert v.alt[a] != v.alt[b]
    assert v.passes_filters[a]
    assert v.passes_filters[b]
    assert _pair_index(a, b) is not None, "a pair at distance 0 is an ordinary pair"


def test_an_indel_whose_alid_is_longer_than_32_bytes() -> None:
    v = td.variants()
    row, ref_len, alt_len = LONG_INDEL
    assert (len(v.ref[row]), len(v.alt[row])) == (ref_len, alt_len)
    assert len(f"{v.chrom[row]}:{v.pos[row]}:{v.ref[row]}:{v.alt[row]}".encode()) > 32
    assert v.passes_filters[row]


def test_rows_below_mac_20_fail_the_filters() -> None:
    v = td.variants()
    for row in FAILS_MAC:
        assert not v.passes_filters[row]
        assert v.drop_reason[row] == "mac"


def test_gwas_z_is_nan_exactly_where_a_row_fails() -> None:
    v, z = td.variants(), td.gwas().z
    assert len(z) == N_VARIANTS
    assert (np.isnan(z) == ~v.passes_filters).all()
    for row in CAUSAL:
        assert abs(z[row]) > 5.45, "each causal variant reaches p < 5e-8"


def test_clump_leads_are_plink2_s() -> None:
    v, z = td.variants(), td.gwas().z
    params = td.clump_params()
    assert (params.p1, params.p2, params.kb, params.r2) == (CLUMP_P1, CLUMP_P1, 3000, (0.01, 0.001))
    for r2, leads in LEADS.items():
        assert td.clump_leads(r2) == tuple(leads)
        for row in leads:
            assert v.passes_filters[row]
            assert math.erfc(abs(z[row]) / math.sqrt(2)) <= params.p1
    assert len(LEADS[0.01]) > len(LEADS[0.001]) >= 2, "the thresholds must give different leads"
    with pytest.raises(ValueError, match=r"0\.05"):
        td.clump_leads(0.05)


def test_cached_results_cannot_be_changed_by_a_test() -> None:
    with pytest.raises(TypeError):
        td.manifest()["files"] = {}
    with pytest.raises(TypeError):
        td.manifest()["pc_check"]["covariates"][0] = "PC20"
    with pytest.raises(AttributeError):
        td.clump_leads(0.01).sort()


def test_the_manifest_names_no_host_path() -> None:
    text = (td.TEST_PANEL / "manifest.json").read_text(encoding="utf-8")
    for host in ("/home/", "/data/opengwasdb", "scratchpad", "/tmp/"):
        assert host not in text, host


def test_the_manifest_check_holds_where_paths_are_spelt_with_backslashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The manifest names files with `/`; a Windows checkout's relative paths use `\` (CI run
    # 45664832290), so the check compares POSIX spellings (opengwas/ogquery#2).
    original = pathlib.Path.relative_to

    def windows(self: pathlib.Path, *other: object, **kw: object) -> pathlib.PureWindowsPath:
        return pathlib.PureWindowsPath(*original(self, *other, **kw).parts)  # type: ignore[arg-type]

    monkeypatch.setattr(pathlib.Path, "relative_to", windows)
    assert str((td.TEST_PANEL / "plink2" / "x").relative_to(td.TEST_PANEL)) == "plink2\\x"
    test_every_file_matches_its_manifest_hash()
