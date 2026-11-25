"""ogld#6's contract: proxies and clumping on the Test Store, with the refusals of ld-store §9.

Expected proxies come from the Test Panel's reference r, quantised and judged on r̂² as the
spec says.

From opengwasdb's `tests/ld/contract/test_query.py` at `c8ebe4ca` (ogquery#2). Its two
comparisons with plink2 -- the clump leads and the proxies at k = 0 -- read a k = 0 store that
opengwasdb's `scripts/ld/make_test_store.py` writes with its build code, so they stay there
(ADR 0002). The clumping invariants that read it here read the committed k = 10 Test Store:
they hold at any k.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import support
from support import alids, panel_data, passing_rows, reference

import ogquery.ld
from ogquery.ld.format import RefusalReason

#: plink2's unphased leads at r² 0.001 on the Test Panel (tests/ld/data/test-panel/README.md) are
#: held there, but a 0.2.0 store refuses r² 0.001 within 3 Mb: its t_partner is 0.01, so the
#: comparison that used them is answered at r² 0.01 only (ld-store §9; the refusal is tested below).

#: The band the plink2 comparison lists (ld-store §3; ogld#16): candidate pairs (p ≤ p1, under
#: 3,000 kb, k = 0) whose true r², plink2's, and whose r̂² at the Test Store's × 511 lie on two
#: sides of r2. Each is within 2|r| × half a step of it: true r² 0.00101–0.00104, read as code 16,
#: r̂² 0.00098. None moves a lead. At × 32767 there were none.
CROSSING = {
    0.01: set(),
    0.001: {(538, 613), (538, 703), (693, 798), (707, 798), (719, 798)},
}


@cache
def proxy_pairs(r2: float, window: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (query row, proxy row, r̂) with r̂² ≥ r2 within `window` bp, both directions."""
    ref = reference(support.K)
    r_hat = ref.code.astype(np.float64) / support.R_SCALE
    keep = (r_hat * r_hat >= r2) & (ref.distance <= window)
    query = np.concatenate([ref.row_i[keep], ref.row_j[keep]])
    proxy = np.concatenate([ref.row_j[keep], ref.row_i[keep]])
    return query, proxy, np.concatenate([r_hat[keep], r_hat[keep]])


def expected_proxies(query: int, r2: float, window: float) -> dict[int, float]:
    """Each partner of Test Panel row `query` with r̂² ≥ r2 within `window` bp: row → r."""
    queries, proxies, r = proxy_pairs(r2, window)
    mine = queries == query
    return dict(zip(proxies[mine].tolist(), r[mine].tolist(), strict=True))


def busiest_query(r2: float, window: float) -> int:
    """The passing row with the most proxies, so the test has something to compare."""
    queries, _, _ = proxy_pairs(r2, window)
    counts = np.bincount(queries, minlength=int(passing_rows().max()) + 1)
    assert counts.max() > 1, "the Test Panel must hold a cluster with proxies at this r²"
    return int(np.argmax(counts))


def test_proxies_default_to_r2_0_8_within_10_mb() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    query = busiest_query(0.8, 10_000_000)
    found = store.proxies(alids([query]))
    assert (found.r2_min, found.window_bp, found.t_far) == (0.8, 10_000_000, support.T_FAR)
    expected = expected_proxies(query, 0.8, 10_000_000)
    assert sorted(found.proxy) == sorted(alids(list(expected)))
    assert (found.r2 >= 0.8).all()
    assert (found.distance <= 10_000_000).all()


def test_proxies_return_signed_r() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    rows = passing_rows()
    sample = rows[:: max(1, len(rows) // 100)]
    found = store.proxies(alids(sample), r2=0.5, kb=3000)
    queries, proxies, r = proxy_pairs(0.5, 3_000_000)
    mine = np.isin(queries, sample)
    names = zip(alids(queries[mine]), alids(proxies[mine]), strict=True)
    expected = dict(zip(names, r[mine].tolist(), strict=True))
    assert any(r < 0 for r in expected.values()), "the fixture must hold negative r"
    assert any(r > 0 for r in expected.values()), "the fixture must hold positive r"
    got = {(q, p): r for q, p, r in zip(found.query, found.proxy, found.r, strict=True)}
    assert got == expected
    np.testing.assert_array_equal(found.r2, found.r * found.r)


def test_proxies_are_ordered_and_never_the_query_itself() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    query = busiest_query(0.5, 3_000_000)
    found = store.proxies(alids([query]), r2=0.5, kb=3000)
    assert alids([query])[0] not in set(found.proxy)
    keys = list(zip(-found.r2, found.distance, found.proxy, strict=True))
    assert keys == sorted(keys), "rows go by r² descending, then distance, then ALID"


def test_proxies_below_t_far_within_3_mb_read_the_near_band() -> None:
    # r2 = 0.01 < t_far is answered within 3 Mb: every pair there is stored (ld-store §9).
    store = ogquery.ld.open(support.TEST_STORE)
    query = busiest_query(0.01, 3_000_000)
    found = store.proxies(alids([query]), r2=0.01, kb=3000)
    expected = expected_proxies(query, 0.01, 3_000_000)
    assert len(expected) > len(expected_proxies(query, support.T_FAR, 3_000_000)), (
        "the fixture must hold pairs below t_far, or this reads nothing the index lacks"
    )
    assert sorted(found.proxy) == sorted(alids(list(expected)))


def test_proxies_list_absent_queries() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    found = store.proxies(["not-an-alid", *alids(passing_rows()[:1])])
    assert found.absent == ("not-an-alid",)


@pytest.mark.parametrize(
    ("r2", "kb", "reason"),
    [
        (0.01, 5_000, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.001, 3_000, RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        (0.8, 10_001, RefusalReason.WINDOW_OVER_FAR),
        (1.5, 1_000, RefusalReason.INVALID),
        (0.5, -1, RefusalReason.INVALID),
    ],
)
def test_each_proxies_refusal(r2: float, kb: float, reason: RefusalReason) -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    with pytest.raises(ogquery.ld.Refusal) as refused:
        store.proxies(alids(passing_rows()[:1]), r2=r2, kb=kb)
    assert refused.value.reason is reason


# ------------------------------------------------------------------ clump


def p_value(z: np.ndarray) -> np.ndarray:
    return np.array([math.erfc(abs(x) / math.sqrt(2)) for x in z])


def candidates(p_max: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The passing rows at p ≤ p_max: ALIDs, |z| and rows."""
    z = panel_data().gwas().z
    rows = passing_rows()
    keep = rows[p_value(z[rows]) <= p_max]
    assert len(keep) > 2, f"the simulated GWAS must have hits at p ≤ {p_max}"
    return alids(keep), np.abs(z[keep]), keep


@pytest.mark.parametrize("r2", [0.01, 0.001])
def test_the_pairs_the_store_s_step_moves_across_r2_are_listed_and_within_it(r2: float) -> None:
    # Reference r and the store's own step, with no query: the pairs are listed at both
    # thresholds, though a 0.2.0 store answers only the first (ld-store §9).
    # Every pair of the Test Panel the step moves across r2 lies within 2|r| × half a step of it;
    # among the plink2 comparison's candidates they are exactly CROSSING's.
    ogquery.ld.open(support.TEST_STORE)
    ref = reference(0)
    params = panel_data().clump_params()
    *_, rows = candidates(params.p1)
    window = ref.distance < params.kb * 1000
    crossing = window & ((ref.r**2 > r2) != (support.stored_at(ref.r) ** 2 > r2))
    assert crossing.sum() > 100, "the step must move many pairs for this to mean anything"
    band = np.abs(ref.r**2 - r2) <= 2 * np.abs(ref.r) * support.HALF_STEP + support.SLACK
    assert band[crossing].all(), "a pair moved across r2 lies within the step of it"
    among = crossing & np.isin(ref.row_i, rows) & np.isin(ref.row_j, rows)
    found = zip(ref.row_i[among].tolist(), ref.row_j[among].tolist(), strict=True)
    assert set(found) == CROSSING[r2]


def test_every_member_reaches_its_lead_and_every_lead_leads_itself() -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    names, score, _ = candidates(1e-4)
    clumps = store.clump(names, score, r2=0.01, kb=3000)
    assert list(clumps.alid) == names
    is_lead = clumps.lead == clumps.alid
    assert is_lead.any()
    assert (~is_lead).any(), "the fixture must clump some variants into others"
    assert (clumps.r[is_lead] == 1.0).all()
    assert (clumps.r[~is_lead] ** 2 >= 0.01).all()
    lead_score = dict(zip(clumps.alid, clumps.score, strict=True))
    assert all(lead_score[lead] >= s for lead, s in zip(clumps.lead, clumps.score, strict=True))


def test_a_stricter_threshold_gives_a_prefix_of_the_leads() -> None:
    # ogld#6: the leads at p ≤ p1 (5e-8) are the leads at p ≤ 1e-4 with p ≤ p1, so the engine
    # can store one clumping per configuration.
    store = ogquery.ld.open(support.TEST_STORE)
    loose_names, loose_score, _ = candidates(1e-4)
    strict_names, strict_score, _ = candidates(panel_data().clump_params().p1)
    assert len(loose_names) > len(strict_names), "the looser set must add variants to test"
    loose = store.clump(loose_names, loose_score, r2=0.01, kb=3000).leads()
    strict = store.clump(strict_names, strict_score, r2=0.01, kb=3000).leads()
    assert list(strict) == [lead for lead in loose if lead in set(strict_names)]


@pytest.mark.parametrize(
    ("r2", "kb", "reason"),
    [
        (0.001, 10_000, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.001, 3_000, RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        (0.01, 3_000.001, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.1, 10_000.001, RefusalReason.WINDOW_OVER_FAR),
        (-0.1, 1_000, RefusalReason.INVALID),
    ],
)
def test_each_clump_refusal(r2: float, kb: float, reason: RefusalReason) -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    names = alids(passing_rows()[:3])
    with pytest.raises(ogquery.ld.Refusal) as refused:
        store.clump(names, np.array([6.0, 5.0, 4.0]), r2=r2, kb=kb)
    assert refused.value.reason is reason


@pytest.mark.parametrize("score", [[6.0, math.nan, 4.0], [6.0, -1.0, 4.0], [6.0, 5.0]])
def test_a_score_that_is_not_one_per_alid_finite_and_non_negative_is_invalid(
    score: list[float],
) -> None:
    store = ogquery.ld.open(support.TEST_STORE)
    with pytest.raises(ogquery.ld.Refusal) as refused:
        store.clump(alids(passing_rows()[:3]), np.array(score), r2=0.1, kb=1000)
    assert refused.value.reason is RefusalReason.INVALID
