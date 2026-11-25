"""The semantics core: how r is stored, which pairs a store keeps, and which requests it answers.

Every expected value is independent of the code under test. They are literals from the
spec's tables (`docs/spec/ld-store.md` §3, §8, §9), from the epic's t_far table, and from
the walkthroughs of `proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html`,
which computed them in JavaScript. None is recomputed the way `ogquery.ld.format` computes it.
Hypothesis holds the properties: sign symmetry, and monotonicity at the boundaries.

From opengwasdb's `tests/ld/test_format.py` at `c8ebe4ca` (ogquery#2): the reader's part. The tests
of quantising, the step, the far code floor and t_far, which only a build or validation computes,
stay there with them. t_far and a code are computed here as `store_copies` writes ADR 0087 and
ld-store §3 out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from store_copies import codes_at, t_far

import ogquery.ld.format as fmt

#: The builder's distances (opengwasdb ADR 0087), which a reader never uses: it reads the ones a
#: store records. They stay in opengwasdb's format (ogquery#2); these tests judge pairs at them.
NEAR_BP = 3_000_000
FAR_BP = 10_000_000

# r in the closed interval the format stores.
R = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
# t_far's range for any panel the formula admits: the 0.001 floor up to just below 1.
T = st.floats(min_value=0.001, max_value=0.999, allow_nan=False)


# ------------------------------------------------------------------ §3 the encoding

#: The scales the decisions name: HGDP+1kGP's, UK Biobank EUR's and the container's widest
#: (owner, 10 November 2026; ADR 0090).
SCALES = [511, 2047, 32767]


def test_every_encoding_shares_the_spec_s_container() -> None:
    assert fmt.REncoding.dtype == np.dtype("int16")
    assert fmt.REncoding.reserved_code == -32768
    assert fmt.REncoding.rounding == "half_to_even"


@pytest.mark.parametrize("scale", [1, 511, 2047, 32767])
def test_any_whole_scale_from_1_to_32767_is_an_encoding(scale: int) -> None:
    assert fmt.REncoding(scale).scale == scale


@pytest.mark.parametrize("scale", [0, -511, 32768, 511.0, True, "511", None])
def test_a_scale_the_container_cannot_hold_is_refused_naming_it(scale: object) -> None:
    with pytest.raises(fmt.InputError, match=f"not {scale!r}"):
        fmt.REncoding(scale)


@pytest.mark.parametrize(("scale", "past"), [(511, 512), (511, -32767), (2047, 2048), (1, 2)])
def test_a_code_past_the_scale_is_never_read(scale: int, past: int) -> None:
    # In the int16 container a × 511 store's 512 is a number, but r̂ = 512 / 511 is no r.
    encoding = fmt.REncoding(scale)
    assert not encoding.holds(np.array([past]))[0]
    with pytest.raises(ValueError, match=str(past)):
        encoding.dequantise(np.array([0, past], dtype=np.int16))


def test_dequantise_returns_float64() -> None:
    out = fmt.REncoding(511).dequantise(np.array([511, -511, 0, 256], dtype=np.int16))
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, -1.0, 0.0, 256 / 511]


# ------------------------------------------------------------------ §3, §7 the record

#: ld-store §7's `encoding` of the HGDP+1kGP stores.
HGDP = {"r_dtype": "int16", "r_scale": 511, "r_reserved_code": -32768, "rounding": "half_to_even"}


UNIMPLEMENTED: dict[str, object] = {
    "an int8 container": {**HGDP, "r_dtype": "int8"},
    "another reserved code": {**HGDP, "r_reserved_code": 0},
    "rounding half away from zero": {**HGDP, "rounding": "half_away_from_zero"},
    "a scale past the container": {**HGDP, "r_scale": 32768},
    "a scale of 0": {**HGDP, "r_scale": 0},
    "a fractional scale": {**HGDP, "r_scale": 511.5},
    "a scale written as text": {**HGDP, "r_scale": "511"},
    "a key ogld does not know": {**HGDP, "r_offset": 0},
    "no scale": {k: v for k, v in HGDP.items() if k != "r_scale"},
    "no record": None,
}


@pytest.mark.parametrize("case", list(UNIMPLEMENTED))
def test_an_encoding_ogld_does_not_implement_is_refused_naming_it(case: str) -> None:
    record = UNIMPLEMENTED[case]
    with pytest.raises(fmt.InputError) as refused:
        fmt.REncoding.of(record)
    assert repr(record) in str(refused.value), refused.value


# ------------------------------------------------------------------ §8 pair answers

#: t_far at k = 10 on the samples the EUR and AMR builds keep (ADR 0088, decision 11).
EUR_K10 = t_far(665, 10)
AMR_K10 = t_far(341, 10)
#: The HGDP+1kGP stores' encoding, in which §8's walkthroughs read r (ADR 0090).
HGDP_R = fmt.REncoding(511)


def stored(r: float) -> float:
    """r as a × 511 store holds it: its code, as `store_copies.codes_at` writes ld-store §3's rule
    out, read back by the encoding."""
    return float(HGDP_R.dequantise(codes_at(r, 511).astype(np.int16)))


def kind(distance: int, r_hat: float, tf: float = EUR_K10, *, same: bool = True) -> fmt.AnswerKind:
    return fmt.answer_kind(
        same_chromosome=same,
        distance=distance,
        r_hat=r_hat,
        t_far=tf,
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
    )


def test_exactly_3_mb_is_the_near_band_and_one_more_base_is_not() -> None:
    # The walkthrough "Exactly 3 Mb", EUR, k = 10.
    weak = stored(0.05)
    assert kind(3_000_000, weak) is fmt.AnswerKind.STORED
    assert kind(3_000_001, weak) is fmt.AnswerKind.BOUND
    assert kind(3_000_001, stored(0.30)) is fmt.AnswerKind.STORED


def test_exactly_10_mb_is_measured_and_one_more_base_is_not() -> None:
    strong = stored(0.30)
    assert kind(10_000_000, strong) is fmt.AnswerKind.STORED
    assert kind(10_000_001, strong) is fmt.AnswerKind.NOT_MEASURED
    assert kind(10_000_001, 1.0) is fmt.AnswerKind.NOT_MEASURED


def test_another_chromosome_is_not_measured_at_any_distance() -> None:
    for distance in (0, 1, 3_000_000, 5_000_000):
        assert kind(distance, 1.0, same=False) is fmt.AnswerKind.NOT_MEASURED


def test_two_records_at_one_position_are_a_near_band_pair() -> None:
    assert kind(0, 0.0) is fmt.AnswerKind.STORED


def test_r_squared_equal_to_t_far_is_stored_one_code_below_is_the_bound() -> None:
    # The walkthrough "r² equal to t_far": EUR k = 10's floor is 92 at × 511 (§3).
    assert kind(5_000_000, 92 / 511) is fmt.AnswerKind.STORED
    assert kind(5_000_000, 91 / 511) is fmt.AnswerKind.BOUND
    assert kind(5_000_000, -92 / 511) is fmt.AnswerKind.STORED
    # At × 32767 it is 5850, and the rule is the same: r̂² ≥ t_far.
    assert kind(5_000_000, 5850 / 32767) is fmt.AnswerKind.STORED
    assert kind(5_000_000, 5849 / 32767) is fmt.AnswerKind.BOUND


def test_amr_s_t_far_bounds_a_pair_eur_stores() -> None:
    # The walkthrough "AMR's t_far of 0.062": r = 0.24 (r² ≈ 0.0576), 5 Mb apart.
    r_hat = stored(0.24)
    assert kind(5_000_000, r_hat, AMR_K10) is fmt.AnswerKind.BOUND
    assert kind(5_000_000, r_hat, EUR_K10) is fmt.AnswerKind.STORED


def test_a_store_that_measured_less_than_10_mb_says_so() -> None:
    # An imported store may record far_bp below 10 Mb (ld-store §7); the reader uses the record.
    out = fmt.answer_kind(
        same_chromosome=True,
        distance=9_000_000,
        r_hat=1.0,
        t_far=0.001,
        near_bp=3_000_000,
        far_bp=8_000_000,
    )
    assert out is fmt.AnswerKind.NOT_MEASURED


@pytest.mark.parametrize("bad", [{"distance": -1}, {"r_hat": math.nan}, {"r_hat": 1.5}])
def test_answer_kind_refuses_impossible_inputs(bad: dict[str, float]) -> None:
    args: dict[str, float] = {"distance": 5_000_000, "r_hat": 0.5} | bad
    with pytest.raises(ValueError, match=r"distance|r̂"):
        fmt.answer_kind(
            same_chromosome=True,
            distance=int(args["distance"]),
            r_hat=args["r_hat"],
            t_far=EUR_K10,
            near_bp=NEAR_BP,
            far_bp=FAR_BP,
        )


@given(R, T, st.integers(min_value=0, max_value=12_000_000), st.booleans())
def test_the_answer_does_not_depend_on_the_sign(r: float, tf: float, d: int, same: bool) -> None:
    r_hat = stored(r)
    assert kind(d, r_hat, tf, same=same) is kind(d, -r_hat, tf, same=same)


def test_answer_kinds_reads_the_kind_of_every_cell_of_a_matrix() -> None:
    # Five variants: three on chr1 at 1, 2, and 4.5 Mb and 12 Mb, one on chr2.
    chrom = np.array(["1", "1", "1", "1", "2"])
    position = np.array([1_000_000, 2_000_000, 4_500_000, 12_000_000, 1_000_000])
    nan = math.nan
    r = np.array(
        [
            [1.0, 0.2, nan, nan, nan],
            [0.2, 1.0, 0.3, nan, nan],
            [nan, 0.3, 1.0, nan, nan],
            [nan, nan, nan, 1.0, nan],
            [nan, nan, nan, nan, 1.0],
        ]
    )
    s, b, m = (
        fmt.AnswerKind.STORED.value,
        fmt.AnswerKind.BOUND.value,
        fmt.AnswerKind.NOT_MEASURED.value,
    )
    expected = [
        [s, s, b, m, m],  # 1→4.5 Mb is 3.5 Mb: unstored, so the Bound; 1→12 Mb is 11 Mb
        [s, s, s, b, m],  # 2→12 Mb is 10 Mb exactly: unstored, so the Bound
        [b, s, s, b, m],  # 4.5→12 Mb is 7.5 Mb
        [m, b, b, s, m],
        [m, m, m, m, s],
    ]
    out = fmt.answer_kinds(
        r, chrom=chrom, position=position, t_far=EUR_K10, near_bp=NEAR_BP, far_bp=FAR_BP
    )
    assert out.tolist() == expected


def test_answer_kinds_refuses_an_unstored_near_band_pair() -> None:
    # Every pair within near_bp is stored, so a NaN there is a broken read, not an answer.
    with pytest.raises(ValueError, match="Near Band"):
        fmt.answer_kinds(
            np.array([[1.0, math.nan], [math.nan, 1.0]]),
            chrom=np.array(["1", "1"]),
            position=np.array([10, 20]),
            t_far=EUR_K10,
            near_bp=NEAR_BP,
            far_bp=FAR_BP,
        )


def kinds_of_pair(r: float, *, distance: int, same: bool = True) -> str:
    return str(
        fmt.answer_kinds(
            np.array([[1.0, r], [r, 1.0]]),
            chrom=np.array(["1", "1" if same else "2"]),
            position=np.array([0, distance]),
            t_far=EUR_K10,
            near_bp=NEAR_BP,
            far_bp=FAR_BP,
        )[0, 1]
    )


def test_a_number_where_the_store_holds_nothing_is_refused_not_labelled() -> None:
    # A cell beyond far_bp, or across chromosomes, holds nothing; a number there is a fill
    # or a broken read, and kinds() must not call it Not Measured.
    assert kinds_of_pair(math.nan, distance=11_000_000) == "not_measured"
    with pytest.raises(ValueError, match="holds nothing"):
        kinds_of_pair(0.0, distance=11_000_000)
    with pytest.raises(ValueError, match="holds nothing"):
        kinds_of_pair(0.3, distance=1_000, same=False)


def test_a_far_cell_below_t_far_is_refused_not_called_the_bound() -> None:
    # Every stored pair beyond near_bp has r̂² ≥ t_far, so a smaller number there, 0
    # included, is not a Stored r: it is a fill or a broken read.
    assert kinds_of_pair(math.nan, distance=5_000_000) == "bound"
    assert kinds_of_pair(0.3, distance=5_000_000) == "stored"
    with pytest.raises(ValueError, match="below t_far"):
        kinds_of_pair(0.0, distance=5_000_000)
    with pytest.raises(ValueError, match="below t_far"):
        kinds_of_pair(0.1, distance=5_000_000)


# ------------------------------------------------------------------ §9 refusals


def judge(
    r2: float,
    kb: float,
    tf: float = EUR_K10,
    far_bp: int = FAR_BP,
    t_partner: float | None = None,
) -> fmt.Decision:
    """The judgement `clump` and `proxies` make: with a store's t_partner it is
    `judge_partner_request`, and without one `judge_request`, which is a 0.1.0 store's table."""
    if t_partner is not None:
        return fmt.judge_partner_request(
            r2=r2,
            kb=kb,
            t_far=tf,
            near_bp=NEAR_BP,
            far_bp=far_bp,
            t_partner=t_partner,
            store="a store",
        )
    return fmt.judge_request(r2=r2, kb=kb, t_far=tf, near_bp=NEAR_BP, far_bp=far_bp)


UKB = t_far(49766, 10)


@pytest.mark.parametrize(
    ("r2", "kb", "tf", "refusal"),
    [
        # "Exactly 10 Mb"
        (0.1, 10_000, EUR_K10, None),
        (0.1, 10_000.001, EUR_K10, fmt.RefusalReason.WINDOW_OVER_FAR),
        # "A window over 10 Mb": refused on every panel, whatever r2 is
        (0.8, 10_001, EUR_K10, fmt.RefusalReason.WINDOW_OVER_FAR),
        (0.8, 10_001, UKB, fmt.RefusalReason.WINDOW_OVER_FAR),
        (0.001, 10_000, UKB, None),
        (0.001, 10_000, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        # "r² equal to t_far"
        (EUR_K10, 5_000, EUR_K10, None),
        (EUR_K10 - 1e-6, 5_000, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        # "AMR's t_far of 0.062"
        (0.05, 5_000, AMR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.05, 5_000, EUR_K10, None),
        (0.05, 3_000, AMR_K10, None),
        # §9: a store that records no t_partner — every 0.1.0 store — answers W ≤ 3 Mb at any r2;
        # the queries pass t_partner from 0.2.0, which the test below covers
        (0.0, 3_000, EUR_K10, None),
        (0.0, 3_000.001, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        (1.0, 0, EUR_K10, None),
        # §9: invalid
        (-0.01, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (1.01, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (math.nan, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, -1, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, math.nan, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, math.inf, EUR_K10, fmt.RefusalReason.INVALID),
    ],
)
def test_the_refusal_table(
    r2: float, kb: float, tf: float, refusal: fmt.RefusalReason | None
) -> None:
    decision = judge(r2, kb, tf)
    assert decision.refusal is refusal
    assert decision.answered is (refusal is None)


@pytest.mark.parametrize(
    ("r2", "kb", "t_partner", "refusal"),
    [
        # r² equal to t_partner is answered: the Partner Rows keep r̂² ≥ t_partner (ld-store §9)
        (0.01, 3_000, 0.01, None),
        # One code step below the Partner Floor, at × 511: code 52 is the floor at t_partner
        # 0.01, and code 51's r̂² 0.00996 is below it (ld-store §3; partner-rows §3)
        ((51 / 511) ** 2, 3_000, 0.01, fmt.RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        (math.nextafter(0.01, 0.0), 3_000, 0.01, fmt.RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        (0.0, 3_000, 0.01, fmt.RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        (0.0, 0, 0.01, fmt.RefusalReason.THRESHOLD_BELOW_T_PARTNER),
        # A store whose t_partner is 0.001, as UK Biobank EUR's is, answers r² 0.001 within 3 Mb,
        # and beyond it only above t_far (ld-store §9)
        (0.001, 3_000, 0.001, None),
        (0.001, 5_000, 0.001, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        # Above near_bp the table is unchanged by t_partner
        (0.1, 10_000, 0.01, None),
        (0.01, 10_000, 0.01, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
    ],
)
def test_the_partner_threshold_splits_the_table_within_near_bp(
    r2: float, kb: float, t_partner: float, refusal: fmt.RefusalReason | None
) -> None:
    decision = judge(r2, kb, t_partner=t_partner)
    assert decision.refusal is refusal


def test_a_threshold_below_the_partner_floor_names_it() -> None:
    decision = judge(0.001, 3_000, t_partner=0.01)
    with pytest.raises(fmt.Refusal) as caught:
        decision.raise_if_refused()
    assert caught.value.reason is fmt.RefusalReason.THRESHOLD_BELOW_T_PARTNER
    assert "t_partner" in str(caught.value)
    assert "Partner Floor" in str(caught.value)


def test_windows_are_compared_with_the_distance_the_store_measured() -> None:
    # A store whose far_bp is 8 Mb refuses a 9 Mb window that a 10 Mb store answers.
    assert judge(0.5, 9_000, far_bp=10_000_000).answered
    assert judge(0.5, 9_000, far_bp=8_000_000).refusal is fmt.RefusalReason.WINDOW_OVER_FAR


def test_a_refusal_names_its_reason() -> None:
    decision = judge(0.001, 10_000)
    with pytest.raises(fmt.Refusal) as caught:
        decision.raise_if_refused()
    assert caught.value.reason is fmt.RefusalReason.THRESHOLD_BELOW_T_FAR
    assert "t_far" in str(caught.value)
    judge(0.5, 1_000).raise_if_refused()  # an answered decision raises nothing


def test_an_answered_decision_carries_its_window_in_bp() -> None:
    assert judge(0.8, 10_000).window_bp == 10_000_000
    assert judge(0.8, 2.5).window_bp == 2_500


@given(
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=12_000),
    st.floats(min_value=0, max_value=12_000),
    st.sampled_from([EUR_K10, AMR_K10, UKB]),
)
def test_a_stricter_request_is_answered_whenever_a_looser_one_is(
    r2_a: float, r2_b: float, kb_a: float, kb_b: float, tf: float
) -> None:
    # Raising r2 or narrowing the window never turns an answered request into a refusal.
    loose_r2, strict_r2 = sorted((r2_a, r2_b))
    narrow, wide = sorted((kb_a, kb_b))
    if judge(loose_r2, wide, tf).answered:
        assert judge(strict_r2, narrow, tf).answered


@pytest.mark.parametrize(
    ("radius", "far_bp", "refusal"),
    [
        (5_000_000, 10_000_000, None),
        (5_000_001, 10_000_000, fmt.RefusalReason.REGION_OVER_FAR),
        (0, 10_000_000, None),
        (4_000_001, 8_000_000, fmt.RefusalReason.REGION_OVER_FAR),
        (-1, 10_000_000, fmt.RefusalReason.INVALID),
    ],
)
def test_a_region_is_at_most_the_far_distance_wide(
    radius: int, far_bp: int, refusal: fmt.RefusalReason | None
) -> None:
    # Owner, 7 November 2026: a region's width, 2 × radius, is capped at 10 Mb.
    assert fmt.judge_region(radius=radius, far_bp=far_bp).refusal is refusal
