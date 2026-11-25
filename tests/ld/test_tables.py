"""The parts of the skeleton that are built: result tables, BuildInputs, LDStore's facts, the CLI.

These are containers and plumbing, implemented with the skeleton because every ticket
uses them. The queries that fill them are tests/ld/contract/'s.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import ogquery.ld
from ogquery.ld.format import t_far
from ogquery.ld.store import StoreReader

EUR_K10 = t_far(771, 10)


def variants(
    n: int, *, chrom: str = "22", start: int = 1_000_000, step: int = 1_000_000
) -> ogquery.ld.Variants:
    positions = np.arange(start, start + n * step, step, dtype=np.int64)
    return ogquery.ld.Variants(
        chrom=np.array([chrom] * n),
        position=positions,
        alid=np.array([f"{chrom}:{p}:A:G" for p in positions]),
        a1=np.array(["A"] * n),
        a2=np.array(["G"] * n),
        a1_frequency=np.full(n, 0.2),
    )


def test_a_table_refuses_columns_of_different_lengths() -> None:
    good = variants(3)
    assert len(good) == 3
    assert list(good.columns()) == ["chrom", "position", "alid", "a1", "a2", "a1_frequency"]
    with pytest.raises(ValueError, match="equally long"):
        ogquery.ld.Variants(**{**good.columns(), "position": np.arange(2, dtype=np.int64)})


def matrix(
    r: list[list[float]], fill: float | None = None, step: int = 1_000_000
) -> ogquery.ld.LDMatrix:
    return ogquery.ld.LDMatrix(
        r=np.array(r),
        variants=variants(len(r), step=step),
        absent=(),
        t_far=EUR_K10,
        near_bp=3_000_000,
        far_bp=10_000_000,
        fill=fill,
    )


def test_an_ld_matrix_must_be_square_over_its_variants() -> None:
    with pytest.raises(ValueError, match="variants"):
        ogquery.ld.LDMatrix(
            r=np.eye(2),
            variants=variants(3),
            absent=(),
            t_far=EUR_K10,
            near_bp=3_000_000,
            far_bp=10_000_000,
            fill=None,
        )


def test_an_ld_matrix_says_what_each_nan_means() -> None:
    # Variants 4 Mb apart: 0-1 is a Far Pair candidate, 0-2 is 8 Mb, 0-3 is 12 Mb.
    nan = math.nan
    out = matrix(
        [[1, nan, 0.4, nan], [nan, 1, nan, nan], [0.4, nan, 1, nan], [nan, nan, nan, 1]],
        step=4_000_000,
    ).kinds()
    assert out[0].tolist() == ["stored", "bound", "stored", "not_measured"]


@pytest.mark.parametrize("fill", [0.0, 0.5, -9.0])
def test_a_filled_matrix_has_no_kinds(fill: float) -> None:
    # A filled cell cannot be told from a stored one, so kinds() is defined only without a
    # fill: ask for the matrix again with fill=None.
    filled = matrix([[1, fill], [fill, 1]], fill=fill, step=4_000_000)
    with pytest.raises(ValueError, match="fill"):
        filled.kinds()


def test_a_hand_built_clumping_orders_its_leads_on_chr_prefixed_labels() -> None:
    # The engine reads `chr1` as `1` (normalise_chromosome), and so does leads(), as it did
    # before order_key (ogld#14): ties by chromosome rank, then position, then ALID.
    names = ["chr10:5:A:C", "chr2:50:A:C", "chr1:500:A:C", "chr1:100:A:G", "chr1:100:A:C"]
    clumps = ogquery.ld.Clumps(
        alid=np.array(names),
        chrom=np.array(["chr10", "chr2", "chr1", "chr1", "chr1"]),
        position=np.array([5, 50, 500, 100, 100], dtype=np.int64),
        score=np.array([5.0, 5.0, 3.0, 5.0, 5.0]),
        lead=np.array(names),
        r=np.ones(5),
        absent=(),
        r2=0.1,
        window_bp=1000.0,
    )
    expected = ["chr1:100:A:C", "chr1:100:A:G", "chr2:50:A:C", "chr10:5:A:C", "chr1:500:A:C"]
    assert list(clumps.leads()) == expected


class FakeReader(StoreReader):
    """A StoreReader whose provenance is three facts; nothing else is read."""

    @property
    def provenance(self) -> Any:
        return {"t_far": 0.0274, "near_bp": 3_000_000, "far_bp": 8_000_000}


def test_an_ld_store_reports_the_facts_its_store_recorded() -> None:
    opened = ogquery.ld.LDStore(FakeReader())
    assert (opened.t_far, opened.near_bp, opened.far_bp) == (0.0274, 3_000_000, 8_000_000)
