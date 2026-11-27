"""Genotype Stores answer through the same public queries as the independent Test Store
(opengwas/ogld#91).

From opengwasdb `964ac1b2`'s `tests/ld/test_genotype_store.py` (opengwas/ogquery#11), on the
committed Genotype Store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import support
from support import TEST_STORE

import ogquery.ld
from ogquery.ld.format import claims_partner_rows

pytestmark = pytest.mark.genotype


def test_matrix_equals_the_independent_test_store(genotype_store: Path) -> None:
    rows = support.passing_rows()
    assert len(rows) > 100
    names = [*support.alids(rows), "absent", support.alids(rows[:1])[0]]
    expected = ogquery.ld.open(TEST_STORE).matrix(names)
    actual = ogquery.ld.open(genotype_store).matrix(names)
    assert actual.absent == expected.absent == ("absent",)
    np.testing.assert_array_equal(actual.r, expected.r)
    np.testing.assert_array_equal(actual.kinds(), expected.kinds())


def test_provenance_claims_partner_rows_for_both_store_kinds(genotype_store: Path) -> None:
    """`claims_partner_rows` is the capability rule: both kinds answer `clump` from t_partner."""
    ld = ogquery.ld.open(TEST_STORE).provenance
    sidecar = ogquery.ld.open(genotype_store).provenance
    assert claims_partner_rows(ld), "the LD Store records its Partner Threshold"
    assert claims_partner_rows(sidecar), "the sidecar answers clump and proxies from t_partner"
    without = {key: value for key, value in dict(ld).items() if key != "t_partner"}
    assert not claims_partner_rows(without), "a store that records no threshold claims none"
