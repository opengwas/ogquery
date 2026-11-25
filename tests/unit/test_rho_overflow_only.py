"""A Hybrid Dense Component's Rho Matrix records its Overflow-only Analyses, read exactly
(opengwasdb#496). The contract bundle (`tests/contract/`) asks the readers over real
releases; these hold the strict parse of the record, which a release the builders write
never breaks."""

from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from ogquery.rho import (
    OVERFLOW_ONLY_ATTR,
    OverflowOnlyAnalysisError,
    recorded_overflow_only,
)


def _group(tmp_path: Path, **attrs: object) -> zarr.Group:
    group = zarr.open_group(tmp_path / "rho", mode="w")
    group.attrs.update(attrs)
    return group


def test_a_recorded_list_is_read_as_a_set(tmp_path: Path):
    group = _group(tmp_path, **{OVERFLOW_ONLY_ATTR: [1, 3]})
    assert recorded_overflow_only(group, 4, "rho") == frozenset({1, 3})
    assert recorded_overflow_only(_group(tmp_path, **{OVERFLOW_ONLY_ATTR: []}), 4, "rho") == set()


def test_a_matrix_that_records_none_is_refused_not_read_as_empty(tmp_path: Path):
    with pytest.raises(ValueError, match="does not record its Overflow-only Analyses"):
        recorded_overflow_only(_group(tmp_path), 4, "rho")


@pytest.mark.parametrize(
    "listed", [["1"], [1.0], [True], [1, 1], [3, 1], [4], [-1], "1", 1, None]
)
def test_a_malformed_record_is_refused_not_coerced(tmp_path: Path, listed: object):
    group = _group(tmp_path, **{OVERFLOW_ONLY_ATTR: listed})
    with pytest.raises(ValueError, match="ascending analysis_index"):
        recorded_overflow_only(group, 4, "rho")


def test_the_refusal_names_the_analysis_for_a_port_to_compare():
    refused = OverflowOnlyAnalysisError("trait_c")
    assert refused.fields == {"analysis_id": "trait_c"}
    assert isinstance(refused, LookupError)
