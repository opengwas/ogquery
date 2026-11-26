"""A Hybrid's Rho reads, from its Dense Component's Rho Matrix (opengwasdb#495/#496).

Ported from opengwasdb's `tests/test_hybrid_rho.py`, on copies of the contract bundle's
`hybrid_overflow_only` and `dense_rho` releases instead of its builder: the reads (a pair, a
row, the matrix; refusing an Overflow-only Analysis by name, an unknown one first, a Rho
Matrix that records no Overflow-only Analyses or records them wrongly) and `differences` on
the `rho` section. The build (`build_dense_rho` on a Hybrid), its validation and the 0.2
Dense Component stay in opengwasdb: they need its builders.

`hybrid_overflow_only`: `trait_a`/`trait_b` correlated (Rho 0.617), `trait_c` Overflow-only
(`overflow_only: [2]`), `trait_d` Dense with no null support (NaN, not refused).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from ogquery.analyses import UnknownAnalysisError
from ogquery.digests import differences
from ogquery.facade import query_release
from ogquery.release import open_release
from ogquery.rho import OVERFLOW_ONLY_ATTR, OverflowOnlyAnalysisError

BUNDLE = Path(__file__).parents[1] / "contract"
OVERFLOW_ONLY = "trait_c"
INTERIOR = 0.617


@pytest.fixture
def hybrid(tmp_path: Path) -> Path:
    return Path(shutil.copytree(BUNDLE / "releases" / "hybrid_overflow_only", tmp_path / "h"))


def _rho_group(release: Path, mode: str = "r") -> zarr.Group:
    return zarr.open_group(release / "dense" / "data.zarr" / "rho", mode=mode)


def _query(release: Path):
    return query_release(open_release(release))


def test_the_reader_answers_for_dense_analyses_and_refuses_an_overflow_only_one(hybrid: Path):
    with _query(hybrid) as query:
        matrix = query.rho_matrix()
        assert matrix["analysis_id"].tolist() == ["trait_a", "trait_b", "trait_d"]
        assert float(matrix["rho"][0, 1]) == pytest.approx(INTERIOR, abs=1e-3)
        # `trait_d` is Dense without support: NaN, not refused.
        assert np.isnan(matrix["rho"][2, :2]).all() and (matrix["n_null"][2, :2] == 0).all()

        row = query.rho_row("trait_a")
        assert row["analysis_id"].tolist() == ["trait_b", "trait_d"]
        assert float(row["rho"][0]) == pytest.approx(INTERIOR, abs=1e-3)

        pair = query.rho("trait_b", "trait_a")
        assert pair["analysis_id_a"].tolist() == ["trait_b"]
        assert float(pair["rho"][0]) == pytest.approx(INTERIOR, abs=1e-3)
        assert len(query.rho(["trait_a", "trait_b", "trait_d"])["rho"]) == 3

        for ask in (
            lambda: query.rho("trait_a", OVERFLOW_ONLY),
            lambda: query.rho([OVERFLOW_ONLY]),
            lambda: query.rho_row(OVERFLOW_ONLY),
            lambda: query.rho_matrix(["trait_a", OVERFLOW_ONLY]),
        ):
            with pytest.raises(OverflowOnlyAnalysisError) as refused:
                ask()
            assert refused.value.fields == {"analysis_id": OVERFLOW_ONLY}
        # An Analysis the release does not hold is refused as unknown first.
        with pytest.raises(UnknownAnalysisError):
            query.rho(OVERFLOW_ONLY, "absent")


def test_without_a_rho_matrix_the_hybrid_answers_empty_and_still_refuses_unknown(hybrid: Path):
    shutil.rmtree(hybrid / "dense" / "data.zarr" / "rho")
    manifest = hybrid / "dense" / "manifest.json"
    declared = json.loads(manifest.read_text(encoding="utf-8"))
    declared["arrays"] = {k: v for k, v in declared.get("arrays", {}).items() if k != "rho"}
    manifest.write_text(json.dumps(declared), encoding="utf-8")
    with _query(hybrid) as query:
        assert query.rho_matrix()["rho"].shape == (0, 0)
        assert len(query.rho_row(OVERFLOW_ONLY)["rho"]) == 0
        with pytest.raises(UnknownAnalysisError):
            query.rho_row("absent")


def test_a_matrix_that_does_not_record_its_overflow_only_analyses_is_refused(hybrid: Path):
    """A Rho Matrix built before #496 does not say which Analyses have no Rho, so a NaN in
    it could be either; the reader refuses it."""
    del _rho_group(hybrid, "r+").attrs[OVERFLOW_ONLY_ATTR]
    with _query(hybrid) as query, pytest.raises(ValueError, match=OVERFLOW_ONLY_ATTR):
        query.rho_matrix()


@pytest.mark.parametrize(
    "recorded",
    [["2"], [2.0], [True, 2], [2, 2], [5], -1, [3, 2], [-1, 2]],
    ids=[
        "string", "float", "bool", "repeated", "out-of-range", "not-a-list",
        "descending", "negative-element",
    ],
)
def test_a_malformed_record_is_refused_not_coerced(hybrid: Path, recorded: object):
    """Each is refused as `recorded_overflow_only` reads it, as opengwasdb `355449d7`'s
    in-tree reader does (same function): `[3, 2]` is valid indices but not ascending, and
    `[-1, 2]` has a negative element inside a list, as `-1` alone is not a list."""
    _rho_group(hybrid, "r+").attrs[OVERFLOW_ONLY_ATTR] = recorded
    with _query(hybrid) as query, pytest.raises(ValueError, match=OVERFLOW_ONLY_ATTR):
        query.rho_row("trait_a")


def test_a_changed_rho_digest_is_a_difference():
    """`differences` -- the rewrite proof's comparison, which has no readable rows --
    compares the `rho` section as it does every other (opengwasdb#497)."""
    expected = BUNDLE / "expected" / "dense_rho.json"
    digests = json.loads(expected.read_text(encoding="utf-8"))["digests"]
    assert digests["rho"]["rho_matrix"], "the fixture asks no Rho question"
    assert differences(digests, digests) == []
    changed = copy.deepcopy(digests)
    changed["rho"]["rho_matrix"]["rho"] = "0" * 64
    assert differences(digests, changed) == ["rho/rho_matrix/rho"]
