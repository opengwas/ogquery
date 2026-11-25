"""The tests' sorted spelling (`spelling.py`) is the Test Store's: every stored ALID, A1 and A2
is what it forms from the Test Panel's passing record (ogquery#2)."""

from __future__ import annotations

import support
import zarr
from spelling import sorted_identity


def test_the_tests_spelling_is_the_test_stores() -> None:
    v = support.panel_data().variants()
    rows = support.passing_rows()
    formed = [
        sorted_identity(str(v.chrom[i]), int(v.pos[i]), str(v.ref[i]), str(v.alt[i])) for i in rows
    ]
    stored = zarr.open_group(str(support.TEST_STORE), mode="r")["variants"]
    assert len(formed) == stored["alid"].shape[0] > 1000
    assert stored["alid"][:].tolist() == [f.alid for f in formed]
    assert stored["a1"][:].tolist() == [f.a1 for f in formed]
    assert stored["a2"][:].tolist() == [f.a2 for f in formed]
    assert any(len(f.a1) != len(f.a2) for f in formed), "the panel must hold an indel"
