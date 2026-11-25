"""ogld#16: each LD Store records its encoding, and the reader decodes with it or refuses it.

From opengwasdb's `tests/ld/test_encoding.py` at `c8ebe4ca` (ogquery#2): its reader's part, on a
copy of the committed Test Store whose recorded encoding a test edits. Its stores at × 2047 and
× 32767, the Partner Floor and validation at each scale need opengwasdb's build and stay there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from store_copies import R_SCALE, TEST_STORE, copy_store

import ogquery.ld

UNIMPLEMENTED = {
    "an int8 container": {"r_dtype": "int8", "r_scale": 127},
    "a scale past int16": {"r_scale": 32768},
    "rounding half away from zero": {"rounding": "half_away_from_zero"},
}


def test_the_reader_decodes_at_the_scale_the_store_records() -> None:
    store = ogquery.ld.open(TEST_STORE)
    assert store.provenance["encoding"]["r_scale"] == R_SCALE
    assert store._reader.encoding.scale == R_SCALE


@pytest.mark.parametrize("case", list(UNIMPLEMENTED))
def test_an_encoding_the_reader_does_not_implement_is_refused_at_open_naming_it(
    case: str, tmp_path: Path
) -> None:
    copy = copy_store(tmp_path / "store")
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    recorded: dict[str, Any] = doc["attributes"]["ld_store"]["encoding"]
    recorded.update(UNIMPLEMENTED[case])
    meta.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ogquery.ld.InputError) as refused:
        ogquery.ld.open(copy)
    assert repr(recorded) in str(refused.value), refused.value
    assert str(copy) in str(refused.value)
    assert "ogquery.ld implements" in str(refused.value), "the refusal names this reader"
