"""ogquery reads a Genotype Store as opengwasdb `964ac1b2` did (opengwas/ogquery#11).

The reader came from that commit (ADR 0002, amended), so it must give that commit's answers:
every question `genotype_answers.QUESTIONS` asks -- a matrix, regions, proxies, a clump, and the
refusals -- is asked of `ogquery.ld.open` and compared with what `964ac1b2`'s `opengwasdb.ld.open`
answered on the same store (`data/test-panel/genotype-answers/`, recorded in `data/SOURCE`):
every array exactly, by dtype; every refusal by class, reason and text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import genotype_answers
import numpy as np
import pytest

import ogquery.ld

#: Every test here reads the Genotype Store: none imports the `genotype` extra's modules at
#: collection, so the suite collects where the extra is not installed (Windows; CI's no-genotype
#: run, which deselects these by their mark).
pytestmark = pytest.mark.genotype

EXPECTED_ARRAYS, EXPECTED_VALUES = genotype_answers.load()


def _equal(name: str, found: tuple[dict[str, Any], dict[str, Any]]) -> None:
    arrays, values = found
    assert values == EXPECTED_VALUES[name]
    expected = {
        key.removeprefix(name + "/"): array
        for key, array in EXPECTED_ARRAYS.items()
        if key.startswith(name + "/")
    }
    assert arrays.keys() == expected.keys()
    for key, array in expected.items():
        assert arrays[key].dtype == array.dtype, key
        np.testing.assert_array_equal(arrays[key], array, err_msg=key)


def test_the_answers_hold_every_question() -> None:
    pins = {f"refused-pin-{fault}" for fault in genotype_answers.PIN_FAULTS}
    names = {*genotype_answers.QUESTIONS, *pins}
    assert names == EXPECTED_VALUES.keys()
    answered = {name for name, values in EXPECTED_VALUES.items() if "refusal" not in values}
    assert answered == {
        "matrix",
        "proxies",
        "clump",
        *(f"region-{c}-{f}" for c in (20000000, 25000000) for f in (None, 0.0)),
    }


@pytest.mark.parametrize("name", sorted(genotype_answers.QUESTIONS))
def test_each_answer_is_964ac1b2_s(genotype_store: Path, name: str) -> None:
    ask = genotype_answers.QUESTIONS[name]
    _equal(name, genotype_answers.answer(ogquery.ld.open, genotype_store, ask))


@pytest.mark.parametrize("fault", genotype_answers.PIN_FAULTS)
def test_a_changed_pin_is_refused_on_open_as_964ac1b2_refused_it(
    genotype_store: Path, tmp_path: Path, fault: str
) -> None:
    copy = genotype_answers.place(tmp_path)
    genotype_answers.edit_pin(copy, fault)
    _equal(f"refused-pin-{fault}", genotype_answers.answer(ogquery.ld.open, copy, lambda s: s))


def test_a_matrix_read_in_parallel_slices_is_964ac1b2_s(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The matrix's 267 rows, read 128 at a time on the decoding threads, with a duplicate and
    # out of order: the same answer as the one read whole.
    from ogquery.ld.genotypes import reader as genotype_reader

    monkeypatch.setattr(genotype_reader, "_READ_ROWS", 128)
    ask = genotype_answers.QUESTIONS["matrix"]
    _equal("matrix", genotype_answers.answer(ogquery.ld.open, genotype_store, ask))
