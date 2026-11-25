"""The range PheWAS of a release's digests is bounded by variants and by cells (#492).

Ported from opengwasdb's `tests/test_convert_legacy.py`, with the narrowing's record
(`narrowed_range_question`) that the helper returns beside it. What stays in opengwasdb
needs its conversion builders: the rewrite recording a narrowed proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ogquery import digests

RECORDS = [{"chromosome": "1", "position": p} for p in range(1, 30_001)]
RECORDS.append({"chromosome": "2", "position": 25_000})


def test_the_range_question_is_bounded_by_variants_and_by_cells() -> None:
    """A large release's range PheWAS is cut to `RANGE_ROWS_ASKED` variants and to
    `RANGE_CELLS_ASKED` cells; a small one is asked over its whole span."""
    assert digests._range_last(RECORDS[:500] + RECORDS[-1:], 2)["position"] == 25_000
    assert digests._range_last(RECORDS, 2)["position"] == 10_000
    assert digests._range_last(RECORDS, 2_514)["position"] == 2_500_000 // 2_514
    assert digests._range_last(RECORDS, 241)["position"] == 10_000, "OGS-00015's"


def test_a_range_asked_over_its_whole_span_is_not_narrowed() -> None:
    assert digests.narrowed_range_question(RECORDS[:500] + RECORDS[-1:], 2) is None


def test_a_narrowed_range_says_what_it_dropped() -> None:
    assert digests.narrowed_range_question(RECORDS, 241) == {
        "chromosome": "1",
        "start": 1,
        "end": 10_000,
        "unbounded_end": 25_000,
        "variants": 10_000,
        "unbounded_variants": 25_000,
        "analyses": 241,
    }


@pytest.mark.parametrize("n_analyses", [0, 1])
def test_a_release_with_no_or_one_analysis_is_bounded_by_variants(n_analyses: int) -> None:
    narrowed = digests.narrowed_range_question(RECORDS, n_analyses)
    assert narrowed is not None and narrowed["variants"] == digests.RANGE_ROWS_ASKED


def test_the_digest_names_its_facade_through_a_private_subclass() -> None:
    class StoreQuery: ...

    class _Converting(StoreQuery): ...

    assert digests.adapter_name(_Converting()) == "StoreQuery"
    assert digests.expects_refusal(_Converting(), digests.UNSTORED_SHAPES[0])


# ── the entry points: the proof output ───────────────────────────────────────

RELEASE = Path(__file__).parents[1] / "contract" / "releases" / "dense_completed"


def test_release_answers_records_a_narrowed_range_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole = digests.release_digests(RELEASE, query=digests.open_query)
    assert "narrowed_range_question" not in whole
    monkeypatch.setattr(digests, "RANGE_ROWS_ASKED", 3)
    narrowed = digests.release_digests(RELEASE, query=digests.open_query)
    record = narrowed["narrowed_range_question"]
    assert record["variants"] == 3 and record["unbounded_variants"] > 3
    assert record["end"] < record["unbounded_end"] and record["analyses"] > 0


def test_an_explicit_opener_is_called_even_when_it_is_falsey() -> None:
    class Opener:
        called: list[Path] = []

        def __bool__(self) -> bool:
            return False

        def __call__(self, path: Path) -> Any:
            self.called.append(path)
            return digests.open_query(path)

    opener = Opener()
    digests.release_digests(RELEASE, query=opener)
    assert opener.called == [RELEASE]
