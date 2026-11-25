"""A caller of ogquery, type-checked with `mypy --strict` against the installed wheel.

opengwasdb runs strict mypy over code that imports ogquery (#475). Without `py.typed`
in the wheel, mypy refuses the import (`import-untyped`); CI's `distributions` job
checks this file from outside the checkout, so only the installed package is seen.
"""

from __future__ import annotations

from pathlib import Path

from ogquery.facade import query_release
from ogquery.release import VariantTableReaderRequired, open_release


def first_analysis_rows(path: Path) -> int:
    try:
        release = open_release(path)
        with query_release(release) as q:
            first = q.analyses_table()[0]["analysis_id"]
            return int(q.analysis(first)["z"].size)
    except VariantTableReaderRequired as refused:
        raise SystemExit(f"{refused.format_version}: {refused}") from refused
