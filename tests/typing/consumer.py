"""A caller of ogquery, type-checked with `mypy --strict` against the installed wheel.

opengwasdb runs strict mypy over code that imports ogquery (#475). Without `py.typed`
in the wheel, mypy refuses the import (`import-untyped`); CI's `distributions` job
checks this file from outside the checkout, so only the installed package is seen.
"""

from __future__ import annotations

from pathlib import Path

from ogquery.facade import query_release
from ogquery.ld.format import VariantFilters
from ogquery.release import VariantTableReaderRequired, open_release


def first_analysis_rows(path: Path) -> int:
    try:
        release = open_release(path)
        with query_release(release) as q:
            first = q.analyses_table()[0]["analysis_id"]
            return int(q.analysis(first)["z"].size)
    except VariantTableReaderRequired as refused:
        raise SystemExit(f"{refused.format_version}: {refused}") from refused


def ld_answers(location: str, alids: list[str]) -> tuple[int, int, bool]:
    """The LD read path, as opengwasdb and ogdb-api call it (ogquery#2)."""
    import ogquery.ld
    from ogquery.opening import HttpExtraMissing

    try:
        store = ogquery.ld.open(location)
    except HttpExtraMissing as missing:
        raise SystemExit(str(missing)) from missing
    with store:
        matrix: ogquery.ld.LDMatrix = store.matrix(alids)
        clumps: ogquery.ld.Clumps = store.clump(alids, [1.0] * len(alids), r2=0.1, kb=1000)
        held: ogquery.ld.Variants = store.variants_in("22", 1, 50_000_000)
        present = bool(store.has(alids).all())
        return int(matrix.r.shape[0]) + len(held), len(clumps.leads()), present


def x_store_filters() -> VariantFilters:
    """An X Store's `variant_filters`, as opengwasdb's builder writes it (ADR 0109; #519)."""
    return {
        "filter": ["PASS", "."],
        "mac_min": 50,
        "info_min": None,
        "autosomes_only": False,
        "ploidy": {
            "chromosomes": ["X"],
            "sex_source": "psam SEX (1 male, 2 female)",
            "males": 300,
            "females": 365,
            "male_x_coding": "0/2",
            "x_pars_grch38": [[10001, 2781479], [155701383, 156030895]],
            "mac_min_haploid": 10,
        },
    }
