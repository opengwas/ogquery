"""Every public read an index can reach is guarded, or exempt for a stated reason (ogquery#13).

opengwasdb ADR 0109 §2: a splice renumbers `analysis_index` and `variant_index`, so every
entry point that takes one must be told the index space it came from (``expect_space=``, or
``expect_spaces=`` for a reader's cells) and refuse another. A hand-kept list of guarded
methods missed some (review round 2), so this sweep is mechanical: it walks every public
method of every class `ogquery` defines and finds each that takes an index-bearing value,
by name or by its `Cells` annotation. Each must take the keyword or be in `EXEMPT`, with its
reason. A new method that takes an index fails here until it is one or the other.

`ogquery.ld` is not walked: its rows are an LD Store's own variant rows, which a splice of an
OpenGWAS release does not renumber, and its genotype half needs the `genotype` extra.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterator

import pytest

import ogquery

#: A parameter name that holds an Analysis or variant index, a row or column of a reader's
#: cells, a run of them, or a flat position derived from them.
INDEX_NAME = re.compile(
    r"^(analysis_index|analysis_indices|variant_index|variant_indices|indices|index|"
    r"row|rows|row_indices|col|cols|col_indices|first|low|high|first_variant|last_variant|"
    r"positions|slot|i|j)$"
)
GUARDS = ("expect_space", "expect_spaces")

_POSITIONS = (
    "flat positions in this sequence, which no answer carries: they are produced by this "
    "reader's own guarded `span`, `segment_positions`, `pair_positions` and `windows`"
)
_PLANE = (
    "a plane decoder bound to its arrays, not to a release; the readers that open it "
    "(`DenseGridReader`, `RaggedCSRReader`, `ByVariantReader`) guard every index they pass it"
)
_SIDECAR = (
    "a sidecar decoder holding no release identity; the Variant Axis is its reader, and "
    "guards every index it passes it (`VariantAxis`, `PanelVariantAxis`)"
)
_SELECTOR = (
    "builds a selector or record; reading it is guarded (`read`, `inspect`, `z`, `windows`), "
    "and an unstamped record cannot be resolved (`resolve` refuses it)"
)
_OWN = "positions within this record itself, not a release's index"

#: Each method that takes an index-bearing value and no space, and why that is safe.
EXEMPT: dict[str, str] = {
    "ogquery.alid_index.AlidIndex.alids": _SIDECAR,
    "ogquery.alid_index.FixedWidthAlidIndex.alids": _SIDECAR,
    "ogquery.alid_sidecar.AlidSidecar.alid": _SIDECAR,
    "ogquery.alid_sidecar.AlidSidecar.alid_bytes_at_slot": _SIDECAR,
    "ogquery.alid_sidecar.AlidSidecar.alids": _SIDECAR,
    "ogquery.panel_axis.PanelAlidIndex.alids": _SIDECAR,
    "ogquery.position_sidecar.PositionSidecar.chromosomes_of": _SIDECAR,
    "ogquery.position_sidecar.PositionSidecar.misplaced": _SIDECAR,
    "ogquery.position_sidecar.PositionSidecar.placements": _SIDECAR,
    "ogquery.rsid_index.RsidIndex.rsids": _SIDECAR,
    "ogquery.source_alid_sidecar.SourceAlidSidecar.values": _SIDECAR,
    "ogquery.associations.cells.Cells.analyses": _SELECTOR,
    "ogquery.associations.cells.Cells.analysis": _SELECTOR,
    "ogquery.associations.cells.Cells.cross": _SELECTOR,
    "ogquery.associations.cells.Cells.pairs": _SELECTOR,
    "ogquery.associations.cells.Cells.positions": _SELECTOR,
    "ogquery.associations.cells.Cells.variants": _SELECTOR,
    "ogquery.associations.record.Associations.from_decoded": _SELECTOR,
    "ogquery.associations.record.Inspection.of": _SELECTOR,
    "ogquery.associations.record.Associations.take": _OWN,
    "ogquery.associations.record.Inspection.take": _OWN,
    "ogquery.csr.RaggedCSRReader.analysis_index_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.eaf_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.eaf_at_read": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.imputed_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.n_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.se_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.variant_index_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.z_at": _POSITIONS,
    "ogquery.encoding.codec.SparseExactTable.lookup": _PLANE,
    "ogquery.encoding.codec.WindowedExactTable.lookup": _PLANE,
    "ogquery.encoding.codec.StoreDecoder.decode_eaf": _PLANE,
    "ogquery.encoding.codec.StoreDecoder.decode_se": _PLANE,
    "ogquery.encoding.codec.StoreDecoder.decode_z": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.points": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.read_block": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.read_column": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.read_points": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.read_row": _PLANE,
    "ogquery.encoding.planes.DenseEafPlane.read_rows": _PLANE,
    "ogquery.encoding.planes.DenseSePlane.block": _PLANE,
    "ogquery.encoding.planes.DenseSePlane.column": _PLANE,
    "ogquery.encoding.planes.DenseSePlane.points": _PLANE,
    "ogquery.encoding.planes.DenseSePlane.row": _PLANE,
    "ogquery.encoding.planes.DenseSePlane.rows": _PLANE,
    "ogquery.encoding.planes.DenseZPlane.block": _PLANE,
    "ogquery.encoding.planes.DenseZPlane.column": _PLANE,
    "ogquery.encoding.planes.DenseZPlane.points": _PLANE,
    "ogquery.encoding.planes.DenseZPlane.row": _PLANE,
    "ogquery.encoding.planes.DenseZPlane.rows": _PLANE,
    "ogquery.encoding.planes.RaggedEafPlane.at": _PLANE,
    "ogquery.encoding.planes.RaggedEafPlane.read_at": _PLANE,
    "ogquery.encoding.planes.RaggedSePlane.at": _PLANE,
    "ogquery.encoding.planes.RaggedSePlane.slice": _PLANE,
}


def _modules() -> Iterator[object]:
    for found in pkgutil.walk_packages(ogquery.__path__, "ogquery."):
        if found.name == "ogquery.ld" or found.name.startswith("ogquery.ld."):
            continue
        yield importlib.import_module(found.name)


def _methods() -> Iterator[tuple[str, inspect.Signature]]:
    """Every public method, classmethod and staticmethod a class of `ogquery` defines."""
    for module in _modules():
        for class_name, owner in vars(module).items():
            if not inspect.isclass(owner) or owner.__module__ != module.__name__:  # type: ignore[attr-defined]
                continue
            if class_name.startswith("_"):
                continue
            for name, raw in vars(owner).items():
                if name.startswith("_") or isinstance(raw, property):
                    continue
                function = raw.__func__ if isinstance(raw, classmethod | staticmethod) else raw
                if not inspect.isfunction(function):
                    continue
                yield f"{owner.__module__}.{class_name}.{name}", inspect.signature(function)
            # A reader's constructor takes no index; its methods are walked above.


def _takes_an_index(signature: inspect.Signature) -> list[str]:
    return [
        p.name
        for p in signature.parameters.values()
        if INDEX_NAME.match(p.name) or "Cells" in str(p.annotation)
    ]


def _sweep() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    guarded: dict[str, list[str]] = {}
    unguarded: dict[str, list[str]] = {}
    for name, signature in _methods():
        taken = _takes_an_index(signature)
        if not taken:
            continue
        guards = [p for p in signature.parameters.values() if p.name in GUARDS]
        if guards:
            assert all(p.kind is p.KEYWORD_ONLY for p in guards), name
            guarded[name] = taken
        else:
            unguarded[name] = taken
    return guarded, unguarded


def test_every_method_taking_an_index_is_guarded_or_exempt_with_a_reason() -> None:
    _, unguarded = _sweep()
    missing = sorted(set(unguarded) - set(EXEMPT))
    assert not missing, f"take an index but no expect_space(s) and are not exempt: {missing}"


def test_every_exemption_names_a_method_that_takes_an_index_unguarded() -> None:
    _, unguarded = _sweep()
    stale = sorted(set(EXEMPT) - set(unguarded))
    assert not stale, f"exempt but no longer an unguarded index-taking method: {stale}"
    assert all(reason.strip() for reason in EXEMPT.values())


@pytest.mark.parametrize(
    "name",
    [
        "ogquery.by_variant.ByVariantReader.rows_for_variant_range",
        "ogquery.by_variant.ByVariantReader.decode",
        "ogquery.csr.RaggedCSRReader.se_slice",
        "ogquery.associations.dense.DenseGridReader.z",
        "ogquery.associations.dense.DenseGridReader.windows",
        "ogquery.associations.csr.CsrReader.z",
        "ogquery.associations.csr.CsrReader.windows",
        # Bare `i`/`j` Analysis indices are found too.
        "ogquery.rho.DenseRhoReader.row",
        "ogquery.rho.DenseRhoReader.pair",
    ],
)
def test_the_reads_review_round_2_found_are_found_and_guarded(name: str) -> None:
    guarded, _ = _sweep()
    assert name in guarded
