"""Every public read an index can reach is guarded, or exempt for a stated reason (ogquery#13).

opengwasdb ADR 0109 §2: a splice renumbers `analysis_index` and `variant_index`, so every
entry point that takes one must be told the index space it came from (``expect_space=``, or
``expect_spaces=`` for a reader's cells) and refuse another. A hand-kept list of guarded
methods missed some (review round 2), so this sweep is mechanical: it walks every public
method of every class `ogquery` defines, inherited ones included, and finds each that takes
an index-bearing value, by name or by its `Cells` annotation. Each must take the keyword or
be in `EXEMPT`, with its reason. A new method that takes an index fails here until it is one
or the other.

Naming the keyword is not enough (review round 3). The keyword has no default, unless the
method is in `CONDITIONAL`, where it is required only when an index is given or the decoder
is bound to an axis. Every guarded method must be in the behavioural table of
`test_release_guard.py` (`entry_points`), where it is called with its own space, another's
and none, and the table holds nothing the sweep does not find.

`ogquery.ld` is not walked: its rows are an LD Store's own variant rows, which a splice of an
OpenGWAS release does not renumber, and its genotype half needs the `genotype` extra. A
`Protocol` is not walked either: it states a signature, and its implementations are walked.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

import ogquery

#: A parameter name that holds an Analysis or variant index, a row or column of a reader's
#: cells, a run or range of them, or a flat position derived from them.
INDEX_NAME = re.compile(
    r"^(analysis_index|analysis_indices|variant_index|variant_indices|indices|index|"
    r"row|rows|row_indices|variant_rows|col|cols|col_indices|first|low|high|first_variant|"
    r"last_variant|variant|positions|slot|i|j|start|stop|end)$"
)
GUARDS = ("expect_space", "expect_spaces")

_CONDITIONAL_INDEX = "required when an index or region is given; with none it reads everything"
_BOUND = (
    "required when the decoder is bound to an axis, as every one a Variant Axis hands out is "
    "(`VariantAxis.alid_index`, `.rsid_index`, `.position_sidecar`, `.source_alid_sidecar`); "
    "one opened from its files holds no release identity"
)

#: Each guarded method whose keyword may be omitted, and when it is required.
CONDITIONAL: dict[str, str] = {
    "ogquery.top_hits.TopHitTier.read": _CONDITIONAL_INDEX,
    "ogquery.top_hits.TopHitTier.clumps": _CONDITIONAL_INDEX,
    "ogquery.csr.RaggedCSRReader.se_slice": _CONDITIONAL_INDEX,
    "ogquery.associations.dense.DenseGridReader.windows": _CONDITIONAL_INDEX,
    "ogquery.associations.csr.CsrReader.windows": _CONDITIONAL_INDEX,
    "ogquery.alid_sidecar.AlidSidecar.alid": _BOUND,
    "ogquery.alid_sidecar.AlidSidecar.alids": _BOUND,
    "ogquery.alid_index.FixedWidthAlidIndex.alids": _BOUND,
    "ogquery.panel_axis.PanelAlidIndex.alids": _BOUND,
    "ogquery.rsid_index.RsidIndex.rsids": _BOUND,
    "ogquery.position_sidecar.PositionSidecar.chromosomes_of": _BOUND,
    "ogquery.position_sidecar.PositionSidecar.placements": _BOUND,
    "ogquery.position_sidecar.PositionSidecar.misplaced": _BOUND,
    "ogquery.source_alid_sidecar.SourceAlidSidecar.values": _BOUND,
    "ogquery.source_alid_sidecar.SourceAlidSidecar.sourced": _BOUND,
    "ogquery.source_alid_sidecar.SourceAlidSidecar.differing": _BOUND,
}

_POSITIONS = (
    "flat positions in this sequence, not an Analysis or variant index: no query answer "
    "carries one, and a caller gets them from this reader's own guarded `span`, "
    "`segment_positions`, `pair_positions` and `windows`. A position a caller caches from "
    "one release and reads in another is not caught here"
)
_PLANE = (
    "a plane or codec decoder bound to its arrays, not to a release, and opened by no axis "
    "or facade attribute: the readers that open it (`DenseGridReader`, `RaggedCSRReader`, "
    "`ByVariantReader`) guard every index they pass it"
)
_SELECTOR = (
    "builds a selector or record; reading it is guarded (`read`, `inspect`, `z`, `windows`), "
    "and an unstamped record cannot be resolved (`resolve` refuses it)"
)
_OWN = "positions within this record itself, not a release's index"
_GENOMIC = "a genomic position on a chromosome, which a splice does not renumber"
_SLOT = "a slot of the sidecar's sorted keys, which only its own search returns"

#: Each method that takes an index-bearing value and no space, and why that is safe.
EXEMPT: dict[str, str] = {
    "ogquery.alid_sidecar.AlidSidecar.alid_bytes_at_slot": _SLOT,
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
    "ogquery.csr.RaggedCSRReader.eaf_slice_read": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.imputed_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.n_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.n_slice": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.se_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.variant_index": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.variant_index_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.z_at": _POSITIONS,
    "ogquery.csr.RaggedCSRReader.z_slice": _POSITIONS,
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
    "ogquery.encoding.planes.RaggedEafPlane.read_slice": _PLANE,
    "ogquery.encoding.planes.RaggedEafPlane.slice": _PLANE,
    "ogquery.encoding.planes.RaggedSePlane.at": _PLANE,
    "ogquery.encoding.planes.RaggedSePlane.slice": _PLANE,
    "ogquery.facade.HybridStoreQuery.range_phewas": _GENOMIC,
    "ogquery.facade.RaggedStoreQuery.range_by_analysis": _GENOMIC,
    "ogquery.facade.RaggedStoreQuery.range_phewas": _GENOMIC,
    "ogquery.facade.StoreQuery.range_phewas": _GENOMIC,
    "ogquery.facade._StoreQueryBase.variants_in_range": _GENOMIC,
    "ogquery.panel_axis.PanelVariantAxis.range": _GENOMIC,
    "ogquery.panel_axis.PanelVariantAxis.range_indices": _GENOMIC,
    "ogquery.panel_axis.PanelVariantAxis.tabix_range": _GENOMIC,
    "ogquery.position_sidecar.PositionSidecar.rows_in": _GENOMIC,
    "ogquery.variant_axis.NoVariantTable.fetch": _GENOMIC,
    "ogquery.variant_axis.VariantAxis.range": _GENOMIC,
    "ogquery.variant_axis.VariantAxis.range_indices": _GENOMIC,
    "ogquery.variant_axis.VariantAxis.tabix_range": _GENOMIC,
}

Method = tuple[str, inspect.Signature]


def _modules() -> Iterator[Any]:
    for found in pkgutil.walk_packages(ogquery.__path__, "ogquery."):
        if found.name == "ogquery.ld" or found.name.startswith("ogquery.ld."):
            continue
        yield importlib.import_module(found.name)


def _ours(module: str) -> bool:
    return module.startswith("ogquery.") and not module.startswith("ogquery.ld")


def methods_of(owner: type, *, ours: Any = None) -> Iterator[Method]:
    """Every public method, classmethod and staticmethod `owner` has, its inherited ones
    too, each named where it is defined (``module.Class.method``): those `ogquery` defines,
    or, for a test's own class, those `ours` accepts by module name."""
    ours = _ours if ours is None else ours
    if getattr(owner, "_is_protocol", False):
        return
    for name in dir(owner):
        if name.startswith("_"):
            continue
        raw = inspect.getattr_static(owner, name)
        if isinstance(raw, property):
            continue
        function = raw.__func__ if isinstance(raw, classmethod | staticmethod) else raw
        if not inspect.isfunction(function) or not ours(function.__module__):
            continue
        yield f"{function.__module__}.{function.__qualname__}", inspect.signature(function)


def _methods() -> Iterator[Method]:
    seen: set[str] = set()
    for module in _modules():
        for class_name, owner in vars(module).items():
            if not inspect.isclass(owner) or owner.__module__ != module.__name__:
                continue
            if class_name.startswith("_"):
                continue
            for name, signature in methods_of(owner):
                if name not in seen:
                    seen.add(name)
                    yield name, signature


def _takes_an_index(signature: inspect.Signature) -> list[str]:
    return [
        p.name
        for p in signature.parameters.values()
        if INDEX_NAME.match(p.name) or "Cells" in str(p.annotation)
    ]


def sweep(methods: Iterable[Method]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(guarded, unguarded)``: each method that takes an index, by whether it takes
    `expect_space(s)`."""
    guarded: dict[str, list[str]] = {}
    unguarded: dict[str, list[str]] = {}
    for name, signature in methods:
        taken = _takes_an_index(signature)
        if not taken:
            continue
        guards = [p for p in signature.parameters.values() if p.name in GUARDS]
        (guarded if guards else unguarded)[name] = taken
    return guarded, unguarded


def violations(methods: Iterable[Method]) -> list[str]:
    """Each way `methods` break the rule: an index taken unguarded and not exempt, or a
    guard that is not keyword-only, or that defaults where it is not `CONDITIONAL`."""
    methods = list(methods)
    signatures = dict(methods)
    guarded, unguarded = sweep(methods)
    found = [f"{name}: takes {taken} unguarded" for name, taken in unguarded.items()
             if name not in EXEMPT]
    for name in guarded:
        for p in signatures[name].parameters.values():
            if p.name not in GUARDS:
                continue
            if p.kind is not p.KEYWORD_ONLY:
                found.append(f"{name}: {p.name} is not keyword-only")
            if p.default is not p.empty and name not in CONDITIONAL:
                found.append(f"{name}: {p.name} defaults to {p.default!r}")
    return found


def guarded_methods() -> dict[str, list[str]]:
    return sweep(_methods())[0]


def test_every_method_taking_an_index_is_guarded_or_exempt_with_a_reason() -> None:
    assert violations(_methods()) == []


def test_every_exemption_and_condition_names_a_method_the_sweep_finds_so() -> None:
    guarded, unguarded = sweep(_methods())
    assert sorted(set(EXEMPT) - set(unguarded)) == []
    assert sorted(set(CONDITIONAL) - set(guarded)) == []
    assert all(reason.strip() for reason in [*EXEMPT.values(), *CONDITIONAL.values()])


class _Injected:
    """What review round 3 added to see whether the sweep would notice: it did not."""

    def read(self, variant: int) -> None: ...

    def rows(self, variant_rows: list[int], *, expect_space: Any = None) -> None: ...

    def sourced(self, start: int, stop: int) -> None: ...

    def fine(self, i: int, *, expect_space: Any) -> None: ...


def test_an_injected_unguarded_or_defaulted_read_is_caught() -> None:
    found = violations(methods_of(_Injected, ours=lambda module: module == __name__))
    assert any(".read: takes ['variant'] unguarded" in v for v in found), found
    assert any(".rows: expect_space defaults to None" in v for v in found), found
    assert any(".sourced: takes ['start', 'stop'] unguarded" in v for v in found), found
    assert not any(".fine" in v for v in found), found


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
        "ogquery.rho.DenseRhoReader.row",
        "ogquery.rho.DenseRhoReader.pair",
        # Review round 3: the sidecar decoders an axis hands out.
        "ogquery.alid_sidecar.AlidSidecar.alids",
        "ogquery.panel_axis.PanelAlidIndex.alids",
        "ogquery.source_alid_sidecar.SourceAlidSidecar.sourced",
    ],
)
def test_the_reads_the_reviews_found_are_found_and_guarded(name: str) -> None:
    assert name in guarded_methods()
