"""The LD Store format as a reader reads it: encoding, thresholds, answers, refusals, provenance.

The semantics core of opengwasdb's `docs/spec/ld-store.md`, read side: a store's encoding of r,
`REncoding`, which owns dequantisation and the far code floor (§3; ADR 0090), t_far (§3), the
row and band geometry (§2, §4, §5, in `ogquery.ld.format.geometry`), the answer kind of a pair
(§8) and the decision for a request (§9). It also names the provenance types (§7). Every other
layer takes these from here, so no component rounds, thresholds or refuses by a rule of its own.

The build side of the format -- the build's default distances, the validation rules, the keys a
build derives and the geometry only a build or validation computes -- stays in opengwasdb, which
imports this (ogquery#2; ADR 0002).
"""

from ogquery.ld.format.encoding import T_FAR_MIN, T_FAR_Q, Codes, REncoding, t_far
from ogquery.ld.format.geometry import near_row_lengths
from ogquery.ld.format.provenance import (
    FORMAT_VERSION,
    PARTNER_ROWS_SINCE,
    READABLE_FORMATS,
    Adjustment,
    Build,
    Counts,
    Encoding,
    Panel,
    PcCheck,
    PcDiagnostic,
    PcFile,
    PcNull,
    PcScan,
    Provenance,
    Source,
    SourceKind,
    VariantFilters,
    claims_partner_rows,
    release_of,
)
from ogquery.ld.format.semantics import (
    AnswerKind,
    Decision,
    InputError,
    Refusal,
    RefusalReason,
    answer_kind,
    answer_kinds,
    judge_partner_request,
    judge_region,
    judge_request,
    judge_window,
)

__all__ = [
    "FORMAT_VERSION",
    "PARTNER_ROWS_SINCE",
    "READABLE_FORMATS",
    "T_FAR_MIN",
    "T_FAR_Q",
    "Adjustment",
    "AnswerKind",
    "Build",
    "Codes",
    "Counts",
    "Decision",
    "Encoding",
    "InputError",
    "Panel",
    "PcCheck",
    "PcDiagnostic",
    "PcFile",
    "PcNull",
    "PcScan",
    "Provenance",
    "REncoding",
    "Refusal",
    "RefusalReason",
    "Source",
    "SourceKind",
    "VariantFilters",
    "answer_kind",
    "answer_kinds",
    "claims_partner_rows",
    "judge_partner_request",
    "judge_region",
    "judge_request",
    "judge_window",
    "near_row_lengths",
    "release_of",
    "t_far",
]
