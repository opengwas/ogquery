"""A Store Release's `format_version`: which ones exist, and whether this build reads one.

The version tables -- the release series this build reads and the versions it
writes (ADR 0041, spec §21) -- and the parse and check every open runs before it
reads anything else (ADR 0038 §2). What a writer stamps and refuses to write is
`opengwasdb.store.open`'s; it builds on these.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol

from ogquery.genome import SHA256_HEX

log = logging.getLogger(__name__)

#: The highest compatible remainder this build fully understands, per readable
#: **release series** (ADR 0041, spec §21). A series absent from this mapping is
#: rejected; a remainder above the one recorded here is read with a warning,
#: because the compatible axis is defined as one an older reader still reads
#: correctly.
#:
#: Two entries since #245: `0.1` (Zarr v2, read but never written again) and
#: `0.2` (Zarr v3 with sharding, what every builder writes from #247 on), read to
#: 0.2.3: `0.2.1` adds #430's region table, `0.2.2` #381's indel ALIDs spelt
#: reference first (ADR 0079), `0.2.3` #338's `n_missing_se_out_of_range` (ADR
#: 0068). The four pre-reset versions are not in it and are not readable -- see
#: `PRE_RESET_FORMAT_VERSIONS`.
#:
#: `0.3` since #467: 0.2.3's features, with the variable-length ALID sidecar in place
#: of the fixed-width `|S32` index (ADR 0105). A reader of the 0.2 series would find
#: no ALID index and misread every ALID lookup, so 0.3.0 is a new series that an
#: older package refuses by name (ADR 0041). #468-#472 add the rest of 0.3.0 in the
#: same version: the position and `source_alid` sidecars, and what nothing reads.
#:
#: `0.4` since #517 (issue opengwas/ogquery#19): every key a CPRA and every statistic
#: signed for ALT (spec §4.0, §5), on 0.3.0's features. A 0.3 reader would sign its
#: statistics for the key's first allele, REF, so it refuses 0.4.0 by series (spec §21.2).
SUPPORTED_FORMAT_VERSIONS: Mapping[tuple[int, ...], tuple[int, ...]] = MappingProxyType(
    {(0, 1): (0,), (0, 2): (3,), (0, 3): (0,), (0, 4): (0,)}
)

#: The format_versions this build writes, oldest first. A build stamps each release
#: the lowest whose features it uses (`format_version_for`; spec §21.3, ADR 0038 §3
#: as amended by the owner's decision of 2026-11-11, issue #430), and reads every
#: one in `SUPPORTED_FORMAT_VERSIONS`: supporting the *writing* of historical
#: formats would mean keeping every retired encoder alive and tested, for a use
#: case nobody has, but these two are written by the same encoders.
#:
#: `0.2.0` since #247: Zarr v3 with the sharding codec (ADR 0057), at the Dense
#: inner chunk and shard shapes ADR 0058 decided. The builders write it through
#: `opengwasdb.store.arrays`; `0.1.0` stays readable until a later decision deletes
#: the v2 reader (ADR 0057). `0.2.1` since #430: 0.2.0 plus a Ragged release's
#: optional, manifest-declared `regions.tsv`, so a 0.2.0 release is a 0.2.1 one
#: without the file. `0.2.2` since #381: 0.2.1 with an indel's ALID spelt reference
#: first rather than sorted (ADR 0079), `regions.tsv` still optional. `0.2.3`
#: since #338: 0.2.2 whose `analyses.tsv` records `n_missing_se_out_of_range` on
#: every row, and which holds no association whose SE is above 65,504 (ADR 0068).
#: Package v0.9.0 wrote 0.2.0 and 0.2.1 with sorted indels. `0.3.0` since #467: 0.2.3
#: with the variable-length ALID sidecar (ADR 0105). This build writes 0.3.0 for every
#: release it builds, completes or rewrites from 0.2.3 (`opengwasdb.store.rewrite`),
#: and 0.2.0 for one it converts from 0.1.0 (`opengwasdb.store.convert`), whose
#: indels it cannot re-spell. The 0.2 versions stay in this table because a pass
#: that adds arrays to a 0.2 release (`ogdb build-by-variant`) writes them, and a
#: 0.2.2 or 0.2.3 source is completed.
#: `0.4.0` since #517: CPRA keys and ALT-signed statistics (spec §4.0, §5, §21.3); written
#: by the builders and Reference Completion once epic #525's engine slices land, and never
#: by a rewrite or the converter. It is last: the indexes above name the 0.2 and 0.3 versions.
WRITTEN_FORMAT_VERSIONS: tuple[str, ...] = (
    "0.2.0",
    "0.2.1",
    "0.2.2",
    "0.2.3",
    "0.3.0",
    "0.4.0",
)

#: The last 0.2 version: 0.2.2's reference-first indels with 0.2.3's SE count, every
#: feature 0.3.0 keeps. The one version `ogdb rewrite-release` derives 0.3.0 from.
LAST_0_2_FORMAT_VERSION = WRITTEN_FORMAT_VERSIONS[3]

#: The first version whose ALID index is the variable-length sidecar (ADR 0105).
ALID_SIDECAR_FORMAT_VERSION = WRITTEN_FORMAT_VERSIONS[4]

#: The first version whose keys are CPRAs and whose statistics are signed for ALT (spec §4.0).
CPRA_FORMAT_VERSION = WRITTEN_FORMAT_VERSIONS[5]

#: The first `format_version` whose indel ALIDs spell the reference allele first
#: (ADR 0079). Every earlier one sorts an indel's alleles, so its deletion reads
#: like the insertion of the same sequence; nothing rewrites such a Store.
REFERENCE_FIRST_INDELS_SINCE = (0, 2, 2)


def indels_reference_first(version: str) -> bool:
    """Whether a Store of ``version`` spells an indel's ALID reference first.

    A Store before `REFERENCE_FIRST_INDELS_SINCE` is read as it was written: its
    indel ALIDs keep their sorted spelling, and a check or a join against it
    spells a source's indel the same way (ADR 0079).
    """
    return parse_format_version(version) >= REFERENCE_FIRST_INDELS_SINCE


def format_version_for(
    *,
    carries_regions: bool,
    reference_first_indels: bool,
    records_se_out_of_range: bool = True,
    alid_sidecar: bool = True,
    cpra_keys: bool = False,
    signed_for_alt: bool = False,
) -> str:
    """The `format_version` a release this build writes is stamped: the lowest whose
    features it uses (spec §21.3). The one rule every writer stamps by -- the builders,
    completion, the converter -- so a release no older reader can fully read says so,
    and every other one reads exactly as before.

    `carries_regions`: the release carries `regions.tsv`, which 0.2.1 adds.
    `reference_first_indels`: its indel ALIDs spell the reference first, which 0.2.2
    adds (ADR 0079); every release this build builds or completes does, one it
    converts from 0.1.0 does not.
    `records_se_out_of_range`: its `analyses.tsv` counts, on every row, the
    associations left out for an SE above 65,504, which 0.2.3 adds (ADR 0068). Every
    release this build builds or completes does. 0.2.3 is 0.2.2 and the count, so a
    release whose indels are sorted is stamped by its indels, whatever it counts.
    `alid_sidecar`: it carries the variable-length ALID sidecar rather than the
    fixed-width index, which 0.3.0 adds (ADR 0105) on all of 0.2.3's features. Every
    release this build builds, completes or rewrites does; one it converts from 0.1.0
    does not. A release with the sidecar and without one of 0.2.3's features is no
    version, and is refused rather than stamped.
    `cpra_keys` and `signed_for_alt`: its keys are CPRAs and its statistics are signed for
    ALT, which 0.4.0 adds (spec §4.0, §5, §21.3) on 0.3.0's features. They go together: a
    release with one and not the other is no version, and is refused rather than stamped.
    """
    if cpra_keys != signed_for_alt:
        raise ValueError(
            "a release's keys are CPRAs exactly when its statistics are signed for ALT "
            "(0.4.0, spec §21.3); this one has "
            f"cpra_keys={cpra_keys} and signed_for_alt={signed_for_alt}"
        )
    if cpra_keys:
        if not (reference_first_indels and records_se_out_of_range):
            raise ValueError(
                "a release whose keys are CPRAs is 0.4.0, which keeps 0.3.0's features: it "
                "also counts the SEs left out (ADR 0108 §8); this one does not "
                f"(reference_first_indels={reference_first_indels}, "
                f"records_se_out_of_range={records_se_out_of_range})"
            )
        return CPRA_FORMAT_VERSION
    if alid_sidecar:
        if not (reference_first_indels and records_se_out_of_range):
            raise ValueError(
                "a release carrying the variable-length ALID sidecar is 0.3.0, which also "
                "spells its indels reference first and counts the SEs left out (ADR 0105); "
                f"this one does not (reference_first_indels={reference_first_indels}, "
                f"records_se_out_of_range={records_se_out_of_range})"
            )
        return ALID_SIDECAR_FORMAT_VERSION
    if reference_first_indels:
        return WRITTEN_FORMAT_VERSIONS[3 if records_se_out_of_range else 2]
    return WRITTEN_FORMAT_VERSIONS[1] if carries_regions else WRITTEN_FORMAT_VERSIONS[0]

#: The format_version a release that uses no optional feature is stamped: what
#: every build of this package writes for a release without `regions.tsv`, and
#: what opengwasdb-stores' engine update probes as the format a locked engine
#: writes.
CURRENT_FORMAT_VERSION = format_version_for(carries_regions=False, reference_first_indels=True)

#: The Zarr on-disk format each readable release series is stored in.  The
#: series-to-layout coupling is part of the format contract, so it lives beside
#: the version tables rather than in each caller that needs it (#264 review).
ZARR_FORMAT_BY_SERIES: Mapping[tuple[int, ...], int] = MappingProxyType(
    {(0, 1): 2, (0, 2): 3, (0, 3): 3, (0, 4): 3}
)

#: The first `format_version` whose keys are CPRAs, `chrom:pos:REF:ALT`, and whose statistics
#: are signed for ALT: the series that names its sidecars `variant_cpra_*` and
#: `variant_source_cpra_*` and its Variant Table columns `ref`, `alt`, `cpra` and
#: `source_cpra` (spec §4.0, §21.5). Read by the release's series, never by which files
#: exist (ADR 0105 §5).
CPRA_SINCE = (0, 4, 0)


def carries_cpra(version: str) -> bool:
    """Whether a Store of ``version`` is keyed by CPRA (0.4.0 on), not by ALID (0.3 and
    earlier), and so signs its statistics for ALT, not for the key's first allele."""
    return parse_format_version(version) >= CPRA_SINCE

#: The first `format_version` whose ALID index is the variable-length sidecar
#: (`ogquery.alid_sidecar`, ADR 0105) and not the fixed-width pair.
ALID_SIDECAR_SINCE = tuple(int(part) for part in ALID_SIDECAR_FORMAT_VERSION.split("."))


def carries_alid_sidecar(version: str) -> bool:
    """Whether a Store of ``version`` indexes its ALIDs with the variable-length sidecar."""
    return parse_format_version(version) >= ALID_SIDECAR_SINCE


#: The first `format_version` whose `index.sqlite` has no `metadata` table -- the
#: `dense` layout blob, `schema_version`, `n_variants`, `n_analyses` -- (ADR 0102,
#: #471): `manifest.json` is the one copy of the layout. 0.3.0 is one format version,
#: so it is the ALID sidecar's.
INDEX_METADATA_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_index_metadata(version: str) -> bool:
    """Whether a Store of ``version`` carries `index.sqlite`'s `metadata` table.

    Every release of such a version carries `index.sqlite` (a Ragged one's may be
    empty). From 0.3.0 it holds only `completion_quality`, so only a release that holds
    its own carries one (`opengwasdb.store.open.ships_index`).
    """
    return parse_format_version(version) < INDEX_METADATA_DROPPED_SINCE


#: The first `format_version` with no `variant_aliases` (ADR 0102, #471): an rsID the
#: rsID index lacks is a miss, and the Store Reader never opens `index.sqlite`.
VARIANT_ALIASES_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_variant_aliases(version: str) -> bool:
    """Whether a Store of ``version`` may name an rsID in `index.sqlite`'s
    `variant_aliases`, which a Dense or Hybrid query falls back to (#476 retires it)."""
    return parse_format_version(version) < VARIANT_ALIASES_DROPPED_SINCE


#: The first `format_version` whose Top-Hit tiers store `z` but neither `abs_z` nor
#: `p_value`, which the validator recomputes from `z` (ADR 0102, #471).
STORED_P_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_stored_p(version: str) -> bool:
    """Whether a Store of ``version``'s Top-Hit tiers store `abs_z` and `p_value`."""
    return parse_format_version(version) < STORED_P_DROPPED_SINCE


#: The first `format_version` that answers a range from the position sidecar
#: (`ogquery.position_sidecar`, ADR 0105 §6) and not through tabix.
#: 0.3.0 is one format version: every release that carries the ALID sidecar carries
#: this one too.
POSITION_SIDECAR_SINCE = ALID_SIDECAR_SINCE


def carries_position_sidecar(version: str) -> bool:
    """Whether a Store of ``version`` carries the position sidecar a range is answered from."""
    return parse_format_version(version) >= POSITION_SIDECAR_SINCE


#: The first `format_version` with no `variant_offsets.npy` (ADR 0102, ADR 0105 §7,
#: #469): its Store Reader never reads `variants.tsv.gz`, so nothing seeks a row's BGZF
#: offset; a row's identity and rsID come from the sidecars, and the variant count from
#: the ALID sidecar. 0.3.0 is one format version, so it is the ALID sidecar's.
VARIANT_OFFSETS_DROPPED_SINCE = ALID_SIDECAR_SINCE


def carries_variant_offsets(version: str) -> bool:
    """Whether a Store of ``version`` carries `variant_offsets.npy`, each row's BGZF
    offset in its Variant Table, and is read through that table: the 0.2 and 0.1 series."""
    return parse_format_version(version) < VARIANT_OFFSETS_DROPPED_SINCE


#: The first `format_version` that carries each row's `source_alid` in its own sidecar
#: (`ogquery.source_alid_sidecar`, ADR 0105 §8, #470) and not only in the
#: Variant Table's eighth column. 0.3.0 is one format version, so it is the ALID sidecar's.
SOURCE_ALID_SIDECAR_SINCE = ALID_SIDECAR_SINCE


def carries_source_alid_sidecar(version: str) -> bool:
    """Whether a Store of ``version`` names each row's `source_alid` from its sidecar."""
    return parse_format_version(version) >= SOURCE_ALID_SIDECAR_SINCE


#: The first `format_version` whose Hybrid Dense Component carries no Variant Axis
#: and no `analyses.tsv` of its own: it reads the shared ones through
#: `dense_to_shared.npy` (#472, ADR 0102 §4). 0.3.0 is one format version.
DENSE_COMPONENT_SHARES_AXIS_SINCE = ALID_SIDECAR_SINCE


def dense_component_shares_axis(version: str) -> bool:
    """Whether a Hybrid Dense Component of ``version`` reads its variants and Analyses
    from the release that nests it, rather than carrying its own copy of each."""
    return parse_format_version(version) >= DENSE_COMPONENT_SHARES_AXIS_SINCE


def zarr_format_for_version(version: str, *, source: str = "release") -> int:
    """The Zarr on-disk format a readable ``format_version`` is stored in.

    Runs the reader's own version check first, so a caller can ask this alone
    and get both "is it readable" and "what layout is it" -- a caller that
    instead pattern-matched a version string would be the second place the
    version-to-layout rule lives, and the two would eventually disagree.
    """
    series, _ = split_format_version(version)
    check_format_version(version, source=source)
    try:
        return ZARR_FORMAT_BY_SERIES[series]
    except KeyError:
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, series "
            f"{_series_text(series)}, which this build reads but has no Zarr on-disk "
            "layout for; the version table and the layout table disagree"
        ) from None

#: The versions the format carried before the reset, and what each one was.
#: Every one is two-component, so the parser rejects it on shape alone; naming
#: them here is what turns that rejection into an instruction rather than a
#: complaint about punctuation (issue #143). No store in any of them was ever
#: published outside the core team, and this build decodes none of them: the
#: bytes of a `3.0` release are the bytes of a `0.1.0` one, but `0.1`, `1.0`
#: and `2.0` are genuinely different encodings and reading them was deleted
#: rather than deprecated.
PRE_RESET_FORMAT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "0.1": "the first pre-release format, float16 throughout",
        "1.0": "fixed-point z (issue #114)",
        "2.0": "residual-coded eaf (issue #116)",
        "3.0": "residual-coded se (issue #118)",
    }
)


class UnsupportedFormatVersion(Exception):
    """A release declares a format_version this build cannot interpret."""


class MalformedFormatVersion(UnsupportedFormatVersion):
    """A release's format_version is not ``MAJOR.MINOR.PATCH``.

    A subclass rather than a `ValueError`: to a caller deciding whether it can
    read a release, "the version is nonsense" and "the version is from the
    future" are the same answer, and both must stop the read.
    """


def parse_format_version(version: str) -> tuple[int, int, int]:
    """``"0.1.0"`` -> ``(0, 1, 0)``. Raises `MalformedFormatVersion` otherwise.

    Three components, not two, and that shape change is the safety mechanism of
    the reset rather than a cosmetic choice (issue #143, ADR 0041). Pre-reset
    releases are stamped ``0.1``, and a reset that reused that string would
    give two different formats one name -- a store that reads as plausible and
    is wrong. A two-component value has no reading here at all, so every such
    release is refused loudly instead.
    """
    text = str(version)
    was = PRE_RESET_FORMAT_VERSIONS.get(text)
    if was is not None:
        raise UnsupportedFormatVersion(
            f"format_version {text!r} is a pre-release format ({was}), which this "
            "build no longer reads: the format was reset to "
            f"{CURRENT_FORMAT_VERSION} and the decoders for every earlier version "
            "were deleted (ADR 0041, issue #143). Rebuild the release from source "
            "-- a store in one of those formats cannot be validated, completed or "
            "queried by this build"
        )
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MalformedFormatVersion(
            f"format_version {text!r} is not MAJOR.MINOR.PATCH (ADR 0041, spec §21)"
        )
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def split_format_version(version: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``"0.1.0"`` -> ``((0, 1), (0,))``: a version's series and its remainder.

    The **series** is the part a compatible change may not move, and the
    **remainder** is the part it may. Which is which follows semantic
    versioning's own rule, that the *leftmost non-zero component is the
    breaking one*: below ``1.0.0`` a format is declaring itself unsettled, so
    ``MINOR`` carries an incompatible change and ``PATCH`` does not; from
    ``1.0.0`` it is ``MAJOR``, and ``MINOR`` joins the remainder.

    Both halves come from one function because they are one rule. A caller that
    took the series here and worked out the remainder for itself would be
    stating that rule a second time, and the two would eventually disagree --
    `check_format_version` compares what this returns and knows nothing about
    which component is which.
    """
    major, minor, patch = parse_format_version(version)
    if major == 0:
        return (0, minor), (patch,)
    return (major,), (minor, patch)


def check_format_version(version: str, *, source: str = "release") -> None:
    """Reject a `format_version` this build cannot interpret (ADR 0041 §2).

    Rejection is on the **release series** alone: a build carries the series it
    implements, not a list of every version it has ever seen. A *newer
    remainder* within a known series is accepted and warned about -- accepting
    it is the definition of a compatible change, and the warning is how a
    mis-classified one (a change that should have moved the series) becomes
    visible instead of silently returning partial data.
    """
    series, remainder = split_format_version(version)
    known = SUPPORTED_FORMAT_VERSIONS.get(series)
    if known is None:
        readable = [_series_text(known_series) for known_series in SUPPORTED_FORMAT_VERSIONS]
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, a release series this "
            f"build does not implement; readable series: {sorted(readable)}"
        )
    if remainder > known:
        log.warning(
            "%s declares format_version=%s, a newer compatible version than this build "
            "knows (%s). Reading it as that: anything added after it is not visible "
            "here, and if any of it changes how existing arrays decode it was "
            "misclassified as compatible.",
            source,
            version,
            _series_text(series) + "." + ".".join(str(part) for part in known),
        )


def _series_text(series: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in series)


#: The vocabularies of a release's `cpra_normal_form` (spec §2, §21.2). A 0.4.0 release holds
#: exactly `CPRA_NORMAL_FORM_VALUES`; the derived forms of older releases use the others.
CPRA_NORMAL_FORM_VALUES: Mapping[str, str] = MappingProxyType(
    {"alleles": "ref_alt", "signed_for": "alt", "indels": "left_aligned"}
)

#: The one reference assembly a 0.4.0 release has: GRCh38, the one a Byte Genome exists for
#: (spec §4.0).
CPRA_ASSEMBLY: Final = "GRCh38"


class _Absent:
    """The type of `ABSENT`: a manifest key that is not there, as distinct from one that is
    `null` (spec §2 says an earlier release's `cpra_normal_form` is "absent, not `null`")."""

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT: Final = _Absent()


@dataclass(frozen=True)
class CpraSequence:
    """The reference sequence a CPRA is relative to: its assembly and sequence digest."""

    assembly: str
    digest: str


@dataclass(frozen=True)
class CpraNormalForm:
    """The form a release's keys and signs take (ADR 0108 §9, spec §21.2): which alleles a
    key spells in which order, which allele a statistic is signed for, how indels are
    spelt, and the sequence they were formed against (`None` where a stamp names none).

    Two releases join only on equal forms (`require_joinable`)."""

    alleles: str
    signed_for: str
    indels: str
    sequence: CpraSequence | None

    def describe(self) -> str:
        sequence = (
            "no recorded sequence"
            if self.sequence is None
            else f"{self.sequence.assembly} sequence {self.sequence.digest}"
        )
        return (
            f"alleles={self.alleles}, signed_for={self.signed_for}, "
            f"indels={self.indels}, {sequence}"
        )


#: The form of a 0.2.2 to 0.3.x release, derived from its stamp (spec §21.2).
_ALID_FORM = CpraNormalForm(
    "snv_sorted_indel_reference_first", "a1", "as_spelt", None
)
#: The form of a release before 0.2.2, or a converted one whose indels sort.
_SORTED_FORM = CpraNormalForm("sorted", "a1", "as_spelt", None)
#: The deviation a converted release lists when it sorts its indels (ADR 0108 §9).
SORTED_INDELS_DEVIATION = "variant_table.indel_order"


class UnsupportedCpraNormalForm(UnsupportedFormatVersion):
    """A release's `cpra_normal_form` is absent where 0.4.0 requires it, present where an
    earlier format forbids it, or holds a value outside spec §2's: refused at open, as a
    value outside a controlled vocabulary is (spec §21.2)."""


class CpraNormalFormMismatch(ValueError):
    """Two releases whose `CpraNormalForm`s differ were asked to join. A join on the key
    string would match an SNV whose REF sorts first and mis-sign it; there is no override
    (ADR 0108 §9)."""


class MixedCpraNormalForms(UserWarning):
    """A release was opened whose form differs from one already open in this process.
    Opening is not a join, so it warns; a caller that joins makes it an error with
    ``warnings.simplefilter("error", MixedCpraNormalForms)``."""


class _HasManifestFields(Protocol):
    @property
    def format_version(self) -> str: ...
    @property
    def reference_assembly(self) -> str: ...
    @property
    def provenance(self) -> dict[str, Any]: ...
    @property
    def cpra_normal_form_entry(self) -> Any: ...


def cpra_normal_form(manifest: _HasManifestFields) -> CpraNormalForm:
    """A release's `CpraNormalForm`: a 0.4.0 release's manifest entry, checked against spec
    §2, and for an earlier release the form its stamp gives (spec §21.2). The one place the
    stamp is read for it. Raises `UnsupportedCpraNormalForm` for an entry 0.4.0 does not
    allow, and for one an earlier release carries."""
    entry = manifest.cpra_normal_form_entry
    version = manifest.format_version
    if not carries_cpra(version):
        if entry is not ABSENT:
            raise UnsupportedCpraNormalForm(
                f"a release of format_version {version} declares cpra_normal_form, which "
                "only 0.4.0 and later releases carry (spec §2): its form is derived from "
                "its stamp"
            )
        return _derived_form(manifest)
    return _declared_form(entry, manifest)


def _derived_form(manifest: _HasManifestFields) -> CpraNormalForm:
    converted = manifest.provenance.get("converted_from")
    deviations = converted.get("deviations") if isinstance(converted, dict) else None
    listed = (
        isinstance(deviations, (list, tuple)) and SORTED_INDELS_DEVIATION in deviations
    )
    if listed or not indels_reference_first(manifest.format_version):
        return _SORTED_FORM
    return _ALID_FORM


def _declared_form(entry: Any, manifest: _HasManifestFields) -> CpraNormalForm:
    version = manifest.format_version
    if entry is ABSENT or entry is None:
        raise UnsupportedCpraNormalForm(
            f"a release of format_version {version} must declare cpra_normal_form (spec "
            "§2), and this one does not"
        )
    if manifest.reference_assembly != CPRA_ASSEMBLY:
        raise UnsupportedCpraNormalForm(
            f"a release of format_version {version} has reference_assembly "
            f"{manifest.reference_assembly!r}, but a 0.4.0 release's is {CPRA_ASSEMBLY} (spec "
            "§4.0): no other assembly has a Byte Genome"
        )
    expected = dict(CPRA_NORMAL_FORM_VALUES)
    if not isinstance(entry, dict) or set(entry) != {*expected, "sequence"}:
        raise UnsupportedCpraNormalForm(
            f"cpra_normal_form of a {version} release must be an object of exactly "
            f"{sorted([*expected, 'sequence'])} (spec §2); this one is {entry!r}"
        )
    for key, value in expected.items():
        if entry[key] != value or not isinstance(entry[key], str):
            raise UnsupportedCpraNormalForm(
                f"cpra_normal_form.{key} of a {version} release must be {value!r} (spec §2); "
                f"this one is {entry[key]!r}"
            )
    sequence = entry["sequence"]
    if not isinstance(sequence, dict) or set(sequence) != {"assembly", "digest"}:
        raise UnsupportedCpraNormalForm(
            "cpra_normal_form.sequence must be an object of exactly ['assembly', 'digest'] "
            f"(spec §2); this one is {sequence!r}"
        )
    assembly, digest = sequence["assembly"], sequence["digest"]
    if assembly != manifest.reference_assembly or not isinstance(assembly, str):
        raise UnsupportedCpraNormalForm(
            f"cpra_normal_form.sequence.assembly is {assembly!r}, but the release's "
            f"reference_assembly is {manifest.reference_assembly!r}; spec §2 requires them equal"
        )
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        raise UnsupportedCpraNormalForm(
            "cpra_normal_form.sequence.digest must be 64 lower-case hex characters "
            f"(spec §2); this one is {digest!r}"
        )
    form = CpraNormalForm(
        entry["alleles"], entry["signed_for"], entry["indels"], CpraSequence(assembly, digest)
    )
    _require_declarations(manifest, form)
    return form


_REFERENCE_SEQUENCE_KEYS = ("assembly", "digest", "sha256", "format", "format_version", "path")


def _require_declarations(manifest: _HasManifestFields, form: CpraNormalForm) -> None:
    """A 0.4.0 release MUST also declare `provenance.reference_sequence` and
    `provenance.admission`, each of the JSON types spec §2 gives. Their content is the engine's
    validation rules' (§20.0a); here only what a reader relies on is checked: the entries are
    there, their shape and types, and the two equalities §2 states."""
    provenance = manifest.provenance
    where = "provenance.reference_sequence"
    sequence = provenance.get("reference_sequence", ABSENT)
    if not isinstance(sequence, dict) or set(sequence) != {*_REFERENCE_SEQUENCE_KEYS, "source"}:
        raise UnsupportedCpraNormalForm(
            f"a 0.4.0 release must declare {where}, an object of exactly "
            f"{sorted([*_REFERENCE_SEQUENCE_KEYS, 'source'])} (spec §2); this one is "
            f"{sequence!r}"
        )
    for key in _REFERENCE_SEQUENCE_KEYS:
        if not isinstance(sequence[key], str):
            raise UnsupportedCpraNormalForm(
                f"{where}.{key} must be a JSON string (spec §2); this one is {sequence[key]!r}"
            )
    if not isinstance(sequence["source"], dict):
        raise UnsupportedCpraNormalForm(
            f"{where}.source must be an object (spec §2); this one is {sequence['source']!r}"
        )
    if form.sequence is not None and sequence["assembly"] != form.sequence.assembly:
        raise UnsupportedCpraNormalForm(
            f"{where}.assembly is {sequence['assembly']!r}, not the release's "
            f"{form.sequence.assembly!r} (spec §2)"
        )
    if form.sequence is not None and sequence["digest"] != form.sequence.digest:
        raise UnsupportedCpraNormalForm(
            f"{where}.digest is {sequence['digest']!r}, not cpra_normal_form.sequence.digest "
            f"{form.sequence.digest!r} (spec §2 requires them equal)"
        )
    if not SHA256_HEX.fullmatch(sequence["sha256"]):
        raise UnsupportedCpraNormalForm(
            f"{where}.sha256 must be 64 lower-case hex characters (spec §2); this one is "
            f"{sequence['sha256']!r}"
        )
    _require_admission(provenance.get("admission", ABSENT))


def _require_admission(admission: Any) -> None:
    where = "provenance.admission"
    if not isinstance(admission, dict) or set(admission) != {"rules", "dropped"}:
        raise UnsupportedCpraNormalForm(
            f"a 0.4.0 release must declare {where}, an object of exactly ['dropped', 'rules'] "
            f"(spec §2); this one is {admission!r}"
        )
    if not isinstance(admission["rules"], str):
        raise UnsupportedCpraNormalForm(
            f"{where}.rules must be a JSON string (spec §2); this one is {admission['rules']!r}"
        )
    dropped = admission["dropped"]
    if not isinstance(dropped, dict) or any(
        not isinstance(reason, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in dropped.items()
    ):
        raise UnsupportedCpraNormalForm(
            f"{where}.dropped must be an object of reasons to JSON integers of at least 0 "
            f"(spec §2); this one is {dropped!r}"
        )


def cpra_declarations(manifest: _HasManifestFields) -> tuple[Any, Any, Any]:
    """What a 0.4.0 manifest declares as spec §2 says a Hybrid's two manifests must hold equal:
    `cpra_normal_form`, `provenance.reference_sequence` and `provenance.admission`, each
    `ABSENT` where the manifest has none."""
    return (
        manifest.cpra_normal_form_entry,
        manifest.provenance.get("reference_sequence", ABSENT),
        manifest.provenance.get("admission", ABSENT),
    )


def require_equal_declarations(
    a: _HasManifestFields, a_name: str, b: _HasManifestFields, b_name: str
) -> None:
    """Refuse (`UnsupportedCpraNormalForm`) two manifests of one Hybrid, `a_name` and `b_name`
    being where each was read, that declare other than the same (spec §2: "each equal in the
    two"). Judged before either is read for its own form."""
    if cpra_declarations(a) != cpra_declarations(b):
        raise UnsupportedCpraNormalForm(
            f"{a_name} and {b_name} are one Hybrid, whose cpra_normal_form, "
            "provenance.reference_sequence and provenance.admission must each be equal in its "
            "two manifests (spec §2), and they are not"
        )


class _HasForm(Protocol):
    @property
    def cpra_normal_form(self) -> CpraNormalForm: ...
    @property
    def release_id(self) -> str: ...


def require_joinable(a: _HasForm, b: _HasForm) -> None:
    """Raise `CpraNormalFormMismatch` unless releases ``a`` and ``b`` (a `StoreRelease`, or
    any reader exposing `cpra_normal_form` and `release_id`) have equal forms. Every join
    calls it before comparing a key (spec §21.2). There is no override: the remedy is to
    rebuild the older release."""
    form_a, form_b = a.cpra_normal_form, b.cpra_normal_form
    if form_a == form_b:
        return
    raise CpraNormalFormMismatch(
        f"cannot join release {a.release_id!r} ({form_a.describe()}) with release "
        f"{b.release_id!r} ({form_b.describe()}): their keys or signs differ, so a join "
        "on the key would match an SNV whose REF sorts first and mis-sign it. Rebuild the "
        "older release from its sources at format 0.4.0 (spec §21.2); there is no override"
    )


#: Every release opened in this process, by form: the first release of each form. Process
#: wide, as spec §21.2 says; `_forget_opened_forms` clears it (tests).
_OPENED: dict[CpraNormalForm, str] = {}
_WARNED_PAIRS: set[frozenset[CpraNormalForm]] = set()


def record_opened(form: CpraNormalForm, release_id: str) -> None:
    """Record that a release of ``form`` was opened, warning `MixedCpraNormalForms` once per
    pair of forms when another form is already open. Called by `open_release` and by the LD
    Store's open (#462)."""
    for other_form, other_id in list(_OPENED.items()):
        pair = frozenset((form, other_form))
        if other_form == form or pair in _WARNED_PAIRS:
            continue
        message = (
            f"release {release_id!r} ({form.describe()}) is open beside release "
            f"{other_id!r} ({other_form.describe()}): their forms differ, so a join of the two "
            "is refused (require_joinable). To make that an error, use "
            'warnings.simplefilter("error", MixedCpraNormalForms)'
        )
        log.warning(message)
        warnings.warn(message, MixedCpraNormalForms, stacklevel=3)
        # Remembered only once the warning has been issued: a caller that makes it an error
        # is refused again on retry, never let through (ADR 0108 §9).
        _WARNED_PAIRS.add(pair)
    _OPENED.setdefault(form, release_id)


def _forget_opened_forms() -> None:
    """Clear the registry `record_opened` keeps (for tests; nothing else forgets)."""
    _OPENED.clear()
    _WARNED_PAIRS.clear()
