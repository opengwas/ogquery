"""The CPRA and `source_cpra` sidecars (format 0.4.0): the ALID ones under 0.4.0's names.

Spec §4.0 says the CPRA sidecar is ``variant_cpra_utf8.npy``, ``variant_cpra_offsets.npy``,
``variant_cpra_keys.npy`` and ``variant_cpra_order.npy``, "laid out as 0.3.0's ALID sidecar
is, with CPRA for ALID in every rule", and the `source_cpra` sidecar is
``variant_source_cpra_state.npy``, ``variant_source_cpra_utf8.npy`` and
``variant_source_cpra_offsets.npy``, "laid out as 0.3.0's `source_alid` sidecar is" (§21.5
maps each name). So the two classes here are the 0.3.0 ones with other file names: one
decoder of each layout, never two. Which of them a release reads is its `format_version`'s
(`alid_index.alid_index_kind`, `variant_axis.require_source_alid_sidecar_files`), never which
files happen to be present (ADR 0105 §5).

A row's text is a CPRA, `chrom:pos:REF:ALT`: the sidecar's methods keep the 0.3.0 names
(`alid`, `alids`, `search`), which here mean the row's CPRA and the row of a CPRA.
"""

from __future__ import annotations

from typing import ClassVar

from ogquery.alid_sidecar import AlidSidecar
from ogquery.source_alid_sidecar import SourceAlidSidecar

CPRA_UTF8_FILENAME = "variant_cpra_utf8.npy"
CPRA_OFFSETS_FILENAME = "variant_cpra_offsets.npy"
CPRA_KEYS_FILENAME = "variant_cpra_keys.npy"
CPRA_ORDER_FILENAME = "variant_cpra_order.npy"

#: The four files, in the order `alid_sidecar.encode_alids` returns its arrays.
CPRA_SIDECAR_FILENAMES: tuple[str, ...] = (
    CPRA_UTF8_FILENAME,
    CPRA_OFFSETS_FILENAME,
    CPRA_KEYS_FILENAME,
    CPRA_ORDER_FILENAME,
)

SOURCE_CPRA_STATE_FILENAME = "variant_source_cpra_state.npy"
SOURCE_CPRA_UTF8_FILENAME = "variant_source_cpra_utf8.npy"
SOURCE_CPRA_OFFSETS_FILENAME = "variant_source_cpra_offsets.npy"

#: The three files, in the order `source_alid_sidecar.encode_source_alids` returns its arrays.
SOURCE_CPRA_SIDECAR_FILENAMES: tuple[str, ...] = (
    SOURCE_CPRA_STATE_FILENAME,
    SOURCE_CPRA_UTF8_FILENAME,
    SOURCE_CPRA_OFFSETS_FILENAME,
)


class CpraSidecar(AlidSidecar):
    """A 0.4.0 release's CPRA sidecar: `AlidSidecar`'s layout in the `variant_cpra_*` files."""

    FILENAMES: ClassVar[tuple[str, ...]] = CPRA_SIDECAR_FILENAMES
    WHAT: ClassVar[str] = "CPRA sidecar"


class SourceCpraSidecar(SourceAlidSidecar):
    """A 0.4.0 release's `source_cpra` sidecar: `SourceAlidSidecar`'s layout in the
    `variant_source_cpra_*` files."""

    FILENAMES: ClassVar[tuple[str, ...]] = SOURCE_CPRA_SIDECAR_FILENAMES
    WHAT: ClassVar[str] = "source_cpra sidecar"
