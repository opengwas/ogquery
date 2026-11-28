"""The Indexed Variant Subset adapter (ADR 0061): a Dense release's Analysis-major copy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ogquery.associations.cells import Cells, require_within
from ogquery.associations.record import Associations, Inspection, Status
from ogquery.guard import VARIANT, CellSpaces, IndexSpace, require_cells

if TYPE_CHECKING:
    from ogquery.indexed import IndexedSubset


class SubsetReader:
    """Reads one Analysis at a time from a published Indexed Variant Subset.

    Opening refuses an unknown, stale, incomplete or corrupt subset with the
    `IndexedSubsetError` family, before any value is decoded. Like the grid it
    copies, a subset stores no association at a missing cell, so `read` returns
    the present cells, in Store Variant Index order.
    """

    carries_n = False

    def __init__(
        self, subset: IndexedSubset, *, source: str, variant_space: IndexSpace | None = None
    ) -> None:
        #: Its columns are the Analyses of the release it indexes; its rows that release's
        #: variant axis unless the caller says they are another (`variant_space`).
        self.spaces = CellSpaces(
            IndexSpace(subset.identity, VARIANT) if variant_space is None else variant_space,
            subset.index_space,
        )
        self._subset = subset
        self._source = source

    @classmethod
    def open(
        cls, store_path: str | Path, name: str, *, variant_space: IndexSpace | None = None
    ) -> SubsetReader:
        # Imported here: the subset module's own validation classifies cells
        # through this package, so a module-level import would be circular.
        from ogquery.indexed import open_indexed_subset

        return cls(
            open_indexed_subset(store_path, name),
            source=f"Indexed Variant Subset {name!r} of {store_path}",
            variant_space=variant_space,
        )

    def read(
        self, cells: Cells, *, observed_only: bool = False, expect_spaces: CellSpaces
    ) -> Associations:
        """The Analysis's present cells; raises `InvalidAssociations` on an invalid one."""
        require_cells(self.spaces, expect_spaces, "the cells")
        found = self.inspect(cells, expect_spaces=expect_spaces).associations(self._source)
        keep = found.status != Status.MISSING
        if observed_only:
            keep &= found.status != Status.IMPUTED
        return replace(
            found.where(keep),
            variant_space=self.spaces.variant,
            analysis_space=self.spaces.analysis,
        )

    def inspect(self, cells: Cells, *, expect_spaces: CellSpaces) -> Inspection:
        require_cells(self.spaces, expect_spaces, "the cells")
        if cells.kind != "analysis":
            raise TypeError(f"an Indexed Variant Subset reads one Analysis, not {cells.kind!r}")
        col = cells.analysis_index
        require_within(cells, self._source, n_analyses=self._subset.n_analyses)
        decoded = self._subset.decode_analysis(col, expect_space=self._subset.index_space)
        eaf = (
            np.full(len(decoded.z), np.nan, dtype=np.float32)
            if decoded.eaf is None
            else decoded.eaf
        )
        return Inspection.of(
            decoded.variant_index, col, decoded.z, decoded.se, eaf, decoded.imputed
        )
