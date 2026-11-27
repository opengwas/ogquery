"""PC-adjusted, A1-signed r for on-demand rows: a Block and `r` (ADR 0099; opengwasdb's).

From opengwasdb `964ac1b2`'s `opengwasdb/ld/genotypes/adjust.py` (opengwas/ogquery#11; ADR 0002,
amended): the names a Genotype Store's read reaches. The PC Adjustment that makes a Block's
projections and norms -- `covariates`, `residuals`, `prepare`, `Covariates` and `FLAT` -- is a
build's, and stays in opengwasdb, which imports these two for its tiles. The covariates are an
intercept and the kept PCs in sample order; at k = 0 the intercept alone centres the dosages,
giving Pearson r on complete genotypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Block:
    """ALT dosages and their PC projections, residual norms and A1 signs (ADR 0099).

    Rows retain the sample order of the covariate basis. Two blocks passed to `r` must use
    the same samples and basis; a row-against-window call is a one-row slice of a block.
    """

    calls: npt.NDArray[np.float64]
    projection: npt.NDArray[np.float64]
    norm: npt.NDArray[np.float64]
    sign: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.calls)

    def __getitem__(self, rows: slice) -> Block:
        return Block(self.calls[rows], self.projection[rows], self.norm[rows], self.sign[rows])


def r(rows: Block, span: Block) -> npt.NDArray[np.float64]:
    """A1-signed r for a block against a span, including a single row against a window.

    For complete hard calls the float64 BLAS raw dot is exact: every partial sum is a
    whole number at most 4n. Only the (k + 1)-term correction and normalisation round.
    Dosage inputs use the same formula, without the integer exactness claim (ADR 0099).
    Quantise the result only through `ogquery.ld.format.REncoding`.
    """
    out = rows.calls @ span.calls.T
    out -= rows.projection @ span.projection.T
    out /= rows.norm[:, None]
    out /= span.norm[None, :]
    out *= rows.sign[:, None]
    out *= span.sign[None, :]
    return out
