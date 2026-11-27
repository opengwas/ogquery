"""Genotype Stores: on-demand LD computed from the pgens a store pins (ADR 0099, ADR 0100).

The read side of opengwasdb's `opengwasdb.ld.genotypes`, moved here from opengwasdb `964ac1b2`
(opengwas/ogquery#11; ADR 0002, amended), by its module names: `reader` (`GenotypeReader`, which
`ogquery.ld.open` opens a Genotype Store with), `pgen` (`Genotypes`, a pgen open for reading),
`adjust` (`Block` and `r`) and `layout` (the sidecar's root key, arrays and input pins). What builds
or validates one -- the PC Adjustment, the sidecar's writer, its validation -- stays there.

It needs pgenlib and threadpoolctl, the `genotype` extra (`pip install 'ogquery[genotype]'`), and is
the one package of ogquery that imports them. Importing this package imports neither: `pgen` imports
pgenlib when a pgen is opened, and only `reader` imports threadpoolctl.
"""

from ogquery.ld.genotypes.adjust import Block, r
from ogquery.ld.genotypes.pgen import Genotypes, import_pgenlib

__all__ = ["Block", "Genotypes", "import_pgenlib", "r"]
