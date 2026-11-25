"""Opening and reading the Zarr arrays of an LD Store (layer 1; ogld#4).

The store layer knows Zarr and the codes; nothing above it does. Queries read through the
`ogquery.ld.query.reader.Reader` protocol, which `StoreReader` satisfies. `VariantTable` is the
reader's variant half — the Variant Set's rows and the ALID lookup — which any reader composes
(ogld#94). Writing and validating a store are opengwasdb's (ADR 0002).
"""

from ogquery.ld.store.reader import StoreReader
from ogquery.ld.store.tables import Variants
from ogquery.ld.store.variants import VariantTable

__all__ = ["StoreReader", "VariantTable", "Variants"]
