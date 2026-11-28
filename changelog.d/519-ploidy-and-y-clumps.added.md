- **`VariantFilters` gains `ploidy: NotRequired[Ploidy]`, and `ld_stores_y` is a known, optional
  key of the `clumps` attribute (opengwas/opengwasdb#519, ADR 0109).** `Ploidy` types what the
  engine writes for a Store holding X, Y or MT, so its provenance type-checks with no `cast`; the
  `autosomes_only` docstring now says what it means for X, Y and MT. The LD Store format version
  is unchanged. A Clumps group without `ld_stores_y` reads as before; when it holds one, a
  non-object value, a non-object entry, or an entry lacking a key of `Y_ENTRY_KEYS` (an
  `ld_stores` entry's keys without the row counts) is refused as `MalformedClumps`, named. Tests
  show a Genotype Store answers `matrix` and `proxies` on X, Y and MT with literal r: the reader
  needed no change.
