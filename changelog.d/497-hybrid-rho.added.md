- **A Hybrid release answers `rho()`, `rho_row()` and `rho_matrix()` from its Dense
  Component's Rho Matrix (opengwas/opengwasdb#497).** A Hybrid's matrix is
  `dense/data.zarr/rho` and records its Overflow-only Analyses (`overflow_only`, read
  exactly); naming one raises `OverflowOnlyAnalysisError` (`fields`: `analysis_id`), never
  a NaN, and a row or the whole matrix leaves them out. `ogquery.digests` asks the Rho
  readers (a `rho` section) and `differences` compares it. The contract bundle is
  opengwasdb's bundle 2 (`355449d7`), which carries a Rho Matrix on its Dense and Hybrid
  releases.

  A Hybrid Rho Matrix built before opengwasdb#496 records no `overflow_only`, so every
  Rho query of that release now refuses with a `ValueError` naming the matrix; rebuild it
  with `opengwasdb build-dense-rho` on the Hybrid release.
