**The indices a query answers with are scoped to an index space, and the entry points a cached integer can reach take the space it came from and refuse a mismatch (opengwas/ogquery#13, opengwasdb ADR 0109 §2).** A splice renumbers `analysis_index` and `variant_index`, so an index cached from another release is still in range and names a different Analysis or variant without failing. An `IndexSpace` is the Store and release (`ReleaseIdentity`) and the axis it numbers (`variant`, a Hybrid's `variant:panel` rows, `analysis`), so two Stores that share a `release_id`, and a Hybrid's panel and shared numberings, are told apart.

**Constructors keep their signatures.** `VariantAxis`, `PanelVariantAxis`, `DenseRhoReader`, `TopHitTier(s)`, `RaggedCSRReader`, `ByVariantReader`, `CsrReader`, `DenseGridReader`, `SubsetReader`, `IndexedSubset` and `AnalysesIndex` learn their space from the release they open, never from a caller (`ogquery.manifest.release_scope_of`, the one place it is decided). A Hybrid's Dense Component is known as one by its real path: a `dense/` directory under a Hybrid manifest of its own release. Its Analyses are the Hybrid's and its rows the Hybrid's `variant:panel` rows, however it is opened. A `dense/` under a Hybrid of another release, or under a manifest that cannot be read, is refused, and so is `open_release` of it, rather than read as a release of its own. Where no manifest describes the store a reader refuses with `ReleaseIdUnknown`. An optional `identity=` is accepted only where no manifest is reachable (`CsrReader.open` now forwards it), and refused where one says otherwise. Ids that are missing, empty or the text `None` refuse (`StoreManifest` is the one parser; a null `release_id` no longer becomes `'None'`).

**A manifest is bound to the Zarr store it describes.** A `data.zarr` is read only beside a manifest that describes it: the layout (a `ragged/` component or none, and the store's recorded `layout`), the completion state the store records (a Dense grid's, or a Ragged release's sequence), and its dimensions (Analyses against `analyses.tsv`, grid rows or by-variant index entries against the variant axis, or a Dense Component's `dense_to_shared.npy`). A release whose `data.zarr` is a link into another release is refused. `StoreRelease.top_hit_tiers(root)` refuses arrays that are not the release's own `data.zarr`, so a Hybrid's tiers cannot be read from its Dense Component's. **The limit:** a store that agrees with another release's manifest on all of these, an exact structural twin, cannot be told from its own, since nothing in a store names the release that wrote it. That needs a writer-stamped identity in the store.

**What breaks is the keyword**, required and keyword-only, on:

- `expect_space=`:
  - `VariantAxis` and `PanelVariantAxis`: `by_index`, `by_indices`, `identity_by_indices`, `rsids_by_indices`, `source_alids_by_indices` and `rows_by_indices`, plus `VariantAxis.sidecar_records`;
  - `DenseRhoReader.row`, `pair` and `matrix`;
  - `IndexedSubset.decode_analysis`;
  - `ByVariantReader.rows_for_variant`, `rows_for_variant_range` and `decode`;
  - `RaggedCSRReader.span`.
- `expect_space=`, required only when an index is given: `TopHitTier.read` and `clumps` (an `analysis_index`), and `RaggedCSRReader.se_slice` (an `analysis_index`).
- `expect_spaces=`:
  - `RaggedCSRReader.segment_positions`, `eaf_pairs`, `n_pairs` and `pair_positions`;
  - `read`, `inspect` and `z` of `DenseGridReader` and `CsrReader`;
  - `read` and `inspect` of `SubsetReader`.
- `expect_spaces=`, required only with a region: `windows` of `DenseGridReader` and `CsrReader`. `windows()` over the whole grid or component needs none.
- `Associations.remap_variants(to_shared, *, variant_space=)`: the space the remapped rows are in, required. Pass `None` for an unstamped record.

A mismatch raises `ogquery.guard.ReleaseMismatchError` (a `ValueError`) naming both spaces. A test walks every public method of every `ogquery` class, and fails on one that takes an index with neither keyword unless it is exempt with a stated reason (`tests/unit/test_index_space_sweep.py`).

`Associations` carries `variant_space` and `analysis_space` outside its mapping view, so no digest changes. The readers stamp what they read, and the query facades verify the stamp rather than overwrite it. `resolve` and `resolve_rows` refuse a result stamped with another space or with none, and `Associations.concat` refuses records of different spaces. Also: `digests.Reference` serialises (`to_json`, `from_json`).
