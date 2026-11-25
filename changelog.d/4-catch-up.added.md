- **The Store Reader catches up with opengwasdb `c8ebe4ca`: it reads the Clumps and
  `narrowed_range_question` bounds the conversion proof's range question
  (opengwas/ogquery#4).** `ogquery.top_hits` gains the Clumps names (`CLUMPS_*`,
  `TierClumps`, `MalformedClumps`, `declares_clumps_of`, `TopHitTier.clumps`,
  `TopHitTiers.by_key`/`keys`), and `StoreManifest.clumps` and `StoreRelease.top_hit_tiers`
  carry the manifest's declaration to them; a `clumps/` group the manifest does not declare,
  a declaration with no group, or a group not laid out as spec §18.1 says is refused as
  `MalformedClumps`, never read. `ogquery.digests` gains `RANGE_ROWS_ASKED`,
  `RANGE_CELLS_ASKED`, `narrowed_range_question` and `adapter_name`. Both come from
  opengwasdb#418 (Clumps) and opengwasdb#492 (`rewrite-release --convert-legacy`); the
  contract bundle's bytes are unchanged and `SOURCE` now names `c8ebe4ca`, as ADR 0001's
  ordering rule for changes made before opengwasdb#475 says.
