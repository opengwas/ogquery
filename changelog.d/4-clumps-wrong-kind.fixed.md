- **`TopHitTier.clumps` refuses a `clumps` array, a `lead` group and a scalar `lead` as
  `MalformedClumps`, by name, where it raised `AttributeError` or `IndexError`
  (opengwas/ogquery#4).** Format spec §18.1 says a `clumps/` group not laid out as it
  describes is refused as `MalformedClumps`. The fix is ogquery's only: opengwasdb's in-tree
  copy is deleted by opengwasdb#475, which re-exports this module.
