- **`ogquery.cpra` holds a variant's VCF normal form, CPRA formation's one owner
  (opengwas/ogquery#14, opengwasdb ADR 0108 §7).** `trim_alleles`, `normalise_indel`, the
  `ReferenceSequence` protocol and their errors (`NotAnIndel`, `ReferenceAlleleMismatch`,
  `ContigNotHeld`, `PositionOutsideContig`, `UnreadableReferenceBase`) moved unchanged from
  opengwasdb `0ba413e7`'s `opengwasdb.variants.left_align`, with its tests, so opengwasdb can
  re-export them.
