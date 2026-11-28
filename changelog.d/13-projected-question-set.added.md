`ogquery.digests` poses a reference release's questions of another release and projects the answers onto the reference's index space (opengwas/ogquery#13, opengwasdb ADR 0109 §8b).

- `reference_of(q)` takes a release's question set and its `analysis_id` and ALID index maps as a `Reference`.
- `release_answers(path, reference=...)` and `release_digests(path, reference=...)` ask them by `analysis_id` and ALID only, never by rsID. A permitted rename then cannot change which rows are asked about, and the rsID misses are not asked.
- They drop every row the reference has no index for **before** any limit. `top_hits/limit` is posed without one and limited after the projection, in the canonical `(variant_index, analysis_index)` order, which is not the facade's analysis-major order.
- They refuse **every** Analysis of the reference the release does not hold, sampled or not (`refusals/unheld/<analysis_id>`), rather than answer it empty.
- A projected record is stamped with the reference's index spaces.

`projected_arrays(path, reference)` returns the decoded, projected answers themselves: the numbers of the splice's **answer proof**. It is not the exhaustive **cell proof**, which opengwasdb#521 owns. It also returns the projected variant and Analysis tables, column by column, with a variant's names (`rsid`, `source_alid`) apart under `names`.

`array_differences(a, b, eaf_tolerance=...)` compares them before any hashing:

- exactly, except `eaf`, which is compared within the tolerance. The tolerance must be finite and non-negative, or the comparison is refused;
- NaN at the same place, and the same infinity, are equal everywhere;
- Rho answers bitwise, so a signed zero differs from its opposite;
- a name difference is reported as `names/...` and fails no numeric comparison.

Without `reference`, nothing changes.
