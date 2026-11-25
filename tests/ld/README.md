# The LD read path's tests

The tests of `ogquery.ld`: `LDStore`, `matrix`, `region`, `proxies`, `clump`, `variants_in`,
`count_in` and `has`, their results and refusals, the store reader beneath them, and the
shared identity primitives they read by (`ogquery.alid`, `ogquery.identity`). They came from
opengwasdb's `tests/ld/` at
[`c8ebe4ca`](https://github.com/opengwas/opengwasdb/commit/c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4)
with the read path (ogquery#2, [ADR 0002](../../docs/adr/0002-the-ld-read-path-is-ogquery-s.md)).

```bash
pixi run -e test test tests/ld                    # with the `http` extra, as CI's matrix
pixi run -e test-nohttp test tests/ld -m "not http"
```

## The data is opengwasdb's, read only

`data/test-panel/` is the Test Panel's answers and its Test Store, and `data/identity/` the
sorted spelling's committed answers. **opengwasdb regenerates them** (its
`scripts/ld/make_test_panel.py` and `make_test_store.py`, which need its build code), and
ogquery holds a copy, read only, as it holds `tests/contract/`. `data/SOURCE` names the commit
and each file's git blob id there; `test_source.py` holds every file to it byte for byte, and
`data/.gitattributes` keeps every checkout's bytes. A test that fails against this data is a
wrong answer: never edit a file here to make one pass. A change to what the LD reader answers
moves in [ADR 0001](../../docs/adr/0001-ogquery-is-the-store-reader-extracted-from-opengwasdb.md)'s
order: the reader changes first, opengwasdb regenerates, and ogquery takes the files of that
opengwasdb commit and rewrites `SOURCE` from its `git ls-tree`.

ogquery holds what these tests read: the Test Store, the reference r (`reference/r.npz`), plink2's
r and clump leads (`plink2/`), the simulated GWAS (`gwas/z.tsv`), `manifest.json`, `README.md`,
`variants.tsv` and `panel.pvar` (which the loader, `testdata.py`, reads the rows from). The
genotypes, PC File and dosages stay in opengwasdb, with the tests that read them.

## What stayed in opengwasdb

Every test of building, writing, deriving or validating a store, of the CLI, of the Genotype
Store, and of the Pan-UKB import and the benchmarks stays there, as does any test that needs a
store only build code can make. The read-side tests among them, so a reader of this suite knows
what it does not hold:

- **plink2 at k = 0** (`contract/test_query.py`): the clump leads, and the proxies outside the
  step's band. Both read a k = 0 store that `make_test_store.py` writes; the Test Store is k = 10.
  The plink2 r itself is held to the reference r here (`test_testdata.py`).
- **The local ALID companion** (`test_alid_mmap_lifetime.py`, `test_variant_table.py`): a
  companion is derived by `derive_alid_mmap`, and the Test Store holds none.
- **A lookup's memory over a wide index** (`test_alid_index_reads.py`, one test): it writes an
  index with the builder's `layout.put_numbers`.
- **The Partner Rows decoded whole** (`test_partner_rows.py`): it decodes frames with the build's
  `decode_frame`. The reader's own frame reads are `test_partner_row_reads.py`'s, here.
- **Stores at other scales** (`test_encoding.py`), **a leftover Proxy Index** and **Far Pairs a
  build derives** (`test_proxy_index_removed.py`, `test_far_pairs.py`): each writes its store.
- **The build's variant order** (`test_variant_order.py`, `test_sorted_spelling_store.py`), the
  **sorted identity a build forms** (`contract/test_identity.py`,
  `test_sorted_spelling_conformance.py`) and the **build's geometry** (`test_geometry.py`'s
  `chromosome_ranges`, `spans_at`, `near_pair_of`).
