# The LD read path's tests

The tests of `ogquery.ld`: `LDStore`, `matrix`, `region`, `proxies`, `clump`, `variants_in`,
`count_in` and `has`, their results and refusals, the store reader beneath them, and the
shared identity primitives they read by (`ogquery.alid`, `ogquery.identity`). They came from
opengwasdb's `tests/ld/` at
[`c8ebe4ca`](https://github.com/opengwas/opengwasdb/commit/c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4)
with the read path (ogquery#2, [ADR 0002](../../docs/adr/0002-the-ld-read-path-is-ogquery-s.md)).
The Genotype Store's read-side tests came from
[`964ac1b2`](https://github.com/opengwas/opengwasdb/commit/964ac1b28a3096553559d4a8565d877a884470e9)
with its reader (opengwas/ogquery#11, ADR 0002 amended): `test_genotype_answers.py`,
`test_genotype_queries.py` (and its subprocess, `genotype_threads.py`), `test_genotype_store.py`,
and the read parts of `test_genotype_refusals.py` and `test_reader_kind.py`. They carry the
`genotype` mark and need the `genotype` extra; on Windows, where pgenlib publishes no wheel, they
are skipped, naming the platform. `test_genotype_extra.py` needs no extra: it blocks the modules.

```bash
pixi run -e test test tests/ld                    # with both extras, as CI's matrix
pixi run -e test-nohttp test tests/ld -m "not http and not genotype"
pixi run -e test-nogenotype test tests/ld -m "not genotype"
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
`variants.tsv` and `panel.pvar` (which the loader, `testdata.py`, reads the rows from), and the
genotypes the Genotype Store pins (`panel.pgen`, `panel.psam`). The PC File, keep list and dosages
stay in opengwasdb, with the tests that read them.

One store is generated, not copied: `store-k0/`, the Test Store at k = 0, which the plink2
comparisons read. opengwasdb's tests write it in a scratch directory and commit none, so it was
written once by opengwasdb `c8ebe4ca`'s own `scripts/ld/make_test_store.py`; `SOURCE` records the
command, the commit and the inputs' blob ids, and pins its bytes as it pins the copies. It holds
an ALID companion derived on Linux, which the Windows run of the suite reads.

Two more are generated the same way (opengwas/ogquery#11). `genotype-store/` is the Test Panel's
Genotype Store, which no opengwasdb commit holds either: it was built once by opengwasdb
`964ac1b2`'s own `opengwasdb ld build --genotypes`, beside the `panel.pgen` and `panel.psam` it
pins. `genotype-answers/` is what `964ac1b2`'s `opengwasdb.ld.open` answered on it to the questions
`genotype_answers.py` asks, which `test_genotype_answers.py` asks `ogquery.ld.open` and compares.
`SOURCE` records both as it records `store-k0/`, and holds `genotype_answers.py` to the bytes that
asked, so changing a question means writing the answers again in `964ac1b2`'s environment. The
store pins its inputs' mtimes, which git does not keep: tests read a copy (`genotype_store`, in
`conftest.py`) whose inputs carry the recorded mtime, and never the files here.

## What stayed in opengwasdb

Every test of building, writing, deriving or validating a store, of the CLI, of building or
validating a Genotype Store, and of the Pan-UKB import and the benchmarks stays there, as does any
test that needs a store only build code can make. The read-side tests among them, so a reader of this suite knows
what it does not hold:

- **The ALID companion's lifetime** (`test_alid_mmap_lifetime.py`, `test_alid_mmap.py`): mappings
  released under concurrent lookups, and derivation's failure modes. A companion's reads are here
  (`test_variant_table.py`, `test_alid_companion.py`, on the k = 0 store's and a test-written one).
- **A lookup's memory over a wide index** (`test_alid_index_reads.py`, one test): it writes an
  index with the builder's `layout.put_numbers`.
- **The Partner Rows decoded whole** (`test_partner_rows.py`): it decodes frames with the build's
  `decode_frame`. The reader's own frame reads are `test_partner_row_reads.py`'s, here.
- **Stores at other scales** (`test_encoding.py`) and **Far Pairs a build derives**
  (`test_far_pairs.py`): each writes its store. The refusal of an encoding this reader does not
  implement is here, on an edited copy.
- **The encoding's build side** (`test_format.py`): quantising, the step, the far code floor and
  t_far, which only a build or validation computes; the Partner Floor (`test_encoding.py`).
- **Validation and the source scans** (`test_proxy_index_removed.py`): the leftover-index and
  0.1.0 reads are here; validation's view of them and the scans of opengwasdb's source are not.
- **A Genotype Store built or validated** (`test_genotype_validation.py`, `test_genotypes.py`,
  `genotype_codes.py`, the build parts of `test_genotype_refusals.py` and `test_reader_kind.py`):
  each builds a sidecar, compares with the build's PC Adjustment, or validates. Its reads are here,
  on the committed Genotype Store.
- **The build's variant order** (`test_variant_order.py`, `test_sorted_spelling_store.py`), the
  **sorted identity a build forms** (`contract/test_identity.py`,
  `test_sorted_spelling_conformance.py`) and the **build's geometry** (`test_geometry.py`'s
  `chromosome_ranges`, `spans_at`, `near_pair_of`).
