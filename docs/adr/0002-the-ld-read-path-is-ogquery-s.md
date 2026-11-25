# The LD read path is ogquery's

## Status

Accepted with [ogquery#2](https://github.com/opengwas/ogquery/issues/2), part of epic
[opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450). The owner ruled
the choices the move needed (layout, HTTP, Genotype Stores, the shared primitives, golden data,
the public reads, history) on the survey of opengwasdb's LD read path; this records them and
what the move found. Switching `opengwasdb.ld` to read through ogquery is a later opengwasdb
change, after this is released.

## Context

opengwasdb's `opengwasdb.ld` (formerly ogld) builds, validates and reads LD Stores: per-ancestry
LD for clumping, proxies and LD matrices. Its readers -- opengwasdb's own Top-Hit Query
(opengwasdb#419) and ogdb-api, which will serve LD (opengwas/ogld#79) -- need only the read side,
and ogdb-api's LD workers should not import opengwasdb at all. But importing `opengwasdb.ld`
imported its builder (`ld/build`), its validator, its writer, and the Genotype Store reader,
whose module imports threadpoolctl and whose reads need pgenlib, a compiled package. ogquery
already holds the GWAS read path on numpy, zarr and numcodecs alone (ADR 0001), so the LD read
path belongs beside it.

## Decision

**The read side of `opengwasdb.ld` moves into `ogquery/ld/`**, from opengwasdb
[`c8ebe4ca`](https://github.com/opengwas/opengwasdb/commit/c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4),
by its module names: `opengwasdb.ld.format.*` is `ogquery.ld.format.*`, `opengwasdb.ld.store.reader`
is `ogquery.ld.store.reader`, `opengwasdb.ld.query.ldstore` is `ogquery.ld.query.ldstore`, and so
on. `ogquery.ld.open` opens an LD Store; `LDStore.matrix`, `region`, `proxies` and `clump` answer
as before, with their results (`LDMatrix`, `Proxies`, `Clumps`, `Variants`) and refusals
(`Refusal`, `RefusalReason`, `InputError`). Beneath them come the store reader, the chunk cache
and location handling, decoding, the Partner Rows' frame reads, the variant table and the read
side of the LD format.

**Modules that mixed reads and builds were split by name.** A name a read reaches moved; a name
only a build, a write or validation calls stayed in opengwasdb, which will import the read side
from here. So `frames.py` brought `FramePayload`, `FrameCache` and `PartnerFrames` and left
`encode_frame`, `decode_frame` and `FramesGroup`; `layout.py` brought the fill-value rules, the
chromosome record's reader and the ALID companion's reader and left every array creator;
`geometry.py` left `chromosome_ranges`, `spans_at` and `near_pair_of`, which only a build or
validation computes; `provenance.py` left `Sourced`, `REQUIRED_KEYS` and `VALIDATION_RULES`;
`tables.py` left `Partners`; `store/reader.py` left `open_variant_table`; `encoding.py` left the
build's own distances, `NEAR_BP` and `FAR_BP`, which a reader never uses. Three names with no
caller in either side came with the format, because they state it and have no build code in them:
`answer_kind`, `judge_request` and `judge_window`. The package initialisers import the read side
eagerly and are small: nothing build-side exists here to load.

**What stays in opengwasdb:** building and writing a store, deriving one (Partner Rows, Far Pairs,
the ALID index and companion), validation and its rules, the CLI, the Pan-UKB import, and the
**Genotype Store** reader. That reader needs pgenlib and threadpoolctl, which break ogquery's
dependency rule, so `ogquery.ld.open` refuses a Genotype Store with an `InputError` naming
`opengwasdb.ld.open`. The seam stays open for it: `ogquery.ld.Reader` is the public protocol every
query reads through, and `ogquery.ld.query.kind.open_kind` still names both kinds, so opengwasdb's
`open` dispatches a Genotype Store root to its own reader and plugs it into the same `LDStore`.
threadpoolctl therefore leaves the read path.

**HTTP is an extra, and one open owns it.** An LD Store is read from a local path or an http(s)
URL. zarr opens a URL through fsspec's HTTP filesystem, which runs on aiohttp; neither is a core
dependency. `ogquery.arrays.open_group` -- already the one place a reader calls
`zarr.open_group` -- is the LD open too, and gains the http(s) branch: it checks that fsspec and
aiohttp are installed before zarr is asked, and without them raises
`ogquery.arrays.HttpExtraMissing` (an `ImportError`) naming `pip install 'ogquery[http]'`. Any
other URL scheme is refused rather than read as a local path. The extra is
`http = ["fsspec>=2024.12", "aiohttp>=3.10"]`, opengwasdb's floors. Local LD Stores keep their
`os.pread` frame reads (`PreadSource`).

**A local frame read works on Windows.** opengwasdb ran on Linux, and Windows has no `os.pread`
and opens a descriptor in text mode. `PreadSource` now opens its descriptor with `O_BINARY`
where there is one and reads through `location._read_at`: `os.pread` where it exists, and
otherwise a seek and a read under one process-wide lock, which gives the same bytes and
serialises only local frame reads. No format change was needed.

**The shared primitives exist once, here.** The LD read path spells and orders variants by
helpers opengwasdb held beside its builders; their bodies moved unchanged and are the owners
opengwasdb will re-export: `chromosome_rank` and `chromosome_sort_key` in `ogquery.identity`
(with `normalise_chromosome` and the autosomes, already there), and the sorted-spelling read
helpers -- `sorted_alid_locus`, `spells_indel_reference_first`, `ReferenceFirstIndel`,
`refuse_reference_first_indel` -- in `ogquery.alid` (with `is_indexable_alid`, already there). A
record's sorted identity as a build forms it (`sorted_identity`) stays in opengwasdb. The LD
Store's own `VariantTable` stays a separate ALID lookup from the GWAS sidecars': it reads the LD
Store's format (opengwasdb ADR 0096), which this move does not change. Run-length helpers whose
semantics differ (frames, labels, slices) stay separate.

**The reads ogdb-api needs are public** (opengwas/ogld#79): `LDStore.variants_in(chrom, start,
stop)`, the Variant Set's variants in a range, read from the variant table before any r;
`LDStore.count_in`, its length, from positions alone; and `LDStore.has(alids)`, one boolean per
ALID, true exactly where the queries find it present. Each runs inside the reader's `request()`
as the queries do and normalises the chromosome as `region` does; `variants_in` judges no radius,
since it reads no pair. A range whose start exceeds its stop is an `InputError`, never an empty
answer. The component pool's size (`_query_workers`) stays private.

**The LD Store format is unchanged.** `software.ogld` and the LD format version stay as they are.

**The golden data is opengwasdb's, read only here, in ADR 0001's order.** The LD Test Panel's
answers, its Test Store and its loader came to `tests/ld/` as copies of `c8ebe4ca`'s files.
`tests/ld/data/SOURCE` names that commit and each file's git blob id, and a test holds every
file to it byte for byte, offline. opengwasdb's builders regenerate them, so a change to what the
LD reader answers moves as a Store Reader change does: **the reader changes first; opengwasdb's
generator then runs that reader to regenerate the files; ogquery takes that opengwasdb commit's
files with `SOURCE` naming it.** Until opengwasdb reads LD through ogquery, its own reader is
the one that answers, and ogquery makes the same change after. Tests that need a store only
build code can derive stay in opengwasdb (`tests/ld/README.md` lists the read-side ones).

**History is a plain copy.** The modules were split between read and build, so a path's history
would describe code that is not here; the first commit names opengwasdb `c8ebe4ca` as the source.

## Consequences

- Until opengwasdb reads LD through ogquery, the LD read path exists twice, as the Store Reader
  did before opengwasdb#475. That change makes `opengwasdb.ld`'s read modules re-export these and
  deletes its copies.
- `import ogquery.ld` imports `ogquery.arrays`, whose import sets zarr's process-wide
  configuration as every Store Reader process has it (opengwasdb ADR 0056): one-worker
  `FusedCodecPipeline` among it. ogld's readers ran under zarr's default pipeline, and opengwasdb
  ADR 0094's amendment (opengwasdb#460) measured `proxies` 17-29 % slower cold under the fused
  one, with clumps, regions and warm reads not slower. An LD-only process now reads under it too.
- `import zarr` imports fsspec whenever fsspec is installed, whatever imports zarr; ogquery's LD
  path adds nothing to it, and without the `http` extra loads neither.
- The plink2 comparisons at k = 0 need a k = 0 store, which only opengwasdb's build writes, so
  they are tested there until opengwasdb commits one for ogquery to copy.
