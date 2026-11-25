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
by its module names: `opengwasdb.ld.format.*` is `ogquery.ld.format.*`,
`opengwasdb.ld.store.reader` is `ogquery.ld.store.reader`, `opengwasdb.ld.query.ldstore` is
`ogquery.ld.query.ldstore`, and so on. `ogquery.ld.open` opens an LD Store; `LDStore.matrix`,
`region`, `proxies` and `clump` answer as before, with their results (`LDMatrix`, `Proxies`,
`Clumps`, `Variants`) and refusals (`Refusal`, `RefusalReason`, `InputError`). Beneath them come the
store reader, the chunk cache and location handling, decoding, the Partner Rows' frame reads, the
variant table and the read side of the LD format.

**Modules that mixed reads and builds were split by name.** A name a read reaches moved; a name
only a build, a write or validation calls stayed in opengwasdb, which will import the read side
from here. So `frames.py` brought `FramePayload`, `FrameCache` and `PartnerFrames` and left
`encode_frame`, `decode_frame` and `FramesGroup`; `layout.py` brought the fill-value rules, the
chromosome record's reader and the ALID companion's reader and left every array creator;
`geometry.py` left `chromosome_ranges`, `spans_at` and `near_pair_of`, which only a build or
validation computes; `provenance.py` left `Sourced`, `REQUIRED_KEYS` and `VALIDATION_RULES`;
`tables.py` left `Partners`; `store/reader.py` left `open_variant_table`. `encoding.py` keeps
what decoding and the refusals need -- `REncoding`'s container, `of` (which encodings this reader
implements), `holds`, `require_codes` and `dequantise` -- and left quantising and the step
(`record`, `quantise`, `step`, `max_error`), the floors (`far_code_floor`, `partner_code_floor`,
`least_r`), the masks a build and a Genotype Store keep pairs by (`stored_mask`, `partner_mask`),
t_far itself (`t_far`, `T_FAR_Q`, `T_FAR_MIN`) and the build's distances (`NEAR_BP`, `FAR_BP`):
a reader uses the t_far, floors and distances a store records. opengwasdb's `REncoding` extends
this one with them. Scanning the other split modules member by member left nothing else that
only a build or validation calls.

**An accepted deviation: three names state the format and came with it** (owner's ruling, fix
round 1). `answer_kind`, `judge_request` and `judge_window` have no caller in either repository's
production code, so the survey put them with the build. They contain no build code and state the
format's semantics, and without them 18 of its tests of the refusal table and the answer kinds
would have stayed behind; so they are here. The package initialisers import the read side eagerly
and are small: nothing build-side exists here to load.

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
dependency. The one place a reader calls `zarr.open_group` is `ogquery.opening.open_group`, which
the GWAS read path imports as `ogquery.arrays.open_group` (re-exported, by the name it always
had) and the LD read path from `ogquery.opening`. It gains the http(s) branch: it checks that
fsspec and aiohttp are installed before zarr is asked, and without them raises
`ogquery.opening.HttpExtraMissing` (an `ImportError`) naming `pip install 'ogquery[http]'`. The
scheme is case-insensitive and reaches zarr lower-cased (`HTTP://` opens; fsspec knows only
`http`). Any other URL scheme is refused rather than read as a local path. The extra is
`http = ["fsspec>=2024.12", "aiohttp>=3.10"]`, opengwasdb's floors. Local LD Stores keep their
`os.pread` frame reads (`PreadSource`).

**The LD read path leaves zarr's configuration alone.** The GWAS read path's process-wide zarr
configuration (opengwasdb ADR 0056: the one-worker `FusedCodecPipeline`, Blosc's threads, empty
chunks written) is set by importing `ogquery.arrays`, and stays there: every GWAS read imports
that module, as before. The opener moved out of it into `ogquery.opening`, which sets nothing,
and `ogquery.ld` imports only that. So `import ogquery.ld` and an LD query leave zarr's
configuration at its defaults, as ogld's readers had it (`tests/ld/test_zarr_config.py`). Under
the fused pipeline opengwasdb#460 measured `proxies` 17-29 % slower cold on real LD Stores (its
ADR 0094 amendment); this ticket's review measured the Test Store's `proxies` (294 ALIDs, 50
fresh readers a pipeline, alternated) at a median of 40.7 ms under zarr's default pipeline and
43.2 ms under the fused one. A process that imports both read paths reads LD under the GWAS
configuration, as an opengwasdb process does today.

**An accepted exception to the import boundary: zarr loads fsspec.** With the `http` extra
installed, `import zarr` itself imports fsspec (zarr 3.4's storage module), whatever imports
zarr; so `import ogquery.ld` loads fsspec there even for a local query. ogquery cannot prevent it
and adds nothing to it: `tests/ld/test_ld_imports.py` measures the LD path against what numpy,
zarr and numcodecs load alone, allows fsspec in that baseline only, and says why. Without the
extra nothing loads it, and aiohttp is never loaded by an import.

**A local frame read works on Windows.** opengwasdb ran on Linux, and Windows has no `os.pread`
and opens a descriptor in text mode. `PreadSource` now opens its descriptor with `O_BINARY`
where there is one and reads through `location._read_at`: `os.pread` where it exists, and
otherwise a seek and a read under one process-wide lock, which gives the same bytes and
serialises only local frame reads. No format change was needed. An ALID companion (ADR 0098)
pins the canonical files it was derived from by their relative paths, which a Linux derivation
spells with `/`; the reader now forms its own as POSIX paths too, so a companion derived on
Linux is read on Windows (`tests/ld/test_alid_companion.py`).

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
ALID, true exactly where the queries find it present. None of them opens a key of the Near Band,
the Far Pairs, the Partner Rows or a Proxy Index, so a store whose LD components are damaged
still answers them: they read `Reader.rows_in`, the variant table's. `region` reads
`Reader.region_rows` instead, which opens the chromosome's Near Band first, so a broken one
refuses even a region with no variant in it, as opengwasdb `c8ebe4ca` refused it. Each runs
inside the reader's `request()` as the queries do and normalises the chromosome as `region`
does; `variants_in` judges no radius, since it reads no pair. A range whose start exceeds its
stop is an `InputError`, never an empty answer. The component pool's size (`_query_workers`)
stays private. `region_rows` is a member of the `Reader` protocol, so opengwasdb's Genotype
Store reader implements it too, as its `rows_in`.

**The LD Store format is unchanged.** `software.ogld` and the LD format version stay as they are.

**The golden data is opengwasdb's, read only here, in ADR 0001's order.** The LD Test Panel's
answers, its Test Store and its loader came to `tests/ld/` as copies of `c8ebe4ca`'s files.
`tests/ld/data/SOURCE` names that commit and each file's git blob id, and a test holds every
file to it byte for byte, offline. opengwasdb's builders regenerate these copies, so a change to
what the LD reader answers moves as a Store Reader change does: **the reader changes first;
opengwasdb's generator then runs that reader to regenerate the files; ogquery takes that
opengwasdb commit's files with `SOURCE` naming it.** Until opengwasdb reads LD through ogquery,
its own reader is the one that answers, and ogquery makes the same change after. Tests that need
a store only build code can derive stay in opengwasdb (`tests/ld/README.md` lists the read-side
ones).

**One store is generated, and no opengwasdb commit holds it yet.** The k = 0 Test Store
(`tests/ld/data/test-panel/store-k0/`), which the plink2 comparisons read, is one opengwasdb's
tests write in a scratch directory and do not commit. It was written once by `c8ebe4ca`'s own
`scripts/ld/make_test_store.py`, from a `git archive` of that commit in its own environment;
`SOURCE` records it apart from the copies -- the command, the commit, the blob ids of the files
it read there -- and pins what it wrote. The opengwasdb switch lane commits the same store there;
a refresh then copies it like the rest.

**History is a plain copy.** The modules were split between read and build, so a path's history
would describe code that is not here; the first commit names opengwasdb `c8ebe4ca` as the source.

## Consequences

- Until opengwasdb reads LD through ogquery, the LD read path exists twice, as the Store Reader
  did before opengwasdb#475. That change makes `opengwasdb.ld`'s read modules re-export these and
  deletes its copies.
- opengwasdb's `REncoding` becomes a subclass of this one carrying the build-side members, and
  its Genotype Store reader keeps the masks and floors it calls.
- An LD-only process, such as ogdb-api's LD workers, reads under zarr's defaults; a process that
  also reads GWAS Stores reads LD under their configuration.
- The k = 0 Test Store is the one test fixture here no opengwasdb commit holds, until the switch
  lane commits it there.
