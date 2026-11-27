# The LD read path is ogquery's

## Status

Accepted with [ogquery#2](https://github.com/opengwas/ogquery/issues/2), part of epic
[opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450). The owner ruled
the choices the move needed (layout, HTTP, Genotype Stores, the shared primitives, golden data,
the public reads, history) on the survey of opengwasdb's LD read path; this records them and
what the move found. Switching `opengwasdb.ld` to read through ogquery is a later opengwasdb
change, after this is released.

**Amended 26 November 2026** ([opengwas/ogquery#11](https://github.com/opengwas/ogquery/issues/11)):
the owner ruled that Genotype Stores are read by ogquery, behind a `genotype` extra. That reverses
what this record said of the Genotype Store reader and of the encoding's masks and quantising;
"Amendment: Genotype Stores are ogquery's", below, records it, and the paragraphs it reverses say
so where they stand.

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
this one with them. (Amended: `quantise`, `stored_mask` and `partner_mask` came with the Genotype
Store's reader, below.) Scanning the other split modules member by member left nothing else that
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
threadpoolctl therefore leaves the read path. (Reversed by the amendment below: the reader is
ogquery's, behind the `genotype` extra.)

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
  its Genotype Store reader keeps the masks and floors it calls. *(Superseded by the amendment
  below: the reader and the masks are here; opengwasdb keeps the floors.)*
- An LD-only process, such as ogdb-api's LD workers, reads under zarr's defaults; a process that
  also reads GWAS Stores reads LD under their configuration.
- The k = 0 Test Store is the one test fixture here no opengwasdb commit holds, until the switch
  lane commits it there. *(Superseded by the amendment below: the Test Panel's Genotype Store and
  opengwasdb's answers on it are generated here too.)*

## Amendment: Genotype Stores are ogquery's (26 November 2026)

**The ruling.** The owner ruled on 26 November 2026 that Genotype Stores are read by ogquery
([opengwas/ogquery#11](https://github.com/opengwas/ogquery/issues/11), epic
[opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450)): the Data API serves
the HGDP+1kGP panels from Genotype Stores and reads every Store through ogquery. opengwasdb's
half, reading through ogquery and dropping its copy, is
[opengwas/opengwasdb#512](https://github.com/opengwas/opengwasdb/issues/512).

**The reader moves, in this record's order.** opengwasdb's Genotype Store read side moved from
opengwasdb [`964ac1b2`](https://github.com/opengwas/opengwasdb/commit/964ac1b28a3096553559d4a8565d877a884470e9)
by its module names, split by name as above: `opengwasdb.ld.genotypes.{reader,pgen,adjust,layout}`
is `ogquery.ld.genotypes.{reader,pgen,adjust,layout}`. There is no sidecar read module:
`sidecar.py` writes and `validation.py` validates, and both stay. A name a read reaches moved:
`GenotypeReader` and its handle pool and BLAS guard; `layout`'s `KEY`, `VERSION`, `ARRAYS`,
`check_input`, `Fileset`, `open_filesets` and `mapped_arrays`; `adjust`'s `Block` and `r`, which
opengwasdb's tiles import and will re-export. **`Genotypes` moved whole** -- `range`,
`variant_count` and the int8 reads, which only a build calls, too -- because it is the one owner
of reading a pgen, which a build and a read both do. What stays: `covariates`, `residuals`,
`prepare`, `Covariates` and `FLAT` (the PC Adjustment), `input_pin` and `fileset_record` (what a
writer records), `SidecarWriter` and validation. `layout.KEY` is `ogquery.ld.query.kind.KEY`,
not a second spelling. `mapped_arrays` (which opengwasdb's validation calls),
`GenotypeReader.open` (a second root read, which both repositories' tests call; validation
dispatches to `validate_sidecar` instead) and the reader's envelope checks are as copied: each
has a caller, and none was removed. Behaviour is identical -- the same answers, refusals and
text -- but for two deliberate changes: the missing extra, and an http(s) URL whose root is a
Genotype Store (both below).

**The encoding brings what the reader calls.** `quantise`, `stored_mask` and `partner_mask` moved
into this package's `REncoding`, with `_R_SLACK`, the float slack `quantise` refuses beyond;
`quantise` needs neither `step` nor `record`, which stay. This reverses that part of the
`encoding.py` paragraph above. It adds members and changes none, so no LD Store read changes. The
floors (`far_code_floor`, `partner_code_floor`, `least_r`), t_far and the build's distances stay
in opengwasdb: the reader uses the values a store records.

**Opening one.** `ogquery.ld.open` reads the root once (`open_kind`) and hands a Genotype Store
root to `GenotypeReader.from_opened`, as opengwasdb's `open` did. Before it imports
`ogquery.ld.genotypes.reader` it calls `ogquery.opening.require_genotype`, modelled on
`require_http`: it finds pgenlib and threadpoolctl (`find_spec`, no import), and without either
raises `GenotypeExtraMissing`, an `ImportError` naming `pip install 'ogquery[genotype]'` and the
modules it lacks. That replaces `import_pgenlib`'s `InputError` naming `opengwasdb[ld-build]`,
the first of the two deliberate changes of text; `import_pgenlib` now calls the same check. The
second is the local-only refusal of a URL (below).

**The genotype extra, and the import boundary.** `genotype = ["pgenlib>=0.94",
"threadpoolctl>=3.5"]`, with no platform marker: pip decides. pgenlib 0.94 is the first with
Python 3.14 wheels. threadpoolctl's floor is 3.5, not opengwasdb's 3.0: measured beside numpy
2.0.0's wheel, threadpoolctl 3.0 to 3.4 find no BLAS (`threadpool_info()` is empty), so the
one-thread limit holds nothing, and 3.5 is the first that finds numpy 2's `scipy_openblas`.
`dev` carries both on every platform but Windows. A Clean Install is unchanged: numpy, zarr and
numcodecs. pgenlib and threadpoolctl are imported only under `ogquery/ld/genotypes/`, and only
`ogquery.ld.genotypes.reader` imports threadpoolctl at import, so `import ogquery`,
`import ogquery.ld`, and opening or querying an LD Store or a GWAS Store load neither
(`tests/unit/test_store_reader_imports.py`, `tests/ld/test_ld_imports.py`).

**Platforms.** pgenlib publishes wheels for Linux (x86_64 and aarch64) and macOS (universal2),
and none for Windows, in any version. CI installs the extra and runs the Genotype Store tests on
Linux and macOS, on Python 3.12 and 3.14, and at the floors: on the operating systems of CI's
matrix, on x86_64 for Linux. Linux aarch64's wheels are not run, as CI runs no aarch64 leg for
anything; on Windows the tests marked
`genotype` are skipped with a reason naming the platform, never silently. The one-thread limit
holds only a BLAS threadpoolctl controls (OpenBLAS, MKL): NumPy's macOS 14+ arm64 wheels link
Accelerate, which it cannot control, and there the limit does nothing and the test of it is
skipped, naming that; on Linux a BLAS it cannot see fails that test. A `no-genotype` job runs
the rest of the suite without the extra. pixi's `genotype-wheels` feature adds the extra on
linux-64, osx-64 and osx-arm64 only (pixi makes a feature named `genotype` from the extra itself,
on every platform), so every environment still solves on win-64.

**Local only.** A Genotype Store is opened from a local path: its reader finds the pgens it pins
by path, relative to its root, and memory-maps its arrays, as opengwasdb's
`docs/spec/genotype-store.md` specifies. An LD Store still opens over HTTP(S). An http(s) URL
whose root is a Genotype Store is a new refusal, an `InputError` saying a Genotype Store is read
from a local path only, raised before the extra is checked (installing it would not make a URL
readable); opengwasdb `964ac1b2`'s reader raised a missing-input error at the first pinned pgen
there, naming a path `Path` had mangled from the URL.

**One more store is generated, as the k = 0 Test Store was.** No opengwasdb commit holds a
Genotype Store: its tests build one per session. The Test Panel's was built once by `964ac1b2`'s
own `opengwasdb ld build --genotypes`, from a `git archive` of that commit in its own environment,
beside the `panel.pgen` and `panel.psam` it pins, which came with it; and `964ac1b2`'s
`opengwasdb.ld.open` answered the ported tests' questions on it (`tests/ld/genotype_answers.py`).
`tests/ld/data/SOURCE` records both apart from the copies. The store pins its inputs' mtimes, which
git does not keep: a test reads a copy whose inputs it sets to the recorded mtime.

**Consequences.**
- opengwasdb's half re-exports `ogquery.ld.genotypes` and drops its copy. Until then the reader
  exists twice, as the LD read path did.
- opengwasdb's `REncoding` overrides three members this one now has, with the same bodies; its
  claim that ogquery's encoding raises `AttributeError` for them no longer holds.
- A build that opens a pgen through `Genotypes` without pgenlib is now told to install
  `ogquery[genotype]`, not `opengwasdb[ld-build]`; opengwasdb's half decides what its build says.
