# CPRA formation is ogquery's, against a memory-mapped byte genome

## Status

Accepted with [opengwas/ogquery#14](https://github.com/opengwas/ogquery/issues/14), the first
slice of [opengwas/opengwasdb#517](https://github.com/opengwas/opengwasdb/issues/517) under
epic [opengwas/opengwasdb#525](https://github.com/opengwas/opengwasdb/issues/525). The rule
is opengwasdb's ADR 0108 (`docs/adr/0108-indel-alids-are-normalised-against-the-release-assembly.md`,
as revised 28 Nov 2026: every variant keyed by its CPRA, `chrom:pos:REF:ALT` on GRCh38, signed
for ALT), and its §7
records the owner's ruling of 28 Nov 2026 that CPRA formation and its reference live here:
ogdb-api uses only ogquery. This record states what ogquery built for it: the byte genome's
format, and the choices `to_cpra` made where ADR 0108 left one open.

## Context

A CPRA's REF is GRCh38's, so every caller that forms one -- the engine's builders,
completion's panel keys, the LD import, and #518's alignment tool behind the Data API --
needs the sequence, and needs the same sequence, or two of them spell one variant two ways.
opengwasdb reads GRCh38 from an indexed FASTA through pysam (`FastaSequence`), which ogquery
may not import (ADR 0001). ADR 0108 measured the alternative, the 25 primary contigs as one
memory-mapped byte file: REF for 36 M sites in 0.5 s warm, 13-19 ns a site, with workers
sharing one page-cache copy. That file is the byte genome, and ogquery owns its format.

## Decision

### The byte genome's format (`ogquery.genome`, format version 1.0)

**The data file** is raw uint8: each contig's bases, upper-cased, as their ASCII bytes,
concatenated with no separator, in the variant order (1-22, X, Y, MT;
`identity.chromosome_sort_key`). A base is stored as the source has it, upper-cased: A, C, G,
T, N, or any other IUPAC nucleotide code. None is ever rewritten: an `N` stays an `N`, and
what to do with one is the reader's decision (`to_cpra` calls it `off_sequence`). A byte
that is not an IUPAC code (`-`, a newline, a digit) is refused by the writer. GRCh38's is
3,088,286,401 bytes.

**The sidecar** is `<data file>.json` beside it:

```json
{
  "format": "ogquery-byte-genome",
  "format_version": "1.0",
  "assembly": "GRCh38",
  "sequence_digest": "263979d2475a76b5396c58dbeee0138d3548cf5011f53f2189662466c789e258",
  "data_sha256": "<sha256 of the data file>",
  "contigs": [
    {"label": "1", "offset": 0, "length": 248956422, "sha256": "<sha256 of its bases>"},
    "...",
    {"label": "MT", "offset": 3088269832, "length": 16569, "sha256": "..."}
  ],
  "provenance": {"source": "...", "source_sha256": "...", "derived_by": "...", "derived_utc": "..."}
}
```

- Labels are the canonical ones (`normalise_chromosome`: `1`, not `chr1`; `MT`, not `chrM`),
  primary contigs only (`is_primary_contig`).
- `provenance` is a free object the writer's caller supplies: the source FASTA and its
  checksum, who derived the file and when. ogquery records it and never interprets it.
- **The sequence digest** is exactly ADR 0108 §9's, as `FastaSequence.digest` computes it: the
  SHA-256, in lower-case hex, of one ASCII line per contig, in the variant order -- its label,
  a tab, its length in bases, a tab, the lower-case hex SHA-256 of its upper-cased sequence,
  and a newline. It covers the sequence alone: line wrapping, compression, case, contig
  naming and contigs off the primary assembly do not enter it, so a byte genome written from
  a FASTA has that FASTA's digest. `genome.sequence_digest` computes it from the per-contig
  lines. The data file's SHA-256 is a different thing, the file's identity, which a release
  records as `provenance.reference_sequence` (ADR 0108 §8).

**The writer**, `write_byte_genome(path, contigs, *, assembly, provenance)`, takes
`(label, chunks)` pairs, so a 3 GB genome is streamed, never held. It refuses
(`RefusedContig`) a non-primary or non-canonical label, a contig given twice, contigs out of
the variant order, an empty contig, a byte that is not an IUPAC code, and no contig at all.
It hashes as it writes, writes both files under temporary names beside their targets, and
publishes the data file under its name and then the sidecar, so a sidecar exists only beside a
complete data file; a refusal leaves nothing behind. It never overwrites an existing data
file or sidecar, a dangling symlink included (`FileExistsError`): a reference in use is
replaced by writing a new one. Each name is published by a hard link (`os.link`), which
refuses a name already taken in the same step that takes it, so a second writer that appears
after the first check cannot be overwritten either, as a check followed by `os.replace`
would allow; if the sidecar's name is taken, the data file just published is withdrawn.

**The reader**, `ByteGenome.open(path)`, reads the sidecar and maps the data with
`numpy.memmap` in mode `r`. It refuses (`UnreadableByteGenome`), loudly, before mapping:

- a missing sidecar, or one that is not JSON, not this format, or missing a field, or
  whose checksums (`sequence_digest`, `data_sha256`, each contig's `sha256`) are not 64
  lower-case hex characters;
- **an unknown format version**: anything but `1.0`, since a reader cannot know what a
  later version changed;
- **offsets that do not tile the file**: the first contig at 0, each next one where the one
  before ends, the labels canonical, primary, unique and in the variant order;
- **a data file whose size is not the contigs' total**: a truncated copy, or a data file
  that is not this sidecar's. The size is taken (`os.fstat`) from the one open file that is
  then mapped, so a file replaced between the two is never mapped as a prefix.

It implements the `ReferenceSequence` protocol: `fetch(chrom, start, end) -> str` (0-based,
half-open, upper-cased; `ContigNotHeld` and `PositionOutsideContig`, as `FastaSequence`
raises them; a label is read as an identity reads it, so `chr1` is contig `1`) and
`digest()`. `bases(chrom, positions)` gathers the base at each 1-based position of a numpy
array as uint8, in one vectorised index, for a builder looking up REF at millions of SNVs.

**`digest()` is computed, never recorded.** It hashes the mapped bytes once per handle and
caches the answer, as `FastaSequence.digest` does: 4.5 s for GRCh38 warm. The sidecar's
recorded digest is never returned as if it were computed. `verify()` compares what the bytes
give -- each contig's SHA-256, the sequence digest, the data file's SHA-256, all from one
pass -- with what the sidecar records, and raises `ByteGenomeMismatch` naming every
difference. `open` does not verify: a 3 GB hash on every open would cost the API its start-up,
and the size and tiling checks catch truncation. A caller that must know the bytes are the
recorded ones (an install, `validate --reference`) calls `verify()`; a caller that records the
reference (a build's manifest) records `digest()`.

**What stays in the engine.** Reading a FASTA (pysam) and deriving the file, and the canonical
install under `/data/opengwasdb/reference/` with its provenance, are opengwasdb's lane.

### CPRA formation (`ogquery.cpra`)

**The normal form moved here unchanged.** `ReferenceSequence`, `trim_alleles`,
`normalise_indel` and their errors are opengwasdb `0ba413e7`'s
`opengwasdb/variants/left_align.py`, byte for byte but for the import of `ogquery.identity`
and the module docstring; its tests came with it, and run against both an in-memory sequence
and a byte genome.

**`to_cpra(chrom, pos, a1, a2, *, sequence, declared_ref=None)`** follows ADR 0108 §2-§5 and
returns a frozen `CpraResult` (`status`, `cpra`, `chrom`, `pos`, `ref`, `alt`, `a1_is_alt`;
#518's `effect_is_alt` where `a1` is the effect allele). On a status other than `ok` every
other field is None. Where ADR 0108 left a choice open, it takes these (accepted by the
orchestrator in review, 28 Nov 2026):

- **A contig the sequence does not hold raises `ContigNotHeld`**, as malformed alleles raise
  the `identity` errors: a CPRA exists only on the primary assembly, so a caller filters
  other contigs first (`is_primary_contig`), and a byte genome missing a primary contig is
  the wrong file. It is not a status.
- **A position before 1 is `off_sequence`**, as one past a contig's end is. It is judged
  before any trim, so `AG/AA` at position 0 never becomes `G/A` at 1.
- **A declared indel whose REF span holds an `N` is `off_sequence`**, not `ref_mismatch`, as
  an SNV at an `N` is: the span is read and judged before the declared REF is compared, in
  both paths. A left shift that reaches an `N` (`UnreadableReferenceBase`) is `off_sequence`
  too.
- **An undeclared indel is `indel_unresolved` before the genome is read**, wherever it lies.
- **A declared equal-length substitution is checked whole, then trimmed.** ADR 0108 §3
  drops a declared REF the genome does not carry and keeps nothing unchecked, so the whole
  declared allele is compared with the genome's span: `CC/TC` declaring `CC` where the
  genome reads `CA` is `ref_mismatch`, as `bcftools norm -c e` reports it, not the `C:T`
  its trimmed REF would match. An **undeclared** one is trimmed first (ruling 9;
  `trim_alleles` is symmetric in its alleles) and then oriented by the genome's span at the
  trimmed position, so `CC/TC` at a genome `CA` is `C:T` (ruled in review, 28 Nov 2026). A
  multi-base substitution that stays one is formed like an SNV, REF being the genome's span.
- **A declared indel's REF has one owner**, `normalise_indel`, whose
  `ReferenceAlleleMismatch` `to_cpra` reports as `ref_mismatch`; `to_cpra` reads the span
  first only to judge `off_sequence`.
- **A sequence that answers with another number of bases than asked raises
  `ShortReferenceRead`**: a broken `ReferenceSequence` is not a mismatch.
- `declared_ref` names `"a1"` or `"a2"`, not an allele string, so a caller cannot name an
  allele that is neither.

### Measured on GRCh38

Measured on 28 Nov 2026 (a verification, not a test; the scripts and their output are under
the orchestration run's `scratch/ogq14/`, not in this repository):

- **Derived** from `/data/opengwasdb/reference/grch38/GRCh38.primary.fa.gz` (sha256
  `41b23c1f…68d6`, UCSC hg38's primary contigs), read with Python's `gzip` module and written
  through `write_byte_genome`, in 76 s. The data file is 3,088,286,401 bytes, sha256
  `73f33822468e1d17da51cef5faca7d7228d3cf02f04c639a56373e23bd2c2134`.
- **Its `digest()` is `263979d2475a76b5396c58dbeee0138d3548cf5011f53f2189662466c789e258`**,
  the digest opengwasdb computed through `FastaSequence` for that FASTA (ADR 0108 §9), in
  4.5 s; `verify()` passes.
- **Read against pysam** (0.24) on the same FASTA: 10,000 seeded random positions per contig
  (`bases`) and 1,000 spans of 1-200 bases per contig (`fetch`), 275,000 reads and 2.77 M bases in all: no disagreement.
- **`to_cpra` against `bcftools norm -f <FASTA> -m-any -c e`** (1.24), on every indel of
  OGS-00004 (3,373,064, reference-first, declared REF): all 3,373,064 agree, the 1,074 that
  bcftools respells among them, and so does the 10,000-row sample of those 1,074 plus seeded
  random others. None was `ref_mismatch` or `off_sequence`. `to_cpra` took 7.5 µs an indel
  in Python on a host at load 100.
- **`bases()` on 1 M random SNV sites**, warm, in 25 per-contig gathers: 12.1 ns a site
  sorted and 10.8 ns unsorted (median of five), within ADR 0108's 13-19 ns.

## Consequences

- opengwasdb's `opengwasdb.variants` re-exports `ogquery.cpra` and drops `left_align.py`,
  and derives and installs the GRCh38 byte genome through `write_byte_genome`, in its own
  lane. 0.4.0 manifests name `ByteGenome.digest()` as `cpra_normal_form.sequence.digest`.
- A byte genome is opened by every build and by the Data API; it is 3.1 GB on disk, read
  through the page cache, and nothing ogquery returns from it holds a view of the map.
- A change to the format that a 1.0 reader would misread moves `format_version`, and every
  reader refuses a version it does not know.
