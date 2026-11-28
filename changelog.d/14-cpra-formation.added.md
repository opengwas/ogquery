- **`ogquery.cpra` holds a variant's VCF normal form, CPRA formation's one owner
  (opengwas/ogquery#14, opengwasdb ADR 0108 §7).** `trim_alleles`, `normalise_indel`, the
  `ReferenceSequence` protocol and their errors (`NotAnIndel`, `ReferenceAlleleMismatch`,
  `ContigNotHeld`, `PositionOutsideContig`, `UnreadableReferenceBase`) moved unchanged from
  opengwasdb `0ba413e7`'s `opengwasdb.variants.left_align`, with its tests, so opengwasdb can
  re-export them.
- **`ogquery.genome`: the byte genome, the memory-mapped uint8 reference CPRA formation reads
  (opengwas/ogquery#14, ADR 0003).** `write_byte_genome(path, contigs, *, assembly, provenance)`
  writes a release assembly's primary contigs as one raw file of upper-cased ASCII bases, in the
  variant order, beside a JSON sidecar (contig offsets, lengths and SHA-256s, opengwasdb ADR 0108
  §9's sequence digest, the file's SHA-256 and the caller's provenance), streaming chunks and
  publishing each name by a hard link, so an existing file is never replaced. Written from
  `/data/opengwasdb/reference/grch38/GRCh38.primary.fa.gz`, GRCh38's is 3,088,286,401 bytes,
  and its digest is `263979d2475a76b5396c58dbeee0138d3548cf5011f53f2189662466c789e258`, the
  FASTA's own; `bases()` reads REF at about 11-12 ns a site, warm (1 M random SNV sites), and
  275,000 reads agree with pysam's.
  `ByteGenome.open` maps it with numpy alone and refuses an unknown format
  version, a data file of the wrong size and offsets that do not tile it; it implements
  `ReferenceSequence` (`fetch`, `digest`), and `bases(chrom, positions)` gathers the base at
  millions of 1-based positions at once. `digest()` is computed from the bytes, never read from
  the sidecar, and `verify()` compares the two.
