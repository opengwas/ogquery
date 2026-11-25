- **LD Stores: `ogquery.ld` reads them, as `opengwasdb.ld` did, with no build code
  (opengwas/ogquery#2, ADR 0002).** `ogquery.ld.open(path_or_url)` opens an LD Store, and
  `LDStore.matrix`, `region`, `proxies` and `clump` answer as opengwasdb `c8ebe4ca`'s reader
  answers, with the same results (`LDMatrix`, `Proxies`, `Clumps`, `Variants`) and refusals
  (`Refusal`, `RefusalReason`, `InputError`); the LD Store format is unchanged. Three reads
  ogdb-api needs are public (opengwas/ogld#79): `LDStore.variants_in(chrom, start, stop)`, the
  Variant Set in a range read before any r, `LDStore.count_in`, its length, and
  `LDStore.has(alids)`, one boolean per ALID. Reading over HTTP(S) needs the new `http` extra
  (`pip install 'ogquery[http]'`: fsspec and aiohttp); without it an `http(s)://` open raises
  `ogquery.opening.HttpExtraMissing`, naming the extra, and local Stores read as before, on
  Windows too (a local frame is a seek and a read where there is no `os.pread`). A
  Genotype Store is refused, naming `opengwasdb.ld.open`; `ogquery.ld.Reader` is the protocol
  its reader plugs into. `ogquery.identity` gains `chromosome_rank` and `chromosome_sort_key`,
  and `ogquery.alid` the sorted-spelling reads (`sorted_alid_locus`,
  `spells_indel_reference_first`, `ReferenceFirstIndel`, `refuse_reference_first_indel`), the
  owners opengwasdb will re-export. `import ogquery.ld` loads no opengwasdb, pgenlib,
  threadpoolctl, aiohttp, scipy or pysam, and leaves zarr's process-wide configuration at its
  defaults: the one read-only open is now `ogquery.opening.open_group`, which sets nothing
  (`ogquery.arrays.open_group` re-exports it, and importing `ogquery.arrays` still configures the
  GWAS read path as before).
