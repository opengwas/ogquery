- **Genotype Stores: `ogquery.ld.open` reads them, behind a new `genotype` extra
  (opengwas/ogquery#11, ADR 0002 amended).** A Genotype Store, opened from a local path, answers
  `matrix`, `region`, `proxies` and `clump` by computing r from the pgens it pins; its reader,
  `ogquery.ld.genotypes` (`GenotypeReader`, `Genotypes`, `Block`, `r`), moved from opengwasdb
  `964ac1b2` and gives that commit's answers and refusals, checked on the Test Panel and on the
  real HGDP+1kGP AMR store. It needs `pip install 'ogquery[genotype]'` (pgenlib 0.94 or later,
  threadpoolctl 3.5 or later); without it, opening a Genotype Store raises
  `ogquery.opening.GenotypeExtraMissing`, an `ImportError` naming the extra, where it raised an
  `InputError` naming opengwasdb. pgenlib publishes no Windows wheel, so Genotype Stores are
  tested on Linux and macOS. Its one-BLAS-thread limit holds only a BLAS threadpoolctl controls,
  so it does nothing with NumPy's Accelerate wheels on macOS. A plain `pip install ogquery` still installs only numpy, zarr and
  numcodecs, and LD and GWAS Stores load neither new module. Opening a Genotype Store by an
  http(s) URL is a new refusal, an `InputError` saying it is read from a local path only, where
  opengwasdb's reader raised a missing-input error naming a mangled path. `ogquery.ld.format.REncoding`
  gains `quantise`, `stored_mask` and `partner_mask`, which the reader keeps pairs by.
