# ogquery

The light, read-only query package for OpenGWAS **GWAS Stores** and **LD Stores**.
It is the Store Reader: the one read path that the OpenGWASDB builder and
validator, the Data API (ogdb-api), opengwaspy and opengwasr all use.

> **Status: under construction.** Tracked by
> [opengwas/opengwasdb#450](https://github.com/opengwas/opengwasdb/issues/450).
> Nothing here is usable yet.

- **Core dependencies:** numpy, zarr and numcodecs only. No pysam, no SQLite,
  nothing to compile. It runs on Linux, macOS and Windows.
- **Optional extras:** reading LD Stores over HTTP (fsspec, aiohttp) and
  reading Genotype Stores.
- **Contract:** the Store format specs plus golden fixture results, which any
  reader, including a future Rust core shipped inside the same wheels, must
  reproduce.

Most users never install ogquery by name. They install
[opengwaspy](https://github.com/opengwas/opengwaspy), whose Local Source reads
through it, or [opengwasr](https://github.com/opengwas/opengwasr), which reaches
it through opengwaspy.

## Licence

GPL-3.0-only. See [LICENSE](./LICENSE). Copyright © University of Bristol.
Contributions from outside the University of Bristol need a signed
[Contributor License Agreement](./CLA.md).
