- **ogquery 0.1.0: opengwasdb's Store Reader as its own package, on Linux, macOS
  and Windows (opengwas/ogquery#1, ADR 0001).** `pip install ogquery` installs
  numpy (2.0 or later), zarr (3.4 or later) and numcodecs (0.17 or later) and
  nothing else, all as wheels, on Python 3.12 to 3.14. The modules are
  `opengwasdb.store_reader`'s, by the same names (`ogquery.release.open_release`,
  `ogquery.facade.query_release`, ...), `opengwasdb.store.query_digests` is
  `ogquery.digests`, and the package carries `py.typed`. It reads format 0.3.0
  Store Releases alone and answers the Store Reader contract bundle, opengwasdb
  `69c29452`'s, on all three operating systems. A 0.2.x or 0.1.0 release needs a
  reader for its Store Variant Table, `open_release(path, legacy=...)`, as
  opengwasdb passes; without one it is refused with `VariantTableReaderRequired`,
  which names its `format_version` and the way to 0.3.0: `ogdb rewrite-release`
  for a 0.2.3 release, a rebuild from source for an earlier one.
