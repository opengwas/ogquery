"""The one read-only open of a Store's Zarr group, local or over HTTP(S) (ogquery#2).

`open_group` is the one place a reader calls `zarr.open_group`: a GWAS Store Release's open
(`ogquery.arrays.open_group`, which re-exports it) and an LD Store's (`ogquery.ld`). It sets
none of zarr's process-wide configuration. The GWAS read path's configuration -- the fused codec
pipeline, Blosc's threads, empty chunks written (ADR 0056) -- is applied by `ogquery.arrays`,
which every GWAS read imports; the LD read path does not import it, so an LD-only process reads
under zarr's defaults, as ogld's readers did. opengwasdb measured `proxies` 17-29 % slower cold
under the fused pipeline on real LD Stores (its ADR 0094 amendment, #460; ADR 0002).

An http(s) URL opens through fsspec and aiohttp, which only the `http` extra installs. A Genotype
Store is read with pgenlib and threadpoolctl, which only the `genotype` extra installs:
`require_genotype` is that extra's check, as `require_http` is the `http` extra's.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import zarr
from zarr.core.sync import sync as zarr_sync
from zarr.storage import LocalStore

#: The schemes `open_group` reads over the network, and what reading them imports: zarr
#: opens an http(s) URL through fsspec's HTTP filesystem, which is async and runs on aiohttp.
HTTP_SCHEMES: Final = ("http", "https")
HTTP_MODULES: Final = ("fsspec", "aiohttp")
#: What installs `HTTP_MODULES` (pyproject.toml's `http` extra).
HTTP_EXTRA: Final = "ogquery[http]"


class HttpExtraMissing(ImportError):
    """An http(s) location, with the `http` extra not installed: the error names the extra,
    rather than leaving zarr to fail on whichever module it imports first."""


#: What reading a Genotype Store imports: pgenlib decodes the pgens it pins, and threadpoolctl
#: holds BLAS to one thread a request (`ogquery.ld.genotypes`, the one package that imports them).
GENOTYPE_MODULES: Final = ("pgenlib", "threadpoolctl")
#: What installs `GENOTYPE_MODULES` (pyproject.toml's `genotype` extra).
GENOTYPE_EXTRA: Final = "ogquery[genotype]"


class GenotypeExtraMissing(ImportError):
    """A Genotype Store, with the `genotype` extra not installed: the error names the extra,
    rather than leaving the reader to fail on whichever module it imports first."""


def open_group(path: str | os.PathLike[str]) -> Any:
    """Open a Store's Zarr group read-only: a local path, or an http(s) URL.

    The one place a reader calls `zarr.open_group`. The stored metadata decides the
    Zarr format, so a 0.1.0 (Zarr v2) release opens as a 0.2.x (v3) one does. Every
    write -- and every other open mode -- goes through `opengwasdb.store.arrays`
    (`open_group`, `open_group_for_write`).

    A local path opens on zarr's `LocalStore`. An http(s) URL opens as zarr opens
    one, through fsspec and aiohttp, which only the `http` extra installs: without
    them this raises `HttpExtraMissing`, naming `ogquery[http]`, before zarr is
    asked. Any other URL scheme is a `ValueError`: no reader here reads it, and
    read as a local path it would name a directory that is not there.
    """
    location = os.fspath(path)
    scheme = _scheme(location)
    if scheme in HTTP_SCHEMES:
        require_http(location)
        # A scheme is case-insensitive (RFC 3986 §3.1), and fsspec knows `http`, not `HTTP`:
        # the URL reaches zarr with its scheme lower-cased and the rest as given.
        url = scheme + location[len(scheme) :]
        return zarr.open_group(url, mode="r")
    if scheme is not None:
        raise ValueError(
            f"{location}: a Store is opened from a local path or an http(s) URL, not a "
            f"{scheme}:// URL"
        )
    store: LocalStore = zarr_sync(LocalStore.open(root=Path(location), mode="r", read_only=True))
    return zarr.open_group(store, mode="r", zarr_format=None)


def is_http(location: str | os.PathLike[str]) -> bool:
    """Whether `location` is an http(s) URL, by the scheme test `open_group` reads one by
    (`_scheme`, in any case; a Windows drive, `C:\\`, is a local path)."""
    return _scheme(os.fspath(location)) in HTTP_SCHEMES


def require_http(location: str) -> None:
    """Raise `HttpExtraMissing`, naming the extra and the modules it lacks, unless fsspec and
    aiohttp are both installed. They are found, not imported: zarr imports them, and ogquery
    imports nothing outside its three dependencies itself."""
    missing = _missing(HTTP_MODULES)
    if missing:
        raise HttpExtraMissing(
            f"{location} is an http(s) URL, and reading one needs {' and '.join(HTTP_MODULES)}, "
            f"which ogquery installs only with its `http` extra: pip install '{HTTP_EXTRA}' "
            f"(not importable: {', '.join(missing)})"
        )


def require_genotype(location: str | None = None) -> None:
    """Raise `GenotypeExtraMissing`, naming the extra and the modules it lacks, unless pgenlib and
    threadpoolctl are both installed; `location` is the Genotype Store being opened, if one is.
    They are found, not imported: only `ogquery.ld.genotypes` imports them, once this passes."""
    missing = _missing(GENOTYPE_MODULES)
    if missing:
        subject = (
            "reading a Genotype Store"
            if location is None
            else f"{location} is a Genotype Store, and reading one"
        )
        raise GenotypeExtraMissing(
            f"{subject} needs {' and '.join(GENOTYPE_MODULES)}, which ogquery installs only "
            f"with its `genotype` extra: pip install '{GENOTYPE_EXTRA}' "
            f"(not importable: {', '.join(missing)})"
        )


def _missing(modules: tuple[str, ...]) -> list[str]:
    """Which of an extra's `modules` cannot be imported, in order: found by `find_spec`, never
    imported, so checking an extra loads nothing (`require_http`, `require_genotype`)."""
    return [name for name in modules if importlib.util.find_spec(name) is None]


def _scheme(location: str) -> str | None:
    """The URL scheme of `location`, lower-cased, or None for a local path. A Windows drive
    (`C:\\` or `C:/`) is a path: a scheme is at least two letters and is followed by `//`."""
    parts = urlsplit(location)
    if len(parts.scheme) < 2 or not location[len(parts.scheme) + 1 :].startswith("//"):
        return None
    return parts.scheme.lower()
