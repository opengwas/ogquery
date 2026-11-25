"""Reading over HTTP needs the `http` extra, and a missing one says so (ogquery#2; ADR 0002).

zarr opens an http(s) URL through fsspec's HTTP filesystem, which runs on aiohttp. Neither is
one of ogquery's three dependencies, so `pip install ogquery` reads local Stores only, and
`ogquery.arrays.open_group` -- the one open every Store goes through -- checks for both before
zarr is asked, raising `HttpExtraMissing` naming `ogquery[http]`. These tests block the two
imports in a fresh interpreter, so they hold with the extra installed or not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ogquery import arrays

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"


def _without(blocked: tuple[str, ...], statements: str) -> dict[str, object]:
    """Run `statements` in a fresh interpreter in which `blocked` cannot be imported, and return
    the JSON it prints."""
    code = (
        "import json, sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        f"{statements}\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    found: dict[str, object] = json.loads(done.stdout)
    return found


OPEN_A_URL = (
    "import ogquery.ld\n"
    "from ogquery.arrays import HttpExtraMissing\n"
    "try:\n"
    "    ogquery.ld.open('http://127.0.0.1:9/store/')\n"
    "    out = {'raised': None}\n"
    "except Exception as error:\n"
    "    out = {'raised': type(error).__name__, 'message': str(error),\n"
    "           'import_error': isinstance(error, ImportError),\n"
    "           'missing_extra': isinstance(error, HttpExtraMissing)}\n"
)


@pytest.mark.parametrize(
    "blocked", [("fsspec", "aiohttp"), ("fsspec",), ("aiohttp",)], ids=["both", "fsspec", "aiohttp"]
)
def test_a_url_without_the_extra_names_it(blocked: tuple[str, ...]) -> None:
    out = _without(blocked, OPEN_A_URL + "print(json.dumps(out))")
    assert out["raised"] == "HttpExtraMissing", out
    assert out["import_error"] and out["missing_extra"]
    message = str(out["message"])
    assert "pip install 'ogquery[http]'" in message
    for name in blocked:
        assert name in message.split("not importable:")[1], message


def test_a_local_store_reads_without_the_extra() -> None:
    statements = (
        "import ogquery.ld\n"
        f"store = ogquery.ld.open({str(TEST_STORE)!r})\n"
        "rows = store.variants_in('22', 1, 3_000_000_000).alid.tolist()[:40]\n"
        "m = store.matrix(rows)\n"
        "c = store.clump(rows, list(range(40)), r2=0.1, kb=1000)\n"
        "p = store.proxies(rows, r2=0.5, kb=1000)\n"
        "out = {'shape': list(m.r.shape), 'clumped': len(c), 'proxies': len(p),\n"
        "       'loaded': [n for n in ('fsspec', 'aiohttp') if sys.modules.get(n) is not None]}\n"
        "print(json.dumps(out))"
    )
    out = _without(("fsspec", "aiohttp"), statements)
    assert out == {"shape": [40, 40], "clumped": 40, "proxies": out["proxies"], "loaded": []}
    assert isinstance(out["proxies"], int) and out["proxies"] > 0


@pytest.mark.parametrize(
    ("location", "scheme"),
    [
        ("https://example.org/store", "https"),
        ("HTTP://example.org/store", "http"),
        ("s3://bucket/store", "s3"),
        ("file:///data/store", "file"),
        ("/data/store", None),
        ("store", None),
        ("C:\\data\\store", None),
        ("C:/data/store", None),
        ("c:store", None),
    ],
)
def test_a_location_s_scheme(location: str, scheme: str | None) -> None:
    # A Windows drive is a path, not a one-letter scheme.
    assert arrays._scheme(location) == scheme


@pytest.mark.parametrize("location", ["s3://bucket/store", "file:///data/store"])
def test_a_url_of_another_scheme_is_refused_not_read_as_a_path(location: str) -> None:
    with pytest.raises(ValueError, match=r"local path or an http\(s\) URL, not a \w+:// URL"):
        arrays.open_group(location)
