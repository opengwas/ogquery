"""A directory served over HTTP from this machine, for the tests that read a Store by URL.

One owner of the server's lifecycle: started on a free port, and on exit shut down, its thread
joined and its socket closed. `contract/conftest.py` serves the Test Store with it, and
`test_open.py` a Genotype Store. Each request is answered by `support.RangeHandler`, which honours
the byte ranges Zarr over HTTP asks for.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from support import RangeHandler


@contextmanager
def serve(directory: Path) -> Iterator[str]:
    """`directory` served over HTTP on 127.0.0.1, as its URL, until the block ends."""
    handler = partial(RangeHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    # shutdown() waits out one poll: 0.05 s, not serve_forever's default 0.5 s, per test.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
