#!/usr/bin/env python3
"""Serve the immutable graph and static Wiki included in a release image."""
from __future__ import annotations

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReleaseHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", "/graph.html")
            self.end_headers()
            return
        super().do_GET()


def main() -> int:
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    outputs = data_dir / "outputs"
    if not (outputs / "graph.html").is_file():
        raise FileNotFoundError(f"release graph is missing: {outputs}")
    host = os.environ.get("OKFOLIO_HOST", "0.0.0.0")
    port = int(os.environ.get("OKFOLIO_PORT", "8080"))
    handler = partial(ReleaseHandler, directory=str(outputs))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"serving={outputs} address=http://{host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
