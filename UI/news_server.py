#!/usr/bin/env python3
"""Serve the Trending-news page (UI/trending.html), the advanced dashboard (UI/news.html) and their read-only API.

  .venv/bin/python UI/news_server.py 8878                 # http://127.0.0.1:8878/
  (reads the live PostgreSQL schema `news_engine`; credentials from the environment, see deploy/dashboard.env)

Independent of UI/api_server.py (the social-media dashboard), which is left untouched. Binds to localhost
unless --host is given. Serves only news.html + assets/ and /api/news/*. Nothing here writes to any database.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import news_api as NA  # noqa: E402

API = NA.NewsAPI()
STATIC_OK = ("/trending.html", "/assets/trending.css", "/assets/trending.js", "/assets/fortress.css", "/news.html", "/assets/dash.css", "/assets/news.css",
             "/assets/news.js", "/favicon.ico")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(NA.clean(obj), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _api(self, fn):
        try:
            self._json(fn())
        except NA.ApiError as e:
            self._json({"error": str(e)}, e.status)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            self._json({"error": f"internal error: {type(e).__name__}"}, 500)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            return self._json({"ok": True})
        if u.path.startswith("/api/news/"):
            return self._api(lambda: NA.route_get(API, u.path, parse_qs(u.query)))
        if u.path in ("/", ""):
            self.path = "/trending.html"           # the simple Top-trending page; the advanced dashboard is /news.html
            u = urlparse(self.path)
        if u.path not in STATIC_OK:
            return self._json({"error": "not found"}, 404)
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/news/"):
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "bad Content-Length"}, 400)
        if n <= 0 or n > NA.MAX_BODY:
            return self._json({"error": f"body must be 1..{NA.MAX_BODY} bytes"}, 413 if n > NA.MAX_BODY else 400)
        try:
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._json({"error": "body must be a JSON object"}, 400)
        self._api(lambda: NA.route_post(API, u.path, body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8878)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"News dashboard on http://{a.host}:{a.port}/  (PostgreSQL schema: {API.db_name}; read-only)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
