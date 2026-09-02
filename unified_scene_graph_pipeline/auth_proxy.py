"""Minimal non-logging bearer-auth reverse proxy for the private vLLM service."""

from __future__ import annotations

import hmac
import http.client
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


UPSTREAM_HOST = os.environ.get("UPSTREAM_HOST", "qwen")
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "8000"))
API_KEY = os.environ["INFERENCE_API_KEY"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, "Bearer " + API_KEY)

    def _forward(self) -> None:
        if not self._authorized():
            body = b'{"error":"unauthorized"}\n'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in {"authorization", "host", "content-length", "connection", "accept-encoding"}
        }
        connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=900)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"transfer-encoding", "connection", "content-length", "content-encoding"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (ConnectionError, OSError, socket.timeout):
            payload = b'{"error":"upstream unavailable"}\n'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            connection.close()

    do_GET = _forward
    do_POST = _forward


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
