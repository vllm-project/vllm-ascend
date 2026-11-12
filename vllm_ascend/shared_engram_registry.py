# SPDX-License-Identifier: Apache-2.0
"""Early owner membership on the existing conductor/kvmaster pod.

This small trusted-network registry is independent of model health. Membership
freezes once all owners register; replacements require a new deployment.
"""

import argparse
import ipaddress
import json
import secrets
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_CHUNK_ROWS = 1 << 22

MAX_REGISTRATION_BYTES = 4096


class OwnerRegistry:
    def __init__(self, count):
        if count not in (2, 4):
            raise ValueError("Engram registry requires two or four owners")
        self.count, self.identity = count, None
        self.members = {}
        self.membership = None
        self.deployment_id = secrets.token_hex(16)
        self.lock = threading.Lock()

    def register(self, request):
        if not isinstance(request, dict) or set(request) != {"node_id", "host", "port", "identity"}:
            raise ValueError("Invalid Engram owner registration")
        node, identity = request["node_id"], request["identity"]
        if not isinstance(node, str) or not 1 <= len(node) <= 253:
            raise ValueError("Invalid Engram owner identity")
        if not isinstance(identity, str) or len(identity) != 64 or set(identity) - set("0123456789abcdef"):
            raise ValueError("Invalid Engram checkpoint identity")
        host = str(ipaddress.ip_address(request["host"]))
        port = request["port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid Engram owner port")
        peer = dict(host=host, port=port)
        with self.lock:
            if self.identity not in (None, identity):
                raise ValueError("Engram checkpoint mismatch")
            if node in self.members and self.members[node] != peer:
                raise ValueError("Engram owner changed endpoint; restart the deployment")
            if node not in self.members:
                if self.membership is not None or peer in self.members.values():
                    raise ValueError("Engram membership is frozen or endpoint is duplicated")
                self.identity = identity
                self.members[node] = peer
            if len(self.members) == self.count and self.membership is None:
                self.membership = dict(
                    schema=1,
                    deployment_id=self.deployment_id,
                    chunk_rows=DEFAULT_CHUNK_ROWS,
                    owners=[self.members[key] for key in sorted(self.members)],
                )
            return self._status()

    def _status(self):
        return dict(
            state="ready" if self.membership is not None else "collecting",
            registered=len(self.members),
            expected=self.count,
            identity=self.identity,
            membership=self.membership,
        )

    def status(self):
        with self.lock:
            return self._status()


def make_server(host, port, registry):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def respond(self, code, result):
            payload = json.dumps(result).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/members":
                self.respond(404, dict(error="Unknown registry path"))
                return
            result = registry.status()
            self.respond(200 if result["state"] == "ready" else 202, result)

        def do_POST(self):
            if self.path != "/register":
                self.respond(404, dict(error="Unknown registry path"))
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REGISTRATION_BYTES:
                    raise ValueError("Invalid registration size")
                result = registry.register(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError) as exc:
                self.respond(409, dict(error=str(exc)))
                return
            self.respond(200 if result["state"] == "ready" else 202, result)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29698)
    parser.add_argument("--owners", type=int, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Existing kvmaster command after --")
    args = parser.parse_args()
    server = make_server(args.host, args.port, OwnerRegistry(args.owners))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    child = subprocess.Popen(command) if command else None
    try:
        while not stop.wait(1):
            if child is not None and child.poll() is not None:
                return child.returncode
            if not thread.is_alive():
                raise RuntimeError("Engram registry server stopped")
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        server.shutdown()
        server.server_close()
        thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
