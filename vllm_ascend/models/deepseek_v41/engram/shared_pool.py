# SPDX-License-Identifier: Apache-2.0
"""Startup manifests and authenticated control leases; no table data on TCP.

NFS may have local-only locks, so leases live in the owner process, not files.
Use a fresh private directory per generation and a trusted cluster network.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import threading
import time
from pathlib import Path

import torch

from .shared_host import MappedHostBuffer, retry_pending_rollbacks

SCHEMA = 1
DTYPES = {"int8": torch.int8, "float32": torch.float32}
MAX_CONTROL_BYTES = 1 << 20


def _close_buffers(buffers, library_path):
    first_error = None
    for buffer in reversed(list(buffers.values())):
        try:
            buffer.close()
        except Exception as exc:
            first_error = first_error or exc
    try:
        retry_pending_rollbacks(library_path)
    except Exception as exc:
        first_error = first_error or exc
    if first_error is not None:
        raise first_error


def checkpoint_identity(root):
    """Hash config and index, not tensor payloads; use with an immutable model ref."""
    root = Path(root)
    digest = hashlib.sha256()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _private(root):
    stat = root.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError("Shared Engram directory must be uid-owned and mode0700")


def _send(wire, value):
    pending = memoryview((json.dumps(value) + "\n").encode())
    while pending:
        written = wire.write(pending)
        if not written:
            raise ConnectionError("Engram control connection stopped accepting data")
        pending = pending[written:]


def _receive(wire):
    line = wire.readline(MAX_CONTROL_BYTES + 1)
    if not line or len(line) > MAX_CONTROL_BYTES or not line.endswith(b"\n"):
        raise ConnectionError("Incomplete or oversized Engram control message")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected an Engram control object")
    return value


def owner_alive(root):
    try:
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        with (
            socket.create_connection(tuple(manifest["endpoint"]), timeout=5) as conn,
            conn.makefile("rwb", buffering=0) as wire,
        ):
            _send(wire, dict(op="ping", token=manifest["token"]))
            return _receive(wire).get("alive", False)
    except (OSError, ValueError, KeyError):
        return False


class SharedPoolOwner:
    def __init__(self, root, identity, *, device, library_path=None, host="127.0.0.1", port=0, network_bootstrap=False):
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        _private(self.root)
        self.device, self.library_path = device, library_path
        self.network_bootstrap = network_bootstrap
        self.buffers = {}
        self.closed = False
        self._condition = threading.Condition()
        self._leases = set()
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(128)
        self._listener.settimeout(0.5)
        self.manifest = dict(
            schema=SCHEMA,
            identity=identity,
            generation=secrets.token_hex(16),
            state="loading",
            tensors={},
            endpoint=[host, self._listener.getsockname()[1]],
            token=secrets.token_hex(32),
        )
        self._publish_manifest()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _publish_manifest(self):
        temporary = self.root / "manifest.tmp"
        with temporary.open("w") as file:
            json.dump(self.manifest, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(self.root / "manifest.json")

    def _accept(self):
        while not self.closed:
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        acquired = False
        try:
            with conn:
                conn.settimeout(30)
                with conn.makefile("rwb", buffering=0) as wire:
                    request = _receive(wire)
                    token = request.get("token")
                    public_bootstrap = request.get("op") == "bootstrap" and self.network_bootstrap
                    if not public_bootstrap and (
                        not isinstance(token, str) or not hmac.compare_digest(token, self.manifest["token"])
                    ):
                        _send(wire, dict(error="Invalid Engram capability"))
                        return
                    with self._condition:
                        if request.get("op") == "ping":
                            _send(
                                wire,
                                dict(
                                    alive=not self.closed,
                                    state=self.manifest["state"],
                                    identity=self.manifest["identity"],
                                    generation=self.manifest["generation"],
                                    leases=len(self._leases),
                                ),
                            )
                            return
                        if request.get("op") == "bootstrap":
                            if request.get("identity") != self.manifest["identity"]:
                                _send(wire, dict(error="Engram checkpoint identity mismatch"))
                                return
                            if self.manifest["state"] != "ready":
                                _send(wire, dict(error="Engram pool is not ready"))
                                return
                            descriptors = {
                                entry["descriptor"]: base64.b64encode(
                                    (self.root / entry["descriptor"]).read_bytes()
                                ).decode("ascii")
                                for entry in self.manifest["tensors"].values()
                            }
                            _send(wire, dict(manifest=self.manifest, descriptors=descriptors))
                            return
                        if (
                            request.get("op") != "attach"
                            or self.manifest["state"] != "ready"
                            or request.get("identity") != self.manifest["identity"]
                            or request.get("generation") != self.manifest["generation"]
                        ):
                            _send(wire, dict(error="Engram pool is not ready or model-compatible"))
                            return
                        self._leases.add(conn)
                        acquired = True
                        _send(wire, dict(manifest=self.manifest))
                    conn.settimeout(None)
                    if _receive(wire) != {"op": "release"}:
                        raise ValueError("Expected mapping-release acknowledgement")
                    with self._condition:
                        self._leases.remove(conn)
                        acquired = False
                        _send(wire, dict(released=True))
                        self._condition.notify_all()
        except (OSError, ValueError):
            if acquired:
                # EOF is not proof that a device mapping was safely released.
                # Retain this orphaned lease; operator recovery must first stop
                # all consumers before terminating the owner generation.
                print("ENGRAM_ORPHANED_LEASE owner memory retained", flush=True)

    def add(self, key, rows, width, dtype, *, global_rows=None, row_start=0):
        if self.manifest["state"] != "loading" or key in self.buffers:
            raise ValueError("Cannot add to a published pool or duplicate a tensor")
        if global_rows is not None and not 0 <= row_start < row_start + rows <= global_rows:
            raise ValueError("Invalid Engram shard range")
        name = str(dtype).removeprefix("torch.")
        if name not in DTYPES:
            raise ValueError("Only INT8 codes and FP32 scales are supported")
        descriptor = hashlib.sha256(key.encode()).hexdigest()
        buffer = MappedHostBuffer(
            self.root / descriptor, rows, width, dtype, owner=True, device=self.device, library_path=self.library_path
        )
        self.buffers[key] = buffer
        self.manifest["tensors"][key] = dict(rows=rows, width=width, dtype=name, descriptor=descriptor)
        if global_rows is not None:
            self.manifest["tensors"][key].update(global_rows=global_rows, row_start=row_start)
        return buffer

    def ready(self):
        with self._condition:
            if self.manifest["state"] != "loading" or not self.buffers:
                raise ValueError("Only a populated loading pool can be published")
            self.manifest["state"] = "ready"
            self._publish_manifest()

    def close(self, timeout=0):
        if self.closed:
            return
        with self._condition:
            self.manifest["state"] = "draining"
            self._publish_manifest()
            deadline = time.monotonic() + timeout
            while self._leases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Shared Engram consumers have not released their mappings")
                self._condition.wait(remaining)
            _close_buffers(self.buffers, self.library_path)
            self.manifest["state"] = "closed"
            self._publish_manifest()
            self.closed = True
        self._listener.close()
        self._thread.join(timeout=1)


class SharedPoolClient:
    def __init__(self, root, identity, *, device, library_path=None):
        self.root = Path(root)
        _private(self.root)
        discovery = json.loads((self.root / "manifest.json").read_text())
        self.buffers, self.views = {}, []
        self.closed = False
        self.device, self.library_path = device, library_path
        self._conn = socket.create_connection(tuple(discovery["endpoint"]), timeout=30)
        self._wire = self._conn.makefile("rwb", buffering=0)
        try:
            _send(
                self._wire,
                dict(op="attach", token=discovery["token"], identity=identity, generation=discovery["generation"]),
            )
            response = _receive(self._wire)
            if "error" in response:
                raise ValueError(response["error"])
            self.manifest = response["manifest"]
            if (
                self.manifest.get("schema") != SCHEMA
                or self.manifest.get("identity") != identity
                or self.manifest.get("generation") != discovery["generation"]
            ):
                _send(self._wire, dict(op="release"))
                _receive(self._wire)
                raise ValueError("Incompatible shared Engram manifest")
        except BaseException:
            self._wire.close()
            self._conn.close()
            raise

    def view(self, key, start, end, *, rows, width, dtype, chunk_rows=None):
        if self.closed:
            raise RuntimeError("Shared Engram client is closed")
        metadata = self.manifest["tensors"][key]
        if (metadata["rows"], metadata["width"], metadata["dtype"]) != (rows, width, str(dtype).removeprefix("torch.")):
            raise ValueError("Shared Engram tensor layout mismatch")
        descriptor = hashlib.sha256(key.encode()).hexdigest()
        if metadata["descriptor"] != descriptor:
            raise ValueError("Invalid Engram descriptor name")
        if key not in self.buffers:
            self.buffers[key] = MappedHostBuffer(
                self.root / descriptor,
                rows,
                width,
                dtype,
                owner=False,
                device=self.device,
                library_path=self.library_path,
            )
        view = (
            self.buffers[key].view(start, end)
            if chunk_rows is None
            else self.buffers[key].view(start, end, chunk_rows=chunk_rows)
        )
        self.views.append(view)
        return view

    def close(self):
        if self.closed:
            return
        for view in self.views:
            view.close()
        _close_buffers(self.buffers, self.library_path)
        _send(self._wire, dict(op="release"))
        if _receive(self._wire) != {"released": True}:
            raise RuntimeError("Shared Engram release was not acknowledged")
        self._wire.close()
        self._conn.close()
        self.closed = True
