# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent node-local immutable MXFP8 tables.

The owner CLI does not import the model or initialize the NPU.
Consumers receive size-sealed memfds, register and then protect their mappings.
"""

import argparse
import array
import contextlib
import fcntl
import hashlib
import json
import mmap
import os
import select
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import NamedTuple

PROTOCOL_VERSION = 1
# Linux UAPI value on Ascend host architectures, including older Python builds.
SO_PEERPIDFD = getattr(socket, "SO_PEERPIDFD", 77)
COPY_BYTES = 16 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
INDEX_FILES = (
    "quant_model_weights.safetensors.index.json",
    "model.safetensors.index.json",
)


def checkpoint_layout(model_path):
    """Read only checkpoint headers and fingerprint the selected table sources."""
    root = Path(model_path).resolve()
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    text = config.get("text_config", config)
    layers = text["engram_layer_ids"]
    rows = text["engram_num_embeddings"]
    width = text["engram_head_dim"]
    if len(layers) != len(rows) or width % 32:
        raise ValueError("Invalid Engram model layout")
    indices = []
    for filename in INDEX_FILES:
        path = root / filename
        if path.is_file():
            indices.append(json.loads(path.read_bytes())["weight_map"])
    tables = {}
    for layer, row_count in zip(layers, rows):
        key = f"layers.{layer}.engram.embed.weight"
        index = next((index for index in indices if key in index), None)
        if index is None:
            raise ValueError(f"Missing indexed Engram table: {key}")
        parts = {}
        for kind, name, shape, dtypes in (
            ("codes", key, [row_count, width], ("F8_E4M3", "F8_E4M3FN")),
            ("scales", key.removesuffix(".weight") + ".scale", [row_count, width // 32], ("F8_E8M0",)),
        ):
            shard = (root / index[name]).resolve()
            if not shard.is_relative_to(root):
                raise ValueError("Checkpoint shard escapes the model directory")
            with shard.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                if header_size > MAX_MESSAGE_BYTES * 64:
                    raise ValueError("Safetensors header is too large")
                header = json.loads(source.read(header_size))
                entry = header[name]
                stat = os.fstat(source.fileno())
            start, end = entry["data_offsets"]
            size = row_count * shape[1]
            offset = 8 + header_size + start
            if (
                entry["shape"] != shape
                or entry["dtype"] not in dtypes
                or end - start != size
                or start < 0
                or offset + size > stat.st_size
            ):
                raise ValueError(f"Unsupported MXFP8/E8M0 table header: {name}")
            parts[kind] = {
                "name": name,
                "shape": shape,
                "dtype": entry["dtype"],
                "shard": str(shard),
                "offset": offset,
                "size": size,
                "source_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
            }
        tables[key] = parts
    identity = {"config_sha256": hashlib.sha256(config_bytes).hexdigest(), "tables": tables}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "fingerprint": fingerprint,
        "tables": tables,
        "allocation_bytes": sum(p["size"] for t in tables.values() for p in t.values()),
    }


def _recv_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise EOFError("Engram control connection closed")
        data.extend(part)
    return bytes(data)


def _send_json(connection, message):
    data = json.dumps(message).encode()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Engram control message too large")
    connection.sendall(struct.pack("!I", len(data)) + data)


def _recv_json(connection):
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Engram control message too large")
    return json.loads(_recv_exact(connection, size))


def request(endpoint, message, timeout=30):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(endpoint)
        _send_json(connection, message)
        return _recv_json(connection)


def _close_fds(fds):
    for fd in fds:
        with contextlib.suppress(OSError):
            os.close(fd)


def _peer_pidfd(connection: socket.socket, peer_pid: int) -> int | None:
    # SO_PEERPIDFD identifies the connecting process even across PID namespaces.
    with contextlib.suppress(OSError):
        return connection.getsockopt(socket.SOL_SOCKET, SO_PEERPIDFD)
    # Older kernels only support pidfd_open. PID 0 is not visible to this owner.
    if peer_pid > 0:
        with contextlib.suppress(OSError):
            return os.pidfd_open(peer_pid)
    return None


class _Lease(NamedTuple):
    pidfd: int | None
    stale: bool = False


class NodeTableLease:
    """A control lease kept until device synchronization and unregister finish."""

    def __init__(self, endpoint, generation, fingerprint, key, timeout=1800):
        deadline = time.monotonic() + timeout
        self.connection: socket.socket | None = None
        self.fds: list[int] = []
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(min(30, max(1, deadline - time.monotonic())))
            try:
                connection.connect(endpoint)
                _send_json(
                    connection,
                    {
                        "op": "attach",
                        "version": PROTOCOL_VERSION,
                        "generation": generation,
                        "fingerprint": fingerprint,
                        "key": key,
                    },
                )
                response = _recv_json(connection)
                if response.get("state") == "loading":
                    connection.close()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Engram owner is still loading")
                    time.sleep(0.2)
                    continue
                if not response.get("ok"):
                    raise RuntimeError(f"Engram attach rejected: {response}")
                received, ancillary, flags, _ = connection.recvmsg(
                    1, socket.CMSG_SPACE(2 * array.array("i").itemsize), socket.MSG_CMSG_CLOEXEC
                )
                for level, kind, data in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        handles = array.array("i")
                        handles.frombytes(data)
                        self.fds.extend(handles)
                if received != b"F" or flags & socket.MSG_CTRUNC or len(self.fds) != 2:
                    raise RuntimeError("Invalid Engram FD response")
                self.metadata = response
                self.connection = connection
                break
            except (TimeoutError, FileNotFoundError, ConnectionRefusedError):
                connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Engram owner unavailable") from None
                time.sleep(0.2)
            except BaseException:
                connection.close()
                _close_fds(self.fds)
                self.fds.clear()
                raise

    def release(self) -> None:
        connection = self.connection
        if connection is None:
            return
        self.connection = None
        try:
            _send_json(connection, {"op": "release", "lease": self.metadata["lease"]})
            reply = _recv_json(connection)
            if not reply.get("ok"):
                raise RuntimeError(f"Engram release rejected: {reply}")
        finally:
            with contextlib.suppress(OSError):
                connection.close()
            _close_fds(self.fds)
            self.fds.clear()


def _load_sealed_memfd(part):
    fd = os.memfd_create("engram-" + part["name"], os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, part["size"])
        with mmap.mmap(fd, part["size"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE) as target:
            # Fault pages only after this hint. posix_fallocate materializes
            # 4 KiB shmem pages before the VMA can request transparent huge
            # pages, greatly increasing A5 registration time and page tables.
            target.madvise(mmap.MADV_HUGEPAGE)
            buffer = bytearray(min(COPY_BYTES, part["size"]))
            with open(part["shard"], "rb", buffering=0) as source:
                stat = os.fstat(source.fileno())
                actual = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
                if actual != part["source_stat"]:
                    raise ValueError("Checkpoint changed before owner load")
                source.seek(part["offset"])
                offset = 0
                while offset < part["size"]:
                    size = min(len(buffer), part["size"] - offset)
                    view = memoryview(buffer)[:size]
                    if source.readinto(view) != size:
                        raise EOFError("Short Engram checkpoint read")
                    target[offset : offset + size] = view
                    offset += size
        # CANN pins host ranges with write access even for read-only lookups.
        # Consumers must register before applying mprotect(PROT_READ).
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return fd
    except BaseException:
        os.close(fd)
        raise


class NodeTableOwner:
    """Own immutable backing across independent engine lifetimes."""

    def __init__(self, endpoint, generation, model_path):
        self.endpoint = endpoint
        self.generation = generation
        self.model_path = model_path
        self.layout = checkpoint_layout(model_path)
        self.state = "loading"
        self.error: str | None = None
        self.handles: dict[str, list[int]] = {}
        self.leases: dict[str, _Lease] = {}
        self.lock = threading.Lock()

    def load(self):
        loaded: dict[str, list[int]] = {}
        try:
            available = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            if self.layout["allocation_bytes"] > available * 0.9:
                raise MemoryError("Not enough host memory for one Engram table")
            for key, parts in self.layout["tables"].items():
                handles: list[int] = []
                loaded[key] = handles
                for kind in ("codes", "scales"):
                    handles.append(_load_sealed_memfd(parts[kind]))
                print(f"Loaded {key}: {sum(p['size'] for p in parts.values())} bytes", flush=True)
            if checkpoint_layout(self.model_path) != self.layout:
                raise ValueError("Checkpoint changed while loading")
            with self.lock:
                if self.state == "loading":
                    self.handles = loaded
                    loaded = {}
                    self.state = "ready"
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"
            print(self.error, flush=True)
        finally:
            for handles in loaded.values():
                _close_fds(handles)

    def status(self):
        return {
            "ok": self.state != "failed",
            "state": self.state,
            "generation": self.generation,
            "fingerprint": self.layout["fingerprint"],
            "allocation_bytes": self.layout["allocation_bytes"],
            "backing_inodes": {key: [os.fstat(fd).st_ino for fd in handles] for key, handles in self.handles.items()},
            "active_leases": sum(not lease.stale for lease in self.leases.values()),
            "stale_leases": sum(lease.stale for lease in self.leases.values()),
            "error": self.error,
        }

    def _remove_lease(self, lease: str) -> None:
        pidfd = self.leases.pop(lease).pidfd
        if pidfd is not None:
            _close_fds((pidfd,))

    def _reap_stale_leases(self) -> None:
        # Called under self.lock. Readable pidfds indicate an exited process.
        stale = {info.pidfd: lease for lease, info in self.leases.items() if info.stale and info.pidfd is not None}
        if stale:
            poller = select.poll()
            for pidfd in stale:
                poller.register(pidfd, select.POLLIN)
            for pidfd, event in poller.poll(0):
                if event & select.POLLIN:
                    self._remove_lease(stale[pidfd])

    def handle(self, connection):
        lease = None
        delivered = False
        released = False
        try:
            connection.settimeout(30)
            peer_pid, peer_uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if peer_uid != os.getuid():
                raise ValueError("Engram control peer UID mismatch")
            message = _recv_json(connection)
            with self.lock:
                op = message.get("op")
                if op == "status":
                    _send_json(connection, self.status())
                    return
                if op == "drain":
                    if self.state in ("loading", "ready"):
                        self.state = "draining"
                    _send_json(connection, self.status())
                    return
                if op != "attach":
                    raise ValueError("Unknown Engram control operation")
                if self.state != "ready":
                    _send_json(connection, self.status() | {"ok": False})
                    return
                key = message.get("key")
                if (
                    message.get("version") != PROTOCOL_VERSION
                    or message.get("generation") != self.generation
                    or message.get("fingerprint") != self.layout["fingerprint"]
                    or key not in self.handles
                ):
                    raise ValueError("Engram generation, checkpoint or layout mismatch")
                lease = uuid.uuid4().hex
                self.leases[lease] = _Lease(_peer_pidfd(connection, peer_pid))
                handles = self.handles[key]
                _send_json(
                    connection,
                    {
                        "ok": True,
                        "lease": lease,
                        "generation": self.generation,
                        "parts": self.layout["tables"][key],
                        "inodes": [os.fstat(fd).st_ino for fd in handles],
                    },
                )
                if connection.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", handles))]) != 1:
                    raise OSError("Engram table descriptors were not sent")
                delivered = True
            connection.settimeout(None)
            message = _recv_json(connection)
            if message != {"op": "release", "lease": lease}:
                raise ValueError("Invalid Engram lease release")
            with self.lock:
                self._remove_lease(lease)
                released = True
            _send_json(connection, {"ok": True})
        except Exception as exc:
            with contextlib.suppress(OSError):
                _send_json(connection, {"ok": False, "error": str(exc)})
        finally:
            if lease is not None and not released:
                with self.lock:
                    if lease in self.leases:
                        if delivered:
                            # EOF alone does not prove asynchronous NPU reads are over.
                            self.leases[lease] = _Lease(self.leases[lease].pidfd, stale=True)
                        else:
                            self._remove_lease(lease)
            connection.close()

    def serve(self):
        endpoint = Path(self.endpoint)
        if not endpoint.parent.is_dir():
            raise ValueError("Create a private Engram endpoint directory first")
        with open(str(endpoint) + ".lock", "a") as owner_lock:
            fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if endpoint.exists():
                endpoint.unlink()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(endpoint))
                os.chmod(endpoint, 0o600)
                server.listen(64)
                server.settimeout(1)
                loader = threading.Thread(target=self.load, daemon=True)
                loader.start()
                try:
                    while True:
                        with self.lock:
                            self._reap_stale_leases()
                            if self.state == "failed" or (
                                self.state == "draining" and not self.leases and not loader.is_alive()
                            ):
                                break
                        try:
                            connection, _ = server.accept()
                        except TimeoutError:
                            continue
                        threading.Thread(target=self.handle, args=(connection,), daemon=True).start()
                finally:
                    loader.join()
                    endpoint.unlink(missing_ok=True)
                    with self.lock:
                        for handles in self.handles.values():
                            _close_fds(handles)
                        self.handles.clear()
                        for lease in list(self.leases):
                            self._remove_lease(lease)
                        self.state = "closed"
            if self.error:
                raise RuntimeError(self.error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("serve", "status", "wait", "drain"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--generation")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.operation == "serve":
        if not args.model or not args.generation:
            parser.error("serve requires --model and --generation")
        NodeTableOwner(args.endpoint, args.generation, args.model).serve()
        return
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            result = request(args.endpoint, {"op": "drain" if args.operation == "drain" else "status"})
            if args.generation and result["generation"] != args.generation:
                raise ValueError("Engram generation mismatch")
            if args.model and result["fingerprint"] != checkpoint_layout(args.model)["fingerprint"]:
                raise ValueError("Engram checkpoint mismatch")
            if args.operation != "wait" or result["state"] == "ready":
                print(json.dumps(result, indent=2), flush=True)
                if not result["ok"]:
                    raise RuntimeError(result["error"])
                return
            if result["state"] not in ("loading", "ready"):
                raise RuntimeError(f"Engram owner unavailable: {result}")
        except (TimeoutError, FileNotFoundError, ConnectionRefusedError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Engram owner")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
