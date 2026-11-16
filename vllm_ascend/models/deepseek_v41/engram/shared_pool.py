# SPDX-License-Identifier: Apache-2.0
"""Network discovery and control leases for a trusted cluster network."""

import hashlib
import json
import logging
import socket
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

from .shared_host import CHUNK_ROWS, GIB, MappedHostBuffer, MappingCleanupError, MemfabricRuntime

SCHEMA = 2
DTYPES = {"int8": torch.int8, "float32": torch.float32}
MAX_CONTROL_BYTES = 1 << 20
# Keep failed startup rollbacks alive until process exit; EOF is not a safe release.
_failed_startup_cleanups: list[Callable[[], None]] = []


def rollback_shared_engram(close: Callable[[], None]) -> bool:
    try:
        close()
    except BaseException:
        _failed_startup_cleanups.append(close)
        logging.getLogger(__name__).exception("Shared Engram startup cleanup failed; retaining mapping")
        return False
    return True


def _close_all(items):
    first_error = None
    for item in items:
        try:
            item.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
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


def parse_endpoint(endpoint):
    host, sep, port = endpoint.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("shared Engram endpoint must be host:port")
    return host, int(port)


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


def shard_range(rows, index, count):
    chunks = (rows + CHUNK_ROWS - 1) // CHUNK_ROWS
    if type(index) is not int or type(count) is not int or count < 1 or not 0 <= index < count or chunks < count:
        raise ValueError("invalid shared Engram shard layout")
    return chunks * index // count * CHUNK_ROWS, min(chunks * (index + 1) // count * CHUNK_ROWS, rows)


class SharedPoolOwner:
    def __init__(
        self,
        endpoint,
        identity,
        *,
        device,
        store_url=None,
        world_size=65,
        shard_index=0,
        shard_count=1,
        generation=None,
    ):
        if not 2 <= world_size <= 1024:
            raise ValueError("MemFabric world size must include the owner and consumers")
        if type(shard_count) is not int or type(shard_index) is not int or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shared Engram shard layout")
        if shard_count > 1 and (not store_url or not generation):
            raise ValueError("sharded owners require a common store URL and generation")
        if world_size <= shard_count:
            raise ValueError("MemFabric world size must include all owners and consumers")
        host, port = parse_endpoint(endpoint)
        self.endpoint = endpoint
        self.shard_index, self.shard_count = shard_index, shard_count
        self.generation = generation or uuid.uuid4().hex
        self.device = device
        self.runtime = None
        self.buffers = {}
        self.closed = False
        self._condition = threading.Condition()
        self._leases = set()
        self._listener = socket.socket()
        try:
            self._listener.bind((host, port))
            self._listener.listen(128)
            self._listener.settimeout(0.5)
            if store_url is None:
                with socket.socket() as reservation:
                    reservation.bind((host, 0))
                    store_url = f"tcp://{host}:{reservation.getsockname()[1]}"
            elif store_url == f"tcp://{endpoint}":
                raise ValueError("MemFabric store and owner control endpoints must differ")
        except BaseException:
            self._listener.close()
            raise
        self.manifest = dict(
            schema=SCHEMA,
            identity=identity,
            state="loading",
            tensors={},
            backend="memfabric_bm",
            memfabric=dict(store_url=store_url, world_size=world_size),
            endpoint=[host, self._listener.getsockname()[1]],
            generation=self.generation,
            shard=dict(index=shard_index, count=shard_count),
        )
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

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
                    with self._condition:
                        if request.get("op") == "ping":
                            _send(wire, dict(alive=not self.closed))
                            return
                        if (
                            request.get("op") != "attach"
                            or self.manifest["state"] != "ready"
                            or request.get("identity") != self.manifest["identity"]
                        ):
                            _send(wire, dict(error="Engram pool is not ready or model-compatible"))
                            return
                        self._leases.add(conn)
                        acquired = True
                        _send(wire, dict(manifest=self.manifest))
                    conn.settimeout(None)
                    if _receive(wire) != {"op": "release", "generation": self.generation}:
                        raise ValueError("Expected mapping-release acknowledgement")
                    with self._condition:
                        self._leases.remove(conn)
                        acquired = False
                        self._condition.notify_all()
                    _send(wire, dict(released=True))
        except (OSError, ValueError):
            if acquired:
                # EOF is not proof that a device mapping was safely released.
                # Retain this orphaned lease; operator recovery must first stop
                # all consumers before terminating the owner generation.
                print("ENGRAM_ORPHANED_LEASE owner memory retained", flush=True)

    def add(self, key, rows, width, dtype, *, total_rows=None, first_row=0, space_rows=None):
        if self.manifest["state"] != "loading" or key in self.buffers:
            raise ValueError("Cannot add to a published pool or duplicate a tensor")
        name = str(dtype).removeprefix("torch.")
        if name not in DTYPES:
            raise ValueError("Only INT8 codes and FP32 scales are supported")
        if self.runtime is None:
            settings = self.manifest["memfabric"]
            self.runtime = MemfabricRuntime.acquire(
                self.generation, settings["store_url"], settings["world_size"], self.device, owner=self.shard_index == 0
            )
        object_id = len(self.buffers)
        try:
            buffer = MappedHostBuffer(self.runtime, object_id, rows, width, dtype, owner=True, space_rows=space_rows)
        except MappingCleanupError as exc:
            self.buffers[key] = exc.buffer
            raise
        self.buffers[key] = buffer
        self.manifest["tensors"][key] = dict(
            rows=rows,
            width=width,
            dtype=name,
            object_id=object_id,
            owner_rank=self.runtime.rank,
            total_rows=total_rows if total_rows is not None else rows,
            first_row=first_row,
            space_rows=space_rows or rows,
        )
        return buffer

    def add_packed(self, specs):
        """Allocate one rank contribution containing all logical tensors."""
        if self.manifest["state"] != "loading" or self.buffers or self.shard_count < 2:
            raise ValueError("Packed Engram requires an empty sharded owner")
        metadata = {}
        rank_bytes = space_bytes = 0
        for key, total_rows, width, dtype in specs:
            if key in metadata or dtype not in (torch.int8, torch.float32) or width <= 0:
                raise ValueError("Invalid packed Engram tensor")
            first, last = shard_range(total_rows, self.shard_index, self.shard_count)
            chunks = (total_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
            max_rows = min(total_rows, ((chunks + self.shard_count - 1) // self.shard_count) * CHUNK_ROWS)
            row_bytes = width * torch.empty((), dtype=dtype).element_size()
            metadata[key] = dict(
                rows=last - first,
                width=width,
                dtype=str(dtype).removeprefix("torch."),
                object_id=0,
                total_rows=total_rows,
                first_row=first,
                byte_offset=rank_bytes,
            )
            rank_bytes += (last - first) * row_bytes
            space_bytes += max_rows * row_bytes
        if not metadata:
            raise ValueError("Packed Engram requires at least one tensor")
        settings = self.manifest["memfabric"]
        self.runtime = MemfabricRuntime.acquire(
            self.generation, settings["store_url"], settings["world_size"], self.device, owner=self.shard_index == 0
        )
        try:
            buffer = MappedHostBuffer(self.runtime, 0, rank_bytes, 1, torch.int8, owner=True, space_rows=space_bytes)
        except MappingCleanupError as exc:
            self.buffers["__packed__"] = exc.buffer
            raise
        self.buffers["__packed__"] = buffer
        for meta in metadata.values():
            meta.update(owner_rank=self.runtime.rank, rank_bytes=rank_bytes, space_bytes=space_bytes)
        self.manifest["packing"] = "byte-v1"
        self.manifest["tensors"] = metadata
        return buffer

    def ready(self):
        with self._condition:
            if self.manifest["state"] != "loading" or not self.buffers:
                raise ValueError("Only a populated loading pool can be published")
            self.manifest["state"] = "ready"

    def close(self, timeout=0):
        if self.closed:
            return
        with self._condition:
            self.manifest["state"] = "draining"
            deadline = time.monotonic() + timeout
            while self._leases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Shared Engram consumers have not released their mappings")
                self._condition.wait(remaining)
            _close_all(reversed(list(self.buffers.values())))
            if self.runtime is not None:
                self.runtime.close()
                self.runtime = None
            self.manifest["state"] = "closed"
            self.closed = True
        self._listener.close()
        self._thread.join(timeout=1)


class SharedPoolClient:
    def __init__(self, endpoint, identity, *, device):
        self.endpoint = endpoint
        self.buffers, self.views = {}, []
        self.closed = False
        self.device = device
        self.runtime = None
        self._conn = socket.create_connection(parse_endpoint(endpoint), timeout=30)
        self._wire = self._conn.makefile("rwb", buffering=0)
        try:
            _send(self._wire, dict(op="attach", identity=identity))
            response = _receive(self._wire)
            if "error" in response:
                raise ValueError(response["error"])
            self.manifest = response["manifest"]
            if (
                self.manifest.get("schema") != SCHEMA
                or self.manifest.get("identity") != identity
                or self.manifest.get("backend") != "memfabric_bm"
                or not self.manifest.get("generation")
            ):
                _send(self._wire, dict(op="release", generation=self.manifest.get("generation")))
                _receive(self._wire)
                raise ValueError("Incompatible shared Engram manifest")
        except BaseException:
            self._wire.close()
            self._conn.close()
            raise

    def view(self, key, start, end, *, rows, width, dtype):
        if self.closed:
            raise RuntimeError("Shared Engram client is closed")
        metadata = self.manifest["tensors"][key]
        if (metadata["rows"], metadata["width"], metadata["dtype"]) != (rows, width, str(dtype).removeprefix("torch.")):
            raise ValueError("Shared Engram tensor layout mismatch")
        if not isinstance(metadata["object_id"], int) or not 0 <= metadata["object_id"] < 64:
            raise ValueError("Invalid MemFabric Engram object id")
        if self.runtime is None:
            settings = self.manifest["memfabric"]
            self.runtime = MemfabricRuntime.acquire(
                self.manifest["generation"], settings["store_url"], settings["world_size"], self.device, owner=False
            )
        if key not in self.buffers:
            try:
                self.buffers[key] = MappedHostBuffer(
                    self.runtime,
                    metadata["object_id"],
                    rows,
                    width,
                    dtype,
                    owner=False,
                    owner_rank=metadata["owner_rank"],
                )
            except MappingCleanupError as exc:
                self.buffers[key] = exc.buffer
                raise
        view = self.buffers[key].view(start, end)
        self.views.append(view)
        return view

    def close(self):
        if self.closed:
            return
        first_error = None
        for items in (self.views, reversed(list(self.buffers.values()))):
            try:
                _close_all(items)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        _send(self._wire, dict(op="release", generation=self.manifest["generation"]))
        if _receive(self._wire) != {"released": True}:
            raise RuntimeError("Shared Engram release was not acknowledged")
        self._wire.close()
        self._conn.close()
        self.closed = True


class ShardedPoolSlice:
    """Chunk pointers indexed from the global chunk containing the TP slice."""

    def __init__(self, tensor, ptrs, row_offset, device):
        self.tensor, self.ptrs, self.row_offset, self.device = tensor, ptrs, row_offset, device

    def close(self):
        if self.ptrs is not None:
            with torch.npu.device(self.device):
                torch.npu.synchronize()
            self.ptrs = self.tensor = None


class ShardedPoolClient:
    def __init__(self, endpoints, identity, *, device):
        self.clients, self.views = [], []
        self.device = device
        self._packed_mapping = None
        try:
            for index, endpoint in enumerate(endpoints):
                client = SharedPoolClient(endpoint, identity, device=device)
                self.clients.append(client)
                manifest = client.manifest
                if manifest.get("shard") != {"index": index, "count": len(endpoints)}:
                    raise ValueError("shared Engram owner shard layout mismatch")
                if index and any(manifest[k] != self.clients[0].manifest[k] for k in ("generation", "memfabric")):
                    raise ValueError("shared Engram owners use different MemFabric groups")
            manifests = [client.manifest for client in self.clients]
            keys = list(manifests[0]["tensors"])
            if not keys or any(m.get("packing") != "byte-v1" or list(m["tensors"]) != keys for m in manifests):
                raise ValueError("shared Engram owners have different packed tensor layouts")
            ranks = set()
            for index, manifest in enumerate(manifests):
                owner_ranks = {meta["owner_rank"] for meta in manifest["tensors"].values()}
                if (
                    len(owner_ranks) != 1
                    or any(
                        type(rank) is not int or not 0 <= rank < manifest["memfabric"]["world_size"]
                        for rank in owner_ranks
                    )
                    or ranks.intersection(owner_ranks)
                ):
                    raise ValueError("shared Engram owners must have distinct ranks")
                ranks.update(owner_ranks)
                offset = capacity = 0
                for key in keys:
                    meta = manifest["tensors"][key]
                    ref = manifests[0]["tensors"][key]
                    if any(meta[k] != ref[k] for k in ("total_rows", "width", "dtype", "object_id", "space_bytes")):
                        raise ValueError("shared Engram tensor layout mismatch")
                    if (
                        meta["object_id"] != 0
                        or meta["dtype"] not in DTYPES
                        or type(meta["width"]) is not int
                        or meta["width"] <= 0
                    ):
                        raise ValueError("invalid packed Engram tensor metadata")
                    first, last = shard_range(meta["total_rows"], index, len(endpoints))
                    if (meta["first_row"], meta["rows"], meta["byte_offset"]) != (first, last - first, offset):
                        raise ValueError("shared Engram row range or packed offset mismatch")
                    row_bytes = meta["width"] * torch.empty((), dtype=DTYPES[meta["dtype"]]).element_size()
                    chunks = (meta["total_rows"] + CHUNK_ROWS - 1) // CHUNK_ROWS
                    max_rows = min(meta["total_rows"], ((chunks + len(endpoints) - 1) // len(endpoints)) * CHUNK_ROWS)
                    offset += (last - first) * row_bytes
                    capacity += max_rows * row_bytes
                if any(
                    meta["rank_bytes"] != offset or meta["space_bytes"] != capacity
                    for meta in manifest["tensors"].values()
                ):
                    raise ValueError("shared Engram packed capacity mismatch")
            self._packed_spec = [(m["generation"], m["memfabric"], m["tensors"]) for m in manifests]
        except BaseException:
            rollback_shared_engram(self.close)
            raise

    def view(self, key, start, end, *, rows, width, dtype):
        meta = self.clients[0].manifest["tensors"][key]
        if (meta["total_rows"], meta["width"], meta["dtype"]) != (rows, width, str(dtype).removeprefix("torch.")):
            raise ValueError("shared Engram tensor layout mismatch")
        if not 0 <= start < end <= rows:
            raise ValueError("invalid shared Engram slice")
        pointers = []
        aligned_start = start // CHUNK_ROWS * CHUNK_ROWS
        primary = self.clients[0]
        if primary.runtime is None:
            settings = primary.manifest["memfabric"]
            primary.runtime = MemfabricRuntime.acquire(
                primary.manifest["generation"], settings["store_url"], settings["world_size"], self.device, owner=False
            )
        if self._packed_mapping is None:
            # One process-wide BM object is shared by the layer clients.
            with MemfabricRuntime._lock:
                mapping = getattr(primary.runtime, "_packed_mapping", None)
                if mapping is None:
                    try:
                        mapping = MappedHostBuffer(
                            primary.runtime,
                            0,
                            meta["rank_bytes"],
                            1,
                            torch.int8,
                            owner=False,
                            owner_rank=meta["owner_rank"],
                            space_rows=meta["space_bytes"],
                        )
                    except MappingCleanupError as exc:
                        primary.buffers["__packed_cleanup__"] = exc.buffer
                        raise
                    primary.runtime._packed_mapping = mapping
                    primary.runtime._packed_spec = self._packed_spec
                    primary.runtime._packed_refs = 0
                elif (
                    primary.runtime._packed_spec != self._packed_spec
                    or mapping.rows != meta["rank_bytes"]
                    or mapping.space_size != ((meta["space_bytes"] + GIB - 1) // GIB * GIB)
                ):
                    raise ValueError("shared Engram packed mapping layout mismatch")
                primary.runtime._packed_refs += 1
                self._packed_mapping = mapping
        mapping = self._packed_mapping
        for client in self.clients:
            shard = client.manifest["tensors"][key]
            first, last = shard["first_row"], shard["first_row"] + shard["rows"]
            chunk_start, chunk_end = max(aligned_start, first), min(end, last)
            for row in range(chunk_start, chunk_end, CHUNK_ROWS):
                row_bytes = width * torch.empty((), dtype=dtype).element_size()
                offset = shard["byte_offset"] + (row - first) * row_bytes
                size = min(CHUNK_ROWS, chunk_end - row) * row_bytes
                pointers.append(mapping.device_pointer(offset, size, owner_rank=shard["owner_rank"]))
        tensor = torch.empty((end - start, width), dtype=dtype, device="meta")
        view = ShardedPoolSlice(
            tensor, torch.tensor(pointers, dtype=torch.int64, device=self.device), start - aligned_start, self.device
        )
        self.views.append(view)
        return view

    def close(self):
        _close_all(self.views)
        # Drop this layer's mapping reference before releasing any owner lease.
        if self._packed_mapping is not None:
            runtime = self.clients[0].runtime
            with MemfabricRuntime._lock:
                if runtime._packed_refs == 1:
                    self._packed_mapping.close()
                    runtime._packed_mapping = None
                    runtime._packed_spec = None
                runtime._packed_refs -= 1
                self._packed_mapping = None
        if self.clients:
            self.clients[0].close()
            _close_all(self.clients[1:])
