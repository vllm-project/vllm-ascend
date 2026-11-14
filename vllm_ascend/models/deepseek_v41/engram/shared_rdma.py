# SPDX-License-Identifier: Apache-2.0
"""Immutable Engram owner and bounded HIXL row reader.

Control messages carry IDs; registered staging buffers carry packed rows.
This protocol is for a trusted serving network, not an exposed storage API.
"""

import ctypes
import functools
import hashlib
import json
import logging
import os
import socket
import socketserver
import struct
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import safe_open

MIN_ROWS = 65536
MAX_ROWS = 262144
MAX_CONTROL_BYTES = 65536

# A failed transfer may still access its registered receive buffer.
_held_readers = set()
_held_readers_lock = threading.Lock()
# Failed owner rollback must retain its engine and table references.
_held_owners = set()
_held_owners_lock = threading.Lock()


class EngramCapacityError(ValueError):
    """A local lookup exceeds capacity before any transport operation."""


def capacity_for_batch(configured, max_tokens, layer_head_counts, tp_size):
    """Size for one layer; combined lookups may fall back to per-layer reads."""
    if tp_size <= 0 or any(heads % tp_size for heads in layer_head_counts):
        raise ValueError("shared Engram heads must be divisible by a positive TP size")
    required = max((max_tokens * (heads // tp_size) for heads in layer_head_counts), default=0)
    capacity = max(MIN_ROWS, configured, required)
    if capacity > MAX_ROWS:
        raise ValueError(f"shared Engram lookup needs {capacity} rows, exceeding the supported limit {MAX_ROWS}")
    return capacity


def _hold_reader(reader):
    reader.failed = True
    with _held_readers_lock:
        _held_readers.add(reader)


def _reader_operation(function):
    @functools.wraps(function)
    def run(self, *args, **kwargs):
        with self._operation_lock:
            if self.failed or self.closed:
                raise RuntimeError("remote Engram reader is failed or closed")
            try:
                return function(self, *args, **kwargs)
            except EngramCapacityError:
                raise
            except BaseException:
                _hold_reader(self)
                raise

    return run


def _receive_exact(wire, size):
    result = bytearray(size)
    view = memoryview(result)
    while view:
        count = wire.recv_into(view)
        if not count:
            raise ConnectionError("shared Engram control connection closed")
        view = view[count:]
    return result


def _send(wire, value):
    data = json.dumps(value).encode()
    if len(data) > MAX_CONTROL_BYTES:
        raise ValueError("oversized Engram control message")
    wire.sendall(struct.pack("!I", len(data)) + data)


def _receive(wire):
    size = struct.unpack("!I", _receive_exact(wire, 4))[0]
    if not 0 < size <= MAX_CONTROL_BYTES:
        raise ValueError("invalid Engram control message size")
    value = json.loads(_receive_exact(wire, size))
    if not isinstance(value, dict):
        raise ValueError("expected Engram control object")
    return value


def _configure_socket(wire):
    wire.settimeout(120)
    wire.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def checkpoint_metadata(model):
    """Identify the immutable checkpoint and sample raw rows before allocation."""
    root = Path(model)
    digest = hashlib.sha256()
    index_path = next(
        (
            root / name
            for name in ("quant_model_weights.safetensors.index.json", "model.safetensors.index.json")
            if (root / name).is_file()
        ),
        None,
    )
    if index_path is None:
        raise ValueError("remote Engram requires a checkpoint index")
    for filename in ("config.json", index_path.name):
        digest.update(filename.encode())
        digest.update((root / filename).read_bytes())
    config = json.loads((root / "config.json").read_text())
    config = config.get("text_config", config)
    index = json.loads(index_path.read_text())["weight_map"]
    if config["engram_head_dim"] != 256:
        raise ValueError("remote Engram requires width256 INT8 checkpoint rows")
    tables = {}
    for layer, rows in zip(config["engram_layer_ids"], config["engram_num_embeddings"], strict=True):
        sample = hashlib.sha256()
        scale_dtype = code_dtype = None
        for suffix, width in (("weight", 256), ("scale", 8)):
            key = f"layers.{layer}.engram.embed.{suffix}"
            with safe_open(root / index[key], framework="pt", device="cpu") as file:
                tensor = file.get_slice(key)
                dtype = tensor.get_dtype()
                if tensor.get_shape() != [rows, width]:
                    raise ValueError(f"unsupported checkpoint shape: {key}")
                if suffix == "weight":
                    if dtype not in ("I8", "INT8", "F8_E4M3"):
                        raise ValueError("remote Engram requires INT8 or FP8 E4M3 codes")
                    code_dtype = dtype
                if suffix == "scale":
                    if dtype not in ("F8_E8M0", "F32"):
                        raise ValueError("remote Engram requires E8M0 or FP32 scales")
                    scale_dtype = dtype
                for row in (0, 1, rows // 2, rows - 1):
                    # Model construction sets a default NPU device; safetensors
                    # slicing must stay on CPU before selecting these few rows.
                    with torch.device("cpu"):
                        sample.update(tensor[row : row + 1].contiguous().view(torch.uint8).numpy().tobytes())
        tables[str(layer)] = dict(
            rows=rows,
            row_bytes=264 if scale_dtype == "F8_E8M0" else 288,
            scale_dtype=scale_dtype,
            code_dtype=code_dtype,
            samples=sample.hexdigest(),
        )
    return dict(identity=digest.hexdigest(), tables=tables, index_file=index_path.name, protocol=2)


class HixlEngine:
    """Use the installed C++ API without depending on its Python ABI."""

    def __init__(self, endpoint, device, library_path=None, *, listen_port=None, local_comm_res=None):
        self.device = device
        path = library_path or Path(__file__).resolve().parents[3] / "libengram_hixl.so"
        self.lib = ctypes.CDLL(str(path))
        ptr, size, char = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p
        definitions = {
            "engram_hixl_create_v2": [char, ctypes.c_int, char, char, ctypes.POINTER(ptr)],
            "engram_hixl_register": [ptr, size, size, ctypes.c_int, ctypes.POINTER(ptr)],
            "engram_hixl_deregister": [ptr, ptr],
            "engram_hixl_connect": [ptr, char],
            "engram_hixl_read": [ptr, char, size, size, size],
            "engram_hixl_close": [ptr],
            "engram_pack_rows": [size, size, size, ptr, size, size],
        }
        for name, args in definitions.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = ctypes.c_int
        self.lib.engram_hixl_error.restype = char
        self.handle = ptr()
        self.check(
            self.lib.engram_hixl_create_v2(
                endpoint.encode(),
                device,
                str(listen_port).encode() if listen_port is not None else None,
                os.fsencode(local_comm_res) if local_comm_res else None,
                ctypes.byref(self.handle),
            )
        )

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.engram_hixl_error().decode())

    def register(self, tensor):
        handle = ctypes.c_void_p()
        self.check(
            self.lib.engram_hixl_register(
                self.handle,
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                int(tensor.device.type == "cpu"),
                ctypes.byref(handle),
            )
        )
        return handle

    def deregister(self, handle):
        self.check(self.lib.engram_hixl_deregister(self.handle, handle))

    def close(self):
        if self.handle:
            self.check(self.lib.engram_hixl_close(self.handle))
            self.handle = ctypes.c_void_p()


def _physical_device_id(device):
    from vllm_ascend.platform import NPUPlatform

    return NPUPlatform.visible_device_id_to_physical_device_id(device)


class RemoteReader:
    """One rank's reusable connection and finite HBM receive buffer."""

    def __init__(self, model, endpoint, *, capacity, library_path=None, listen_port=None, local_comm_res=None):
        if not 0 < capacity <= MAX_ROWS:
            raise ValueError("invalid remote Engram row capacity")
        self.closed = False
        self.failed = False
        self._operation_lock = threading.Lock()
        self._last_use = None
        self.engine = None
        self.buffer = None
        metadata = checkpoint_metadata(model)
        host, port = endpoint.rsplit(":", 1)
        self.wire = socket.create_connection((host, int(port)), 120)
        _configure_socket(self.wire)
        try:
            _send(
                self.wire,
                dict(
                    op="attach",
                    metadata=metadata,
                    capacity=capacity,
                    physical_device=_physical_device_id(torch.npu.current_device()),
                ),
            )
            ready = _receive(self.wire)
            if "error" in ready:
                raise RuntimeError(ready["error"])
            if ready.get("metadata") != metadata or not ready.get("generation"):
                raise ValueError("remote Engram checkpoint or layout mismatch")
            self.metadata = metadata
            self.supports_batch = ready.get("batch_lookup", False)
            self.capacity = capacity
            self.generation = ready["generation"]
            self.remote_address = ready["address"]
            self.peer = ready["engine"].encode()
            row_bytes = max(t["row_bytes"] for t in metadata["tables"].values())
            self.buffer = torch.empty((capacity * row_bytes,), dtype=torch.uint8, device="npu")
            # Every TP/DP worker needs a unique HIXL engine identity, even on one Pod IP.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind((self.wire.getsockname()[0], 0))
                address, port = reservation.getsockname()
            self.engine = HixlEngine(
                f"{address}:{port}",
                torch.npu.current_device(),
                library_path,
                listen_port=listen_port,
                local_comm_res=local_comm_res,
            )
            self.registration = self.engine.register(self.buffer)
            self.engine.check(self.engine.lib.engram_hixl_connect(self.engine.handle, self.peer))
        except BaseException:
            # Keep registered local memory alive if initialization did not complete.
            if self.engine is not None:
                _hold_reader(self)
            else:
                self.wire.close()
            raise

    @staticmethod
    def copy_hashes(indices, head_start, local_heads):
        """Copy both layers before either layer enqueues its TP exchange."""
        return (
            indices[:, :, head_start : head_start + local_heads]
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
            .numpy()
        )

    def _lookup_ids(self, table, indices, cpu_ids):
        if self.closed:
            raise RuntimeError("lookup on closed remote Engram reader")
        layer = str(table._checkpoint_key.split(".")[1])
        spec = self.metadata["tables"][layer]
        if table.num_embeddings != spec["rows"]:
            raise ValueError("remote Engram row count differs from model")
        if cpu_ids is None:
            selected = indices[:, table.head_start : table.head_start + table.part_n_hash_cols]
            cpu_ids = selected.to(device="cpu", dtype=torch.int64).contiguous().numpy()
        if cpu_ids.shape != (indices.shape[0], table.part_n_hash_cols):
            raise ValueError("remote Engram CPU IDs differ from the local head layout")
        owned = (cpu_ids >= table.vocab_start_idx) & (cpu_ids < table.vocab_end_idx)
        ids = np.where(owned, cpu_ids, -1).astype(np.int64, copy=False).reshape(-1)
        return layer, spec, ids

    def _transfer_rows(self, request, ids, byte_count):
        if len(ids) > self.capacity:
            raise EngramCapacityError("Engram batch exceeds configured remote capacity")
        if self._last_use is not None:
            self._last_use.synchronize()
        _send(self.wire, dict(request, generation=self.generation))
        self.wire.sendall(ids.astype("<i8", copy=False).tobytes())
        ready = _receive(self.wire)
        if ready != {"ready": len(ids), "generation": self.generation}:
            raise RuntimeError(f"remote Engram lookup failed: {ready}")
        if len(ids):
            assert self.engine is not None and self.buffer is not None
            self.engine.check(
                self.engine.lib.engram_hixl_read(
                    self.engine.handle,
                    self.peer,
                    self.buffer.data_ptr(),
                    self.remote_address,
                    byte_count,
                )
            )

    @_reader_operation
    def lookup(self, table, indices, out, *, cpu_ids=None):
        layer, spec, ids = self._lookup_ids(table, indices, cpu_ids)
        self._transfer_rows(dict(op="lookup", layer=layer, count=len(ids)), ids, len(ids) * spec["row_bytes"])
        if len(ids):
            from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

            dequantize_packed_engram(
                self.buffer, out, len(ids), spec["row_bytes"], spec["scale_dtype"], spec.get("code_dtype", "I8")
            )
        self._last_use = torch.npu.Event()
        self._last_use.record(torch.npu.current_stream())

    @_reader_operation
    def lookup_layers(self, tables, gathered, cpu_hashes):
        """Use one RPC/read when all layers fit the existing receive buffer."""
        if not self.supports_batch or len(tables) < 2:
            return None
        if gathered.shape[0] * sum(table.part_n_hash_cols for table in tables) > self.capacity:
            return None
        entries = [self._lookup_ids(table, gathered[:, slot], cpu_hashes[:, slot]) for slot, table in enumerate(tables)]
        ids = np.concatenate([entry[2] for entry in entries])
        byte_count = sum(len(rows) * spec["row_bytes"] for _, spec, rows in entries)
        request = dict(op="lookup_batch", lookups=[dict(layer=layer, count=len(rows)) for layer, _, rows in entries])
        self._transfer_rows(request, ids, byte_count)
        from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

        outputs = []
        offset = 0
        assert self.buffer is not None
        for table, (_, spec, rows) in zip(tables, entries):
            output = torch.empty(
                (gathered.shape[0], table.part_n_hash_cols, table.dim),
                dtype=torch.bfloat16,
                device=gathered.device,
            )
            size = len(rows) * spec["row_bytes"]
            if len(rows):
                dequantize_packed_engram(
                    self.buffer[offset : offset + size],
                    output,
                    len(rows),
                    spec["row_bytes"],
                    spec["scale_dtype"],
                    spec.get("code_dtype", "I8"),
                )
            offset += size
            outputs.append(output)
        self._last_use = torch.npu.Event()
        self._last_use.record(torch.npu.current_stream())
        return outputs

    def close(self):
        with self._operation_lock:
            if self.closed:
                return
            if self.failed:
                raise RuntimeError("failed remote Engram reader requires external recovery")
            try:
                # Only dequantization consumes this registered buffer.
                if self._last_use is not None:
                    self._last_use.synchronize()
                assert self.engine is not None
                self.engine.close()
                _send(self.wire, dict(op="close", generation=self.generation))
                if _receive(self.wire) != {"closed": True}:
                    raise RuntimeError("owner did not retire remote Engram lease")
                self.wire.close()
                self.buffer = None
                self.closed = True
            except BaseException:
                _hold_reader(self)
                raise


def _lookup_plan(request, tables, capacity):
    operation = request.get("op")
    if operation == "lookup":
        entries = [request]
    elif operation == "lookup_batch":
        entries = request.get("lookups")
    else:
        raise ValueError("unknown Engram operation")
    if not isinstance(entries, list) or not 0 < len(entries) <= len(tables):
        raise ValueError("invalid Engram layer batch")
    plan = []
    count = offset = 0
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid Engram lookup entry")
        layer, rows = entry.get("layer"), entry.get("count", -1)
        if (
            not isinstance(layer, str)
            or layer not in tables
            or layer in seen
            or type(rows) is not int
            or not 0 <= rows <= capacity
        ):
            raise ValueError("invalid Engram lookup shape")
        seen.add(layer)
        plan.append((layer, rows, count, offset))
        count += rows
        offset += rows * tables[layer].shape[1]
    if count > capacity:
        raise ValueError("Engram layer batch exceeds remote capacity")
    return plan, count


def _pack_lookup_rows(engine, stage, tables, plan, ids):
    # Validate every layer before writing any leased buffer bytes.
    if ids.ndim != 1 or len(ids) != sum(count for _, count, _, _ in plan):
        raise ValueError("invalid Engram ID payload size")
    for layer, count, start, _ in plan:
        selected = ids[start : start + count]
        if np.any((selected < -1) | (selected >= len(tables[layer]))):
            raise ValueError("invalid Engram row ID")
    for layer, count, start, offset in plan:
        table = tables[layer]
        selected = ids[start : start + count]
        engine.check(
            engine.lib.engram_pack_rows(
                stage.data_ptr() + offset,
                table.data_ptr(),
                table.shape[1],
                selected.ctypes.data,
                count,
                len(table),
            )
        )


class SharedOwner(socketserver.ThreadingTCPServer):
    """Immutable DRAM table, with one leased host staging range per connection."""

    allow_reuse_address = True

    def __init__(
        self,
        model,
        endpoint,
        engine_endpoint,
        *,
        device=0,
        library_path=None,
        listen_port=None,
        local_comm_res=None,
        fabric="roce",
    ):
        self.metadata = checkpoint_metadata(model)
        self.generation = uuid.uuid4().hex
        self.device = device
        self.physical_device = _physical_device_id(device)
        self.engine_endpoint = engine_endpoint
        self.library_path = library_path
        self.listen_port = listen_port
        self.local_comm_res = local_comm_res
        self.fabric = fabric
        self.engine = HixlEngine(
            engine_endpoint, device, library_path, listen_port=listen_port, local_comm_res=local_comm_res
        )
        self.alternate_engine = None
        self.alternate_endpoint = None
        self.tables = {}
        self.sessions = {}
        self.retained = []
        self.lock = threading.Lock()
        self.pending_handlers = 0
        try:
            root = Path(model)
            index = json.loads((root / self.metadata["index_file"]).read_text())["weight_map"]
            for layer, spec in self.metadata["tables"].items():
                packed = torch.empty((spec["rows"], spec["row_bytes"]), dtype=torch.uint8)
                for suffix, start, end in (("weight", 0, 256), ("scale", 256, spec["row_bytes"])):
                    key = f"layers.{layer}.engram.embed.{suffix}"
                    with safe_open(root / index[key], framework="pt", device="cpu") as file:
                        tensor = file.get_slice(key)
                        for row in range(0, spec["rows"], 1 << 20):
                            stop = min(row + (1 << 20), spec["rows"])
                            packed[row:stop, start:end].copy_(tensor[row:stop].contiguous().view(torch.uint8))
                self.tables[layer] = packed
                print(json.dumps(dict(stage="TABLE_LOADED", layer=layer, bytes=packed.numel())), flush=True)
            host, port = endpoint.rsplit(":", 1)
            super().__init__((host, int(port)), _OwnerConnection, bind_and_activate=False)
            self.server_bind()
            self.server_activate()
        except BaseException:
            # No request handlers exist until construction returns and serve_forever starts.
            sock = getattr(self, "socket", None)
            if sock is not None:
                with suppress(OSError):
                    sock.close()
            try:
                self.engine.close()
            except BaseException:
                with _held_owners_lock:
                    _held_owners.add(self)
                logging.getLogger(__name__).exception(
                    "Failed to close shared Engram owner after initialization failure"
                )
            raise

    def engine_for_reader(self, address, physical_device):
        # A2 Device RoCE cannot connect a NIC to itself. Reuse the same DRAM
        # through another owner device only for the colocated same-device rank.
        if self.fabric == "ub":
            return self.engine, self.engine_endpoint
        if address != self.server_address[0] or physical_device != self.physical_device:
            return self.engine, self.engine_endpoint
        with self.lock:
            if self.alternate_engine is None:
                devices = [d for d in range(torch.npu.device_count()) if d != self.device]
                if not devices:
                    raise RuntimeError("Colocated Engram requires another visible owner NPU for RoCE")
                host, port = self.engine_endpoint.rsplit(":", 1)
                self.alternate_endpoint = f"{host}:{int(port) + 1}"
                self.alternate_engine = HixlEngine(
                    self.alternate_endpoint,
                    devices[0],
                    self.library_path,
                    listen_port=self.listen_port,
                    local_comm_res=self.local_comm_res,
                )
        return self.alternate_engine, self.alternate_endpoint

    def process_request(self, request, client_address):
        with self.lock:
            self.pending_handlers += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.lock:
                self.pending_handlers -= 1
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.lock:
                self.pending_handlers -= 1

    def server_close(self):
        with self.lock:
            if self.pending_handlers or self.sessions or self.retained:
                raise RuntimeError("stop consumers and drain retained Engram leases before owner teardown")
        super().server_close()
        with self.lock:
            if self.pending_handlers or self.sessions or self.retained:
                raise RuntimeError("stop consumers and drain retained Engram leases before owner teardown")
        if self.alternate_engine is not None:
            self.alternate_engine.close()
        self.engine.close()


class _OwnerConnection(socketserver.BaseRequestHandler):
    def handle(self):
        owner = cast(SharedOwner, self.server)
        wire = self.request
        _configure_socket(wire)
        stage = registration = None
        retired = False
        session = uuid.uuid4().hex
        try:
            attach = _receive(wire)
            capacity = attach.get("capacity", 0)
            if attach.get("op") != "attach" or attach.get("metadata") != owner.metadata or not 0 < capacity <= MAX_ROWS:
                raise ValueError("incompatible remote Engram attach")
            physical_device = attach.get("physical_device")
            if type(physical_device) is not int or physical_device < 0:
                raise ValueError("remote Engram attach requires a physical device ID")
            engine, endpoint = owner.engine_for_reader(self.client_address[0], physical_device)
            torch.npu.set_device(engine.device)
            row_bytes = max(t["row_bytes"] for t in owner.metadata["tables"].values())
            stage = torch.empty(capacity * row_bytes, dtype=torch.uint8, pin_memory=True)
            registration = engine.register(stage)
            with owner.lock:
                owner.sessions[session] = stage
            # A valid lease may stay idle throughout model loading/graph capture.
            wire.settimeout(None)
            _send(
                wire,
                dict(
                    metadata=owner.metadata,
                    generation=owner.generation,
                    address=stage.data_ptr(),
                    engine=endpoint,
                    batch_lookup=True,
                ),
            )
            while True:
                request = _receive(wire)
                if request.get("generation") != owner.generation:
                    raise ValueError("stale Engram generation")
                if request.get("op") == "close":
                    engine.deregister(registration)
                    retired = True
                    _send(wire, {"closed": True})
                    break
                plan, count = _lookup_plan(request, owner.tables, capacity)
                ids = np.frombuffer(_receive_exact(wire, count * 8), dtype="<i8")
                _pack_lookup_rows(engine, stage, owner.tables, plan, ids)
                _send(wire, dict(ready=count, generation=owner.generation))
        except Exception as error:
            with suppress(OSError):
                _send(wire, {"error": str(error)})
        finally:
            with owner.lock:
                owner.sessions.pop(session, None)
                if registration is not None and not retired:
                    # EOF is not proof that a remote RDMA request has retired.
                    owner.retained.append((engine, stage, registration))
            wire.close()
