# SPDX-License-Identifier: Apache-2.0
"""Immutable Engram owner and bounded HIXL row reader for A2 RoCE.

Control messages carry IDs; registered staging buffers carry packed rows.
This protocol is for a trusted serving network, not an exposed storage API.
"""

import ctypes
import hashlib
import json
import socket
import socketserver
import struct
import threading
import uuid
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MAX_ROWS = 65536
MAX_CONTROL_BYTES = 65536


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
    for filename in ("config.json", "quant_model_weights.safetensors.index.json"):
        digest.update(filename.encode())
        digest.update((root / filename).read_bytes())
    config = json.loads((root / "config.json").read_text())
    config = config.get("text_config", config)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    if config["engram_head_dim"] != 256:
        raise ValueError("remote Engram requires width256 INT8 checkpoint rows")
    tables = {}
    for layer, rows in zip(config["engram_layer_ids"], config["engram_num_embeddings"], strict=True):
        sample = hashlib.sha256()
        scale_dtype = None
        for suffix, width in (("weight", 256), ("scale", 8)):
            key = f"layers.{layer}.engram.embed.{suffix}"
            with safe_open(root / index[key], framework="pt", device="cpu") as file:
                tensor = file.get_slice(key)
                dtype = tensor.get_dtype()
                if tensor.get_shape() != [rows, width]:
                    raise ValueError(f"unsupported checkpoint shape: {key}")
                if suffix == "weight" and dtype not in ("I8", "INT8"):
                    raise ValueError("remote Engram currently requires INT8 codes")
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
            samples=sample.hexdigest(),
        )
    return dict(identity=digest.hexdigest(), tables=tables)


class HixlEngine:
    """Use the installed C++ API without depending on its Python ABI."""

    def __init__(self, endpoint, device, library_path=None):
        self.device = device
        path = library_path or Path(__file__).resolve().parents[3] / "libengram_hixl.so"
        self.lib = ctypes.CDLL(str(path))
        ptr, size, char = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p
        definitions = {
            "engram_hixl_create": [char, ctypes.c_int, ctypes.POINTER(ptr)],
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
        self.check(self.lib.engram_hixl_create(endpoint.encode(), device, ctypes.byref(self.handle)))

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


class RemoteReader:
    """One rank's reusable connection and finite HBM receive buffer."""

    def __init__(self, model, endpoint, *, capacity, library_path=None):
        if not 0 < capacity <= MAX_ROWS:
            raise ValueError("invalid remote Engram row capacity")
        self.closed = False
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
                dict(op="attach", metadata=metadata, capacity=capacity, device=torch.npu.current_device()),
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
            self.engine = HixlEngine(f"{address}:{port}", torch.npu.current_device(), library_path)
            self.registration = self.engine.register(self.buffer)
            self.engine.check(self.engine.lib.engram_hixl_connect(self.engine.handle, self.peer))
        except BaseException:
            # Without a completed attach/drain, owner staging is retained conservatively.
            if self.engine is not None:
                self.engine.close()
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
            raise ValueError("Engram batch exceeds configured remote capacity")
        if self._last_use is not None:
            self._last_use.synchronize()
        _send(self.wire, dict(request, generation=self.generation))
        self.wire.sendall(ids.astype("<i8", copy=False).tobytes())
        ready = _receive(self.wire)
        if ready != {"ready": len(ids), "generation": self.generation}:
            raise RuntimeError(f"remote Engram lookup failed: {ready}")
        if len(ids):
            self.engine.check(
                self.engine.lib.engram_hixl_read(
                    self.engine.handle,
                    self.peer,
                    self.buffer.data_ptr(),
                    self.remote_address,
                    byte_count,
                )
            )

    def lookup(self, table, indices, out, *, cpu_ids=None):
        layer, spec, ids = self._lookup_ids(table, indices, cpu_ids)
        self._transfer_rows(dict(op="lookup", layer=layer, count=len(ids)), ids, len(ids) * spec["row_bytes"])
        if len(ids):
            from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

            dequantize_packed_engram(self.buffer, out, len(ids), spec["row_bytes"], spec["scale_dtype"])
        self._last_use = torch.npu.Event()
        self._last_use.record(torch.npu.current_stream())

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
        for table, (_, spec, rows) in zip(tables, entries):
            output = torch.empty(
                (gathered.shape[0], table.part_n_hash_cols, table.dim),
                dtype=torch.bfloat16,
                device=gathered.device,
            )
            size = len(rows) * spec["row_bytes"]
            if len(rows):
                dequantize_packed_engram(
                    self.buffer[offset : offset + size], output, len(rows), spec["row_bytes"], spec["scale_dtype"]
                )
            offset += size
            outputs.append(output)
        self._last_use = torch.npu.Event()
        self._last_use.record(torch.npu.current_stream())
        return outputs

    def close(self):
        if self.closed:
            return
        # Only dequantization consumes this registered buffer. Later graph and
        # collective work uses separate BF16 results and need not be drained.
        if self._last_use is not None:
            self._last_use.synchronize()
        self.engine.close()
        _send(self.wire, dict(op="close", generation=self.generation))
        if _receive(self.wire) != {"closed": True}:
            raise RuntimeError("owner did not retire remote Engram lease")
        self.wire.close()
        self.buffer = None
        self.closed = True


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

    def __init__(self, model, endpoint, engine_endpoint, *, device=0, library_path=None):
        self.metadata = checkpoint_metadata(model)
        self.generation = uuid.uuid4().hex
        self.device = device
        self.engine_endpoint = engine_endpoint
        self.library_path = library_path
        self.engine = HixlEngine(engine_endpoint, device, library_path)
        self.alternate_engine = None
        self.alternate_endpoint = None
        self.tables = {}
        self.sessions = {}
        self.retained = []
        self.lock = threading.Lock()
        root = Path(model)
        index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
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
        super().__init__((host, int(port)), _OwnerConnection)

    def engine_for_reader(self, address, device):
        # A2 Device RoCE cannot connect a NIC to itself. Reuse the same DRAM
        # through another owner device only for the colocated same-device rank.
        if address != self.server_address[0] or device != self.device:
            return self.engine, self.engine_endpoint
        with self.lock:
            if self.alternate_engine is None:
                devices = [d for d in range(torch.npu.device_count()) if d != self.device]
                if not devices:
                    raise RuntimeError("Colocated Engram requires another visible owner NPU for RoCE")
                host, port = self.engine_endpoint.rsplit(":", 1)
                self.alternate_endpoint = f"{host}:{int(port) + 1}"
                self.alternate_engine = HixlEngine(self.alternate_endpoint, devices[0], self.library_path)
        return self.alternate_engine, self.alternate_endpoint

    def server_close(self):
        with self.lock:
            if self.sessions or self.retained:
                raise RuntimeError("stop consumers and drain retained Engram leases before owner teardown")
        super().server_close()
        if self.alternate_engine is not None:
            self.alternate_engine.close()
        self.engine.close()


class _OwnerConnection(socketserver.BaseRequestHandler):
    def handle(self):
        owner = self.server
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
            engine, endpoint = owner.engine_for_reader(self.client_address[0], attach.get("device"))
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
                    owner.retained.append((stage, registration))
            wire.close()
