# SPDX-License-Identifier: Apache-2.0
"""Qwen BF16 PLE tables over the shared Engram HIXL protocol."""

import socket
import socketserver
import threading
import uuid

import numpy as np
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.shared_rdma import (
    MAX_ROWS,
    HixlEngine,
    RemoteReader,
    SharedOwner,
    _configure_socket,
    _hold_reader,
    _OwnerConnection,
    _reader_operation,
    _receive,
    _send,
)

from .ple_checkpoint import PLECheckpoint, checkpoint_identity


def ple_metadata(model, prefix: str, parts: int, width: int):
    checkpoint = PLECheckpoint(model, prefix, parts, width)
    metadata = {
        "identity": checkpoint_identity(model, prefix),
        "tables": {"ple": {"rows": checkpoint.rows, "row_bytes": width * 2, "dtype": "BF16"}},
    }
    return checkpoint, metadata


class PLERemoteOwner(SharedOwner):
    """Own one complete table; retain the existing HIXL lease protocol."""

    def __init__(self, model, prefix, parts, width, endpoint, engine_endpoint, *, device=0, library_path=None):
        checkpoint, self.metadata = ple_metadata(model, prefix, parts, width)
        self.generation = uuid.uuid4().hex
        self.device, self.engine_endpoint, self.library_path = device, engine_endpoint, library_path
        self.shard = {"index": 0, "count": 1}
        self.alternate_engine = self.alternate_endpoint = None
        self.sessions, self.retained = {}, []
        self.lock = threading.Lock()
        table = torch.empty((checkpoint.rows, width), dtype=torch.bfloat16, device="cpu")
        for start in range(0, checkpoint.rows, 262144):
            checkpoint.copy_rows(table[start : min(start + 262144, checkpoint.rows)], start)
        self.tables = {"ple": table.view(torch.uint8).reshape(checkpoint.rows, width * 2)}
        self.engine = HixlEngine(engine_endpoint, device, library_path)
        try:
            host, port = endpoint.rsplit(":", 1)
            socketserver.ThreadingTCPServer.__init__(self, (host, int(port)), _OwnerConnection)
        except BaseException:
            self.engine.close()
            raise


class PLERemoteReader(RemoteReader):
    """Fetch selected rows outside graph capture into a stable NPU buffer."""

    def __init__(self, model, prefix, parts, width, endpoint, *, capacity, library_path=None):
        if not 0 < capacity <= MAX_ROWS:
            raise ValueError("Invalid shared PLE row capacity")
        checkpoint, metadata = ple_metadata(model, prefix, parts, width)
        self.rows, self.width, self.capacity = checkpoint.rows, width, capacity
        self.closed = self.failed = False
        self._operation_lock = threading.Lock()
        self._last_use = self.engine = self.buffer = None
        host, port = endpoint.rsplit(":", 1)
        self.wire = socket.create_connection((host, int(port)), 120)
        _configure_socket(self.wire)
        try:
            _send(self.wire, dict(op="attach", metadata=metadata, capacity=capacity, device=torch.npu.current_device()))
            ready = _receive(self.wire)
            if ready.get("metadata") != metadata or not ready.get("generation"):
                raise ValueError("Shared PLE owner checkpoint or layout mismatch")
            self.metadata, self.generation = metadata, ready["generation"]
            self.remote_address, self.peer = ready["address"], ready["engine"].encode()
            self.buffer = torch.empty((capacity * width * 2,), dtype=torch.uint8, device="npu")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind((self.wire.getsockname()[0], 0))
                address, local_port = reservation.getsockname()
            self.engine = HixlEngine(f"{address}:{local_port}", torch.npu.current_device(), library_path)
            self.registration = self.engine.register(self.buffer)
            self.engine.check(self.engine.lib.engram_hixl_connect(self.engine.handle, self.peer))
            self.ple_external_checkpoint_names = checkpoint.weight_names
        except BaseException:
            if self.engine is not None:
                _hold_reader(self)
            else:
                self.wire.close()
            raise

    def fetch_into(self, ids: torch.Tensor, output: torch.Tensor, *, vocab_start: int, vocab_end: int) -> None:
        """Populate graph input from global IDs, with unowned IDs masked to zero."""
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Shared PLE HIXL fetch must run outside graph capture")
        self._fetch_into(ids, output, vocab_start=vocab_start, vocab_end=vocab_end)

    @_reader_operation
    def _fetch_into(self, ids: torch.Tensor, output: torch.Tensor, *, vocab_start: int, vocab_end: int) -> None:
        if not 0 <= vocab_start <= vocab_end <= self.rows:
            raise ValueError("Invalid shared PLE vocabulary range")
        if ids.dtype not in (torch.int32, torch.int64) or output.dtype != torch.bfloat16:
            raise ValueError("Shared PLE requires integer IDs and BF16 output")
        if output.shape != (*ids.shape, self.width) or not output.is_contiguous():
            raise ValueError("Shared PLE output does not match the IDs")
        assert self.buffer is not None
        if output.device != self.buffer.device or ids.numel() > self.capacity:
            raise ValueError("Shared PLE output device or row capacity is invalid")
        selected = ids.to(device="cpu", dtype=torch.int64).contiguous().numpy().reshape(-1)
        owned = (selected >= vocab_start) & (selected < vocab_end)
        selected = np.where(owned, selected, -1).astype(np.int64, copy=False)
        self._transfer_rows(
            dict(op="lookup", layer="ple", count=len(selected)), selected, len(selected) * self.width * 2
        )
        self._last_use = torch.npu.current_stream()
        if len(selected):
            output.copy_(self.buffer[: len(selected) * self.width * 2].view(torch.bfloat16).reshape(output.shape))
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._last_use = event
