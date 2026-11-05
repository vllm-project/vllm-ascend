# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIXL READ session for the Ascend sharded_rdt transport.

Registered storage stays owned by this object until successful deregistration.
After a failed transfer the caller must terminate the sync, never retry using
the same ring or fall back to another transport.
"""

import threading
import time
from typing import Any

import torch
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

ALIGNMENT = 2 << 20
MAX_REGIONS = 4000
TRANSFER_TIMEOUT_S = 300.0
POLL_INTERVAL_S = 0.001


class HixlSession:
    def __init__(self, device: torch.device, *, binding: Any = None) -> None:
        # Lazy runtime imports keep factory discovery usable on CPU hosts.
        import acl

        if binding is None:
            try:
                import hixl as binding
            except ImportError:
                from vllm_ascend.distributed.eplb import hixl_compat as binding

                binding.ensure_available()
        self._binding = binding
        self._acl = acl
        self._device = device
        torch.npu.set_device(device)
        self._context, status = acl.rt.get_context()
        self._check(status, "get ACL context")
        self.endpoint = join_host_port(get_ip(), get_open_port())
        self._engine = binding.Hixl()
        self._regions: dict[int, tuple[Any, Any, int, int]] = {}
        self._peers: set[str] = set()
        self._requests: set[Any] = set()
        self._lock = threading.RLock()
        self._closed = False
        self._error: Exception | None = None
        self._check(self._engine.initialize(self.endpoint, {}), "initialize")

    def _check(self, status: int, operation: str) -> None:
        if status != self._binding.SUCCESS:
            error = RuntimeError(f"HIXL {operation} failed: status={status}")
            self._error = error
            raise error

    def _activate(self) -> None:
        if self._closed:
            raise RuntimeError("HIXL session is closed")
        if self._error is not None:
            raise RuntimeError("HIXL session failed; terminate the worker") from self._error
        self._check(self._acl.rt.set_context(self._context), "set ACL context")

    def register(self, tensor: torch.Tensor) -> None:
        with self._lock:
            self._activate()
            if tensor.device.type != "npu" or not tensor.is_contiguous():
                raise ValueError("HIXL requires contiguous NPU buffers")
            storage = tensor.untyped_storage()
            address, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
            if address in self._regions:
                return
            if address % ALIGNMENT or size % ALIGNMENT or size <= 0:
                raise ValueError("HIXL ring buffers must be aligned to 2 MiB")
            if len(self._regions) >= MAX_REGIONS:
                raise ValueError("HIXL registration limit exceeded")
            status, handle = self._engine.register_mem(
                self._binding.MemDesc(address, size),
                self._binding.MemType.MEM_DEVICE,
            )
            self._check(status, "register memory")
            self._regions[address] = (handle, storage, address, size)

    def allocate(self, nbytes: int) -> torch.Tensor:
        size = (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        owner = torch.empty(size + ALIGNMENT, dtype=torch.uint8, device=self._device)
        offset = (-owner.data_ptr()) % ALIGNMENT
        tensor = owner[offset : offset + size]
        self.register(tensor)
        return tensor

    def read(self, endpoint: str, target: torch.Tensor, address: int, nbytes: int) -> Any:
        with self._lock:
            self._activate()
            if nbytes <= 0 or nbytes > target.numel() * target.element_size():
                raise ValueError("Invalid HIXL READ size")
            if not any(
                start <= target.data_ptr() and target.data_ptr() + nbytes <= start + size
                for _handle, _storage, start, size in self._regions.values()
            ):
                raise ValueError("HIXL READ target was not registered")
            if endpoint not in self._peers:
                self._check(self._engine.connect(endpoint), "connect")
                self._peers.add(endpoint)
            operations = [
                self._binding.TransferOpDesc(
                    local_addr=target.data_ptr(),
                    remote_addr=address,
                    len=nbytes,
                )
            ]
            status, request = self._engine.transfer_async(
                endpoint,
                self._binding.TransferOp.READ,
                operations,
            )
            self._check(status, "READ")
            self._requests.add(request)
            return request

    def wait(self, request: Any) -> None:
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        try:
            while True:
                with self._lock:
                    self._activate()
                    if request not in self._requests:
                        raise ValueError("Unknown or already consumed HIXL request")
                    status, state = self._engine.get_transfer_status(request)
                    waiting = self._binding.TransferStatus.WAITING
                    if status != self._binding.SUCCESS or state != waiting:
                        self._requests.remove(request)
                    self._check(status, "query READ")
                    if state == self._binding.TransferStatus.COMPLETED:
                        return
                    if state != waiting:
                        raise RuntimeError(f"HIXL READ failed: state={state}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("HIXL READ timed out; registered buffers retained")
                time.sleep(POLL_INTERVAL_S)
        except Exception as error:
            self._error = error
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._activate()
            if self._requests:
                raise RuntimeError("Drain HIXL READs before closing the session")
            for peer in tuple(self._peers):
                self._check(self._engine.disconnect(peer), "disconnect")
                self._peers.remove(peer)
            for address, (handle, _storage, _start, _size) in tuple(self._regions.items()):
                self._check(self._engine.deregister_mem(handle), "deregister memory")
                del self._regions[address]
            status = self._engine.finalize()
            if status is not None:
                self._check(status, "finalize")
            self._closed = True
