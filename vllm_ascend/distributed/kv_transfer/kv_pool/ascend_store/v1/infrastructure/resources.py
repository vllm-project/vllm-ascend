from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger

from ..backend import (
    BackendSpec,
    BufferRegistrationError,
    KVStoreBackend,
    ObjectSessions,
    Registration,
)
from .execution.executor import Executor

if TYPE_CHECKING:
    from ..topology import KVPoolGroupTopology


_GVA_REGISTRATION_ALIGNMENT = 2 * 1024 * 1024


class KVPoolResources:
    """Manage resources that need to be started, stopped, or explicitly released.

    Own Executors, Backend object sessions, buffer registrations, and the Backend
    client. Keep KV tensors alive until cleanup succeeds. The Worker submits work
    and decides when to start, stop, and release these resources; mappings and
    per-step execution Sessions remain Worker-owned.
    """

    def __init__(
        self,
        backend: KVStoreBackend,
        backend_spec: BackendSpec,
        num_blocks: int,
        groups: tuple[KVPoolGroupTopology, ...],
        *,
        align_shared_storage: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self._backend_sessions: ObjectSessions | None = None
        self.num_blocks = num_blocks
        self._groups = groups
        self._align_shared_storage = align_shared_storage
        self._registered_regions: tuple[tuple[int, int], ...] = ()
        self.kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None
        self._buffer_registration: Registration | None = None
        self._binding_started = False
        self._closed = False
        self._executors: tuple[Executor, ...] | None = None

    def bind_backend_sessions(self, sessions: ObjectSessions) -> None:
        if self._closed or self._binding_started or self._backend_sessions is not None:
            raise RuntimeError("Backend sessions must be bound once before KV caches")
        self._backend_sessions = sessions

    def bind_executors(self, executors: Sequence[Executor]) -> None:
        if self._closed or self._binding_started or self._executors is not None:
            raise RuntimeError("Executors must be bound once before KV caches")
        self._executors = tuple(executors)

    def start_executors(self) -> None:
        """Call after KV buffer registration; failed starts retain all executors for cleanup."""
        if self._closed:
            raise RuntimeError("KV resources are closed")
        for executor in self._executors or ():
            executor.start()

    def stop_executors(self) -> None:
        """Attempt every drain even after a failure, retaining the first error."""
        close_error: BaseException | None = None
        for executor in self._executors or ():
            try:
                executor.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    logger.exception("Failed to stop executor after an earlier drain failure")
        if close_error is not None:
            raise close_error

    @property
    def executors_stopped(self) -> bool:
        return all(executor.stopped for executor in self._executors or ())

    def prepare_kv_caches(self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._binding_started:
            raise RuntimeError("KV caches are already bound")
        self._binding_started = True
        self.kv_caches = kv_caches
        return self._extract_kv_layout(kv_caches)

    def register_kv_buffers(self) -> None:
        """Acquire registrations only after the Worker has bound its mappings."""
        addresses, sizes = [], []
        for start, end in self._registered_regions:
            addresses.append(start)
            sizes.append(end - start)
        try:
            self._buffer_registration = self.backend.register_buffer(addresses, sizes)
        except BufferRegistrationError as error:
            self._buffer_registration = error.registration
            raise

    def close(self) -> None:
        """Call after the Worker observes results and confirms Store source release.

        Stopped threads alone do not prove source safety. Release sessions before
        registrations and the client; stop at a failed cleanup and retain ownership
        so the caller can retry without releasing resources still in use.
        """
        if self._closed:
            return
        if not self.executors_stopped:
            raise RuntimeError("Executors must stop before KV resources are released")
        if self._backend_sessions is not None:
            self._backend_sessions.close()
        if self._buffer_registration is not None:
            self._buffer_registration.close()
        self.backend.close()
        self.kv_caches = None
        self._closed = True

    def _extract_kv_layout(
        self,
        kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    ) -> dict[str, Any]:
        base_addresses: dict[int, list[int]] = {}
        block_lengths: dict[int, list[int]] = {}
        block_strides: dict[int, list[int]] = {}
        layer_entry_offsets: dict[int, list[int]] = {}
        registered_regions: dict[int, tuple[int, int]] = {}

        for group in self._groups:
            bases: list[int] = []
            lengths: list[int] = []
            strides: list[int] = []
            layer_offsets = [0]
            for layer in group.layers:
                for layer_name in layer.layer_names:
                    cache_or_caches = kv_caches[layer_name]
                    caches = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else tuple(cache_or_caches)
                    for cache in caches:
                        # NoPE MLA exposes an empty RoPE view whose data_ptr() is zero.
                        if not cache.numel():
                            continue
                        assert cache.shape[0] % self.num_blocks == 0, (
                            "The external block size must be an integer multiple of the kernel block size."
                        )
                        block_scale = cache.shape[0] // self.num_blocks
                        block_length = cache[0].numel() * cache.element_size() * block_scale
                        block_stride = cache.stride(0) * cache.element_size() * block_scale
                        address = cache.data_ptr()
                        bases.append(address)
                        lengths.append(block_length)
                        strides.append(block_stride)
                        # Payload lengths exclude padding; registration must still cover the last kernel block.
                        element_size = cache.element_size()
                        kernel_block_span = element_size + sum(
                            (size - 1) * stride * element_size
                            for size, stride in zip(cache.shape[1:], cache.stride()[1:], strict=True)
                        )
                        region_end = (
                            address
                            + (self.num_blocks * block_scale - 1) * cache.stride(0) * element_size
                            + kernel_block_span
                        )
                        storage_key = cache.untyped_storage().data_ptr()
                        previous = registered_regions.get(storage_key)
                        registered_regions[storage_key] = (
                            (min(previous[0], address), max(previous[1], region_end))
                            if previous is not None
                            else (address, region_end)
                        )
                layer_offsets.append(len(bases))
            base_addresses[group.group_id] = bases
            block_lengths[group.group_id] = lengths
            block_strides[group.group_id] = strides
            layer_entry_offsets[group.group_id] = layer_offsets

        if self._align_shared_storage:
            # Hybrid tensor views can start inside a larger allocation. Native GVA
            # registration needs an aligned start, independently of object offsets.
            for storage_key, (start, end) in registered_regions.items():
                aligned_start = start // _GVA_REGISTRATION_ALIGNMENT * _GVA_REGISTRATION_ALIGNMENT
                if aligned_start < storage_key:
                    raise ValueError("Shared KV storage cannot satisfy the GVA alignment boundary")
                registered_regions[storage_key] = aligned_start, end

        registration: dict[str, Any] = {
            "base_addresses": base_addresses,
            "block_lengths": block_lengths,
            "block_strides": block_strides,
            "layer_entry_offsets": layer_entry_offsets,
        }
        self._registered_regions = tuple(registered_regions.values())
        return registration
