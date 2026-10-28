from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

from ..backend import (
    BackendSpec,
    BufferRegistrationError,
    KVStoreBackend,
    ObjectSessions,
    Registration,
)
from .execution import Executor

if TYPE_CHECKING:
    from .configuration.topology import KVPoolGroupTopology


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
        groups: tuple[KVPoolGroupTopology, ...],
        *,
        align_shared_storage: bool = False,
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self._backend_sessions: ObjectSessions | None = None
        self._groups = groups
        self._align_shared_storage = align_shared_storage
        self._registered_regions: tuple[tuple[int, int], ...] = ()
        self.kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None
        self._buffer_registration: Registration | None = None
        self._binding_started = False
        self._closed = False
        self._executors: tuple[Executor, ...] | None = None
        self._execution_cleanup: Callable[[], None] | None = None

    def bind_execution_cleanup(self, close: Callable[[], None]) -> None:
        """Retain a route-owned, retryable and idempotent close operation before work starts.

        Run it only after executor drain and session cleanup, before releasing
        registrations or the client. Retain it on failure for a later close retry.
        """
        if self._closed or self._binding_started or self._execution_cleanup is not None:
            raise RuntimeError("Execution cleanup must be bound once before KV caches")
        self._execution_cleanup = close

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

    def prepare_kv_caches(self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]) -> None:
        if self._closed:
            raise RuntimeError("KV resources are closed")
        if self._binding_started:
            raise RuntimeError("KV caches are already bound")
        self._binding_started = True
        self.kv_caches = kv_caches
        self._registered_regions = self._collect_registered_regions(kv_caches)

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
        if self._execution_cleanup is not None:
            self._execution_cleanup()
        if self._buffer_registration is not None:
            self._buffer_registration.close()
        self.backend.close()
        self.kv_caches = None
        self._closed = True

    def _collect_registered_regions(
        self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
    ) -> tuple[tuple[int, int], ...]:
        """Cover each view's actual storage span, including gaps excluded from payloads."""
        registered_regions: dict[int, tuple[int, int]] = {}
        for group in self._groups:
            for layer in group.layers:
                for layer_name in layer.layer_names:
                    cache_or_caches = kv_caches[layer_name]
                    caches = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else cache_or_caches
                    for cache in caches:
                        # NoPE MLA exposes an empty RoPE view whose data_ptr() is zero.
                        if not cache.numel():
                            continue
                        address = cache.data_ptr()
                        element_size = cache.element_size()
                        region_end = (
                            address
                            + element_size
                            + sum(
                                (size - 1) * stride * element_size
                                for size, stride in zip(cache.shape, cache.stride(), strict=True)
                            )
                        )
                        storage_key = cache.untyped_storage().data_ptr()
                        previous = registered_regions.get(storage_key)
                        registered_regions[storage_key] = (
                            (min(previous[0], address), max(previous[1], region_end))
                            if previous is not None
                            else (address, region_end)
                        )
        if self._align_shared_storage:
            # GVA registration alignment is independent of payload and object offsets.
            for storage_key, (start, end) in registered_regions.items():
                aligned_start = start // _GVA_REGISTRATION_ALIGNMENT * _GVA_REGISTRATION_ALIGNMENT
                if aligned_start < storage_key:
                    raise ValueError("Shared KV storage cannot satisfy the GVA alignment boundary")
                registered_regions[storage_key] = aligned_start, end
        return tuple(registered_regions.values())
