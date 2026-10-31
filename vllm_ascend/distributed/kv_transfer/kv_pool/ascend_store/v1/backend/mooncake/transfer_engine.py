"""Own the process-wide engine and registration references used only by v1."""

import threading
from typing import Any

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

from ..registration import BufferRegistration, rollback_buffer_registration


class RegisteredTransferEngine:
    """Keep v1 registrations isolated from legacy connector registrations.

    The upstream owner supplies native initialization and its failure policy.
    Each v1 client holds a retryable registration; the final reference releases
    the native region, while the engine itself remains process-owned.
    """

    def __init__(self) -> None:
        self._engine_owner = GlobalTE()
        self._registration_lock = threading.RLock()
        self._registered_regions: dict[int, tuple[int, int]] = {}

    def get_transfer_engine(self, hostname: str, device_name: str | None) -> Any:
        return self._engine_owner.get_transfer_engine(hostname, device_name)

    def acquire_registration(self, addresses: list[int], sizes: list[int]) -> BufferRegistration:
        if len(addresses) != len(sizes):
            raise ValueError(f"addresses and sizes must have the same length: {len(addresses)} != {len(sizes)}")
        registration = BufferRegistration(self._release_region)
        with self._registration_lock:
            assert self._engine_owner.transfer_engine is not None, "Transfer engine must be initialized"
            try:
                for address, size in zip(addresses, sizes, strict=True):
                    self._acquire_region(address, size)
                    registration.acquire(address, size)
            except BaseException as error:
                rollback_buffer_registration(registration, error)
                raise
        return registration

    def _acquire_region(self, ptr: int, size: int) -> None:
        if ptr < 0 or size <= 0:
            raise ValueError(f"Invalid Mooncake region: address={ptr}, size={size}")
        current = self._registered_regions.get(ptr)
        if current is not None:
            registered_size, ref_count = current
            if registered_size != size:
                raise ValueError(f"Mooncake address {ptr} is registered with size {registered_size}, not {size}")
            self._registered_regions[ptr] = (registered_size, ref_count + 1)
            return
        transfer_engine = self._engine_owner.transfer_engine
        assert transfer_engine is not None, "Transfer engine must be initialized"
        native_result = transfer_engine.register_memory(ptr, size)
        if native_result != 0:
            raise RuntimeError(f"Mooncake memory registration failed: ptr={ptr}, size={size}, ret={native_result}")
        self._registered_regions[ptr] = (size, 1)

    def _release_region(self, ptr: int, size: int) -> None:
        with self._registration_lock:
            current = self._registered_regions.get(ptr)
            if current is None:
                raise RuntimeError(f"Mooncake address {ptr} is not registered")
            registered_size, ref_count = current
            if registered_size != size:
                raise RuntimeError(f"Mooncake address {ptr} is registered with size {registered_size}, not {size}")
            if ref_count > 1:
                self._registered_regions[ptr] = (registered_size, ref_count - 1)
                return
            assert self._engine_owner.transfer_engine is not None, "Transfer engine must be initialized"
            native_result = self._engine_owner.transfer_engine.unregister_memory(ptr)
            if native_result != 0:
                raise RuntimeError(
                    f"Mooncake memory deregistration failed: ptr={ptr}, size={size}, ret={native_result}"
                )
            del self._registered_regions[ptr]


transfer_engine = RegisteredTransferEngine()
