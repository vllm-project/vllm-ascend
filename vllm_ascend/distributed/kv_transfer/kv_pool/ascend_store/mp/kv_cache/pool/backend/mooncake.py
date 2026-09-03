"""Mooncake backend owned by one multiprocess Worker service."""

from typing import Any

import torch
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

from .....backend.mooncake_backend import MooncakeBackend

_MOONCAKE_MP_MIN_VERSION = "mooncake-transfer-engine-npu>=0.3.12.post1"
_MOONCAKE_MP_REQUIRED_API = "register_memory(address, size, location)"


def _mooncake_register_version_error(exc: TypeError) -> RuntimeError:
    """Explain a location-argument rejection from an old mooncake binding.

    The rejection is either an outdated package or an old mooncake module
    (typically under the CANN python path) shadowing the pip-installed one,
    so the message includes where mooncake.engine was actually loaded from.
    """
    try:
        import mooncake.engine as mooncake_engine

        loaded_from = getattr(mooncake_engine, "__file__", "<unknown>")
    except Exception:
        loaded_from = "<mooncake.engine not importable>"
    return RuntimeError(
        f"AscendStore MP with Mooncake requires {_MOONCAKE_MP_MIN_VERSION} "
        f"(needs {_MOONCAKE_MP_REQUIRED_API}); the loaded binding rejected the "
        f"location argument: {exc}. mooncake.engine was loaded from {loaded_from!r} "
        "and may be shadowing the pip-installed package."
    )


class MPMooncakeBackend(MooncakeBackend):
    """Own the Mooncake buffer registration for one MP Worker service."""

    def __init__(self, parallel_config: Any, device_index: int, lazy_init: bool = False):
        self.device_index = device_index
        self._mp_registered_ptrs: list[int] = []
        super().__init__(parallel_config, lazy_init=lazy_init)

    def set_device(self) -> None:
        torch.npu.set_device(self.device_index)

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None:
        if self._use_fabric_mem:
            return
        if self._mp_registered_ptrs:
            raise RuntimeError("Mooncake buffers are already registered for this Worker")

        self.set_device()
        transfer_engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        location = f"npu:{self.device_index}"
        registered: list[int] = []
        with global_te.register_buffer_lock:
            try:
                for ptr, length in zip(ptrs, lengths):
                    result = transfer_engine.register_memory(ptr, length, location)
                    if result != 0:
                        raise RuntimeError(
                            f"Mooncake memory registration failed with code {result}: "
                            f"address=0x{ptr:x}, length={length}"
                        )
                    registered.append(ptr)
            except BaseException as exc:
                for ptr in reversed(registered):
                    transfer_engine.unregister_memory(ptr)
                if isinstance(exc, TypeError) and "register_memory" in str(exc):
                    raise _mooncake_register_version_error(exc) from exc
                raise
        self._mp_registered_ptrs = registered

    def unregister_buffer(self) -> None:
        if self._use_fabric_mem or not self._mp_registered_ptrs:
            return

        transfer_engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        failed: list[tuple[int, object]] = []
        released: set[int] = set()
        with global_te.register_buffer_lock:
            for ptr in reversed(self._mp_registered_ptrs):
                result = transfer_engine.unregister_memory(ptr)
                if result != 0:
                    failed.append((ptr, result))
                else:
                    released.add(ptr)
        self._mp_registered_ptrs = [ptr for ptr in self._mp_registered_ptrs if ptr not in released]
        if failed:
            raise RuntimeError(f"Mooncake memory unregistration failed: {failed!r}")

    def close(self) -> None:
        try:
            self.unregister_buffer()
        finally:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()
