from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.utils.utils import validate_uint64_transfer_columns


def _resolve_batch_arrays_fn(engine: Any, method_name: str) -> Callable[..., int] | None:
    """Return Mooncake's array-transfer API when explicitly exposed and callable (>= v0.3.14)."""
    # Probe statically first so dynamic __getattr__ objects (e.g. unconfigured Mock())
    # do not synthesize a fake callable attribute.
    if inspect.getattr_static(engine, method_name, None) is None:
        return None
    array_fn = getattr(engine, method_name, None)
    return array_fn if callable(array_fn) else None


def _batch_transfer_sync(
    engine: Any,
    session_id: str,
    src_addrs: npt.NDArray[np.uint64],
    dst_addrs: npt.NDArray[np.uint64],
    lengths: npt.NDArray[np.uint64],
    method_name: str,
) -> int:
    if validate_uint64_transfer_columns(src_addrs, dst_addrs, lengths) == 0:
        return 0
    array_fn = _resolve_batch_arrays_fn(engine, f"{method_name}_arrays")
    if array_fn is not None:
        return array_fn(session_id, src_addrs, dst_addrs, lengths)
    return getattr(engine, method_name)(session_id, src_addrs.tolist(), dst_addrs.tolist(), lengths.tolist())


def batch_transfer_sync_write(
    engine: Any,
    session_id: str,
    src_addrs: npt.NDArray[np.uint64],
    dst_addrs: npt.NDArray[np.uint64],
    lengths: npt.NDArray[np.uint64],
) -> int:
    """Submit contiguous uint64 write descriptors, falling back to list API on < v0.3.14."""
    return _batch_transfer_sync(engine, session_id, src_addrs, dst_addrs, lengths, "batch_transfer_sync_write")


def batch_transfer_sync_read(
    engine: Any,
    session_id: str,
    src_addrs: npt.NDArray[np.uint64],
    dst_addrs: npt.NDArray[np.uint64],
    lengths: npt.NDArray[np.uint64],
) -> int:
    """Submit contiguous uint64 read descriptors, falling back to list API on < v0.3.14."""
    return _batch_transfer_sync(engine, session_id, src_addrs, dst_addrs, lengths, "batch_transfer_sync_read")


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._initialization_attempted = False
        self.supports_write_arrays: bool = False
        self.supports_read_arrays: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        # Buffers unregistered since the last full (re)registration, so a
        # failure part-way through either direction can be resumed.
        self._unregistered_ptrs: set[int] = set()
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    if self._initialization_attempted:
                        raise RuntimeError(
                            "TransferEngine initialization previously failed. "
                            "Restart the process before retrying to avoid leaking native resources."
                        )
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    # Mooncake 0.3.11.post1 leaks transport resources on failed
                    # initialization (kvcache-ai/Mooncake#2332). Mark the attempt
                    # before native construction, including exceptions. Revisit
                    # retries once the minimum supported version includes that
                    # fix and repeated-failure resource cleanup is verified.
                    self._initialization_attempted = True
                    engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"TransferEngine initialization failed with ret_value: {ret_value}. "
                            "Restart the process before retrying."
                        )
                    self.supports_write_arrays = (
                        _resolve_batch_arrays_fn(engine, "batch_transfer_sync_write_arrays") is not None
                    )
                    self.supports_read_arrays = (
                        _resolve_batch_arrays_fn(engine, "batch_transfer_sync_read_arrays") is not None
                    )
                    missing_array_apis = [
                        name
                        for name, supported in (
                            ("batch_transfer_sync_write_arrays", self.supports_write_arrays),
                            ("batch_transfer_sync_read_arrays", self.supports_read_arrays),
                        )
                        if not supported
                    ]
                    if missing_array_apis:
                        logger.info(
                            "Mooncake Transfer Engine does not expose callable %s; falling back to "
                            "the list API. Install mooncake-transfer-engine v0.3.14 or newer "
                            "to use the array API.",
                            ", ".join(missing_array_apis),
                        )
                    self.transfer_engine = engine
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            # Register once per process. Once recorded, buffers dropped by
            # unregister_buffer() come back through reregister_buffer(), so a
            # stray call here cannot register a still-registered buffer twice.
            if self.registered_buffers:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.registered_buffers = list(zip(ptrs, sizes))

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if len(self._unregistered_ptrs) == len(self.registered_buffers):
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            # Record each buffer as it goes, so that after a failure part-way
            # reregister_buffer() restores exactly the ones this call dropped.
            # Ascend Direct unregister_memory also disconnects peers.
            for ptr, _ in self.registered_buffers:
                if ptr in self._unregistered_ptrs:
                    continue
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    raise RuntimeError(f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}")
                self._unregistered_ptrs.add(ptr)

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if not self._unregistered_ptrs:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            for ptr, size in self.registered_buffers:
                if ptr not in self._unregistered_ptrs:
                    continue
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, ret_value={ret_value}"
                    )
                self._unregistered_ptrs.discard(ptr)

    def disconnect_all_peers(self) -> None:
        with self.register_buffer_lock:
            if self.transfer_engine is None:
                return
            ret = self.transfer_engine.disconnect_all_peers()
            if ret != 0:
                raise RuntimeError(f"Mooncake peer disconnection failed with ret_value: {ret}")


global_te = GlobalTE()
