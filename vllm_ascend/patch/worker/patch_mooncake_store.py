# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Optional legacy Ascend fabric support for Mooncake Store.

Enable with enable_ascend_fabric_patch=true in MooncakeStoreConnector's
kv_connector_extra_config. It defaults to false; HDK 26.1 or later can disable it.
The switch does not automatically detect the HDK version. This patch targets
the upstream Store worker, not AscendStoreConnector or MooncakeConnectorV1.
The option controls every replacement below, including NPU thread binding.
When disabled or imported without a config context, no replacements are made.
Fabric handles KV-cache registration, which must not create its agent.
"""

import os
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.config.kv_transfer import KVTransferConfig

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker
except ModuleNotFoundError as exc:
    if exc.name != "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store":
        raise
    # Retain compatibility with vLLM revisions before the Store package split.
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker


@dataclass
class AscendMooncakeStoreConfig(worker.MooncakeStoreConfig):
    use_ascend_fabric_mem: bool = field(
        init=False, default_factory=lambda: os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_ascend_fabric_mem and self.protocol != "ascend":
            raise ValueError("Ascend fabric memory requires protocol='ascend'")


class DeferredFabricStore:
    """Defer backend setup until serving or a Store operation after capture."""

    def __init__(self, store: Any, device_id: int):
        self._store = store
        self._device_id = device_id
        self._setup_args: tuple[Any, ...] = ()
        self._setup_kwargs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._ready = False
        self._closed = False
        self._thread_state = threading.local()

    def setup(self, *args: Any, **kwargs: Any) -> int:
        self._setup_args = args
        self._setup_kwargs = kwargs
        return 0

    def register_buffer(self, address: int, size: int) -> int:
        # Fabric owns memory registration, matching AscendStore's fabric path.
        return 0

    def _ensure_initialized(self) -> None:
        if not getattr(self._thread_state, "device_bound", False):
            torch.npu.set_device(self._device_id)
            self._thread_state.device_bound = True
        if self._closed:
            raise RuntimeError("Mooncake store is closed")
        if self._ready:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("Mooncake store is closed")
            if self._ready:
                return
            ret = self._store.setup(*self._setup_args, **self._setup_kwargs)
            if ret != 0:
                raise RuntimeError(f"Initialize MooncakeDistributedStore failed: {ret}")
            self._ready = True

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._store, name)

        @wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            self._ensure_initialized()
            return method(*args, **kwargs)

        return call

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._ready:
                self._store.close()
                self._ready = False


COMMON_SETUP_STORE = getattr(worker.MooncakeStoreWorker, "_setup_store", None)
COMMON_TRANSFER_INIT = worker.KVTransferThread.__init__
COMMON_LOOKUP = worker.MooncakeStoreWorker.lookup


def _setup_store(self, config: AscendMooncakeStoreConfig) -> None:
    if config.use_ascend_fabric_mem:
        self.store = DeferredFabricStore(self.store, torch.npu.current_device())
    assert COMMON_SETUP_STORE is not None
    COMMON_SETUP_STORE(self, config)


@wraps(COMMON_LOOKUP)
def _lookup(self, *args: Any, **kwargs: Any):
    # Lookup runs on the RPC thread, before the first serving callback. Match
    # AscendStore's cold lookup behavior and keep initial fabric setup on the
    # serving thread instead of initializing device resources from this thread.
    if isinstance(self.store, DeferredFabricStore) and not self.store._ready:
        return worker.MooncakeLookupResult(0)
    return COMMON_LOOKUP(self, *args, **kwargs)


@wraps(COMMON_TRANSFER_INIT)
def _transfer_init(self, *args: Any, **kwargs: Any) -> None:
    self._ascend_device_id = torch.npu.current_device()
    self._ascend_device_bound = False
    COMMON_TRANSFER_INIT(self, *args, **kwargs)


def _bind_request_device(handle_request):
    @wraps(handle_request)
    def call(self, *args: Any, **kwargs: Any):
        if not self._ascend_device_bound:
            torch.npu.set_device(self._ascend_device_id)
            self._ascend_device_bound = True
        return handle_request(self, *args, **kwargs)

    return call


def _initialize_on_serving(method):
    @wraps(method)
    def call(self, *args: Any, **kwargs: Any):
        if isinstance(self.store, DeferredFabricStore):
            self.store._ensure_initialized()
        return method(self, *args, **kwargs)

    return call


def _get_store_extra_configs(kv_transfer_config: KVTransferConfig | None) -> list[dict[str, Any]]:
    if kv_transfer_config is None:
        return []
    configs = [
        {
            "kv_connector": kv_transfer_config.kv_connector,
            "kv_connector_extra_config": kv_transfer_config.kv_connector_extra_config,
        }
    ]
    if kv_transfer_config.kv_connector == "MultiConnector":
        configs = kv_transfer_config.kv_connector_extra_config.get("connectors", [])
    return [
        config.get("kv_connector_extra_config", {})
        for config in configs
        if config.get("kv_connector") == "MooncakeStoreConnector"
    ]


def apply_mooncake_store_patch(kv_transfer_config: KVTransferConfig | None = None) -> None:
    """Install the compatibility patch only when explicitly enabled."""
    if kv_transfer_config is None:
        config = get_current_vllm_config_or_none()
        kv_transfer_config = config.kv_transfer_config if config is not None else None
    extra_configs = _get_store_extra_configs(kv_transfer_config)
    if not extra_configs:
        return
    enabled = [config.get("enable_ascend_fabric_patch", False) for config in extra_configs]
    if any(type(value) is not bool for value in enabled):
        raise ValueError("enable_ascend_fabric_patch must be a JSON boolean")
    if enabled and any(value != enabled[0] for value in enabled):
        raise ValueError("MooncakeStoreConnector instances in one worker must use the same enable_ascend_fabric_patch")
    if not any(enabled) or COMMON_SETUP_STORE is None:
        return
    if worker.MooncakeStoreWorker._setup_store is _setup_store:
        return

    worker.KVTransferThread.__init__ = _transfer_init
    worker.KVCacheStoreSendingThread._handle_request = _bind_request_device(
        worker.KVCacheStoreSendingThread._handle_request
    )
    worker.KVCacheStoreRecvingThread._handle_request = _bind_request_device(
        worker.KVCacheStoreRecvingThread._handle_request
    )
    worker.MooncakeStoreConfig = AscendMooncakeStoreConfig
    worker.MooncakeStoreWorker._setup_store = _setup_store
    worker.MooncakeStoreWorker.lookup = _lookup
    worker.MooncakeStoreWorker.start_load_kv = _initialize_on_serving(worker.MooncakeStoreWorker.start_load_kv)
    worker.MooncakeStoreWorker.wait_for_save = _initialize_on_serving(worker.MooncakeStoreWorker.wait_for_save)
    worker.MooncakeStoreWorker.build_connector_worker_meta = _initialize_on_serving(
        worker.MooncakeStoreWorker.build_connector_worker_meta
    )


# Imported by the worker patch entry; no config context means disabled.
apply_mooncake_store_patch()
