# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import weakref
from typing import Any

import torch


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.element_size() * tensor.numel())


class _DSAPDMooncakeKVRegistry:
    """Process-local bridge between KV allocation and Mooncake setup.

    Tensor references are weak so reinitializing a model runner cannot keep an
    obsolete KV cache alive. Registration happens during startup; connector
    threads only read the registry after initialization.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cpu_buffers: list[tuple[int, int]] = []
        self._offloaded_tensors: list[
            tuple[int, int, weakref.ReferenceType[torch.Tensor]]
        ] = []

    def register_cpu_buffer(self, ptr: int, size: int) -> None:
        if ptr <= 0 or size <= 0:
            return
        entry = (int(ptr), int(size))
        with self._lock:
            if entry not in self._cpu_buffers:
                self._cpu_buffers.append(entry)

    def register_offloaded_tensor(self, tensor: torch.Tensor) -> None:
        ptr = int(tensor.data_ptr())
        size = _tensor_nbytes(tensor)
        if ptr <= 0 or size <= 0:
            return

        with self._lock:
            live_entries = []
            already_registered = False
            for start, length, tensor_ref in self._offloaded_tensors:
                existing = tensor_ref()
                if existing is None:
                    continue
                live_entries.append((start, length, tensor_ref))
                already_registered = already_registered or (
                    start == ptr and length == size
                )
            if not already_registered:
                live_entries.append((ptr, size, weakref.ref(tensor)))
            self._offloaded_tensors = live_entries

    def clear(self) -> None:
        with self._lock:
            self._cpu_buffers.clear()
            self._offloaded_tensors.clear()

    def get_register_location(self, ptr: int, size: int) -> str | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        with self._lock:
            for start, length in self._cpu_buffers:
                if ptr >= start and end <= start + length:
                    return "cpu"
        return None

    def resolve_offloaded_tensor(
        self,
        ptr: int,
        size: int,
    ) -> tuple[torch.Tensor, int] | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        resolved = None
        with self._lock:
            live_entries = []
            for start, length, tensor_ref in self._offloaded_tensors:
                tensor = tensor_ref()
                if tensor is None:
                    continue
                live_entries.append((start, length, tensor_ref))
                if resolved is None and ptr >= start and end <= start + length:
                    resolved = (tensor, ptr - start)
            self._offloaded_tensors = live_entries
        return resolved


_DSA_PD_MOONCAKE_KV_REGISTRY = _DSAPDMooncakeKVRegistry()


def register_dsa_pd_mooncake_cpu_kv_buffer(ptr: int, size: int) -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.register_cpu_buffer(ptr, size)


def register_dsa_pd_mooncake_offloaded_kv_tensor(tensor: torch.Tensor) -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.register_offloaded_tensor(tensor)



def reset_dsa_pd_mooncake_kv_registries() -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.clear()


def clear_dsa_pd_mooncake_kv_buffer_registries_for_testing() -> None:
    reset_dsa_pd_mooncake_kv_registries()


def get_dsa_pd_mooncake_kv_register_location(ptr: int, size: int) -> str | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.get_register_location(ptr, size)


def get_dsa_pd_mooncake_offloaded_kv_tensor(
    ptr: int,
    size: int,
) -> tuple[torch.Tensor, int] | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.resolve_offloaded_tensor(ptr, size)


def is_dsa_pd_mooncake_offloaded_kv_buffer(ptr: int, size: int) -> bool:
    return get_dsa_pd_mooncake_offloaded_kv_tensor(ptr, size) is not None

def _iter_connector_names(kv_transfer_config: Any) -> list[str]:
    connector_names: list[str] = []
    connector_name = getattr(kv_transfer_config, "kv_connector", "")
    if connector_name:
        connector_names.append(str(connector_name))

    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    connectors = extra_config.get("connectors", [])
    if isinstance(connectors, dict):
        connectors = connectors.values()

    for connector in connectors:
        if isinstance(connector, dict):
            nested_name = connector.get("kv_connector", "")
        else:
            nested_name = getattr(connector, "kv_connector", "")
        if nested_name:
            connector_names.append(str(nested_name))

    return connector_names


def kv_transfer_uses_mooncake(kv_transfer_config: Any) -> bool:
    if kv_transfer_config is None:
        return False
    return any("mooncake" in name.lower() for name in _iter_connector_names(kv_transfer_config))


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_dsa_sparse_attention_config(vllm_config: Any) -> Any:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return _config_get(additional_config, "dsa_sparse_attention_config", {}) or {}


def _is_dsa_sparse_model(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return bool(
        hf_text_config is not None
        and hasattr(hf_text_config, "index_topk")
        and not hasattr(hf_text_config, "compress_ratios")
    )


def _is_mooncake_kv_consumer(vllm_config: Any) -> bool:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return bool(
        kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_consumer", False)
        and kv_transfer_uses_mooncake(kv_transfer_config)
    )


def should_use_dsa_pd_mooncake_long_context_kv(vllm_config: Any) -> bool:
    """Return whether sparse KV capacity is backed by Mooncake CPU offload."""
    dsa_sparse_attention_config = _get_dsa_sparse_attention_config(vllm_config)
    return should_use_dsa_pd_mooncake_cpu_kv(
        vllm_config,
        use_sparse=_is_dsa_sparse_model(vllm_config),
        is_kv_consumer=_is_mooncake_kv_consumer(vllm_config),
        enable_cpu_kv_store=bool(
            _config_get(dsa_sparse_attention_config, "enable_cpu_kv_store", False)
        ),
        dsa_sparse_attention_mode=_config_get(dsa_sparse_attention_config, "mode", None),
    )


def should_use_dsa_pd_mooncake_cpu_kv(
    vllm_config: Any,
    *,
    use_sparse: bool,
    is_kv_consumer: bool,
    enable_cpu_kv_store: bool,
    dsa_sparse_attention_mode: str | None = None,
) -> bool:
    if dsa_sparse_attention_mode == "baseline":
        return False
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return bool(
        use_sparse
        and is_kv_consumer
        and enable_cpu_kv_store
        and kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_consumer", False)
        and kv_transfer_uses_mooncake(kv_transfer_config)
    )


def empty_mooncake_cpu_kv_memory(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError("DSA PD Mooncake offloaded KV requires torch_npu.empty_with_swapped_memory.") from e

    empty_with_swapped_memory = getattr(torch_npu, "empty_with_swapped_memory", None)
    if empty_with_swapped_memory is None:
        raise RuntimeError("DSA PD Mooncake offloaded KV requires torch_npu.empty_with_swapped_memory.")
    return empty_with_swapped_memory(shape, dtype=dtype, device="npu")
