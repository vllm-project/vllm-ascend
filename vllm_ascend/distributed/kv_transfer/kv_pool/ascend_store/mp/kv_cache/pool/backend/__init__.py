"""Backend variants that are safe to own inside KVCacheServer."""

import importlib
from typing import Any

import torch

_BACKEND_CLASSES = {
    "memcache": ("memcache", "MPMemcacheBackend"),
    "mooncake": ("mooncake", "MPMooncakeBackend"),
    "yuanrong": ("yuanrong", "MPYuanrongBackend"),
}


def create_mp_backend(backend_name: str, parallel_config: object, device_index: int | None, lazy_init: bool) -> Any:
    """Create only the configured backend so optional SDKs stay optional."""
    if device_index is None:
        raise ValueError("Worker KV cache mapping did not provide an NPU device index")

    backend_config = _BACKEND_CLASSES.get(backend_name.lower())
    if backend_config is None:
        raise ValueError(f"Unsupported AscendStore backend {backend_name!r}")

    module_name, class_name = backend_config
    torch.npu.set_device(device_index)
    module = importlib.import_module(f"{__name__}.{module_name}")
    backend_class = getattr(module, class_name)
    return backend_class(parallel_config, device_index, lazy_init=lazy_init)


__all__ = ["create_mp_backend"]
