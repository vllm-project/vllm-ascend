from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Integral
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import set_assigned_physical_gpu_ids

# QoS range supported by the pooled KV store backends. A larger value
# means a higher transfer priority.
QOS_VALUE_MIN = 0
QOS_VALUE_MAX = 4


def parse_qos_from_extra_config(extra_config: dict[str, Any] | None) -> int | None:
    """Parse and validate the ``qos_priority`` field of kv_connector_extra_config.

    Returns None when the field is absent; otherwise the QoS integer in
    [QOS_VALUE_MIN, QOS_VALUE_MAX]. Only integers are supported; an invalid
    value fails fast with a clear error instead of an obscure failure inside
    the store backends.
    """
    if not extra_config or "qos_priority" not in extra_config:
        return None
    qos = extra_config["qos_priority"]
    if isinstance(qos, bool) or not isinstance(qos, int) or not (QOS_VALUE_MIN <= qos <= QOS_VALUE_MAX):
        raise ValueError(
            f"Invalid qos_priority {qos!r} in kv_connector_extra_config: "
            f"QoS must be an integer in [{QOS_VALUE_MIN}, {QOS_VALUE_MAX}]."
        )
    return qos


class BatchResultShapeError(RuntimeError):
    """Raised when a backend returns malformed per-key batch results."""


def require_aligned_batch_results(
    operation: str,
    keys: list[str],
    results: Iterable[int] | None,
) -> list[int]:
    """Return integer results after validating one result per input key."""
    try:
        raw_values = list(results) if results is not None else []
    except (TypeError, ValueError) as exc:
        raise BatchResultShapeError(f"{operation} returned non-integer batch results") from exc
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_values):
        raise BatchResultShapeError(f"{operation} returned non-integer batch results")
    values = [int(value) for value in raw_values]
    if len(values) != len(keys):
        raise BatchResultShapeError(f"{operation} returned {len(values)} results for {len(keys)} keys")
    return values


def set_scheduler_device(parallel_config: ParallelConfig) -> None:
    assigned_ids = parallel_config.assigned_physical_gpu_ids
    if assigned_ids is not None:
        set_assigned_physical_gpu_ids(assigned_ids)
        device_id = current_platform.logical_device_id_to_visible_device_id(0)
    else:
        device_id = torch.npu.current_device()
    torch.npu.set_device(device_id)


class Backend(ABC):
    store: Any | None = None
    # Whether the connector must filter existing keys before calling put().
    requires_exists_before_put: bool = True

    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        pass

    @classmethod
    def create_scheduler_client(cls, parallel_config: ParallelConfig):
        return cls(parallel_config)

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return self.exists(keys)

    def batch_get_key_info(self, keys: list[str]):
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_key_info")

    def batch_alloc(self, keys: list[str], sizes: list[int], lease_ttl_ms: int = 0) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_alloc")

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_add_lease")

    def batch_remove_lease(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_remove_lease")

    def batch_write_finish(self, keys: list[str], results: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_write_finish")

    def validate_layerwise_support(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support block-key layerwise transfer")

    def batch_put_start(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_put_start")

    def batch_copy_put(
        self,
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
    ) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_copy_put")

    def batch_commit(self, keys: list[str]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_commit")

    def batch_revoke(self, keys: list[str]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_revoke")

    def batch_get_start(self, keys: list[str]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_start")

    def batch_copy_get(
        self,
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_copy_get")

    def batch_get_end(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_get_end")

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
