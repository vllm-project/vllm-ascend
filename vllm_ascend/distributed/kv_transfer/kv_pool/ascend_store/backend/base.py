from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig

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

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_alloc")

    def batch_add_lease(self, keys: list[str], lease_ttl_ms: int = 0) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_add_lease")

    def batch_remove_lease(self, keys: list[str]) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_remove_lease")

    def batch_write_finish(self, keys: list[str], results: list[int]) -> list[int]:
        raise NotImplementedError(f"{type(self).__name__} does not support batch_write_finish")

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
