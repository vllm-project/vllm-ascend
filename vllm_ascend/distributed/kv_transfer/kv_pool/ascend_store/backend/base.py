from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig

# QoS range supported by the pooled KV store backends. A larger value
# means a higher transfer priority.
QOS_VALUE_MIN = 0
QOS_VALUE_MAX = 4


def parse_qos_from_extra_config(extra_config: dict[str, Any] | None) -> int | None:
    """Parse and validate the ``qos`` field of kv_connector_extra_config.

    Returns None when the field is absent; otherwise the QoS integer in
    [QOS_VALUE_MIN, QOS_VALUE_MAX]. Only integers are supported; an invalid
    value fails fast with a clear error instead of an obscure failure inside
    the store backends.
    """
    if not extra_config or "qos" not in extra_config:
        return None
    qos = extra_config["qos"]
    if isinstance(qos, bool) or not isinstance(qos, int) or not (QOS_VALUE_MIN <= qos <= QOS_VALUE_MAX):
        raise ValueError(
            f"Invalid qos {qos!r} in kv_connector_extra_config: "
            f"QoS must be an integer in [{QOS_VALUE_MIN}, {QOS_VALUE_MAX}]."
        )
    return qos


def fetch_qos_from_current_config() -> int | None:
    """Fetch the ``qos`` field of kv_connector_extra_config from the current
    vLLM config.

    The store backends call this inside their QoS injection helpers so no QoS
    needs to be passed through ``__init__``. Returns None when no current
    config is available in this process (e.g. standalone store usage) or when
    the field is absent; an invalid value still fails fast via
    ``parse_qos_from_extra_config``.
    """
    from vllm.config import get_current_vllm_config

    try:
        vllm_config = get_current_vllm_config()
    except AssertionError:
        # vLLM >= 0.27.1 (main) raises when no config has been set in this
        # process (e.g. stores used outside the vLLM engine).
        vllm_config = None
    if vllm_config is None:
        return None
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    return parse_qos_from_extra_config(extra_config)


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
