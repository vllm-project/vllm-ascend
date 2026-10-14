# SPDX-License-Identifier: Apache-2.0
"""Initialize upstream Ascend engines under a shared, startup-only lock.

Both pools must use this adapter. Native construction and initialize() can read
the process environment, so the lock covers both, including configuration reads.
Requests never enter this scope. Unrelated native engine initialization must not
run concurrently with QoS pool construction in the same worker.
"""

import copy
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from vllm_ascend import envs

# A single lock for PD and Store: separate per-pool locks cannot protect environ.
INIT_LOCK = RLock()
RESOURCE_CONFIG_KEY = "transports/ascend_direct/global_resource_config"


class TransferEngine(Protocol):
    def initialize(self, host: str, metadata: str, protocol: str, device: str) -> int: ...
    def get_rpc_port(self) -> int: ...
    def get_engine(self) -> Any: ...
    def register_memory(self, ptr: int, size: int) -> int: ...
    def unregister_memory(self, ptr: int) -> int: ...
    def batch_transfer_sync_read(
        self, session: str, local: Sequence[int], remote: Sequence[int], sizes: Sequence[int]
    ) -> int: ...
    def batch_transfer_sync_write(
        self, session: str, local: Sequence[int], remote: Sequence[int], sizes: Sequence[int]
    ) -> int: ...


@lru_cache(maxsize=1)
def _config_directory() -> tempfile.TemporaryDirectory[str]:
    # Retain snapshots for the worker lifetime, even if a caller retains an
    # engine beyond pool.close(). TemporaryDirectory removes them at normal
    # process exit. Only engine construction allocates a file, never a request.
    return tempfile.TemporaryDirectory(prefix="vllm-ascend-qos-")


def _derive_config(original: Any, resource: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(original, dict):
        raise ValueError("TENT configuration must be an object")
    result = copy.deepcopy(original)
    result.pop(RESOURCE_CONFIG_KEY, None)
    transports = result.setdefault("transports", {})
    if not isinstance(transports, dict):
        raise ValueError("TENT transports must be an object")
    direct = transports.setdefault("ascend_direct", {})
    if not isinstance(direct, dict):
        raise ValueError("TENT ascend_direct must be an object")
    direct["global_resource_config"] = json.dumps(resource, separators=(",", ":"))
    return result


def initialize_engine(
    factory: Callable[[], TransferEngine],
    host: str,
    device_name: str,
    qos: int,
    resource_config: Mapping[str, Any] | None = None,
) -> TransferEngine:
    """Create one lane using upstream initialize(host, metadata, protocol, device)."""
    if type(qos) is not int or not 0 <= qos <= 7:
        raise ValueError("QoS must be an integer in 0..7")
    resource_env = envs.MOONCAKE_ASCEND_RESOURCE_ENV
    tent_env = envs.MOONCAKE_TENT_CONFIG_ENV
    with INIT_LOCK:
        previous = {name: os.environ.get(name) for name in (resource_env, tent_env)}
        base = json.loads(previous[resource_env] or "{}") if resource_config is None else dict(resource_config)
        if not isinstance(base, dict) or "store" in base or "comm_resource_config" in base:
            raise ValueError("resource_config requires literal flat comm_resource_config.* keys")
        resource = dict(base, **{"comm_resource_config.qos": qos})
        original_path = previous[tent_env]
        original = json.loads(Path(original_path).read_text()) if original_path else {}
        derived = _derive_config(original, resource)
        fd, name = tempfile.mkstemp(prefix=f"tent-q{qos}-", suffix=".json", dir=_config_directory().name)
        with os.fdopen(fd, "w") as stream:
            json.dump(derived, stream)
        try:
            os.environ[resource_env] = json.dumps(resource, separators=(",", ":"))
            os.environ[tent_env] = name
            engine = factory()
            rc = engine.initialize(host, "P2PHANDSHAKE", "ascend", device_name or "")
            if rc != 0:
                raise RuntimeError(f"QoS {qos}: upstream engine initialization failed: {rc}")
            return engine
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
