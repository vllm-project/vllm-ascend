# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary probe: which memcache local_service protocol can init and register.

Not part of the curated suite. Run explicitly with /e2e while bringing up the
A5 pool cases; delete once the working protocol is known.
"""

import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager

PROTOCOLS = ["device_uboe", "device_urma", "host_shm", "host_tcp"]
PROBE_TIMEOUT_SECONDS = 180
PROBE_DRAM_SIZE = "1GB"

PROBE_SCRIPT = """
import torch
import torch_npu
from memcache_hybrid import DistributedObjectStore

torch.npu.set_device(0)
store = DistributedObjectStore()
print("probe: init start", flush=True)
rc = store.init(0, init_bm=True)
print(f"probe: init rc={rc}", flush=True)
buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="npu")
print("probe: register start", flush=True)
rc = store.register_buffer(buffer.data_ptr(), buffer.numel())
print(f"probe: register rc={rc}", flush=True)
print(f"probe: unregister rc={store.unregister_buffer(buffer.data_ptr(), buffer.numel())}", flush=True)
store.close()
print("probe: done", flush=True)
"""


def _build_config(protocol: str) -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": protocol,
                "ock.mmc.local_service.dram.size": PROBE_DRAM_SIZE,
            },
        },
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_memcache_protocol_probe(protocol: str) -> None:
    started = time.perf_counter()
    with SingleNodeMemcacheManager(_build_config(protocol), f"probe_{protocol}") as pool:
        with patch.dict(os.environ, pool.server_envs):
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", PROBE_SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                elapsed = time.perf_counter() - started
                print(
                    f"[pool_probe] protocol={protocol} rc={completed.returncode} "
                    f"elapsed={elapsed:.1f}s stdout={completed.stdout.strip()!r} "
                    f"stderr={completed.stderr.strip()[-2000:]!r}",
                    flush=True,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                partial = exc.stdout or exc.stderr or ""
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(
                    f"[pool_probe] protocol={protocol} TIMEOUT after {elapsed:.1f}s "
                    f"stdout={partial.strip()!r}",
                    flush=True,
                )
