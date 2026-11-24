# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary probe: memcache init with MF_HYBM_ENABLE_4K_PAGE=1.

MemFabric documents that machines without hugepages must export
MF_HYBM_ENABLE_4K_PAGE=1 or smem_bm_create fails. The pod reports 0 hugepages.
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

PROBE_TIMEOUT_SECONDS = 120
PROBE_DRAM_SIZE = "1GB"
PROBE_ENVS = {
    "MF_HYBM_ENABLE_4K_PAGE": "1",
    "MF_GROUP_JOIN_MAX_TIMEOUT": "60",
    "ASCEND_CONNECT_TIMEOUT": "60000",
    "ASCEND_TRANSFER_TIMEOUT": "60000",
}

PROBE_SCRIPT = """
import subprocess

import torch
import torch_npu
from memcache_hybrid import DistributedObjectStore

hp = subprocess.run(["bash", "-lc", "grep -E 'HugePages_Total|Hugepagesize' /proc/meminfo"], capture_output=True, text=True)
print(f"probe: hugepages {hp.stdout.strip()!r}", flush=True)
hal = "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so"
cmd = "nm -D " + hal + " | grep -i drvNotifyIdAddrOffset"
nm = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(f"probe: hal_symbol rc={nm.returncode} out={nm.stdout.strip()!r}", flush=True)
torch.npu.set_device(0)
store = DistributedObjectStore()
print("probe: init start", flush=True)
rc = store.init(0, init_bm=True)
print(f"probe: init rc={rc}", flush=True)
store.close()
print("probe: done", flush=True)
"""


def _build_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {
                "ock.mmc.log_level": "info",
                "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
            },
            "local": {
                "ock.mmc.log_level": "info",
                "ock.mmc.local_service.world_size": 1,
                "ock.mmc.local_service.protocol": "device_urma",
                "ock.mmc.local_service.dram.size": PROBE_DRAM_SIZE,
            },
        },
    )


def test_memcache_4k_page_probe() -> None:
    started = time.perf_counter()
    with SingleNodeMemcacheManager(_build_config(), "probe_4k") as pool:
        with patch.dict(os.environ, {**pool.server_envs, **PROBE_ENVS}):
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", PROBE_SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                text = completed.stdout.strip()
                outcome = f"rc={completed.returncode}"
            except subprocess.TimeoutExpired as exc:
                text = exc.stdout or exc.stderr or ""
                if isinstance(text, bytes):
                    text = text.decode(errors="replace")
                text = text.strip()
                outcome = "TIMEOUT"
            elapsed = time.perf_counter() - started
            print(
                f"[pool_probe] 4k_page {outcome} elapsed={elapsed:.1f}s "
                f"head={text[:1200]!r} tail={text[-1200:]!r}",
                flush=True,
            )
