# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary probe: host-side memcache protocols plus the HAL symbol check.

Device protocols block after SMEM inside the HYBM GVA path, whose precheck
reports a missing ``drvNotifyIdAddrOffset`` in libascend_hal.so. Test the two
host protocols and print the exported-symbol result plus the head of the log.
Delete this file once the cause is known.
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

CASES = ["host_rdma", "host_urma"]
PROBE_TIMEOUT_SECONDS = 120
PROBE_DRAM_SIZE = "1GB"
PROBE_ENVS = {
    "MF_GROUP_JOIN_MAX_TIMEOUT": "60",
    "MF_QP_READY_CHECK_TIMEOUT_BASE": "30",
    "ASCEND_CONNECT_TIMEOUT": "60000",
    "ASCEND_TRANSFER_TIMEOUT": "60000",
}
HEAD_CHARS = 1500
TAIL_CHARS = 1500

PROBE_SCRIPT = """
import subprocess

import torch
import torch_npu
from memcache_hybrid import DistributedObjectStore

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


def _build_config(protocol: str) -> MemcacheKVPoolConfig:
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
                "ock.mmc.local_service.protocol": protocol,
                "ock.mmc.local_service.dram.size": PROBE_DRAM_SIZE,
            },
        },
    )


@pytest.mark.parametrize("protocol", CASES)
def test_memcache_host_protocol_probe(protocol: str) -> None:
    started = time.perf_counter()
    with SingleNodeMemcacheManager(_build_config(protocol), f"probe_host_{protocol}") as pool:
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
                f"[pool_probe] protocol={protocol} {outcome} elapsed={elapsed:.1f}s "
                f"head={text[:HEAD_CHARS]!r} tail={text[-TAIL_CHARS:]!r}",
                flush=True,
            )
