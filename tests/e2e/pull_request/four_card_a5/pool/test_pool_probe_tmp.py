# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Temporary probe: capture the memcache MetaService log while init blocks.

Protocol and world_size are already ruled out: every case reaches
``initialize store ... OK`` and then ``DistributedObjectStore.init`` never
returns. This run starts one LocalService, dumps the MetaService log on
timeout, and uses an explicit metrics port so a busy 8000 cannot hide the
failure. Delete this file once the cause is known.
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

PROBE_TIMEOUT_SECONDS = 180
PROBE_DRAM_SIZE = "1GB"
PROBE_ENVS = {
    "MF_GROUP_JOIN_MAX_TIMEOUT": "60",
    "MF_QP_READY_CHECK_TIMEOUT_BASE": "30",
    "ASCEND_CONNECT_TIMEOUT": "60000",
    "ASCEND_TRANSFER_TIMEOUT": "60000",
}
META_LOG_TAIL_CHARS = 6000

PROBE_SCRIPT = """
import subprocess

import torch
import torch_npu
from memcache_hybrid import DistributedObjectStore

hal = "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so"
cmd = "nm -D " + hal + " | grep -i drvNotifyIdAddrOffset"
nm = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(f"probe: hal_symbol rc={nm.returncode} out={nm.stdout.strip()!r} err={nm.stderr.strip()[-300:]!r}", flush=True)
torch.npu.set_device(0)
store = DistributedObjectStore()
print("probe: init start", flush=True)
rc = store.init(0, init_bm=True)
print(f"probe: init rc={rc}", flush=True)
store.close()
print("probe: done", flush=True)
"""


def _build_config() -> MemcacheKVPoolConfig:
    metrics_port = get_open_port()
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {
                "ock.mmc.log_level": "info",
                "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{metrics_port}",
            },
            "local": {
                "ock.mmc.log_level": "debug",
                "ock.mmc.local_service.world_size": 1,
                "ock.mmc.local_service.protocol": "device_urma",
                "ock.mmc.local_service.dram.size": PROBE_DRAM_SIZE,
            },
        },
    )


def _read_tail(path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-META_LOG_TAIL_CHARS:]
    except OSError as exc:
        return f"<cannot read {path}: {exc}>"


@pytest.mark.parametrize("round_index", [0])
def test_memcache_init_probe(round_index: int) -> None:
    started = time.perf_counter()
    with SingleNodeMemcacheManager(_build_config(), f"probe_init_{round_index}") as pool:
        with patch.dict(os.environ, {**pool.server_envs, **PROBE_ENVS}):
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", PROBE_SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                outcome = f"rc={completed.returncode} stdout={completed.stdout.strip()!r}"
            except subprocess.TimeoutExpired as exc:
                partial = exc.stdout or exc.stderr or ""
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                outcome = f"TIMEOUT stdout={partial.strip()[-4000:]!r}"
            elapsed = time.perf_counter() - started
            print(f"[pool_probe] case=init elapsed={elapsed:.1f}s {outcome}", flush=True)
            print(f"[pool_meta_log] path={pool.log_path} tail={_read_tail(pool.log_path)!r}", flush=True)
