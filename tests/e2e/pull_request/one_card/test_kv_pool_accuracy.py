# SPDX-License-Identifier: Apache-2.0
"""PR-level accuracy guard for the AscendStore KV pool modes."""

import json

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Explain why keeping a record of important events can be useful."

pytestmark = pytest.mark.e2e_model(MODEL)


def _server_args(port: int, kv_transfer_config: dict | None = None) -> list[str]:
    args = [
        "--trust-remote-code",
        "--served-model-name",
        "kv-pool-pr-test",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.7",
        "--seed",
        "42",
        "--enforce-eager",
        "--port",
        str(port),
    ]
    if kv_transfer_config is not None:
        args.extend(["--kv-transfer-config", json.dumps(kv_transfer_config)])
    return args


@pytest.mark.parametrize("use_layerwise", [False, True], ids=["non-layerwise", "layerwise"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_memcache_kv_pool_matches_uncached_output(tmp_path, use_layerwise: bool) -> None:
    """Generated text must match the uncached baseline with the pool enabled."""
    pytest.importorskip("memcache_hybrid")
    baseline_port = get_open_port()
    with RemoteOpenAIServer(
        maybe_model_redirect(MODEL),
        _server_args(baseline_port),
        server_port=baseline_port,
        auto_port=False,
    ) as server:
        baseline_response = requests.post(
            server.url_root + "/v1/completions",
            json={"model": "kv-pool-pr-test", "prompt": PROMPT, "temperature": 0, "max_tokens": 32},
            timeout=180,
        )
        baseline_response.raise_for_status()
        expected = baseline_response.json()["choices"][0]["text"]

    pool_config = MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 1,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "256MB",
            },
        },
    )
    with SingleNodeMemcacheManager(pool_config, f"{tmp_path.name}-{use_layerwise}") as pool:
        port = get_open_port()
        transfer_config = {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lookup_rpc_port": "0",
                "backend": "memcache",
                "use_layerwise": use_layerwise,
            },
        }
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            _server_args(port, transfer_config),
            server_port=port,
            auto_port=False,
            env_dict=pool.server_envs,
        ) as server:
            response = requests.post(
                server.url_root + "/v1/completions",
                json={"model": "kv-pool-pr-test", "prompt": PROMPT, "temperature": 0, "max_tokens": 32},
                timeout=180,
            )
            response.raise_for_status()
            assert response.json()["choices"][0]["text"] == expected
