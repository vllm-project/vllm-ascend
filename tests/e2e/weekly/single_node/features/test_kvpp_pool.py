# SPDX-License-Identifier: Apache-2.0
import json
import time

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.kvpp import MODEL, PROMPTS, complete, output_texts, server_args
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager

pytestmark = pytest.mark.e2e_model(MODEL)

POLL_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.2


def _reset_prefix_cache_after_save(server):
    # A completed HTTP request may still have an asynchronous pool save in
    # flight. The reset succeeds only after those block references are released.
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.post(server.url_for("reset_prefix_cache"), timeout=POLL_TIMEOUT_SECONDS)
        response.raise_for_status()
        if response.json()["success"]:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail("Prefix cache reset did not succeed after waiting for pool saves")


def _loaded_keys(server):
    response = requests.get(server.url_for("metrics"), timeout=POLL_TIMEOUT_SECONDS)
    response.raise_for_status()
    return sum(
        float(line.split()[-1])
        for line in response.text.splitlines()
        if line.startswith("vllm:ascend_store_load_get_keys_total{")
    )


def _wait_for_loaded_keys(server, previous):
    # Connector metrics reach the API process asynchronously.
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    loaded_keys = previous
    while time.monotonic() < deadline:
        loaded_keys = _loaded_keys(server)
        if loaded_keys > previous:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"Replay did not load new pool keys: before={previous}, after={loaded_keys}")


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@pytest.mark.parametrize(
    "use_v2_runner,pcp_size,enable_kvpp",
    [
        pytest.param(False, 1, True, id="mrv1-pcp1-kvpp"),
        pytest.param(True, 1, False, id="mrv2-pcp1-replicated"),
        pytest.param(True, 1, True, id="mrv2-pcp1-kvpp"),
        pytest.param(True, 2, False, id="mrv2-pcp2-replicated"),
        pytest.param(True, 2, True, id="mrv2-pcp2-kvpp"),
    ],
)
@wait_until_npu_memory_free()
def test_kvpp_memcache_reload(tmp_path, use_v2_runner, pcp_size, enable_kvpp):
    pytest.importorskip("memcache_hybrid")
    args = server_args()
    tp_size = int(args[args.index("--tensor-parallel-size") + 1])
    config = MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {
                "ock.mmc.log_level": "info",
                "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
            },
            "local": {
                "ock.mmc.log_level": "info",
                "ock.mmc.local_service.world_size": tp_size * pcp_size,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "1GB",
            },
        },
    )
    # Each parameter case starts its own store and cannot hit another layout's keys.
    with SingleNodeMemcacheManager(config, tmp_path.name) as pool:
        port = get_open_port()
        args += [
            "--port",
            str(port),
            "--prefill-context-parallel-size",
            str(pcp_size),
            "--additional-config",
            json.dumps({"enable_kvpp": enable_kvpp}),
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "0",
                        "backend": "memcache",
                        "use_layerwise": False,
                        "load_async": True,
                    },
                }
            ),
        ]
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={
                **pool.server_envs,
                "VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2_runner)),
                "VLLM_SERVER_DEV_MODE": "1",
            },
        ) as server:
            # Use a prompt spanning cache blocks so the replay exercises pool loading.
            prompt = PROMPTS[1]
            expected = output_texts(complete(server.url_root, prompt))
            _reset_prefix_cache_after_save(server)
            loaded_keys_before = _loaded_keys(server)
            assert output_texts(complete(server.url_root, [prompt, prompt])) == expected * 2
            _wait_for_loaded_keys(server, loaded_keys_before)
