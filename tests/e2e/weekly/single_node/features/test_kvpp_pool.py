# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.kvpp import MODEL, PROMPTS, complete, output_texts, server_args
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMemcacheManager

pytestmark = pytest.mark.e2e_model(MODEL)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_memcache_reload(tmp_path):
    pytest.importorskip("memcache_hybrid")
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
                "ock.mmc.local_service.world_size": 2,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "1GB",
            },
        },
    )
    with SingleNodeMemcacheManager(config, tmp_path.name) as pool:
        port = get_open_port()
        args = server_args() + [
            "--port",
            str(port),
            "--additional-config",
            '{"enable_kvpp":true}',
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
        # KV pool is off the V2 blacklist, so this case defaults to V2 without
        # an explicit runner env pin.
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={**pool.server_envs, "VLLM_SERVER_DEV_MODE": "1"},
        ) as server:
            # Use a prompt spanning cache blocks so the replay exercises pool loading.
            prompt = PROMPTS[1]
            expected = output_texts(complete(server.url_root, prompt))
            requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
            assert output_texts(complete(server.url_root, [prompt, prompt])) == expected * 2
            metrics = requests.get(server.url_for("metrics"), timeout=30)
            metrics.raise_for_status()
            loaded_keys = sum(
                float(line.split()[-1])
                for line in metrics.text.splitlines()
                if line.startswith("vllm:ascend_store_load_get_keys_total{")
            )
            assert loaded_keys > 0
