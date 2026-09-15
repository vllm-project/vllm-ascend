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


def metric_total(metrics_text: str, name: str) -> float:
    return sum(
        float(line.split()[-1]) for line in metrics_text.splitlines() if line.startswith((f"{name}{{", f"{name} "))
    )


def metric_delta(before_text: str, after_text: str, name: str) -> float:
    return metric_total(after_text, name) - metric_total(before_text, name)


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
def test_kvpp_memcache_mixed_hbm_external_hit(tmp_path):
    pytest.importorskip("memcache_hybrid")
    block_size = 128
    results = {}
    for lookup_hash_mode in ("full", "suffix"):
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
        with SingleNodeMemcacheManager(config, f"{tmp_path.name}-{lookup_hash_mode}") as pool:
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
                            "lookup_hash_mode": lookup_hash_mode,
                        },
                    }
                ),
            ]
            with RemoteOpenAIServer(
                maybe_model_redirect(MODEL),
                args,
                server_port=port,
                auto_port=False,
                env_dict={**pool.server_envs, "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_SERVER_DEV_MODE": "1"},
            ) as server:
                tokenized = requests.post(
                    server.url_for("tokenize"),
                    json={"model": "kvpp-test", "prompt": PROMPTS[1]},
                    timeout=30,
                )
                tokenized.raise_for_status()
                long_prompt = tokenized.json()["tokens"][: 3 * block_size]
                assert len(long_prompt) == 3 * block_size
                short_prefix = long_prompt[:block_size]

                # Persist the complete prompt externally, then retain only its
                # first block in HBM. FULL and SUFFIX must locate and load the
                # same remaining KV despite sending different hash payloads.
                complete(server.url_root, long_prompt)
                requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
                complete(server.url_root, short_prefix)

                before = requests.get(server.url_for("metrics"), timeout=30)
                before.raise_for_status()
                outputs = output_texts(complete(server.url_root, long_prompt))
                after = requests.get(server.url_for("metrics"), timeout=30)
                after.raise_for_status()

                prefix_hits = metric_delta(before.text, after.text, "vllm:prefix_cache_hits_total")
                external_hits = metric_delta(before.text, after.text, "vllm:external_prefix_cache_hits_total")
                load_keys = metric_delta(before.text, after.text, "vllm:ascend_store_load_get_keys_total")
                sent_hashes = metric_delta(before.text, after.text, "vllm:ascend_store_lookup_hashes_sent_total")
                omitted_hashes = metric_delta(
                    before.text,
                    after.text,
                    "vllm:ascend_store_lookup_hashes_omitted_total",
                )
                assert prefix_hits >= block_size
                assert external_hits > 0
                assert load_keys > 0
                if lookup_hash_mode == "full":
                    assert (sent_hashes, omitted_hashes) == (3, 0)
                else:
                    assert (sent_hashes, omitted_hashes) == (2, 1)
                results[lookup_hash_mode] = (outputs, prefix_hits, external_hits, load_keys)

    assert results["full"] == results["suffix"]
