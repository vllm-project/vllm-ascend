# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 owner-sharded Mooncake write/reload with layerwise transfers."""

import json

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.kvpp import PROMPTS, complete, output_texts, server_args
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "Eco-Tech/GLM-5.2-w4a8"
TP_SIZE = 8

pytestmark = pytest.mark.e2e_model(MODEL)


def external_cache_hits(server):
    response = requests.get(server.url_for("metrics"), timeout=30)
    response.raise_for_status()
    return sum(
        float(line.split()[-1])
        for line in response.text.splitlines()
        if line.startswith("vllm:external_prefix_cache_hits_total{")
    )


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@pytest.mark.parametrize("use_mtp", [False, True])
@wait_until_npu_memory_free()
def test_kvpp_mooncake_layerwise_reload(tmp_path, use_mtp):
    pytest.importorskip("mooncake.store")
    pool_config = MooncakeKVPoolConfig(
        master_port=get_open_port(),
        metrics_port=get_open_port(),
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "global_segment_size": 1 << 30,
            "local_buffer_size": 1 << 30,
        },
    )
    with SingleNodeMooncakeManager(pool_config, tmp_path.name) as pool:
        port = get_open_port()
        args = server_args(tp_size=TP_SIZE, gpu_memory_utilization=0.9) + [
            "--port",
            str(port),
            "--additional-config",
            '{"enable_kvpp":true}',
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "0",
                        "backend": "mooncake",
                        "use_layerwise": True,
                        "load_async": False,
                    },
                }
            ),
        ]
        if use_mtp:
            args += ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={**pool.server_envs, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_SERVER_DEV_MODE": "1"},
        ) as server:
            # 128-token chunks span multiple cache blocks and exercise later
            # forwards with history, including shared-index SFA layers.
            prompt = PROMPTS[1]
            expected = output_texts(complete(server.url_root, prompt))
            hits_before = external_cache_hits(server)
            requests.post(server.url_for("reset_prefix_cache"), timeout=30).raise_for_status()
            assert output_texts(complete(server.url_root, [prompt, prompt])) == expected * 2
            assert external_cache_hits(server) > hits_before
