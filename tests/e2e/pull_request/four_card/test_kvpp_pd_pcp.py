# SPDX-License-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""PD disaggregation with prefill context parallelism on the prefill engine.

PCP replicates KV across prefill ranks; the transfer path must select one
complete replica per request (hash-stable on both engines). The prefill engine
runs TP1+PCP2 and the decode engine TP2 on four cards.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kvpp import MODEL, PROMPTS, complete, output_texts, server_args
from tests.e2e.conftest import RemotePDServer, wait_until_npu_memory_free

pytestmark = pytest.mark.e2e_model(MODEL)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP,EP,PCP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_pd_pcp_disaggregation():
    model = maybe_model_redirect(MODEL)
    p_port, d_port = get_open_port(), get_open_port()
    p_url, d_url = f"http://127.0.0.1:{p_port}", f"http://127.0.0.1:{d_port}"
    servers = [
        [
            model,
            # P: TP1 + PCP2 (two cards); D: TP2 (two cards).
            *(server_args(tp_size=1) if role == "kv_producer" else server_args()),
            "--port",
            str(port),
            "--additional-config",
            json.dumps({"enable_kvpp": enabled}),
            "--prefill-context-parallel-size",
            "2" if role == "kv_producer" else "1",
            "--kv-transfer-config",
            json.dumps({"kv_connector": "MooncakeConnectorV2", "kv_role": role, "kv_port": kv_port}),
        ]
        for port, enabled, role, kv_port in (
            (p_port, True, "kv_producer", 26810),
            (d_port, False, "kv_consumer", 27010),
        )
    ]
    with RemotePDServer(servers, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_SERVER_DEV_MODE": "1"}):
        expected = [output_texts(complete(p_url, prompt)) for prompt in PROMPTS]
        requests.post(p_url + "/reset_prefix_cache", timeout=30).raise_for_status()

        def generate(prompt):
            prefill = complete(p_url, prompt, max_tokens=1, min_tokens=1, kv_transfer_params={"do_remote_decode": True})
            return output_texts(complete(d_url, prompt, kv_transfer_params=prefill["kv_transfer_params"]))

        assert [generate(prompt) for prompt in PROMPTS] == expected
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(generate, PROMPTS)) == expected
