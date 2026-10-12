# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kvpp import MODEL, PROMPTS, complete, output_texts, server_args
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

pytestmark = pytest.mark.e2e_model(MODEL)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP,PP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_pipeline_parallel():
    results = []
    for enabled in (False, True):
        port = get_open_port()
        args = server_args() + [
            "--port",
            str(port),
            "--pipeline-parallel-size",
            "2",
            "--additional-config",
            json.dumps({"enable_kvpp": enabled}),
        ]
        with RemoteOpenAIServer(
            maybe_model_redirect(MODEL),
            args,
            server_port=port,
            auto_port=False,
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
        ) as server:
            results.append(
                [
                    output_texts(complete(server.url_root, PROMPTS)),
                    output_texts(complete(server.url_root, PROMPTS[1])),
                ]
            )
    assert results[0] == results[1]
