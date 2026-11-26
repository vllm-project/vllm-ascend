# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.common.multi_node.utils import get_available_port
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MODEL = "wemaster/deepseek_mtp_main_random_bf16"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="multi_instance",
    parallel="DP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_both_dp_children_serve_with_pcp():
    # The supervisor assigns consecutive HTTP ports to its children.
    first_port = get_available_port()
    while (second_port := get_available_port(first_port + 1)) != first_port + 1:
        first_port = second_port
    supervisor_port = get_available_port(second_port + 1)
    rpc_port = get_available_port(supervisor_port + 1)

    model = maybe_model_redirect(MODEL)
    host = "127.0.0.1"
    args = [
        "--host",
        host,
        "--port",
        str(first_port),
        "--data-parallel-multi-port-external-lb",
        "--data-parallel-supervisor-port",
        str(supervisor_port),
        "--data-parallel-rpc-port",
        str(rpc_port),
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "2",
        "--prefill-context-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--enable-expert-parallel",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "128",
    ]
    # The supervisor exposes /health after both child engines are ready.
    with RemoteOpenAIServer(
        model,
        args,
        server_host=host,
        server_port=supervisor_port,
        auto_port=False,
        env_dict={"HCCL_BUFFSIZE": "1024"},
        max_wait_seconds=600,
    ):
        for port in (first_port, second_port):
            response = requests.post(
                f"http://{host}:{port}/v1/completions",
                json={"model": model, "prompt": "The capital of France is", "max_tokens": 1, "ignore_eos": True},
                timeout=180,
            )
            response.raise_for_status()
            assert response.json()["usage"]["completion_tokens"] == 1
