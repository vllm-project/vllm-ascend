import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            "NONE",
            marks=pytest.mark.e2e_coverage(
                arch="dense",
                feature="prefix_caching,chunked_prefill",
                parallel="TP",
                deploy="pd_mix",
                hardware="A2",
                quantization="BF16",
                graph_mode="eager",
            ),
        ),
        pytest.param(
            "FULL",
            marks=pytest.mark.e2e_coverage(
                arch="dense",
                feature="prefix_caching,chunked_prefill",
                parallel="TP",
                deploy="pd_mix",
                hardware="A2",
                quantization="BF16",
                graph_mode="full_graph",
            ),
        ),
    ],
)
@wait_until_npu_memory_free()
def test_guard_preserves_sequential_and_concurrent_generation(use_v2, graph_mode):
    model = "Qwen/Qwen3-0.6B"
    prompt = "Explain how a key value cache works. " * 40
    arguments = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "256",
        "--block-size",
        "128",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.3",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--compilation-config",
        json.dumps({"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [1, 2, 4]}),
    ]
    if graph_mode == "NONE":
        arguments.append("--enforce-eager")
    results = []
    for enabled in (False, True):
        port = get_open_port()
        environment = {
            "VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2)),
            "VLLM_ASCEND_KV_GUARD": str(int(enabled)),
            "VLLM_ASCEND_KV_GUARD_ACTION": "raise",
        }
        with (
            RemoteOpenAIServer(
                model,
                [*arguments, "--port", str(port)],
                env_dict=environment,
                seed=1234,
                server_host="127.0.0.1",
                server_port=port,
                auto_port=False,
            ) as server,
            server.get_client() as client,
        ):

            def generate(request_index):
                response = client.completions.create(model=model, prompt=prompt, temperature=0, max_tokens=8, seed=1234)
                assert response.choices[0].text
                return response.choices[0].text

            sequential = [generate(request_index) for request_index in range(2)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                concurrent = list(executor.map(generate, range(4)))
            results.append(sequential + concurrent)
    assert results[0] == results[1]
