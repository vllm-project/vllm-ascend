import json
import os
from typing import Any

import openai
import pytest
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases

MODELS = [
    "vllm-ascend/Qwen3-Next-80B-A3B-Instruct",
]

MODES = ["aclgraph"]

TENSOR_PARALLELS = [4]
MAX_NUM_BATCHED_TOKENS = [1024, 4096, 8192, 32768]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}

batch_size_dict = {
    "linux-aarch64-a2-4": 64,
    "linux-aarch64-a3-4": 64,
}
VLLM_CI_RUNNER = os.getenv("VLLM_CI_RUNNER", "linux-aarch64-a2-4")
performance_batch_size = batch_size_dict.get(VLLM_CI_RUNNER, 1)

aisbench_cases = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 4 * performance_batch_size,
    "max_out_len": 1500,
    "batch_size": performance_batch_size,
    "baseline": 1,
    "threshold": 0.97
}, {
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/gsm8k-lite",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
    "max_out_len": 32768,
    "batch_size": 32,
    "top_k": 20,
    "baseline": 95,
    "threshold": 5
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("max_num_batched_tokens", MAX_NUM_BATCHED_TOKENS)
async def test_models(model: str, mode: str, tp_size: int,
                      max_num_batched_tokens: int) -> None:
    port = get_open_port()
    env_dict = {
        "OMP_NUM_THREADS": "10",
        "OMP_PROC_BIND": "false",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    compilation_config = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    server_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--port",
        str(port),
        "--max-model-len",
        "40960",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.8",
        "--max-num-seqs",
        "64",
    ]
    if mode == "aclgraph":
        server_args.extend(
            ["--compilation-config",
             json.dumps(compilation_config)])
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False) as server:
        client = server.get_async_client()
        batch = await client.completions.create(
            model=model,
            prompt=prompts,
            **request_keyword_args,
        )
        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "empty response"
        print(choices)
        if mode == "single":
            return
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
