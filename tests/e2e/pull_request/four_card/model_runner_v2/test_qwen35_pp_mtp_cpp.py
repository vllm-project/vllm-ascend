# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPP-enabled variant of test_qwen35_pp_mtp.py.

This file is the chunk pipeline parallel (profiling_chunk_config) companion
to tests/e2e/pull_request/four_card/model_runner_v2/test_qwen35_pp_mtp.py.
The two files are intentionally kept separate so the base MTP e2e case and
the CPP e2e case can fail independently without affecting each other.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from vllm.utils.network_utils import get_open_port

MODEL = os.environ.get("QWEN35_DENSE_MODEL", "Qwen/Qwen3.5-27B")
MAX_BATCHED_TOKENS = 16384


@pytest.mark.e2e_model("Qwen/Qwen3.5-27B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,aclgraph,chunked_prefill,long_sequence",
    parallel="PP,TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_qwen35_pp_mtp_cpp_full_decode_only() -> None:
    port = get_open_port()
    server_args = [
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--seed",
        "1024",
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        str(MAX_BATCHED_TOKENS),
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.88",
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True}),
        "--compilation-config",
        json.dumps({"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--additional-config",
        json.dumps({
            "enable_cpu_binding": True,
            "scheduler_config": {"profiling_chunk_config": {"enabled": True}},
        }),
        "--port",
        str(port),
    ]
    with (
        RemoteOpenAIServer(
            MODEL,
            server_args,
            server_host="127.0.0.1",
            server_port=port,
            auto_port=False,
            max_wait_seconds=600,
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
        ) as server,
        server.get_client(timeout=180) as client,
    ):

        def run_prefill(question: str):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=0,
                max_tokens=1,
                extra_body={
                    "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            assert response.usage is not None
            assert response.usage.completion_tokens == 1
            return response

        for expression in ["6 plus 7", "3 times 4", "9 minus 5"]:
            run_prefill(f"What is {expression}? Give the final integer.")

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    run_prefill,
                    f"What is {3 + i % 6} plus {4 + i % 5}? Give the final integer.",
                )
                for i in range(32)
            ]
            for future in futures:
                future.result()

        # Force a second prefill chunk and return exactly one output token.
        padding = "This paragraph is padding for a long context test. " * 2000
        response = run_prefill(
            padding + "\nIgnore the padding above. What is 6 plus 7? Give the final integer."
        )
        assert response.usage.prompt_tokens > MAX_BATCHED_TOKENS
