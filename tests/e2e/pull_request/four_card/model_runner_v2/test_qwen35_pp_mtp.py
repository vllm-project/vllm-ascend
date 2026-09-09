# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import regex as re
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

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
def test_qwen35_pp_mtp_full_decode_only() -> None:
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
        json.dumps({"enable_cpu_binding": True}),
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

        def check_answer(question: str, expected: int):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=0,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            choice = response.choices[0]
            assert choice.finish_reason == "stop", choice
            text = choice.message.content or ""
            numbers = re.findall(r"-?\d+", text.rsplit("</think>", 1)[-1])
            assert numbers and int(numbers[-1]) == expected, text
            return response

        for expression, answer in [("19 times 23", 437), ("125 plus 378", 503), ("144 divided by 12", 12)]:
            check_answer(f"What is {expression}? Check the arithmetic and give the final integer.", answer)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    check_answer,
                    f"What is {19 + i} times {23 + i}? Check the arithmetic and give the final integer.",
                    (19 + i) * (23 + i),
                )
                for i in range(32)
            ]
            for future in futures:
                future.result()

        # Force a second prefill chunk before exercising decode graph replay.
        padding = "This paragraph is padding for a long context test. " * 2000
        response = check_answer(
            padding
            + "\nIgnore the padding above. What is 125 plus 378? Check the arithmetic and give the final integer.",
            503,
        )
        assert response.usage.prompt_tokens > MAX_BATCHED_TOKENS
