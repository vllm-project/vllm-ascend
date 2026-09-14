# SPDX-License-Identifier: Apache-2.0
"""Inputs and serving arguments shared by KVPP integration tests."""

import requests

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
PROMPTS = [
    "The capital of France is",
    "This is background information. " * 80 + "The capital of France is",
]


def server_args():
    return [
        "--served-model-name",
        "kvpp-test",
        "--trust-remote-code",
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--async-scheduling",
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--block-size",
        "128",
        "--num-gpu-blocks-override",
        "64",
        "--gpu-memory-utilization",
        "0.8",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
    ]


def complete(url, prompt, **kwargs):
    response = requests.post(
        url + "/v1/completions",
        json={"model": "kvpp-test", "prompt": prompt, "temperature": 0, "max_tokens": 16, "ignore_eos": True, **kwargs},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def output_texts(result):
    assert all(choice["finish_reason"] == "length" for choice in result["choices"])
    return [choice["text"] for choice in result["choices"]]
