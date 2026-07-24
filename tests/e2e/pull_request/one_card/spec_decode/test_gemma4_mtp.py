# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP speculative decoding acceptance test.

Run: pytest tests/e2e/pull_request/one_card/spec_decode/test_gemma4_mtp.py
"""

import os

from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAIN_MODEL = "google/gemma-4-31B-it"
DRAFT_MODEL = "google/gemma-4-31B-it-assistant"


def test_gemma4_mtp_acceptance():
    golden = [0.70, 0.45, 0.25]

    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    max_tokens = 256

    with VllmRunner(
        MAIN_MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        disable_log_stats=False,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "model": DRAFT_MODEL,
        },
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[12],
        ),
    ) as spec_vllm_model:
        _ = spec_vllm_model.generate_greedy(example_prompts, max_tokens)
        metrics = spec_vllm_model.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * 3
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    assert num_drafts > 0, "No spec decode drafts were generated"
    acceptance_per_pos = [n / num_drafts for n in num_accepted_tokens_per_pos]

    match = all((a >= b) or (b - a < 0.10) for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden} (num_drafts={num_drafts})"
    cleanup_dist_env_and_memory()
