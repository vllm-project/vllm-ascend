"""Smoke-test Gemma4 hybrid-attention MTP with per-group KV metadata."""

import os
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

TARGET_MODEL = os.environ.get("GEMMA4_TARGET_MODEL", "google/gemma-4-31B-it")
DRAFT_MODEL = os.environ.get("GEMMA4_DRAFT_MODEL", "google/gemma-4-31B-it-assistant")

_GRAPH_ENV = {
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}


def _collect_acceptance_rates(vllm_model: VllmRunner) -> tuple[int, list[float]]:
    num_drafts = 0
    accepted_per_position = [0, 0, 0]
    for metric in vllm_model.model.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for position, count in enumerate(metric.values):
                if position < 3:
                    accepted_per_position[position] += count
    assert num_drafts > 0
    return num_drafts, [count / num_drafts for count in accepted_per_position]


@pytest.mark.e2e_model(TARGET_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, _GRAPH_ENV)
@wait_until_npu_memory_free()
def test_gemma4_31b_mtp_k3_acceptance_tp4() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    with VllmRunner(
        TARGET_MODEL,
        tensor_parallel_size=4,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        skip_mm_profiling=True,
        disable_log_stats=False,
        speculative_config={
            "method": "mtp",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 3,
            "enforce_eager": False,
        },
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[8],
        ),
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=64)
        num_drafts, acceptance_rates = _collect_acceptance_rates(vllm_model)

    assert len(outputs) == len(prompts)
    assert all(output_ids for output_ids, _ in outputs)
    assert num_drafts >= len(prompts)
    assert acceptance_rates[0] > 0.5, acceptance_rates
    assert acceptance_rates[-1] > 0.1, acceptance_rates
