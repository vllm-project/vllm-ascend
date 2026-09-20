"""Performance guard for profiling-based dynamic chunk sizing (PP scenario).

Measures Time-To-First-Token (TTFT) on 64k-token prefill requests with
profiling_chunk_config enabled. The test runs against Qwen3-30B-A3B
served with PP=2, TP=2 (4 NPU cards total).

The same scenario is executed with Model Runner V1 and Model Runner V2,
both with Short Request First (SRF) disabled and enabled. Within each SRF
mode, the only configuration difference between the two model-runner runs
is VLLM_USE_V2_MODEL_RUNNER.

Test flow:
  1. Create an LLM engine with profiling_chunk_config enabled.
  2. Run NUM_WARMUP sequential requests (64k tokens, max_tokens=1) to warm
     up both the NPU and the profiling predictor.
  3. Run NUM_TEST sequential requests, recording TTFT for each.
  4. Assert that the median TTFT does not exceed BASELINE_TTFT_S seconds.
"""

import os
import statistics
import time
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-30B-A3B"

# ~64k tokens
_WORD = "hello "
INPUT_64K_TOKENS = _WORD * (384_000 // len(_WORD))

NUM_WARMUP = 5
NUM_TEST = 5

# NOTE: Any changes to this baseline must be approved by team members.
# Measured on Qwen3-30B-A3B, PP=2, TP=2, 64k prefill, profiling_chunk enabled.
BASELINE_TTFT_S = 5.45


@pytest.mark.parametrize(
    ("use_v2_model_runner", "enable_srf"),
    [
        pytest.param("0", False, id="MRV1"),
        pytest.param("1", False, id="MRV2"),
        pytest.param("0", True, id="MRV1-SRF"),
        pytest.param("1", True, id="MRV2-SRF"),
    ],
)
@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    },
)
@wait_until_npu_memory_free()
def test_profiling_chunk_ttft_performance(
    use_v2_model_runner: str,
    enable_srf: bool,
) -> None:
    runner_name = "MRV2" if use_v2_model_runner == "1" else "MRV1"
    scenario_name = f"{runner_name}+SRF" if enable_srf else runner_name

    # Keep all other runtime settings identical across cases.
    with (
        patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": use_v2_model_runner}),
        VllmRunner(
            MODEL,
            max_model_len=70000,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            block_size=128,
            enable_expert_parallel=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=12288,
            distributed_executor_backend="mp",
            enforce_eager=True,
            async_scheduling=False,
            additional_config={
                "scheduler_config": {
                    "profiling_chunk_config": {
                        "enabled": True,
                        "smooth_factor": 0.9,
                    },
                    "short_request_first_config": {
                        "enabled": enable_srf,
                    },
                },
                "enable_cpu_binding": False,
            },
            hf_overrides={
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 1000,
                    "factor": 5,
                    "original_max_position_embeddings": 262144,
                }
            },
        ) as vllm_model,
    ):
        # With max_tokens=1, total latency ≈ prefill time ≈ TTFT
        prompts = [INPUT_64K_TOKENS]

        # ── Warmup ──────────────────────────────────────────────────────
        for _ in range(NUM_WARMUP):
            vllm_model.generate_greedy(prompts, max_tokens=1)

        # ── Measurement ─────────────────────────────────────────────────
        ttfts: list[float] = []
        for _ in range(NUM_TEST):
            start = time.perf_counter()
            vllm_model.generate_greedy(prompts, max_tokens=1)
            ttfts.append(time.perf_counter() - start)

        median_ttft = statistics.median(ttfts)
        ttft_str = ", ".join(f"{t:.2f}s" for t in ttfts)
        print(
            f"\n[profiling_chunk perf][{scenario_name}] "
            f"TTFT per request: [{ttft_str}]"
            f"\n[profiling_chunk perf][{scenario_name}] "
            f"Median TTFT: {median_ttft:.2f}s  "
            f"(baseline: {BASELINE_TTFT_S}s)"
        )

        assert median_ttft <= BASELINE_TTFT_S, (
            f"[{scenario_name}] TTFT performance regression: "
            f"median TTFT {median_ttft:.2f}s exceeds baseline "
            f"{BASELINE_TTFT_S}s. Individual TTFTs: [{ttft_str}]"
        )
