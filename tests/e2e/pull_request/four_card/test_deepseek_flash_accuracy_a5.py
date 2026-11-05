# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-checkpoint A5 Flash fixed-token generation regressions.

Use the same prompts, five-token greedy generation, and exact token-ID
assertions as the A3 V4 PR regression.
Use the A5 FP4/FP8 checkpoints and TP1/DP4/EP instead of A3 W4A8/TP4.
V4.1 keeps its complete Engram tables in shared host memory. Pro is excluded.
"""

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.deepseek_a5_utils import configure_a5

DP_SIZE = 4
MAX_OUTPUT_TOKENS = 5
PROMPTS = (
    "Hello, my name is",
    "What is the meaning of life?",
)


@pytest.fixture(scope="module", autouse=True)
def a5_runtime():
    with pytest.MonkeyPatch.context() as monkeypatch:
        configure_a5(monkeypatch)
        monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "3600")
        yield


def _engine_args(tokenizer_mode: str) -> dict:
    args = {
        "tensor_parallel_size": 1,
        "data_parallel_size": DP_SIZE,
        "distributed_executor_backend": "mp",
        "dp_start_timeout": 3600,
        "dp_request_timeout": 600,
        "enable_expert_parallel": True,
        "quantization": "deepseek_v4_fp8",
        "tokenizer_mode": tokenizer_mode,
        "max_model_len": 2048,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 1,
        "kv_cache_memory_bytes": 512 * 1024**2,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "seed": 0,
        "safetensors_load_strategy": "lazy",
        "model_loader_extra_config": {"enable_multithread_load": True, "num_threads": 16},
        "additional_config": {"enable_cpu_binding": False},
    }
    if tokenizer_mode == "deepseek_v41":
        args.update(
            block_size=128,
            limit_mm_per_prompt={"image": 0},
            engram_config={"cpu_offload": True, "dp_shared_memory": True},
            swa_bounded_replay=True,
        )
    else:
        args.update(block_size=32, kv_cache_dtype="fp8", attention_config={"indexer_kv_dtype": "fp8"})
    return args


def _check_real_weight_greedy(model: str, tokenizer_mode: str, expected_token_ids: list[list[int]]) -> None:
    # The shared DP runner dispatches to every rank directly, without relying
    # on HTTP load balancing to assign a very small prompt batch to all ranks.
    with DPVllmRunner(maybe_model_redirect(model), **_engine_args(tokenizer_mode)) as runner:
        outputs = runner.generate_greedy(list(PROMPTS), MAX_OUTPUT_TOKENS)
        print(f"{model} greedy outputs: {outputs!r}", flush=True)
        assert len(outputs) == len(PROMPTS)
        for (output_ids, output_str), expected in zip(outputs, expected_token_ids, strict=True):
            assert len(output_str) > 0
            assert len(output_ids) > 0
            assert output_ids == expected, (model, output_ids, expected)


@pytest.mark.e2e_model("deepseek-ai/DeepSeek-V4-Flash-0731")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,mixed_lengths",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_deepseek_v4_flash_a5_real_weight_greedy() -> None:
    expected_token_ids = [
        [19923, 14, 1026, 2329, 344, 680, 2852, 95, 305, 342],
        [3085, 344, 270, 5281, 294, 1988, 33, 1780, 905, 3180, 1780, 905],
    ]
    _check_real_weight_greedy("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek_v4", expected_token_ids)


@pytest.mark.e2e_model("deepseek-ai/DeepSeek-V4.1-Flash")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,cpu_offloading,mixed_lengths",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_deepseek_v41_flash_a5_real_weight_greedy() -> None:
    # The released V4.1 checkpoint ends these raw completions before the
    # five-token limit; include EOS in the expected IDs, as returned by vLLM.
    expected_token_ids = [
        [19923, 14, 1026, 2329, 344, 1613, 1],
        [3085, 344, 270, 5281, 294, 1988, 33, 1],
    ]
    _check_real_weight_greedy("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41", expected_token_ids)
