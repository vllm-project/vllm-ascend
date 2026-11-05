# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 V4.1 replay functional tests with the complete Flash checkpoint.

Exercise prefix reuse on DP4, including a prefill chunk smaller than the SWA
window. Bounded replay deliberately truncates SWA dependencies, so these checks
validate scheduling and cache reuse without requiring replay-on/off token parity.
"""

from pathlib import Path

import pytest
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.deepseek_a5_utils import (
    SWA_REPLAY_LOG,
    LoggedServer,
    assert_dp_engines,
    cached_tokens,
    completions,
    configure_a5,
    engine_args,
    prompt,
)

LONG_PROMPT_TOKENS = 1153
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SWA_WINDOW_TOKENS = 128
DP_SIZE = 4


def _args() -> dict:
    args = engine_args()
    args.update(
        # A5 QSMLA requires all 64 query heads on each rank. Keep V4.1 at TP1;
        # TP4 shards the official 64 heads to 16 per rank and fails validation.
        tensor_parallel_size=1,
        data_parallel_size=DP_SIZE,
        data_parallel_size_local=DP_SIZE,
        # Keep single-sequence batches to control the replay boundary cases.
        max_num_seqs=1,
        async_scheduling=False,
        worker_extension_cls="tests.e2e.pull_request.deepseek_a5_utils.A5WorkerProbe",
        tokenizer_mode="deepseek_v41",
        quantization="deepseek_v4_fp8",
        block_size=128,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
        engram_config={"cpu_offload": True, "dp_shared_memory": True},
        safetensors_load_strategy="lazy",
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 16},
    )
    return args


def _assert_replay(server: LoggedServer, prompts: list[list[int]]) -> None:
    server.collective_rpc("install_swa_replay_probe")
    # completions validates output lengths, token IDs and finite logprobs for
    # both cold and cached requests. Approximate replay need not match tokens.
    uncached = completions(server, prompts)
    assert all(cached_tokens(result) == 0 for result in uncached)
    cached = completions(server, prompts)
    assert any(cached_tokens(result) > SWA_WINDOW_TOKENS for result in cached), (
        "No DP engine reused a prefix longer than the SWA window"
    )
    assert_dp_engines(server, DP_SIZE)
    server.assert_log(SWA_REPLAY_LOG)


@pytest.fixture(scope="module", autouse=True)
def a5_runtime():
    with pytest.MonkeyPatch.context() as monkeypatch:
        configure_a5(monkeypatch)
        monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
        yield


@pytest.fixture(scope="module")
def v41_model(a5_runtime):
    model = maybe_model_redirect(MODEL)
    tokenizer = get_tokenizer(model, tokenizer_mode="deepseek_v41")
    lengths = (LONG_PROMPT_TOKENS, SWA_WINDOW_TOKENS + 1, SWA_WINDOW_TOKENS, SWA_WINDOW_TOKENS - 1) * DP_SIZE
    prompts = [
        prompt(length, salt=i * 137, tokenizer=tokenizer)["prompt_token_ids"] for i, length in enumerate(lengths)
    ]
    return model, prompts


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,chunked_prefill,cpu_offloading,mixed_lengths",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="MXFP4,MXFP8",
    graph_mode="eager",
)
@pytest.mark.parametrize("chunk_size", [512, 64])
@wait_until_npu_memory_free()
def test_deepseek_v41_a5_bounded_replay(v41_model, chunk_size: int, tmp_path: Path) -> None:
    model, prompts = v41_model
    args = dict(_args(), swa_bounded_replay=True, max_num_batched_tokens=chunk_size)
    with LoggedServer(model, args, tmp_path / f"v41-replay-{chunk_size}.log") as server:
        _assert_replay(server, prompts)
