# SPDX-License-Identifier: Apache-2.0
import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.kvpp_utils import (
    BLOCK_SIZE,
    NUM_BLOCKS,
    SchedulerTrace,
    assert_worker_state,
    compare_outputs,
    generate,
    run_basic_comparison,
    token_prompt,
)

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

pytestmark = pytest.mark.e2e_model(MODEL)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
@wait_until_npu_memory_free()
def test_kvpp_chunked_prefill_and_decode(monkeypatch, v2):
    run_basic_comparison(monkeypatch, MODEL, v2=v2)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_kvpp_prefix_hit_and_block_reuse(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    results = []
    for enabled in (False, True):
        with VllmRunner(
            maybe_model_redirect(MODEL),
            dtype="bfloat16",
            tensor_parallel_size=2,
            enforce_eager=True,
            async_scheduling=True,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_model_len=8 * BLOCK_SIZE,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=NUM_BLOCKS,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.8,
            seed=42,
            additional_config={"enable_kvpp": enabled},
        ) as runner:
            assert_worker_state(runner, enabled, 2, 1, NUM_BLOCKS)
            tokenizer = runner.model.get_tokenizer()
            prefix = token_prompt(
                tokenizer, "Explain the importance of storing historical information. ", 6 * BLOCK_SIZE
            )
            warmup = prefix + token_prompt(tokenizer, "First answer: ", 16)
            repeated = prefix + token_prompt(tokenizer, "Second answer: ", 16)
            outputs = []
            with monkeypatch.context() as observation:
                trace = SchedulerTrace(runner, observation)
                outputs.extend(generate(runner, [warmup]))
                hit = generate(runner, [repeated])
                assert hit[0].num_cached_tokens > 0
                outputs.extend(hit)
                seen_blocks, reused = set(), False
                first_blocks = set()
                for index in range(12):
                    prompt = token_prompt(
                        tokenizer, f"Unique record {index:04d}: describe this distinct observation. ", 6 * BLOCK_SIZE
                    )
                    first = tuple(prompt[:BLOCK_SIZE])
                    assert first not in first_blocks
                    first_blocks.add(first)
                    wave = generate(runner, [prompt])
                    assert wave[0].num_cached_tokens == 0
                    written = trace.written_blocks[wave[0].request_id]
                    assert written
                    reused |= bool(seen_blocks & written)
                    seen_blocks.update(written)
                    outputs.extend(wave)
                assert reused, "No completed request's block was overwritten by a later request"
                outputs.extend(generate(runner, [warmup]))
            results.append(outputs)
    compare_outputs(*results)
