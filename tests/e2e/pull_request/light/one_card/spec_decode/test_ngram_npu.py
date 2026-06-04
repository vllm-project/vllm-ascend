from __future__ import annotations

from typing import Any

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.light.one_card.spec_decode.utils import calculate_acceptance_per_pos


@pytest.mark.parametrize("num_speculative_tokens", [3])
def test_ngram_npu_async_acceptance(
    test_prompts: list[list[dict[str, Any]]],
    num_speculative_tokens: int,
    model_name: str,
):
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    speculative_config = {
        "method": "ngram_gpu",
        "prompt_lookup_max": 2,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": num_speculative_tokens,
    }

    compilation_config = CompilationConfig(
        cudagraph_mode="PIECEWISE",
        cudagraph_capture_sizes=[12],
    )

    with VllmRunner(
        model_name,
        max_model_len=2048,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        async_scheduling=True,
    ) as llm:
        outputs = llm.model.chat(test_prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = [0.50, 0.30, 0.20]

    match = all(abs(a - b) < 1.0 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match
