from __future__ import annotations

from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import MODELS, calculate_acceptance_per_pos


def test_qwen3_vl_eagle(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    vl_model_name: str,
):
    with VllmRunner(
        vl_model_name,
        max_model_len=1024,
        cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as ref_llm:
        ref_llm.model.chat(test_prompts, sampling_config)


@pytest.mark.parametrize("method", MODELS.keys())
@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("draft_tensor_parallel_size", [1])
@pytest.mark.parametrize("disable_padded_drafter_batch", [False])
@pytest.mark.parametrize("async_scheduling", [True])
def test_qwen_eagle3_acceptance(
    method: str,
    num_speculative_tokens: int,
    draft_tensor_parallel_size: None | int,
    disable_padded_drafter_batch: bool,
    async_scheduling: bool,
):
    main_model_name = MODELS[method]["main"]
    spec_model_name = MODELS[method]["spec"]

    tokenizer_path = resolve_tokenizer_args(main_model_name)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [
        {"role": "user", "content": "Hello, my name is"},
        {"role": "user", "content": "The president of the United States is"},
        {"role": "user", "content": "The capital of France is"},
        {"role": "user", "content": "The future of AI is"},
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": draft_tensor_parallel_size,
        "disable_padded_drafter_batch": disable_padded_drafter_batch,
        "model": spec_model_name,
    }

    compilation_config = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[12])

    with VllmRunner(
        main_model_name,
        max_model_len=2048,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        async_scheduling=async_scheduling,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = [0.68, 0.40, 0.18]

    match = all(abs(a - b) < 0.08 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
