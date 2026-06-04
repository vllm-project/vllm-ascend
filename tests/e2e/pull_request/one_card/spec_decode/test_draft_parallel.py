from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import (
    BASELINES,
    DRAFT_PARALLEL_MODELS,
    calculate_acceptance_per_pos,
)


@pytest.mark.parametrize("method", DRAFT_PARALLEL_MODELS.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
@pytest.mark.parametrize("draft_tensor_parallel_size", [1])
def test_parallel_drafting_acceptance(
    method: str,
    num_speculative_tokens: int,
    draft_tensor_parallel_size: None | int,
):
    main_model_name = DRAFT_PARALLEL_MODELS[method]["main"]
    spec_model_name = DRAFT_PARALLEL_MODELS[method]["spec"]

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

    prompts = [{"role": "user", "content": "Hello, your name is"}]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "draft_model",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": draft_tensor_parallel_size,
        "parallel_drafting": True,
    }

    compilation_config = CompilationConfig(
        cudagraph_mode="PIECEWISE",
        cudagraph_capture_sizes=[12],
    )

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
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
    golden = BASELINES[method]

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match
