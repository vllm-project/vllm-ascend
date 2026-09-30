from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import BASELINES, DFLASH, calculate_acceptance_per_pos

MAX_NUM_SEQS = 256
DYNAMIC_DFLASH_BASELINES = {
    # DFlash drafts a block in parallel, so changing K changes the block input
    # shape and its acceptance profile. Do not reuse the K=8 baseline for K=4.
    ("dflash", 4): [0.8, 0.5, 0.3, 0.2],
}


@pytest.mark.parametrize("method", DFLASH.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
@pytest.mark.parametrize(
    "dynamic_num_speculative_tokens",
    [
        pytest.param(None, id="static"),
        pytest.param(4, id="dynamic"),
    ],
)
def test_dflash_acceptance(
    method: str,
    num_speculative_tokens: int,
    dynamic_num_speculative_tokens: int | None,
):
    main_model_name = DFLASH[method]["main"]
    spec_model_name = DFLASH[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
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
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if dynamic_num_speculative_tokens is not None:
        speculative_config["num_speculative_tokens_per_batch_size"] = [
            [1, MAX_NUM_SEQS, dynamic_num_speculative_tokens]
        ]
        dynamic_capture_size = len(prompts) * (dynamic_num_speculative_tokens + 1)
        capture_sizes = [dynamic_capture_size, 9, 18]
        cudagraph_mode = "PIECEWISE"
    else:
        capture_sizes = [9, 18]
        cudagraph_mode = "FULL_DECODE_ONLY"

    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=capture_sizes,
    )

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
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
    effective_num_speculative_tokens = (
        num_speculative_tokens if dynamic_num_speculative_tokens is None else dynamic_num_speculative_tokens
    )
    if dynamic_num_speculative_tokens is None:
        golden = BASELINES[method][:effective_num_speculative_tokens]
    else:
        golden = DYNAMIC_DFLASH_BASELINES[(method, dynamic_num_speculative_tokens)]

    assert len(acceptance_per_pos) == num_speculative_tokens
    active_acceptance_per_pos = acceptance_per_pos[:effective_num_speculative_tokens]
    match = all(abs(a - b) < 0.1 for a, b in zip(active_acceptance_per_pos, golden, strict=True))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
    if dynamic_num_speculative_tokens is not None:
        assert all(rate == 0 for rate in acceptance_per_pos[dynamic_num_speculative_tokens:])
