from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import (
    BASELINES,
    DFLASH,
    DSPARK,
    calculate_acceptance_per_pos,
)

# Confidence-based dynamic verify-length via additional_config.dynamic_spec_config.
# Distinct from batch-size based dynamic K (num_speculative_tokens_per_batch_size).
DYNAMIC_CASES = {
    "dspark": {
        "models": DSPARK,
        "num_speculative_tokens": 7,
        "prompt": "What is your name? Please introduce yourself in detail.",
        "method_params": {
            "initial_verify_budget_per_req": 3,
            "budget_update_interval": 1,
            "budget_threshold": 0.7,
        },
        "compilation_config": CompilationConfig(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=[7, 8],
        ),
    },
    "dflash": {
        "models": DFLASH,
        "num_speculative_tokens": 8,
        "prompt": "What is your name? Please introduce yourself in detail.",
        "method_params": {
            "initial_verify_budget_per_req": 3,
            "budget_update_interval": 1,
            "budget_threshold": 0.7,
        },
        "compilation_config": CompilationConfig(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=[9, 18],
        ),
    },
}


@pytest.mark.parametrize("method", DYNAMIC_CASES.keys())
def test_dynamic_spec_acceptance(method: str):
    case = DYNAMIC_CASES[method]
    models = case["models"]
    main_model_name = models[method]["main"]
    spec_model_name = models[method]["spec"]
    num_speculative_tokens = case["num_speculative_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [{"role": "user", "content": case["prompt"]}]
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
        "enforce_eager": True,
        "method": method,
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }

    additional_config = {
        "dynamic_spec_config": {
            "method": method,
            "method_params": case["method_params"],
        },
    }

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        additional_config=additional_config,
        compilation_config=case["compilation_config"],
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
    golden = BASELINES[f"{method}_dynamic"]

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
