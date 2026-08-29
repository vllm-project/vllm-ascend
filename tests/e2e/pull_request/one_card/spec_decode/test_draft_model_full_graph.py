from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import calculate_acceptance_per_pos

# Independent draft model for the draft_model spec-decode method (same
# tokenizer family / vocabulary as the target).
MAIN_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"
NUM_SPECULATIVE_TOKENS = 5
# Target capture sizes are R*(K+1) based ([6, 12] for R in {1, 2}); the
# drafter's derived table becomes [7, 14] (R*(K+2)).
CAPTURE_SIZES = [6, 12]
# Per-position acceptance under the FULL-graph drafter must stay within
# this tolerance of the eager-drafter baseline: graph compilation changes
# kernel tiling / accumulation order, which is numeric noise under BF16 +
# greedy argmax, not a systematic draft-quality regression.
ACCEPTANCE_TOLERANCE = 0.15


def _run_once(additional_config: dict | None):
    tokenizer_path = resolve_tokenizer_args(MAIN_MODEL)[1]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=256)

    prompts = [
        {"role": "user", "content": "Hello, your name is"},
        {"role": "user", "content": "The capital of France is"},
    ]
    prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

    speculative_config = {
        "method": "draft_model",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=CAPTURE_SIZES,
    )

    with VllmRunner(
        MAIN_MODEL,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        additional_config=additional_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, NUM_SPECULATIVE_TOKENS, Counter, Vector)
    return outputs, acceptance_per_pos


@pytest.mark.parametrize(
    "additional_config",
    [
        # Default: drafter runs eager (upstream PIECEWISE-only semantics);
        # the target model still runs FULL graphs.
        None,
        # draft_model_full_graph: drafter gets its own R*(K+2) capture
        # table and K+2-based dispatch.
        {"draft_model_full_graph": True},
    ],
    ids=["drafter_eager_default", "drafter_full_graph"],
)
def test_draft_model_full_graph_acceptance(additional_config: dict | None):
    outputs, acceptance_per_pos = _run_once(additional_config)

    for output in outputs:
        assert len(output.outputs[0].token_ids) > 0
        print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")

    assert acceptance_per_pos, "no acceptance metrics reported"
    print(f"acceptance_per_pos: {acceptance_per_pos}")
    # The draft is a real model of the same family; the first position
    # (the sampled seed token re-draft) should be accepted most of the time.
    assert acceptance_per_pos[0] > 0.5


def test_draft_model_full_graph_matches_eager_drafter_acceptance():
    # The K+2 dispatch/padding path must not corrupt draft quality: the
    # FULL-graph drafter's per-position acceptance stays close to the
    # eager-drafter baseline.
    _, eager = _run_once(None)
    _, full_graph = _run_once({"draft_model_full_graph": True})

    assert len(eager) == len(full_graph)
    for i, (a, b) in enumerate(zip(eager, full_graph)):
        assert abs(a - b) < ACCEPTANCE_TOLERANCE, (
            f"position {i}: full-graph drafter acceptance {b} deviates from "
            f"eager-drafter baseline {a} beyond tolerance {ACCEPTANCE_TOLERANCE}"
        )
