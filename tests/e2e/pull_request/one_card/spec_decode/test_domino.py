from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import (
    BASELINES,
    DOMINO,
    calculate_acceptance_per_pos,
)

POSITION_ACCEPTANCE_TOLERANCE = 0.1
CANDIDATE_ACCEPTANCE_LENGTH_TOLERANCE = 0.1


@pytest.mark.parametrize("method", DOMINO.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
@pytest.mark.parametrize("enable_reduce_sample", [False, True])
def test_domino_acceptance(
    method: str,
    num_speculative_tokens: int,
    enable_reduce_sample: bool,
):
    """Domino speculative decoding E2E acceptance test.

    Domino is selected by ``method="dflash"`` + a Domino checkpoint
    (``projector_type == "domino"``); vllm-ascend auto-detects it and dispatches
    to ``AscendDominoProposer``. Run on a single NPU with the
    ``Huang2020/Qwen3-8B-Domino-b16`` checkpoint pre-cached (the prod env is
    offline).

    The draft uses aclgraph (NPUGraph) capture: the fused correction loop
    (triton kernels + fixed-k Python loop + fixed bonus buffer) is captured by
    the DFlash ``dummy_run``.
    """
    main_model_name = DOMINO[method]["main"]
    spec_model_name = DOMINO[method]["spec"]

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
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[9, 18]
    )

    results = {}
    for candidate_topk in (0, 64):
        additional_config = {
            "enable_reduce_sample": enable_reduce_sample,
            "domino_candidate_topk": candidate_topk,
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
            compilation_config=compilation_config,
            enable_prefix_caching=False,
        ) as llm:
            outputs = llm.model.generate(prompts, sampling_params)
            metrics = llm.model.get_metrics()

        output_token_ids = [tuple(output.outputs[0].token_ids) for output in outputs]
        acceptance_per_pos = calculate_acceptance_per_pos(
            metrics, num_speculative_tokens, Counter, Vector
        )
        results[candidate_topk] = (output_token_ids, acceptance_per_pos)

        for output in outputs:
            print(f"candidate_topk={candidate_topk}, prompt={output.prompt!r}")
            print(f"Generated text: {output.outputs[0].text!r}")
            print(f"Output tokens: {output.outputs[0].token_ids}")
        print(
            f"candidate_topk={candidate_topk}, "
            f"acceptance_per_pos={acceptance_per_pos}"
        )

    full_output_ids, full_acceptance = results[0]
    candidate_output_ids, candidate_acceptance = results[64]
    assert candidate_output_ids == full_output_ids, (
        "Top-64 Domino changed greedy output tokens relative to full-vocabulary "
        "Domino. The target verifier should preserve final output semantics."
    )

    full_acceptance_length = sum(full_acceptance)
    candidate_acceptance_length = sum(candidate_acceptance)
    assert (
        candidate_acceptance_length
        >= full_acceptance_length - CANDIDATE_ACCEPTANCE_LENGTH_TOLERANCE
    ), (
        "Top-64 Domino acceptance length regressed relative to full-vocabulary "
        f"Domino: full={full_acceptance_length:.3f}, "
        f"candidate={candidate_acceptance_length:.3f}."
    )

    golden = BASELINES[method]
    print(f"full_acceptance_per_pos: {full_acceptance}")
    print(f"candidate_acceptance_per_pos: {candidate_acceptance}")
    print(f"golden: {golden}")

    if not enable_reduce_sample:
        # Regression coverage: Domino correction must run even when Ascend's
        # reduce-sampling optimization is disabled.
        assert full_acceptance[0] > 0.8
        assert candidate_acceptance[0] > 0.8

    # While the baseline is placeholder zeros, do not assert - just print the
    # measured acceptance so the user can record it into BASELINES. Once a real
    # baseline is recorded, enforce it as a lower bound below.
    if all(g == 0 for g in golden):
        pytest.skip(
            "Domino acceptance baseline is a placeholder. Record the measured "
            "acceptance_per_pos into BASELINES['domino'] and re-run."
        )

    # Acceptance above the recorded baseline is an improvement, not a
    # regression. Keep a small tolerance for run-to-run variation on NPU.
    regressions = [
        (position, actual, expected)
        for position, (actual, expected) in enumerate(zip(full_acceptance, golden))
        if actual < expected - POSITION_ACCEPTANCE_TOLERANCE
    ]
    assert not regressions, f"Domino acceptance regressed: {regressions}"
