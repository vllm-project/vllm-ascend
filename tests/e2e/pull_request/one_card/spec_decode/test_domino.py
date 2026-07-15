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


@pytest.mark.parametrize("method", DOMINO.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
def test_domino_acceptance(
    method: str,
    num_speculative_tokens: int,
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

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics, num_speculative_tokens, Counter, Vector
    )
    golden = BASELINES[method]
    print(f"acceptance_per_pos: {acceptance_per_pos}")
    print(f"golden: {golden}")

    # While the baseline is placeholder zeros, do not assert - just print the
    # measured acceptance so the user can record it into BASELINES. Once a real
    # baseline is recorded, this falls back to the DFlash-style assertion.
    if all(g == 0 for g in golden):
        pytest.skip(
            "Domino acceptance baseline is a placeholder. Record the measured "
            "acceptance_per_pos into BASELINES['domino'] and re-run."
        )

    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match
