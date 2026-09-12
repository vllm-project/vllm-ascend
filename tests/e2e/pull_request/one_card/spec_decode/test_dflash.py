from __future__ import annotations

import pytest
from vllm.config import CompilationConfig

from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH
from tests.e2e.pull_request.utils import SPEC_DECODE_PROMPTS, _run_speculative_decoding

MAX_NUM_SEQS = 256

EXPECTED_ACCEPTANCE_LENGTH_STATIC = 2.80
EXPECTED_ACCEPTANCE_LENGTH_DYNAMIC = 2.60


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

    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
    }
    num_prompts = len(SPEC_DECODE_PROMPTS)
    if dynamic_num_speculative_tokens is not None:
        speculative_config["num_speculative_tokens_per_batch_size"] = [
            [1, MAX_NUM_SEQS, dynamic_num_speculative_tokens]
        ]
        capture_sizes = [
            num_prompts * (dynamic_num_speculative_tokens + 1),
            num_prompts * (num_speculative_tokens + 1),
        ]
        cudagraph_mode = "PIECEWISE"
        expected_acceptance_length = EXPECTED_ACCEPTANCE_LENGTH_DYNAMIC
    else:
        capture_sizes = [num_prompts * (num_speculative_tokens + 1)]
        cudagraph_mode = "FULL_DECODE_ONLY"
        expected_acceptance_length = EXPECTED_ACCEPTANCE_LENGTH_STATIC

    _run_speculative_decoding(
        model_name=main_model_name,
        speculative_config=speculative_config,
        expected_acceptance_length=expected_acceptance_length,
        runner_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "gpu_memory_utilization": 0.8,
            "compilation_config": CompilationConfig(
                cudagraph_mode=cudagraph_mode,
                cudagraph_capture_sizes=capture_sizes,
            ),
            "enable_prefix_caching": False,
        },
        example_prompts=SPEC_DECODE_PROMPTS,
        max_tokens=256,
        is_moe=False,
        enable_thinking=False,
    )
