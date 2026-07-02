from __future__ import annotations

from typing import Any

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.e2e_features("eager_mode", "ngram")
@pytest.mark.e2e_model("LLM-Research/Meta-Llama-3.1-8B-Instruct")
def test_ngram(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    with VllmRunner(
        model_name,
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
        max_model_len=1024,
        cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as runner:
        runner.model.chat(test_prompts, sampling_config)
