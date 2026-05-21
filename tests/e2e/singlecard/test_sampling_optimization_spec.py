# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Repeat the word blue five times:",
    "Complete the pattern: red, blue, red, blue,",
]


def _sampling_optimization_config():
    return {
        "sampling_config": {
            "enable_sampling_optimization": True,
            "logits_processing_mode": "default",
        }
    }


def _runner_kwargs(additional_config=None):
    kwargs = {
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4],
        "gpu_memory_utilization": 0.7,
    }
    if additional_config is not None:
        kwargs["additional_config"] = additional_config
    return kwargs


def _generate_outputs(sampling_params: SamplingParams, **runner_kwargs):
    with VllmRunner(MODEL, **_runner_kwargs(**runner_kwargs)) as runner:
        outputs = runner.generate(PROMPTS, sampling_params, use_tqdm=False)
    return [(token_ids[0], texts[0]) for token_ids, texts in outputs]


def test_sampling_optimization_ngram_spec_decode_matches_target_model() -> None:
    sampling_params = SamplingParams(
        max_tokens=12,
        temperature=0.0,
    )
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": 3,
    }

    reference_outputs = _generate_outputs(sampling_params)
    optimized_spec_outputs = _generate_outputs(
        sampling_params,
        additional_config=_sampling_optimization_config(),
        speculative_config=speculative_config,
    )

    check_outputs_equal(
        outputs_0_lst=reference_outputs,
        outputs_1_lst=optimized_spec_outputs,
        name_0="target model",
        name_1="sampling optimization spec decode",
    )
