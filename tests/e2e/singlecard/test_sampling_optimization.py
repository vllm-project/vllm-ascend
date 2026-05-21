# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "List three colors:",
]


def _sampling_optimization_config(logits_processing_mode: str = "default"):
    return {
        "sampling_config": {
            "enable_sampling_optimization": True,
            "logits_processing_mode": logits_processing_mode,
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


def _generate_outputs(sampling_params: SamplingParams, additional_config=None):
    with VllmRunner(MODEL, **_runner_kwargs(additional_config)) as runner:
        outputs = runner.generate(PROMPTS, sampling_params, use_tqdm=False)
    return [(token_ids[0], texts[0]) for token_ids, texts in outputs]


def test_sampling_optimization_greedy_matches_default_sampler() -> None:
    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.0,
        repetition_penalty=1.05,
        frequency_penalty=0.1,
    )

    reference_outputs = _generate_outputs(sampling_params)
    optimized_outputs = _generate_outputs(
        sampling_params,
        additional_config=_sampling_optimization_config(),
    )

    check_outputs_equal(
        outputs_0_lst=reference_outputs,
        outputs_1_lst=optimized_outputs,
        name_0="default sampler",
        name_1="sampling optimization",
    )


def test_sampling_optimization_stochastic_smoke() -> None:
    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        seed=0,
    )

    outputs = _generate_outputs(
        sampling_params,
        additional_config=_sampling_optimization_config(),
    )

    assert len(outputs) == len(PROMPTS)
    for output_token_ids, output_text in outputs:
        assert output_token_ids
        assert output_text
