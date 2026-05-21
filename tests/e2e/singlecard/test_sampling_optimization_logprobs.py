# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "A short fact about Saturn:",
    "One sentence about compilers:",
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


def _generate_logprobs(additional_config=None):
    sampling_params = SamplingParams(
        max_tokens=6,
        temperature=0.0,
        logprobs=2,
    )
    with VllmRunner(MODEL, **_runner_kwargs(additional_config)) as runner:
        return runner.generate_w_logprobs(PROMPTS, sampling_params, use_tqdm=False)


def _sampled_logprob_values(output):
    output_token_ids, _output_text, output_logprobs = output
    assert output_logprobs is not None
    assert len(output_token_ids) == len(output_logprobs)

    values = []
    for token_id, step_logprobs in zip(output_token_ids, output_logprobs):
        assert token_id in step_logprobs
        logprob = step_logprobs[token_id]
        values.append(float(getattr(logprob, "logprob", logprob)))
    return values


def _sampled_logprob_ranks(output):
    output_token_ids, _output_text, output_logprobs = output
    assert output_logprobs is not None
    ranks = []
    for token_id, step_logprobs in zip(output_token_ids, output_logprobs):
        assert token_id in step_logprobs
        rank = getattr(step_logprobs[token_id], "rank", None)
        assert rank is not None
        ranks.append(int(rank))
    return ranks


def test_sampling_optimization_logprobs_match_default_sampler() -> None:
    reference_outputs = _generate_logprobs()
    optimized_outputs = _generate_logprobs(_sampling_optimization_config())

    check_outputs_equal(
        outputs_0_lst=[output[:2] for output in reference_outputs],
        outputs_1_lst=[output[:2] for output in optimized_outputs],
        name_0="default sampler",
        name_1="sampling optimization",
    )

    for reference_output, optimized_output in zip(reference_outputs, optimized_outputs):
        assert _sampled_logprob_values(optimized_output) == pytest.approx(
            _sampled_logprob_values(reference_output),
            abs=1e-3,
        )
        assert _sampled_logprob_ranks(optimized_output) == _sampled_logprob_ranks(reference_output)
