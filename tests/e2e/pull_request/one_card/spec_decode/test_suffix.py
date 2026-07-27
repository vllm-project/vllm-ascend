from __future__ import annotations

from typing import Any

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def test_suffix_acceptance(
    test_prompts: list[list[dict[str, Any]]],
    sampling_config: SamplingParams,
    model_name: str,
):
    num_draft = []
    num_accept = []
    with VllmRunner(
        model_name,
        speculative_config={
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": 1000,
            "num_speculative_tokens": 10,
        },
        max_model_len=1024,
        max_cudagraph_capture_size=16,
        disable_log_stats=False,
    ) as runner:
        for i in range(10):
            runner.model.chat(test_prompts[i], sampling_config)
            metrics = runner.model.get_metrics()
            for metric in metrics:
                print(metric)
                if metric.name == "vllm:spec_decode_num_draft_tokens":
                    num_draft.append(metric.value)
                if metric.name == "vllm:spec_decode_num_accepted_tokens":
                    num_accept.append(metric.value)

    first_accept_tokens = num_accept[0]
    first_draft_tokens = num_draft[0]
    first_accept_rate = first_accept_tokens / first_draft_tokens

    last_accept_tokens = num_accept[-1] - num_accept[-2]
    last_draft_tokens = num_draft[-1] - num_draft[-2]
    last_accept_rate = last_accept_tokens / last_draft_tokens

    assert first_accept_tokens < last_accept_tokens
    assert first_accept_rate < last_accept_rate
    assert last_accept_rate > 0.60
