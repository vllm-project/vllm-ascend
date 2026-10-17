# SPDX-License-Identifier: Apache-2.0
"""Numerical engine oracle across mixed requests and repeated cache-slot reuse."""

import warnings

import pytest
from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


def _assert_outputs_compatible(reference: CompletionOutput, actual: CompletionOutput) -> None:
    assert len(actual.token_ids) == len(reference.token_ids)
    for position, (expected, observed) in enumerate(zip(reference.token_ids, actual.token_ids)):
        if expected == observed:
            continue
        # Follow upstream tests/models/utils.py::check_logprobs_close: BF16
        # batch/prefill changes can swap near-tied greedy choices, even with
        # speculation disabled. Compare both choices at the first divergence;
        # later conditional distributions have different token histories.
        assert reference.logprobs is not None and actual.logprobs is not None
        expected_probs, actual_probs = reference.logprobs[position], actual.logprobs[position]
        assert expected in actual_probs and observed in expected_probs, (
            f"Different predictions at token {position}: reference={expected_probs}, recover={actual_probs}"
        )
        warnings.warn(
            f"BF16 greedy choices differ at token {position} but are mutual top-5 predictions: "
            f"reference={expected_probs}, recover={actual_probs}",
            stacklevel=2,
        )
        break


@pytest.mark.parametrize("async_scheduling", [False, True])
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_gdn_recover_matches_non_speculative_target(monkeypatch, async_scheduling):
    if get_current_hardware_profile().device_adaptor_family is not DeviceAdaptorFamily.STANDARD:
        pytest.skip("Integrated GDN recovery requires A2/A3")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    prompts = [
        "Explain a hash table in simple terms.",
        "Write a Python function for binary search.",
        "The following is a list of integers: " + "1, 2, 3, 4, " * 48 + "continue it.",
        "Explain why a cache can reduce latency.",
    ]
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, logprobs=5) for n in (2, 9, 48, 96)]
    kwargs = dict(
        model_name=maybe_model_redirect("Qwen/Qwen3.5-4B"),
        dtype="bfloat16",
        language_model_only=True,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        # Leave room for the attention backend's temporary KV materialization.
        gpu_memory_utilization=0.6,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
    )
    # The target without speculation is an independent engine oracle. Native
    # speculative kernels have a separate ragged-state indexing dependency.
    with VllmRunner(**kwargs) as runner:
        reference = [o.outputs[0] for o in runner.model.generate(prompts, params)]
    with VllmRunner(
        **kwargs,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True},
        additional_config={"enable_gdn_recover": True},
    ) as runner:
        for order in ([0, 1, 2, 3], [3, 1, 0, 2], [2, 0, 3, 1]):
            outputs = runner.model.generate([prompts[i] for i in order], [params[i] for i in order])
            for i, output in zip(order, outputs):
                _assert_outputs_compatible(reference[i], output.outputs[0])
