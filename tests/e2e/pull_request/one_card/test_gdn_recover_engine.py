# SPDX-License-Identifier: Apache-2.0
"""Greedy engine oracle across mixed requests and repeated cache-slot reuse."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


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
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in (2, 9, 48, 96)]
    kwargs = dict(
        model_name="Qwen/Qwen3.5-4B",
        dtype="bfloat16",
        language_model_only=True,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
    )
    # The target without speculation is an independent engine oracle. Native
    # speculative kernels have a separate ragged-state indexing dependency.
    with VllmRunner(**kwargs) as runner:
        reference = [tuple(o.outputs[0].token_ids) for o in runner.model.generate(prompts, params)]
    with VllmRunner(
        **kwargs,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True},
        additional_config={"enable_gdn_recover": True},
    ) as runner:
        for order in ([0, 1, 2, 3], [3, 1, 0, 2], [2, 0, 3, 1]):
            outputs = runner.model.generate([prompts[i] for i in order], [params[i] for i in order])
            assert [tuple(o.outputs[0].token_ids) for o in outputs] == [reference[i] for i in order]
