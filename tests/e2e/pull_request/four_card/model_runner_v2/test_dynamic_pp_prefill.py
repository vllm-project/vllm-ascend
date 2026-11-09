# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os

import pytest
from vllm import SamplingParams
from vllm.distributed.utils import get_pp_indices

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"
PROMPTS = [
    "Explain why the sky is blue in one sentence.",
    "Calculate 17 plus 25. The answer is",
    "Describe the purpose of a pipeline in parallel computing.",
    "Translate the word 'hello' into Chinese.",
]


@wait_until_npu_memory_free()
def _generate(additional_config):
    with VllmRunner(
        MODEL,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        async_scheduling=True,
        seed=7,
        additional_config={"enable_dsa_cp": False, "enable_flashcomm1": False, **additional_config},
    ) as runner:
        params = SamplingParams(temperature=0, seed=7, max_tokens=1, logprobs=5)
        # Hold batch shapes fixed across cuts; quantized kernels need not be
        # numerically invariant to asynchronous request packing.
        outputs = [output for prompt in PROMPTS for output in runner.generate_w_logprobs([prompt], params)]
        num_layers = runner.model.llm_engine.vllm_config.model_config.hf_config.num_hidden_layers
        _, base_boundary = get_pp_indices(num_layers, 0, 2)
        return outputs, runner.model.collective_rpc("get_dynamic_pp_stats"), base_boundary


@pytest.fixture(scope="module")
def fixed_pp_outputs():
    previous = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    try:
        yield _generate({})
    finally:
        if previous is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous


@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize(
    "offset",
    [None, -1, 0, 1, "adaptive"],
    ids=["buffer-only", "move-downstream", "base", "move-upstream", "adaptive"],
)
def test_dsv4_prefill_matches_fixed_pipeline(fixed_pp_outputs, offset):
    """Use a real checkpoint to check buffering and both boundary directions."""
    config = {"pp_send_buffer_size": 3}
    if offset is not None:
        config["dynamic_pp_config"] = {
            "enabled": True,
            "redundant_layers": 2,
            "initial_offset": 0 if offset == "adaptive" else offset,
            "adaptive": offset == "adaptive",
            "cooldown_steps": 1,
        }
    actual, stats, base_boundary = _generate(config)
    baseline_outputs, _, _ = fixed_pp_outputs
    assert any(row["buffer_sends"] > 0 for row in stats)
    if offset is not None:
        assert all(row["enabled"] and row["batches"] > 0 for row in stats)
        assert len({row["boundary"] for row in stats}) == 1, "PP/TP workers disagreed on the final boundary"
        assert len({row["boundary_version"] for row in stats}) == 1, "PP/TP workers disagreed on the boundary version"
    if offset == "adaptive":
        # A balanced run may correctly hold its initial cut. Forced movement
        # would reintroduce the posted-send-watermark regression.
        assert any(
            row["device_feedback"] is not None and row["device_feedback"]["completed_timing_samples"] > 0
            for row in stats
        ), "Adaptive run never observed device-completed work"
    elif offset is not None:
        assert all(row["boundary"] == base_boundary + offset for row in stats), "Requested offset did not take effect"
    assert len(actual) == len(baseline_outputs)
    for output, baseline in zip(actual, baseline_outputs):
        assert output[0] == baseline[0], "Dynamic PP changed the greedy output token"
        assert output[2] is not None and baseline[2] is not None
        for token, logprob in baseline[2][0].items():
            assert token in output[2][0], "Dynamic PP changed the top logprob candidates"
            assert output[2][0][token].logprob == pytest.approx(logprob.logprob, abs=1e-3)
