# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Optional two-card acceptance test; requires a compatible ZercMoE wheel."""

from importlib.util import find_spec

import pytest
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import _check_decode_token, _check_prefill_token
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
PROMPTS = [
    "Explain how expert parallelism distributes tokens and combines expert outputs. " * 12,
    "Describe how prefill and decode use a language model's key and value cache. " * 14,
]


class ZercMoEProbeWorker:
    def install_zercmoe_probe(self):
        from vllm_ascend.ops.fused_moe.moe_comm_method import ZercMoECommImpl

        self._zercmoe_calls = 0
        original = ZercMoECommImpl.fused_experts

        def counted_call(comm, *args, **kwargs):
            self._zercmoe_calls += 1
            return original(comm, *args, **kwargs)

        ZercMoECommImpl.fused_experts = counted_call

    def read_zercmoe_calls(self):
        return self._zercmoe_calls


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="zercmoe,aclgraph,logprobs",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@pytest.mark.parametrize("runner_v2", [False, True])
@wait_until_npu_memory_free(0.7)
def test_zercmoe_prefill_matches_baseline_and_keeps_decode_graphs(monkeypatch, runner_v2):
    if find_spec("zercmoe") is None:
        pytest.skip("A compatible ZercMoE wheel is required")
    if not get_current_hardware_profile().supports(HardwareCapability.ZERCMOE):
        pytest.skip("ZercMoE is supported only on A2")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(runner_v2)))
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ZERC_MOE", "0")
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, logprobs=20)
    settings = {
        "dtype": "bfloat16",
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 4,
        "enable_prefix_caching": False,
        "seed": 1024,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    }
    with VllmRunner(MODEL, **settings, additional_config={"enable_fused_mc2": 0, "enable_zerc_moe": 0}) as runner:
        baseline = runner.model.generate(PROMPTS, params)

    with VllmRunner(
        MODEL,
        **settings,
        worker_extension_cls=f"{__name__}.ZercMoEProbeWorker",
        additional_config={
            "enable_fused_mc2": 1,
            "enable_zerc_moe": 1,
            "enable_flashcomm1": False,
            "zercmoe_ipport": f"tcp://127.0.0.1:{get_open_port()}",
        },
    ) as runner:
        runner.model.collective_rpc("install_zercmoe_probe")
        actual = runner.model.generate(PROMPTS, params)
        # Accuracy alone could pass after a silent fallback. Check both EP ranks
        # actually executed the fused operator on these long prefills.
        assert all(count > 0 for count in runner.model.collective_rpc("read_zercmoe_calls"))

    for index, (base, output) in enumerate(zip(baseline, actual, strict=True)):
        base_seq, actual_seq = base.outputs[0], output.outputs[0]
        assert len(base_seq.token_ids) == len(actual_seq.token_ids) == 3
        _check_prefill_token(base_seq, actual_seq, index, 0.0689)
        for token_index in (1, 2):
            _check_decode_token(base_seq, actual_seq, token_index, index, 0.1378)
