# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import model as model_module


@pytest.mark.parametrize(
    "section,option,value",
    [
        (None, None, None),
        ("env", "VLLM_MOE_SKIP_PADDING", False),
        ("runner", "v2", False),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "expert_placement_strategy", "round_robin"),
        ("parallel", "enable_eplb", True),
        ("ascend", "mix_placement", True),
        ("ascend", "enable_force_eplb", True),
        ("communication", "method", MoECommType.MC2),
        ("communication", "method", MoECommType.FUSED_MC2),
        ("communication", "method", MoECommType.ALLTOALL),
    ],
)
@pytest.mark.parametrize(
    "dp_size,pcp_size,tp_size",
    [(1, 1, 1), (1, 1, 8), (2, 1, 1), (2, 1, 4), (1, 4, 1), (2, 4, 1), (2, 2, 2)],
)
def test_skip_moe_padding_requires_supported_layout(monkeypatch, section, option, value, dp_size, pcp_size, tp_size):
    parallel = SimpleNamespace(
        data_parallel_size=dp_size,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=tp_size,
        expert_placement_strategy="linear",
        enable_eplb=False,
    )
    ascend = SimpleNamespace(mix_placement=False, enable_force_eplb=False)
    if section in ("parallel", "ascend"):
        setattr({"parallel": parallel, "ascend": ascend}[section], option, value)
    config = SimpleNamespace(parallel_config=parallel, speculative_config=None)
    ep_size = dp_size * pcp_size * tp_size
    monkeypatch.setattr(model_module, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", section != "env")
    monkeypatch.setattr(model_module, "use_v2_model_runner", lambda config: section != "runner")
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        model_module,
        "select_moe_comm_method",
        lambda num_tokens, config: value if section == "communication" else MoECommType.ALLGATHER,
    )
    enabled = section is None and ep_size > 1
    assert model_module._can_skip_padding_tokens_in_moe(config) is enabled
    if enabled:
        config.speculative_config = SimpleNamespace(enable_adaptive_verification=True)
        assert model_module._can_skip_padding_tokens_in_moe(config)
