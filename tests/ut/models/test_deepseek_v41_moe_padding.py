# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import model as model_module


@pytest.mark.parametrize(
    "section,option,value",
    [
        (None, None, None),
        ("env", "VLLM_MOE_SKIP_PADDING", False),
        ("runner", "v2", False),
        ("parallel", "use_sequence_parallel_moe", False),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "use_ubatching", True),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "tensor_parallel_size", 1),
        ("parallel", "expert_placement_strategy", "round_robin"),
        ("parallel", "enable_eplb", True),
        ("ascend", "mix_placement", True),
        ("ascend", "enable_force_eplb", True),
        ("speculative", "enable_adaptive_verification", True),
        ("communication", "method", MoECommType.MC2),
        ("communication", "method", MoECommType.FUSED_MC2),
        ("communication", "method", MoECommType.ALLTOALL),
    ],
)
def test_skip_moe_padding_requires_supported_layout(monkeypatch, section, option, value):
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=True,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        use_ubatching=False,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        expert_placement_strategy="linear",
        enable_eplb=False,
    )
    ascend = SimpleNamespace(mix_placement=False, enable_force_eplb=False)
    speculative = SimpleNamespace(enable_adaptive_verification=False)
    if section in ("parallel", "ascend", "speculative"):
        setattr({"parallel": parallel, "ascend": ascend, "speculative": speculative}[section], option, value)
    config = SimpleNamespace(parallel_config=parallel, speculative_config=speculative)
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", section != "env")
    monkeypatch.setattr(model_module, "use_v2_model_runner", lambda config: section != "runner")
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        model_module,
        "select_moe_comm_method",
        lambda num_tokens, config: value if section == "communication" else MoECommType.ALLGATHER,
    )
    assert model_module._can_skip_moe_padding(config) is (section is None)
    if section is None:
        config.speculative_config = None
        assert model_module._can_skip_moe_padding(config)


@pytest.mark.parametrize("masked", [False, True])
def test_forward_publishes_only_the_current_routing_mask(monkeypatch, masked):
    context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    model = Mock()
    ids = torch.tensor([17, 42], dtype=torch.int32)
    mask = torch.tensor([False, True]) if masked else None
    model_module.AscendDeepseekV41LLMForCausalLM.forward(
        SimpleNamespace(model=model), ids, torch.arange(2), routing_padding_mask=mask
    )
    if masked:
        assert context.additional_kwargs["routing_padding_mask"] is mask
    else:
        assert "routing_padding_mask" not in context.additional_kwargs
    assert model.call_args.args[0] is ids
