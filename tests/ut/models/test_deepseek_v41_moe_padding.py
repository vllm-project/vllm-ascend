# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState


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


@pytest.mark.parametrize(
    "enabled,mode,phases,expected",
    [
        (True, CUDAGraphMode.NONE, (True,), True),
        (True, CUDAGraphMode.NONE, (True, False), False),
        (True, CUDAGraphMode.NONE, (False,), False),
        (True, CUDAGraphMode.FULL, (True,), False),
        (True, CUDAGraphMode.PIECEWISE, (True,), False),
        (False, CUDAGraphMode.NONE, (True,), False),
    ],
)
def test_only_pure_eager_prefill_bypasses_route_filter(monkeypatch, enabled, mode, phases, expected):
    monkeypatch.setattr(AscendModelState, "prepare_inputs", lambda *args: {})
    state = EngramModelState.__new__(EngramModelState)
    state.model = SimpleNamespace(skip_moe_padding=enabled)
    state._cg_mode = mode
    batch = SimpleNamespace(has_prefill=any(phases), is_prefilling_np=np.array(phases))

    assert state.prepare_inputs(batch, None) == ({"is_pure_eager_prefill": True} if expected else {})


@pytest.mark.parametrize("rows", [13, 16])
@pytest.mark.parametrize("pure_eager_prefill", [False, True])
def test_forward_preserves_global_padding_layout(monkeypatch, rows, pure_eager_prefill):
    mask = torch.arange(rows) >= 6
    context = SimpleNamespace(is_padding=mask, additional_kwargs={})
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 8)
    model = Mock()
    ids = torch.arange(rows, dtype=torch.int32)
    model_module.AscendDeepseekV41LLMForCausalLM.forward(
        SimpleNamespace(model=model, skip_moe_padding=True),
        ids,
        torch.arange(rows),
        is_pure_eager_prefill=pure_eager_prefill,
    )
    if pure_eager_prefill or rows == 16:
        assert context.is_padding is mask
    else:
        torch.testing.assert_close(context.is_padding, torch.arange(16) >= 6)
    assert context.additional_kwargs == ({"skip_moe_padding_filter": True} if pure_eager_prefill else {})
    assert model.call_args.args[0] is ids
