# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.worker.v2 import input_batch as input_batch_module
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


@pytest.mark.parametrize(
    "section,option,value",
    [
        (None, None, None),
        ("env", "VLLM_MOE_SKIP_PADDING", False),
        ("runner", "v2", False),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "tensor_parallel_size", 1),
        ("parallel", "expert_placement_strategy", "round_robin"),
        ("parallel", "enable_eplb", True),
        ("ascend", "mix_placement", True),
        ("ascend", "enable_force_eplb", True),
        ("communication", "method", MoECommType.MC2),
        ("communication", "method", MoECommType.FUSED_MC2),
        ("communication", "method", MoECommType.ALLTOALL),
    ],
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_skip_moe_padding_requires_supported_layout(monkeypatch, section, option, value, sequence_parallel):
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=sequence_parallel,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        expert_placement_strategy="linear",
        enable_eplb=False,
    )
    ascend = SimpleNamespace(mix_placement=False, enable_force_eplb=False)
    if section in ("parallel", "ascend"):
        setattr({"parallel": parallel, "ascend": ascend}[section], option, value)
    config = SimpleNamespace(parallel_config=parallel, speculative_config=None)
    monkeypatch.setattr(model_module.envs, "VLLM_MOE_SKIP_PADDING", section != "env")
    monkeypatch.setattr(model_module, "use_v2_model_runner", lambda config: section != "runner")
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(
        model_module,
        "select_moe_comm_method",
        lambda num_tokens, config: value if section == "communication" else MoECommType.ALLGATHER,
    )
    assert model_module._can_skip_padding_tokens_in_moe(config) is (section is None)
    if section is None:
        config.speculative_config = SimpleNamespace(enable_adaptive_verification=True)
        assert model_module._can_skip_padding_tokens_in_moe(config)


@pytest.mark.parametrize("rows", [13, 16])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("has_graph_padding", [False, True])
def test_forward_preserves_global_padding_layout(monkeypatch, rows, sequence_parallel, has_graph_padding):
    parallel = SimpleNamespace(use_sequence_parallel_moe=sequence_parallel, tensor_parallel_size=8)
    config = SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr(input_batch_module, "get_current_vllm_config_or_none", lambda: config)
    buffers = AscendInputBuffers(1, rows, torch.device("cpu"))
    assert buffers.max_num_tokens == buffers.input_ids.numel() == buffers.positions.numel() == rows
    for tensor in (buffers.input_ids, buffers.positions):
        assert tensor.untyped_storage().nbytes() == rows * tensor.element_size()
    assert buffers.is_padding.numel() == (16 if sequence_parallel else rows)
    mask = buffers.is_padding[:rows]
    valid_rows = 6 if has_graph_padding else rows
    mask[valid_rows:].fill_(True)
    context = SimpleNamespace(is_padding=mask)
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 8)
    model = Mock(use_sequence_parallel=sequence_parallel, skip_padding_tokens_in_moe=True)
    ids = torch.arange(rows, dtype=torch.int32)
    model_module.AscendDeepseekV41LLMForCausalLM.forward(
        SimpleNamespace(model=model),
        ids,
        torch.arange(rows),
    )
    if not sequence_parallel or rows == 16:
        assert context.is_padding is mask
        assert not buffers.is_padding[rows:].any()
    else:
        assert context.is_padding.data_ptr() == mask.data_ptr()
        torch.testing.assert_close(context.is_padding, torch.arange(16) >= valid_rows)
    torch.testing.assert_close(mask, torch.arange(rows) >= valid_rows)
    assert model.call_args.args[0] is ids
