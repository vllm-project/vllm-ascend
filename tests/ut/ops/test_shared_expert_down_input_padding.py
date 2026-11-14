# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for moving shared-expert token padding before Down."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    SharedExpertMLPPath,
    SharedExpertParallelMode,
    SubmittedSharedExpertGateUp,
)


@pytest.fixture
def shared(monkeypatch):
    monkeypatch.setattr(moe_utils, "_PAD_ZERO_BLOCKS", {})
    value = AscendSharedExperts.__new__(AscendSharedExperts)
    value.layer = SimpleNamespace(expert_gate=None)
    value.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=16))
    return value


@pytest.mark.parametrize("rows", [1, 8, 15, 16, 17, 31, 32, 64])
def test_down_input_padding_preserves_rows_and_zero_fills(shared, rows):
    x = torch.randint(-127, 128, (rows, 384), dtype=torch.int8)
    scale = torch.rand(rows, dtype=torch.float32)
    out_x, out_scale = shared._pad_a8_down_input(x, scale, SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY)
    padded_rows = ((rows + 15) // 16) * 16
    assert out_x.shape == (padded_rows, 384)
    assert out_scale.shape == (padded_rows,)
    torch.testing.assert_close(out_x[:rows], x, atol=0, rtol=0)
    torch.testing.assert_close(out_scale[:rows], scale, atol=0, rtol=0)
    assert torch.count_nonzero(out_x[rows:]) == 0
    assert torch.count_nonzero(out_scale[rows:]) == 0
    if rows == padded_rows:
        assert out_x is x and out_scale is scale


@pytest.mark.parametrize("unsupported", ["mode", "gate", "rank", "scale_rank", "length", "x_dtype", "scale_dtype"])
def test_unsupported_down_inputs_keep_original_path(shared, unsupported):
    x = torch.ones((8, 384), dtype=torch.int8)
    scale = torch.ones(8, dtype=torch.float32)
    mode = SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    if unsupported == "mode":
        mode = SharedExpertParallelMode.TENSOR_PARALLEL
    elif unsupported == "gate":
        shared.layer.expert_gate = MagicMock()
    elif unsupported == "rank":
        x = x.unsqueeze(0)
    elif unsupported == "scale_rank":
        scale = scale.unsqueeze(1)
    elif unsupported == "length":
        scale = scale[:-1]
    elif unsupported == "x_dtype":
        x = x.float()
    elif unsupported == "scale_dtype":
        scale = scale.to(torch.bfloat16)
    result = shared._pad_a8_down_input(x, scale, mode)
    assert result[0] is x and result[1] is scale


def test_linear_rows_match_padding_after_down(shared):
    torch.manual_seed(42)
    x = torch.randint(-10, 11, (8, 32), dtype=torch.int8)
    weight = torch.randint(-10, 11, (32, 64), dtype=torch.int8)
    scale = torch.rand(8)
    padded_x, padded_scale = shared._pad_a8_down_input(x, scale, SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY)
    reference = torch.nn.functional.pad((x.float() @ weight.float()) * scale[:, None], (0, 0, 0, 8))
    actual = (padded_x.float() @ weight.float()) * padded_scale[:, None]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_fused_mlp_pads_down_inputs_before_quant_matmul(shared, monkeypatch):
    x = torch.arange(8 * 4, dtype=torch.int8).reshape(8, 4)
    scale = torch.arange(1, 9, dtype=torch.float32)
    weight = torch.ones((4, 3), dtype=torch.int8)
    shared.layer.gate_up_proj = SimpleNamespace(weight_scale_fp32=torch.ones(8))
    shared.layer.down_proj = SimpleNamespace(weight=weight, weight_scale=torch.ones(3))
    shared.situ_activation = SimpleNamespace(beta=1.0, linear_beta=0.0)
    shared.multistream_overlap = False
    monkeypatch.setattr(
        shared_experts_module.torch.ops,
        "_C_ascend",
        SimpleNamespace(dequant_situ_quant=lambda **_: (x, scale)),
    )

    def matmul(activation, kernel_weight, _weight_scale, *, pertoken_scale, **_kwargs):
        return (activation.float() @ kernel_weight.float()) * pertoken_scale[:, None]

    monkeypatch.setattr(shared_experts_module.torch_npu, "npu_quant_matmul", matmul, raising=False)
    submitted = SubmittedSharedExpertGateUp(
        hidden_states=torch.ones((8, 3)),
        gate_up=torch.ones((8, 8), dtype=torch.int32),
        gate_up_input_scale=torch.ones(8),
        local_dp_metadata=None,
        path=SharedExpertMLPPath.A8_INT_FUSED,
    )
    actual = shared._finish_shared_mlp(
        submitted, RoutedMoEMilestones(), SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    )
    reference = torch.nn.functional.pad((x.float() @ weight.float()) * scale[:, None], (0, 0, 0, 8))
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
