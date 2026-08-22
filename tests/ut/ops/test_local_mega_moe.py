from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import (
    FusedMC2CommImpl,
    _as_local_mega_moe_tensor_list,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2


def _make_comm_impl():
    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    return comm_impl


def test_local_mega_moe_splits_stacked_expert_tensors():
    stacked_weights = torch.empty(4, 8, 16)
    stacked_scales = torch.empty(4, 8)

    weights = _as_local_mega_moe_tensor_list([stacked_weights], 2, "weight")
    scales = _as_local_mega_moe_tensor_list([stacked_scales], 1, "scale")

    assert len(weights) == 4
    assert all(weight.shape == (8, 16) for weight in weights)
    assert len(scales) == 4
    assert all(scale.shape == (8,) for scale in scales)


def test_local_mega_moe_splits_flat_expert_scales():
    flat_scales = torch.empty(32)
    scales = _as_local_mega_moe_tensor_list([flat_scales], 1, "scale", num_experts=4)

    assert len(scales) == 4
    assert all(scale.shape == (8,) for scale in scales)
    assert all(scale.untyped_storage().data_ptr() == flat_scales.untyped_storage().data_ptr() for scale in scales)


def test_local_mega_moe_preserves_existing_expert_lists():
    weights = [torch.empty(8, 16), torch.empty(8, 16)]
    normalized = _as_local_mega_moe_tensor_list(weights, 2, "weight")
    assert normalized is weights


@patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config")
def test_swigluoai_uninterleave_uses_local_mega_moe(mock_get_ascend_config):
    mock_get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=1)
    comm_impl = _make_comm_impl()
    comm_impl._apply_local_mega_moe = MagicMock(return_value=(torch.empty(1), torch.empty(1)))

    with patch.object(torch.ops._C_ascend, "mega_moe", create=True):
        comm_impl.fused_experts(
            SimpleNamespace(
                activation="swigluoai_uninterleave",
                weights=SimpleNamespace(w1_scale=torch.empty(1), w2_scale=torch.empty(1)),
            )
        )

    comm_impl._apply_local_mega_moe.assert_called_once()


@patch("vllm_ascend.ops.fused_moe.moe_comm_method._MEGA_MOE_SUPPORTED", True)
@patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config")
def test_other_activation_keeps_existing_mega_moe(mock_get_ascend_config):
    mock_get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=1)
    comm_impl = _make_comm_impl()
    comm_impl._apply_cann_mega_moe = MagicMock(return_value=(torch.empty(1), torch.empty(1)))

    comm_impl.fused_experts(
        SimpleNamespace(activation="silu", weights=SimpleNamespace(w1_scale=torch.empty(1), w2_scale=torch.empty(1)))
    )

    comm_impl._apply_cann_mega_moe.assert_called_once()
