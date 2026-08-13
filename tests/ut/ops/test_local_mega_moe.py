from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2


def _make_comm_impl():
    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    return comm_impl


@patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config")
def test_swigluoai_uninterleave_uses_local_mega_moe(mock_get_ascend_config):
    mock_get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=1)
    comm_impl = _make_comm_impl()
    comm_impl._apply_local_mega_moe = MagicMock(return_value=(torch.empty(1), torch.empty(1)))

    comm_impl.fused_experts(SimpleNamespace(activation="swigluoai_uninterleave", weights=SimpleNamespace(
        w1_scale=torch.empty(1), w2_scale=torch.empty(1))))

    comm_impl._apply_local_mega_moe.assert_called_once()


@patch("vllm_ascend.ops.fused_moe.moe_comm_method._MEGA_MOE_SUPPORTED", True)
@patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config")
def test_other_activation_keeps_existing_mega_moe(mock_get_ascend_config):
    mock_get_ascend_config.return_value = SimpleNamespace(enable_fused_mc2=1)
    comm_impl = _make_comm_impl()
    comm_impl._apply_cann_mega_moe = MagicMock(return_value=(torch.empty(1), torch.empty(1)))

    comm_impl.fused_experts(SimpleNamespace(activation="silu", weights=SimpleNamespace(
        w1_scale=torch.empty(1), w2_scale=torch.empty(1))))

    comm_impl._apply_cann_mega_moe.assert_called_once()
