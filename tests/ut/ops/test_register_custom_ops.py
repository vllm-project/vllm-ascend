from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops import register_custom_ops


@patch.object(register_custom_ops._EXTRA_CTX, "flash_comm_v1_enabled", False)
@patch("vllm_ascend.ops.register_custom_ops.enable_sp_by_pass", return_value=True)
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_world_size", return_value=4)
@patch("vllm_ascend.ops.register_custom_ops.get_ep_group")
def test_fake_ep_communication_uses_ep_world_size(
    mock_get_ep_group,
    mock_get_tp_world_size,
    mock_enable_sp_by_pass,
):
    """SP-pass fake communication must model the dp4/tp4 EP group."""
    mock_get_ep_group.return_value = MagicMock(world_size=16)
    input_ = torch.empty(16, 8)

    gathered = register_custom_ops._maybe_all_gather_and_maybe_unpad_fake(input_, True, is_ep_comm=True)
    reduced = register_custom_ops._maybe_pad_and_reduce_fake(gathered, is_ep_comm=True)

    assert gathered.shape == (256, 8)
    assert reduced.shape == input_.shape
    mock_get_tp_world_size.assert_not_called()
