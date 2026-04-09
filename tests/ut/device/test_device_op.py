from unittest.mock import patch

import pytest
import torch
from types import SimpleNamespace

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def test_npu_moe_init_routing_falls_back_to_v2_when_custom_binary_missing():
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    expected = (
        torch.randn(4, 4),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int64),
        None,
    )

    with patch(
        "vllm_ascend.device.device_op.get_current_vllm_config",
        side_effect=AssertionError,
    ), patch(
        "torch.ops._C_ascend.npu_moe_init_routing_custom",
        side_effect=RuntimeError(
            "Parse dynamic kernel config fail. Op MoeInitRoutingCustom does not has any binary."
        ),
        create=True,
    ) as mock_custom, patch(
        "torch_npu.npu_moe_init_routing_v2",
        return_value=expected,
    ) as mock_v2:
        result = BaseDeviceAdaptor.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            active_num=4,
            expert_num=2,
        )

    assert result is expected
    mock_custom.assert_called_once()
    mock_v2.assert_called_once()


def test_npu_moe_init_routing_reraises_unrelated_runtime_error():
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    with patch(
        "vllm_ascend.device.device_op.get_current_vllm_config",
        side_effect=AssertionError,
    ), patch(
        "torch.ops._C_ascend.npu_moe_init_routing_custom",
        side_effect=RuntimeError("some other failure"),
        create=True,
    ):
        with pytest.raises(RuntimeError, match="some other failure"):
            BaseDeviceAdaptor.npu_moe_init_routing(
                hidden_states,
                topk_ids,
                active_num=4,
                expert_num=2,
            )


def test_npu_moe_init_routing_prefers_v2_for_gemma4():
    hidden_states = torch.randn(2, 4)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    expected = (
        torch.randn(4, 4),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int64),
        None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="gemma4"),
        )
    )

    with patch(
        "vllm_ascend.device.device_op.get_current_vllm_config",
        return_value=vllm_config,
    ), patch(
        "torch_npu.npu_moe_init_routing_v2",
        return_value=expected,
    ) as mock_v2, patch(
        "torch.ops._C_ascend.npu_moe_init_routing_custom",
        create=True,
    ) as mock_custom:
        result = BaseDeviceAdaptor.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            active_num=4,
            expert_num=2,
        )

    assert result is expected
    mock_v2.assert_called_once()
    mock_custom.assert_not_called()


def test_a5_device_adaptor_keeps_mxfp_scale_override():
    scale = torch.randn(2, 4)

    normalized = A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(scale)

    assert normalized.shape == (2, 2, 2)
    assert torch.equal(normalized.reshape(2, 4), scale)
