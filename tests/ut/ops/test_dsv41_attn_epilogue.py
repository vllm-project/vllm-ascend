# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.dsv41_attn_epilogue import prepare_attn_epilogue_weights


@pytest.mark.parametrize("name", ["wo_a", "wo_b"])
@pytest.mark.parametrize("layout", ["nd", "nz"])
def test_pack_preserves_weight_and_scale_bytes(monkeypatch, name, layout):
    from vllm_ascend.ops import dsv41_attn_epilogue as module

    weight_bytes = torch.arange(8 * 64).remainder(255).to(torch.uint8).reshape(8, 64)
    scale = torch.arange(16).to(torch.uint8).reshape(8, 1, 2)
    layer = SimpleNamespace(
        prefix=f"model.layers.0.self_attn.{name}",
        weight=torch.nn.Parameter(weight_bytes.view(torch.float8_e4m3fn), requires_grad=False),
        weight_scale=torch.nn.Parameter(scale, requires_grad=False),
    )
    source_ptr = layer.weight.data_ptr()
    pack = Mock(side_effect=lambda tensor, fmt: tensor)
    monkeypatch.setattr(module.torch_npu, "npu_format_cast", pack)
    prepare_attn_epilogue_weights(layer, layout, 4)
    expected_w = weight_bytes.reshape(2, 4, 64) if name == "wo_a" else weight_bytes
    expected_s = scale.reshape(2, 4, 1, 2) if name == "wo_a" else scale
    if layout == "nz":
        expected_w = expected_w.transpose(-2, -1).contiguous()
        expected_s = expected_s.transpose(-3, -2).contiguous()
        pack.assert_called_once()
        assert pack.call_args.args[1] == 29
    else:
        pack.assert_not_called()
        assert layer.weight.data_ptr() == source_ptr
    torch.testing.assert_close(layer.weight.view(torch.uint8), expected_w)
    torch.testing.assert_close(layer.weight_scale, expected_s)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("capturing", [False, True])
def test_forward_folds_inverse_once_and_preserves_destination(monkeypatch, fused, capturing):
    from vllm_ascend.attention import dsa_v41 as module

    active, padded = 2, 4
    hidden = torch.zeros(padded, 5120, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    raw_attention = torch.randn(active, 64, 512, dtype=torch.bfloat16)
    cos, sin = torch.ones(active, 1, 1, 64), torch.zeros(active, 1, 1, 64)
    metadata = SimpleNamespace(
        swa=SimpleNamespace(num_actual_tokens=active),
        positions=torch.arange(padded),
        rope=lambda name, tokens: (cos, sin),
    )
    impl = object.__new__(module.AscendDSAV41Impl)
    impl._get_layer_metadata = Mock(return_value=metadata)
    impl._prepare_inputs_and_caches = Mock(return_value=(None, None))
    impl._select_sparse_indices = Mock(return_value=None)
    impl._forward_attention = Mock(return_value=raw_attention)
    impl._project_output = Mock()
    rotary = Mock()
    monkeypatch.setattr(module.AscendDSAV41Impl, "_apply_rotary", rotary)
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    epilogue = Mock()
    stream, erase = Mock(), Mock()
    order = Mock()
    order.attach_mock(epilogue, "epilogue")
    order.attach_mock(stream, "stream")
    order.attach_mock(erase, "erase")
    monkeypatch.setattr(module.torch.npu, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda device: stream)
    monkeypatch.setitem(sys.modules, "torch_npu.npu.utils", SimpleNamespace(_erase_stream=erase))
    projection = SimpleNamespace(
        weight=torch.nn.Parameter(torch.empty(1), requires_grad=False),
        weight_scale=torch.nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False),
    )
    attn = SimpleNamespace(
        fused_attn_epilogue=epilogue if fused else None,
        wo_a=projection,
        wo_b=projection,
        rope_head_dim=64,
        rotary_emb=SimpleNamespace(layername="test"),
    )
    assert impl.forward(attn, None, hidden, output) is output
    if fused and not capturing:
        stream.record_event.assert_called_once_with()
        stream.wait_event.assert_called_once_with(stream.record_event.return_value)
        erase.assert_called_once_with(raw_attention, stream)
        assert [entry[0] for entry in order.mock_calls] == [
            "epilogue",
            "stream.record_event",
            "stream.wait_event",
            "erase",
        ]
    else:
        stream.record_event.assert_not_called()
        erase.assert_not_called()
    if fused:
        epilogue.assert_called_once()
        assert epilogue.call_args.args[0] is raw_attention
        for weight in epilogue.call_args.args[1:3]:
            assert type(weight) is torch.Tensor
            assert weight.data_ptr() == projection.weight.data_ptr()
        assert epilogue.call_args.args[-1] is output
        assert epilogue.call_args.args[5].shape == (active, 64)
        rotary.assert_not_called()
        impl._project_output.assert_not_called()
    else:
        rotary.assert_called_once_with(attn, raw_attention, cos, sin, inverse=True)
        impl._project_output.assert_called_once_with(attn, raw_attention, hidden, projected=output)
