# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.dsv41_attn_epilogue import prepare_attn_epilogue_weights


@pytest.mark.parametrize("name", ["wo_a", "wo_b"])
def test_prepare_preserves_shared_weight_and_scale_bytes(name):
    weight_bytes = torch.arange(8 * 64).remainder(255).to(torch.uint8).reshape(8, 64)
    scale = torch.arange(16).to(torch.uint8).reshape(8, 1, 2)
    layer = SimpleNamespace(
        prefix=f"model.layers.0.self_attn.{name}",
        weight=torch.nn.Parameter(weight_bytes.view(torch.float8_e4m3fn), requires_grad=False),
        weight_scale=torch.nn.Parameter(scale, requires_grad=False),
    )
    source_ptr = layer.weight.data_ptr()
    source_scale_ptr = layer.weight_scale.data_ptr()
    prepare_attn_epilogue_weights(layer, 4)
    expected_w = weight_bytes.reshape(2, 4, 64) if name == "wo_a" else weight_bytes.t()
    expected_s = scale.reshape(2, 4, 1, 2) if name == "wo_a" else scale.transpose(0, 1)
    assert layer.weight.data_ptr() == source_ptr
    assert layer.weight_scale.data_ptr() == source_scale_ptr
    torch.testing.assert_close(layer.weight.view(torch.uint8), expected_w)
    torch.testing.assert_close(layer.weight_scale, expected_s)


@pytest.mark.parametrize(
    ("available", "padded", "fused"),
    [(False, 4, False), (True, 4, True), (True, 384, True), (True, 385, False)],
)
@pytest.mark.parametrize("capturing", [False, True])
def test_forward_folds_inverse_once_and_preserves_destination(monkeypatch, available, padded, fused, capturing):
    from vllm_ascend.attention import dsa_v41 as module

    active = 2
    hidden = torch.zeros(padded, 8, dtype=torch.bfloat16)
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
    wo_a = SimpleNamespace(
        weight=torch.nn.Parameter(torch.empty(2, 3, 4), requires_grad=False),
        weight_scale=torch.nn.Parameter(torch.zeros((2, 3, 1, 2), dtype=torch.uint8), requires_grad=False),
    )
    wo_b = SimpleNamespace(
        weight=torch.nn.Parameter(torch.empty(4, 3).t(), requires_grad=False),
        weight_scale=torch.nn.Parameter(torch.zeros((4, 1, 2), dtype=torch.uint8).transpose(0, 1), requires_grad=False),
    )
    attn = SimpleNamespace(
        fused_attn_epilogue=epilogue if available else None,
        wo_a=wo_a,
        wo_b=wo_b,
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
        for weight, projection in zip(epilogue.call_args.args[1:3], (wo_a, wo_b)):
            assert type(weight) is torch.Tensor
            assert weight.data_ptr() == projection.weight.data_ptr()
        assert epilogue.call_args.args[2].shape == (4, 3)
        assert epilogue.call_args.args[2].is_contiguous()
        assert epilogue.call_args.args[4].shape == (4, 1, 2)
        assert epilogue.call_args.args[-1] is output
        assert epilogue.call_args.args[5].shape == (active, 64)
        rotary.assert_not_called()
        impl._project_output.assert_not_called()
    else:
        epilogue.assert_not_called()
        rotary.assert_called_once_with(attn, raw_attention, cos, sin, inverse=True)
        impl._project_output.assert_called_once_with(attn, raw_attention, hidden, projected=output)


def test_optional_epilogue_provider(monkeypatch):
    from vllm_ascend.ops import dsv41_attn_epilogue as module

    loader = module.get_attn_epilogue_out.__wrapped__
    provider = Mock(side_effect=ModuleNotFoundError(name="ops.attn_epilogue"))
    monkeypatch.setattr(module, "import_module", provider)
    assert loader() is None
    provider.side_effect = None
    provider.return_value = SimpleNamespace()
    assert loader() is None
    op = Mock()
    provider.return_value.attn_epilogue_out_op = op
    assert loader() is op


@pytest.mark.parametrize(
    "device_type,producer,packaged",
    [("A5", False, True), ("A2", False, True), ("A5", True, True), ("A5", False, False)],
)
def test_model_selects_epilogue_automatically(monkeypatch, device_type, producer, packaged):
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.models.deepseek_v41 import model
    from vllm_ascend.ops import dsv41_attn_epilogue as epilogue_module
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod

    scheme = Mock(spec=AscendW8A8MXFP8DSDynamicLinearMethod)

    def new_projection(*args, **kwargs):
        return SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))

    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(model, name, new_projection)
    monkeypatch.setattr(model, "RMSNorm", Mock())
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(model.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        model, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType[device_type])
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer, is_kv_consumer=not producer),
        lora_config=None,
    )
    monkeypatch.setattr(model, "get_current_vllm_config", lambda: vllm_config)
    op = Mock() if packaged else None
    load = Mock(return_value=op)
    monkeypatch.setattr(epilogue_module, "get_attn_epilogue_out", load)
    config = SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=8,
        sliding_window=128,
        rms_norm_eps=1e-6,
    )
    attn = SimpleNamespace()
    model.init_attention_projections(attn, config, None, "attn", reduce_results=False)

    eligible = device_type == "A5" and not producer
    assert load.call_count == int(eligible)
    assert attn.fused_attn_epilogue is (op if eligible else None)
    assert hasattr(attn.wo_a, "_dsv41_epilogue_nd") == (eligible and packaged)
    assert hasattr(attn.wo_b, "_dsv41_epilogue_nd") == (eligible and packaged)
