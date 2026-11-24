# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.attention.gdn_attn_builder import GDNCausalConv1dMetadata, GDNPrefillMetadata
from vllm_ascend.ops import kimi_kda
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic, w8a8_mxfp8


@pytest.mark.parametrize("chip", ["A2", "A3", "A5", "_310P"])
@pytest.mark.parametrize("backend", ["mega_recurrent_kda_int8", "mega_recurrent_kda_mx"])
def test_mega_backend_device_pair(monkeypatch, tmp_path, chip, backend):
    from vllm_ascend import ascend_config
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile

    monkeypatch.setattr(ascend_config, "is_950", lambda: chip == "A5")
    monkeypatch.setattr(
        ascend_config, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType[chip])
    )
    config = ascend_config.KDABackendConfig(backend=backend, operator_package_dir=str(tmp_path))
    allowed = chip in ("A2", "A3") if backend == "mega_recurrent_kda_int8" else chip == "A5"
    if allowed:
        config.validate_device()
    else:
        with pytest.raises(ValueError, match="MegaKDA backend requires"):
            config.validate_device()


def test_int8_qkv_loader_and_native_prefill(monkeypatch):
    from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiK3DeltaAttention

    method = w8a8_dynamic.AscendW8A8DynamicLinearMethod()
    layer = nn.Module()
    layer.quant_method = method
    weights = torch.randint(-127, 128, (384, 7168), dtype=torch.int8)
    scale = torch.arange(384, dtype=torch.bfloat16).reshape(-1, 1)
    layer.weight = nn.Parameter(weights.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(scale.clone(), requires_grad=False)
    layer.weight_offset = nn.Parameter(torch.zeros_like(scale), requires_grad=False)
    assert kimi_kda._configure_mega_qkv(layer, "mega_recurrent_kda_int8") == 0
    cast = MagicMock(side_effect=lambda weight, fmt: weight)
    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_format_cast", cast)
    method.process_weights_after_loading(layer)
    assert cast.call_count == 1 and cast.call_args.args[1] == 29
    torch.testing.assert_close(layer.weight, weights.T)
    torch.testing.assert_close(layer.weight_scale, scale.flatten())

    x = torch.zeros((3, 7168), dtype=torch.bfloat16)
    codes = torch.zeros_like(x, dtype=torch.int8)
    token_scale = torch.ones(3)
    quant = MagicMock(return_value=(codes, token_scale))
    expected = torch.zeros((3, 384), dtype=torch.bfloat16)
    mm = MagicMock(return_value=expected)
    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_dynamic_quant", quant, raising=False)
    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_quant_matmul", mm, raising=False)
    assert AscendKimiK3DeltaAttention._matmul_fused_qkv(SimpleNamespace(in_proj_qkvgfab=layer), x) is expected
    assert mm.call_args.args[1] is layer.weight and mm.call_args.args[2] is layer.weight_scale
    assert mm.call_args.kwargs["pertoken_scale"] is token_scale


@pytest.mark.parametrize(
    "backend,method_cls",
    [
        ("mega_recurrent_kda_int8", w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod),
        ("mega_recurrent_kda_mx", w8a8_dynamic.AscendW8A8DynamicLinearMethod),
    ],
)
def test_mega_rejects_wrong_checkpoint(backend, method_cls):
    layer = SimpleNamespace(quant_method=SimpleNamespace(quant_method=method_cls.__new__(method_cls)))
    with pytest.raises(ValueError, match="MegaKDA requires"):
        kimi_kda._configure_mega_qkv(layer, backend)


def test_mega_qkv_layout_and_reload(monkeypatch):
    layer = nn.Module()
    rows, hidden = 96, 7168
    raw = torch.randint(0, 255, (rows, hidden), dtype=torch.uint8)
    scales = torch.randint(0, 255, (rows, hidden // 32), dtype=torch.uint8)
    layer.weight = nn.Parameter(raw.view(torch.float8_e4m3fn).clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(scales.clone(), requires_grad=False)
    layer._mega_kda_mx_qkv = True
    method = w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod.__new__(w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod)
    method.group_size = 32
    cast = MagicMock(side_effect=lambda t, fmt: t.clone())
    monkeypatch.setattr(w8a8_mxfp8.torch_npu, "npu_format_cast", cast)
    method.process_weights_after_loading(layer)
    assert cast.call_args.args[0].shape == (1, hidden, rows)
    assert cast.call_args.args[1] == 29
    torch.testing.assert_close(layer.weight.view(torch.uint8), raw.T.unsqueeze(0))
    torch.testing.assert_close(layer.weight_scale, scales.view(rows, -1, 2).transpose(0, 1).unsqueeze(0))
    assert layer.weight_scale.is_contiguous()
    pointers = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    method.restore_weights_for_rl_loading(layer)
    torch.testing.assert_close(layer.weight.view(torch.uint8), raw)
    torch.testing.assert_close(layer.weight_scale, scales)
    layer.weight_scale.data.bitwise_xor_(1)
    method.process_weights_after_loading(layer)
    assert pointers == (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    torch.testing.assert_close(layer.weight_scale, scales.bitwise_xor(1).view(rows, -1, 2).transpose(0, 1).unsqueeze(0))


@pytest.mark.parametrize("int8", [False, True], ids=["mx", "int8"])
@pytest.mark.parametrize("mode", ["dummy", "decode", "first_token", "long_prefill", "mixed", "decode_prefill"])
def test_mega_kda_model_dispatch(monkeypatch, mode, int8):
    x = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    indices = torch.tensor([0, 2, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=int(mode in ("first_token", "long_prefill", "mixed", "decode_prefill")),
        num_prefill_tokens={"first_token": 1, "long_prefill": 5, "mixed": 3, "decode_prefill": 3}.get(mode, 0),
        num_decodes=2 if mode == "decode_prefill" else 0,
        num_decode_tokens=2 if mode == "decode_prefill" else 0,
        num_spec_decodes=int(mode == "mixed"),
        num_spec_decode_tokens=2 if mode == "mixed" else 0,
        num_actual_tokens=5,
        non_spec_token_indx=indices if mode == "mixed" else None,
        spec_token_indx=torch.tensor([1, 4], dtype=torch.int32) if mode == "mixed" else None,
        spec_sequence_masks=object() if mode == "mixed" else None,
    )
    if mode == "decode_prefill":
        prefix = torch.tensor([0, 1, 2, 5], dtype=torch.int32)
        state_slots = torch.tensor([2, 5, 7], dtype=torch.int32)
        conv_slots = torch.tensor([3, 1, 9], dtype=torch.int32)
        initial = torch.tensor([True, True, False])
        metadata.non_spec_query_start_loc = prefix
        metadata.non_spec_state_indices_tensor = state_slots
        metadata.prefill_query_start_loc = prefix[2:] - 2
        metadata.prefill_state_indices = state_slots[2:]
        metadata.prefill_has_initial_state = initial[2:]
        metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(prefix, conv_slots, initial), chunk=object()
        )
    context = SimpleNamespace(attn_metadata=None if mode == "dummy" else {"layer": metadata}, additional_kwargs={})
    monkeypatch.setattr(kimi_kda, "get_forward_context", lambda: context)
    for hook in (
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
    ):
        monkeypatch.setattr(kimi_kda, hook, MagicMock())
    upstream = MagicMock(
        side_effect=lambda layer, op, meta, hidden, weights, *, output, only_spec: output.copy_(hidden + 1)
    )
    monkeypatch.setattr(kimi_kda, "run_upstream_mega_kda", upstream)
    proj = MagicMock(side_effect=lambda hidden: (hidden, None, None, None))

    def native(hidden, g1, g2, beta, out, *, beta_is_preprocessed, native_prefill_metadata):
        assert beta_is_preprocessed
        assert native_prefill_metadata.num_spec_decodes == 0
        assert native_prefill_metadata.num_actual_tokens == hidden.size(0)
        assert native_prefill_metadata.spec_sequence_masks is None
        if mode == "decode_prefill":
            assert native_prefill_metadata.num_decodes == 0
            phase = native_prefill_metadata.non_spec_prefill_metadata
            assert phase.chunk is metadata.non_spec_prefill_metadata.chunk
            assert phase.causal_conv1d.query_start_loc.tolist() == [0, 3]
            assert phase.causal_conv1d.cache_indices.tolist() == [9]
            assert phase.causal_conv1d.initial_state_mode.tolist() == [False]
            assert native_prefill_metadata.non_spec_state_indices_tensor.tolist() == [7]
            torch.testing.assert_close(hidden, x[2:])
        out.copy_((hidden + 2).reshape_as(out))

    layer = SimpleNamespace(
        prefix="layer",
        _mega_kda_op=object(),
        _mega_kda_scale_alg=0 if int8 else 1,
        _fused_bfg_output_sizes=(1, 4, 4),
        _mega_output_norm_weight=MagicMock(),
        hidden_size=4,
        in_proj_qkvgfab=SimpleNamespace(
            _mega_kda_int8_qkv=int8,
            _mxfp8_transformed=not int8,
            weight=torch.zeros((4, 12), dtype=torch.int8) if int8 else MagicMock(),
            weight_scale=torch.ones(12, dtype=torch.bfloat16) if int8 else MagicMock(),
        ),
        fused_bfg_proj=SimpleNamespace(
            weight=torch.arange(36).view(9, 4), f_a_weight=MagicMock(), f_b_weight=MagicMock()
        ),
        get_parameter=MagicMock(),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(4),
        local_num_heads=1,
        head_dim=4,
        local_projection_size=4,
        _run_overlapped_qkv_bfg=proj,
        _forward=native,
        o_proj=SimpleNamespace(weight=MagicMock(), quant_method=SimpleNamespace(apply=lambda layer, h, bias: h)),
    )
    result = torch.empty_like(x)
    kimi_kda.AscendKimiK3DeltaAttention._forward_upstream_mega_kda(layer, x, result)
    expected = x + 1
    if mode == "dummy":
        expected.zero_()
    elif mode == "long_prefill":
        expected = x + 2
    elif mode == "mixed":
        expected[indices.long()] += 1
        assert metadata.num_spec_decodes == 1
    elif mode == "decode_prefill":
        expected[2:] += 1
        decode = upstream.call_args.args[2]
        assert decode.num_prefills == 0 and decode.num_decodes == 2 and decode.num_actual_tokens == 2
        assert decode.non_spec_decode_metadata.causal_conv1d.query_start_loc.tolist() == [0, 1, 2]
        assert decode.non_spec_decode_metadata.causal_conv1d.cache_indices.tolist() == [3, 1]
        assert decode.non_spec_state_indices_tensor.tolist() == [2, 5]
        assert metadata.non_spec_prefill_metadata.causal_conv1d.query_start_loc.tolist() == [0, 1, 2, 5]
        assert metadata.num_prefills == 1 and metadata.num_decodes == 2
        torch.testing.assert_close(upstream.call_args.args[3], x[:2])
    torch.testing.assert_close(result, expected)
    assert proj.call_count == int(mode in ("long_prefill", "mixed", "decode_prefill"))
    assert upstream.call_count == int(mode in ("decode", "first_token", "mixed", "decode_prefill"))
    if upstream.called:
        assert upstream.call_args.kwargs["output"].data_ptr() == result.data_ptr()
        assert upstream.call_args.kwargs["only_spec"] == (mode == "mixed")
        weights = upstream.call_args.args[4]
        assert weights["qkv_scale_alg"] == (0 if int8 else 1)
        assert weights["qkv_projection_weight"] is layer.in_proj_qkvgfab.weight
        assert weights["qkv_projection_weight_scale"] is layer.in_proj_qkvgfab.weight_scale
        assert weights["decay_projection_a_weight"] is layer.fused_bfg_proj.f_a_weight
        assert weights["decay_projection_b_weight"] is layer.fused_bfg_proj.f_b_weight
        torch.testing.assert_close(weights["beta_projection_weight"], layer.fused_bfg_proj.weight[:1])
        torch.testing.assert_close(weights["output_gate_projection_weight"], layer.fused_bfg_proj.weight[5:])
    for hook in (
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
    ):
        assert getattr(kimi_kda, hook).call_count == int(mode != "dummy")


@pytest.mark.parametrize(
    "tp,reduce,rs,offset", [(1, True, False, 0), (8, False, False, 0), (8, True, False, 1), (8, False, True, 2)]
)
def test_mega_output_collective(monkeypatch, tp, reduce, rs, offset):
    monkeypatch.setattr(kimi_kda, "tensor_model_parallel_all_reduce", lambda x: x + 1)
    monkeypatch.setattr(kimi_kda, "sp_reduce_scatter", lambda x: x + 2)
    layer = SimpleNamespace(
        _mega_kda_op=object(),
        _forward_upstream_mega_kda=lambda x, out: out.copy_(x),
        o_proj=SimpleNamespace(
            custom_op=MagicMock(spec=kimi_kda.KimiOProjMMReduceScatterOp) if rs else None,
            tp_size=tp,
            reduce_results=reduce,
        ),
    )
    x = torch.arange(8).view(2, 4)
    result = kimi_kda.AscendKimiK3DeltaAttention._forward_projections(layer, x, None)
    torch.testing.assert_close(result, x + offset)


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("grouped", [False, True])
def test_mega_native_prefill_qkv_keeps_token_axis(tokens, grouped):
    expected = torch.arange(tokens * 12).reshape(tokens, 12)
    projected = expected.unsqueeze(0) if grouped else expected
    qkv = SimpleNamespace(_mega_kda_mx_qkv=True, quant_method=MagicMock())
    qkv.quant_method.apply.return_value = projected
    layer = SimpleNamespace(in_proj_qkvgfab=qkv)
    result = kimi_kda.AscendKimiK3DeltaAttention._matmul_fused_qkv(layer, object())
    torch.testing.assert_close(result, expected)
    assert result.data_ptr() == projected.data_ptr()
