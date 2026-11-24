# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.ops import kimi_kda
from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8


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


@pytest.mark.parametrize("mode", ["dummy", "decode", "first_token", "long_prefill", "mixed"])
def test_mega_kda_model_dispatch(monkeypatch, mode):
    x = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    indices = torch.tensor([0, 2, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=int(mode in ("first_token", "long_prefill", "mixed")),
        num_prefill_tokens={"first_token": 1, "long_prefill": 5, "mixed": 3}.get(mode, 0),
        num_spec_decodes=int(mode == "mixed"),
        num_spec_decode_tokens=2 if mode == "mixed" else 0,
        num_actual_tokens=5,
        non_spec_token_indx=indices if mode == "mixed" else None,
        spec_token_indx=torch.tensor([1, 4], dtype=torch.int32) if mode == "mixed" else None,
        spec_sequence_masks=object() if mode == "mixed" else None,
    )
    context = SimpleNamespace(attn_metadata=None if mode == "dummy" else {"layer": metadata})
    monkeypatch.setattr(kimi_kda, "get_forward_context", lambda: context)
    for hook in (
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
    ):
        monkeypatch.setattr(kimi_kda, hook, MagicMock())
    upstream = MagicMock(return_value=x + 1)
    monkeypatch.setattr(kimi_kda, "run_upstream_mega_kda", upstream)
    proj = MagicMock(side_effect=lambda hidden: (hidden, None, None, None))

    def native(hidden, g1, g2, beta, out, *, native_prefill_metadata):
        assert native_prefill_metadata.num_spec_decodes == 0
        assert native_prefill_metadata.num_actual_tokens == hidden.size(0)
        assert native_prefill_metadata.spec_sequence_masks is None
        out.copy_((hidden + 2).reshape_as(out))

    layer = SimpleNamespace(
        prefix="layer",
        _mega_kda_op=object(),
        _mega_kda_scale_alg=1,
        _fused_bfg_output_sizes=(1, 4, 4),
        _mega_output_norm_weight=MagicMock(),
        in_proj_qkvgfab=SimpleNamespace(_mxfp8_transformed=True, weight=MagicMock(), weight_scale=MagicMock()),
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
    torch.testing.assert_close(result, expected)
    assert proj.call_count == int(mode in ("long_prefill", "mixed"))
    assert upstream.call_count == int(mode in ("decode", "first_token", "mixed"))
    if upstream.called:
        assert upstream.call_args.kwargs["only_spec"] == (mode == "mixed")
        weights = upstream.call_args.args[4]
        assert weights["qkv_scale_alg"] == 1
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
        _mega_kda_mix=True,
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
