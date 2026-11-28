# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.methods.w4a4 import w4a4_int_moe
from vllm_ascend.quantization.quant_type import QuantType

Method = w4a4_int_moe.AscendW4A4DynamicFusedMoEMethod


def test_registration_and_unquantized_dispatch():
    assert get_scheme_class("W4A4_DYNAMIC", "moe") is Method
    quant = MoEQuantParams(quant_type=QuantType.W4A4)
    assert quant.is_quant
    assert not quant.dispatch_with_quant
    assert quant.get_dst_type is None
    assert not Method.supports_eplb


@pytest.mark.parametrize("eplb,fused_mc2,message", [(True, False, "EPLB"), (False, True, "enable_fused_mc2")])
def test_unsupported_configuration(monkeypatch, eplb, fused_mc2, message):
    monkeypatch.setattr(
        w4a4_int_moe,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=eplb)),
    )
    monkeypatch.setattr(w4a4_int_moe, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=fused_mc2))
    with pytest.raises(ValueError, match=message):
        Method()


def test_scales_preserve_fp32_checkpoint_precision():
    params = Method.__new__(Method).get_dynamic_quant_param(2, 8, 16, torch.bfloat16)
    assert all(value.dtype == torch.float32 for value in params.values())
    assert params["w13_weight_scale"].shape == (2, 16, 1)
    assert params["w2_weight_scale"].shape == (2, 16, 1)


@pytest.mark.parametrize("tokens", [0, 3])
def test_floating_payload_is_quantized_despite_scale_metadata(monkeypatch, tokens):
    method = Method.__new__(Method)
    x = torch.zeros(tokens, 16, dtype=torch.bfloat16)
    packed = torch.zeros(tokens, 2, dtype=torch.int32)
    scale = torch.ones(tokens)
    quantize = Mock(return_value=(packed, scale))
    monkeypatch.setattr(w4a4_int_moe.torch_npu, "npu_dynamic_quant", quantize)
    method._gmm = Mock(return_value=torch.zeros(tokens, 8))
    context = SimpleNamespace(hidden_states=x, dynamic_scale=torch.full((tokens,), 7.0))
    method.apply_gmm1(context)
    quantize.assert_called_once_with(x, dst_type=torch.quint4x2)
    method._gmm.assert_called_once_with(context, packed, scale, "w13")


@pytest.mark.parametrize("dtype,scale", [(torch.int8, torch.ones(1)), (torch.int32, None)])
def test_rejects_invalid_quantized_payload(dtype, scale):
    method = Method.__new__(Method)
    with pytest.raises(ValueError, match="packed INT4 input"):
        method.apply_gmm1(SimpleNamespace(hidden_states=torch.zeros(1, 8, dtype=dtype), dynamic_scale=scale))


def test_packed_input_reuses_its_token_scales():
    method = Method.__new__(Method)
    method._gmm = Mock()
    context = SimpleNamespace(hidden_states=torch.zeros(2, 2, dtype=torch.int32), dynamic_scale=torch.ones(2))
    method.apply_gmm1(context)
    method._gmm.assert_called_once_with(context, context.hidden_states, context.dynamic_scale, "w13")


@pytest.mark.parametrize("offset,outputs,message", [(1.0, 8, "symmetric weight offsets"), (0.0, 7, "divisible by 8")])
def test_invalid_checkpoint_metadata(offset, outputs, message):
    layer = SimpleNamespace(
        w13_weight=torch.zeros(1, outputs, 8, dtype=torch.int8),
        w13_weight_scale=torch.ones(1, outputs, 1),
        w13_weight_offset=torch.full((1, outputs, 1), offset),
    )
    with pytest.raises(ValueError, match=message):
        Method.__new__(Method).process_weights_after_loading(layer)
