# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense loader metadata and real layerwise reload with a mocked MX kernel."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from tests.e2e.common.rlhf.weight_transfer.case import QWEN35_ONLINE_MX_CASES
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.online_mx import (
    AscendOnlineMXLinearMethod,
    AscendOnlineMXMoEMethod,
    _quantize,
    _validate_dense,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("case", QWEN35_ONLINE_MX_CASES, ids=lambda case: case.id)
def test_compressed_tensors_config_selects_online_mx(case):
    config = AscendCompressedTensorsConfig.from_config(case.hf_overrides["quantization_config"])
    args = config.target_scheme_map["Linear"]
    expected = "W8A8_MXFP8" if args["weights"].num_bits == 8 else "W4A8_MXFP"
    assert config._detect_quant_type(args["weights"], args["input_activations"], config.quant_format) == expected
    scheme = SimpleNamespace(quant_type=QuantType.W8A8MXFP if expected == "W8A8_MXFP8" else QuantType.W4A8MXFP)
    with (
        patch.object(config, "_get_linear_scheme", return_value=scheme),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config", return_value=SimpleNamespace(online_quantize_weights=True)
        ),
    ):
        method = config.get_quant_method(MagicMock(spec=LinearBase), "model.layers.3.self_attn.q_proj")
    assert isinstance(method, AscendOnlineMXLinearMethod)
    assert method.quant_method is scheme


def test_moe_dense_shapes_and_receiver_generated_scales():
    method = AscendOnlineMXMoEMethod.__new__(AscendOnlineMXMoEMethod)
    method.quant_method = SimpleNamespace(
        use_expert_weight_list=False,
        quant_type=QuantType.W8A8MXFP,
        process_weights_after_loading=MagicMock(),
    )
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 64, 64, torch.bfloat16)
    assert layer.w13_weight.shape == (2, 128, 64)
    assert layer.w2_weight.shape == (2, 64, 64)
    assert not hasattr(layer, "w13_weight_scale")
    assert layer.w13_weight.dtype == torch.bfloat16

    def kernel(weight, *, dst_type):
        return weight.to(dst_type), torch.zeros(weight.shape[0], 2, dtype=torch.uint8)

    with patch("vllm_ascend.quantization.online_mx.torch_npu.npu_dynamic_mx_quant", side_effect=kernel):
        method.process_weights_after_loading(layer)
    assert layer.w13_weight_scale.shape == (2, 128, 2)
    assert layer.w2_weight_scale.shape == (2, 64, 2)
    method.quant_method.process_weights_after_loading.assert_called_once_with(layer)
    method.quant_method.use_expert_weight_list = True
    with pytest.raises(ValueError, match="EPLB"):
        method.create_weights(torch.nn.Module(), 2, 64, 64, torch.bfloat16)


@pytest.mark.parametrize("shape,dtype", [((4, 32), torch.bfloat16), ((4, 64), torch.float16)])
def test_invalid_dense_payload(shape, dtype):
    with pytest.raises(ValueError):
        _validate_dense(shape, dtype)


@pytest.mark.parametrize("quant_type,packed", [(QuantType.W8A8MXFP, False), (QuantType.W4A8MXFP, True)])
def test_receiver_generates_weights_and_scales(quant_type, packed):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(4, 64, dtype=torch.bfloat16), requires_grad=False)
    calls = []
    scheme = SimpleNamespace(quant_type=quant_type, process_weights_after_loading=lambda layer: calls.append(layer))
    weight = torch.zeros(4, 32 if packed else 64, dtype=torch.uint8 if packed else torch.float8_e4m3fn)
    scale = torch.full((4, 2), 127, dtype=torch.uint8)
    with patch("vllm_ascend.quantization.online_mx.torch_npu.npu_dynamic_mx_quant", return_value=(weight, scale)) as op:
        _quantize(layer, ("weight",), scheme)
    assert op.call_args.kwargs["dst_type"] == (torch_npu.float4_e2m1fn_x2 if packed else torch.float8_e4m3fn)
    assert layer.weight.dtype == weight.dtype
    assert layer.weight.shape == weight.shape
    assert torch.equal(layer.weight_scale, scale)
    assert calls == [layer]


def test_layerwise_reload_restores_bf16_and_updates_captured_scale_storage():
    layer = torch.nn.Module()
    scheme = SimpleNamespace(quant_type=QuantType.W8A8MXFP)

    def native_process(layer):
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

    scheme.process_weights_after_loading = native_process
    method = AscendOnlineMXLinearMethod(scheme)
    layer.quant_method = method
    method.create_weights(layer, 64, [4], 64, 4, torch.bfloat16, weight_loader=default_weight_loader)
    record_metadata_for_reloading(layer)

    def kernel(weight, *, dst_type):
        assert weight.dtype == torch.bfloat16
        code = int(weight[0, 0])
        return weight.to(dst_type), torch.full((4, 2), code, dtype=torch.uint8)

    with patch("vllm_ascend.quantization.online_mx.torch_npu.npu_dynamic_mx_quant", side_effect=kernel):
        layer.weight.data.fill_(1)
        method.process_weights_after_loading(layer)
        weight_ptr, scale_ptr = layer.weight.data_ptr(), layer.weight_scale.data_ptr()
        for value in (2, 3, 2):
            initialize_layerwise_reload(layer)
            assert layer.weight.dtype == torch.bfloat16
            assert not hasattr(layer, "weight_scale")
            layer.weight.weight_loader(layer.weight, torch.full((4, 64), value, dtype=torch.bfloat16))
            assert layer.weight.data_ptr() == weight_ptr
            assert layer.weight_scale.data_ptr() == scale_ptr
            assert torch.equal(layer.weight.float(), torch.full((64, 4), float(value)))
            assert torch.all(layer.weight_scale == value)
