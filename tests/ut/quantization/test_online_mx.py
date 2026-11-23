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
from tests.e2e.common.rlhf.weight_transfer.online_mx import modelslim_description, modelslim_startup_directory
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.online_mx import (
    AscendOnlineMXLinearMethod,
    AscendOnlineMXMoEMethod,
    _quantize,
    _validate_dense,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("quant_type", ["W8A8_MXFP8", "W4A8_MXFP"])
def test_online_modelslim_validation(quant_type):
    config = AscendConfig(online_quantize_weights=True, sparse_kv_offload_config=SimpleNamespace(enabled=False))
    vc = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=False))
    quant = AscendModelSlimConfig({"model.layers.0.q_proj.weight": quant_type})
    config._validate_online_quantization(vc, quant)
    quant.quant_description["other.weight"] = "W8A8_DYNAMIC"
    with pytest.raises(ValueError, match="weight types"):
        config._validate_online_quantization(vc, quant)


@pytest.mark.parametrize("feature", ["static_eplb", "dynamic_eplb", "megamoe"])
def test_online_modelslim_rejects_expert_list_paths(feature):
    config = AscendConfig(online_quantize_weights=True, sparse_kv_offload_config=SimpleNamespace(enabled=False))
    vc = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=feature == "static_eplb"))
    config.eplb_config.dynamic_eplb = feature == "dynamic_eplb"
    # MegaMoE's user values 2/3 have already normalized to 1 at this point.
    config.enable_fused_mc2 = 1
    config._megamoe_supported = feature == "megamoe"
    with pytest.raises(ValueError, match="EPLB or MegaMoE"):
        config._validate_online_quantization(vc, AscendModelSlimConfig({"layer.weight": "W8A8_MXFP8"}))


@pytest.mark.parametrize("group_size", [16, 64, 128])
def test_online_modelslim_rejects_incompatible_scale_groups(group_size):
    config = AscendConfig(online_quantize_weights=True, sparse_kv_offload_config=SimpleNamespace(enabled=False))
    quant = AscendModelSlimConfig({"layer.weight": "W8A8_MXFP8", "group_size": group_size})
    with pytest.raises(ValueError, match="group_size=32"):
        config._validate_online_quantization(SimpleNamespace(), quant)


@pytest.mark.parametrize("description", [{}, {"layer.weight": "W8A8_DYNAMIC"}, {"model_quant_type": "W8A8_MXFP8"}])
def test_online_modelslim_requires_mx_weight_entries(description):
    with pytest.raises(ValueError, match="ModelSlim"):
        AscendConfig(
            online_quantize_weights=True, sparse_kv_offload_config=SimpleNamespace(enabled=False)
        )._validate_online_quantization(SimpleNamespace(), AscendModelSlimConfig(description))


def test_online_modelslim_is_opt_in_and_rejects_other_formats():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    config._validate_online_quantization(SimpleNamespace(), None)
    config.online_quantize_weights = True
    with pytest.raises(ValueError, match="ModelSlim"):
        config._validate_online_quantization(SimpleNamespace(), SimpleNamespace(get_name=lambda: "compressed-tensors"))


@pytest.mark.parametrize("case", QWEN35_ONLINE_MX_CASES, ids=lambda case: case.id)
def test_modelslim_description_file_roundtrip(case, tmp_path):
    params = [
        SimpleNamespace(name="model.layers.0.mlp.experts.0.gate_proj.weight", shape=(64, 64)),
        SimpleNamespace(name="model.layers.0.mlp.gate.weight", shape=(2, 64)),
        SimpleNamespace(name="model.embed_tokens.weight", shape=(64, 64)),
        SimpleNamespace(name="model.layers.0.input_layernorm.weight", shape=(64,)),
        SimpleNamespace(name="lm_head.weight", shape=(64, 64)),
    ]
    source = SimpleNamespace(metadata=params)
    original_config = tmp_path / "config.json"
    original_config.write_text("{}")
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(original_config)),
        modelslim_startup_directory(source, case) as directory,
    ):
        config = AscendModelSlimConfig.from_config(case.hf_overrides["quantization_config"])
        assert not config.quant_description
        config.maybe_update_config(directory)
    expected = modelslim_description(source, case)
    for param in params:
        assert config.quant_description[param.name] == expected[param.name]
    assert expected[params[0].name] in ("W8A8_MXFP8", "W4A8_MXFP")
    assert all(expected[param.name] == "FLOAT" for param in params[1:])


@pytest.mark.parametrize("case", QWEN35_ONLINE_MX_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_modelslim_config_selects_online_mx(case, online, moe):
    prefix = "model.layers.3.mlp.experts" if moe else "model.layers.3.self_attn.q_proj"
    quant_type = case.hf_overrides["quantization_config"]["model_quant_type"]
    config = AscendModelSlimConfig({prefix + ".weight": quant_type})
    scheme = SimpleNamespace(quant_type=QuantType.W8A8MXFP if quant_type == "W8A8_MXFP8" else QuantType.W4A8MXFP)
    with (
        patch("vllm_ascend.quantization.configs.modelslim_config.create_scheme_for_layer", return_value=scheme),
        patch("vllm_ascend.quantization.configs.modelslim_config.is_fused_moe_layer", return_value=moe),
        patch.object(config, "_update_packed_modules_mapping"),
        patch(
            "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config",
            return_value=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_moe"))
            ),
        ),
        patch(
            "vllm_ascend.quantization.configs.modelslim_config.get_ascend_config",
            return_value=SimpleNamespace(online_quantize_weights=online),
        ),
    ):
        method = config.get_quant_method(MagicMock() if moe else MagicMock(spec=LinearBase), prefix)
    expected_method = AscendOnlineMXMoEMethod if moe else AscendOnlineMXLinearMethod
    assert isinstance(method, expected_method) == online
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
