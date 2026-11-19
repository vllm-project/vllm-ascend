# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""BF16 reload storage/lifecycle tests; CPU casts do not validate NPU graphs."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.model_loader.weight_reload import refill_expert_list


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    from vllm.model_executor.model_loader.reload import direct

    from vllm_ascend import ascend_config
    from vllm_ascend.model_loader.weight_reload import install_direct_reload_patch

    # Restore process-wide state even if setup or the test raises. Subsequent
    # upstream and layerwise tests must see the provider/config they started with.
    monkeypatch.setattr(direct, "_checkpoint_views", direct._checkpoint_views)
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", ascend_config._ASCEND_CONFIG)
    monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", ascend_config._INIT_VLLM_CONFIG)
    install_direct_reload_patch()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
            }
        )
    )
    vc = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="bfloat16", enforce_eager=True),
        additional_config={"weight_nz_mode": 0},
    )
    init_ascend_config(vc)
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda value, *a, **kw: value)
    monkeypatch.setattr(torch.npu, "empty_cache", lambda: None)

    with set_current_vllm_config(vc), torch.no_grad():
        yield vc


def register(layer, name, data):
    param = ModelWeightParameter(
        data=data, input_dim=max(0, data.ndim - 1), output_dim=0, weight_loader=default_weight_loader
    )
    layer.register_parameter(name, param)


def _build_bf16_layer(layout, version, vc, monkeypatch, device=None):
    device = torch.device("cpu") if device is None else device
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.up_proj"
    layer.activation = "silu"
    if layout.startswith("dense"):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        layer.quant_method = AscendUnquantizedLinearMethod()
        values = {"weight": torch.empty(32, 64, dtype=torch.bfloat16)}
    else:
        import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as upstream

        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

        monkeypatch.setattr(upstream, "select_unquantized_moe_backend", lambda **kw: (None, None))
        scheme = AscendUnquantizedFusedMoEMethod()
        scheme.dynamic_eplb = layout == "moe-list"
        get_ascend_config().enable_fused_mc2 = int(layout in ("moe-nz", "moe-list"))
        layer.quant_method = scheme
        values = {
            "w13_weight": torch.empty(2, 128, 64, dtype=torch.bfloat16),
            "w2_weight": torch.empty(2, 64, 64, dtype=torch.bfloat16),
        }
    for name, tensor in list(values.items()):
        values[name] = (torch.arange(tensor.numel()).reshape(tensor.shape) % 5 + version).to(
            device=device, dtype=tensor.dtype
        )
        parent, _, leaf = name.rpartition(".")
        owner = layer.get_submodule(parent) if parent else layer
        register(owner, leaf, values[name].clone())
    record_metadata_for_reloading(layer)
    process_weights_after_loading(layer, vc.model_config, device)
    return layer, values


def _runtime(layer):
    tensors = {name: value for name, value in layer.named_parameters() if not getattr(value, "retired", False)}
    tensors.update({name: value for name, value in layer.named_buffers() if value is not None})
    for name in ("w13_weight_list", "w2_weight_list"):
        tensors.update({f"{name}.{index}": value for index, value in enumerate(getattr(layer, name, ()))})
    return tensors


@pytest.mark.parametrize("layout", ["dense-nd", "moe-nd", "moe-nz", "moe-list"])
@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_bf16_reload_matches_cold_start(config, monkeypatch, layout, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, *args, **kwargs: tensor.clone())
    layer, _ = _build_bf16_layer(layout, 1, config, monkeypatch)
    if layout == "moe-list":
        assert all(hasattr(layer, name) == (mode == "direct") for name in ("w13_weight", "w2_weight"))
    parameters = dict(layer.named_parameters())
    held = {name: tensor.detach() for name, tensor in _runtime(layer).items()}
    for version in (1, 3, 1):
        fresh, values = _build_bf16_layer(layout, version, config, monkeypatch)
        start_reload(layer)
        for name, value in values.items():
            parameter = layer.get_parameter(name)
            parameter.weight_loader(parameter, value)
        finish_reload(layer, config.model_config, torch.device("cpu"))
        actual, expected = _runtime(layer), _runtime(fresh)
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            assert torch.equal(tensor, expected[name]), name
            if mode == "direct":
                assert tensor.data_ptr() == held[name].data_ptr(), name
                assert torch.equal(held[name], expected[name]), name
        if mode == "direct":
            assert all(layer.get_parameter(name) is parameter for name, parameter in parameters.items())


def test_expert_layout_change_rejected_before_copy(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    layer = torch.nn.Module()
    layer.experts = [torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(2, 2, dtype=torch.bfloat16)]
    before = [tensor.clone() for tensor in layer.experts]
    with pytest.raises(ValueError, match="expert runtime layout changed"):
        refill_expert_list(layer, "experts", [torch.zeros_like(before[0]), torch.zeros(3, 2, dtype=torch.bfloat16)])
    assert all(torch.equal(actual, expected) for actual, expected in zip(layer.experts, before))


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_runtime_tensor_layout_change_keeps_captured_data(monkeypatch, change):
    from vllm_ascend.model_loader.weight_reload import refill_runtime_tensor

    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    owner = SimpleNamespace()
    original = torch.ones(2, 4, dtype=torch.bfloat16)
    assert refill_runtime_tensor(owner, "projection", original) is original
    incompatible = torch.zeros(4, 2, dtype=torch.bfloat16) if change == "shape" else torch.zeros(2, 4)
    with pytest.raises(ValueError, match="runtime layout changed"):
        refill_runtime_tensor(owner, "projection", incompatible)
    assert torch.equal(original, torch.ones_like(original))
