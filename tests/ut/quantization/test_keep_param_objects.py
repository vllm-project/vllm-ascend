# SPDX-License-Identifier: Apache-2.0
"""Ascend counterpart of upstream #62407's checkpoint-parameter contract.

These CPU tests execute real PWAL and parameter helpers. Device format casts
and communication discovery are substituted, so they do not validate NZ or
MX kernels. Real-NPU checks belong in the accompanying validation report.
"""

import importlib
import json

import pytest
import torch
import torch_npu
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.reload.direct import begin_direct_reload, record_checkpoint_layout
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import get_live_param, kernel_parameters

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

BASE = "vllm_ascend.quantization.methods"


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    from vllm.model_executor.model_loader.reload import direct

    from vllm_ascend import ascend_config
    from vllm_ascend.quantization.reload import install_direct_reload_patch

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
        quant_config=AscendModelSlimConfig(),
        additional_config={"weight_nz_mode": 0},
    )
    init_ascend_config(vc)
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", property(lambda _: False))
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda value, *a, **kw: value)
    monkeypatch.setattr(torch.npu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.Tensor, "npu", lambda value, *a, **kw: value)

    def pack_cpu(value, **kwargs):
        groups = value.to(torch.int64).reshape(*value.shape[:-1], -1, 8) & 15
        return (groups << (torch.arange(8) * 4)).sum(-1).to(torch.int32)

    monkeypatch.setattr(torch_npu, "npu_convert_weight_to_int4pack", pack_cpu)
    with set_current_vllm_config(vc), torch.no_grad():
        yield vc


def module(name, monkeypatch, mega=False):
    mod = importlib.import_module(f"{BASE}.{name}")
    if hasattr(mod, "use_cann_megamoe"):
        monkeypatch.setattr(mod, "use_cann_megamoe", lambda _: mega)
    if hasattr(mod, "get_tensor_model_parallel_world_size"):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    if hasattr(mod, "get_mc2_group"):
        # Exercise the production constructor's unavailable-communicator path.
        monkeypatch.setattr(mod, "get_mc2_group", lambda: None)
    return mod


def register(layer, name, data):
    param = ModelWeightParameter(
        data=data, input_dim=max(0, data.ndim - 1), output_dim=0, weight_loader=default_weight_loader
    )
    param.audit_marker = object()
    layer.register_parameter(name, param)


def snapshot(layer):
    return {
        n: (
            p,
            set(vars(p)),
            p.input_dim,
            p.output_dim,
            p.numel() * p.element_size(),
            p.weight_loader,
            getattr(p, "packed_dim", None),
        )
        for n, p in layer.named_parameters(remove_duplicate=False)
    }


def assert_kept(layer, before):
    after = dict(layer.named_parameters(remove_duplicate=False))
    for name, (param, attrs, input_dim, output_dim, nbytes, loader, packed_dim) in before.items():
        assert after.get(name) is param, f"checkpoint object lost: {name}"
        assert attrs <= vars(param).keys(), name
        assert (param.input_dim, param.output_dim) == (input_dim, output_dim), name
        assert getattr(param, "packed_dim", None) == packed_dim, name
        assert param.weight_loader is loader and callable(loader), name
        available = param.untyped_storage().nbytes() - param.storage_offset() * param.element_size()
        staging = getattr(param, "_direct_reload_staging", None)
        if staging is not None:
            shape, stride, dtype, offset = param._ckpt[1]
            assert (tuple(staging.shape), staging.stride(), staging.dtype, staging.storage_offset()) == (
                shape,
                stride,
                dtype,
                offset,
            )
            assert staging.data_ptr() != param.data_ptr(), name
            available = max(available, staging.untyped_storage().nbytes() - offset * staging.element_size())
        assert available >= nbytes, f"checkpoint storage released: {name}"
    unique = {id(p): p for p in after.values() if p.numel()}
    pointers = [p.untyped_storage().data_ptr() for p in unique.values()]
    assert len(pointers) == len(set(pointers)), "different Parameter objects alias one storage"


@pytest.mark.parametrize("name", ["weight", "w13_weight", "w2_weight"])
def test_packed_checkpoint_name_remains_an_alias(config, monkeypatch, name):
    mod = module("w4a4.w4a4_mxfp4", monkeypatch)
    layer = torch.nn.Module()
    register(layer, name + "_packed", torch.arange(16, dtype=torch.uint8).reshape(4, 4))
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    mod._rename_packed_weight_parameter(layer, name)
    assert_kept(layer, before)
    assert getattr(layer, name) is getattr(layer, name + "_packed")
    assert set(kernel_parameters(layer)) == {name}
    # A second PWAL must accept the alias it created on the first invocation.
    mod._rename_packed_weight_parameter(layer, name)
    begin_direct_reload(layer)
    getattr(layer, name + "_packed").weight_loader(
        getattr(layer, name + "_packed"), torch.full((4, 4), 7, dtype=torch.uint8)
    )
    assert torch.equal(getattr(layer, name), torch.full((4, 4), 7, dtype=torch.uint8))


@pytest.mark.parametrize("mx", [False, True], ids=["int4-flatquant", "mxfp4-flatquant"])
def test_flatquant_keeps_checkpoint_parameters(config, monkeypatch, mx):
    mod = module("w4a4.w4a4_mxfp4_flatquant" if mx else "w4a4.w4a4_flatquant", monkeypatch)
    cls = mod.AscendW4A4MXFP4FlatQuantDynamicLinearMethod if mx else mod.AscendW4A4FlatQuantDynamicLinearMethod
    scheme = cls()
    layer = torch.nn.Module()
    data = {
        "weight": torch.ones(16, 32 if mx else 64, dtype=torch.uint8 if mx else torch.int8),
        "weight_scale": torch.ones((16, 2) if mx else (16, 1), dtype=torch.uint8 if mx else torch.bfloat16),
        "weight_offset": torch.zeros(16, 1, dtype=torch.bfloat16),
        "left_trans": torch.eye(8, dtype=torch.bfloat16),
        "right_trans": torch.eye(8, dtype=torch.bfloat16),
        "clip_ratio": torch.ones(1, dtype=torch.bfloat16),
    }
    for name, value in data.items():
        register(layer, name, value.clone())
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    scheme.process_weights_after_loading(layer)
    assert_kept(layer, before)
    if not mx:
        assert get_live_param(layer, "weight") is None
        assert "weight_packed" in kernel_parameters(layer)
    for factor in [2, 1]:
        begin_direct_reload(layer)
        for name, value in data.items():
            param = layer.get_parameter(name)
            param.weight_loader(param, value if name == "clip_ratio" else value * factor)
        scheme.process_weights_after_loading(layer)
        assert_kept(layer, before)
        assert torch.equal(layer.left_trans, (data["left_trans"] * factor).t())
        if not mx:
            assert torch.equal(layer.weight_packed, mod.pack_int4_weights(data["weight"] * factor))


MOE_CASES = [
    ("w8a8.w8a8_dynamic", "AscendW8A8DynamicFusedMoEMethod", False),
    ("w4a8.w4a8", "AscendW4A8DynamicFusedMoEMethod", False),
    ("w4a8.w4a8", "AscendW4A8DynamicFusedMoEMethod", True),
    ("w4a8.w4a8_mxfp4", "AscendW4A8MXFPDynamicFusedMoEMethod", False),
    ("w4a4.w4a4_mxfp4", "AscendW4A4MXFP4DynamicFusedMoEMethod", True),
    ("w8a8.w8a8_mxfp8", "AscendW8A8MXFP8DynamicFusedMoEMethod", True),
]


@pytest.mark.parametrize("prefix", ["probe", "model.kv_b_proj"])
def test_fp8_checkpoint_survives_dequantization(config, monkeypatch, prefix):
    mod = module("w8a8.fp8_block", monkeypatch)
    monkeypatch.setattr(mod, "_require_fp8_weight_matmul", lambda: None)
    scheme = mod.AscendFp8BlockLinearMethod((4, 4))
    layer = torch.nn.Module()
    layer.prefix = prefix
    register(layer, "weight", torch.ones(16, 24).to(torch.float8_e4m3fn))
    register(layer, "weight_scale_inv", torch.full((4, 6), 0.25))
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    for value in [1, 2, 1]:
        begin_direct_reload(layer)
        layer.weight.weight_loader(layer.weight, torch.full((16, 24), value).to(torch.float8_e4m3fn))
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch.full((4, 6), 0.25))
        scheme.process_weights_after_loading(layer)
        assert_kept(layer, before)
        assert get_live_param(layer, "weight_scale_inv") is None
        assert torch.equal(layer.weight, torch.full((16, 24), value * 0.25, dtype=torch.bfloat16))


@pytest.mark.parametrize("path,classname,mega", MOE_CASES)
def test_moe_list_conversion_keeps_checkpoint_storage(config, monkeypatch, path, classname, mega):
    mod = module(path, monkeypatch, mega)
    ascend = get_ascend_config()
    monkeypatch.setattr(ascend.eplb_config, "dynamic_eplb", not mega)
    scheme = getattr(mod, classname)()
    layer = torch.nn.Module()
    layer.activation = "silu"
    tensors = scheme.get_weight(2, 64, 64, torch.bfloat16)
    tensors.update(scheme.get_dynamic_quant_param(2, 64, 64, torch.bfloat16))
    for name, tensor in tensors.items():
        register(layer, name, torch.ones(tensor.shape, dtype=tensor.dtype))
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    scheme.process_weights_after_loading(layer)
    assert_kept(layer, before)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        assert get_live_param(layer, name) is None
    # Retained sources must still be writable through their checkpoint loaders.
    begin_direct_reload(layer)
    for name, tensor in tensors.items():
        param = layer.get_parameter(name)
        expected = torch.full(tensor.shape, 2, dtype=tensor.dtype)
        param.weight_loader(param, expected)
        assert torch.equal(param.float(), expected.float())


def test_w4a8_linear_reduction_keeps_checkpoint_capacity(config, monkeypatch):
    mod = module("w4a8.w4a8", monkeypatch)
    scheme = mod.AscendW4A8DynamicLinearMethod()
    layer = torch.nn.Module()
    values = {
        "weight": torch.ones(32, 64, dtype=torch.int8),
        "weight_scale": torch.ones(64, 1),
        "weight_offset": torch.zeros(64, 1),
        "scale_bias": torch.ones(64, 16),
    }
    for name, value in values.items():
        register(layer, name, value.clone())
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    for factor in [1, 2, 1]:
        begin_direct_reload(layer)
        for name, value in values.items():
            param = layer.get_parameter(name)
            param.weight_loader(param, value * factor)
        scheme.process_weights_after_loading(layer)
        assert_kept(layer, before)
        assert torch.equal(layer.scale_bias, torch.full((64,), 16.0 * factor))


def test_unquantized_moe_keeps_checkpoint_parameters(config, monkeypatch):
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as upstream

    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    # Backend selection is irrelevant to this CPU PWAL contract; PWAL itself
    # and the parameter helpers are real, including the transpose and copy.
    monkeypatch.setattr(upstream, "select_unquantized_moe_backend", lambda **kw: (None, None))
    scheme = AscendUnquantizedFusedMoEMethod()
    layer = torch.nn.Module()
    values = {"w13_weight": torch.ones(2, 128, 64), "w2_weight": torch.ones(2, 64, 64)}
    for name, value in values.items():
        register(layer, name, value.clone())
    before = snapshot(layer)
    record_checkpoint_layout(layer)
    for factor in [1, 2, 1]:
        begin_direct_reload(layer)
        for name, value in values.items():
            param = layer.get_parameter(name)
            param.weight_loader(param, value * factor)
        scheme.process_weights_after_loading(layer)
        assert_kept(layer, before)
        for name, value in values.items():
            assert torch.equal(layer.get_parameter(name), value.transpose(1, 2) * factor)
