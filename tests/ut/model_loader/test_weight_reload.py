# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""BF16 reload storage/lifecycle tests; CPU casts do not validate NPU graphs."""

import json
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

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
from vllm_ascend.model_loader.weight_reload import ExpertModuleStaging, refill_expert_list, refill_runtime_parameter
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


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


def _build_fused_layer(layout, version, vc, monkeypatch):
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as upstream

    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.ops.fused_moe import routed_experts
    from vllm_ascend.utils import AscendDeviceType

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.up_proj"
    layer.activation = "silu"
    device_type = AscendDeviceType.A5 if layout == "moe-a5-fallback" else AscendDeviceType.A3
    monkeypatch.setattr(upstream, "select_unquantized_moe_backend", lambda **kw: (None, None))
    monkeypatch.setattr(routed_experts, "use_cann_megamoe", lambda _config: layout != "moe-nz")
    monkeypatch.setattr(routed_experts, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    get_ascend_config().enable_fused_mc2 = 1
    layer.quant_method = routed_experts.AscendUnquantizedFusedMoEMethod()
    values = {}
    for name, shape in (("w13_weight", (2, 128, 64)), ("w2_weight", (2, 64, 64))):
        tensor = (torch.arange(torch.Size(shape).numel()).reshape(shape) % 5 + version).to(torch.bfloat16)
        values[name] = tensor
        register(layer, name, tensor.clone())
    record_metadata_for_reloading(layer)
    process_weights_after_loading(layer, vc.model_config, torch.device("cpu"))
    return layer, values


def _runtime(layer):
    tensors = {name: value for name, value in layer.named_parameters() if not getattr(value, "retired", False)}
    tensors.update({name: value for name, value in layer.named_buffers() if value is not None})
    for name in ("w13_weight_list", "w2_weight_list"):
        tensors.update({f"{name}.{index}": value for index, value in enumerate(getattr(layer, name, ()))})
    return tensors


@pytest.mark.parametrize("layout", ["moe-nz", "moe-list", "moe-a5-fallback"])
@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_bf16_reload_matches_cold_start(config, monkeypatch, layout, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, *args, **kwargs: tensor.clone())
    layer, _ = _build_fused_layer(layout, 1, config, monkeypatch)
    if layout == "moe-list":
        assert all(hasattr(layer, name) == (mode == "direct") for name in ("w13_weight", "w2_weight"))
    parameters = dict(layer.named_parameters())
    held = {name: tensor.detach() for name, tensor in _runtime(layer).items()}
    for version in (3, 1):
        fresh, values = _build_fused_layer(layout, version, config, monkeypatch)
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


def test_fused_moe_transpose_temporaries_released_before_nz_cast(config, monkeypatch):
    from vllm_ascend.model_loader import weight_reload
    from vllm_ascend.ops.fused_moe import routed_experts

    layer, values = _build_fused_layer("moe-nz", 1, config, monkeypatch)
    temporaries = {}
    replace = routed_experts.replace_parameter
    refill = weight_reload.refill_nz_weight

    def track_transpose(layer, name, tensor):
        temporaries[name] = weakref.ref(tensor)
        replace(layer, name, tensor)

    def check_refill(layer, name, **kwargs):
        # Square w2 is copied into existing storage; rectangular w13 may be rebound.
        assert "w2_weight" in temporaries
        assert all(ref() is None for ref in temporaries.values())
        refill(layer, name, **kwargs)

    monkeypatch.setattr(routed_experts, "replace_parameter", track_transpose)
    monkeypatch.setattr(weight_reload, "refill_nz_weight", check_refill)
    start_reload(layer)
    for name, value in values.items():
        parameter = layer.get_parameter(name)
        parameter.weight_loader(parameter, value)
    finish_reload(layer, config.model_config, torch.device("cpu"))
    assert "w2_weight" in temporaries


def test_full_update_repairs_partial_moe_postprocessing(config, monkeypatch):
    from vllm.config import WeightTransferConfig

    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine
    from vllm_ascend.model_loader import weight_reload
    from vllm_ascend.worker.worker import NPUWorker

    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, *args: tensor.clone())
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    layer, _ = _build_fused_layer("moe-nz", 1, config, monkeypatch)
    cold_b, checkpoint_b = _build_fused_layer("moe-nz", 3, config, monkeypatch)
    cold_c, checkpoint_c = _build_fused_layer("moe-nz", 5, config, monkeypatch)
    held = {name: tensor.detach() for name, tensor in _runtime(layer).items()}
    original_w2 = held["w2_weight"].clone()

    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.model_runner = Mock()
    worker.model_runner.extensible_kv_cache = None
    worker.model_runner.get_model.return_value = layer
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker._incomplete_weight_updates = set()
    worker._kv_cache_reclaimable = False
    worker._weight_update_kv_cache_loan = ExitStack()
    engine = HCCLWeightTransferEngine(WeightTransferConfig(backend="hccl"), config, torch.device("cpu"), layer)
    worker.weight_transfer_engine = engine

    # Deliver the checkpoint locally. Keep the real worker, engine begin/finish,
    # parameter loaders and fused-MoE post-processing on the path under test.
    def load_checkpoint(values):
        for name, value in values.items():
            parameter = layer.get_parameter(name)
            parameter.weight_loader(parameter, value)

    monkeypatch.setattr(engine, "update_weights", load_checkpoint)
    refill = weight_reload.refill_nz_weight

    def fail_on_w2(layer, name, **kwargs):
        if name == "w2_weight":
            raise RuntimeError("injected w2 refill failure")
        refill(layer, name, **kwargs)

    worker.start_weight_update()
    worker.update_weights(checkpoint_b)
    with monkeypatch.context() as fault:
        fault.setattr(weight_reload, "refill_nz_weight", fail_on_w2)
        with pytest.raises(RuntimeError, match="injected w2 refill failure"):
            worker.finish_weight_update()
    assert torch.equal(held["w13_weight"], cold_b.w13_weight)
    assert torch.equal(held["w2_weight"], original_w2)
    assert not torch.equal(held["w2_weight"], cold_b.w2_weight)

    # C differs from both A and B, so skipping either repair write is observable.
    worker.start_weight_update()
    worker.update_weights(checkpoint_c)
    worker.finish_weight_update()
    actual, expected = _runtime(layer), _runtime(cold_c)
    assert actual.keys() == expected.keys() == held.keys()
    for name, tensor in actual.items():
        assert tensor.data_ptr() == held[name].data_ptr(), name
        assert torch.equal(held[name], expected[name]), name


def test_refill_experts_keeps_owned_storage_and_handles_aliases(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    layer = torch.nn.Module()
    weights = torch.tensor([[1, 2], [3, 4]])
    refill_expert_list(layer, "experts", weights.unbind())
    held = layer.experts
    pointers = [t.data_ptr() for t in held]
    weights.zero_()
    refill_expert_list(layer, "experts", held[::-1])
    assert torch.stack(held).tolist() == [[3, 4], [1, 2]]
    assert layer.experts is held and [t.data_ptr() for t in held] == pointers


def test_expert_layout_change_rejected_before_copy(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    layer = torch.nn.Module()
    layer.experts = [torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(2, 2, dtype=torch.bfloat16)]
    before = [tensor.clone() for tensor in layer.experts]
    with pytest.raises(ValueError, match=r"experts\[1\]: runtime layout changed"):
        refill_expert_list(layer, "experts", [torch.zeros_like(before[0]), torch.zeros(3, 2, dtype=torch.bfloat16)])
    assert all(torch.equal(actual, expected) for actual, expected in zip(layer.experts, before))


def test_expert_npu_format_change_rejected_before_copy(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setattr(torch_npu, "get_npu_format", lambda tensor: tensor.npu_format, raising=False)
    logical_layout = {
        "shape": torch.Size([2, 4]),
        "stride": lambda: (4, 1),
        "dtype": torch.bfloat16,
        "device": SimpleNamespace(type="npu"),
    }
    targets = [SimpleNamespace(**logical_layout, npu_format=ACL_FORMAT_FRACTAL_NZ, copy_=Mock()) for _ in range(2)]
    sources = [
        SimpleNamespace(**logical_layout, npu_format=ACL_FORMAT_FRACTAL_NZ),
        SimpleNamespace(**logical_layout, npu_format=ACL_FORMAT_FRACTAL_ND),
    ]
    layer = SimpleNamespace(experts=targets)
    with pytest.raises(ValueError, match=r"experts\[1\]: physical format changed"):
        refill_expert_list(layer, "experts", sources)
    assert layer.experts is targets
    for target in targets:
        target.copy_.assert_not_called()


@pytest.mark.parametrize("change", ["shape", "dtype", "stride"])
def test_runtime_tensor_layout_change_keeps_captured_data(monkeypatch, change):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    owner = torch.nn.Module()
    owner.projection = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    original = torch.ones(2, 4, dtype=torch.bfloat16)
    refill_runtime_parameter(owner, "projection", original)
    assert owner.projection.data_ptr() == original.data_ptr()
    updated = torch.full_like(original, 2)
    # Direct begin rebinds .data to checkpoint storage before the next refill.
    owner.projection.data = torch.empty_like(original)
    refill_runtime_parameter(owner, "projection", updated)
    assert owner.projection.data_ptr() == original.data_ptr()
    incompatible = {
        "shape": torch.zeros(4, 2, dtype=torch.bfloat16),
        "dtype": torch.zeros(2, 4),
        "stride": torch.zeros(4, 2, dtype=torch.bfloat16).T,
    }[change]
    with pytest.raises(ValueError, match="runtime layout changed"):
        refill_runtime_parameter(owner, "projection", incompatible)
    assert torch.equal(original, updated)
    assert owner.projection.data_ptr() == original.data_ptr()


def test_interleaved_expert_staging_and_full_retry():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = SimpleNamespace(is_act_and_mul=True)
            self.quant_method = SimpleNamespace(_process_expert_weights=self.postprocess)
            self.flushes = 0
            for name, rows in (("w13_weight", 4), ("w2_weight", 2)):
                tensor = torch.zeros(2, rows, 2, dtype=torch.bfloat16)
                param = torch.nn.Parameter(tensor, requires_grad=False)
                param._ckpt = (None, (tuple(tensor.shape), tensor.stride(), tensor.dtype, 0))
                param._direct_reload_runtime = tensor.transpose(1, 2).contiguous()
                param.weight_loader = self.weight_loader
                self.register_parameter(name, param)
            assert ExpertModuleStaging.supports(self)
            self.stage = ExpertModuleStaging(self, ("w13_weight", "w2_weight"))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id if expert_id < 2 else -1

        def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            dst = param.data[expert_id]
            if shard_id in ("w1", "w3"):
                dst = dst.chunk(2, 0)[shard_id == "w3"]
            dst.copy_(loaded_weight)
            return True if return_success else None

        def postprocess(self, layer, names):
            for name in names:
                param = getattr(layer, name)
                param._direct_reload_runtime.copy_(param.data.transpose(1, 2).contiguous())
            self.flushes += 1

        def send(self, shard, expert, value):
            name = "w2_weight" if shard == "w2" else "w13_weight"
            param = getattr(self, name)
            buffer = torch.full((2, 2), value, dtype=torch.bfloat16)
            ok = param.weight_loader(param, buffer, name, shard, expert, return_success=True)
            buffer.zero_()  # The receiver no longer depends on the packed slot.
            return ok

    a, b = Layer(), Layer()
    a.w13_weight.data.fill_(5)
    a.stage.begin()
    b.stage.begin()
    assert a.w13_weight.numel() == 0
    a.w13_weight.data.zero_()  # Upstream begin resets parameters with loader padding.
    assert torch.all(a.w13_weight._direct_reload_runtime == 5)
    assert not a.send("w2", 3, 42)
    assert not a.stage.allocated
    assert a.send("w3", 1, 9)
    assert a.send("w3", 1, 8)
    assert b.send("w2", 0, 7)
    with pytest.raises(RuntimeError, match="incomplete"):
        a.stage.finish()
    a.stage.begin()
    for expert in (1, 0):
        for shard, value in (("w3", 3), ("w2", 2), ("w1", 1)):
            assert a.send(shard, expert, value)
    assert a.flushes == 1 and not a.stage.allocated
    assert b.flushes == 0 and b.stage.allocated == {"w2_weight"}
    a.stage.finish()
    assert torch.all(a.w13_weight[:, :, :2] == 1) and torch.all(a.w13_weight[:, :, 2:] == 3)
    assert torch.all(a.w2_weight == 2)
    assert a.w13_weight.data_ptr() == a.w13_weight._direct_reload_runtime.data_ptr()


@pytest.mark.parametrize("offset", [0, 3])
def test_checkpoint_staging_preserves_strides_and_offset(offset):
    from vllm_ascend.model_loader.weight_reload import _checkpoint_buffer

    param = torch.nn.Parameter(torch.empty(0))
    param._ckpt = (object(), ((2, 3), (5, 1), torch.bfloat16, offset))
    buf = _checkpoint_buffer(param).zero_()
    assert (tuple(buf.shape), buf.stride(), buf.dtype, buf.storage_offset()) == param._ckpt[1]
    assert buf.untyped_storage().nbytes() == (offset + 8) * buf.element_size()


def test_mixed_checkpoint_planning_uses_upstream_before_starting_staging(monkeypatch):
    from vllm.model_executor.model_loader.reload import direct

    from vllm_ascend.model_loader import weight_reload

    model = torch.nn.Module()
    model._ckpt_owner = object()
    model.block = torch.nn.Module()
    stage = Mock()
    a = torch.nn.Parameter(torch.empty(2))
    a._ckpt = (model._ckpt_owner, direct._layout(a), {}, a.data_ptr())
    a._ascend_expert_stage = stage
    model.register_parameter("expert", a)
    b = torch.arange(6).reshape(2, 3)
    b._ckpt = (model._ckpt_owner, ((3, 2), (2, 1), b.dtype, 0), {}, b.data_ptr())
    model.block.register_buffer("plain", b)
    planner = Mock(wraps=weight_reload._UPSTREAM_CHECKPOINT_VIEWS)
    monkeypatch.setattr(weight_reload, "_UPSTREAM_CHECKPOINT_VIEWS", planner)

    [(param, view)] = weight_reload.checkpoint_views(model)
    assert param is b and view.shape == (3, 2) and view.data_ptr() == b.data_ptr()
    assert list(direct.loadable_tensors(planner.call_args.args[0])) == [("block.plain", b)]
    stage.begin.assert_called_once()
    stage.reset_mock()
    b._ckpt = (object(), *b._ckpt[1:])
    with pytest.raises(RuntimeError, match=r"block\.plain is shared"):
        weight_reload.checkpoint_views(model)
    stage.begin.assert_not_called()
