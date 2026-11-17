"""Allocation regressions for review F1-F8; NPU companion executes real casts."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import PackedvLLMParameter
from vllm.model_executor.utils import replace_parameter

from tests.ut.quantization import test_keep_param_objects as fixtures
from tests.ut.quantization.test_keep_param_objects import (
    assert_kept,
    get_ascend_config,
    module,
    register,
    snapshot,
)
from tests.ut.quantization.test_nd_reload_storage import build as old_build
from tests.ut.quantization.test_nd_reload_storage import runtime as old_runtime
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod

# Re-export the existing fixture without redefining an imported name.
config = fixtures.config

CASES = [
    "bf-moe",
    "w8-moe",
    "w4-moe",
    "pdmix",
    "laos",
    "mx16-moe",
    "ds-normal",
    "ds-wo_a",
    "ds-wo_b",
    "ds-e8-normal",
    "ds-e8-wo_a",
    "ds-e8-wo_b",
    "kpool",
    "kpool-bf16",
    "flat-int",
]


def build(case, version, vc, mp, device=None):
    device = torch.device("cpu") if device is None else device
    if case in ("w8-moe", "w4-moe", "flat-int"):
        return old_build(case, version, vc, mp, device)
    layer = torch.nn.Module()
    layer.activation = "silu"
    get_ascend_config().enable_fused_mc2 = 0
    get_ascend_config().eplb_config.dynamic_eplb = False
    if case == "bf-moe":
        import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as upstream

        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

        mp.setattr(upstream, "select_unquantized_moe_backend", lambda **kw: (None, None))
        get_ascend_config().enable_fused_mc2 = 1
        scheme = AscendUnquantizedFusedMoEMethod()
        layer.quant_method = scheme
        values = {
            "w13_weight": torch.empty(2, 128, 64, dtype=torch.bfloat16),
            "w2_weight": torch.empty(2, 64, 64, dtype=torch.bfloat16),
        }
    elif case == "mx16-moe":
        scheme = module("wna16.w4a16_mxfp4", mp).AscendW4A16MXFP4FusedMoEMethod()
        values = scheme.get_weight(2, 64, 64, torch.bfloat16)
        values.update(scheme.get_dynamic_quant_param(2, 64, 64, torch.bfloat16))
        layer.quant_method = AscendFusedMoEMethod(scheme, None)
    elif case in ("laos", "pdmix"):
        scheme = (
            module("w4a4.w4a4_laos_dynamic", mp).AscendW4A4LaosDynamicLinearMethod()
            if case == "laos"
            else module("w8a8.w8a8_pdmix", mp).AscendW8A8PDMixLinearMethod()
        )
        values = scheme.get_weight(64, 32, torch.bfloat16)
        values.update(scheme.get_perchannel_param(32, torch.bfloat16))
        if case == "pdmix":
            values.update(scheme.get_pertensor_param(torch.bfloat16))
        values["weight_scale"] = values["weight_scale"].to(torch.bfloat16)
        layer.quant_method = AscendLinearMethod(scheme)
    elif case.startswith("ds-"):
        vc.model_config.hf_config.o_lora_rank = 64
        scheme = module("w8a8.w8a8_mxfp8", mp).AscendW8A8MXFP8DSDynamicLinearMethod([128, 128])
        layer.prefix = case
        values = {
            "weight": torch.empty(256, 256, dtype=torch.int8),
            "weight_scale": torch.empty(2, 2, dtype=torch.float8_e8m0fnu if "e8" in case else torch.float32),
        }
        layer.quant_method = AscendLinearMethod(scheme)
    elif case.startswith("kpool"):
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend

        layer = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
        torch.nn.Module.__init__(layer)
        layer.head_dim = 16
        layer.wk_weights_proj = torch.nn.Module()
        layer.k_norm = torch.nn.Module()
        for name in ("_key_weight", "_gate_weight", "_head_weight_f32", "_norm_weight_f32", "_norm_bias_f32"):
            layer.register_buffer(name, None, persistent=False)
        values = {
            "wk_weights_proj.weight": torch.empty(20, 64, dtype=torch.float32),
            "index_kpool_compress_gate": torch.empty(16, 64, dtype=torch.float32),
            "k_norm.weight": torch.empty(16, dtype=torch.float32),
            "k_norm.bias": torch.empty(16, dtype=torch.float32),
        }
    else:
        raise AssertionError(case)
    if case == "kpool-bf16":
        values = {name: tensor.to(torch.bfloat16) for name, tensor in values.items()}
    for name, tensor in list(values.items()):
        data = torch.arange(tensor.numel(), device=device).reshape(tensor.shape) % 5 + version
        if tensor.dtype == torch.float8_e8m0fnu:
            data = (127 + data.long() % 3).to(torch.uint8).view(torch.float8_e8m0fnu)
        elif "offset" in name:
            data.zero_()
        elif "scale" in name and tensor.dtype == torch.uint8:
            data = 125 + data % 3
        elif case.startswith("ds-") and name == "weight_scale":
            data = 2.0 ** (data % 3)
        elif tensor.is_floating_point():
            data = data / 16
        values[name] = data.to(tensor.dtype)
        parent, _, key = name.rpartition(".")
        register(layer.get_submodule(parent) if parent else layer, key, values[name].clone())
    before = snapshot(layer)
    process_weights_after_loading(layer, vc.model_config, device)
    return layer, values, before


def runtime(layer):
    result = old_runtime(layer)
    result.update({n: b for n, b in layer.named_buffers() if b is not None})
    return result


def raw(tensor):
    tensor = tensor.detach()
    if tensor.device.type == "npu" and torch_npu.get_npu_format(tensor) not in (0, 2):
        tensor = torch_npu.npu_format_cast(tensor, 2)
    return tensor.cpu().contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("case", CASES)
def test_review_a_a_b_a(config, monkeypatch, case):
    # A real allocation is required: identity casts hide the original F2 bug.
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda t, *a, **kw: t.clone())
    layer, _, before = build(case, 1, config, monkeypatch)
    held = {n: t.detach() for n, t in runtime(layer).items()}
    for version in (1, 3, 1):
        fresh, values, _ = build(case, version, config, monkeypatch)
        start_reload(layer)
        for n, value in values.items():
            p = layer.get_parameter(n)
            p.weight_loader(p, value)
        finish_reload(layer, config.model_config, torch.device("cpu"))
        assert_kept(layer, before)
        current = runtime(layer)
        expected = runtime(fresh)
        assert current.keys() == expected.keys() == held.keys()
        for n, t in current.items():
            assert t.data_ptr() == held[n].data_ptr(), n
            assert torch.equal(raw(t), raw(expected[n])), n
            assert torch.equal(raw(held[n]), raw(expected[n])), n


@pytest.mark.parametrize("mode", ["direct", "layerwise"])
def test_worker_failed_finish_ends_only_direct_session(config, monkeypatch, mode):
    from vllm_ascend.worker.worker import NPUWorker

    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)

    class Engine:
        resets = 0

        def finish_weight_update(self):
            raise ValueError("failed PWAL")

        def reset_weight_update_target(self):
            self.resets += 1

    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.weight_transfer_engine = Engine()
    worker._weight_update_active = True
    worker.model_runner = SimpleNamespace(reset_lora_state=lambda: pytest.fail("failed finish refreshed LoRA"))
    with pytest.raises(ValueError, match="failed PWAL"):
        worker.finish_weight_update()
    assert worker._weight_update_active is (mode != "direct")
    assert worker.weight_transfer_engine.resets == (mode == "direct")


def test_worker_full_retry_after_failed_flatquant_finish(config, monkeypatch):
    from vllm_ascend.worker.worker import NPUWorker

    layer, values, _ = old_build("flat-int", 1, config, monkeypatch)

    class Engine:
        resets = 0

        def start_weight_update(self):
            start_reload(layer)

        def finish_weight_update(self):
            finish_reload(layer, config.model_config, torch.device("cpu"))

        def reset_weight_update_target(self):
            self.resets += 1

    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.weight_transfer_engine = Engine()
    worker._weight_update_active = False
    worker.model_runner = SimpleNamespace(reset_lora_state=lambda: None)
    worker.start_weight_update()
    for n, t in values.items():
        p = layer.get_parameter(n)
        p.weight_loader(p, torch.tensor([0.5]) if n == "clip_ratio" else t)
    with pytest.raises(ValueError, match="cannot change FlatQuant"):
        worker.finish_weight_update()
    worker.start_weight_update()
    for n, t in values.items():
        p = layer.get_parameter(n)
        p.weight_loader(p, t)
    worker.finish_weight_update()
    cold, _, _ = old_build("flat-int", 1, config, monkeypatch)
    assert not worker._weight_update_active and worker.weight_transfer_engine.resets == 2
    assert all(torch.equal(raw(t), raw(runtime(cold)[n])) for n, t in runtime(layer).items())


def test_typed_packed_dimension_restored_for_loader(config):
    layer = torch.nn.Module()
    p = PackedvLLMParameter(
        data=torch.ones(4, 8, dtype=torch.int32),
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.register_parameter("weight", p)
    from vllm.model_executor.model_loader.reload.direct import record_checkpoint_layout

    record_checkpoint_layout(layer)
    replace_parameter(layer, "weight", p.t().contiguous())
    p._packed_dim = 0
    p._input_dim = 0
    p._output_dim = 1
    start_reload(layer)
    assert (p.input_dim, p.output_dim, p.packed_dim, p.packed_factor) == (1, 0, 1, 8)
    p.weight_loader(p, torch.full((4, 8), 3, dtype=torch.int32))
    assert torch.equal(p, torch.full((4, 8), 3, dtype=torch.int32))


@pytest.mark.parametrize("nz", [0, 1])
def test_disk_reload_checks_configured_nz_before_runner(config, monkeypatch, nz):
    from vllm_ascend.worker.worker import NPUWorker

    calls = []
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = SimpleNamespace(reload_weights=lambda: calls.append("reload"))
    get_ascend_config().weight_nz_mode = nz
    if nz:
        with pytest.raises(ValueError, match="FRACTAL_NZ mode is enabled"):
            worker.reload_weights()
        assert not calls
    else:
        worker.reload_weights()
        assert calls == ["reload"]


def test_invalid_physical_format_staging_refuses_before_model_mutation(config):
    from vllm.model_executor.model_loader.reload.direct import record_checkpoint_layout

    layer = torch.nn.Module()
    register(layer, "first", torch.arange(12).reshape(3, 4).float())
    register(layer, "second", torch.arange(12).reshape(3, 4).float())
    record_checkpoint_layout(layer)
    replace_parameter(layer, "first", layer.first.t().contiguous())
    layer.second._direct_reload_staging = torch.empty(2, 6)
    before = {n: (p.shape, p.data_ptr(), p.detach().clone()) for n, p in layer.named_parameters()}
    with pytest.raises(RuntimeError, match="incompatible checkpoint staging"):
        start_reload(layer)
    assert not hasattr(layer, "_direct_reload_begun")
    for n, p in layer.named_parameters():
        shape, pointer, value = before[n]
        assert p.shape == shape and p.data_ptr() == pointer
        assert torch.equal(p, value)


def test_ascend_reload_registration_is_idempotent(config):
    from vllm.model_executor.model_loader.reload import direct

    from vllm_ascend.quantization.reload import checkpoint_views, install_direct_reload_patch

    install_direct_reload_patch()
    install_direct_reload_patch()
    assert direct._checkpoint_views is checkpoint_views


def test_ascend_nd_planner_retains_upstream_capacity_error(config):
    from vllm.model_executor.model_loader.reload.direct import record_checkpoint_layout

    from vllm_ascend.quantization import reload as ascend_reload

    layer = torch.nn.Module()
    register(layer, "weight", torch.ones(4, 8))
    record_checkpoint_layout(layer)
    layer.weight.data = torch.ones(2, 2)
    for planner in (ascend_reload._UPSTREAM_CHECKPOINT_VIEWS, ascend_reload.checkpoint_views):
        with pytest.raises(RuntimeError, match="checkpoint layout needs"):
            planner(layer)


def test_ascend_derived_buffer_refuses_persistent_checkpoint_buffer(config):
    from vllm_ascend.quantization.reload import refill_derived_buffer

    layer = torch.nn.Module()
    layer.register_buffer("checkpoint", torch.ones(4))
    original = layer.checkpoint.clone()
    with pytest.raises(ValueError, match="non-persistent"):
        refill_derived_buffer(layer, "checkpoint", torch.zeros(4))
    assert torch.equal(layer.checkpoint, original)


def test_ascend_derived_buffer_refuses_layout_change_without_rebinding(config):
    from vllm_ascend.quantization.reload import refill_derived_buffer

    layer = torch.nn.Module()
    layer.register_buffer("runtime", torch.ones(4), persistent=False)
    held = layer.runtime
    with pytest.raises(ValueError, match="layout changed"):
        refill_derived_buffer(layer, "runtime", torch.zeros(8))
    assert layer.runtime is held and torch.equal(held, torch.ones(4))


def test_config_fixture_restores_process_state(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader.reload import direct

    from vllm_ascend import ascend_config
    from vllm_ascend.quantization.reload import checkpoint_views

    original_provider = direct._checkpoint_views
    original_config = ascend_config._ASCEND_CONFIG
    original_init_config = ascend_config._INIT_VLLM_CONFIG
    with monkeypatch.context() as patch:
        # Exercise the actual generator fixture, including its teardown and
        # monkeypatch dependency; a leaked provider/config affects later tests.
        setup = fixtures.config.__wrapped__(tmp_path, patch)
        try:
            next(setup)
            assert direct._checkpoint_views is checkpoint_views
            assert ascend_config._ASCEND_CONFIG is not original_config
        finally:
            setup.close()
    assert direct._checkpoint_views is original_provider
    assert ascend_config._ASCEND_CONFIG is original_config
    assert ascend_config._INIT_VLLM_CONFIG is original_init_config
