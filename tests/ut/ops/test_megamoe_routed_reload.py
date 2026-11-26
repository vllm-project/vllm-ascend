# SPDX-License-Identifier: Apache-2.0
"""Behavioral routed-expert and standalone-linear reload tests for A3 MegaMoe.

Every test uses non-square shapes and distinct per-expert/per-channel random
weights and compares reloaded captured views and ALLTOALL/MC2 fallback views
against a freshly loaded reference model.
"""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import WeightTransferConfig, get_current_vllm_config_or_none, set_current_vllm_config
from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine
from vllm_ascend.ops.fused_moe import routed_experts as routed_mod
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.worker import worker as worker_mod
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

# Non-square dimensions catch transposed-axis and expert-stride regressions.
_NUM_EXPERTS = 3
_HIDDEN_SIZE = 16
_INTERMEDIATE_SIZE = 12


def _make_moe_layer(scheme_factory):
    moe = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(moe)
    moe._fse_fuse_gate = False
    moe.ascend_shared_experts = None
    experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(experts)
    moe.routed_experts = experts
    scheme = scheme_factory()
    experts.quant_method = (
        scheme
        if isinstance(scheme, AscendUnquantizedFusedMoEMethod)
        else AscendFusedMoEMethod(scheme, SimpleNamespace())
    )
    experts.quant_method.create_weights(
        experts,
        num_experts=_NUM_EXPERTS,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size_per_partition=_INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    return moe


class ReloadableMoEModel(nn.Module):
    """Multi-layer MoE model with an optional standalone quantized linear."""

    def __init__(self, scheme_factory, *, num_layers: int = 2, linear_cls=None):
        super().__init__()
        self.layers = nn.ModuleList([_make_moe_layer(scheme_factory) for _ in range(num_layers)])
        if linear_cls is not None:
            self.proj = nn.Module()
            self.proj.quant_method = AscendLinearMethod(linear_cls())
            self.proj.quant_method.create_weights(
                self.proj, 16, [80], 16, 80, torch.bfloat16, weight_loader=default_weight_loader
            )

    def load_weights(self, weights):
        loaded = set()
        for name, weight in weights:
            param = self.get_parameter(name)
            param.weight_loader(param, weight)
            loaded.add(name)
        return loaded


def _random_checkpoint(model: nn.Module, seed: int, *, compressed_int4: bool = False) -> dict[str, torch.Tensor]:
    """Generate distinct per-expert, asymmetric weights and positive scales."""
    gen = torch.Generator().manual_seed(seed)
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name.endswith("_offset"):
            tensor = torch.zeros_like(param)
        elif param.dtype == torch.int8:
            lo, hi = (-8, 8) if compressed_int4 else (-100, 101)
            tensor = torch.randint(lo, hi, param.shape, dtype=torch.int8, generator=gen)
        elif param.dtype == torch.int32:
            tensor = torch.randint(-1000, 1001, param.shape, dtype=torch.int32, generator=gen)
        elif name.endswith("weight_scale") or name.endswith("_weight_scale"):
            tensor = (torch.rand(param.shape, generator=gen) * 0.5 + 0.05).to(param.dtype)
        elif name.endswith("_scale_bias"):
            tensor = (torch.rand(param.shape, generator=gen) - 0.5).to(param.dtype)
        else:
            tensor = (torch.randn(param.shape, generator=gen) * 0.25).to(param.dtype)
        checkpoint[name] = tensor
    return checkpoint


def _scheme_of(experts: AscendRoutedExperts):
    qm = experts.quant_method
    return getattr(qm, "quant_method", qm)


@pytest.fixture
def reload_env(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "npu", lambda tensor: tensor, raising=False)
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    monkeypatch.setattr(w4a8, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(w8a8_dynamic, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(w8a8_dynamic, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(w8a8_dynamic, "maybe_trans_nz", lambda weight: weight)
    context = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2, use_mega_moe=True)
    monkeypatch.setattr(w4a8, "_EXTRA_CTX", context)
    monkeypatch.setattr(w8a8_dynamic, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_mod, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_mod, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(routed_mod, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    monkeypatch.setattr(routed_mod, "use_cann_megamoe", lambda _: True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, quantization="ascend", word_embeddings_untied_by_checkpoint=False
        ),
        load_config=SimpleNamespace(device="cpu"),
        lora_config=None,
        quant_config=SimpleNamespace(get_name=lambda: w4a8.ASCEND_QUANTIZATION_METHOD, quant_description={}),
    )

    def build_and_load(
        scheme_factory, *, seed: int, num_layers: int = 2, linear_cls=None, compressed_int4: bool = False
    ):
        with torch.inference_mode(), set_current_vllm_config(config):
            model = ReloadableMoEModel(scheme_factory, num_layers=num_layers, linear_cls=linear_cls)
            record_metadata_for_reloading(model)
            checkpoint = _random_checkpoint(model, seed, compressed_int4=compressed_int4)
            model.load_weights(checkpoint.items())
            process_weights_after_loading(model, config.model_config, torch.device("cpu"))
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.model = model
        runner.vllm_config = config
        runner.ascend_config = SimpleNamespace(megamoe_shared_expert_fusion=False)
        runner.lora_config = runner.speculator = runner.encoder_cache = runner.pooling_runner = None
        return runner, checkpoint

    return SimpleNamespace(build=build_and_load, ctx=context, config=config)


def _assert_tensor_lists_match_and_alias(captured_list, live_list, reference_list):
    for old, live, ref in zip(captured_list, live_list, reference_list, strict=True):
        assert old.shape == live.shape == ref.shape
        assert old.dtype == live.dtype == ref.dtype
        assert old.data_ptr() == live.data_ptr()
        assert old.untyped_storage() == live.untyped_storage()
        torch.testing.assert_close(old, ref, rtol=0, atol=0)


def _assert_not_equal_any(tensors_a, tensors_b):
    assert any(not torch.equal(a, b) for a, b in zip(tensors_a, tensors_b, strict=True))


@pytest.mark.parametrize("dynamic_eplb", [False, True], ids=["registered", "per-expert-list"])
def test_bf16_multilayer_reload_preserves_storage_and_matches_fresh_reference(reload_env, dynamic_eplb):
    def make_scheme():
        scheme = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        nn.Module.__init__(scheme)
        scheme.moe = SimpleNamespace(
            is_act_and_mul=True,
            has_bias=False,
            hidden_dim_unpadded=_HIDDEN_SIZE,
            intermediate_size_per_partition_unpadded=_INTERMEDIATE_SIZE,
        )
        scheme.moe_kernel = None
        scheme.dynamic_eplb = dynamic_eplb
        return scheme

    runner, _ = reload_env.build(make_scheme, seed=11, num_layers=2)
    captured_per_layer = [
        _scheme_of(layer.routed_experts).get_fused_mc2_weights(layer.routed_experts) for layer in runner.model.layers
    ]
    initial_w1_snapshot = [t.clone() for t in captured_per_layer[0].w1]

    # Reload checkpoint B (layerwise checkpoint format) then C (kernel format
    # for registered parameters, or a second layerwise reload when EPLB splits
    # parameters into per-expert tensor lists) and verify both FUSED_MC2 and
    # ALLTOALL/MC2 fallback views match a fresh model.
    for seed, is_ckpt in [(29, True), (53, dynamic_eplb)]:
        fresh, ckpt = reload_env.build(make_scheme, seed=seed, num_layers=2)
        source = ckpt.items() if is_ckpt else fresh.model.named_parameters()
        runner.reload_weights(source, is_checkpoint_format=is_ckpt)

        for layer, fresh_layer, captured in zip(
            runner.model.layers, fresh.model.layers, captured_per_layer, strict=True
        ):
            experts = layer.routed_experts
            fresh_experts = fresh_layer.routed_experts
            scheme = _scheme_of(experts)
            fresh_scheme = _scheme_of(fresh_experts)

            reload_env.ctx.use_mega_moe = True
            live_mega = scheme.get_fused_mc2_weights(experts)
            ref_mega = fresh_scheme.get_fused_mc2_weights(fresh_experts)
            _assert_tensor_lists_match_and_alias(captured.w1, live_mega.w1, ref_mega.w1)
            _assert_tensor_lists_match_and_alias(captured.w2, live_mega.w2, ref_mega.w2)

            # Switch to fallback ALLTOALL/MC2 path and verify updated weights.
            reload_env.ctx.use_mega_moe = False
            live_fallback = scheme.get_mlp_weights(experts)
            ref_fallback = fresh_scheme.get_mlp_weights(fresh_experts)
            for got, ref in zip(live_fallback, ref_fallback, strict=True):
                if isinstance(got, list):
                    for g_t, r_t in zip(got, ref, strict=True):
                        torch.testing.assert_close(g_t, r_t, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(got, ref, rtol=0, atol=0)
            reload_env.ctx.use_mega_moe = True

    _assert_not_equal_any(captured_per_layer[0].w1, initial_w1_snapshot)


@pytest.mark.parametrize("use_expert_weight_list", [False, True], ids=["registered", "per-expert-list"])
@pytest.mark.parametrize("quant_method", [w4a8.ASCEND_QUANTIZATION_METHOD, w4a8.COMPRESSED_TENSORS_METHOD])
def test_w4a8_multilayer_reload_and_fallback_switch_match_fresh_reference(
    reload_env, quant_method, use_expert_weight_list
):
    compressed = quant_method == w4a8.COMPRESSED_TENSORS_METHOD

    def make_scheme():
        scheme = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
        scheme.quant_method = quant_method
        scheme.tp_size = 1
        scheme.use_expert_weight_list = use_expert_weight_list
        return scheme

    runner, _ = reload_env.build(make_scheme, seed=17, num_layers=2, compressed_int4=compressed)
    captured_per_layer = [
        _scheme_of(layer.routed_experts).get_fused_mc2_weights(layer.routed_experts) for layer in runner.model.layers
    ]
    initial_snapshot = [t.clone() for t in captured_per_layer[0].w1]

    # Layerwise checkpoint reload works for both registered and per-expert lists;
    # kernel-format reload updates registered parameters in place.
    steps = [(41, True), (67, use_expert_weight_list)]
    for seed, is_ckpt in steps:
        fresh, ckpt = reload_env.build(make_scheme, seed=seed, num_layers=2, compressed_int4=compressed)
        source = ckpt.items() if is_ckpt else fresh.model.named_parameters()
        runner.reload_weights(source, is_checkpoint_format=is_ckpt)

        for layer, fresh_layer, captured in zip(
            runner.model.layers, fresh.model.layers, captured_per_layer, strict=True
        ):
            experts = layer.routed_experts
            fresh_experts = fresh_layer.routed_experts
            scheme = _scheme_of(experts)
            fresh_scheme = _scheme_of(fresh_experts)

            # 1. FUSED_MC2 (MegaMoe) operands keep storage & match fresh reference.
            reload_env.ctx.use_mega_moe = True
            live = scheme.get_fused_mc2_weights(experts)
            ref = fresh_scheme.get_fused_mc2_weights(fresh_experts)
            _assert_tensor_lists_match_and_alias(captured.w1, live.w1, ref.w1)
            _assert_tensor_lists_match_and_alias(captured.w2, live.w2, ref.w2)
            _assert_tensor_lists_match_and_alias(captured.w1_scale, live.w1_scale, ref.w1_scale)
            _assert_tensor_lists_match_and_alias(captured.w2_scale, live.w2_scale, ref.w2_scale)
            _assert_tensor_lists_match_and_alias(captured.w1_scale_bias, live.w1_scale_bias, ref.w1_scale_bias)
            _assert_tensor_lists_match_and_alias(captured.w2_scale_bias, live.w2_scale_bias, ref.w2_scale_bias)

            # 2. Switch to ALLTOALL / MC2 fallback: zero-copy int32 view & scales match fresh reference.
            reload_env.ctx.use_mega_moe = False
            fallback = scheme.get_fused_mc2_weights(experts)
            ref_fallback = fresh_scheme.get_fused_mc2_weights(fresh_experts)
            for got_list, ref_list in (
                (fallback.w1, ref_fallback.w1),
                (fallback.w2, ref_fallback.w2),
                (fallback.w1_scale, ref_fallback.w1_scale),
                (fallback.w2_scale, ref_fallback.w2_scale),
                (fallback.w1_scale_bias, ref_fallback.w1_scale_bias),
                (fallback.w2_scale_bias, ref_fallback.w2_scale_bias),
            ):
                for g_t, r_t in zip(got_list, ref_list, strict=True):
                    torch.testing.assert_close(g_t, r_t, rtol=0, atol=0)
            if not use_expert_weight_list:
                # 2D int8 MegaMoe view and 3D int32 fallback view share storage zero-copy.
                for w_mega, w_fb in ((live.w1, fallback.w1[0]), (live.w2, fallback.w2[0])):
                    assert w_fb.dtype == torch.int32
                    for expert_view, packed_slice in zip(w_mega, w_fb.view(torch.int8).unbind(0), strict=True):
                        assert expert_view.data_ptr() == packed_slice.data_ptr()
            reload_env.ctx.use_mega_moe = True

    _assert_not_equal_any(captured_per_layer[0].w1, initial_snapshot)


@pytest.mark.parametrize("use_expert_weight_list", [False, True], ids=["registered", "per-expert-list"])
def test_w8a8_multilayer_reload_refills_routed_and_fallback_scales(reload_env, use_expert_weight_list):
    def make_scheme():
        scheme = w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
        scheme.use_expert_weight_list = use_expert_weight_list
        return scheme

    runner, _ = reload_env.build(make_scheme, seed=23, num_layers=2)
    captured_per_layer = [
        _scheme_of(layer.routed_experts).get_fused_mc2_weights(layer.routed_experts) for layer in runner.model.layers
    ]
    fallback_scales_per_layer = [
        _scheme_of(layer.routed_experts).get_mlp_weights(layer.routed_experts) for layer in runner.model.layers
    ]
    initial_scale_snapshot = [t.clone() for t in captured_per_layer[0].w1_scale]

    steps = [(47, True), (71, use_expert_weight_list)]
    for seed, is_ckpt in steps:
        fresh, ckpt = reload_env.build(make_scheme, seed=seed, num_layers=2)
        source = ckpt.items() if is_ckpt else fresh.model.named_parameters()
        runner.reload_weights(source, is_checkpoint_format=is_ckpt)

        for layer, fresh_layer, captured, initial_fb in zip(
            runner.model.layers, fresh.model.layers, captured_per_layer, fallback_scales_per_layer, strict=True
        ):
            experts = layer.routed_experts
            fresh_experts = fresh_layer.routed_experts
            scheme = _scheme_of(experts)
            fresh_scheme = _scheme_of(fresh_experts)

            live = scheme.get_fused_mc2_weights(experts)
            ref = fresh_scheme.get_fused_mc2_weights(fresh_experts)
            _assert_tensor_lists_match_and_alias(captured.w1, live.w1, ref.w1)
            _assert_tensor_lists_match_and_alias(captured.w2, live.w2, ref.w2)
            _assert_tensor_lists_match_and_alias(captured.w1_scale, live.w1_scale, ref.w1_scale)
            _assert_tensor_lists_match_and_alias(captured.w2_scale, live.w2_scale, ref.w2_scale)

            live_fb = scheme.get_mlp_weights(experts)
            ref_fb = fresh_scheme.get_mlp_weights(fresh_experts)
            _assert_tensor_lists_match_and_alias(initial_fb.w1_scale, live_fb.w1_scale, ref_fb.w1_scale)
            _assert_tensor_lists_match_and_alias(initial_fb.w2_scale, live_fb.w2_scale, ref_fb.w2_scale)

    _assert_not_equal_any(captured_per_layer[0].w1_scale, initial_scale_snapshot)


@pytest.mark.parametrize(
    ("moe_cls", "linear_cls"),
    [
        (w4a8.AscendW4A8DynamicFusedMoEMethod, w4a8.AscendW4A8DynamicLinearMethod),
        (w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod, w8a8_dynamic.AscendW8A8DynamicLinearMethod),
    ],
    ids=["w4a8", "w8a8"],
)
def test_reload_and_trainer_update_refresh_standalone_scale_and_routed_weights(
    reload_env, monkeypatch, moe_cls, linear_cls
):
    def make_scheme():
        scheme = moe_cls.__new__(moe_cls)
        scheme.quant_method = w4a8.ASCEND_QUANTIZATION_METHOD
        scheme.tp_size = 1
        scheme.use_expert_weight_list = False
        return scheme

    runner, _ = reload_env.build(make_scheme, seed=31, num_layers=2, linear_cls=linear_cls)
    projection = runner.model.proj
    captured_scale = projection.weight_scale_fp32
    captured_scale_ptr = captured_scale.data_ptr()
    captured_routed = [
        _scheme_of(layer.routed_experts).get_fused_mc2_weights(layer.routed_experts) for layer in runner.model.layers
    ]

    # 1. Layerwise checkpoint reload
    fresh_b, ckpt_b = reload_env.build(make_scheme, seed=59, num_layers=2, linear_cls=linear_cls)
    runner.reload_weights(ckpt_b.items())
    assert projection.weight_scale_fp32 is captured_scale
    assert captured_scale.data_ptr() == captured_scale_ptr
    torch.testing.assert_close(captured_scale, fresh_b.model.proj.weight_scale_fp32, rtol=0, atol=0)

    # 2. Kernel-format reload
    fresh_c, _ = reload_env.build(make_scheme, seed=83, num_layers=2, linear_cls=linear_cls)
    runner.reload_weights(fresh_c.model.named_parameters(), is_checkpoint_format=False)
    assert projection.weight_scale_fp32 is captured_scale
    assert captured_scale.data_ptr() == captured_scale_ptr
    torch.testing.assert_close(captured_scale, fresh_c.model.proj.weight_scale_fp32, rtol=0, atol=0)

    # 3. Trainer HCCL weight-transfer lifecycle (start -> update -> finish)
    fresh_d, ckpt_d = reload_env.build(make_scheme, seed=107, num_layers=2, linear_cls=linear_cls)
    updates = iter(ckpt_d.values())
    config = runner.vllm_config
    config.parallel_config = SimpleNamespace()
    config.speculative_config = None
    runner.extensible_kv_cache = None

    def broadcast(weight, *, src, stream):
        assert get_current_vllm_config_or_none() is config
        weight.copy_(next(updates))

    monkeypatch.setattr(torch.npu, "device", lambda *_: nullcontext(), raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: None, raising=False)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(worker_mod, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=0))

    engine = HCCLWeightTransferEngine(WeightTransferConfig(backend="hccl"), config, torch.device("cpu"), runner.model)
    engine.model_update_group = SimpleNamespace(broadcast=broadcast)
    engine._init_info = SimpleNamespace(packed=False, packed_buffer_size_bytes=None, packed_num_buffers=None)
    engine._last_hccl_update_info = {}
    worker = worker_mod.NPUWorker.__new__(worker_mod.NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.device = torch.device("cpu")
    worker.model_runner, worker.weight_transfer_engine, worker._weight_update_active = runner, engine, False
    worker._weight_update_is_draft = False
    worker._incomplete_weight_updates = set()
    worker._kv_cache_reclaimable = False
    worker._weight_update_kv_cache_loan = ExitStack()
    with torch.inference_mode():
        worker.start_weight_update()
        worker.update_weights(
            {
                "names": list(ckpt_d),
                "dtype_names": [str(t.dtype).removeprefix("torch.") for t in ckpt_d.values()],
                "shapes": [list(t.shape) for t in ckpt_d.values()],
            }
        )
        worker.finish_weight_update()

    assert not worker._weight_update_active
    assert projection.weight_scale_fp32 is captured_scale
    assert captured_scale.data_ptr() == captured_scale_ptr
    torch.testing.assert_close(captured_scale, fresh_d.model.proj.weight_scale_fp32, rtol=0, atol=0)
    for layer, fresh_layer, captured in zip(runner.model.layers, fresh_d.model.layers, captured_routed, strict=True):
        live = _scheme_of(layer.routed_experts).get_fused_mc2_weights(layer.routed_experts)
        ref = _scheme_of(fresh_layer.routed_experts).get_fused_mc2_weights(fresh_layer.routed_experts)
        _assert_tensor_lists_match_and_alias(captured.w1, live.w1, ref.w1)
        _assert_tensor_lists_match_and_alias(captured.w2, live.w2, ref.w2)


@pytest.mark.parametrize(
    ("moe_cls", "linear_cls"),
    [
        (w4a8.AscendW4A8DynamicFusedMoEMethod, w4a8.AscendW4A8DynamicLinearMethod),
        (w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod, w8a8_dynamic.AscendW8A8DynamicLinearMethod),
    ],
    ids=["w4a8_linear", "w8a8_linear"],
)
def test_linear_direct_inplace_reload_refreshes_weight_scale_fp32(reload_env, moe_cls, linear_cls):
    """Worker-V1 / direct process_weights_after_loading updates weight_scale_fp32 in place."""

    def make_scheme():
        scheme = moe_cls.__new__(moe_cls)
        scheme.quant_method = w4a8.ASCEND_QUANTIZATION_METHOD
        scheme.tp_size = 1
        scheme.use_expert_weight_list = False
        return scheme

    runner, _ = reload_env.build(make_scheme, seed=41, num_layers=1, linear_cls=linear_cls)
    projection = runner.model.proj
    captured_scale = projection.weight_scale_fp32
    captured_ptr = captured_scale.data_ptr()

    fresh, ckpt = reload_env.build(make_scheme, seed=73, num_layers=1, linear_cls=linear_cls)
    projection.weight.data = ckpt["proj.weight"].clone()
    projection.weight_scale.data = ckpt["proj.weight_scale"].clone()
    projection.weight_offset.data = ckpt["proj.weight_offset"].clone()
    if "proj.scale_bias" in ckpt:
        projection.scale_bias.data = ckpt["proj.scale_bias"].clone()

    projection.quant_method.process_weights_after_loading(projection)
    assert projection.weight_scale_fp32 is captured_scale
    assert projection.weight_scale_fp32.data_ptr() == captured_ptr
    torch.testing.assert_close(projection.weight_scale_fp32, fresh.model.proj.weight_scale_fp32, rtol=0, atol=0)


def test_reset_lora_state_preserves_super_on_non_034_and_skips_moe_scan_with_lora(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2 import model_runner as mr_mod

    super_calls: list[str] = []
    moe_calls: list[str] = []

    monkeypatch.setattr(GPUModelRunner, "reset_lora_state", lambda self: super_calls.append("super"), raising=False)
    monkeypatch.setattr(mr_mod, "vllm_version_is", lambda ver: False)

    def _scan_modules():
        moe_calls.append("scan")
        return []

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(lora_config=None)
    runner.get_model = lambda: SimpleNamespace(modules=_scan_modules)
    runner.reset_lora_state()
    assert super_calls == ["super"]
    assert moe_calls == []

    monkeypatch.setattr(mr_mod, "vllm_version_is", lambda ver: ver == "0.34.0")
    runner.vllm_config = SimpleNamespace(lora_config=SimpleNamespace(max_loras=1))
    runner.reset_lora_state()
    assert super_calls == ["super", "super"]
    assert moe_calls == []
