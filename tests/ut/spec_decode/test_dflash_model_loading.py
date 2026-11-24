# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from copy import copy
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3MoeConfig
from vllm.config import CUDAGraphMode, DeviceConfig, ModelConfig, SpeculativeConfig, VllmConfig, replace
from vllm.config.parallel import ParallelConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor import model_loader
from vllm.platforms import current_platform
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.spec_decode import utils as spec_utils
from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as standalone_ar

from vllm_ascend import ascend_config, ascend_forward_context, utils
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.patch.worker import patch_dspark
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext, model_capture_wrapper
from vllm_ascend.worker.v2.spec_decode import (
    _wrap_speculator_draft_model_context,
    init_speculator,
    resolve_speculator_draft_vllm_config,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.pcp_utils import build_dspark_draft_config
from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import AscendStandaloneARSpeculator


@pytest.fixture
def megamoe_target(tmp_path, monkeypatch):
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", None)
    monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", None)
    monkeypatch.setattr(ascend_config, "_MEGA_MOE_SUPPORTED", None)
    monkeypatch.setattr(utils, "_IS_MOE_MODEL", None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    cann_ops = ModuleType("cann_ops_transformer")
    cann_ops.__spec__ = ModuleSpec(cann_ops.__name__, loader=None)
    monkeypatch.setitem(sys.modules, cann_ops.__name__, cann_ops)
    monkeypatch.setattr(
        ascend_forward_context, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr(ascend_forward_context, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(patch_dspark, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    # v0.33.0 queries PP through these imported getters; main no longer does.
    # Only replace an existing getter so the fixture follows the real loader.
    for module in (spec_utils, eagle_utils):
        if hasattr(module, "get_pp_group"):
            monkeypatch.setattr(module, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    if hasattr(eagle_utils, "in_single_pp_stage"):
        monkeypatch.setattr(eagle_utils, "in_single_pp_stage", lambda: False)

    # On CPU, wire the real factory into the platform hook used by VllmConfig.
    # Keep each returned config so the test can inspect the draft's own derivation.
    initialized = []

    def check_and_update_config(config):
        initialized.append(ascend_config.init_ascend_config(config))

    monkeypatch.setattr(current_platform, "check_and_update_config", check_and_update_config)
    monkeypatch.setattr(current_platform, "uses_standalone_cpu_model_runner", lambda: False, raising=False)

    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True, max_model_len=128, dtype="bfloat16"
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=2, prefill_context_parallel_size=8, enable_expert_parallel=True
    )
    # Resolve metadata locally, without downloading a draft checkpoint.
    speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )
    speculative_config.draft_model_config = copy(model_config)
    speculative_config.draft_parallel_config = parallel_config
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        additional_config={"enable_fused_mc2": 2},
    )
    target_ascend_config = ascend_config.get_ascend_config()
    assert initialized == [target_ascend_config]

    def assert_target_active():
        assert ascend_config.get_ascend_config() is target_ascend_config
        assert ascend_config._INIT_VLLM_CONFIG is config
        assert ascend_config._MEGA_MOE_SUPPORTED is True
        assert ascend_forward_context.use_cann_megamoe(config)

    assert_target_active()
    return config, initialized, assert_target_active


@pytest.mark.parametrize(
    ("method", "speculator_cls"),
    [
        pytest.param("dflash", AscendDFlashSpeculator, id="dflash"),
        pytest.param("dflash", AscendDFlash2Speculator, id="dflash2"),
        pytest.param("dspark", AscendDSparkSpeculator, id="dspark"),
        pytest.param("mtp", AscendMTPSpeculator, id="mtp"),
    ],
)
def test_draft_loading_preserves_target_megamoe(monkeypatch, megamoe_target, method, speculator_cls):
    target_config, initialized, assert_target_active = megamoe_target
    target_config.speculative_config.method = method
    target = nn.Module()
    target.model = nn.Module()
    draft = nn.Module()
    draft.model = nn.Module()
    draft_config = copy(build_dspark_draft_config(target_config) if method == "dspark" else target_config)
    if method != "mtp":
        # Upstream DFlash initialization replicates DFlash, DFlash2 and DSpark
        # drafts across target PCP ranks. Start at the loader to avoid GPU buffers.
        draft_config.parallel_config = replace(draft_config.parallel_config, prefill_context_parallel_size=1)

    calls = []

    def get_model(*, vllm_config, model_config):
        assert ascend_config.is_draft_config_context()
        assert_target_active()
        assert not utils.lmhead_tp_enable()
        assert model_config is target_config.speculative_config.draft_model_config
        assert vllm_config is not target_config
        assert vllm_config.parallel_config.prefill_context_parallel_size == (8 if method == "mtp" else 1)
        # The replica's EP=1 downgrade belongs to the draft configuration.
        assert initialized[-1] is not initialized[0]
        assert initialized[-1].enable_fused_mc2 == (1 if method == "mtp" else 0)
        calls.append("get_model")
        return draft

    def post_process(config):
        assert config is draft_config
        assert_target_active()
        calls.append("post_process")

    has_post_process = speculator_cls is AscendDFlash2Speculator
    if has_post_process:
        draft.post_process = post_process

    # Stop before model construction and checkpoint I/O. Keep the real
    # upstream loaders and VllmConfig and AscendConfig initialization.
    for module in (dflash_utils, eagle_utils, model_loader):
        monkeypatch.setattr(module, "get_model", get_model)

    spec = speculator_cls.__new__(speculator_cls)
    spec.vllm_config = draft_config
    if method == "dspark":
        spec.use_draft_logits = False
        spec.draft_logits = None
        spec.enable_adaptive_verification = False
    if method == "mtp":
        spec.method = method
        spec.draft_model_config = target_config.speculative_config.draft_model_config
        # Trigger a second reconstruction inside the real MTP weight loader.
        target_config.speculative_config.moe_backend = "auto"
        spec.draft_vllm_config = spec._create_draft_vllm_config()
        assert len(initialized) == 2
        assert_target_active()
    assert spec.load_draft_model(target, set()) is draft

    assert len(initialized) == (3 if method == "mtp" else 2)
    assert calls == (["get_model", "post_process"] if has_post_process else ["get_model"])
    assert not ascend_config.is_draft_config_context()
    assert_target_active()


def _dense_draft_model_config(draft_path):
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=32,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=128,
    ).save_pretrained(draft_path)
    return ModelConfig(
        model=str(draft_path), skip_tokenizer_init=True, enforce_eager=True, max_model_len=128, dtype="bfloat16"
    )


@pytest.mark.parametrize(
    ("method", "speculator_cls", "is_moe"),
    [
        pytest.param("dflash", AscendDFlashSpeculator, False, id="dense-dflash"),
        pytest.param("dflash", AscendDFlashSpeculator, True, id="bf16-moe-dflash"),
        pytest.param("dspark", AscendDSparkSpeculator, True, id="bf16-moe-dspark"),
    ],
)
def test_dflash_draft_uses_own_config_for_eager_and_capture(
    monkeypatch, tmp_path, megamoe_target, method, speculator_cls, is_moe
):
    target_config, _, assert_target_active = megamoe_target
    draft_path = tmp_path / "draft"
    if is_moe:
        target_config.model_config.hf_config.save_pretrained(draft_path)
        draft_model_config = ModelConfig(
            model=str(draft_path), skip_tokenizer_init=True, enforce_eager=True, max_model_len=128, dtype="bfloat16"
        )
    else:
        draft_model_config = _dense_draft_model_config(draft_path)
    draft_model_config.hf_config.dflash_config = {"mask_token_id": 3}

    target_quant_config = AscendModelSlimConfig({"model.layers.0.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC"})
    target_config.model_config.quantization = target_quant_config.get_name()
    target_config.quant_config = target_quant_config
    speculative_config = target_config.speculative_config
    speculative_config.method = method
    speculative_config.draft_model_config = draft_model_config
    speculative_config.draft_parallel_config = speculative_config.create_draft_parallel_config(
        target_config.parallel_config, 1, draft_model_config
    )
    draft_dcp = 1
    speculative_config.draft_decode_context_parallel_size = draft_dcp

    class DraftModel(nn.Module):
        def forward(self):
            assert _EXTRA_CTX.is_draft_model
            assert _EXTRA_CTX.draft_moe_quant_type is QuantType.NONE
            assert not _EXTRA_CTX.use_mega_moe
            return _EXTRA_CTX.moe_comm_type

    target = nn.Module()
    target.model = nn.Module()
    draft = DraftModel()
    draft.model = nn.Module()

    def get_model(*, vllm_config, model_config):
        assert ascend_config.is_draft_config_context()
        assert model_config is draft_model_config
        assert model_config.dtype == torch.bfloat16
        assert model_config.is_moe is is_moe
        if is_moe:
            # Routed experts publish this on the draft ModelConfig during load.
            model_config.draft_moe_quant_type = QuantType.NONE
        return draft

    monkeypatch.setattr(dflash_utils, "get_model", get_model)
    monkeypatch.setattr(model_loader, "get_model", get_model)
    monkeypatch.setattr(dspark_speculator, "get_tp_group", lambda: SimpleNamespace(world_size=1), raising=False)
    monkeypatch.setattr(ascend_forward_context, "_mc2_tokens_capacity", 1024)
    monkeypatch.setattr(ascend_config.get_ascend_config(), "get_mc2_comm_alg", lambda: "fullmesh_v1")
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)

    speculator = init_speculator(target_config, torch.device("cpu"))
    assert isinstance(speculator, speculator_cls)
    target_state = {
        "is_draft_model": False,
        "is_draft_model_prefill": False,
        "draft_moe_quant_type": QuantType.NONE,
        "moe_comm_type": MoECommType.FUSED_MC2,
        "moe_comm_method": object(),
        "use_mega_moe": True,
        "num_tokens": 24,
        "max_tokens_across_dp": 24,
    }
    context = ForwardContext({}, {}, {}, additional_kwargs=target_state.copy())

    def capture_draft():
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        with override_forward_context(context), model_capture_wrapper(speculator, is_draft_model_prefill=False):
            return speculator.model.vllm_config, speculator.model()

    before_load_config = None
    if not is_moe:
        # Capture also has to resolve a draft whose model has no load-time binding.
        speculator.model = draft
        before_load_config, comm = capture_draft()
        assert comm is None

    speculator.load_model(target)
    draft_config = draft._ascend_draft_vllm_config
    if before_load_config is not None:
        assert draft_config is before_load_config

    # Fullmesh v1 caps regular MC2 at 512 tokens for TP1.
    expected_modes = [(24, MoECommType.MC2), (513, MoECommType.ALLTOALL)] if is_moe else [(24, None)]
    for num_tokens, expected in expected_modes:
        assert ascend_forward_context.select_moe_comm_method(num_tokens, target_config) is MoECommType.FUSED_MC2
        target_state.update(num_tokens=num_tokens, max_tokens_across_dp=num_tokens)
        context.additional_kwargs.update(target_state)
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        with override_forward_context(context):
            assert speculator.model() is expected
        capture_config, comm = capture_draft()
        assert comm is expected
        assert capture_config is draft_config
        assert speculator.model is draft
        assert all(context.additional_kwargs[key] == value for key, value in target_state.items())

    resolved_parallel = draft_config.parallel_config
    speculator_parallel = speculator.vllm_config.parallel_config
    assert draft_config.model_config is draft_model_config
    assert resolved_parallel.prefill_context_parallel_size == speculator_parallel.prefill_context_parallel_size == 1
    assert (
        resolved_parallel.decode_context_parallel_size == speculator_parallel.decode_context_parallel_size == draft_dcp
    )
    assert resolved_parallel.is_moe_model is is_moe
    assert draft_config.compilation_config is speculator.vllm_config.compilation_config
    assert draft_config.speculative_config is speculative_config
    assert draft_config.quant_config is None
    assert speculator_parallel.is_moe_model is True
    assert target_config.parallel_config.is_moe_model is True
    assert target_config.quant_config is target_quant_config
    assert_target_active()


def test_eagle_auto_moe_backend_keeps_dense_runtime_config(monkeypatch, tmp_path, megamoe_target):
    target_config, _, assert_target_active = megamoe_target
    spec_config = target_config.speculative_config
    spec_config.method = "eagle"
    spec_config.moe_backend = "auto"
    draft_model_config = _dense_draft_model_config(tmp_path / "eagle")
    spec_config.draft_model_config = draft_model_config
    spec_config.draft_parallel_config = spec_config.create_draft_parallel_config(
        target_config.parallel_config, 1, draft_model_config
    )

    target = nn.Module()
    target.model = nn.Module()
    draft = nn.Module()
    draft.model = nn.Module()
    speculator = init_speculator(target_config, torch.device("cpu"))
    draft_config = speculator.draft_vllm_config
    runtime_parallel = draft_config.parallel_config

    def get_model(*, vllm_config, model_config):
        assert model_config is draft_model_config
        assert vllm_config.parallel_config is not runtime_parallel
        return draft

    monkeypatch.setattr(eagle_utils, "get_model", get_model)
    speculator.load_model(target)

    assert draft._ascend_draft_vllm_config is draft_config
    assert draft_config.parallel_config is runtime_parallel
    assert runtime_parallel.is_moe_model is False
    assert target_config.parallel_config.is_moe_model is True
    assert_target_active()

    # A target-derived parallel config can still carry the target's MoE flag.
    monkeypatch.setattr(runtime_parallel, "is_moe_model", True)
    assert ascend_forward_context.select_moe_comm_method(24, draft_config) is None


def test_standalone_ar_binds_dense_draft_config_without_changing_target(monkeypatch, tmp_path, megamoe_target):
    target_config, initialized, assert_target_active = megamoe_target
    assert utils.is_moe_model(target_config)
    draft_model_config = _dense_draft_model_config(tmp_path / "dense")
    speculative_config = target_config.speculative_config
    speculative_config.method = "draft_model"
    speculative_config.draft_model_config = draft_model_config
    speculative_config.draft_parallel_config = ParallelConfig()

    class DenseDraft(nn.Module):
        def forward(self):
            assert _EXTRA_CTX.is_draft_model
            assert _EXTRA_CTX.moe_comm_type is None
            assert _EXTRA_CTX.moe_comm_method is None
            assert _EXTRA_CTX.use_mega_moe is False
            return _EXTRA_CTX.is_draft_model_prefill

    def load_draft(*, vllm_config, **kwargs):
        assert ascend_config.is_draft_config_context()
        assert_target_active()
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.parallel_config.is_moe_model is False
        return DenseDraft()

    monkeypatch.setattr(standalone_ar, "get_model", load_draft)
    spec = AscendStandaloneARSpeculator.__new__(AscendStandaloneARSpeculator)
    spec.vllm_config = target_config
    spec.speculative_config = speculative_config
    spec.draft_model_config = draft_model_config
    spec.draft_vllm_config = None
    spec.use_local_argmax_reduction = False
    spec.use_acceptance_estimator = False
    spec.use_draft_logits = False
    _wrap_speculator_draft_model_context(spec, target_config).load_model(nn.Module())

    draft = spec.model
    draft_config = spec._draft_vllm_config
    assert draft._ascend_draft_vllm_config is draft_config
    assert draft_config.parallel_config is not target_config.parallel_config
    assert not draft_config.parallel_config.enable_expert_parallel
    assert len(initialized) == 2
    assert not ascend_config.is_draft_config_context()
    assert_target_active()

    target_state = {
        "is_draft_model": False,
        "is_draft_model_prefill": False,
        "draft_moe_quant_type": QuantType.NONE,
        "moe_comm_type": MoECommType.FUSED_MC2,
        "moe_comm_method": object(),
        "use_mega_moe": True,
        "num_tokens": 24,
        "max_tokens_across_dp": 24,
    }
    forward_context = ForwardContext({}, {}, {}, additional_kwargs=target_state.copy())
    select_comm = Mock(wraps=ascend_forward_context.select_moe_comm_method)
    monkeypatch.setattr(ascend_forward_context, "select_moe_comm_method", select_comm)
    with override_forward_context(forward_context):
        for mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            forward_context.cudagraph_runtime_mode = mode
            spec.model()
            assert forward_context.additional_kwargs == target_state

        monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        for is_prefill in (False, True):
            _EXTRA_CTX.is_draft_model_prefill = not is_prefill
            with model_capture_wrapper(spec, is_draft_model_prefill=is_prefill):
                select_comm.reset_mock()
                assert spec.model() is is_prefill
                select_comm.assert_called_once()
            assert _EXTRA_CTX.is_draft_model_prefill is not is_prefill
        assert spec.model is draft
        # Even a different explicit capture config must yield to the binding.
        ModelWithContext(draft, is_draft_model=True, vllm_config=target_config)()
        # A capture without a model binding still resolves the private draft config.
        ModelWithContext(DenseDraft(), is_draft_model=True, speculator=spec)()
        assert all(forward_context.additional_kwargs[key] == value for key, value in target_state.items())

    assert select_comm.call_count > 0
    assert all(call.args[1] is draft_config for call in select_comm.call_args_list)
    assert target_config.parallel_config.is_moe_model is True
    assert_target_active()


@pytest.mark.parametrize(("comm_alg", "per_rank_limit"), [("fullmesh_v1", 512), ("fullmesh_v2", 256)])
@pytest.mark.parametrize(("target_tp", "draft_tp"), [(2, 2), (2, 1), (1, 1)], ids=["tp2-tp2", "tp2-tp1", "tp1-tp1"])
def test_standalone_ar_stays_eager_with_loaded_draft(
    monkeypatch, megamoe_target, comm_alg, per_rank_limit, target_tp, draft_tp
):
    config, _, _ = megamoe_target
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=target_tp, data_parallel_size=2, enable_expert_parallel=True, is_moe_model=True
    )
    config.speculative_config = SpeculativeConfig(
        method="draft_model",
        model=config.model_config.model,
        num_speculative_tokens=2,
        draft_tensor_parallel_size=draft_tp,
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config,
    )
    config.scheduler_config.max_num_seqs = 2048
    config.scheduler_config.max_num_batched_tokens = 4096
    capacity = per_rank_limit * draft_tp
    config.compilation_config.prefill_cudagraph_capture_token_sizes = [128, 256, 512, 1024, 2048]
    config.compilation_config.decode_cudagraph_capture_request_sizes = [128, 256, 512, 1024, 2048]
    config.compilation_config.max_cudagraph_capture_size = 2048

    monkeypatch.setattr(utils, "_IS_DRAFTER_MOE_MODEL", None)
    monkeypatch.setattr(ascend_forward_context, "_mc2_tokens_capacity", 2048)
    monkeypatch.setattr(ascend_config.get_ascend_config(), "get_mc2_comm_alg", lambda: comm_alg)
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cudagraph_utils, "is_breakable_cudagraph_enabled", lambda *args: False)

    class Bf16Draft(nn.Module):
        moe_comm_methods = {}

        def forward(self):
            return _EXTRA_CTX.moe_comm_type

    speculator = init_speculator(config, torch.device("cpu"))
    before_load = resolve_speculator_draft_vllm_config(speculator)
    direct = resolve_speculator_draft_vllm_config(None, config)

    def load_draft(*, vllm_config, **kwargs):
        assert resolve_speculator_draft_vllm_config(speculator) is vllm_config
        assert vllm_config.model_config.dtype == torch.bfloat16
        vllm_config.model_config.draft_moe_quant_type = QuantType.NONE
        return Bf16Draft()

    monkeypatch.setattr(standalone_ar, "get_model", load_draft)
    speculator.load_model(nn.Module())
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    bound_config = speculator.model._ascend_draft_vllm_config
    assert bound_config is speculator._draft_vllm_config
    assert speculator.vllm_config is config
    assert resolve_speculator_draft_vllm_config(None, config) is not bound_config
    context = ForwardContext({}, {}, {})

    def draft_comm(num_tokens):
        context.additional_kwargs.update(num_tokens=num_tokens, max_tokens_across_dp=num_tokens)
        with override_forward_context(context):
            return speculator.model()

    assert draft_comm(capacity // 2) is MoECommType.MC2
    assert draft_comm(capacity) is MoECommType.MC2
    assert draft_comm(2 * capacity) is MoECommType.ALLTOALL
    for manager, query_len in ((speculator.prefill_cudagraph_manager, 4), (speculator.decode_cudagraph_manager, 1)):
        assert manager.cudagraph_mode == CUDAGraphMode.NONE
        assert manager.needs_capture() is False
        assert not manager._capture_descs
        for num_tokens in (capacity // 2, capacity, 2 * capacity):
            desc = manager.dispatch(num_tokens // query_len, num_tokens, query_len, num_active_loras=0)
            assert desc.cg_mode == CUDAGraphMode.NONE
            assert desc.num_tokens == num_tokens

    # The target can call capture even when both draft managers are eager.
    speculator.capture()

    # A config without the derived ParallelConfig can still declare draft TP.
    monkeypatch.setattr(config.speculative_config, "draft_parallel_config", None)
    tp_only = resolve_speculator_draft_vllm_config(None, config)
    for resolved in (before_load, direct, bound_config, tp_only):
        parallel = resolved.parallel_config
        assert (parallel.tensor_parallel_size, parallel.data_parallel_size) == (draft_tp, 1)
        assert parallel.enable_expert_parallel and parallel.is_moe_model
    assert config.parallel_config.tensor_parallel_size == target_tp
    assert config.parallel_config.data_parallel_size == 2
