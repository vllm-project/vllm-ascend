# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from copy import copy
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn
from transformers import Qwen3MoeConfig
from vllm.config import DeviceConfig, ModelConfig, SpeculativeConfig, VllmConfig, replace
from vllm.config.parallel import ParallelConfig
from vllm.model_executor import model_loader
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import utils as spec_utils
from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend import ascend_config, ascend_forward_context, utils
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.patch.worker import patch_dspark
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.pcp_utils import build_dspark_draft_config


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
