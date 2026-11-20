# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from vllm_ascend.ascend_config import AscendConfig, ExpertOffloadConfig
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.distributed.expert_transfer.expert_offload_config import (
    validate_expert_offload_config,
    validate_expert_offload_layer,
)


@pytest.fixture(autouse=True)
def a5_hardware(monkeypatch):
    profile = SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED)
    monkeypatch.setattr(
        "vllm_ascend.distributed.expert_transfer.expert_offload_config.get_current_hardware_profile", lambda: profile
    )


def make_configs():
    vc = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model="/models/target",
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(model_type="other_moe"),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
        load_config=SimpleNamespace(load_format="auto"),
    )
    ac = SimpleNamespace(
        mix_placement=False,
        enable_force_eplb=False,
        enable_fused_mc2=0,
        multistream_overlap_shared_expert=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        rl_config=SimpleNamespace(enabled=False),
    )
    return vc, ac


@pytest.mark.parametrize(
    "options", [{"enabled": True, "num_device_experts": 4}, {"enabled": "true", "num_device_experts": "4"}]
)
def test_default_off_and_explicit_enable(options):
    disabled = ExpertOffloadConfig()
    assert not disabled.enabled and disabled.num_device_experts is None
    validate_expert_offload_config(disabled, None, None)
    config = TypeAdapter(ExpertOffloadConfig).validate_python(options)
    assert config.enabled is True and config.num_device_experts == 4
    validate_expert_offload_config(config, *make_configs())
    validate_expert_offload_layer(config, 8, 2)


@pytest.mark.parametrize(
    "method,async_scheduling", [(None, True), ("dspark", True), ("mtp", True), ("deepseek_mtp", False)]
)
def test_async_multi_request_and_checkpoint_drafting(method, async_scheduling):
    vc, ac = make_configs()
    vc.scheduler_config.async_scheduling = async_scheduling
    if method is not None:
        dspark = method == "dspark"
        vc.model_config.hf_text_config.model_type = "deepseek_v4" if dspark else "glm5_next_text"
        vc.speculative_config = SimpleNamespace(
            method=method,
            enforce_eager=True,
            enable_adaptive_verification=False,
            draft_model_config=SimpleNamespace(
                model=vc.model_config.model,
                hf_config=SimpleNamespace(
                    model_type="deepseek_v4" if dspark else "glm5_next_mtp",
                    architectures=["DSparkDraftModel" if dspark else "Glm5NextMTPModel"],
                ),
            ),
        )
    validate_expert_offload_config(ExpertOffloadConfig(enabled=True, num_device_experts=68), vc, ac)


def test_full_decode_graph_configuration():
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    vc, ac = make_configs()
    vc.model_config.enforce_eager = False
    vc.compilation_config = SimpleNamespace(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    validate_expert_offload_config(ExpertOffloadConfig(enabled=True, num_device_experts=68), vc, ac)


def make_pool_configs(protected=2, suffix=2, *, method="dspark", draft_tokens=5):
    vc, ac = make_configs()
    if method is not None:
        dspark = method == "dspark"
        vc.model_config.hf_text_config.model_type = "deepseek_v4" if dspark else "glm5_next_text"
        vc.speculative_config = SimpleNamespace(
            method=method,
            num_speculative_tokens=draft_tokens,
            enforce_eager=True,
            enable_adaptive_verification=False,
            draft_model_config=SimpleNamespace(
                model=vc.model_config.model,
                hf_config=SimpleNamespace(
                    model_type="deepseek_v4" if dspark else "glm5_next_mtp",
                    architectures=["DSparkDraftModel" if dspark else "Glm5NextMTPModel"],
                ),
            ),
        )
    config = ExpertOffloadConfig(
        enabled=True,
        num_device_experts=84,
        dspark_expert_pool={"enabled": True, "protected_positions": protected, "suffix_experts": suffix},
    )
    ac.expert_offload_config = config
    return config, vc, ac


def test_pool_validation_rejects_enabled_pool_before_disabled_offload_returns():
    config = TypeAdapter(ExpertOffloadConfig).validate_python(
        {"enabled": False, "dspark_expert_pool": {"enabled": True, "protected_positions": 2, "suffix_experts": 2}}
    )
    with pytest.raises(ValueError, match="requires enabled expert offload"):
        validate_expert_offload_config(config, None, None)


@pytest.mark.parametrize("protected,suffix", [(None, 2), (0, 2), (7, 2), (2, None), (2, -1), (2, 7)])
def test_pool_validation_rejects_missing_and_out_of_range_parameters(protected, suffix):
    config, vc, ac = make_pool_configs(protected, suffix)
    with pytest.raises(ValueError, match="protected_positions|suffix_experts"):
        validate_expert_offload_config(config, vc, ac)


@pytest.mark.parametrize("protected,suffix", [(1, 0), (1, 6), (6, 0), (6, 6)])
def test_pool_validation_accepts_inclusive_parameter_boundaries(protected, suffix):
    validate_expert_offload_config(*make_pool_configs(protected, suffix))


@pytest.mark.parametrize(
    "method,draft_tokens", [(None, 5), ("mtp", 5), ("dspark", None), ("dspark", 0), ("dspark", -1)]
)
def test_pool_validation_requires_dspark_with_positive_draft_tokens(method, draft_tokens):
    config, vc, ac = make_pool_configs(method=method, draft_tokens=draft_tokens)
    with pytest.raises(ValueError, match="requires DSpark with a positive number of speculative tokens"):
        validate_expert_offload_config(config, vc, ac)


@pytest.mark.parametrize("offload_enabled", [False, True])
def test_pool_validation_ignores_disabled_pool_options(offload_enabled):
    config = ExpertOffloadConfig(
        enabled=offload_enabled,
        num_device_experts=84 if offload_enabled else None,
        dspark_expert_pool={"enabled": False, "protected_positions": -1, "suffix_experts": 7},
    )
    validate_expert_offload_config(config, *make_configs())


def test_pool_validation_runs_during_config_derivation():
    _, vc, ac = make_pool_configs(protected=0)
    with pytest.raises(ValueError, match="protected_positions"):
        AscendConfig.derive_and_validate(ac, vc)
