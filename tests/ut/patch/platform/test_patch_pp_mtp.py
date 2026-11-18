# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.model import ModelConfig

from tests.ut.patch.platform.release_scheduler_fixture import release_cached_request_abi
from vllm_ascend.patch.platform.patch_pp_mtp import (
    _apply_patch,
    _is_pd_prefill_node,
    _patch_engine_core,
    _patch_model_config_validation,
    _patch_model_runner_output,
    _patch_scheduler_make_cached_request_data,
    _patch_scheduler_update_after_schedule,
    _patch_scheduler_update_from_output,
    _use_pp_ipc_runtime_patch,
    _use_pp_mtp_runtime_patch,
)


@pytest.fixture(autouse=True)
def _release_ipc_payload_abi_on_main(monkeypatch, request):
    if vllm_version_is("0.33.0") or not request.node.originalname.startswith("test_pp_ipc_cached_request_data_"):
        return
    from vllm.v1.core.sched import scheduler as native_module

    legacy_scheduler, _ = release_cached_request_abi()
    monkeypatch.setattr(request.module, "Scheduler", legacy_scheduler)
    monkeypatch.setattr(native_module, "Scheduler", legacy_scheduler)
    _patch_scheduler_make_cached_request_data()


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("qwen3_5_mtp", "Qwen3_5MTP"),
        ("qwen3", "DSparkDraftModel"),
        ("qwen3", "Qwen3DSparkModel"),
        ("kimi_k3", "K3DSparkModel"),
    ],
)
def test_model_config_validates_local_drafter_as_single_pp_rank(monkeypatch, model_type, architecture):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type=model_type)
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=[architecture],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
        # Both supported vLLM lanes read this during model/parallel validation.
        enable_hc_sp=False,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)
    assert parallel_config.pipeline_parallel_size == 2


def test_model_config_keeps_target_model_pp_validation(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "generate"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["UnsupportedForPP"],
    )

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
        # Both supported vLLM lanes read this during model/parallel validation.
        enable_hc_sp=False,
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


def test_pp_ipc_runtime_patch_is_disabled_for_model_runner_v2():
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=object(),
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp=True) is False


def test_pp_ipc_runtime_patch_skips_pd_prefill_node():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            is_kv_consumer=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=object(),
    )

    assert _use_pp_ipc_runtime_patch(vllm_config, use_pp=True) is False


def test_pp_mtp_helpers_and_patch_reentry():
    assert _is_pd_prefill_node(SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_producer"))) is True
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        speculative_config=object(),
    )
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=True) is False
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=False) is False
    vllm_config.speculative_config = None
    assert _use_pp_mtp_runtime_patch(vllm_config, use_pp=True) is False

    _apply_patch()
    _patch_model_runner_output()
    _patch_engine_core()
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_make_cached_request_data()
    _patch_scheduler_update_from_output()
    _patch_model_config_validation()
