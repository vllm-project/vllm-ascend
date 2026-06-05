# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest
from vllm.config.model import ModelConfig
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

import vllm_ascend.patch.platform.patch_pp_mtp as pp_mtp_patch


def test_model_runner_output_accepts_spec_token_ids():
    output = ModelRunnerOutput(
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        sampled_token_ids=[[10]],
        spec_token_ids=[[11, 12]],
    )

    assert output.spec_token_ids == [[11, 12]]
    assert hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "spec_token_ids")


def test_model_runner_output_patch_backfills_legacy_runtime(monkeypatch):
    class LegacyModelRunnerOutput:
        __dataclass_fields__ = {"req_ids": object()}

        def __init__(self, req_ids=None):
            self.req_ids = req_ids or []

    fake_outputs_mod = SimpleNamespace(
        ModelRunnerOutput=LegacyModelRunnerOutput,
        EMPTY_MODEL_RUNNER_OUTPUT=LegacyModelRunnerOutput(),
    )

    import vllm.v1 as vllm_v1

    monkeypatch.setattr(vllm_v1, "outputs", fake_outputs_mod)

    pp_mtp_patch._patch_model_runner_output()

    patched_output_cls = cast(Any, LegacyModelRunnerOutput)
    output = patched_output_cls(
        req_ids=["req0"],
        spec_token_ids=[[100, 101]],
    )
    assert output.spec_token_ids == [[100, 101]]
    assert fake_outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT.spec_token_ids is None


def test_engine_core_post_step_skips_pp_batch_queue_spec_decode_path():
    fake_core = SimpleNamespace(
        batch_queue=object(),
        async_scheduling=False,
        use_spec_decode=True,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=True,
                kv_role="kv_producer",
            ),
        ),
    )

    assert EngineCore.post_step(fake_core, model_executed=True) is None


def test_engine_core_post_step_keeps_decode_dp_mtp_path():
    fake_core = SimpleNamespace(
        batch_queue=object(),
        async_scheduling=False,
        use_spec_decode=True,
        model_executor=SimpleNamespace(take_draft_token_ids=lambda: [[11, 12]]),
        scheduler=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=False,
                kv_role="kv_consumer",
            ),
        ),
    )

    fake_core.scheduler.update_draft_token_ids = lambda draft_token_ids: setattr(
        fake_core, "observed_draft_token_ids", draft_token_ids
    )

    EngineCore.post_step(fake_core, model_executed=True)

    assert fake_core.observed_draft_token_ids == [[11, 12]]


def test_model_config_validates_local_mtp_drafter_as_single_pp_rank(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["Qwen3_5MTP"],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
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
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)
