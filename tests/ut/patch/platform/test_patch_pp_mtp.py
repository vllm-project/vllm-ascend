# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.model import ModelConfig

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


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


@pytest.mark.parametrize(
    ("has_speculative_config", "is_last_rank", "broadcast_pp_output", "expected"),
    [
        (False, False, False, False),
        (False, True, True, False),
        (True, False, False, False),
        (True, True, False, True),
        (True, False, True, True),
    ],
)
def test_pp_mtp_defers_kv_connector_only_on_ranks_that_run_draft(
    monkeypatch,
    has_speculative_config,
    is_last_rank,
    broadcast_pp_output,
    expected,
):
    monkeypatch.setattr(
        "vllm_ascend.worker.model_runner_v1.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=is_last_rank),
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = object() if has_speculative_config else None
    runner.broadcast_pp_output = broadcast_pp_output

    assert runner._should_defer_kv_connector_finalize() is expected
