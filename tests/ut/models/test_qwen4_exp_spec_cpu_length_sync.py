# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Key regressions for the initialization-time CPU length policy."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm_ascend.models.qwen4_exp import cpu_length_sync as policy


@pytest.fixture
def runner():
    try:
        from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAMetadataBuilder

        from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
        from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
    except ModuleNotFoundError as exc:
        if exc.name == "vllm.models.qwen4_exp" or (exc.name or "").startswith("vllm.models.qwen4_exp."):
            pytest.skip("This vLLM version does not provide Qwen4Exp.")
        raise

    group = NS(
        backend=AscendQSABackend,
        get_metadata_builder=lambda _: object.__new__(Qwen4ExpQSAMetadataBuilder),
    )
    draft = object.__new__(AscendMTPSpeculator)
    draft.attn_groups = [[group]]
    draft.attn_architecture, draft.attn_backend = "QSA", AscendQSABackend
    return NS(
        model_state=object.__new__(AscendQwen4ExpModelState),
        model_config=NS(architecture="Qwen4ExpForCausalLM"),
        ascend_config=NS(sparse_kv_offload_config=NS(enabled=False)),
        parallel_config=NS(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        speculator=draft,
        attn_groups=[[group]],
        num_computed_tokens_stream=NS(synchronize=Mock()),
        supports_mm_inputs=False,
    )


def test_incomplete_runtime_keeps_exact_lengths():
    for runtime in (NS(), NS(ascend_config=NS())):
        assert policy.can_skip_qwen_cpu_length_sync(runtime) is False


def test_supported_model_enables_policy_by_default(runner):
    runner.model_state.initialize_spec_cpu_length_sync(runner)
    assert runner.model_state.can_skip_spec_cpu_length_sync() is True


@pytest.mark.parametrize("multimodal", [False, True])
def test_text_and_ordinary_multimodal_use_device_lengths(runner, multimodal):
    if multimodal:
        runner.model_config.architecture = "Qwen4ExpForConditionalGeneration"
        runner.supports_mm_inputs = True
        runner.model_state.encoder_runner = NS(is_realtime=False)
    assert policy.can_skip_qwen_cpu_length_sync(runner) is True


@pytest.mark.parametrize(
    "unsupported",
    [
        "model",
        "architecture",
        "speculator",
        "target",
        "draft",
        "realtime",
        "dcp",
        "pcp",
        "offload",
        "empty_target",
        "empty_draft",
    ],
)
def test_unsupported_consumers_keep_exact_lengths(runner, unsupported):
    if unsupported == "model":
        runner.model_state = NS()
    elif unsupported == "architecture":
        runner.model_config.architecture = "LlamaForCausalLM"
    elif unsupported == "speculator":
        runner.speculator = NS()
    elif unsupported == "realtime":
        runner.supports_mm_inputs = True
        runner.model_state.encoder_runner = NS(is_realtime=True)
    elif unsupported == "dcp":
        runner.parallel_config.decode_context_parallel_size = 2
    elif unsupported == "pcp":
        runner.parallel_config.prefill_context_parallel_size = 2
    elif unsupported == "offload":
        runner.ascend_config.sparse_kv_offload_config.enabled = True
    elif unsupported in ("empty_target", "empty_draft"):
        owner = runner if unsupported == "empty_target" else runner.speculator
        owner.attn_groups = []
    else:
        owner = runner if unsupported == "target" else runner.speculator
        owner.attn_groups = [[NS(backend=runner.attn_groups[0][0].backend, get_metadata_builder=lambda _: object())]]
    assert policy.can_skip_qwen_cpu_length_sync(runner) is False


@pytest.mark.parametrize("eligible", [False, True])
def test_policy_is_frozen_until_reinitialization(runner, monkeypatch, eligible):
    state = runner.model_state
    runner.ascend_config.sparse_kv_offload_config.enabled = not eligible
    state.initialize_spec_cpu_length_sync(runner)
    assert state.can_skip_spec_cpu_length_sync() is eligible
    runner.ascend_config.sparse_kv_offload_config.enabled = eligible
    with monkeypatch.context() as patch:
        patch.setattr(policy, "can_skip_qwen_cpu_length_sync", Mock(side_effect=AssertionError("per-step check")))
        assert state.can_skip_spec_cpu_length_sync() is eligible
    state.initialize_spec_cpu_length_sync(runner)
    assert state.can_skip_spec_cpu_length_sync() is not eligible


def test_pending_snapshot_is_drained_before_enabling(runner):
    state = runner.model_state
    assert state.can_skip_spec_cpu_length_sync() is False

    def synchronize():
        assert state.can_skip_spec_cpu_length_sync() is False

    runner.num_computed_tokens_stream.synchronize.side_effect = synchronize
    state.initialize_spec_cpu_length_sync(runner)
    runner.num_computed_tokens_stream.synchronize.assert_called_once_with()
    assert state.can_skip_spec_cpu_length_sync() is True
    runner.ascend_config.sparse_kv_offload_config.enabled = True
    runner.num_computed_tokens_stream.synchronize.reset_mock()
    state.initialize_spec_cpu_length_sync(runner)
    runner.num_computed_tokens_stream.synchronize.assert_not_called()
    assert state.can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize("stream", [None, NS(synchronize=None)])
def test_missing_snapshot_stream_keeps_exact_lengths(runner, stream):
    runner.num_computed_tokens_stream = stream
    runner.model_state.initialize_spec_cpu_length_sync(runner)
    assert runner.model_state.can_skip_spec_cpu_length_sync() is False


def test_failed_snapshot_drain_keeps_policy_disabled(runner):
    runner.num_computed_tokens_stream.synchronize.side_effect = RuntimeError("D2H failed")
    with pytest.raises(RuntimeError, match="D2H failed"):
        runner.model_state.initialize_spec_cpu_length_sync(runner)
    assert runner.model_state.can_skip_spec_cpu_length_sync() is False
