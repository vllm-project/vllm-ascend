# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops.gdn_recover as recovery
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            enforce_eager=True,
            enable_sleep_mode=False,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_dtype="auto",
        ),
        parallel_config=SimpleNamespace(
            **{
                k: 1
                for k in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            }
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3, enforce_eager=True),
        use_v2_model_runner=False,
        kv_transfer_config=None,
    )


@pytest.fixture
def standard_profile(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.STANDARD),
    )


def test_supported_configuration(standard_profile):
    recovery.validate_gdn_recover_config(_config())


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float16),
        ("model_config", "enable_sleep_mode", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "align"),
        ("cache_config", "mamba_ssm_cache_dtype", "bfloat16"),
        ("cache_config", "mamba_cache_dtype", "float32"),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("speculative_config", "method", "eagle3"),
        ("speculative_config", "num_speculative_tokens", 0),
        ("speculative_config", "num_speculative_tokens", 6),
        ("speculative_config", "enforce_eager", False),
        (None, "use_v2_model_runner", True),
        (None, "kv_transfer_config", object()),
        (None, "speculative_config", None),
    ],
)
def test_reject_unsupported_configuration(standard_profile, owner, field, value):
    config = _config()
    setattr(getattr(config, owner) if owner else config, field, value)
    with pytest.raises(ValueError):
        recovery.validate_gdn_recover_config(config)


def test_reject_other_models_and_hardware(standard_profile, monkeypatch):
    config = _config()
    config.model_config.hf_text_config.model_type = "qwen3_next"
    with pytest.raises(ValueError, match="dense Qwen"):
        recovery.validate_gdn_recover_config(config)
    monkeypatch.setattr(
        recovery,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED),
    )
    with pytest.raises(ValueError, match="A2/A3"):
        recovery.validate_gdn_recover_config(_config())


def _metadata():
    # Spec rows are a subset of the sampler's rows, in a different order.
    return SimpleNamespace(
        recover_state_indices=torch.tensor([5, 2], dtype=torch.int32),
        recover_query_offsets=torch.tensor([0, 2, 6], dtype=torch.int32),
        recover_request_rows=torch.tensor([2, 0], dtype=torch.int64),
    )


@pytest.fixture
def commits(monkeypatch):
    from vllm_ascend.ops.triton.mamba.gdn_recover_batch import GDNRecoverBatchLayout

    calls = []
    monkeypatch.setattr(GDNRecoverBatchLayout, "try_create", lambda _: None)
    monkeypatch.setattr(recovery, "gdn_recover_commit", lambda *args: calls.append(("ssm", args[-1].clone())))
    monkeypatch.setattr(recovery, "gdn_commit_conv_history", lambda *args: calls.append(("conv", args[-1].clone())))
    event = Mock()
    event.record.side_effect = lambda: calls.append(("event", None))
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: Mock())
    return calls, event


def test_commit_uses_forward_row_mapping_and_only_once(commits):
    calls, event = commits
    context = recovery.GDNRecoverContext()
    context.begin()
    metadata = _metadata()
    cache = (torch.empty(1),) * 4
    context.enqueue("target.layer.0", cache, metadata)
    context.enqueue("target.layer.2", cache, metadata)
    # Sampler row 0 accepts 1 draft => two processed inputs; row 2 accepts
    # zero drafts => one mandatory input. Output includes a sampled bonus.
    sampled = torch.tensor([[11, 12, -1, -1], [21, 22, 23, 24], [31, -1, -1, -1]])
    context.commit(sampled)
    assert [c[0] for c in calls] == ["ssm", "conv", "ssm", "conv", "event"]
    for _, counts in calls[:-1]:
        assert counts.tolist() == [1, 2]
    assert not context.pending and not context.active
    context.commit(sampled)
    assert len(calls) == 5, "the same records must never be replayed twice"
    event.record.assert_called_once()
    context.reset()
    event.synchronize.assert_called_once()
    assert context._layout is None


def test_pending_verify_blocks_reuse_and_duplicate_layer(commits):
    context = recovery.GDNRecoverContext()
    context.begin()
    context.enqueue("target.layer.0", (torch.empty(1),) * 4, _metadata())
    with pytest.raises(RuntimeError, match="twice"):
        context.enqueue("target.layer.0", (torch.empty(1),) * 4, _metadata())
    with pytest.raises(RuntimeError, match="not committed"):
        context.begin()
    with pytest.raises(RuntimeError, match="outstanding"):
        context.reset()


def test_profile_forward_does_not_register_commit():
    context = recovery.GDNRecoverContext()
    context.enqueue("profile.layer.0", (torch.empty(1),) * 4, _metadata())
    assert not context.pending
