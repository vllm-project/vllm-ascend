# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The speculative GDN update must pick the kernel that accepts the state dtype.

The native packed-index AscendC kernel rejects an fp32 mamba state with
``aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`` error 161002, and every
Qwen3.5/3.6 checkpoint carries fp32 state regardless of the weight dtype.
"""

import pytest
import torch

from vllm_ascend.ops.gdn import _MAX_RECURRENT_GDN_TOKENS, _use_upstream_recurrent


class _SpecMeta:
    def __init__(self, use_upstream_recurrent: bool):
        self.use_upstream_recurrent = use_upstream_recurrent


class _Meta:
    def __init__(self, use_upstream_recurrent: bool):
        self.spec_decode_metadata = _SpecMeta(use_upstream_recurrent)


def test_fp32_state_forces_strided_kernel():
    """fp32 state is the Qwen3.5/3.6 default, so this must always escalate."""
    meta = _Meta(use_upstream_recurrent=False)
    indices = torch.zeros((2, 2), dtype=torch.int32)
    assert _use_upstream_recurrent(meta, indices, torch.zeros(1, dtype=torch.float32)) is True


def test_fp16_state_keeps_native_kernel_for_narrow_windows():
    meta = _Meta(use_upstream_recurrent=False)
    indices = torch.zeros((2, _MAX_RECURRENT_GDN_TOKENS), dtype=torch.int32)
    assert _use_upstream_recurrent(meta, indices, torch.zeros(1, dtype=torch.float16)) is False


def test_wide_speculative_window_forces_strided_kernel():
    meta = _Meta(use_upstream_recurrent=False)
    indices = torch.zeros((2, _MAX_RECURRENT_GDN_TOKENS + 1), dtype=torch.int32)
    assert _use_upstream_recurrent(meta, indices, torch.zeros(1, dtype=torch.float16)) is True


@pytest.mark.parametrize("flag", [True])
def test_metadata_request_is_honoured(flag: bool):
    meta = _Meta(use_upstream_recurrent=flag)
    indices = torch.zeros((2, 2), dtype=torch.int32)
    assert _use_upstream_recurrent(meta, indices, torch.zeros(1, dtype=torch.float16)) is True


def test_one_dimensional_indices_do_not_trigger_window_rule():
    """The window rule only applies to the 2-D packed index layout."""
    meta = _Meta(use_upstream_recurrent=False)
    indices = torch.zeros(_MAX_RECURRENT_GDN_TOKENS + 4, dtype=torch.int32)
    assert _use_upstream_recurrent(meta, indices, torch.zeros(1, dtype=torch.float16)) is False


def _dispatch_args(state_dtype):
    return {
        "query": torch.zeros(1, 2, 4),
        "key": torch.zeros(1, 2, 4),
        "value": torch.zeros(1, 2, 4),
        "g": torch.zeros(1, 2),
        "beta": torch.zeros(1, 2),
        "ssm_state": torch.zeros(2, dtype=state_dtype),
        "actual_seq_lengths": torch.tensor([2], dtype=torch.int32),
        "ssm_state_indices": torch.zeros(2, dtype=torch.int32),
    }


def _patch_kernels(monkeypatch, gdn, seen):
    def fused(**kwargs):
        # The real kernel returns o.squeeze(0), i.e. a [B, T, H, V] tensor.
        seen["fused"] = kwargs
        return kwargs["q"], None

    def native(*args, **kwargs):
        seen["native"] = kwargs
        return args[0]

    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule", fused)
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", native)


def test_non_spec_dispatch_uses_fused_for_fp32_state(monkeypatch):
    """Non-speculative decode, including the dummy decode captured for ACL graphs,
    must also avoid the native kernel when the state is fp32."""
    import vllm_ascend.ops.gdn as gdn

    seen = {}
    _patch_kernels(monkeypatch, gdn, seen)
    args = _dispatch_args(torch.float32)
    out = gdn._recurrent_gated_delta_rule(args["query"], args["key"], args["value"],
                                          args["ssm_state"], g=args["g"], beta=args["beta"],
                                          scale=0.5,
                                          actual_seq_lengths=args["actual_seq_lengths"],
                                          ssm_state_indices=args["ssm_state_indices"])
    assert "fused" in seen and "native" not in seen
    assert out.shape == (1, 2, 4)


def test_non_spec_dispatch_keeps_native_for_fp16_state(monkeypatch):
    import vllm_ascend.ops.gdn as gdn

    seen = {}
    _patch_kernels(monkeypatch, gdn, seen)
    args = _dispatch_args(torch.float16)
    gdn._recurrent_gated_delta_rule(args["query"], args["key"], args["value"],
                                    args["ssm_state"], g=args["g"], beta=args["beta"],
                                    scale=0.5,
                                    actual_seq_lengths=args["actual_seq_lengths"],
                                    ssm_state_indices=args["ssm_state_indices"])
    assert "native" in seen and "fused" not in seen
