# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention, engram, ihc


@pytest.mark.parametrize("shape", [(0, 16, 96), (1, 16, 96), (6, 16, 96), (513, 96)])
@pytest.mark.parametrize("enabled", [False, True])
def test_engram_cpu_fallback(shape, enabled, monkeypatch):
    op = Mock(side_effect=AssertionError("CPU must not invoke CANN"))
    monkeypatch.setattr(engram.torch_npu, "npu_rms_norm", op)
    x = torch.randn(shape)
    eps = torch.finfo(torch.float32).eps
    actual = engram._engram_inverse_rms(x, torch.ones(shape[-1]), eps, enabled)
    torch.testing.assert_close(actual, (x.square().mean(-1, keepdim=True) + eps).rsqrt(), rtol=0, atol=0)
    op.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, attention.MAX_FUSED_C8_READ_ROWS, attention.MAX_FUSED_C8_READ_ROWS + 1])
def test_bulk_c8_boundary_and_padding(enabled, rows, monkeypatch):
    spec = object.__new__(attention.AscendHYV4FlashAttention)
    spec.use_pd_decode = True
    spec.bulk_c8_decode_reads = enabled
    spec.use_fp8_cache = True
    spec.use_cann_prefill = False
    spec.kv_storage_width = 4
    spec.ops = attention.triton_ops
    monkeypatch.setattr(attention, "HAS_TRITON", True)
    cache = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(1, 4))
    positions = torch.arange(rows)
    positions[0] = -1
    fused = Mock(return_value="fused")
    monkeypatch.setattr(attention.triton_ops, "gather_c8_rows", fused)
    gathered = []

    def gather(cache, table, block_size, logical, columns):
        gathered.append(logical)
        return torch.ones(rows, 4)

    spec._gather = gather
    monkeypatch.setattr(attention, "dequantize_c8_rows", lambda x: x)
    result = spec._gather_attention_rows(cache, None, 16, positions, mask_padding=True)
    if enabled and rows > attention.MAX_FUSED_C8_READ_ROWS:
        fused.assert_not_called()
        assert gathered[0][0] == 0
        assert torch.equal(result[0], torch.zeros(4))
        assert torch.equal(result[1:], torch.ones(rows - 1, 4))
    else:
        assert result == "fused"
        fused.assert_called_once()
        assert not gathered


def test_ihc_tail_cpu_preserves_fallback():
    pre = object.__new__(ihc.FlashHCPreLayer)
    pre.fuse_decode_tail = True
    assert not pre._can_fuse_decode_tail(torch.empty(6, 4, 3072, dtype=torch.bfloat16))


def test_disabled_ihc_needs_no_pre_layer():
    wrapper = object.__new__(ihc.FlashHCLayer)
    wrapper.enable_ihc = False
    wrapper.configure_decode_post()


def test_ihc_decode_configuration_is_automatic(monkeypatch):
    wrapper = object.__new__(ihc.FlashHCLayer)
    wrapper.enable_ihc = True
    wrapper.hc_pre = SimpleNamespace()
    monkeypatch.setattr(ihc, "is_950", lambda: False)
    wrapper.configure_decode_post()
    assert wrapper.hc_pre.fuse_decode_tail is True


def test_native_rms_unsupported_device_uses_reference():
    from vllm_ascend.models.hy_v4_flash.decode_fusions import native_rms_norm

    value = torch.randn(6, 128, dtype=torch.bfloat16)
    norm = torch.nn.RMSNorm(128, eps=1e-6).to(torch.bfloat16)
    torch.testing.assert_close(native_rms_norm(value, norm), norm(value), rtol=0, atol=0)
