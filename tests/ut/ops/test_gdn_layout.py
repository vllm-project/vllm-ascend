# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


@pytest.mark.parametrize("kind", ["head_first", "sequence_first", "mixed_slice"])
def test_a5_prefill_layout_preserves_values_and_output(kind):
    tokens, nk, nv, dim = 7, 2, 4, 16
    qkv = [torch.randn(1, h, tokens, dim).transpose(1, 2) for h in (nk, nk, nv)]
    if kind == "sequence_first":
        qkv = [x.contiguous() for x in qkv]
    elif kind == "mixed_slice":
        qkv = [x[:, 2:] for x in qkv]
    count = qkv[0].shape[1]
    state = torch.randn(1, nv, dim, dim)
    output = torch.randn(1, count, nv, dim)
    gate = torch.randn(1, count, nv)
    meta = SimpleNamespace(cu_seqlens_host=None, chunk_indices_chunk64_host=None, keep_meta=None)

    def fused(q, k, v, g, beta, **kwargs):
        expected_layout = "BNSD" if kind == "head_first" else "BSND"
        assert kwargs["layout"] == expected_layout
        for actual, original in zip((q, k, v), qkv):
            assert actual.is_contiguous()
            expected = original.transpose(1, 2) if kind == "head_first" else original
            torch.testing.assert_close(actual, expected)
            if kind == "head_first":
                assert actual.data_ptr() == original.data_ptr()
        torch.testing.assert_close(g, gate)
        torch.testing.assert_close(beta, gate)
        return output, state

    result, final_state = A5DeviceAdaptor.fla_gdn_prefill(*qkv, gate, gate, state, dim**-0.5, meta, fused)
    assert result is output
    assert final_state is state


@pytest.mark.parametrize("backend", ["base", "cann"])
@pytest.mark.parametrize("sliced", [False, True])
def test_head_first_l2norm_fallback_accepts_multiple_heads(backend, sliced):
    tokens, nk, nv, dim = 7, 2, 4, 128
    qkv = [torch.randn(1, heads, tokens, dim).transpose(1, 2) for heads in (nk, nk, nv)]
    if sliced:
        qkv = [x[:, :2] for x in qkv]
    count = qkv[0].shape[1]
    state = torch.randn(1, nv, dim, dim)
    gate = torch.zeros(1, count, nv)
    output = torch.randn(1, count, nv, dim)

    def norm(x):
        # Match the real upstream L2Norm's first operation. It raises on
        # head-first BSND views unless the caller packs them before norm.
        x.view(-1, x.shape[-1])
        return torch.nn.functional.normalize(x, dim=-1)

    if backend == "base":
        meta = SimpleNamespace(cu_seqlens_host=(0, count), chunk_indices_chunk64_host=((0, 0),), keep_meta=None)

        def fused(q, k, v, *args, **kwargs):
            assert kwargs["layout"] == "BNSD"
            for got, original in zip((q, k), qkv[:2]):
                expected = torch.nn.functional.normalize(original, dim=-1).transpose(1, 2)
                torch.testing.assert_close(got, expected)
            torch.testing.assert_close(v, qkv[2].transpose(1, 2))
            return output, state.transpose(-1, -2).contiguous()

        with patch("vllm_ascend.device.device_op.l2norm_fwd", side_effect=norm):
            actual, _ = BaseDeviceAdaptor.fla_gdn_prefill(*qkv, gate, gate, state, dim**-0.5, meta, fused)
    else:

        def fused(q, k, v, *args, **kwargs):
            for got, original in zip((q, k), qkv[:2]):
                torch.testing.assert_close(got, torch.nn.functional.normalize(original, dim=-1).squeeze(0))
            torch.testing.assert_close(v, qkv[2].squeeze(0))
            return output.squeeze(0), kwargs["initial_state"]

        with (
            patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=norm),
            patch("vllm_ascend.ops.gdn.torch_npu.npu_chunk_gated_delta_rule", side_effect=fused, create=True),
        ):
            actual, _ = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                *qkv, gate, gate, state, torch.tensor([0, count], dtype=torch.int32), dim**-0.5
            )
    torch.testing.assert_close(actual, output)
