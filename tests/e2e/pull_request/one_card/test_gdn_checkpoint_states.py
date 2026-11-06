# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h


def test_fp32_internal_state_matches_truncated_recurrence():
    torch.manual_seed(17)
    device = "npu"
    k = torch.randn(1, 192, 2, 128, device=device, dtype=torch.bfloat16) * 0.01
    w = torch.randn_like(k) * 0.01
    u = torch.randn_like(k) * 0.01
    g = torch.zeros(1, 2, 192, device=device, dtype=torch.float32)
    initial = torch.randn(1, 2, 128, 128, device=device, dtype=torch.float32) * 0.01
    states = torch.empty_like(initial)
    chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial,
        True,
        checkpoint_output=states,
        checkpoint_offsets=torch.tensor([128], dtype=torch.int32, device=device),
    )
    _, _, truncated = chunk_gated_delta_rule_fwd_h(
        k[:, :128].contiguous(),
        w[:, :128].contiguous(),
        u[:, :128].contiguous(),
        g[:, :, :128].contiguous(),
        initial,
        True,
    )
    assert states.dtype == torch.float32
    torch.testing.assert_close(states.transpose(-1, -2), truncated, rtol=1e-5, atol=1e-6)


def test_shared_exporter_preserves_ascend_conv_layout_and_mtp_slots():
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointExporter,
        MambaPrefillCheckpointMetadata,
    )

    device = "npu"
    raw = torch.arange(352 * 16, device=device, dtype=torch.float32).view(352, 16)
    conv = torch.full((5, 6, 16), -42.0, device=device)
    recurrent = torch.full((5, 2, 128, 128), -42.0, device=device)
    states = torch.randn(3, 2, 128, 128, device=device)
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32, device=device),
        state_indices=torch.tensor([2, 0, 4], dtype=torch.int64, device=device),
    )
    MambaPrefillCheckpointExporter(state_len=3).export(
        checkpoint,
        raw_qkv=raw,
        conv_state=conv.transpose(-1, -2),
        recurrent_checkpoint=states,
        recurrent_state=recurrent,
        cu_seqlens=torch.tensor([0, 96, 160, 352], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(conv[2, :3], raw[61:64], rtol=0, atol=0)
    torch.testing.assert_close(conv[4, :3], raw[285:288], rtol=0, atol=0)
    torch.testing.assert_close(recurrent[2], states[0], rtol=0, atol=0)
    torch.testing.assert_close(recurrent[4], states[2], rtol=0, atol=0)
    assert (conv[:, 3:] == -42).all()
    assert (conv[0] == -42).all()
    assert (recurrent[0] == -42).all()


@pytest.mark.parametrize("layout", ["head_first", "token_slice"])
def test_checkpoint_prefill_matches_dense_inputs(layout):
    from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
    from vllm_ascend.ops.triton.fla import chunk

    torch.manual_seed(23)
    lengths = [129, 65, 193]
    total, heads, dim = sum(lengths), 4, 128

    def strided_input():
        if layout == "head_first":
            return torch.randn(heads, total, dim, device="npu", dtype=torch.bfloat16).transpose(0, 1).unsqueeze(0)
        return torch.randn(1, total * 2, heads, dim, device="npu", dtype=torch.bfloat16)[:, ::2]

    q, k, v = (strided_input() for _ in range(3))
    assert not v.is_contiguous()
    g = -torch.rand(1, total * 2, heads, device="npu")[:, ::2] * 0.01
    beta = torch.rand(1, total * 2, heads, device="npu", dtype=torch.bfloat16)[:, ::2]
    initial = torch.randn(3, heads, dim, dim, device="npu", dtype=torch.float32) * 0.01
    cu_cpu = torch.tensor([0, 129, 194, total], dtype=torch.int32)
    cu = cu_cpu.to("npu")
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_num_value_heads=heads)),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )
    metadata = _build_non_spec_chunked_prefill_metadata(builder, cu_cpu, torch.device("npu"))
    offsets = torch.tensor([128, 0, 192], device="npu", dtype=torch.int32)
    snapshots = torch.full_like(initial, float("nan"))
    dense_snapshots = torch.full_like(initial, float("nan"))
    with (
        patch.object(chunk, "get_forward_context", return_value=SimpleNamespace(attn_metadata=None)),
        patch.object(chunk, "get_pcp_group", return_value=SimpleNamespace(world_size=1)),
    ):
        actual = chunk.chunk_gated_delta_rule_checkpoint(q, k, v, g, beta, initial, cu, metadata, snapshots, offsets)
        expected = chunk.chunk_gated_delta_rule_checkpoint(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            initial,
            cu,
            metadata,
            dense_snapshots,
            offsets,
        )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(snapshots[[0, 2]], dense_snapshots[[0, 2]], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(snapshots[[0, 2]]).all()
    assert torch.isnan(snapshots[1]).all()
