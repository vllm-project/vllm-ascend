# SPDX-License-Identifier: Apache-2.0
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
