# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 oracle and changed-input graph checks for streaming ranks."""

import pytest
import torch


def receive_buffer(ranks, tokens, heads, dim, has_local, strided, seed=20261031):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(ranks, tokens, heads, dim, generator=generator).bfloat16()
    logs = torch.randn(ranks, tokens, heads, 1, generator=generator) * 12
    logs[:, 0, 0] = -torch.inf
    values[:, 0, 0] = torch.nan
    logs[0, -1, -1] = torch.nan
    values[0, -1, -1] = torch.nan
    logs[-1, -1, 1] = 1e30
    # Construct the four-slot wire representation directly from IEEE FP32 bits.
    # This oracle does not call the production pack/encode implementation.
    finite = torch.isfinite(logs)
    bits = torch.where(finite, logs, 0).contiguous().view(torch.int32).to(torch.int64)
    exponent = (bits >> 23) & 255
    mantissa = (bits & 0x7FFFFF) | torch.where(exponent > 0, 1 << 23, 0)
    code = torch.where(logs < 0, -(exponent + 1), exponent + 1)
    code = torch.where(logs == 0, 128, code)
    code = torch.where(finite, code, 0)
    fields = torch.cat((code, mantissa >> 16, (mantissa >> 8) & 255, mantissa & 255), -1).bfloat16()
    recv = torch.cat((values, fields), -1).permute(0, 2, 1, 3).contiguous().npu()
    local = local_lse = None
    if has_local:
        local_cpu = torch.randn(tokens, heads, dim, generator=generator).bfloat16()
        local_logs = torch.randn(tokens, heads, 1, generator=generator) * 12
        local_logs[0, 0] = -torch.inf
        local_cpu[0, 0] = torch.nan
        values = torch.cat((values, local_cpu[None]))
        logs = torch.cat((logs, local_logs[None]))
        if strided == "row":
            local = torch.empty(tokens * 2, heads, dim, dtype=torch.bfloat16, device="npu")[::2]
            local_lse = torch.empty(tokens * 2, heads, 1, device="npu")[::2]
            local.copy_(local_cpu)
            local_lse.copy_(local_logs)
        elif strided:
            local = torch.empty(tokens, heads, dim * 2, dtype=torch.bfloat16, device="npu")[..., ::2]
            local_lse = torch.empty(tokens, heads, 2, device="npu")[..., ::2]
            local.copy_(local_cpu)
            local_lse.copy_(local_logs)
        else:
            local, local_lse = local_cpu.npu(), local_logs.npu()
    valid = torch.isfinite(logs)
    maximum = torch.where(valid, logs.double(), -torch.inf).amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.where(valid, (logs.double() - maximum).exp(), 0)
    total = weights.sum(0)
    expected = (torch.where(valid, values.double(), 0) * weights).sum(0) / torch.where(total > 0, total, 1)
    return recv, local, local_lse, expected


@pytest.mark.parametrize(
    "ranks,tokens,heads,dim",
    [
        (1, 48, 6, 512),
        (2, 48, 6, 512),
        (4, 64, 12, 512),
        (8, 48, 12, 512),
        (9, 49, 6, 512),
        (16, 48, 6, 512),
        (17, 49, 6, 512),
        (16, 1, 6, 512),
        (16, 33, 5, 256),
        (16, 128, 12, 512),
        (16, 37, 7, 32),
    ],
)
@pytest.mark.parametrize("has_local,strided", [(False, False), (True, False), (True, True), (True, "row")])
@torch.inference_mode()
def test_streaming_combine_oracle_and_replay(ranks, tokens, heads, dim, has_local, strided):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU hardware is required")
    torch.npu.set_device(0)
    from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine
    from vllm_ascend.ops.triton.dcp.dcp_a2a_batched import _fused_dcp_lse_combine_batched_kernel
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

    recv, local, local_lse, expected = receive_buffer(ranks, tokens, heads, dim, has_local, strided)

    def run():
        return fused_dcp_lse_combine(recv, dim, 1, local_output=local, local_lse=local_lse)

    actual = run()
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.016, atol=0.001)
    shadow_recv = recv.float()
    shadow_local = local.float().contiguous() if has_local else shadow_recv
    shadow_lse = local_lse.contiguous() if has_local else shadow_recv
    shadow = torch.empty((tokens, heads, dim), dtype=torch.float32, device="npu")
    _fused_dcp_lse_combine_batched_kernel[(min(get_vectorcore_num(), (tokens * heads + 7) // 8),)](
        shadow_recv,
        shadow,
        shadow_local,
        shadow_lse,
        *(shadow_local.stride() if has_local else (0, 0, 0)),
        *(shadow_lse.stride()[:2] if has_local else (0, 0)),
        *shadow_recv.stride(),
        *shadow.stride(),
        dim,
        heads,
        tokens * heads,
        DCP_SIZE=ranks,
        BLOCK_RANKS=1 << (ranks - 1).bit_length(),
        BLOCK_D=1 << (dim - 1).bit_length(),
        BLOCK_ROWS=8,
        HAS_LOCAL=has_local,
        COALESCED_LSE=True,
        multibuffer=False,
    )
    torch.testing.assert_close(shadow.cpu().double(), expected, rtol=1e-4, atol=1e-5)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            captured = run()
        # Change both data and LSE fields at the same graph addresses.
        other_recv, other_local, other_lse, other_expected = receive_buffer(
            ranks, tokens, heads, dim, has_local, strided, seed=20261101
        )
        other_recv[..., :dim].mul_(0.5)
        if has_local:
            other_local.mul_(0.5)
        recv.copy_(other_recv.flip(0).contiguous())
        if has_local:
            local.copy_(other_local)
            local_lse.copy_(other_lse)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured.cpu().double(), other_expected * 0.5, rtol=0.016, atol=0.001)
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)
    finally:
        graph.reset()
