import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd


def _chunk_kkt_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    batch, num_kv_heads, max_seqlen, _ = k.shape
    num_v_heads = beta.shape[-1]
    output = torch.zeros(
        batch,
        num_v_heads,
        max_seqlen,
        chunk_size,
        device=k.device,
        dtype=torch.float32,
    )
    sequences = (
        [(batch_idx, 0, max_seqlen) for batch_idx in range(batch)]
        if cu_seqlens is None
        else [(0, int(start), int(end)) for start, end in zip(cu_seqlens[:-1].cpu(), cu_seqlens[1:].cpu())]
    )

    for batch_idx, start, end in sequences:
        for head_idx in range(num_v_heads):
            key_head_idx = head_idx // (num_v_heads // num_kv_heads)
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                key_chunk = k[batch_idx, key_head_idx, chunk_start:chunk_end].float()
                score = key_chunk @ key_chunk.transpose(0, 1)
                gate = g_cumsum[batch_idx, chunk_start:chunk_end, head_idx]
                gate_diff = gate[:, None] - gate[None, :]
                score *= torch.exp(torch.where(gate_diff <= 0, gate_diff, float("-inf")))
                score *= beta[batch_idx, chunk_start:chunk_end, head_idx].float()[:, None]
                output[batch_idx, head_idx, chunk_start:chunk_end, : chunk_end - chunk_start] = torch.tril(
                    score,
                    diagonal=-1,
                )
    return output


@pytest.mark.parametrize("varlen", [False, True])
def test_chunk_scaled_dot_kkt_bhtd_k_bth_gates(varlen: bool):
    torch.manual_seed(0)
    batch, max_seqlen, num_kv_heads, num_v_heads, head_dim = 2, 71, 2, 4, 128
    if varlen:
        batch = 1
        cu_seqlens = torch.tensor([0, 1, 17, max_seqlen], dtype=torch.int64).npu()
    else:
        cu_seqlens = None

    k = torch.randn(batch, num_kv_heads, max_seqlen, head_dim, dtype=torch.bfloat16).npu()
    beta = torch.rand(batch, max_seqlen, num_v_heads, dtype=torch.bfloat16).npu()
    g_cumsum = -torch.rand(batch, max_seqlen, num_v_heads, dtype=torch.float32).npu()

    actual = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )
    expected = _chunk_kkt_reference(k, beta, g_cumsum, 64, cu_seqlens)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
