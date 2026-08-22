import torch

from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd


@torch.inference_mode()
def test_recompute_w_u_fwd_equal_length():
    batch_size, seq_len, key_heads, value_heads = 2, 96, 2, 4
    key_dim = value_dim = chunk_size = 64
    device = "npu"

    k = torch.randn(
        batch_size,
        seq_len,
        key_heads,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        value_heads,
        value_dim,
        dtype=torch.float32,
        device=device,
    )
    batch_values = torch.arange(1, batch_size + 1, dtype=torch.float32, device=device).view(batch_size, 1, 1)
    beta = batch_values.expand(batch_size, seq_len, value_heads).contiguous()
    g_cumsum = batch_values.log().expand_as(beta).contiguous()
    A = torch.zeros(
        batch_size,
        seq_len,
        value_heads,
        chunk_size,
        dtype=torch.float32,
        device=device,
    )
    token_indices = torch.arange(seq_len, device=device)
    A[:, token_indices, :, token_indices % chunk_size] = 1

    w, u = recompute_w_u_fwd(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens=None,
    )

    expected_w = (
        k.repeat_interleave(value_heads // key_heads, dim=2) * beta.unsqueeze(-1) * g_cumsum.exp().unsqueeze(-1)
    )
    expected_u = v * beta.unsqueeze(-1)
    torch.testing.assert_close(w, expected_w)
    torch.testing.assert_close(u, expected_u)
