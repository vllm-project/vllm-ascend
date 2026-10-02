import torch

from vllm_ascend.ops.triton.mamba.selective_scan import selective_scan_fn_npu


def _reference_batch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
) -> torch.Tensor:
    batch, dim, seqlen = u.shape
    dstate = A.shape[-1]
    h = torch.zeros(batch, dim, dstate, dtype=torch.float32)
    out = torch.zeros(batch, dim, seqlen, dtype=torch.float32)
    for t in range(seqlen):
        dt = delta[:, :, t]
        ut = u[:, :, t]
        bt = B[:, :, t]
        ct = C[:, :, t]
        dA = torch.exp(A.view(1, -1, dstate) * dt.unsqueeze(-1))
        h = dA * h + dt.unsqueeze(-1) * bt.unsqueeze(1) * ut.unsqueeze(-1)
        y_t = (ct.unsqueeze(1) * h).sum(-1)
        if D is not None:
            y_t = y_t + D.unsqueeze(0) * ut
        out[:, :, t] = y_t
    return out


def test_selective_scan_batch_matches_reference():
    torch.manual_seed(0)
    batch, dim, dstate, seqlen = 2, 8, 4, 5
    u = torch.randn(batch, dim, seqlen)
    delta = torch.randn(batch, dim, seqlen).abs() * 0.1
    A = -torch.rand(dim, dstate)
    B = torch.randn(batch, dstate, seqlen)
    C = torch.randn(batch, dstate, seqlen)
    D = torch.randn(dim)

    got = selective_scan_fn_npu(u, delta, A, B, C, D, None, None, False, None)
    ref = _reference_batch(u, delta, A, B, C, D)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_selective_scan_varlen_no_item_sync_shape():
    torch.manual_seed(1)
    dim, dstate = 8, 4
    lengths = [3, 0, 2]
    total = sum(lengths)
    u = torch.randn(total, dim)
    delta = torch.randn(total, dim).abs() * 0.1
    A = -torch.rand(dim, dstate)
    B = torch.randn(total, dstate)
    C = torch.randn(total, dstate)
    query_start_loc = torch.tensor([0, 3, 3, 5], dtype=torch.int32)

    out = selective_scan_fn_npu(
        u, delta, A, B, C, None, None, None, False, query_start_loc
    )
    assert out.shape == u.shape
    assert torch.isfinite(out).all()
