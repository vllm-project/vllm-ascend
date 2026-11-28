import pytest
import torch

from vllm_ascend.attention.kimi_c8_dcp import build_c8_dcp_query_plan
from vllm_ascend.attention.kimi_c8_graph import build_c8_graph_query_plan


def combine_dcp_lse(partial_output: torch.Tensor, partial_lse: torch.Tensor, dim: int = 0, *, lse_layout: str = "NT"):
    """Merge [...,T,N,D] outputs; QFMLA LSE is [...,N,T] by default.

    Reference producers using [...,T,N] must request lse_layout="TN".
    Explicit layout remains unambiguous when token and head counts match.
    """
    if lse_layout not in ("NT", "TN"):
        raise ValueError("lse_layout must be NT (QFMLA) or TN (reference)")
    if partial_output.ndim != partial_lse.ndim + 1:
        raise ValueError("partial output must have one value dimension beyond LSE")
    if partial_output.shape[:dim] != partial_lse.shape[:dim]:
        raise ValueError("partial output and LSE rank dimensions do not match")
    lse = partial_lse.float()
    output = partial_output.float()
    if lse_layout == "NT":
        lse = lse.transpose(-1, -2)
    if output.shape[:-1] != lse.shape:
        raise ValueError("partial output and LSE shape mismatch")
    # QFMLA uses +inf for empty rows; existing fused DCP combine ignores every
    # nonfinite LSE. Mask before max and multiplication, including invalid NaN
    # outputs, so one empty rank cannot discard a valid rank's contribution.
    valid = torch.isfinite(lse)
    lse = lse.masked_fill(~valid, -torch.inf)
    output = output.masked_fill(~valid.unsqueeze(-1), 0)
    max_lse = lse.max(dim=dim, keepdim=True).values
    safe_max = torch.where(torch.isfinite(max_lse), max_lse, 0)
    weights = torch.exp(lse - safe_max)
    merged = (output * weights.unsqueeze(-1)).sum(dim=dim)
    denominator = weights.sum(dim=dim)
    merged = merged / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    merged_lse = torch.where(
        denominator > 0,
        safe_max.squeeze(dim) + denominator.log(),
        torch.full_like(denominator, -torch.inf),
    )
    return merged.to(partial_output.dtype), merged_lse


@pytest.mark.parametrize(
    "interleave,length,queries", [(128, 129, 8), (128, 1025, 16), (1664, 1665, 8), (1664, 13313, 16)]
)
def test_native_partition_and_lse_match_global_causal_attention(interleave, length, queries):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(queries, 2, 4, generator=generator)
    k, v = [torch.randn(length, 4, generator=generator) for _ in range(2)]
    owners = (torch.arange(length) // interleave) % 8
    outputs, lses, totals = [], [], torch.zeros(queries, dtype=torch.int32)
    for rank in range(8):
        plan = build_c8_dcp_query_plan(
            [length],
            [0, queries],
            torch.tensor([[7]], dtype=torch.int32),
            world_size=8,
            rank=rank,
            interleave_size=interleave,
        )
        out, lse = torch.full((queries, 2, 4), torch.nan), torch.full((queries, 2), torch.inf)
        local_k, local_v = k[owners == rank], v[owners == rank]
        assert plan.cache_seqlens.tolist() == [int((owners == rank).sum())]
        assert plan.block_table.shape[0] == plan.global_seqlens.numel() == 1
        assert plan.cu_seqlens_q.tolist() == [0, queries]
        assert plan.seqused_q.tolist() == [queries]
        counts = [
            int(((torch.arange(length) < length - queries + token + 1) & (owners == rank)).sum())
            for token in range(queries)
        ]
        for token, count in enumerate(counts):
            if count:
                scores = q[token] @ local_k[:count].T
                out[token], lse[token] = scores.softmax(-1) @ local_v[:count], scores.logsumexp(-1)
        totals += torch.tensor(counts, dtype=torch.int32)
        outputs.append(out)
        lses.append(lse.T)  # Real quant FA returns heads before tokens.
    assert totals.tolist() == list(range(length - queries + 1, length + 1))
    actual, _ = combine_dcp_lse(torch.stack(outputs), torch.stack(lses))
    expected = torch.stack(
        [(q[t] @ k[: length - queries + t + 1].T).softmax(-1) @ v[: length - queries + t + 1] for t in range(queries)]
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("heads", [2, 3])
@pytest.mark.parametrize("empty_lse", [torch.inf, -torch.inf, torch.nan])
def test_nt_tn_layout_and_nonfinite_empty_ranks(heads, empty_lse):
    out = torch.arange(4 * heads * 2, dtype=torch.float32).reshape(2, 2, heads, 2)
    weights = torch.arange(1, 4 * heads + 1, dtype=torch.float32).reshape(2, 2, heads)
    expected = (out * weights[..., None]).sum(0) / weights.sum(0)[..., None]
    for layout, lse in (("TN", weights.log()), ("NT", weights.log().transpose(-1, -2))):
        actual, _ = combine_dcp_lse(out, lse, lse_layout=layout)
        torch.testing.assert_close(actual, expected)
    out[1] = torch.nan
    lse = torch.zeros(2, 2, heads)
    lse[1] = empty_lse
    actual, _ = combine_dcp_lse(out, lse, lse_layout="TN")
    torch.testing.assert_close(actual, out[0])
    empty, empty_lse = combine_dcp_lse(out[1:], lse[1:], lse_layout="TN")
    assert empty.eq(0).all() and empty_lse.isneginf().all()


def test_eager_and_graph_padding_never_exposes_stale_rows():
    table = torch.tensor([[5], [7], [9]], dtype=torch.int32)
    eager = build_c8_dcp_query_plan([0, 1665, 0], [0, 0, 2, 4], table, world_size=8, rank=7, interleave_size=1664)
    graph = build_c8_graph_query_plan(
        torch.tensor([0, 0, 2, 4]),
        torch.tensor([0, 1665, 0]),
        table,
        world_size=8,
        rank=7,
        interleave_size=1664,
    )
    for plan in (eager, graph):
        assert plan.cache_seqlens.tolist() == [0] * 3
        assert plan.block_table.tolist() == [[5], [7], [9]]
        assert plan.seqused_q.tolist() == [0, 2, 0]
    for offsets in ([1, 2], [0]):
        with pytest.raises(ValueError):
            build_c8_dcp_query_plan([8], offsets, table[:1], world_size=8, rank=0, interleave_size=1664)
