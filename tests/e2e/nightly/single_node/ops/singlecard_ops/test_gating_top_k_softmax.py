import pytest
import torch
import torch_npu


@pytest.mark.parametrize(
    'B',
    [1, 16, 64, 128, 32768],
)
@pytest.mark.parametrize(
    'D',
    [8, 16, 32, 64, 128],
)
@pytest.mark.parametrize(
    'top_k',
    [1, 2, 4, 8],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-3, 1e-3),
    ],
)
def test_quant_fpx_linear(B: int, D: int, top_k: int, dtype, atol, rtol):
    x = torch.rand((B, D), dtype=dtype).to("npu")
    # finished = torch.randint(1, size=(B,), dtype=torch.bool).to("npu")
    finished = None
    y, expert_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x,
                                                                    finished,
                                                                    k=top_k)

    topk_weights = x.softmax(dim=-1)
    topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    torch.allclose(y, topk_weights, atol=atol, rtol=rtol)
    torch.allclose(expert_idx, topk_ids, atol=atol, rtol=rtol)
