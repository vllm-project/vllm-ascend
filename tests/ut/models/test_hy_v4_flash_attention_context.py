# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.models.hy_v4_flash import decode_fusions
from vllm_ascend.models.hy_v4_flash.kv_compaction import compact_kv_rows


def test_flash_backend_supports_updatable_graph():
    from vllm_ascend.models.hy_v4_flash.attention import FlashAttentionBackend
    from vllm_ascend.utils import use_updatable_graph

    assert use_updatable_graph(FlashAttentionBackend)


def test_compaction_retains_order_and_zeros_all_invalid_rows():
    selected = torch.tensor([[-1, 8, -1, 2, 3], [-1, -1, -1, -1, -1]])
    kv = torch.arange(20).view(2, 5, 1, 2).float()
    nope, rope = compact_kv_rows(selected, kv, kv + 100)
    torch.testing.assert_close(nope[0, :3], kv[0, [1, 3, 4]])
    torch.testing.assert_close(rope[0, :3], kv[0, [1, 3, 4]] + 100)
    assert torch.count_nonzero(nope[0, 3:]) == 0
    assert torch.count_nonzero(nope[1]) == 0
    assert torch.count_nonzero(rope[1]) == 0


@pytest.mark.parametrize("tuple_kv", [False, True])
def test_fp32_attention_preserves_kv_and_sink(tuple_kv):
    torch.manual_seed(1024)
    query = torch.randn(3, 2, 576, dtype=torch.bfloat16)
    kv = torch.randn(3, 4, 576, dtype=torch.float32)
    selected = torch.tensor([[-1, -1, -1, 0], [-1, 0, 1, -1], [-1, -1, -1, -1]])
    sink = torch.tensor([0.3, -0.5])
    scale = 0.04
    rows = (kv[..., :512, None].transpose(-1, -2), kv[..., 512:, None].transpose(-1, -2), None) if tuple_kv else kv
    result = decode_fusions.native_mla_context(
        query,
        selected,
        rows,
        sink,
        scale,
    )
    logits = (query.float() @ kv.transpose(-1, -2) * scale).masked_fill(selected[:, None] < 0, -torch.inf)
    denominator = torch.cat((logits, sink[None, :, None].expand(3, -1, -1)), -1).softmax(-1)
    expected = (denominator[..., :-1] @ kv[..., :512]).to(query.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result[2]) == 0
