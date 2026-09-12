# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("query_count", [1, 2, 5])
def test_candidate_consumer_preserves_each_query_mask(ratio, query_count):
    """A multi-query tile must reload each row's candidates across KV tiles."""
    torch.manual_seed(71)
    device = "npu"
    heads, width, topk, block_size, kv_len = 64, 128, 1024, 128, 6144
    query = torch.randint(-90, 90, (query_count, heads, width), dtype=torch.int8, device=device)
    key = torch.randint(-90, 90, (kv_len // block_size, block_size, 1, width), dtype=torch.int8, device=device)
    weights = torch.rand(query_count, heads, dtype=torch.float16, device=device)
    query_scale = torch.full((query_count, heads), 0.01, dtype=torch.float16, device=device)
    key_scale = torch.full(key.shape[:-1], 0.01, dtype=torch.float16, device=device)
    qsl = torch.tensor([0, query_count], dtype=torch.int32, device=device)
    lengths = torch.tensor([kv_len], dtype=torch.int32, device=device)
    residual = torch.zeros_like(lengths) if ratio == 2 else None
    table = torch.arange(kv_len // block_size, dtype=torch.int32, device=device).unsqueeze(0)
    # Alternate masks: row 0 excludes the middle KV tile, row 1 excludes the
    # first. A stale mask remains plausible but selects forbidden positions.
    masks = [torch.cat((torch.arange(256), torch.arange(512, 768))), torch.arange(256, 768)]
    candidates_cpu = torch.stack([masks[row % 2] for row in range(query_count)]).int()
    candidates = candidates_cpu.unsqueeze(1).to(device)
    common = dict(
        cu_seqlens_q=qsl,
        seqused_k=lengths,
        cmp_residual_k=residual,
        max_seqlen_q=query_count,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        heads, 1, width, topk, 2, batch_size=1, max_seqlen_k=kv_len, **common
    )
    selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        topk,
        2,
        block_table=table,
        metadata=metadata,
        candidate_topk_index=candidates,
        candidate_mode=2,
        candidate_topk_blocks=512,
        candidate_block_size=8,
        **common,
    )
    for row, indices in enumerate(selected.cpu().reshape(query_count, topk)):
        assert indices.unique().numel() == topk
        visible = (kv_len * ratio - query_count + row + 1) // ratio
        assert ((indices >= 0) & (indices < visible)).all()
        assert torch.isin(indices // 8, candidates_cpu[row]).all(), f"wrong candidate row at query {row}"
