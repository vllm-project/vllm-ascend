# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


@pytest.mark.parametrize("queries", [1, 5])
@pytest.mark.parametrize("context", [1536, 8192, 32768])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("topk", [64, 512])
@torch.inference_mode()
def test_qli_candidate_vector_repeats(queries, context, ratio, mode, topk):
    """Check candidate padding, partial causal blocks and strided PA caches."""
    generator = torch.Generator().manual_seed(1234)
    heads, width = 32, 128
    block_size = 128 if ratio == 1 else 64
    query_cpu = torch.randint(-90, 90, (queries, heads, width), dtype=torch.int8, generator=generator)
    key_cpu = torch.randint(
        -90, 90, (context // block_size, block_size, 1, width), dtype=torch.int8, generator=generator
    )
    weights_cpu = torch.rand((queries, heads), generator=generator).half()
    query_scale_cpu = torch.full((queries, heads), 0.01, dtype=torch.float16)
    key_scale_cpu = torch.full(key_cpu.shape[:-1], 0.01, dtype=torch.float16)
    stride = 147712 if ratio == 1 else 131072
    key = torch.empty_strided(key_cpu.shape, (stride, 128, 128, 1), dtype=torch.int8, device="npu")
    key.copy_(key_cpu)
    key_scale = torch.empty_strided(key_scale_cpu.shape, (stride // 2, 1, 1), dtype=torch.float16, device="npu")
    key_scale.copy_(key_scale_cpu)
    candidates_cpu = torch.stack([torch.randperm(context // 8, generator=generator)[:2048] for _ in range(queries)])
    candidates_cpu = torch.nn.functional.pad(candidates_cpu, (0, 2048 - candidates_cpu.shape[1]), value=-1)
    candidates = candidates_cpu.int().unsqueeze(1).npu() if mode == 2 else None
    common = dict(
        cu_seqlens_q=torch.tensor([0, queries], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([context], dtype=torch.int32, device="npu"),
        cmp_residual_k=torch.tensor([0], dtype=torch.int32, device="npu") if ratio == 2 else None,
        max_seqlen_q=queries,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        heads, 1, width, topk, 2, batch_size=1, max_seqlen_k=context, device="npu:0", **common
    )
    selected, _, candidate_output = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query_cpu.npu(),
        key,
        weights_cpu.npu(),
        query_scale_cpu.npu(),
        key_scale,
        topk,
        2,
        block_table=torch.arange(context // block_size, dtype=torch.int32, device="npu").unsqueeze(0),
        metadata=metadata,
        candidate_topk_index=candidates,
        candidate_mode=mode,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        **common,
    )
    dot = torch.einsum("qhd,kd->qhk", query_cpu.float(), key_cpu.reshape(context, width).float())
    relu = (dot / 1024).clamp_min(0).half().float()
    scores = (relu * (weights_cpu * query_scale_cpu).float().unsqueeze(-1)).sum(1)
    scores *= key_scale_cpu.reshape(context).float()
    selected = selected.cpu().reshape(queries, topk).long()
    for row in range(queries):
        visible = (context * ratio - queries + row + 1) // ratio
        scores[row, visible:] = -torch.inf
        if mode == 2:
            allowed = torch.isin(torch.arange(context) // 8, candidates_cpu[row])
            scores[row, ~allowed] = -torch.inf
        indices = selected[row]
        assert indices.unique().numel() == topk
        assert ((indices >= 0) & (indices < visible)).all()
        torch.testing.assert_close(
            scores[row, indices].sort(descending=True).values,
            scores[row].topk(topk).values,
            rtol=1e-4,
            atol=1e-6,
        )
        if mode == 1:
            blocks = candidate_output.cpu().reshape(queries, -1)[row]
            assert ((blocks == -1) | (blocks >= 0)).all()
            blocks = blocks[blocks >= 0].long()
            expected_blocks = (visible + 7) // 8
            assert blocks.unique().numel() == min(expected_blocks, 2048)
            assert ((blocks >= 0) & (blocks < expected_blocks)).all()
            if expected_blocks <= 2048:
                torch.testing.assert_close(blocks.sort().values, torch.arange(expected_blocks))


@pytest.mark.parametrize("topk", [1, 63, 65, 1024, 2048])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("mode", [1, 2, 3])
def test_qli_adaptive_topk_boundaries(topk, ratio, mode):
    """Exercise partial 64-element widths and the unchanged maximum width."""
    test_qli_candidate_vector_repeats(queries=5, context=8192, ratio=ratio, mode=mode, topk=topk)
