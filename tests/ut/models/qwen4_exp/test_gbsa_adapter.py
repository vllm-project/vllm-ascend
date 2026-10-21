import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tail_tokens", range(4))
def test_qsa_gbsa_uses_original_paged_cache(dtype, tail_tokens):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = []

    def metadata(indices, counts, query_heads, kv_heads, head_dim, block_shape, **kwargs):
        assert block_shape == [1, 4]
        assert head_dim == 128
        assert kwargs["residual_block_mode"] == 1
        assert kwargs["seqused_kv"].tolist() == [8 + tail_tokens]
        return torch.empty(0)

    def attention(query, keys, values, indices, counts, block_shape, **kwargs):
        calls.append((keys.data_ptr(), values.data_ptr(), kwargs["block_table"].clone()))
        results = []
        for row in range(query.shape[0]):
            explicit = indices[0, row, : counts[0, row]].tolist()
            sequence_length = int(kwargs["seqused_kv"][row])
            selected_tokens: list[int] = []
            for block in explicit:
                selected_tokens.extend(range(block * 4, block * 4 + 4))
            selected_tokens.extend(range(sequence_length - tail_tokens, sequence_length))
            table = kwargs["block_table"][row]
            physical = table[torch.tensor(selected_tokens) // 128].long()
            offsets = torch.tensor(selected_tokens) % 128
            selected_keys = keys[physical, offsets].float().repeat_interleave(2, dim=1)
            selected_values = values[physical, offsets].float().repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,shd->hs", query[row].float(), selected_keys)
            probabilities = (scores / 128**0.5).softmax(-1)
            results.append(torch.einsum("hs,shd->hd", probabilities, selected_values))
        return torch.stack(results).to(query.dtype), torch.empty(0)

    namespace = {
        "torch": torch,
        "generic_block_sparse_attention_metadata": metadata,
        "generic_block_sparse_attention": attention,
    }
    exec(compile(module, str(source), "exec"), namespace)

    torch.manual_seed(42 + tail_tokens)
    query = torch.randn(1, 4, 128, dtype=dtype)
    keys = torch.randn(3, 128, 2, 128, dtype=dtype)
    values = torch.randn_like(keys)
    block_table = torch.tensor([[2]], dtype=torch.int32)
    seq_lens = torch.tensor([8 + tail_tokens], dtype=torch.int32)
    groups = torch.tensor([0, 1], dtype=torch.int32)
    compact_indices = torch.cat((groups, torch.full((3,), -1, dtype=torch.int32))).view(1, -1)
    output = torch.empty_like(query)

    result = namespace["qsa_gbsa_attention"](
        query,
        keys,
        values,
        compact_indices,
        block_table,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([7 + tail_tokens], dtype=torch.int32),
        seq_lens,
        output,
    )

    assert result is output
    assert calls[0][0] == keys.data_ptr()
    assert calls[0][1] == values.data_ptr()
    torch.testing.assert_close(calls[0][2], block_table)


def test_qsa_gbsa_uses_per_row_mtp_visible_lengths():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = []

    def metadata(indices, counts, query_heads, kv_heads, head_dim, block_shape, **kwargs):
        calls.append((indices.clone(), counts.clone(), kwargs["seqused_kv"].clone()))
        return torch.empty(0)

    def attention(query, keys, values, indices, counts, block_shape, **kwargs):
        return torch.zeros_like(query), torch.empty(0)

    namespace = {
        "torch": torch,
        "generic_block_sparse_attention_metadata": metadata,
        "generic_block_sparse_attention": attention,
    }
    exec(compile(module, str(source), "exec"), namespace)

    query = torch.randn(4, 4, 128, dtype=torch.bfloat16)
    keys = torch.randn(2, 128, 2, 128, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    compact_indices = torch.tensor(
        [
            [24, 25, -1, -1, -1],
            [24, 25, -1, -1, -1],
            [24, 25, -1, -1, -1],
            [24, 25, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    output = torch.empty_like(query)

    namespace["qsa_gbsa_attention"](
        query,
        keys,
        values,
        compact_indices,
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        torch.tensor([101, 102, 103, 104], dtype=torch.int32),
        torch.tensor([105], dtype=torch.int32),
        output,
    )

    indices, counts, visible_lengths = calls[0]
    assert visible_lengths.tolist() == [102, 103, 104, 105]
    assert counts[0].tolist() == [1, 1, 2, 2]
    assert indices[0, :, :2].tolist() == [
        [24, 25],
        [24, 25],
        [24, 25],
        [24, 25],
    ]


def test_residual_mode_has_arch_specific_kernel_keys():
    root = Path(__file__).resolve().parents[4]
    tiling_source = (
        root / "csrc/attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.cpp"
    ).read_text()
    arch22_kernel = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/arch22/"
        "generic_block_sparse_attention_kernel_arch22.h"
    ).read_text()
    tiling_keys = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/generic_block_sparse_attention_tilingkey.h"
    ).read_text()
    kernel_dispatch = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/generic_block_sparse_attention.cpp"
    ).read_text()

    assert "if (residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE)" in tiling_source
    assert "tilingKey += GBSA_RESIDUAL_BLOCK_MODE_TILING_OFFSET;" in tiling_source
    assert "residualBlockMode_ = tilingData->residualBlockMode;" in arch22_kernel
    assert "GBSA_BF16_TND_PA_BBND_TILING_RESIDUAL 9200001071023222ULL" in tiling_keys
    assert "GBSA_BF16_TND_PA_BBND_FP32SM_TILING_RESIDUAL 9250001071023222ULL" in tiling_keys
    assert "TILING_KEY_IS(GBSA_BF16_TND_PA_BBND_TILING_RESIDUAL);" in kernel_dispatch


def test_a3_qsa_arch22_maps_logical_tokens_into_paged_kv():
    root = Path(__file__).resolve().parents[4]
    arch22_kernel = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/arch22/"
        "generic_block_sparse_attention_kernel_arch22.h"
    ).read_text()
    qk_mmad = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp"
    ).read_text()
    pv_mmad = (
        root / "csrc/attention/generic_block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp"
    ).read_text()
    tiling_source = (
        root / "csrc/attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.cpp"
    ).read_text()

    assert "blockSize_, kStride0_" in arch22_kernel
    assert "blockSize_, vStride0_" in arch22_kernel
    for mmad_source in (qk_mmad, pv_mmad):
        assert "pageIdx = static_cast<uint32_t>(offsetInKV / pageSize)" in mmad_source
        assert "physicalPage = gBlockTable.GetValue(pageIdx)" in mmad_source
        assert "(offsetInKV % pageSize) * strideKV" in mmad_source
    assert "blockSize_ == 128U" in tiling_source
    assert "(128U % blockShapeY_) == 0U" in tiling_source
    assert "GBSA_A3_QSA_MAX_TOPK = 512" in tiling_source
