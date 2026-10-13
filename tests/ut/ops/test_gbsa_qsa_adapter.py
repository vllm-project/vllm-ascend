import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tail_tokens", range(4))
def test_qsa_gbsa_uses_original_paged_cache(dtype, tail_tokens):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gbsa_qsa.py"
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
            selected_tokens = []
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
        seq_lens,
        output,
    )

    assert result is output
    assert calls[0][0] == keys.data_ptr()
    assert calls[0][1] == values.data_ptr()
    torch.testing.assert_close(calls[0][2], block_table)


def test_a3_residual_mode_reuses_runtime_kernel_key():
    root = Path(__file__).resolve().parents[3]
    tiling_source = (
        root
        / "csrc/attention/generic_block_sparse_attention/op_host/"
        "generic_block_sparse_attention_tiling.cpp"
    ).read_text()
    arch22_kernel = (
        root
        / "csrc/attention/generic_block_sparse_attention/op_kernel/arch22/"
        "generic_block_sparse_attention_kernel_arch22.h"
    ).read_text()
    tiling_keys = (
        root
        / "csrc/attention/generic_block_sparse_attention/op_kernel/"
        "generic_block_sparse_attention_tilingkey.h"
    ).read_text()

    assert (
        "if (isArch35 && residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE)"
        in tiling_source
    )
    assert "residualBlockMode_ = tilingData->residualBlockMode;" in arch22_kernel
    assert "GBSA_BF16_TND_PA_BBND_TILING 9200000071023222ULL" in tiling_keys
    assert (
        "GBSA_BF16_TND_PA_BBND_FP32SM_TILING_RESIDUAL "
        "9250001071023222ULL"
        in tiling_keys
    )


def test_a3_qsa_arch22_maps_logical_blocks_into_paged_kv():
    root = Path(__file__).resolve().parents[3]
    arch22_kernel = (
        root
        / "csrc/attention/generic_block_sparse_attention/op_kernel/arch22/"
        "generic_block_sparse_attention_kernel_arch22.h"
    ).read_text()
    tiling_source = (
        root
        / "csrc/attention/generic_block_sparse_attention/op_host/"
        "generic_block_sparse_attention_tiling.cpp"
    ).read_text()

    assert "MAX_SPARSE_BLOCK_CAPACITY = 512" in arch22_kernel
    assert "logicalTokenOffset / blockSize_" in arch22_kernel
    assert "logicalTokenOffset % blockSize_" in arch22_kernel
    assert "tokenOffsetInPage) * strideKVRow" in arch22_kernel
    assert "blockSize_ % blockShapeY_ != 0U" in tiling_source
