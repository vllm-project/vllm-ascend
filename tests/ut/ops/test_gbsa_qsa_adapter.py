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
