from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


def _load_cache_views():
    import math

    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/cache_config.py"
    module = ast.parse(source.read_text())
    names = {"make_slab_view", "make_qsa_kernel_page_cache"}
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch,
        "math": math,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("group_ids", [(0, 1), (31, 32), (95, 96)])
@pytest.mark.parametrize("storage_page_size", [128, 384, 768])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tail_tokens", range(4))
def test_qsa_gbsa_uses_kernel_paged_cache(dtype, tail_tokens, storage_page_size, group_ids):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = []

    def metadata(indices, counts, query_heads, kv_heads, head_dim, block_shape, **kwargs):
        assert block_shape == [1, 4]
        assert head_dim == 128
        assert kwargs["residual_block_mode"] == 0
        assert kwargs["seqused_kv"].tolist() == [sequence_length]
        return torch.empty(0)

    def attention(query, keys, values, indices, counts, block_shape, **kwargs):
        calls.append((keys.data_ptr(), values.data_ptr(), kwargs["block_table"].clone()))
        results = []
        for row in range(query.shape[0]):
            explicit = indices[0, row, : counts[0, row]].tolist()
            sequence_length = int(kwargs["seqused_kv"][row])
            selected_tokens: list[int] = []
            for block in explicit:
                selected_tokens.extend(token for token in range(block * 4, block * 4 + 4) if token < sequence_length)
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
    keys = torch.randn(3, storage_page_size, 2, 128, dtype=dtype)
    values = torch.randn_like(keys)
    sequence_length = (max(group_ids) + 1) * 4 + tail_tokens
    logical_blocks = (sequence_length + storage_page_size - 1) // storage_page_size
    manager_table = torch.tensor([[(index + 2) % 3 for index in range(logical_blocks)]], dtype=torch.int32)
    subpages = storage_page_size // 128
    block_table = (manager_table.unsqueeze(-1) * subpages + torch.arange(subpages)).flatten(1).to(torch.int32)
    seq_lens = torch.tensor([sequence_length], dtype=torch.int32)
    groups = torch.tensor(group_ids, dtype=torch.int32)
    compact_indices = torch.cat((groups, torch.full((3,), -1, dtype=torch.int32))).view(1, -1)
    output = torch.empty_like(query)

    # Aligned storage is contiguous [K128,V128] pairs. MRV2 IDs stay unchanged.
    dtype_size = keys.element_size()
    page_bytes = 128 * 2 * 128 * dtype_size
    block_stride = subpages
    backing = torch.zeros(3 * block_stride * 2 * page_bytes, dtype=torch.int8)
    input_keys, input_values = _load_cache_views()["make_qsa_kernel_page_cache"](
        backing,
        dtype=dtype,
        num_blocks=3,
        manager_block_size=storage_page_size,
        kernel_block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        storage_offset=0,
        block_stride=block_stride * 2 * page_bytes,
    )
    for manager in range(3):
        for subpage in range(subpages):
            page = manager * block_stride + subpage
            input_keys[page].copy_(keys[manager, subpage * 128 : (subpage + 1) * 128])
            input_values[page].copy_(values[manager, subpage * 128 : (subpage + 1) * 128])
    input_table = block_table

    result = namespace["qsa_gbsa_attention"](
        query,
        input_keys,
        input_values,
        compact_indices,
        input_table,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([sequence_length - 1], dtype=torch.int32),
        seq_lens,
        output,
    )

    assert result is output
    selected_tokens = [token for group in group_ids for token in range(group * 4, group * 4 + 4)]
    selected_tokens.extend(range(sequence_length - tail_tokens, sequence_length))
    tokens = torch.tensor(selected_tokens)
    physical = manager_table[0, tokens // storage_page_size].long()
    offsets = tokens % storage_page_size
    ref_keys = keys[physical, offsets].float().repeat_interleave(2, dim=1)
    ref_values = values[physical, offsets].float().repeat_interleave(2, dim=1)
    scores = torch.einsum("hd,shd->hs", query[0].float(), ref_keys) / 128**0.5
    expected_output = torch.einsum("hs,shd->hd", scores.softmax(-1), ref_values).to(dtype)
    torch.testing.assert_close(output[0], expected_output)
    assert calls[0][0] == input_keys.data_ptr()
    assert calls[0][1] == input_values.data_ptr()
    expected = manager_table.unsqueeze(-1) * block_stride + torch.arange(subpages)
    torch.testing.assert_close(calls[0][2], expected.flatten(1).to(torch.int32))


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
    assert counts[0].tolist() == [2, 2, 2, 3]
    assert indices[0, :, :2].tolist() == [
        [24, 25],
        [24, 25],
        [24, 25],
        [24, 25],
    ]
    assert indices[0, 3, 2].item() == 26


def test_qsa_gbsa_reserves_tail_within_512_entry_limit():
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

    compact_indices = torch.cat(
        (
            torch.arange(512, dtype=torch.int32),
            torch.full((3,), -1, dtype=torch.int32),
        )
    ).view(1, -1)
    query = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    keys = torch.randn(17, 128, 2, 128, dtype=torch.bfloat16)
    values = torch.randn_like(keys)

    namespace["qsa_gbsa_attention"](
        query,
        keys,
        values,
        compact_indices,
        torch.arange(17, dtype=torch.int32).view(1, -1),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2048], dtype=torch.int32),
        torch.tensor([2049], dtype=torch.int32),
        torch.empty_like(query),
    )

    indices, counts, visible_lengths = calls[0]
    assert indices.shape == (2, 1, 512)
    assert visible_lengths.tolist() == [2049]
    assert counts[:, 0].tolist() == [512, 512]
    assert indices[0, 0, :511].tolist() == list(range(511))
    assert indices[0, 0, 511].item() == 512


def test_gbsa_uses_cann_transformer_op_api_without_bundled_sources():
    root = Path(__file__).resolve().parents[4]
    bundled_source = root / "csrc/attention/generic_block_sparse_attention"
    adapter = (root / "csrc/aclnn_torch_adapter/gbsa_torch_adpt.h").read_text()
    binding = (root / "csrc/torch_binding.cpp").read_text()
    loader = (root / "csrc/aclnn_torch_adapter/op_api_common.h").read_text()

    assert not bundled_source.exists()
    assert "EXEC_NPU_CMD(aclnnGenericBlockSparseAttentionMetadata" in adapter
    assert "EXEC_NPU_CMD(aclnnGenericBlockSparseAttention" in adapter
    assert '#include "aclnn_torch_adapter/gbsa_torch_adpt.h"' in binding
    assert '"npu_generic_block_sparse_attention_metadata(' in binding
    assert '"npu_generic_block_sparse_attention(' in binding
    assert 'return "libopapi_transformer.so";' in loader


@pytest.mark.parametrize("manager_block_size", [128, 384, 896])
@pytest.mark.parametrize("head_size", [64, 128])
def test_kernel_page_views_preserve_packed_addresses(manager_block_size, head_size):
    namespace = _load_cache_views()
    kernel_size, heads, num_blocks = 128, 1, 3
    page_bytes = kernel_size * heads * head_size * 2
    manager_bytes = manager_block_size * heads * head_size * 2
    block_stride = 2 * manager_bytes
    backing = torch.zeros(num_blocks * block_stride + 32, dtype=torch.int8)
    cache = namespace["make_qsa_kernel_page_cache"](
        backing,
        dtype=torch.float16,
        num_blocks=num_blocks,
        manager_block_size=manager_block_size,
        kernel_block_size=kernel_size,
        num_kv_heads=heads,
        head_size=head_size,
        head_size_v=head_size,
        storage_offset=32,
        block_stride=block_stride,
    )
    key, value = cache
    factor = manager_block_size // kernel_size
    manager_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    virtual_table = (manager_table.unsqueeze(-1) * factor + torch.arange(factor)).flatten(1)
    slots = torch.tensor(
        [
            0,
            127,
            manager_block_size - 1,
            manager_block_size,
            3 * manager_block_size - 1,
            -1,
            3 * manager_block_size + 17,
        ]
    )
    physical = virtual_table
    assert key.shape[1] == value.shape[1] == kernel_size
    assert key.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    for logical_page, physical_page in zip(virtual_table[0].tolist(), physical[0].tolist(), strict=True):
        manager, subpage = divmod(logical_page, factor)
        expected = 32 + manager * block_stride + subpage * 2 * page_bytes
        assert key[physical_page].data_ptr() - backing.data_ptr() == expected
        assert value[physical_page].data_ptr() - backing.data_ptr() == expected + page_bytes
    # The unchanged store kernel consumes already-physical token slots.
    physical_slots = slots[:5]
    blocks, offsets = physical_slots // kernel_size, physical_slots % kernel_size
    actual_offsets = (blocks * key.stride(0) + offsets * key.stride(1)) * 2
    expected_offsets = torch.tensor(
        [
            (slot // manager_block_size) * block_stride
            + ((slot % manager_block_size) // kernel_size) * 2 * page_bytes
            + (slot % kernel_size) * head_size * 2
            for slot in slots[:5].tolist()
        ]
    )
    torch.testing.assert_close(actual_offsets, expected_offsets)
    # Writes to the final subpage of block 2 must not alter block 0 or 1.
    last_page = 2 * (block_stride // (2 * page_bytes)) + factor - 1
    key[last_page].fill_(3)
    value[last_page].fill_(7)
    assert torch.count_nonzero(backing[32 : 32 + 2 * block_stride]) == 0
    k_start = 32 + 2 * block_stride + (factor - 1) * 2 * page_bytes
    torch.testing.assert_close(
        backing[k_start : k_start + page_bytes].view(torch.float16),
        torch.full((page_bytes // 2,), 3, dtype=torch.float16),
    )


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_whole_page_kernel_pairs_reuse_mrv2_ids(num_kv_heads):
    # TP8/TP4 local KV geometry: seven 128-token pairs fill each manager
    # page, including the page shared with the corresponding GDN states.
    manager_size, kernel_size, num_blocks, head_size = 896, 128, 3, 128
    page_bytes = kernel_size * num_kv_heads * head_size * 2
    manager_bytes = 7 * 2 * page_bytes
    backing = torch.zeros(num_blocks * manager_bytes, dtype=torch.int8)
    keys, values = _load_cache_views()["make_qsa_kernel_page_cache"](
        backing,
        dtype=torch.bfloat16,
        num_blocks=num_blocks,
        manager_block_size=manager_size,
        kernel_block_size=kernel_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        storage_offset=0,
        block_stride=manager_bytes,
    )
    assert keys.shape == values.shape == (21, 128, num_kv_heads, 128)
    assert keys.stride(0) * keys.element_size() == 2 * page_bytes
    assert values.data_ptr() - keys.data_ptr() == page_bytes
    # Cross a manager boundary: K6,V6 must be followed immediately by K7,V7.
    keys[6].fill_(3)
    values[6].fill_(5)
    keys[7].fill_(7)
    values[7].fill_(9)
    for page, value in ((12, 3), (13, 5), (14, 7), (15, 9)):
        torch.testing.assert_close(
            backing[page * page_bytes : (page + 1) * page_bytes].view(torch.bfloat16),
            torch.full((page_bytes // 2,), value, dtype=torch.bfloat16),
        )
    # The existing MRV2 IDs and token slots address these interleaved views.
    block_ids = torch.tensor([6, 7, 13, 14, 20])
    slots = torch.tensor([895, 896, 1791, 1792, 2687])
    torch.testing.assert_close(slots // kernel_size, block_ids)
    for slot in slots.tolist():
        page, token = divmod(slot, kernel_size)
        expected = page * 2 * page_bytes + token * num_kv_heads * head_size * 2
        assert keys[page, token].data_ptr() - backing.data_ptr() == expected
