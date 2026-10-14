# SPDX-License-Identifier: Apache-2.0
"""CPU numerical coverage of large-head graph argument and mask construction."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attention
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE


def _reference(query, key_cache, value_cache, block_tables, seq_lens, query_len, scale):
    """Gather only visible tokens per query, independently of the dense mask path."""
    result = torch.empty_like(query)
    block_size = key_cache.shape[1]
    heads_per_kv = query.shape[1] // key_cache.shape[2]
    for request, seq_len in enumerate(seq_lens):
        for offset in range(query_len):
            row = request * query_len + offset
            visible = seq_len - query_len + offset + 1
            keys = torch.stack(
                [key_cache[block_tables[request, t // block_size], t % block_size] for t in range(visible)]
            )
            values = torch.stack(
                [value_cache[block_tables[request, t // block_size], t % block_size] for t in range(visible)]
            )
            for head in range(query.shape[1]):
                kv_head = head // heads_per_kv
                scores = keys[:, kv_head].float() @ query[row, head].float() * scale
                result[row, head] = scores.softmax(dim=0) @ values[:, kv_head].float()
    return result


def _dense_attention(*, query, key, value, atten_mask, scale, **kwargs):
    """Replace only the NPU kernel; execute real production gather/layout/mask code."""
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = query @ key.transpose(-1, -2) * scale
    return (scores.masked_fill(atten_mask, -torch.inf).softmax(dim=-1) @ value,)


def _inputs(query_len):
    generator = torch.Generator().manual_seed(2026)
    head_size = FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
    impl = attention.AscendAttentionBackendImpl.__new__(attention.AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, head_size
    impl.scale = head_size**-0.5
    impl.key_cache = torch.randn(6, 4, 2, head_size, generator=generator)
    impl.value_cache = torch.randn(6, 4, 2, head_size, generator=generator)
    query = torch.randn(2 * query_len, 4, head_size, generator=generator)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([7, 5], dtype=torch.int32),
        seq_lens_device=torch.tensor([7, 5], dtype=torch.int32),
        seq_lens_list=[7, 5],
        # Poison unused columns: they must never enter the attention result.
        block_tables=torch.tensor([[3, 1, 999], [4, 2, -1]], dtype=torch.int32),
    )
    return impl, query, metadata


@pytest.mark.parametrize("query_len", [2, 4])
def test_verify_numerics_follow_dynamic_lengths_and_block_tables(query_len):
    impl, query, metadata = _inputs(query_len)
    output = torch.empty_like(query)
    first = None
    with patch.object(attention.torch_npu, "npu_fusion_attention_v3", side_effect=_dense_attention, create=True):
        for seq_lens in ([7, 5], [5, 7]):
            metadata.seq_lens_device.copy_(torch.tensor(seq_lens, dtype=torch.int32))
            actual = impl._forward_large_head_graph_verify_attention(query, metadata, output)
            expected = _reference(
                query, impl.key_cache, impl.value_cache, metadata.block_tables, seq_lens, query_len, impl.scale
            )
            assert actual is output
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            if first is None:
                first = actual.clone()
                metadata.block_tables[:, :2].copy_(metadata.block_tables[:, :2].flip(0))
                query.mul_(0.5)
                impl.value_cache.add_(0.25)
            else:
                assert not torch.allclose(actual, first)


def test_verify_does_not_observe_future_values():
    impl, query, metadata = _inputs(4)
    with patch.object(attention.torch_npu, "npu_fusion_attention_v3", side_effect=_dense_attention, create=True):
        before = impl._forward_large_head_graph_verify_attention(query, metadata, torch.empty_like(query)).clone()
        # Request zero's last token is visible only to its last verification query.
        impl.value_cache[metadata.block_tables[0, 1], 2].add_(100)
        after = impl._forward_large_head_graph_verify_attention(query, metadata, torch.empty_like(query))
    torch.testing.assert_close(after[:3], before[:3])
    torch.testing.assert_close(after[4:], before[4:])
    assert not torch.allclose(after[3], before[3])


def test_decode_task_updates_buffers_and_numerics_without_rebinding():
    impl, query, metadata = _inputs(1)
    metadata.block_tables = metadata.block_tables[:, :2].contiguous()
    impl._graph_metadata_layer_name = lambda: "layer"
    output = torch.empty_like(query)
    with (
        patch.object(attention, "get_capture_resource", return_value=torch.empty(0)),
        patch.object(attention, "register_task") as register,
    ):
        impl.full_graph_fia_bnsd_large_head(query, metadata, output)
    _, kwargs, provider = register.call_args.args
    assert kwargs["input_layout"] == "BNSD"
    assert kwargs["num_heads"] == 4
    assert kwargs["num_key_value_heads"] == 2
    assert kwargs["query"].data_ptr() == query.data_ptr()
    assert kwargs["out"][0].data_ptr() == output.data_ptr()
    pointers = tuple(t.data_ptr() for t in (provider.block_table, provider.actual_seq_lengths_kv))
    for seq_lens in ([7, 5], [5, 7]):
        metadata.seq_lens_list = seq_lens
        resolved = provider.resolve({"layer": metadata})
        assert pointers == tuple(resolved[k].data_ptr() for k in ("block_table", "actual_seq_lengths_kv"))
        torch.testing.assert_close(resolved["block_table"], metadata.block_tables)
        assert resolved["actual_seq_lengths_kv"].tolist() == seq_lens
        # Execute the captured layout using SDPA, not the paged per-token oracle.
        for request, length in enumerate(resolved["actual_seq_lengths_kv"].tolist()):
            blocks = resolved["block_table"][request].long()
            key = kwargs["key"][blocks].reshape(-1, 2, impl.head_size)[:length].transpose(0, 1)
            value = kwargs["value"][blocks].reshape(-1, 2, impl.head_size)[:length].transpose(0, 1)
            kwargs["out"][0][request].copy_(
                torch.nn.functional.scaled_dot_product_attention(
                    kwargs["query"][request], key, value, enable_gqa=True, scale=kwargs["scale"]
                )
            )
        expected = _reference(query, impl.key_cache, impl.value_cache, metadata.block_tables, seq_lens, 1, impl.scale)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        metadata.block_tables.copy_(metadata.block_tables.flip(0))


def test_replay_padding_clears_stale_rows_without_changing_addresses():
    table = torch.full((3, 3), 9, dtype=torch.int32)
    lengths = torch.full((3,), 99, dtype=torch.int32)
    provider = attention.FIABnsdLargeHeadParamProvider("layer", table, lengths, torch.ones(3, dtype=torch.int32))
    metadata = SimpleNamespace(seq_lens_list=[5], block_tables=torch.tensor([[3, 1]], dtype=torch.int32))
    args = provider.resolve({"layer": metadata})
    assert args["block_table"] is table
    assert args["actual_seq_lengths_kv"] is lengths
    torch.testing.assert_close(table, torch.tensor([[3, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.int32))
    assert lengths.tolist() == [5, 1, 1]


def test_replay_rejects_wider_block_table():
    metadata = SimpleNamespace(seq_lens_list=[5], block_tables=torch.ones((1, 3), dtype=torch.int32))
    with pytest.raises(RuntimeError, match="wider block table"):
        attention.bnsd_large_head_decode_args(metadata, 1, (1, 2))
