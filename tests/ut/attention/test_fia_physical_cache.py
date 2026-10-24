# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.attention.test_fia_non_pa_inputs import _make_impl, _make_metadata
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    FIAParamProvider,
)
from vllm_ascend.attention.utils import fia_physical_block_table, fia_physical_kv_views


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("offset", [0, 17])
def test_physical_views_preserve_selected_pages_and_storage(factor, offset):
    blocks, page, width = 5, 8, 4
    storage = torch.arange(offset + blocks * factor * page * width + page * width)
    key = storage.as_strided((blocks, page, width), (factor * page * width, width, 1), offset)
    value = storage.as_strided(key.shape, key.stride(), offset + page * width)
    physical_key, physical_value, stride = fia_physical_kv_views(key, value)
    ids = torch.tensor([4, 1, 3, 0])
    assert stride == factor
    for source, physical in [(key, physical_key), (value, physical_value)]:
        assert physical.is_contiguous()
        assert physical.data_ptr() == source.data_ptr()
        assert physical.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        torch.testing.assert_close(physical[ids * stride], source[ids])


def test_non_ndh_inner_layout_is_not_reinterpreted():
    key = torch.zeros(4, 8, 16)[..., ::2]
    out_key, out_value, stride = fia_physical_kv_views(key, key)
    assert out_key is key and out_value is key and stride == 1


def test_block_addresses_are_cached_per_metadata_and_keep_sentinels():
    first = SimpleNamespace(block_tables=torch.tensor([[2, -1, 0]], dtype=torch.int32))
    converted = fia_physical_block_table(first, 2)
    torch.testing.assert_close(converted, torch.tensor([[4, -1, 0]], dtype=torch.int32))
    assert fia_physical_block_table(first, 2) is converted
    assert fia_physical_block_table(first, 1) is first.block_tables
    second = SimpleNamespace(block_tables=torch.tensor([[3, 1, 0]], dtype=torch.int32))
    assert fia_physical_block_table(second, 2) is not converted
    torch.testing.assert_close(fia_physical_block_table(second, 2), second.block_tables * 2)


@pytest.mark.parametrize("sliding_window", [None, 16])
def test_graph_provider_resolves_current_step_physical_addresses(sliding_window):
    provider = FIAParamProvider("layer", sliding_window, block_stride=2)
    for table in ([[3, 1]], [[2, 0]]):
        metadata = SimpleNamespace(block_tables=torch.tensor(table), actual_seq_lengths_q=[2], seq_lens_list=[133])
        params = provider.resolve({"layer": metadata})
        torch.testing.assert_close(params["block_table"], metadata.block_tables * 2)
        assert params["actual_seq_lengths_kv"] == [133]


@pytest.mark.parametrize(
    "state",
    [AscendAttentionState.DecodeOnly, AscendAttentionState.PrefillCacheHit, AscendAttentionState.ChunkedPrefill],
)
def test_regular_fia_uses_dense_views_without_copying_cache(state):
    """Exercise the actual parameter path without speculative metadata."""
    backing = torch.arange(5 * 2 * 8 * 2 * 4).reshape(5, 2, 8, 2, 4)
    key, value = backing[:, 0], backing[:, 1]
    impl = _make_impl()
    impl.key_cache, impl.value_cache = key, value
    impl.use_bnsd_kv_cache = False
    impl.sinks = None
    metadata = _make_metadata(state)
    metadata.block_tables = torch.tensor([[4, 1], [3, 0]], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([9, 12])
    metadata.seq_lens_list = [9, 12]
    query = torch.zeros(3, 2, 4)
    output = torch.empty_like(query)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
        patch.object(attn_module.envs_vllm, "VLLM_BATCH_INVARIANT", False),
        patch.object(attn_module, "get_current_hardware_profile", return_value=Mock(supports=lambda _: False)),
        patch.object(attn_module.DeviceOperator, "npu_fused_infer_attention_score") as fia,
    ):
        fia.return_value = (torch.ones_like(query), None)
        impl.forward_fused_infer_attention(query, key, value, metadata, output)
    params = fia.call_args.kwargs
    out_key, out_value, table = params["key"], params["value"], params["block_table"]
    block_size, lengths = params["block_size"], params["actual_seq_lengths_kv"]
    assert out_key.is_contiguous() and out_value.is_contiguous()
    assert block_size == 8 and lengths == [9, 12]
    for source, physical in [(key, out_key), (value, out_value)]:
        assert physical.data_ptr() == source.data_ptr()
        selected = source[metadata.block_tables].flatten(-2)
        torch.testing.assert_close(physical[table], selected)


def test_non_pa_graph_provider_does_not_supply_physical_pages():
    provider = FIAParamProvider("layer", None, is_non_pa=True, block_stride=2)
    metadata = SimpleNamespace(block_tables=torch.tensor([[3, 1]]), actual_seq_lengths_q=[2], seq_lens_list=[2])
    assert provider.resolve({"layer": metadata})["block_table"] is None
