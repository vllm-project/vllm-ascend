# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import attention_v1
from vllm_ascend.attention.kv_cache_view import (
    dense_paged_kv_views,
    dense_paged_kv_write_views,
    remap_paged_block_table,
    remap_paged_slots,
)


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("offset", [0, 17])
def test_dense_paged_kv_views_alias_pages(factor, offset):
    pages, block, hidden = 7, 4, 8
    storage = torch.arange(offset + pages * factor * block * hidden, dtype=torch.float16)
    key = storage.as_strided((pages, block, hidden), (factor * block * hidden, hidden, 1), offset)
    value = storage.as_strided(key.shape, key.stride(), offset + block * hidden)
    table = torch.tensor([[6, 0, 3], [5, 1, 2]], dtype=torch.int32)
    dense_key, dense_value, stride = dense_paged_kv_views(key, value, table)
    assert stride == factor
    assert dense_key.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    assert dense_value.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    for page in range(pages):
        torch.testing.assert_close(dense_key[page * factor], key[page])
        torch.testing.assert_close(dense_value[page * factor], value[page])
    torch.testing.assert_close(remap_paged_block_table(table, stride), table * factor)
    assert table.tolist() == [[6, 0, 3], [5, 1, 2]]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_dense_writer_maps_slots_without_mutating_scheduler_input(dtype):
    pages, block, heads, dim, factor = 7, 4, 2, 8, 3
    storage = torch.zeros((pages * factor + 1) * block * heads * dim, dtype=torch.float16)
    key = storage.as_strided((pages, block, heads, dim), (factor * block * heads * dim, heads * dim, dim, 1))
    value = storage.as_strided(key.shape, key.stride(), block * heads * dim)
    slots = torch.tensor([-1, block - 1, block, pages * block - 1], dtype=dtype)
    before = slots.clone()
    dense_key, dense_value, stride = dense_paged_kv_write_views(key, value, slots)
    mapped = remap_paged_slots(slots, block, stride)
    assert stride == factor
    assert dense_key.is_contiguous() and dense_value.is_contiguous()
    torch.testing.assert_close(mapped, torch.tensor([-1, 3, 12, 75], dtype=dtype))
    torch.testing.assert_close(slots, before)


@pytest.mark.parametrize("entrypoint", ["do_kv_cache_update", "reshape_and_cache"])
def test_backend_writer_passes_aliases_and_mapped_slots(monkeypatch, entrypoint):
    block, pages, dim, factor = 4, 7, 8, 3
    key = torch.zeros(pages * factor * block * dim, dtype=torch.float16).as_strided(
        (pages, block, 1, dim), (factor * block * dim, dim, dim, 1)
    )
    value = torch.zeros(pages * factor * block * dim, dtype=torch.float16).as_strided(key.shape, key.stride())
    backend = attention_v1.AscendAttentionBackendImpl.__new__(attention_v1.AscendAttentionBackendImpl)
    backend.__dict__.update(
        key_cache=key,
        value_cache=value,
        attn_type=attention_v1.AttentionType.DECODER,
        vllm_config=SimpleNamespace(speculative_config=None),
        use_bnsd_kv_cache=False,
        use_pcp=False,
        enable_c8_quant=False,
        sliding_window=None,
        sinks=None,
        kv_sharing_target_layer_name=None,
    )
    monkeypatch.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    monkeypatch.setattr(attention_v1, "is_dcp_enabled", lambda: False)
    monkeypatch.setattr(attention_v1, "notify_kv_cache_written", Mock())
    writer = Mock()
    monkeypatch.setattr(attention_v1.DeviceOperator, "reshape_and_cache", writer)
    slots = torch.tensor([-1, block - 1, block, pages * block - 1], dtype=torch.int32)
    token_kv = torch.zeros((4, 1, dim), dtype=key.dtype)
    if entrypoint == "do_kv_cache_update":
        backend.do_kv_cache_update(None, token_kv, token_kv, [key, value], slots)
    else:
        metadata = SimpleNamespace(slot_mapping=slots, num_actual_tokens=4)
        backend.reshape_and_cache(token_kv, token_kv, token_kv, [key, value], metadata, token_kv)
    args = writer.call_args.kwargs
    assert args["key_cache"].is_contiguous() and args["value_cache"].is_contiguous()
    torch.testing.assert_close(args["slot_mapping"], torch.tensor([-1, 3, 12, 75], dtype=torch.int32))


def test_dense_paged_path_is_limited_to_supported_backend(monkeypatch):
    backend = attention_v1.AscendAttentionBackendImpl.__new__(attention_v1.AscendAttentionBackendImpl)
    backend.__dict__.update(
        attn_type=attention_v1.AttentionType.DECODER,
        vllm_config=SimpleNamespace(speculative_config=None),
        use_bnsd_kv_cache=False,
        use_pcp=False,
        enable_c8_quant=False,
        sliding_window=None,
        sinks=None,
        kv_sharing_target_layer_name=None,
    )
    monkeypatch.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    monkeypatch.setattr(attention_v1, "is_dcp_enabled", lambda: False)
    assert backend._supports_dense_paged_kv()
    backend.enable_c8_quant = True
    assert not backend._supports_dense_paged_kv()

    backend.enable_c8_quant = False
    backend.__dict__.pop("vllm_config")
    assert not backend._supports_dense_paged_kv()


def test_graph_providers_resolve_dynamic_block_table_and_stable_output():
    source = torch.tensor([[1, 3]], dtype=torch.int32)
    current = torch.tensor([[4, 2]], dtype=torch.int32)
    mapped = torch.empty_like(source)
    metadata = SimpleNamespace(
        block_tables=current,
        actual_seq_lengths_q=[1],
        seq_lens_list=[1],
    )
    context = {"layer": metadata}
    table_provider = attention_v1.DensePagedBlockTableParamProvider("layer", mapped)
    fia_provider = attention_v1.DensePagedFIAParamProvider("layer", None, False, mapped)

    params = table_provider.resolve(context)
    torch.mul(params["input"], 3, out=mapped)
    assert params["input"] is current
    assert fia_provider.resolve(context)["block_table"] is mapped
    torch.testing.assert_close(mapped, torch.tensor([[12, 6]], dtype=torch.int32))
