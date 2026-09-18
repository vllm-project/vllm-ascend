# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention import cache_store
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl, AscendSFADSACPMetadataBuilder
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def _metadata():
    return SimpleNamespace(
        block_size=128,
        group_len=torch.ones(2048, dtype=torch.int32),
        group_key_idx=torch.arange(2048, dtype=torch.int32),
        group_key_cache_idx=torch.arange(2048, dtype=torch.int32) + 2048,
    )


@pytest.mark.parametrize("device", list(AscendDeviceType))
@pytest.mark.parametrize("tokens", [0, 33, 2047, 2048])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32])
def test_block_store_eligibility(monkeypatch, device, tokens, dtype):
    monkeypatch.setattr(cache_store, "get_current_hardware_profile", lambda: get_hardware_profile(device))
    key = torch.empty(tokens, 128, dtype=dtype)
    cache = torch.empty(32, 128, 1, 128, dtype=dtype)
    metadata = _metadata()
    operation = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "store_kv_block", operation, raising=False)
    expected = device in (AscendDeviceType.A2, AscendDeviceType.A3) and tokens >= 2048 and dtype != torch.float32
    assert cache_store.try_store_kv_blocks(key, cache, metadata) == expected
    assert operation.call_count == int(expected)
    if expected:
        args = operation.call_args.args
        assert args[0] is key and args[1] is cache
        assert args[2] is metadata.group_len
        assert args[4] is metadata.group_key_cache_idx


@pytest.mark.parametrize("invalid", ["metadata", "stride", "dtype", "block_size", "scratch_size"])
def test_block_store_falls_back_for_unsupported_cache(monkeypatch, invalid):
    monkeypatch.setattr(cache_store, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    key = torch.empty(2048, 1024, dtype=torch.bfloat16)
    cache = torch.empty(32, 64, 1, 1024, dtype=torch.bfloat16)
    metadata = _metadata()
    metadata.block_size = 64
    if invalid == "metadata":
        metadata.group_key_cache_idx = None
    elif invalid == "stride":
        cache = cache[..., ::2]
        key = key[:, ::2]
    elif invalid == "dtype":
        cache = cache.to(torch.float16)
    elif invalid == "block_size":
        metadata.block_size = 0
    elif invalid == "scratch_size":
        metadata.block_size = 128
    operation = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "store_kv_block", operation, raising=False)
    assert not cache_store.try_store_kv_blocks(key, cache, metadata)
    operation.assert_not_called()


def test_main_cache_groups_use_own_full_slots_before_cp_slice(monkeypatch):
    builder = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
    builder.use_block_cache_store = True
    builder.kernel_block_size = 128
    builder.dsa_cp_actual_seq_lengths_query = torch.zeros(2, dtype=torch.int32)
    builder.dsa_cp_actual_seq_lengths_key = torch.zeros(2, dtype=torch.int32)
    actual, padded = 2049, 2056
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=padded,
        num_actual_tokens=actual,
        query_start_loc=torch.tensor([0, actual], dtype=torch.int32),
    )
    slots = torch.arange(padded, dtype=torch.int32) + 8192
    slots[actual:] = -1
    groups = {
        name: torch.empty(actual, dtype=torch.int32) for name in ("group_len", "group_key_idx", "group_key_cache_idx")
    }
    build_groups = Mock(return_value=groups)
    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
            return_value=SimpleNamespace(world_size=8, rank_in_group=7),
        ),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.build_block_cache_groups", build_groups),
    ):
        _, _, _, extra = builder._prepare_parallel_metadata(
            common,
            torch.zeros(padded, 1, 1, 64),
            torch.zeros(padded, 1, 1, 64),
            slots,
            torch.tensor([actual]),
            torch.tensor([actual]),
            None,
        )
    assert build_groups.call_count == 1
    torch.testing.assert_close(build_groups.call_args.args[0], slots[:actual])
    assert build_groups.call_args.args[1] == 128
    assert extra["group_key_cache_idx"] is groups["group_key_cache_idx"]


@pytest.mark.parametrize("use_blocks", [False, True])
def test_main_cache_write_uses_actual_rows_and_own_metadata(use_blocks):
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.enable_sparse_sfa_c8 = True
    key = torch.empty(2056, 656, dtype=torch.int8)
    cache = torch.empty(32, 128, 1, 656, dtype=torch.int8)
    metadata = _metadata()
    metadata.num_actual_tokens = 2049
    slots = torch.arange(2056, dtype=torch.int32) + 4096
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.try_store_kv_blocks", return_value=use_blocks) as store,
        patch("torch_npu.npu_scatter_nd_update_", create=True) as scatter,
    ):
        impl._store_parallel_kv(None, None, None, key, [], (cache,), slots, metadata, False)
    assert store.call_args.args[0].shape == (2049, 656)
    assert store.call_args.args[1] is cache
    assert store.call_args.args[2] is metadata
    assert scatter.call_count == int(not use_blocks)
    if not use_blocks:
        torch.testing.assert_close(scatter.call_args.args[1].flatten(), slots[:2049])


@pytest.mark.parametrize("use_blocks", [False, True])
def test_indexer_cache_write_uses_own_metadata(use_blocks):
    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(indexer)
    indexer.enable_sparse_li_c8 = False
    indexer._dsa_cp_active = True
    indexer._pcp_active = False
    key = torch.empty(2048, 128, dtype=torch.bfloat16)
    cache = torch.empty(32, 128, 1, 128, dtype=torch.bfloat16)
    indexer.k_cache = SimpleNamespace(kv_cache=(cache,))
    metadata = _metadata()
    slots = torch.arange(2048, dtype=torch.int32) + 2048
    with (
        patch("vllm_ascend.attention.indexer.try_store_kv_blocks", return_value=use_blocks) as store,
        patch("torch_npu.npu_scatter_nd_update_", create=True) as scatter,
    ):
        indexer.write_cache(key, None, slots, indexer_attn_metadata=metadata)
    assert store.call_args.args[0] is key
    assert store.call_args.args[1] is cache
    assert store.call_args.args[2] is metadata
    assert scatter.call_count == int(not use_blocks)
    if not use_blocks:
        torch.testing.assert_close(scatter.call_args.args[1].flatten(), slots)
