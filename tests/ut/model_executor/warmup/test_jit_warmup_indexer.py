# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for indexer warmup registration and compile-key coverage."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config, make_registry
from vllm_ascend.ops.triton import indexer_quant_cache_int8 as int8_ops
from vllm_ascend.ops.triton import prepare_indexer_indices as pi

# The registration gates and key spaces under test are only meaningful
# with a real Triton kernel behind them; the without-device CI leg has no
# triton installed.
pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")


def test_keys_cover_both_write_lengths_variants():
    config = make_mock_vllm_config(max_num_batched_tokens=8192)
    with patch.object(pi, "get_vectorcore_num", return_value=4):
        keys = pi._PREPARE_INDEXER_INDICES_KERNEL.get_warmup_keys((8, 4, config))

    assert {key.write_lengths for key in keys} == {False, True}
    # WRITE_LENGTHS bumps BLOCK_COLS to 32, which halves the row budget the
    # runtime enforces: wide-column keys only exist up to that tighter cap.
    base_rows = pi.collect_indexer_warmup_block_rows(8, 4, 8192)
    wl_cap = pi._indexer_max_block_rows(32)
    assert len(keys) == len(base_rows) + len([r for r in base_rows if r <= wl_cap])
    assert max(key.block_rows for key in keys if key.write_lengths) == wl_cap
    for key in keys:
        assert key.topk == 8
        assert key.compress_ratio == 4
        # write_lengths widens BLOCK_COLS to the reduction width.
        assert key.block_cols == (32 if key.write_lengths else 8)
        assert key.positions_dtype == torch.int64


@pytest.mark.parametrize("page_stride", [16384, 20000, 2**31 + 32])
def test_int8_keys_cover_buckets_slot_types_and_offsets(page_stride):
    owner = int8_ops._INDEXER_QUANT_CACHE_INT8_KERNEL
    keys = owner.get_warmup_keys(
        num_slots=1024,
        page_size=128,
        key_page_stride=page_stride,
        scale_page_stride=page_stride // 2,
    )
    assert set(keys) == {
        owner.CompileKey(1024, 128, page_stride, page_stride // 2, 128, 1, 1, dtype, rows, True, aligned)
        for dtype in (torch.int32, torch.int64)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128)
        for aligned in (True, False)
    }


def test_int8_registration_uses_bound_cache_layout_without_allocation():
    key = torch.empty_strided((8, 128, 1, 128), (20000, 128, 128, 1), dtype=torch.int8)
    scale = torch.empty_strided((8, 128, 1, 1), (10000, 1, 1, 1), dtype=torch.float16)
    registry = make_registry()
    register = int8_ops.register_indexer_quant_cache_int8_warmup
    with patch.object(torch, "empty", side_effect=AssertionError("registration must be metadata-only")):
        register(key, scale)
        assert len(registry) == 0
        with registry.activate():
            register(key, scale)
            register(key, scale)
            register(key, scale[..., :0])
        assert len(registry) == 1
    args, kwargs = registry._registrations[int8_ops._INDEXER_QUANT_CACHE_INT8_KERNEL][0]
    assert not args
    assert kwargs == dict(num_slots=1024, page_size=128, key_page_stride=20000, scale_page_stride=10000)


@pytest.mark.parametrize("strided", [False, True])
def test_int8_warmup_covers_runtime_launches_for_every_count(strided):
    owner = int8_ops.IndexerQuantCacheInt8Kernel()
    kernel = Mock(arg_names=owner.kernel.arg_names)
    kernel.__getitem__ = Mock(return_value=Mock())
    owner.kernel = kernel
    key = torch.empty_strided((8, 128, 1, 128), (20000, 128, 128, 1), dtype=torch.int8)
    scale = torch.empty_strided((8, 128, 1, 1), (10000, 1, 1, 1), dtype=torch.float16)
    step = 2 if strided else 1
    x = torch.empty((128, 128 * step), dtype=torch.bfloat16)[:, ::step]
    keys = owner.get_warmup_keys(
        num_slots=1024,
        page_size=128,
        key_page_stride=20000,
        scale_page_stride=10000,
        x_row_stride=x.stride(0),
        x_col_stride=x.stride(1),
        slot_stride=step,
    )
    for compile_key in keys:
        owner.compile(compile_key)
    assert kernel.warmup.call_count == 32
    kernel.__getitem__.assert_not_called()
    assert all(call.kwargs["grid"] == (1,) for call in kernel.warmup.call_args_list)

    def signature(kwargs):
        return {
            name: (value.dtype, value.data_ptr() % 16 == 0) if name.endswith("_ptr") else value
            for name, value in kwargs.items()
            if name not in ("grid", "num_tokens")
        }

    warmed = [signature(call.kwargs) for call in kernel.warmup.call_args_list]
    with patch.object(int8_ops, "_INDEXER_QUANT_CACHE_INT8_KERNEL", owner):
        for dtype in (torch.int32, torch.int64):
            storage = torch.empty(128 * step + 1, dtype=dtype)
            for offset in (0, 1):
                slots = storage[offset : offset + 128 * step : step]
                for count in range(1, 129):
                    int8_ops.indexer_quant_cache_int8(key, scale, x[:count], slots[:count])
                    kernel.__getitem__.assert_called_with((count,))
                    kwargs = kernel.__getitem__.return_value.call_args.kwargs
                    assert kwargs["num_tokens"] == count
                    assert signature(kwargs) in warmed
        kernel.__getitem__.reset_mock()
        int8_ops.indexer_quant_cache_int8(key, scale, x[:0], slots[:0])
        kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "enabled,pcp,bound", [(True, False, True), (False, False, True), (True, True, True), (True, False, False)]
)
def test_int8_owner_registers_only_eligible_bound_cache(enabled, pcp, bound):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/indexer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAIndexerBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "register_k_cache_warmup")
    register = Mock()
    namespace = {
        "HAS_TRITON": True,
        "indexer_quant_cache_int8": int8_ops.indexer_quant_cache_int8,
        "register_indexer_quant_cache_int8_warmup": register,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    cache = (object(), object()) if bound else ()
    owner = SimpleNamespace(
        _fused_k_cache=int8_ops.indexer_quant_cache_int8 if enabled else None,
        _pcp_active=pcp,
        k_cache=SimpleNamespace(kv_cache=cache),
    )
    namespace["register_k_cache_warmup"](owner)
    if enabled and not pcp and bound:
        register.assert_called_once_with(*cache)
    else:
        register.assert_not_called()
