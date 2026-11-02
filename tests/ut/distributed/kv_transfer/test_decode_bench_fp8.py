# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorScheduler,
    DecodeBenchConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVQuantMode

from vllm_ascend.distributed.kv_transfer import decode_bench_connector as connector
from vllm_ascend.distributed.kv_transfer.decode_bench_connector import (
    AscendDecodeBenchConnector,
    AscendDecodeBenchConnectorWorker,
)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
SEED = 1024


class _RejectFp8Writes(TorchDispatchMode):
    """Model a backend without FP8 inplace writes; all other ops run normally."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in ("aten::index_put_", "aten::fill_", "aten::copy_") and args[0].dtype in FP8_DTYPES:
            raise RuntimeError("FP8 tensor write not supported")
        return func(*args, **(kwargs or {}))


def _worker():
    return object.__new__(AscendDecodeBenchConnectorWorker)


def _reference(size, dtype, mean, std):
    worker = object.__new__(DecodeBenchConnectorWorker)
    return worker._make_fill_values(size, torch.device("cpu"), dtype, mean, std).view(torch.uint8)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("method", ["_fill_block_tensor", "_fill_tensor"])
@pytest.mark.parametrize("std", [0.0, 0.2])
def test_reproduces_unsupported_fp8_writes_and_fixes_them(dtype, method, std):
    # Simulate only the unsupported kernel boundary on CPU. Use the actual
    # upstream and adapted implementations, not a replacement filling algorithm.
    raw = torch.full((4, 2, 3), 42, dtype=torch.uint8)
    expected = raw.clone()
    args = ([1, 3], 0.015, std, dtype) if method == "_fill_block_tensor" else (0.015, std, dtype)
    shape = (2, 2, 3) if method == "_fill_block_tensor" else raw.shape
    selected = [1, 3] if method == "_fill_block_tensor" else slice(None)
    torch.manual_seed(SEED)
    expected[selected] = _reference(shape, dtype, 0.015, std)
    with _RejectFp8Writes(), pytest.raises(RuntimeError, match="FP8 tensor write not supported"):
        getattr(object.__new__(DecodeBenchConnectorWorker), method)(raw.clone().view(dtype), *args)
    torch.manual_seed(SEED)
    with _RejectFp8Writes():
        getattr(_worker(), method)(raw.view(dtype), *args)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("mean,std", [(0.015, 0.0), (1e12, 0.0), (-1e12, 0.0), (0.5, 0.2), (1e8, 1e10)])
def test_fp8_encoding_matches_upstream_clamp_and_sampling(dtype, mean, std):
    torch.manual_seed(SEED)
    expected = _reference((3, 4), dtype, mean, std)
    torch.manual_seed(SEED)
    actual = _worker()._make_fill_values((3, 4), torch.device("cpu"), dtype, mean, std)
    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected)
    assert torch.isfinite(actual.float()).all()


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("byte_storage", [False, True])
@pytest.mark.parametrize("std", [0.0, 0.2])
def test_fp8_block_writes_preserve_neighbours_and_padding(dtype, byte_storage, std):
    raw = torch.full((6, 3, 7), 42, dtype=torch.uint8)
    view = raw[:, :2, :5]
    cache = view if byte_storage else view.view(dtype)
    expected = raw.clone()
    torch.manual_seed(SEED)
    expected[[1, 4], :2, :5] = _reference((2, 2, 5), dtype, 0.015, std)
    torch.manual_seed(SEED)
    _worker()._fill_block_tensor(cache, [1, 4, 99], 0.015, std, dtype)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("byte_storage", [False, True])
@pytest.mark.parametrize("std", [0.0, 0.2])
def test_fp8_whole_view_fill_preserves_padding_and_bounds_buffers(monkeypatch, dtype, byte_storage, std):
    raw = torch.full((6, 3, 7), 42, dtype=torch.uint8)
    view = raw[:, :2, :5]
    cache = view if byte_storage else view.view(dtype)
    expected = raw.clone()
    # Force two rows per chunk; this also checks the random startup-fill path.
    monkeypatch.setattr(connector, "_FP8_FILL_CHUNK_ELEMENTS", 20)
    torch.manual_seed(SEED)
    if std:
        for start in range(0, 6, 2):
            expected[start : start + 2, :2, :5] = _reference((2, 2, 5), dtype, 0.015, std)
    else:
        expected[:, :2, :5] = _reference((), dtype, 0.015, std).item()
    torch.manual_seed(SEED)
    _worker()._fill_tensor(cache, 0.015, std, dtype)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (3, 0)])
@pytest.mark.parametrize("std", [0.0, 0.2])
def test_fp8_scalar_and_empty_views(dtype, shape, std):
    raw = torch.full(shape, 42, dtype=torch.uint8)
    torch.manual_seed(SEED)
    expected = _reference(shape, dtype, 0.015, std)
    torch.manual_seed(SEED)
    _worker()._fill_tensor(raw.view(dtype), 0.015, std, dtype)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_empty_or_unallocated_block_selection_does_not_write(dtype):
    raw = torch.full((4, 2, 3), 42, dtype=torch.uint8)
    before = raw.clone()
    for ids in ([], [99]):
        _worker()._fill_block_tensor(raw.view(dtype), ids, 0.015, 0.0, dtype)
    assert torch.equal(raw, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.uint8])
@pytest.mark.parametrize("std", [0.0, 0.2])
def test_non_fp8_fill_behaviour_matches_upstream(dtype, std):
    base = object.__new__(DecodeBenchConnectorWorker)
    for method in ("_fill_block_tensor", "_fill_tensor"):
        actual = torch.full((4, 2, 3), 3, dtype=dtype)
        expected = actual.clone()
        args = ([1, 3], 0.015, std, dtype) if method == "_fill_block_tensor" else (0.015, std, dtype)
        torch.manual_seed(SEED)
        getattr(base, method)(expected, *args)
        torch.manual_seed(SEED)
        getattr(_worker(), method)(actual, *args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("byte_storage", [False, True])
def test_fp8_only_cache_activates_worker_without_changing_scheduler(dtype, byte_storage):
    # Regression: the circular-only adapter previously returned the upstream
    # worker for FP8-only configurations, so none of its NPU fill fixes ran.
    quant = KVQuantMode.FP8_PER_TENSOR if byte_storage else KVQuantMode.NONE
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.uint8 if byte_storage else dtype, kv_quant_mode=quant
    )
    cache = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=spec)],
    )
    cache_dtype = "fp8_e5m2" if dtype == torch.float8_e5m2 else "fp8_e4m3"
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda name, default: default),
    )
    worker = AscendDecodeBenchConnector(config, KVConnectorRole.WORKER, cache)
    scheduler = AscendDecodeBenchConnector(config, KVConnectorRole.SCHEDULER, cache)
    assert type(worker.connector_worker) is AscendDecodeBenchConnectorWorker
    assert type(scheduler.connector_scheduler) is DecodeBenchConnectorScheduler
