# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention import indexer
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.triton.small_indexer_row_store import _supports_layout


@pytest.mark.parametrize("family", list(AscendDeviceType))
def test_hardware_support(family):
    assert get_hardware_profile(family).supports(HardwareCapability.SMALL_INDEXER_ROW_STORE) == (
        family == AscendDeviceType.A3
    )


@pytest.mark.parametrize(
    "rows,width,expected",
    [(0, 128, False), (1, 128, True), (32, 2048, True), (33, 128, False), (2, 17, False), (1, 2064, False)],
)
def test_layout_limits(rows, width, expected):
    cache = torch.empty(2, 128, 1, width, dtype=torch.bfloat16)
    updates = torch.empty(rows, width + 16, dtype=cache.dtype)[:, :width]
    slots = torch.empty(rows, dtype=torch.int64)
    assert _supports_layout(cache, updates, slots) == expected


def test_invalid_layouts_and_address_limits():
    cache = torch.empty(3, 128, 1, 128, dtype=torch.float16, device="meta")
    updates = cache.view(-1, 128)[:2]
    slots = torch.empty(2, dtype=torch.int32, device="meta")
    assert _supports_layout(cache, updates, slots)
    for dest, keys, indices in (
        (cache[1:], updates, slots),
        (cache[:, ::2], updates, slots),
        (cache[..., ::2], updates[:, ::2], slots),
        (cache, updates.bfloat16(), slots),
        (cache, updates, slots.float()),
        (cache, updates, slots[:1]),
        (cache, updates, slots.view(-1, 1)),
        (cache, updates.as_strided((2, 128), (2**31, 1)), slots),
        (torch.empty(2**24, 128, 1, 128, dtype=cache.dtype, device="meta"), updates, slots),
    ):
        assert not _supports_layout(dest, keys, indices)


@pytest.mark.parametrize(
    "enabled,capturing,draft,mode,layout,used",
    [
        (True, True, False, CUDAGraphMode.FULL, True, True),
        (True, True, False, CUDAGraphMode.NONE, True, True),
        (True, False, False, CUDAGraphMode.NONE, True, False),
        (True, True, False, CUDAGraphMode.FULL, False, False),
        (True, True, True, CUDAGraphMode.FULL, True, False),
        (True, False, False, CUDAGraphMode.FULL, True, False),
        (True, True, False, CUDAGraphMode.PIECEWISE, True, False),
        (False, True, False, CUDAGraphMode.FULL, True, False),
    ],
)
def test_write_cache_dispatch(monkeypatch, enabled, capturing, draft, mode, layout, used):
    cache = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    keys = torch.zeros(2, 128, dtype=cache.dtype)
    slots = torch.tensor([0, -1])
    coordinates = torch.tensor([[0, 0], [-1, 127]])
    fallback, small = Mock(), Mock(return_value=layout)
    layer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=(cache,)),
        enable_sparse_li_quant=False,
        _use_small_indexer_rows=enabled,
        _write_cache_tensor=fallback,
    )
    monkeypatch.setattr(indexer, "_EXTRA_CTX", SimpleNamespace(capturing=capturing, is_draft_model=draft))
    monkeypatch.setattr(indexer, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(indexer, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(indexer, "try_store_small_indexer_rows", small)
    indexer.AscendSFAIndexerBackend.write_cache(layer, keys, None, slots, coordinates)
    assert small.called == (enabled and capturing and not draft and mode in (CUDAGraphMode.FULL, CUDAGraphMode.NONE))
    assert fallback.called != used
    if fallback.called:
        assert fallback.call_args.args[0] is cache
        assert fallback.call_args.args[3] is coordinates
