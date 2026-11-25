# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import (
    DEVICE_BUFFER_ALIGNMENT,
    BufferDescriptor,
    MemoryLocation,
    buffer_arrays,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import (
    HiSparseTransferLayout,
    has_hisparse_cache_layout,
    is_hisparse_ascendstore_enabled,
)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"address": 0}, "address and size"),
        ({"size_bytes": 0}, "address and size"),
        ({"alignment_bytes": 0}, "power of two"),
        ({"alignment_bytes": 3}, "power of two"),
        ({"address": 129, "alignment_bytes": 128}, "aligned to 128"),
    ],
)
def test_invalid_memory_geometry_fails_before_registration(kwargs, message):
    values = {"address": 128, "size_bytes": 32, **kwargs}
    with pytest.raises(ValueError, match=message):
        BufferDescriptor(**values)


def test_layout_preserves_exact_host_range_and_aligned_device_prefix():
    device = DEVICE_BUFFER_ALIGNMENT
    layout = HiSparseTransferLayout(10, frozenset({"source"}), frozenset({17}), frozenset({0}))
    buffers = layout.buffer_descriptors({17: (19, 119), device: (device, device + 128)})
    assert buffers == [
        BufferDescriptor(address=19, size_bytes=100, memory_location=MemoryLocation.HOST),
        BufferDescriptor(address=device, size_bytes=128, alignment_bytes=DEVICE_BUFFER_ALIGNMENT),
    ]
    assert buffer_arrays(buffers) == ([19, device], [100, 128])
    assert layout.get_host_num_blocks("source") == 10
    assert layout.get_host_num_blocks("indexer") is None
    with pytest.raises(FrozenInstanceError):
        buffers[0].__setattr__("address", device)


@pytest.mark.parametrize(
    "host_blocks, role, expected", [(None, None, False), (0, None, True), (None, "hisparse_source", True)]
)
def test_layout_predicate_agrees_before_and_after_host_pool_sizing(host_blocks, role, expected):
    config = SimpleNamespace(hisparse_host_num_blocks=host_blocks, kv_cache_groups=[SimpleNamespace(role=role)])
    assert has_hisparse_cache_layout(config) is expected
    for with_store in (False, True):
        transfer = SimpleNamespace(
            has_connector=lambda name, enabled=with_store: enabled and name == "AscendStoreConnector"
        )
        assert is_hisparse_ascendstore_enabled(SimpleNamespace(kv_transfer_config=transfer), config) is (
            expected and with_store
        )
    assert not is_hisparse_ascendstore_enabled(SimpleNamespace(kv_transfer_config=None), config)
    assert not has_hisparse_cache_layout(None)
