# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real tensor descriptors; requires Core's upstream boundary-state Store API."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout")

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import KeyMetadata

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_connector import (
    AscendHashStoreWorker,
    AscendStoreConnector,
    MooncakeStoreWorker,
    _AlignedStore,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_layout import AscendHashStoreLayout


def layout(*, state=False, hnd=False, page=16, unit=4):
    return AscendHashStoreLayout(KeyMetadata("test", 0, 0, 0, 0), page, unit, is_state=state, head_major=hnd)


def payload(descriptor):
    addresses, sizes, _ = descriptor
    return b"".join(ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0]))


def put_payload(descriptor, data):
    addresses, sizes, _ = descriptor
    cursor = 0
    for address, size in zip(addresses[0], sizes[0]):
        ctypes.memmove(address, data[cursor : cursor + size], size)
        cursor += size
    assert cursor == len(data)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_partial_roundtrip(hnd, scale, dtype):
    # Different physical padding on each end. Only [4,12) of block 1
    # may change; block 0, adjacent tokens, layers and padding are guards.
    shape = (3, scale, 2, 16 // scale, 3) if hnd else (3, scale, 16 // scale, 2, 3)
    source_backing = torch.full((3, scale, 128), -9, dtype=dtype)
    target_backing = torch.full((3, scale, 160), -7, dtype=dtype)
    within = (16 // scale) * 6
    source = source_backing[..., :within].view(shape)
    target = target_backing[..., :within].view(shape)
    source.copy_(torch.arange(source.numel()).view(shape))
    before = target_backing.clone()
    expected = target.clone()
    token_axis = 3 if hnd else 2
    for kernel in range(scale):
        for token in range(16 // scale):
            logical = kernel * (16 // scale) + token
            if 4 <= logical < 12:
                index = [1, kernel, slice(None), slice(None), slice(None)]
                index[token_axis] = token
                expected[tuple(index)] = source[tuple(index)]
    src_layout, dst_layout = layout(hnd=hnd), layout(hnd=hnd)
    src_layout.register_kv_caches([source], 3)
    dst_layout.register_kv_caches([target], 3)
    src = src_layout.prepare_values([(4, 12)], [1], [0])
    dst = dst_layout.prepare_values([(4, 12)], [1], [0])
    put_payload(dst, payload(src))
    assert torch.equal(target, expected)
    assert torch.equal(target_backing[..., within:], before[..., within:])


def test_checkpoint_is_complete_not_proportional():
    backing = torch.full((3, 80), -1.0)
    conv = backing[:, :24].view(3, 3, 8)
    state = backing[:, 24:40].view(3, 4, 4)
    conv[1].fill_(2)
    state[1].fill_(3)
    cache = layout(state=True)
    cache.register_kv_caches([conv, state], 3)
    descriptor = cache.prepare_values([(0, 4)], [1], [0])
    assert sum(descriptor[1][0]) == 40 * 4
    assert payload(descriptor) == conv[1].numpy().tobytes() + state[1].numpy().tobytes()


@pytest.mark.parametrize("hnd", [False, True])
def test_ungrouped_attention_with_token_padding(hnd):
    shape = (3, 2, 16, 3) if hnd else (3, 16, 2, 3)
    strides = (256, 80, 5, 1) if hnd else (256, 8, 3, 1)
    backing = torch.full((3 * 256,), -1.0)
    view = torch.as_strided(backing, shape, strides)
    view.copy_(torch.arange(view.numel()).reshape(shape))
    cache = layout(hnd=hnd)
    cache.register_kv_caches([view], 3)
    descriptor = cache.prepare_values([(4, 12)], [1], [0])
    expected = view[1, :, 4:12, :] if hnd else view[1, 4:12, :, :]
    assert payload(descriptor) == expected.contiguous().numpy().tobytes()


@pytest.mark.parametrize(
    "chunks,blocks,shards",
    [([(1, 4)], [1], [0]), ([(12, 20)], [1, 2], [0]), ([(0, 4)], [0], [0]), ([(0, 4)], [3], [0]), ([(0, 4)], [1], [1])],
)
def test_invalid_transfer_rejected(chunks, blocks, shards):
    cache = layout()
    cache.register_kv_caches([torch.zeros(3, 16, 2, 3)], 3)
    with pytest.raises(ValueError):
        cache.prepare_values(chunks, blocks, shards)


def test_overlapping_view_rejected():
    cache = layout()
    with pytest.raises(ValueError, match="Overlapping"):
        cache.register_kv_caches([torch.as_strided(torch.zeros(1000), (3, 16, 2, 3), (1, 6, 3, 1))], 3)


def test_aligned_registration_stays_inside_allocation():
    store = Mock()
    store.register_buffer.return_value = 0
    storage = SimpleNamespace(data_ptr=lambda: 1003, nbytes=lambda: 400)
    view = SimpleNamespace(
        untyped_storage=lambda: storage,
        numel=lambda: 10,
        data_ptr=lambda: 1040,
        shape=(2, 5),
        stride=lambda: (20, 1),
        element_size=lambda: 4,
    )
    wrapper = _AlignedStore(store, [view, view], alignment=16)
    wrapper.register_buffer(1003, 400)
    store.register_buffer.assert_called_once_with(1008, 395)
    wrapper.unregister_buffer(1003)
    store.unregister_buffer.assert_called_once_with(1008)
    view.data_ptr = lambda: 1004
    with pytest.raises(ValueError, match="aligned registration"):
        _AlignedStore(store, [view], alignment=16)


def test_registration_failure_is_fatal():
    store = Mock()
    store.register_buffer.return_value = -1
    view = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 16, nbytes=lambda: 32),
        numel=lambda: 1,
        data_ptr=lambda: 16,
        shape=(1,),
        stride=lambda: (1,),
        element_size=lambda: 4,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        _AlignedStore(store, [view], alignment=16).register_buffer(16, 32)


def test_mrv2_registration_uses_core_capacity_without_mutating_config():
    worker = object.__new__(AscendHashStoreWorker)
    worker._capacity_only = False
    original = worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker._kv_cache_config = SimpleNamespace(num_blocks=3)
    worker._kv_cache_groups = []
    worker.store = Mock()
    with patch.object(MooncakeStoreWorker, "register_kv_caches") as register:
        worker.register_kv_caches({})
        register.assert_called_once_with({})
    assert worker.cache_config.num_gpu_blocks == 3
    assert original.num_gpu_blocks is None
    worker.cache_config.num_gpu_blocks = 4
    with pytest.raises(ValueError, match="capacity disagrees"):
        worker.register_kv_caches({})


def test_npu_event_shared_by_all_save_jobs():
    worker = object.__new__(AscendHashStoreWorker)
    worker._capacity_only, worker.can_put = False, True
    worker.kv_send_thread = Mock()
    requests = [SimpleNamespace(can_save=value) for value in (True, False, True)]
    with patch.object(torch, "npu", create=True) as npu:
        worker.wait_for_save(SimpleNamespace(requests=requests))
        npu.Event.assert_called_once()
        npu.Event.return_value.record.assert_called_once()
    assert worker.kv_send_thread.add_request.call_count == 2
    assert requests[0].current_event is requests[2].current_event


@pytest.mark.parametrize(
    "extra",
    [
        {"backend": "memcache"},
        {"use_layerwise": True},
        {"load_async": False},
        {"consumer_is_to_put": True},
        {"store_tp_size": 2},
        {"enable_store_tp_lcm": True},
        {"store_chunk_size": 16},
    ],
)
def test_hash_mode_rejects_unsupported_configuration(extra):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"pool_prefix_mode": "hash", **extra})
    )
    with pytest.raises(ValueError, match="rank-local Mooncake"):
        AscendStoreConnector(config, None)


def test_non_hash_entrypoint_rejected():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"pool_prefix_mode": "block_only"})
    )
    with pytest.raises(ValueError, match="original module"):
        AscendStoreConnector(config, None)


def test_uncovered_registration_span_rejected():
    view = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 16, nbytes=lambda: 32),
        numel=lambda: 9,
        data_ptr=lambda: 16,
        shape=(9,),
        stride=lambda: (1,),
        element_size=lambda: 4,
    )
    with pytest.raises(ValueError, match="exceeds its storage"):
        _AlignedStore(Mock(), [view], alignment=16)
