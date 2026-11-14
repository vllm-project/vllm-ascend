# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.hisparse.layout import get_hisparse_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    KVCacheTensor,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.hisparse.backend import AscendHiSparseBackend
from vllm_ascend.hisparse.binding import align_hisparse_slab_views, reserve_hisparse_slab_alignment

SLAB_ALIGNMENT = 2 * 1024 * 1024


def aligned_tensor(size: int) -> torch.Tensor:
    backing = torch.empty(size + SLAB_ALIGNMENT, dtype=torch.int8)
    offset = -backing.data_ptr() % SLAB_ALIGNMENT
    return backing[offset : offset + size]


@pytest.mark.parametrize("misalignment", [0, 1, SLAB_ALIGNMENT - 1])
@pytest.mark.parametrize("indexer_c8", [False, True])
def test_hisparse_groups_register_with_independent_host_blocks(indexer_c8: bool, misalignment: int) -> None:
    block_size = 128
    num_device_blocks = 4
    num_host_blocks = 5
    source_name = "model.layers.0.self_attn"
    indexer_name = "model.layers.0.indexer.k_cache"
    source_spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_role=SparseCacheRole.SPARSE,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn if indexer_c8 else torch.bfloat16,
        cache_role=SparseCacheRole.INDEXER,
        cache_sparse_li_c8=indexer_c8,
        scale_dim=1 if indexer_c8 else 0,
        scale_dtype=torch.float32 if indexer_c8 else torch.int8,
    )
    hf_text_config = SimpleNamespace(index_topk=128)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            hisparse_config=SimpleNamespace(device_buffer_size=256, eager_host_mirror=True),
        ),
        model_config=SimpleNamespace(hf_config=hf_text_config, hf_text_config=hf_text_config),
        speculative_config=None,
    )
    groups = get_hisparse_kv_cache_groups(config, {source_name: source_spec, indexer_name: indexer_spec})
    assert groups is not None
    resident_name = groups[2].layer_names[0]
    hot_name = groups[3].layer_names[0]
    indexer_stride = 17408 if indexer_c8 else 98304
    indexer_offset = 128 if indexer_c8 else 32768
    size = num_device_blocks * indexer_stride
    host_tensor = KVCacheTensor(
        size=num_host_blocks * 32768, layers=[source_name], layer_stride=0, block_stride=32768, host_resident=True
    )
    device_tensors = [
        KVCacheTensor(size, [indexer_name], indexer_spec.page_size_bytes, indexer_stride, offset=indexer_offset),
        KVCacheTensor(size, [resident_name], 0, indexer_stride),
        KVCacheTensor(size, [hot_name], 0, indexer_stride),
    ]
    cache_config = KVCacheConfig(
        num_device_blocks, [host_tensor, *device_tensors], groups, hisparse_host_num_blocks=num_host_blocks
    )
    source = aligned_tensor(num_host_blocks * 32768).view(torch.bfloat16).view(num_host_blocks, 128, 1, 128)
    padded_size = size + SLAB_ALIGNMENT - 1
    backing = aligned_tensor(padded_size + misalignment)
    storage = backing.untyped_storage()
    begin = backing.storage_offset() + misalignment
    owner = torch.empty(0, dtype=torch.int8).set_(storage[begin : begin + padded_size], 0, (padded_size,), (1,))
    assert owner.data_ptr() % SLAB_ALIGNMENT == misalignment
    context = {source_name: SimpleNamespace(), indexer_name: SimpleNamespace()}
    with (
        reserve_hisparse_slab_alignment(cache_config),
        patch("vllm.v1.worker.utils.torch.zeros", return_value=owner) as allocate,
    ):
        assert [tensor.size for tensor in cache_config.kv_cache_tensors] == [host_tensor.size, *([padded_size] * 3)]
        device_config = replace(cache_config, kv_cache_tensors=cache_config.kv_cache_tensors[1:])
        caches = allocate_kv_cache(device_config, torch.device("cpu"), KVCacheLayout.BLHNC)
        caches[source_name] = source
        AscendHiSparseBackend().prepare_kv_cache_views(cache_config, caches, context)
        allocate.assert_called_once_with(padded_size, dtype=torch.int8, device=torch.device("cpu"))
    raw = caches[resident_name]
    assert raw.data_ptr() == owner.data_ptr() + (-owner.data_ptr() % SLAB_ALIGNMENT)
    assert raw.storage_offset() == 0
    assert raw.untyped_storage().nbytes() == raw.numel() == size
    assert caches[source_name] is source
    for view in (caches[hot_name], *caches[indexer_name]):
        assert view.untyped_storage().data_ptr() == raw.data_ptr()
    assert caches[indexer_name][0].data_ptr() == raw.data_ptr() + indexer_offset
    assert [tensor.size for tensor in cache_config.kv_cache_tensors] == [host_tensor.size, *([size] * 3)]
    # The storage is an alias of the Core allocation, with no copy.
    raw[0] = 37
    assert owner[-owner.data_ptr() % SLAB_ALIGNMENT].item() == 37
    del caches[hot_name]

    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.use_recoverssm = False
    worker.kv_cache_config = cache_config
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    worker.use_kvpp = False
    worker.use_hybrid = True
    worker.use_layerwise = False
    worker.use_mla = True
    worker.use_sparse = True
    worker.tp_mismatch = False
    worker.num_kv_cache_groups = len(groups)
    worker.num_layers = 1
    worker.hf_config = SimpleNamespace(num_hidden_layers=1)
    worker.kv_cache_group_families = ["default"] * len(groups)
    worker._global_to_local_layer = {}
    worker._extra_config = {}
    worker.grouped_block_size = [128] * len(groups)
    worker.cache_transfer_granularity = 128
    worker.group_uses_align_state = [False] * len(groups)
    worker.dcp_size = 1
    worker.token_database = Mock()
    worker.m_store = Mock()
    worker._start_kv_transfer_threads = Mock()

    worker.register_kv_caches(caches)
    assert worker.group_block_len[0] == [32768]
    assert worker.group_block_stride[0] == [32768]
    assert worker.group_block_len[1] == ([16384, 512] if indexer_c8 else [32768])
    assert worker.group_block_stride[1] == ([17408, 17408] if indexer_c8 else [98304])
    assert all(worker.group_block_len[index] == [] for index in range(2, len(groups)))
    worker.m_store.register_buffer.assert_called_once()
    ptrs, lengths = worker.m_store.register_buffer.call_args.args
    assert worker.m_store.register_buffer.call_args.kwargs == {"check_result": True}
    assert lengths[ptrs.index(source.data_ptr())] == num_host_blocks * 32768
    assert worker._hisparse_host_resident_groups == frozenset({0})
    if indexer_c8:
        assert lengths[ptrs.index(raw.data_ptr())] == indexer_offset + (num_device_blocks - 1) * 17408 + 16896
    worker.kv_cache_config.kv_cache_tensors = [
        KVCacheTensor(size=5 * 32768, layers=[source_name], layer_stride=0, block_stride=32768, host_resident=True),
        KVCacheTensor(
            size=raw.numel(),
            layers=[indexer_name],
            layer_stride=0,
            block_stride=17408 if indexer_c8 else 98304,
        ),
    ]
    layout_id = worker._hisparse_layout_id()
    worker.kv_cache_config.num_blocks += 1
    worker.kv_cache_config.hisparse_host_num_blocks += 1
    assert worker._hisparse_layout_id() == layout_id
    worker.kv_cache_config.kv_cache_tensors[1].block_stride += 512
    assert worker._hisparse_layout_id() != layout_id
    worker.kv_cache_config.kv_cache_tensors[1].block_stride -= 512
    spec = groups[1].kv_cache_spec
    replacement = replace(indexer_spec, block_size=64)
    groups[1].kv_cache_spec = (
        replace(spec, block_size=64, kv_cache_specs={indexer_name: replacement})
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else replacement
    )
    assert worker._hisparse_layout_id() != layout_id


def test_hisparse_alignment_is_scoped_to_the_padded_allocation():
    tensor = KVCacheTensor(128, ["cache"], 0, 128)
    config = KVCacheConfig(1, [tensor], [])
    original = torch.zeros(128, dtype=torch.int8)
    caches = {"cache": original}
    align_hisparse_slab_views(config, caches)
    assert caches["cache"] is original
    with pytest.raises(ValueError, match="reserved alignment padding"), reserve_hisparse_slab_alignment(config):
        align_hisparse_slab_views(config, caches)
    align_hisparse_slab_views(config, caches)
    assert caches["cache"] is original


def test_memcache_splits_host_and_device_batches_without_reordering(caplog) -> None:
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend.store = Mock()
    backend._store_initialized = True
    backend._lazy_init = False
    backend._pending_buffers = None
    backend.store.batch_get_into_layers.side_effect = [[0, -7], [0]]

    keys = ["host", "device1", "device2"]
    addresses = [[1100], [1100], [4000]]
    sizes = [[100], [100], [100]]
    backend.store.batch_put_from_layers.return_value = [0]
    assert backend.put(keys[:1], addresses[:1], sizes[:1], host=True) == [True]
    backend.store.batch_put_from_layers.assert_called_once_with(["host"], [[1100]], [[100]], MmcDirect.COPY_H2G.value)
    assert backend.get(keys, addresses, sizes, host_keys=[True, False, False]) == [0, 0, -7]
    assert backend.store.batch_get_into_layers.call_args_list == [
        call(["device1", "device2"], [[1100], [4000]], [[100], [100]], MmcDirect.COPY_G2L.value),
        call(["host"], [[1100]], [[100]], MmcDirect.COPY_G2H.value),
    ]
    assert backend.get(keys, addresses, sizes, host_keys=[True]) is None
    assert backend.put(keys, addresses[:1], sizes, host=True) is False
    assert "Invalid HiSparse MemCache GET metadata" in caplog.text
    assert "Invalid HiSparse MemCache PUT metadata" in caplog.text


def test_memcache_bounds_each_hisparse_host_get_lease() -> None:
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend.store = Mock()
    backend._store_initialized = True
    backend._lazy_init = False
    backend.store.batch_get_into_layers.side_effect = [[0], [-3106], [0]]
    key_size = 130 * 1024 * 1024

    assert backend.get(["one", "two", "three"], [[1], [2], [3]], [[key_size]] * 3, host_keys=[True] * 3) == [
        0,
        -3106,
        0,
    ]
    assert backend.store.batch_get_into_layers.call_args_list == [
        call([key], [[index]], [[key_size]], MmcDirect.COPY_G2H.value)
        for index, key in enumerate(("one", "two", "three"), 1)
    ]


@pytest.mark.parametrize(
    ("extra_config", "kv_role"),
    [
        ({"backend": "mooncake"}, "kv_both"),
        ({"backend": "memcache", "use_layerwise": True}, "kv_both"),
        ({"backend": "memcache"}, "kv_consumer"),
    ],
)
def test_hisparse_rejects_unsupported_store_mode(extra_config, kv_role):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra_config),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError, match="requires bulk MemCache transfers"):
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize("extra_config", [{"backend": "memcache"}, {"backend": "memcache", "load_async": False}])
def test_hisparse_requires_async_store_load(extra_config):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config=extra_config),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError, match="requires load_async=true"):
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize(
    ("has_hisparse", "protocol", "error"),
    [
        (True, "host_tcp", None),
        (True, "host_shm", "requires .*host_tcp.*host_shm"),
        (True, "device_sdma", "requires .*host_tcp.*device_sdma"),
        (True, "device_urma", "requires .*host_tcp.*device_urma"),
        (True, "", "requires .*host_tcp"),
        (True, None, "MMC_LOCAL_CONFIG_PATH.*not set"),
        (True, "host_tcp\nock.mmc.local_service.protocol=device_sdma", "duplicate.*protocol"),
        (False, None, None),
        (False, "device_sdma", None),
    ],
)
def test_bulk_store_protocol_is_scoped_to_hisparse(monkeypatch, tmp_path, has_hisparse, protocol, error):
    monkeypatch.delenv("MMC_LOCAL_CONFIG_PATH", raising=False)
    if protocol is not None:
        path = tmp_path / "mmc-local.conf"
        value = f"  ock.mmc.local_service.protocol = {protocol}\n" if protocol else ""
        path.write_text("# ock.mmc.local_service.protocol = host_shm\n\n" + value, encoding="utf-8")
        monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(path))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"backend": "memcache", "load_async": True},
        ),
    )
    cache = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(role="hisparse_source" if has_hisparse else None, layer_names=[])]
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.create_kv_pool_scheduler"
    ) as scheduler:
        if error:
            with pytest.raises(ValueError, match=error):
                AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)
            scheduler.assert_not_called()
        else:
            connector = AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)
            assert connector.connector_scheduler is scheduler.return_value


def test_memcache_hisparse_registration_rolls_back_on_sdk_error():
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend.store = Mock()
    backend._store_initialized = True
    backend._pending_buffers = None
    backend.store.register_buffer.side_effect = [0, -7, 0]
    backend.store.unregister_buffer.return_value = 0

    with pytest.raises(RuntimeError, match="buffer registration failed: size=20, result=-7"):
        backend.register_buffer([100, 200, 300], [10, 20, 30], check_result=True)

    assert backend.store.register_buffer.call_args_list == [call(100, 10), call(200, 20)]
    backend.store.unregister_buffer.assert_called_once_with(100, 10)
    assert backend._pending_buffers is None

    # Preserve the existing behavior for regular AscendStore users.
    backend.store.register_buffer.side_effect = [-7]
    backend.register_buffer([200], [20])
    backend.store.unregister_buffer.assert_called_once_with(100, 10)
