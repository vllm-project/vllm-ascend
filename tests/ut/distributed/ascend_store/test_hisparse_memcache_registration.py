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
    KVCacheGroupRole,
    KVCacheGroupSpec,
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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.buffer import BufferDescriptor, MemoryLocation
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import (
    aligned_hisparse_allocation_config,
    build_pool_key_namespace,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import get_bulk_by_memory_location
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.hisparse.binding import prepare_hisparse_cache_views

SLAB_ALIGNMENT = 2 * 1024 * 1024


def _checkpoint(model, revision=None, *, model_weights=""):
    return SimpleNamespace(
        model=model,
        model_weights=model_weights,
        revision=revision,
        code_revision=None,
        dtype=torch.bfloat16,
        quantization=None,
    )


def _key_config(target, draft=None):
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(draft_model_config=draft, draft_load_config=None)
        if draft is not None
        else None,
        load_config=SimpleNamespace(load_format="auto"),
    )


def test_hisparse_pool_keys_track_checkpoint_identity():
    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    config = _key_config(_checkpoint("team-a/glm", "v1"))

    def key():
        return build_pool_key_namespace(config, cache, 1, 128)

    original = key()
    assert original.startswith("hisparse-ascendstore-v1:")
    for model, revision in [("team-b/glm", "v1"), ("team-a/glm", "v2")]:
        config.model_config = _checkpoint(model, revision)
        assert key() != original
        assert build_pool_key_namespace(config, None, 1, 128) == "glm"

    # Object-store staging directories and public serving aliases can change on each rank or restart.
    config.model_config = _checkpoint("/tmp/worker-1", "v1", model_weights="s3://weights/glm")
    staged = key()
    config.model_config.model = "/tmp/worker-2"
    config.model_config.served_model_name = "new-endpoint"
    assert key() == staged
    config.speculative_config = SimpleNamespace(
        draft_model_config=_checkpoint("draft/glm", "v1"), draft_load_config=None
    )
    with_draft = key()
    assert with_draft != staged
    config.speculative_config.draft_model_config.revision = "v2"
    assert key() != with_draft


@pytest.mark.parametrize("model_role", ["target", "draft"])
def test_pool_keys_use_resolved_hub_commit(model_role):
    from huggingface_hub import ResolvedRevision

    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    target, draft = _checkpoint("target"), _checkpoint("draft")
    config = _key_config(target, draft)
    checkpoint = target if model_role == "target" else draft
    checkpoint.revision = ResolvedRevision("a" * 40, initial="main")
    first = build_pool_key_namespace(config, cache, 1, 128)
    checkpoint.revision = "a" * 40
    assert build_pool_key_namespace(config, cache, 1, 128) == first
    checkpoint.revision = ResolvedRevision("b" * 40, initial="main")
    assert build_pool_key_namespace(config, cache, 1, 128) != first


@pytest.mark.parametrize("model_role", ["target", "draft"])
def test_dummy_weights_cannot_reuse_checkpoint_pool(model_role):
    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    config = _key_config(_checkpoint("team/glm"), _checkpoint("team/draft"))
    loader = config.load_config
    if model_role == "draft":
        config.speculative_config.draft_load_config = loader = SimpleNamespace(load_format="auto")
    original = build_pool_key_namespace(config, cache, 1, 128)
    loader.load_format = "safetensors"
    assert build_pool_key_namespace(config, cache, 1, 128) == original
    loader.load_format = "dummy"
    assert build_pool_key_namespace(config, cache, 1, 128) != original


@pytest.mark.parametrize("model_role", ["target", "draft"])
@pytest.mark.parametrize("load_format", ["sharded_state", "runai_streamer_sharded"])
def test_hisparse_pool_keys_distinguish_shard_patterns(model_role, load_format, tmp_path):
    from safetensors.torch import load_file, save_file
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    target = _checkpoint(str(tmp_path) if model_role == "target" else "team/glm")
    draft = _checkpoint(str(tmp_path) if model_role == "draft" else "team/draft")
    config = _key_config(target, draft)
    keys = []
    for prefix, weight in [("a", 1.0), ("b", 2.0)]:
        save_file({"weight": torch.tensor([weight])}, tmp_path / f"{prefix}-rank-0-part-0.safetensors")
        pattern = f"{prefix}-rank-{{rank}}-part-{{part}}.safetensors"
        loader_config = LoadConfig(load_format=load_format, model_loader_extra_config={"pattern": pattern})
        loader = get_model_loader(loader_config)
        assert isinstance(loader, ShardedStateLoader)
        (shard,) = tmp_path.glob(loader.pattern.format(rank=0, part="*"))
        assert load_file(shard)["weight"].item() == weight
        if model_role == "target":
            config.load_config = loader_config
        else:
            config.speculative_config.draft_load_config = loader_config
        key = build_pool_key_namespace(config, cache, 1, 128)
        assert build_pool_key_namespace(config, cache, 1, 128) == key
        keys.append(key)
    assert keys[0] != keys[1]


@pytest.mark.parametrize("model_role", ["target", "draft"])
@pytest.mark.parametrize(
    ("config_attr", "field", "original", "override"),
    [
        ("hf_config", "rope_theta", 10000.0, 20000.0),
        (
            "hf_text_config",
            "rope_scaling",
            {"rope_type": "linear", "factor": 2.0},
            {"rope_type": "linear", "factor": 4.0},
        ),
        (
            "hf_text_config",
            "rope_parameters",
            {"full_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
            {"full_attention": {"rope_theta": 20000.0, "rope_type": "default"}},
        ),
        ("hf_text_config", "rms_norm_eps", 1e-6, 1e-5),
    ],
)
def test_hisparse_pool_keys_track_effective_model_config(model_role, config_attr, field, original, override):
    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    target = _checkpoint("team/glm")
    draft = _checkpoint("draft/glm")
    config = _key_config(target, draft)
    model = target if model_role == "target" else draft
    config_values = {field: original}
    setattr(model, config_attr, SimpleNamespace(to_dict=lambda: config_values))
    before = build_pool_key_namespace(config, cache, 1, 128)

    config_values[field] = override

    assert build_pool_key_namespace(config, cache, 1, 128) != before
    assert build_pool_key_namespace(config, None, 1, 128) == "glm"


def test_hisparse_pool_keys_ignore_hf_config_metadata_and_field_order():
    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    model = _checkpoint("team/glm")
    config = _key_config(model)
    first_config = {
        "_name_or_path": "/tmp/worker-1",
        "text_config": {
            "_name_or_path": "/tmp/worker-1",
            "transformers_version": "5.1.0",
            "rope_parameters": {"rope_type": "linear", "factor": 2.0},
        },
    }
    second_config = {
        "text_config": {
            "rope_parameters": {"factor": 2.0, "rope_type": "linear"},
            "transformers_version": "5.2.0",
            "_name_or_path": "/tmp/worker-2",
        },
        "_name_or_path": "/tmp/worker-2",
    }
    model.hf_config = SimpleNamespace(to_dict=lambda: first_config)
    before = build_pool_key_namespace(config, cache, 1, 128)

    model.hf_config = SimpleNamespace(to_dict=lambda: second_config)

    assert build_pool_key_namespace(config, cache, 1, 128) == before
    assert first_config["_name_or_path"] == "/tmp/worker-1"
    assert second_config["_name_or_path"] == "/tmp/worker-2"


def test_relative_checkpoint_keys_follow_the_local_directory(tmp_path, monkeypatch):
    cache = KVCacheConfig(2, [], [], hisparse_host_num_blocks=3)
    keys = []
    for parent in (tmp_path / "team-a", tmp_path / "team-b"):
        model = parent / "weights"
        model.mkdir(parents=True)
        monkeypatch.chdir(parent)
        config = _key_config(_checkpoint("./weights"))
        relative = build_pool_key_namespace(config, cache, 1, 128)
        config.model_config.model = str(model)
        assert build_pool_key_namespace(config, cache, 1, 128) == relative
        keys.append(relative)
    assert keys[0] != keys[1]


def aligned_tensor(size: int) -> torch.Tensor:
    backing = torch.empty(size + SLAB_ALIGNMENT, dtype=torch.int8)
    offset = -backing.data_ptr() % SLAB_ALIGNMENT
    return backing[offset : offset + size]


@pytest.fixture
def memcache_backend() -> MemcacheBackend:
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend.store = Mock()
    backend._store_initialized = True
    backend._lazy_init = False
    backend._pending_buffers = None
    backend._checked_buffer_sizes = {}
    return backend


@pytest.mark.parametrize(
    ("misalignment", "storage_tail", "view_tail"),
    [(0, 0, 0), (1, 0, 0), (SLAB_ALIGNMENT - 1, 0, 0), (1, 4096, 0), (1, 4096, 4096)],
)
@pytest.mark.parametrize("indexer_c8", [False, True])
def test_hisparse_groups_register_with_independent_host_blocks(
    indexer_c8: bool, misalignment: int, storage_tail: int, view_tail: int
) -> None:
    pytest.importorskip("vllm.v1.hisparse.backend", reason="requires the HiSparse backend extension")
    from vllm_ascend.hisparse.backend import AscendHiSparseBackend

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
        load_config=SimpleNamespace(load_format="auto"),
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
    backing = aligned_tensor(padded_size + misalignment + storage_tail)
    storage = backing.untyped_storage()
    begin = backing.storage_offset() + misalignment
    owner = torch.empty(0, dtype=torch.int8).set_(
        storage[begin : begin + padded_size + storage_tail], 0, (padded_size + view_tail,), (1,)
    )
    assert owner.data_ptr() % SLAB_ALIGNMENT == misalignment
    context = {source_name: SimpleNamespace(), indexer_name: SimpleNamespace()}
    allocation_config = aligned_hisparse_allocation_config(cache_config)
    with patch("vllm.v1.worker.utils.torch.zeros", return_value=owner) as allocate:
        assert [tensor.size for tensor in allocation_config.kv_cache_tensors] == [
            host_tensor.size,
            *([padded_size] * 3),
        ]
        device_config = replace(allocation_config, kv_cache_tensors=allocation_config.kv_cache_tensors[1:])
        caches = allocate_kv_cache(device_config, torch.device("cpu"), KVCacheLayout.BLHNC)
        caches[source_name] = source
        AscendHiSparseBackend().prepare_kv_cache_views(allocation_config, caches, context)
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
    with patch.object(worker, "_start_kv_transfer_threads"):
        worker.register_kv_caches(caches)
    assert worker.group_block_len[0] == [32768]
    assert worker.group_block_stride[0] == [32768]
    assert worker.group_block_len[1] == ([16384, 512] if indexer_c8 else [32768])
    assert worker.group_block_stride[1] == ([17408, 17408] if indexer_c8 else [98304])
    assert all(worker.group_block_len[index] == [] for index in range(2, len(groups)))
    worker.m_store.register_buffer.assert_called_once()
    (buffers,) = worker.m_store.register_buffer.call_args.args
    ptrs = [buffer.address for buffer in buffers]
    lengths = [buffer.size_bytes for buffer in buffers]
    assert worker.m_store.register_buffer.call_args.kwargs == {}
    assert [buffer for buffer in buffers if buffer.memory_location is MemoryLocation.HOST] == [
        BufferDescriptor(
            address=source.data_ptr(), size_bytes=num_host_blocks * 32768, memory_location=MemoryLocation.HOST
        )
    ]
    assert all(
        buffer.alignment_bytes == SLAB_ALIGNMENT
        for buffer in buffers
        if buffer.memory_location is MemoryLocation.DEVICE
    )
    assert lengths[ptrs.index(source.data_ptr())] == num_host_blocks * 32768
    assert worker._transfer_layout is not None
    assert worker._transfer_layout.host_memory_group_ids == frozenset({0})
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
    config.model_config = _checkpoint("team/glm")

    def pool_key():
        return build_pool_key_namespace(
            config, worker.kv_cache_config, worker.dcp_size, worker.cache_transfer_granularity
        )

    original_key = pool_key()
    worker.kv_cache_config.num_blocks += 1
    worker.kv_cache_config.hisparse_host_num_blocks += 1
    assert pool_key() == original_key
    worker.kv_cache_config.kv_cache_tensors[1].block_stride += 512
    assert pool_key() != original_key
    worker.kv_cache_config.kv_cache_tensors[1].block_stride -= 512
    spec = groups[1].kv_cache_spec
    replacement = replace(indexer_spec, block_size=64)
    groups[1].kv_cache_spec = (
        replace(spec, block_size=64, kv_cache_specs={indexer_name: replacement})
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else replacement
    )
    assert pool_key() != original_key


def test_hisparse_alignment_is_scoped_to_the_padded_allocation():
    tensor = KVCacheTensor(128, ["cache"], 0, 128)
    config = KVCacheConfig(1, [tensor], [])
    original = torch.zeros(128, dtype=torch.int8)
    caches = {"cache": original}
    prepare_hisparse_cache_views(config, caches, {})
    assert caches["cache"] is original
    with pytest.raises(ValueError, match="reserved alignment padding"):
        prepare_hisparse_cache_views(aligned_hisparse_allocation_config(config), caches, {})
    prepare_hisparse_cache_views(config, caches, {})
    assert caches["cache"] is original


@pytest.mark.parametrize("raw_view", [lambda buf: buf[1:], lambda buf: buf[::2], lambda buf: buf[:128]])
def test_hisparse_rejects_incomplete_raw_view_with_larger_storage(raw_view):
    tensor = KVCacheTensor(128, ["resident"], 0, 128)
    config = aligned_hisparse_allocation_config(KVCacheConfig(1, [tensor], []))
    reserved_bytes = config.kv_cache_tensors[0].size
    raw = raw_view(torch.empty(2 * reserved_bytes, dtype=torch.int8))
    caches = {"resident": raw}
    with pytest.raises(ValueError, match="cover the reserved slab from its base"):
        prepare_hisparse_cache_views(config, caches, {})
    assert caches["resident"] is raw
    assert config.kv_cache_tensors[0].size == reserved_bytes


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"address": 0}, "address and size"),
        ({"size_bytes": 0}, "address and size"),
        ({"alignment_bytes": 0}, "power of two"),
        ({"alignment_bytes": 3}, "power of two"),
        ({"address": 129, "alignment_bytes": 128}, "aligned to 128"),
    ],
)
def test_buffer_descriptor_rejects_invalid_geometry(overrides, message):
    with pytest.raises(ValueError, match=message):
        BufferDescriptor(**{"address": 128, "size_bytes": 32, **overrides})


def test_register_hisparse_views_requires_sized_host_pool():
    spec = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    source = KVCacheGroupSpec(["source"], spec, host_resident=True, role=KVCacheGroupRole.HISPARSE_SOURCE)
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.use_recoverssm = False
    worker.kv_cache_config = KVCacheConfig(1, [], [source])
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    worker.m_store = Mock()
    with pytest.raises(ValueError, match="Host pool must be sized"):
        worker.register_kv_caches({})
    worker.m_store.register_buffer.assert_not_called()


@pytest.mark.parametrize("failure", ["bad-indexer-shape", "bad-slab-owner", "missing-context"])
def test_hisparse_late_view_failure_leaves_all_bindings_untouched(failure):
    names = ["indexer.0", "indexer.1", "indexer.2"]
    size = 3 * 128 * 128 * torch.bfloat16.itemsize
    padded_size = size + SLAB_ALIGNMENT - 1
    spec = AscendSFAIndexerCacheSpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, cache_role=SparseCacheRole.INDEXER
    )
    tensors = [KVCacheTensor(size, names[:2], 32768, size), KVCacheTensor(size, names[2:], 32768, size)]
    config = KVCacheConfig(1, tensors, [KVCacheGroupSpec(names, spec)])
    slab = torch.zeros(padded_size, dtype=torch.int8)
    typed = slab[:size].view(torch.bfloat16).view(3, 1, 1, 128, 128)
    original = dict(zip(names, typed.unbind(), strict=True))
    context = {name: SimpleNamespace(hisparse_block_stride=99) for name in names}
    error: type[Exception] = ValueError
    if failure == "bad-slab-owner":
        original[names[-1]] = original[names[-1]].clone()
        message = "share one backing allocation"
    elif failure == "bad-indexer-shape":
        original[names[-1]] = original[names[-1]].flatten()
        message = "Unexpected HiSparse indexer cache shape"
    else:
        del context[names[-1]]
        error = KeyError
        message = names[-1]

    caches = original.copy()
    allocation_config = aligned_hisparse_allocation_config(config)
    with pytest.raises(error, match=message):
        prepare_hisparse_cache_views(allocation_config, caches, context)
    assert all(caches[name] is original[name] for name in names)
    assert [tensor.size for tensor in allocation_config.kv_cache_tensors] == [padded_size, padded_size]
    assert all(layer.hisparse_block_stride == 99 for layer in context.values())
    assert config.kv_cache_tensors is tensors
    assert [tensor.size for tensor in tensors] == [size, size]


def test_memcache_splits_host_and_device_batches_without_reordering(memcache_backend, caplog) -> None:
    backend = memcache_backend
    backend.store.batch_get_into_layers.side_effect = [[0, -7], [0]]

    keys = ["host", "device1", "device2"]
    addresses = [[1100], [1100], [4000]]
    sizes = [[100], [100], [100]]
    backend.store.batch_put_from_layers.return_value = [0]
    assert backend.put(keys[:1], addresses[:1], sizes[:1], memory_location=MemoryLocation.HOST) == [True]
    backend.store.batch_put_from_layers.assert_called_once_with(["host"], [[1100]], [[100]], MmcDirect.COPY_H2G.value)
    assert get_bulk_by_memory_location(
        backend,
        keys,
        addresses,
        sizes,
        memory_locations=[MemoryLocation.HOST, MemoryLocation.DEVICE, MemoryLocation.DEVICE],
    ) == [0, 0, -7]
    assert backend.store.batch_get_into_layers.call_args_list == [
        call(["device1", "device2"], [[1100], [4000]], [[100], [100]], MmcDirect.COPY_G2L.value),
        call(["host"], [[1100]], [[100]], MmcDirect.COPY_G2H.value),
    ]
    with pytest.raises(ValueError, match="memory location per key"):
        get_bulk_by_memory_location(backend, keys, addresses, sizes, memory_locations=[MemoryLocation.HOST])
    assert backend.put(keys, addresses[:1], sizes, memory_location=MemoryLocation.HOST) is False
    assert "Invalid HiSparse MemCache PUT metadata" in caplog.text


def test_memcache_bounds_each_hisparse_host_get_lease(memcache_backend) -> None:
    backend = memcache_backend
    backend.store.batch_get_into_layers.side_effect = [[0], [-3106], [0]]
    key_size = 130 * 1024 * 1024

    assert get_bulk_by_memory_location(
        backend, ["one", "two", "three"], [[1], [2], [3]], [[key_size]] * 3, memory_locations=[MemoryLocation.HOST] * 3
    ) == [
        0,
        -3106,
        0,
    ]
    assert backend.store.batch_get_into_layers.call_args_list == [
        call([key], [[index]], [[key_size]], MmcDirect.COPY_G2H.value)
        for index, key in enumerate(("one", "two", "three"), 1)
    ]


def test_memcache_oversized_hisparse_get_only_splits_the_host_batch(memcache_backend) -> None:
    backend = memcache_backend
    mib = 1024 * 1024
    device_size = 130 * mib
    host_size = 257 * mib
    backend.store.batch_get_into_layers.side_effect = [[0, 0], [0], [0], [0]]

    results = get_bulk_by_memory_location(
        backend,
        ["device1", "host1", "host-large", "device2", "host2"],
        [[1], [2], [3, 4], [5], [6]],
        [[device_size], [mib], [host_size - mib, mib], [device_size], [mib]],
        memory_locations=[
            MemoryLocation.DEVICE,
            MemoryLocation.HOST,
            MemoryLocation.HOST,
            MemoryLocation.DEVICE,
            MemoryLocation.HOST,
        ],
    )
    assert results == [0] * 5
    assert backend.store.batch_get_into_layers.call_args_list == [
        call(["device1", "device2"], [[1], [5]], [[device_size], [device_size]], MmcDirect.COPY_G2L.value),
        call(["host1"], [[2]], [[mib]], MmcDirect.COPY_G2H.value),
        call(["host-large"], [[3, 4]], [[host_size - mib, mib]], MmcDirect.COPY_G2H.value),
        call(["host2"], [[6]], [[mib]], MmcDirect.COPY_G2H.value),
    ]


@pytest.mark.parametrize("results", [[0], [0, 0, 0]])
def test_memcache_hisparse_get_fails_when_sdk_result_count_differs(memcache_backend, caplog, results) -> None:
    backend = memcache_backend
    backend.store.batch_get_into_layers.side_effect = [[0], results]

    result = get_bulk_by_memory_location(
        backend,
        ["host1", "device", "host2"],
        [[1], [2], [3]],
        [[100]] * 3,
        memory_locations=[MemoryLocation.HOST, MemoryLocation.DEVICE, MemoryLocation.HOST],
    )
    assert result is None
    assert "BatchResultShapeError" in caplog.text
    assert f"HiSparse Memcache GET returned {len(results)} results for 2 keys" in caplog.text


@pytest.mark.parametrize(
    ("extra_config", "kv_role"),
    [
        ({"backend": "memcache", "use_layerwise": True}, "kv_both"),
        ({"backend": "memcache"}, "kv_consumer"),
    ],
)
def test_hisparse_rejects_unsupported_store_mode(extra_config, kv_role):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra_config),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError, match="requires whole-block transfers"):
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize("extra_config", [{"backend": "memcache"}, {"backend": "memcache", "load_async": False}])
def test_hisparse_requires_async_store_load(extra_config):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config=extra_config),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError, match="requires load_async=true"):
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)


def test_hisparse_rejects_partial_chunk_saves():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"backend": "memcache", "load_async": True, "discard_partial_chunks": False},
        ),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError, match="requires discard_partial_chunks=true"):
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize(
    ("extra_config", "error"),
    [
        pytest.param({}, r"timed-out PUT.*backend=memcache.*host_tcp", id="default-mooncake"),
        pytest.param({"backend": "yuanrong"}, "does not support mixed Host/Device bulk transfers", id="yuanrong"),
    ],
)
def test_hisparse_rejects_incompatible_backend_before_creating_pool(role, extra_config, error):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=1),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"load_async": True, **extra_config},
        ),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
    with (
        patch(f"{module}.create_kv_pool_scheduler") as scheduler,
        patch(f"{module}.create_kv_pool_worker") as worker,
        pytest.raises(ValueError, match=error),
    ):
        AscendStoreConnector(config, role, cache)
    scheduler.assert_not_called()
    worker.assert_not_called()


@pytest.mark.parametrize(("save_decode", "eager"), [(True, False), (False, False), (True, True), (False, True)])
def test_whole_block_put_requires_current_host_rows(save_decode, eager):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=eager)),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"backend": "memcache", "load_async": True, "save_decode_cache": save_decode},
        ),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with (
        patch.object(MemcacheBackend, "validate_hisparse_colocated_config"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.create_kv_pool_scheduler"
        ) as scheduler,
    ):
        if not eager:
            with pytest.raises(ValueError, match="requires eager_host_mirror=true"):
                AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)
            scheduler.assert_not_called()
        else:
            assert (
                AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache).connector_scheduler
                is scheduler.return_value
            )


def test_hisparse_reports_all_incompatible_bulk_store_options(monkeypatch, tmp_path):
    path = tmp_path / "mmc-local.conf"
    path.write_text("ock.mmc.local_service.protocol=device_sdma\n", encoding="utf-8")
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(path))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer",
            kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, "load_async": False},
        ),
    )
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(role="hisparse_source", layer_names=[])])
    with pytest.raises(ValueError) as error:
        AscendStoreConnector(config, KVConnectorRole.SCHEDULER, cache)
    message = str(error.value)
    assert "requires whole-block transfers with kv_role=kv_both" in message
    assert "requires load_async=true" in message
    assert "ock.mmc.local_service.protocol=host_tcp" in message
    assert "device_sdma" in message


@pytest.mark.parametrize(
    ("has_hisparse", "protocol", "error"),
    [
        (True, "host_tcp", None),
        (True, "host_shm", "requires .*host_tcp.*host_shm"),
        (True, "host_tcp # inline comment", "requires .*host_tcp.*inline comment"),
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
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace(eager_host_mirror=True)),
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


@pytest.mark.parametrize(("first", "second"), [("device_sdma", "host_tcp"), ("host_tcp", "device_sdma")])
def test_memcache_lazy_initialization_uses_first_protocol_for_regular_store(monkeypatch, tmp_path, first, second):
    path = tmp_path / "mmc-local.conf"
    path.write_text(
        f"ock.mmc.local_service.protocol={first}\nock.mmc.local_service.protocol={second}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(path))
    with patch.object(MemcacheBackend, "_setup_store") as setup_store:
        backend = MemcacheBackend(SimpleNamespace(), device_id=0, lazy_init=True)
    if first == "device_sdma":
        setup_store.assert_not_called()
        assert backend.store is None
    else:
        setup_store.assert_called_once_with()
        assert backend.store is setup_store.return_value


def test_memcache_hisparse_registration_rolls_back_on_sdk_error(memcache_backend):
    backend = memcache_backend
    backend.store.register_buffer.side_effect = [0, -7, 0]
    backend.store.unregister_buffer.return_value = 0

    with pytest.raises(RuntimeError, match="buffer registration failed: size=20, result=-7"):
        backend.register_buffer(
            [
                BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
                BufferDescriptor(address=200, size_bytes=20),
                BufferDescriptor(address=300, size_bytes=30),
            ]
        )

    assert backend.store.register_buffer.call_args_list == [call(100, 10), call(200, 20)]
    backend.store.unregister_buffer.assert_called_once_with(100, 10)
    assert backend._pending_buffers is None

    # Preserve the existing behavior for regular AscendStore users.
    backend.store.register_buffer.side_effect = [-7]
    backend.register_buffer([BufferDescriptor(address=200, size_bytes=20)])
    backend.store.unregister_buffer.assert_called_once_with(100, 10)


@pytest.mark.parametrize("rollback_result", [-23, RuntimeError("SDK unregister error")])
def test_memcache_hisparse_registration_keeps_error_if_rollback_fails(memcache_backend, caplog, rollback_result):
    backend = memcache_backend
    registration_error = ValueError("SDK rejected registration")
    backend.store.register_buffer.side_effect = [0, 0, registration_error]
    backend.store.unregister_buffer.side_effect = [rollback_result, 0]

    with pytest.raises(ValueError, match="SDK rejected registration") as raised:
        backend.register_buffer(
            [
                BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
                BufferDescriptor(address=200, size_bytes=20),
                BufferDescriptor(address=300, size_bytes=30),
            ]
        )

    assert raised.value is registration_error
    assert backend.store.unregister_buffer.call_args_list == [call(200, 20), call(100, 10)]
    assert backend._pending_buffers is None
    assert "HiSparse MemCache buffer rollback failed: size=20" in caplog.text


def test_memcache_hisparse_rollback_keeps_earlier_registration_and_allows_retry(memcache_backend):
    backend = memcache_backend
    backend.store.register_buffer.side_effect = [0, 0, -7, 0, 0]
    backend.store.unregister_buffer.return_value = 0
    backend.register_buffer(
        [
            BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
            BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
        ]
    )

    with pytest.raises(RuntimeError, match="result=-7"):
        backend.register_buffer(
            [
                BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
                BufferDescriptor(address=200, size_bytes=20),
                BufferDescriptor(address=300, size_bytes=30),
            ]
        )
    backend.store.unregister_buffer.assert_called_once_with(200, 20)
    backend.register_buffer(
        [
            BufferDescriptor(address=100, size_bytes=10, memory_location=MemoryLocation.HOST),
            BufferDescriptor(address=200, size_bytes=20),
        ]
    )
    with pytest.raises(RuntimeError, match="already registered with size=10, requested size=11"):
        backend.register_buffer([BufferDescriptor(address=100, size_bytes=11, memory_location=MemoryLocation.HOST)])
    backend.unregister_buffer([100], [10])
    backend.register_buffer([BufferDescriptor(address=100, size_bytes=11, memory_location=MemoryLocation.HOST)])
    assert backend.store.register_buffer.call_args_list == [
        call(100, 10),
        call(200, 20),
        call(300, 30),
        call(200, 20),
        call(100, 11),
    ]
