# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ownership and page addressing at the Ascend HiSparse boundary."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.kv_cache_planning import _get_kv_cache_bytes_per_block
from vllm.v1.hisparse import binding as common_binding
from vllm.v1.hisparse.coordinator import HiSparseCoordinator
from vllm.v1.hisparse.layout import _build_hisparse_kv_cache_tensors, get_hisparse_kv_cache_groups
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheLayout,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.hisparse import backend as backend_module
from vllm_ascend.distributed.kv_transfer.hisparse import binding
from vllm_ascend.distributed.kv_transfer.hisparse.backend import AscendHiSparseBackend
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import get_kv_cache_coordinator


def test_explicit_dma_extension_load_keeps_global_op_policy(monkeypatch):
    namespace = SimpleNamespace()
    bootstrap = MagicMock()
    ops_type = MagicMock()
    monkeypatch.setattr(torch.ops, "_C_ascend", namespace)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.hisparse", SimpleNamespace(HiSparseNPUOps=ops_type))
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", SimpleNamespace(bootstrap_custom_op_env=bootstrap))

    def load(name):
        assert name == "vllm_ascend.vllm_ascend_C"
        namespace.swap_blocks_batch = MagicMock()

    loader = MagicMock(side_effect=load)
    monkeypatch.setattr(backend_module.importlib, "import_module", loader)
    backend = AscendHiSparseBackend()
    backend.check_available()
    backend.check_available()

    bootstrap.assert_called_once_with()
    loader.assert_called_once_with("vllm_ascend.vllm_ascend_C")


def test_lru_init_avoids_unsupported_int16_range(monkeypatch):
    original = torch.arange
    dtypes = []

    def arange(*args, **kwargs):
        dtypes.append(kwargs["dtype"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", arange)
    values = AscendHiSparseBackend().create_lru_init(8, torch.device("cpu"))

    assert dtypes == [torch.int32]
    assert values.dtype == torch.int16
    assert values.tolist() == list(range(8))


def test_backend_current_stream_tracks_context_changes(monkeypatch):
    active = [object()]
    monkeypatch.setattr(torch.npu, "current_stream", lambda: active[0])
    backend = AscendHiSparseBackend()
    first = backend.current_stream()
    active[0] = object()
    assert backend.current_stream() is active[0]
    assert backend.current_stream() is not first


@pytest.mark.parametrize("enable_caching", [False, True])
def test_hisparse_uses_core_role_managers_and_host_pool(enable_caching):
    source = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    indexer = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec(["source"], source, role=KVCacheGroupRole.HISPARSE_SOURCE, host_resident=True),
        KVCacheGroupSpec(["indexer"], indexer, role=KVCacheGroupRole.HISPARSE_INDEXER),
        KVCacheGroupSpec(["resident"], HiSparseResidentSpec(block_size=128, page_size=source.page_size_bytes)),
        KVCacheGroupSpec(
            ["hot"], HiSparseHotSpec(block_size=128, page_size=source.page_size_bytes, blocks_per_request=1)
        ),
    ]
    config = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups, hisparse_host_num_blocks=128)
    core = get_kv_cache_coordinator(
        config,
        max_model_len=1024,
        max_in_flight_tokens=128,
        enable_caching=enable_caching,
        hash_block_size=128,
        scheduler_block_size=128,
    )
    assert type(core).__module__ == "vllm.v1.core.kv_cache_coordinator"
    sparse = HiSparseCoordinator(config, core.single_type_managers, 1024, num_reprefillable_tokens=0)
    assert sparse.host_group_id == 0
    assert sparse.host_manager is core.single_type_managers[0]
    assert sparse.host_manager.block_pool is not core.block_pool
    assert sparse.gpu_pool is core.block_pool
    assert len(sparse.resident_managers) == len(sparse.hot_managers) == 1


def test_registration_retains_mapping_until_explicit_release(monkeypatch):
    mapping = MagicMock()
    mapped_type = MagicMock(return_value=mapping)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.hisparse", SimpleNamespace(MappedHostTensor=mapped_type))
    pool = torch.empty(4096, dtype=torch.int8)
    backend = AscendHiSparseBackend()

    backend.pin_tensor(pool)
    mapped_type.assert_called_once_with(pool)
    assert pool._hisparse_registration is mapping
    mapping.close.assert_not_called()
    backend.unpin_tensor(pool)
    mapping.close.assert_called_once_with()
    assert not hasattr(pool, "_hisparse_registration")


def test_group_layers_share_one_resolver_workspace(monkeypatch):
    workspace_type = MagicMock()
    ops_type = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.hisparse",
        SimpleNamespace(ResolverWorkspace=workspace_type, HiSparseNPUOps=ops_type),
    )
    group = SimpleNamespace(backend_state=None)
    mapping = object()
    runtime = SimpleNamespace(
        index_group=group,
        max_swap_rows=4,
        region_stride=8,
        max_union_rows=4,
        device=torch.device("cpu"),
        registered_host_pool=SimpleNamespace(_hisparse_registration=mapping),
    )
    backend = AscendHiSparseBackend()

    backend.create_cache_ops(runtime)
    backend.create_cache_ops(runtime)

    workspace_type.assert_called_once()
    assert group.backend_state is workspace_type.return_value
    assert all(call.args == (mapping, group.backend_state) for call in ops_type.call_args_list)


def test_indexer_expanded_view_addresses_shared_hma_pages():
    """Scaling page IDs must reach the same bytes without copying the cache."""
    spec = AscendSFAIndexerCacheSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32)
    backing = torch.arange(96, dtype=torch.float32)
    cache = torch.as_strided(backing, (3, 1, 4, 4), (32, 16, 4, 1), storage_offset=16)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["indexer"], kv_cache_spec=spec)])
    layer = SimpleNamespace()

    caches = {"indexer": cache}
    backend = AscendHiSparseBackend()
    backend.validate_kv_cache_config(config)
    backend.prepare_kv_cache_views(config, caches, {"indexer": layer})

    expanded = caches["indexer"][0]
    assert expanded.is_contiguous()
    assert expanded.data_ptr() == cache.data_ptr()
    assert expanded.untyped_storage().data_ptr() == backing.data_ptr()
    assert expanded.storage_offset() == cache.storage_offset()
    assert layer.hisparse_block_stride == 2
    for block in range(3):
        torch.testing.assert_close(expanded[block * layer.hisparse_block_stride, :, 0], cache[block, 0])
    expanded[4, 3, 0, 3] = -1
    assert backing[-1] == -1


@pytest.mark.parametrize(
    "num_indexers,num_layers,expected_stride", [(1, 1, 294912), (21, 78, 884736), (78, 78, 2654208)]
)
def test_indexer_alignment_flows_through_real_planner_and_shared_views(num_indexers, num_layers, expected_stride):
    indexer = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    source = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    specs = binding.align_hisparse_indexer_specs(
        {"source": source, **{f"indexer_{i}": indexer for i in range(num_indexers)}}
    )
    assert specs.pop("source") is source
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    groups = [KVCacheGroupSpec(list(specs), uniform)]
    groups.extend(
        KVCacheGroupSpec(
            [f"hot_{i}"], HiSparseHotSpec(block_size=128, page_size=source.page_size_bytes, blocks_per_request=1)
        )
        for i in range(num_layers)
    )
    layout = KVCacheLayout.BLHNC
    stride = _get_kv_cache_bytes_per_block(groups, layout)
    assert stride == expected_stride
    tensors = _build_hisparse_kv_cache_tensors(groups, 3, stride * 3, layout, stride)
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=tensors, kv_cache_groups=groups)
    caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    original = {name: caches[name] for name in specs}
    layers = {name: SimpleNamespace() for name in specs}
    binding._bind_indexer_views(caches, specs, layers)
    for i, name in enumerate(specs):
        view = caches[name][0]
        assert view.data_ptr() == original[name].data_ptr()
        factor = layers[name].hisparse_block_stride
        assert factor * indexer.page_size_bytes == stride
        view[2 * factor].fill_(i + 1)
    # Check after all writes so a later layer cannot silently overwrite another.
    for i, name in enumerate(specs):
        torch.testing.assert_close(original[name][2], torch.full_like(original[name][2], i + 1))


@pytest.mark.parametrize("second_is_leader,skip_topk", [(False, False), (True, True)])
def test_runtime_index_groups_drive_specs_and_physical_payloads(monkeypatch, second_is_leader, skip_topk):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.v1.hisparse.binding import allocate_hisparse_kv_caches, bind_hisparse_kv_caches

    from vllm_ascend.worker.v2 import attn_utils

    source = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, is_index_group_leader=True
    )
    indexer = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=128),
        attention_config=SimpleNamespace(
            indexer_kv_dtype="bf16", hisparse_config=SimpleNamespace(device_buffer_size=4096, eager_host_mirror=True)
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(index_topk=2048),
            hf_text_config=SimpleNamespace(index_head_dim=128),
        ),
    )
    layers = {}
    handles = []
    group = object()
    kernel_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda _spec: [128])
    for i, is_leader in enumerate((True, second_is_leader)):
        runtime = object.__new__(HiSparseRuntime)
        runtime.is_group_leader = is_leader
        runtime.index_group = object() if i and is_leader else group
        runtime.row_width = 576
        runtime.kv_dtype = torch.bfloat16
        runtime.backend = MagicMock()
        handle = HiSparseCacheHandle(runtime)
        handles.append(handle)
        layer = MagicMock(spec=MLAAttention)
        layer.kv_sharing_target_layer_name = None
        layer.get_kv_cache_spec.return_value = source
        layer.get_attn_backend.return_value = kernel_backend
        layer.hisparse_cache = handle
        layer.is_draft_layer = False
        # Presence of an indexer does not imply a leader. A dynamic MTP skip
        # must not turn an existing leader into a physical-layout follower.
        layer.impl = SimpleNamespace(indexer=object(), skip_topk=skip_topk if i else False, runtime_has_indexer=True)
        layers[f"layer{i}.attn"] = layer
        if is_leader:
            cache = MagicMock(spec=DeepseekV32IndexerCache)
            cache.kv_sharing_target_layer_name = None
            cache.get_kv_cache_spec.return_value = indexer
            cache.get_attn_backend.return_value = kernel_backend
            layers[f"layer{i}.indexer.k_cache"] = cache
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _config: False)
    monkeypatch.setattr(attn_utils, "kv_cache_dtype_str_to_dtype", lambda *_args: torch.bfloat16)
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(is_sparse_li_c8_layer=lambda _name: False, is_sparse_li_c4_layer=lambda _name: False),
    )

    specs = attn_utils.get_kv_cache_spec(config)
    assert [specs[f"layer{i}.attn"].is_index_group_leader for i in range(2)] == [True, second_is_leader]
    groups = get_hisparse_kv_cache_groups(config, specs)
    assert groups is not None
    resident_groups = [g for g in groups if isinstance(g.kv_cache_spec, HiSparseResidentSpec)]
    hot_groups = [g for g in groups if isinstance(g.kv_cache_spec, HiSparseHotSpec)]
    assert len(resident_groups) == len(hot_groups) == 1 + second_is_leader
    assert [len(g.layer_names) for g in hot_groups] == ([1, 1] if second_is_leader else [2])

    device_groups = [g for g in groups if not g.host_resident]
    stride = _get_kv_cache_bytes_per_block(device_groups, KVCacheLayout.BLHNC)
    tensors = _build_hisparse_kv_cache_tensors(device_groups, 3, stride * 3, KVCacheLayout.BLHNC, stride)
    host_stride = groups[0].kv_cache_spec.page_size_bytes
    tensors[:0] = _build_hisparse_kv_cache_tensors(
        groups[:1], 3, host_stride * 3, KVCacheLayout.LBNHC, host_stride, host_resident=True
    )
    cache_config = KVCacheConfig(
        num_blocks=3, kv_cache_tensors=tensors, kv_cache_groups=groups, hisparse_host_num_blocks=3
    )
    host_pool = SimpleNamespace(registered=None, shared_region=None)

    def allocate_host(size):
        host_pool.registered = torch.zeros(size, dtype=torch.int8)
        return host_pool.registered

    host_pool.allocate = allocate_host
    caches = allocate_hisparse_kv_caches(
        cache_config, torch.device("cpu"), KVCacheLayout.BLHNC, [128] * len(groups), host_pool
    )
    block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros((1, 3), dtype=torch.int32) for _ in groups],
        slot_mappings=[torch.zeros(3, dtype=torch.int64) for _ in groups],
    )
    bound = bind_hisparse_kv_caches(
        forward_context=layers,
        kv_cache_config=cache_config,
        kv_caches=caches,
        block_tables=block_tables,
        host_pool=host_pool,
    )
    assert bound == handles
    assert handles[0].runtime.resident_source_index == 0
    assert handles[1].runtime.resident_source_index == int(second_is_leader)
    if not second_is_leader:
        assert handles[0].runtime.index_group is handles[1].runtime.index_group
        assert handles[0].runtime.hot_block_table is handles[1].runtime.hot_block_table

    # Shared index groups use one physical row ID in different layer views.
    # Independent groups instead own different block IDs in the shared pool.
    blocks = (1, 2 if second_is_leader else 1)
    for handle, block, value in zip(handles, blocks, (11, 29), strict=True):
        handle.runtime.hot.cache[block, 7].fill_(value)
    for handle, block, value in zip(handles, blocks, (11, 29), strict=True):
        view = handle.view
        assert view is not None
        physical_row = block * view.attention_block_stride + 7
        torch.testing.assert_close(view.cache[block, 7], torch.full((576,), value, dtype=torch.bfloat16))
        torch.testing.assert_close(view.attention_cache.view(-1, 576)[physical_row], view.cache[block, 7])


def test_quantized_indexer_is_rejected_before_pool_allocation(monkeypatch):
    spec = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8, scale_dim=1)
    allocator = MagicMock()
    monkeypatch.setattr(common_binding, "get_hisparse_backend", AscendHiSparseBackend)
    monkeypatch.setattr(common_binding, "get_hisparse_host_pool", allocator)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["indexer"], kv_cache_spec=spec)])

    with pytest.raises(ValueError, match="unquantized"):
        common_binding.init_hisparse_kv_cache(config, torch.device("cpu"), [128], None, {}, None)

    allocator.assert_not_called()


def test_invalid_indexer_stride_releases_initialized_host_pool(monkeypatch):
    spec = AscendSFAIndexerCacheSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32)
    backing = torch.empty(128)
    cache = torch.as_strided(backing, (3, 1, 4, 4), (33, 16, 4, 1))
    monkeypatch.setattr(common_binding, "get_hisparse_backend", AscendHiSparseBackend)
    monkeypatch.setattr(common_binding, "allocate_hisparse_kv_caches", lambda *args: {"indexer": cache})
    registered = torch.empty(4096, dtype=torch.int8)
    pool = SimpleNamespace(shared_region=None, registered=registered)
    monkeypatch.setattr(common_binding, "get_hisparse_host_pool", lambda *args: pool)
    release = MagicMock()
    drop = MagicMock()
    monkeypatch.setattr(common_binding, "release_pinned_state", release)
    monkeypatch.setattr(common_binding, "drop_hisparse_host_pool", drop)
    bind = MagicMock()
    monkeypatch.setattr(common_binding, "bind_hisparse_kv_caches", bind)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["indexer"], kv_cache_spec=spec)])
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC)
    )

    with pytest.raises(ValueError, match="whole number"):
        common_binding.init_hisparse_kv_cache(
            config, torch.device("cpu"), [4], vllm_config, {"indexer": SimpleNamespace()}, None
        )

    bind.assert_not_called()
    release.assert_called_once_with([], [registered])
    drop.assert_called_once_with(vllm_config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_backend_resident_gather_preserves_packed_bytes_and_page_gaps(dtype):
    # Include arbitrary FP8 bit patterns and RoPE/scale bytes, not FP8 numbers.
    width = 656 if dtype != torch.bfloat16 else 576
    page_stride = 3 * width
    offset = width
    storage = torch.arange((offset + 2 * page_stride) * dtype.itemsize, dtype=torch.int64).to(torch.uint8)
    source = storage.view(dtype).as_strided((2, 2, width), (page_stride, width, 1), storage_offset=offset)
    rows = torch.tensor([3, 0, 2, 3, -1])
    destination = torch.empty((5, width), dtype=dtype)
    raw_source = source.view(torch.uint8)
    safe = rows.clamp_min(0)
    expected = raw_source[safe // 2, safe % 2]
    before = storage.clone()

    AscendHiSparseBackend().gather_resident_rows(source, destination, rows)

    assert torch.equal(destination.view(torch.uint8), expected)
    assert torch.equal(storage, before)


@pytest.mark.parametrize("bad_width", [False, True])
def test_backend_resident_gather_rejects_geometry_before_writing(bad_width):
    source = torch.ones((2, 2, 4))
    destination = torch.full((2, 5 if bad_width else 4), 17.0)
    rows = torch.tensor([0, 1]) if bad_width else torch.tensor([0])
    with pytest.raises(ValueError, match="row counts and payload byte widths"):
        AscendHiSparseBackend().gather_resident_rows(source, destination, rows)
    assert torch.all(destination == 17)


def test_backend_keeps_private_events_and_rejects_unsupported_ipc(monkeypatch):
    events = MagicMock()
    monkeypatch.setattr(torch.npu, "Event", events)
    backend = AscendHiSparseBackend()
    assert not backend.supports_shared_host_pool
    assert backend.create_event() is events.return_value
    assert backend.create_event(interprocess=False) is events.return_value
    with pytest.raises(NotImplementedError, match="private per-rank"):
        backend.create_event(interprocess=True)
    with pytest.raises(NotImplementedError, match="export"):
        backend.export_event(object())
    with pytest.raises(NotImplementedError, match="import"):
        backend.import_event(torch.device("cpu"), b"invalid-handle")
    assert events.call_count == 2


def test_backend_rejects_cuda_prefill_layout_without_touching_payloads():
    payload = torch.arange(656, dtype=torch.int64).to(torch.uint8)
    before = payload.clone()
    with pytest.raises(NotImplementedError, match="native SFA prefill"):
        AscendHiSparseBackend().gather_fp8_prefill(
            payload,
            payload,
            payload,
            payload,
            1,
            host_cache=payload,
            host_row_ids=payload,
            device_row_ids=payload,
        )
    assert torch.equal(payload, before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_common_prefill_staging_uses_ascend_payload_backend(dtype):
    width = 656 if dtype == torch.float8_e4m3fn else 576
    source = torch.arange(4 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).view(dtype).view(2, 2, width)
    runtime = object.__new__(HiSparseRuntime)
    runtime.backend = AscendHiSparseBackend()
    runtime.cache_ops = SimpleNamespace(hisparse_gather_plan=lambda *args: None)
    rows = torch.tensor([[3, 0, 2, -1]], dtype=torch.int32)
    plan = SimpleNamespace(
        block_size=2,
        row_ids=rows,
        gpu_row_ids=rows,
        block_table=torch.zeros((1, 2), dtype=torch.int32),
        dst_rows=torch.arange(4, dtype=torch.int32).view(1, 4),
        miss_mask=torch.zeros((1, 4), dtype=torch.bool),
    )

    staged = runtime.gather_prefill_cache(source, plan, source)

    safe = rows[0].long().clamp_min(0)
    expected = source.view(torch.uint8)[safe // 2, safe % 2]
    assert torch.equal(staged.view(torch.uint8).flatten(0, 1), expected)


def test_ascend_speculative_prefetch_preserves_tentative_write_policy():
    """MTP verification stages followers but writes all tentative query rows."""
    runtime = SimpleNamespace(
        backend=AscendHiSparseBackend(),
        max_swap_rows=16,
        max_num_reqs=4,
        eager_host_mirror=False,
        begin_forward=MagicMock(),
    )
    handle = HiSparseCacheHandle(runtime)
    handle.view = SimpleNamespace(cache=object())
    handle.slot_mapping = torch.arange(16)
    for tokens in [16, 4, 16]:
        metadata = SimpleNamespace(
            num_actual_tokens=tokens,
            num_decode_tokens=tokens,
            num_reqs=tokens // 4,
            max_query_len=4,
            req_id_per_token=None,
        )
        handle._prepare_for_batch(metadata)
        assert handle.prefetch_decode_group
        assert not handle.decode_batch
        assert handle.host_mirror_required
        _, slots, count = handle.write_target(tokens, tokens)
        assert count == tokens and slots.numel() == tokens
    handle._prepare_for_batch(None)
    assert not handle.prefetch_decode_group
    assert not handle.host_mirror_required
