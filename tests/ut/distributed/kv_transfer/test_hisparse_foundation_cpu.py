# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU boundary checks using the installed Core and Ascend implementations."""

import math
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.indexer import AscendHiSparseIndexerBackend
from vllm_ascend.attention.sfa_hisparse import (
    AscendHiSparseImpl,
    AscendHiSparseMetadataBuilder,
    finish_hisparse_kv_update,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.hisparse import cache_ops as hisparse_ops
from vllm_ascend.hisparse import host_pool
from vllm_ascend.hisparse.binding import (
    _bind_indexer_views,
    align_hisparse_indexer_specs,
)
from vllm_ascend.hisparse.c8 import (
    c8_storage_view,
    compact_c8_indices,
    map_c8_physical_indices,
    pack_c8_components,
)
from vllm_ascend.ops.triton import sfa_indexer_store

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm_ascend.hisparse.backend import AscendHiSparseBackend  # noqa: E402


@pytest.fixture(params=[False, True], ids=["standalone", "profiled"])
def prepare_workspaces(request):
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from vllm_ascend.hisparse.backend import prepare_hisparse_workspaces

    reset_workspace_manager()
    try:
        if request.param:
            init_workspace_manager(torch.device("cpu"))

        @torch.inference_mode()
        def prepare(runtimes):
            layers = {str(i): NS(hisparse_cache=NS(runtime=runtime)) for i, runtime in enumerate(runtimes)}
            layers.update(ordinary_layer=NS(), unbound_layer=NS(hisparse_cache=None))
            prepare_hisparse_workspaces(layers)

        yield prepare if request.param else None
    finally:
        reset_workspace_manager()


@pytest.mark.parametrize("mxfp8", [False, True])
def test_c8_prolog_requires_matching_fp8_weight_scheme(mxfp8):
    from vllm_ascend.attention.utils import PreprocessType
    from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod, AscendW8A8MXFP8DynamicLinearMethod

    method = AscendW8A8MXFP8DynamicLinearMethod if mxfp8 else AscendW8A8DynamicLinearMethod
    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.fused_qkv_a_proj = NS(quant_method=NS(quant_method=method.__new__(method)))
    impl.is_kv_consumer, impl.has_indexer = False, False
    impl.dcp_group = None
    impl.qk_rope_head_dim = 64
    impl.q_a_layernorm = impl.kv_a_layernorm = NS()
    impl.q_proj = NS(_chunk_size=0)
    impl.enable_sparse_sfa_turboquant = False
    expected = PreprocessType.PROLOG_V3 if mxfp8 else None
    assert impl._fused_preprocess_type() is expected


@pytest.mark.parametrize("hisparse", [False, True])
def test_indexer_backend_choice_survives_model_construction_context(hisparse):
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import HiSparseConfig
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.patch.worker import patch_attn_utils  # noqa: F401

    config = VllmConfig()
    config.attention_config.hisparse_config = HiSparseConfig() if hisparse else None
    expected = AscendHiSparseIndexerBackend if hisparse else AscendSFAIndexerBackend
    with set_current_vllm_config(config):
        cache = DeepseekV32IndexerCache(132, torch.uint8, "regression.indexer.k_cache", CacheConfig())
        assert cache.get_attn_backend() is expected
    # Core's post-load block-size discovery runs after the construction context
    # exits. A later model's context must not change an existing cache either.
    assert cache.get_attn_backend() is expected
    other = VllmConfig()
    other.attention_config.hisparse_config = None if hisparse else HiSparseConfig()
    with set_current_vllm_config(other):
        assert cache.get_attn_backend() is expected


def test_failed_unregister_retains_host_owner_for_safe_retry(monkeypatch):
    tensor = torch.empty(64, dtype=torch.uint8)
    events = []

    class Registration:
        def __init__(self, allocation):
            self.tensor = allocation

        def close(self):
            events.append("close")
            if len(events) == 1:
                raise RuntimeError("driver did not retire registration")

    monkeypatch.setattr(host_pool, "MappedHostTensor", Registration)
    backend = AscendHiSparseBackend()
    backend.pin_tensor(tensor)
    registration = tensor._hisparse_registration
    assert registration.tensor is tensor
    with pytest.raises(RuntimeError, match="driver did not retire"):
        backend.unpin_tensor(tensor)
    assert tensor._hisparse_registration is registration
    backend.unpin_tensor(tensor)
    assert not hasattr(tensor, "_hisparse_registration")
    assert events == ["close", "close"]


def test_private_host_allocator_keeps_other_backends_on_core_path(monkeypatch):
    from vllm.v1.hisparse import backend as core_backend
    from vllm.v1.hisparse import runtime

    from vllm_ascend.hisparse.host_pool import install

    class OtherBackend(core_backend.HiSparseBackend):
        def pin_tensor(self, tensor):
            tensor._test_registration = self

    backend = OtherBackend()
    monkeypatch.setattr(runtime, "get_hisparse_backend", lambda: backend)
    monkeypatch.setattr(core_backend, "get_hisparse_backend", lambda: backend)
    # Restore the process-wide binding when this test exits.
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", runtime.allocate_pinned_host_pool)
    install()
    installed = runtime.allocate_pinned_host_pool
    install()
    assert runtime.allocate_pinned_host_pool is installed
    payload, registered = installed(1001)
    assert registered._test_registration is backend
    assert payload.numel() == 1001 and registered.data_ptr() % 4096 == 0
    payload.fill_(17)
    torch.testing.assert_close(registered[:1001], payload)


def test_mapped_addresses_retain_offsets_and_reject_expired_or_external_views():
    backing = torch.arange(128, dtype=torch.uint8)
    mapping = host_pool.MappedHostTensor.__new__(host_pool.MappedHostTensor)
    mapping.tensor = backing
    mapping.host_address = backing.data_ptr()
    mapping.nbytes = backing.numel()
    mapping.device_ptr = 4096
    mapping.closed = False
    assert mapping.address_of(backing[17:43]) == 4113
    with pytest.raises(ValueError, match="outside"):
        mapping.address_of(torch.empty(256, dtype=torch.uint8))
    with pytest.raises(ValueError, match="contiguous"):
        mapping.address_of(backing[::2])
    mapping.closed = True
    with pytest.raises(ValueError, match="live mapped"):
        mapping.address_of(backing)


@pytest.mark.parametrize("max_rows,hot_rows,union_rows", [(1, 16, 8), (9, 48, 24), (37, 128, 72)])
def test_resolver_scratch_follows_group_geometry_and_is_not_shared_across_groups(
    monkeypatch, max_rows, hot_rows, union_rows, prepare_workspaces
):
    # Loading a device extension is the only mocked boundary; the group-owned
    # tensors below are real CPU storage, not a replacement resolver algorithm.
    native = NS(hisparse_resolver_validate_scratch=lambda hot, union: None)
    monkeypatch.setattr(hisparse_ops, "_load_resolver", lambda: native)
    backend = AscendHiSparseBackend()

    def group():
        owner = NS(
            copy_stream=NS(device=torch.device("cpu")),
            shared_topk=NS(device_topk_rows=torch.empty((max_rows, union_rows), dtype=torch.int32)),
        )
        leader = NS(
            index_group=owner,
            max_swap_rows=max_rows,
            region_stride=hot_rows,
            max_union_rows=union_rows,
            device=torch.device("cpu"),
            kv_dtype=torch.bfloat16,
            registered_host_pool=NS(_hisparse_registration=object()),
        )
        owner.leader = leader
        if prepare_workspaces is not None:
            prepare_workspaces([leader, leader])
            pending = leader._ascend_profiled_cache_ops
            prepare_workspaces([leader])
            assert leader._ascend_profiled_cache_ops is pending
        leader.cache_ops = backend.create_cache_ops(leader)
        if prepare_workspaces is not None:
            assert leader.cache_ops.workspace is pending[0]
        assert not hasattr(leader, "_ascend_profiled_cache_ops")
        follower = NS(**vars(leader))
        return leader, backend.create_cache_ops(follower)

    first, follower = group()
    second, _ = group()
    assert follower.workspace is first.cache_ops.workspace
    assert follower.host_mapping is first.registered_host_pool._hisparse_registration
    assert second.cache_ops.workspace is not first.cache_ops.workspace
    first_workspace, second_workspace = first.cache_ops.workspace, second.cache_ops.workspace
    assert first_workspace.first_refs.shape == (max_rows * union_rows,)
    assert first_workspace.errors.shape == (max_rows, 8)
    assert first_workspace.max_union_rows == union_rows
    assert not torch.is_inference(first_workspace.first_refs)
    assert not torch.is_inference(first_workspace.errors)
    first_workspace.errors.fill_(17)
    first_workspace.first_refs.fill_(13)
    second_workspace.first_refs.zero_()
    assert torch.all(follower.workspace.first_refs == 13)
    assert torch.count_nonzero(second_workspace.first_refs) == 0


@pytest.mark.parametrize("topk", [0, 17])
def test_resolver_scratch_rejects_topk_outside_hot_capacity(topk):
    with pytest.raises(ValueError, match="TopK capacity"):
        hisparse_ops.ResolverWorkspace(4, 16, 8, torch.device("cpu"), max_topk=topk)


def test_missing_native_resolver_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(hisparse_ops.torch.ops, "_C_ascend", NS())
    with pytest.raises(RuntimeError, match="requires its compiled AscendC resolver"):
        hisparse_ops._load_resolver()


@pytest.fixture
def indexer_spec():
    return AscendSFAIndexerCacheSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32, block_stride_alignment=3
    )


@pytest.mark.parametrize("role", [SparseCacheRole.SPARSE, SparseCacheRole.INDEXER])
def test_indexer_spec_merge_retains_core_cache_role(indexer_spec, role):
    spec = replace(indexer_spec, cache_role=role)
    merged = AscendSFAIndexerCacheSpec.merge([spec, spec])
    assert merged.cache_role is role
    assert merged.block_stride_alignment == spec.block_stride_alignment


@pytest.mark.parametrize("blocks,stride", [(1, 1), (3, 2), (5, 3)])
def test_expanded_indexer_alias_preserves_shared_slab_offsets(indexer_spec, blocks, stride):
    spec = indexer_spec
    page = 16
    backing = torch.arange(blocks * stride * page + page, dtype=torch.float32)
    original = backing.as_strided((blocks, 1, 4, 4), (stride * page, page, 4, 1), storage_offset=page)
    layer = NS()
    caches = {"indexer": original}
    _bind_indexer_views(caches, {"indexer": spec}, {"indexer": layer})
    view = caches["indexer"][0]
    assert view.data_ptr() == original.data_ptr()
    assert view.storage_offset() == original.storage_offset()
    assert view.untyped_storage().data_ptr() == backing.data_ptr()
    assert layer.hisparse_block_stride == stride
    for block in range(blocks):
        torch.testing.assert_close(view[block * stride, :, 0], original[block, 0])
    view[(blocks - 1) * stride, 1, 0, 2] = -73
    assert original[-1, 0, 1, 2] == -73


@pytest.mark.parametrize("c8", [False, True])
def test_indexer_page_alignment_preserves_input_specs(indexer_spec, c8):
    source = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32)
    specs = {"source": source, "indexer": indexer_spec}
    packed_row_bytes = None
    if c8:
        indexer_spec = replace(
            indexer_spec,
            block_size=128,
            head_size=128,
            dtype=torch.float8_e4m3fn,
            scale_dim=1,
            scale_dtype=torch.float32,
            cache_sparse_li_c8=True,
        )
        packed_row_bytes = 656
        specs.update(
            indexer=indexer_spec,
            resident=HiSparseResidentSpec(block_size=128, page_size=128 * 656, block_stride_alignment=128),
            hot=HiSparseHotSpec(block_size=128, page_size=128 * 656, blocks_per_request=1, block_stride_alignment=128),
        )
    result = align_hisparse_indexer_specs(specs, packed_row_bytes=packed_row_bytes)
    assert result["source"] is source
    assert specs["indexer"] is indexer_spec and indexer_spec.block_stride_alignment == 3
    if c8:
        alignment = math.lcm(*(spec.get_block_stride_alignment() for name, spec in result.items() if name != "source"))
        assert alignment % 656 == 0 and alignment % 128 == 0
        assert result["indexer"].page_size_padded == 128 * (128 + 4)
        assert indexer_spec.page_size_padded is None
        for name in ("resident", "hot"):
            assert specs[name].block_stride_alignment == 128
            assert packed_row_bytes is not None
            assert result[name].block_stride_alignment == math.lcm(128, packed_row_bytes)
    else:
        assert result["indexer"].get_block_stride_alignment() == math.lcm(3, indexer_spec.page_size_bytes)


@pytest.mark.parametrize("field", ["scale_dim", "cache_sparse_li_c8", "cache_sparse_li_c4"])
def test_unsupported_indexer_formats_are_rejected_before_pool_binding(indexer_spec, field):
    spec = replace(indexer_spec, **{field: 1})
    config = NS(kv_cache_groups=[NS(layer_names=["indexer"], kv_cache_spec=spec)])
    with pytest.raises(ValueError, match="C4|C8|scales"):
        AscendHiSparseBackend().validate_kv_cache_config(config)


def test_capture_metadata_prevents_warmup_host_mirrors(monkeypatch):
    capture_metadata = NS(hisparse_for_capture=False)
    monkeypatch.setattr(
        AscendSFAMetadataBuilder, "build_for_cudagraph_capture", lambda *args, **kwargs: capture_metadata
    )
    builder = AscendHiSparseMetadataBuilder.__new__(AscendHiSparseMetadataBuilder)
    builder.build_for_cudagraph_capture(NS())
    real_metadata = NS(hisparse_for_capture=False)
    submitted = []
    cache = NS(finish_kv_update=lambda: submitted.append(True))
    finish_hisparse_kv_update(cache, capture_metadata)
    assert submitted == []
    finish_hisparse_kv_update(cache, real_metadata)
    assert submitted == [True]


def test_c8_packing_preserves_opaque_rope_scale_and_nan_payloads():
    rows = 3
    raw = [torch.arange(rows * width).to(torch.uint8).reshape(rows, width) for width in (512, 128, 16)]
    components = [values.view(torch.float8_e4m3fn) for values in raw]
    packed = pack_c8_components(*components)
    assert packed.dtype == torch.uint8 and packed.shape == (rows, 656)
    torch.testing.assert_close(packed, torch.cat(raw, dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("stride_rows", [128, 136])
def test_c8_storage_mapping_preserves_shared_slab_offsets(stride_rows):
    pages, block, width, offset = 3, 128, 656, 1312
    backing = torch.full((offset + pages * stride_rows * width,), 17, dtype=torch.uint8)
    cache = backing.view(torch.float8_e4m3fn).as_strided(
        (pages, block, width), (stride_rows * width, width, 1), storage_offset=offset
    )
    view, actual_block, actual_stride, capacity = c8_storage_view(cache)
    assert (actual_block, actual_stride, capacity) == (block, stride_rows, pages * block)
    assert view.data_ptr() == cache.data_ptr()
    assert view.untyped_storage().data_ptr() == backing.data_ptr()
    indices = torch.tensor([[0, 127, 128, 255, 256, 383, -1, 384]], dtype=torch.int32)
    mapped = map_c8_physical_indices(indices, actual_block, actual_stride, capacity)
    expected = torch.tensor(
        [[0, 127, stride_rows, stride_rows + 127, 2 * stride_rows, 2 * stride_rows + 127, -1, -1]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(mapped, expected, rtol=0, atol=0)
    view[2 * stride_rows, 0, 1:2].view(torch.uint8).fill_(123)
    assert backing[offset + 2 * stride_rows * width + 1] == 123


@pytest.mark.parametrize("pages,stride_rows,max_rows", [(2, 128, 9), (3, 136, 37)])
def test_c8_decode_lengths_follow_binding_geometry_and_survive_rebind(
    monkeypatch, pages, stride_rows, max_rows, prepare_workspaces
):
    monkeypatch.setattr(
        hisparse_ops, "_load_resolver", lambda: NS(hisparse_resolver_validate_scratch=lambda hot, union: None)
    )
    backend = AscendHiSparseBackend()
    group = NS(copy_stream=NS(device=torch.device("cpu")))
    runtime = NS(
        index_group=group,
        max_swap_rows=max_rows,
        region_stride=128,
        max_union_rows=64,
        device=torch.device("cpu"),
        kv_dtype=torch.float8_e4m3fn,
        registered_host_pool=NS(_hisparse_registration=object()),
    )
    group.shared_topk = NS(device_topk_rows=torch.empty((max_rows, runtime.max_union_rows), dtype=torch.int32))
    group.leader = runtime

    def bind(num_pages):
        backing = torch.empty(num_pages * stride_rows * 656, dtype=torch.float8_e4m3fn)
        cache = backing.as_strided((num_pages, 128, 656), (stride_rows * 656, 656, 1))
        runtime.hot = NS(cache=cache, attention_cache=cache, block_size=128)
        if prepare_workspaces is not None:
            prepare_workspaces([runtime])
            pending = runtime._ascend_profiled_cache_ops
            prepare_workspaces([runtime])
            assert runtime._ascend_profiled_cache_ops is pending
        ops = backend.create_cache_ops(runtime)
        if prepare_workspaces is not None:
            assert ops.workspace is pending[0]
        assert not hasattr(runtime, "_ascend_profiled_cache_ops")
        return ops

    first = bind(pages)
    query, key = first.attention_lengths
    for tensor in (first.workspace.first_refs, first.workspace.errors, query, key):
        assert not torch.is_inference(tensor)
    first.workspace.errors.fill_(17)
    for rows in (1, max_rows, 3, 1):
        torch.testing.assert_close(query[rows : rows + 1], torch.tensor([rows], dtype=torch.int32))
    expected = (pages - 1) * stride_rows + 128
    torch.testing.assert_close(key, torch.tensor([expected], dtype=torch.int32))
    second = bind(pages + 1)
    if prepare_workspaces is not None:
        assert second.workspace is first.workspace
    torch.testing.assert_close(second.attention_lengths[1], torch.tensor([expected + stride_rows], dtype=torch.int32))
    torch.testing.assert_close(key, torch.tensor([expected], dtype=torch.int32))
    assert key.data_ptr() != second.attention_lengths[1].data_ptr()


def test_c8_compaction_preserves_duplicate_order_and_empty_rows():
    indices = torch.tensor([[7, -1, 2, 7, -1, 0], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    expected = torch.tensor([[7, 2, 7, 0, -1, -1], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(compact_c8_indices(indices), expected, rtol=0, atol=0)


def test_bf16_hisparse_indexer_passes_through_page_scatter_indices(monkeypatch):
    cache = torch.empty((2, 4, 128), dtype=torch.bfloat16)
    keys = torch.zeros((2, 128), dtype=torch.bfloat16)
    slots = torch.tensor([2, 4], dtype=torch.int32)
    scatter_indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    writer = AscendHiSparseIndexerBackend.__new__(AscendHiSparseIndexerBackend)
    torch.nn.Module.__init__(writer)
    writer.k_cache = NS(kv_cache=(cache,))
    writer.enable_sparse_li_c8 = False
    writer.enable_sparse_li_c4 = False
    calls = []

    def write_cache_tensor(cache_arg, updates, slots_arg, indices_arg):
        calls.append(indices_arg)
        assert cache_arg is cache
        assert slots_arg is slots
        torch.testing.assert_close(updates, keys)

    monkeypatch.setattr(writer, "_write_cache_tensor", write_cache_tensor)
    writer.write_cache(keys, None, slots, scatter_indices)
    assert len(calls) == 1 and calls[0] is scatter_indices


@pytest.mark.parametrize("padding", [0, 256])
def test_c8_indexer_views_and_writes_preserve_neighboring_slab_bytes(monkeypatch, indexer_spec, padding):
    spec = replace(
        indexer_spec,
        block_size=128,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
    )
    pages, offset = 3, 256
    key_bytes = 128 * 128
    stride = spec.page_size_bytes + padding
    raw = torch.full((offset + pages * stride + 256,), 165, dtype=torch.uint8)
    tensor_config = KVCacheTensor(
        size=raw.numel(), layers=["indexer"], layer_stride=spec.page_size_bytes, block_stride=stride, offset=offset
    )
    (cache,) = create_kv_cache_views(raw.view(torch.int8), spec, pages, KVCacheLayout.BLHNC, tensor_config)
    assert cache.shape == (pages, 1, 128, 132)
    layer = NS()
    caches = {"indexer": cache}
    _bind_indexer_views(caches, {"indexer": spec}, {"indexer": layer})
    key, scale = caches["indexer"]
    assert key.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr() == raw.data_ptr()
    assert layer.hisparse_block_stride == 1
    slots = torch.tensor([127, 0, 128, 383, -1, 256], dtype=torch.int32)
    values_raw = torch.arange(6 * 128).to(torch.uint8).reshape(6, 128)
    values = values_raw.view(torch.float8_e4m3fn)
    scales = torch.arange(6, dtype=torch.float32).reshape(6, 1) + 0.5

    def store(quant, scales, slots, cache, scale_cache, *, block_size, key_block_stride, scale_block_stride):
        # Emulate only the device-store boundary; production views and addresses
        # are checked against the independent byte oracle below.
        assert block_size == 128 and key_block_stride == stride and scale_block_stride == stride // 4
        assert cache.dtype == torch.int8 and scale_cache.dtype == torch.int32
        for i, slot in enumerate(slots.tolist()):
            if 0 <= slot < pages * block_size:
                page, row = divmod(slot, block_size)
                cache[page, row, 0].copy_(quant[i])
                scale_cache[page, row, 0, 0] = scales[i]

    monkeypatch.setattr(sfa_indexer_store, "store_indexer_key_scale", store)
    writer = AscendHiSparseIndexerBackend.__new__(AscendHiSparseIndexerBackend)
    torch.nn.Module.__init__(writer)
    writer.k_cache = NS(kv_cache=caches["indexer"])
    writer.enable_sparse_li_c8 = True
    writer.enable_sparse_li_c4 = False
    expected = raw.clone()
    scale_bytes = scales.view(torch.uint8).reshape(6, 4)
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        page, row = divmod(slot, 128)
        start = offset + page * stride + row * 128
        expected[start : start + 128] = values_raw[i]
        start = offset + page * stride + key_bytes + row * 4
        expected[start : start + 4] = scale_bytes[i]
    scatter_indices = torch.stack((slots // spec.block_size, slots % spec.block_size), dim=-1)
    writer.write_cache(values, scales, slots, scatter_indices)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
