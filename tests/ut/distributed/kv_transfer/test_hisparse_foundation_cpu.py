# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-isolated CPU boundary checks; these do not qualify NPU execution."""

import ast
import ctypes
import importlib.util
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def definitions(relative, names, **bindings):
    path = ROOT / "vllm_ascend" / relative
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch, **bindings}
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def backend_type():
    return definitions(
        "distributed/kv_transfer/hisparse/backend.py", {"AscendHiSparseBackend"}, HiSparseBackend=object
    )["AscendHiSparseBackend"]


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

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.hisparse", NS(MappedHostTensor=Registration))
    backend = backend_type()()
    backend.pin_tensor(tensor)
    registration = tensor._hisparse_registration
    assert registration.tensor is tensor
    with pytest.raises(RuntimeError, match="driver did not retire"):
        backend.unpin_tensor(tensor)
    assert tensor._hisparse_registration is registration
    backend.unpin_tensor(tensor)
    assert not hasattr(tensor, "_hisparse_registration")
    assert events == ["close", "close"]


def test_mapped_addresses_retain_offsets_and_reject_expired_or_external_views():
    cls = definitions("ops/hisparse.py", {"MappedHostTensor"}, ctypes=ctypes)["MappedHostTensor"]
    backing = torch.arange(128, dtype=torch.uint8)
    mapping = cls.__new__(cls)
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
    monkeypatch, max_rows, hot_rows, union_rows
):
    operations = definitions("ops/hisparse.py", {"ResolverWorkspace", "HiSparseNPUOps"})
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.hisparse", NS(**operations))
    backend = backend_type()()

    def group():
        owner = NS()
        leader = NS(
            index_group=owner,
            max_swap_rows=max_rows,
            region_stride=hot_rows,
            max_union_rows=union_rows,
            device=torch.device("cpu"),
            registered_host_pool=NS(_hisparse_registration=object()),
        )
        owner.leader = leader
        leader.cache_ops = backend.create_cache_ops(leader)
        follower = NS(**vars(leader))
        return leader, backend.create_cache_ops(follower)

    first, follower = group()
    second, _ = group()
    assert follower.workspace is first.cache_ops.workspace
    assert follower.host_mapping is first.registered_host_pool._hisparse_registration
    assert second.cache_ops.workspace is not first.cache_ops.workspace
    first_workspace, second_workspace = first.cache_ops.workspace, second.cache_ops.workspace
    assert first_workspace.row_keys.shape == (max_rows * hot_rows,)
    assert first_workspace.cache_keys.shape == (max_rows, hot_rows)
    assert first_workspace.max_union_rows == union_rows
    first_workspace.row_keys.fill_(13)
    second_workspace.row_keys.zero_()
    assert torch.all(follower.workspace.row_keys == 13)
    assert torch.count_nonzero(second_workspace.row_keys) == 0


@dataclass(frozen=True)
class IndexerSpec:
    block_size: int = 4
    num_kv_heads: int = 1
    head_size: int = 4
    block_stride_alignment: int = 3
    scale_dim: int = 0
    cache_sparse_li_c8: bool = False
    cache_sparse_li_c4: bool = False
    dtype: torch.dtype = torch.float32
    scale_dtype: torch.dtype = torch.float32
    sfa_dcp_replicated_indexer_size: int = 1
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * self.dtype.itemsize + self.scale_dim * self.scale_dtype.itemsize)
        )

    def get_block_stride_alignment(self):
        return self.block_stride_alignment


def binding_functions(**bindings):
    return definitions(
        "distributed/kv_transfer/hisparse/binding.py",
        {
            "align_hisparse_indexer_specs",
            "get_hisparse_indexer_specs",
            "_bind_indexer_views",
            "_validate_indexer_spec",
            "_quantized_indexer_views",
            "write_hisparse_indexer_c8",
        },
        AscendSFAIndexerCacheSpec=IndexerSpec,
        UniformTypeKVCacheSpecs=type("Uniform", (), {}),
        math=math,
        replace=replace,
        **bindings,
    )


@pytest.mark.parametrize("blocks,stride", [(1, 1), (3, 2), (5, 3)])
def test_expanded_indexer_alias_preserves_shared_slab_offsets(blocks, stride):
    functions = binding_functions()
    spec = IndexerSpec()
    page = 16
    backing = torch.arange(blocks * stride * page + page, dtype=torch.float32)
    original = backing.as_strided((blocks, 1, 4, 4), (stride * page, page, 4, 1), storage_offset=page)
    layer = NS()
    caches = {"indexer": original}
    functions["_bind_indexer_views"](caches, {"indexer": spec}, {"indexer": layer})
    view = caches["indexer"][0]
    assert view.data_ptr() == original.data_ptr()
    assert view.storage_offset() == original.storage_offset()
    assert view.untyped_storage().data_ptr() == backing.data_ptr()
    assert layer.hisparse_block_stride == stride
    for block in range(blocks):
        torch.testing.assert_close(view[block * stride, :, 0], original[block, 0])
    view[(blocks - 1) * stride, 1, 0, 2] = -73
    assert original[-1, 0, 1, 2] == -73


def test_alignment_updates_only_indexer_specs_without_changing_inputs():
    functions = binding_functions()
    source, indexer = NS(get_block_stride_alignment=lambda: 1), IndexerSpec()
    values = {"source": source, "indexer": indexer}
    result = functions["align_hisparse_indexer_specs"](values)
    assert result["source"] is source
    assert result["indexer"].get_block_stride_alignment() == math.lcm(3, indexer.page_size_bytes)
    assert values["indexer"] is indexer and indexer.block_stride_alignment == 3


def test_c8_page_alignment_reserves_scale_bytes_and_whole_main_rows():
    @dataclass(frozen=True)
    class StorageSpec:
        block_stride_alignment: int = 128

        def get_block_stride_alignment(self):
            return self.block_stride_alignment

    functions = binding_functions(HiSparseResidentSpec=StorageSpec, HiSparseHotSpec=StorageSpec)
    storage = StorageSpec()
    indexer = IndexerSpec(
        block_size=128, head_size=128, dtype=torch.float8_e4m3fn, scale_dim=1, cache_sparse_li_c8=True
    )
    result = functions["align_hisparse_indexer_specs"](
        {"resident": storage, "hot": storage, "indexer": indexer}, packed_row_bytes=656
    )
    stride_alignment = math.lcm(*(spec.get_block_stride_alignment() for spec in result.values()))
    assert stride_alignment % 656 == 0 and stride_alignment % 128 == 0
    assert result["indexer"].page_size_padded == 128 * (128 + 4)
    assert storage.block_stride_alignment == 128 and indexer.page_size_padded is None


@pytest.mark.parametrize("field", ["scale_dim", "cache_sparse_li_c8", "cache_sparse_li_c4"])
def test_unsupported_indexer_formats_are_rejected_before_pool_binding(field):
    functions = binding_functions()
    spec = replace(IndexerSpec(), **{field: 1})
    config = NS(kv_cache_groups=[NS(layer_names=["indexer"], kv_cache_spec=spec)])
    with pytest.raises(ValueError, match="C4|C8|scales"):
        functions["get_hisparse_indexer_specs"](config)


@pytest.mark.parametrize("enable_caching", [False, True])
def test_hisparse_roles_preserve_core_admission_and_explicit_hash_budget(enable_caching):
    calls = []
    marker = object()
    functions = definitions(
        "patch/platform/patch_kv_cache_coordinator.py",
        {"get_kv_cache_coordinator"},
        KVCacheGroupRole=NS(HISPARSE_SOURCE="source"),
        _select_kv_token_budget=lambda maximum, inflight: maximum if inflight is None else inflight,
        _orig_get_kv_cache_coordinator=lambda **kwargs: (calls.append(kwargs), marker)[1],
    )
    config = NS(kv_cache_groups=[NS(role="source"), NS(role="resident"), NS(role="hot")])
    result = functions["get_kv_cache_coordinator"](
        config,
        max_model_len=65536,
        max_in_flight_tokens=129,
        num_hash_lookahead_tokens=7,
        enable_caching=enable_caching,
    )
    assert result is marker and len(calls) == 1
    assert calls[0]["kv_cache_config"] is config
    assert calls[0]["max_in_flight_tokens"] == 129
    assert calls[0]["num_hash_lookahead_tokens"] == 7
    assert calls[0]["enable_caching"] is enable_caching
    assert "eagle_attn_layer_names" not in calls[0]


def test_capture_metadata_prevents_warmup_host_mirrors():
    class Parent:
        def build_for_cudagraph_capture(self, common, **kwargs):
            assert common is common_input and kwargs["draft_index"] is draft
            return self.build()

        def build(self):
            return NS(hisparse_for_capture=False)

    methods = definitions(
        "attention/sfa_hisparse.py",
        {"AscendHiSparseMetadataBuilder", "finish_hisparse_kv_update"},
        AscendSFAMetadataBuilder=Parent,
    )
    cls = methods["AscendHiSparseMetadataBuilder"]
    builder = cls.__new__(cls)
    common_input, draft = object(), object()
    capture_metadata = builder.build_for_cudagraph_capture(common_input, draft_index=draft)
    real_metadata = builder.build()
    finish = methods["finish_hisparse_kv_update"]
    submitted = []
    cache = NS(finish_kv_update=lambda: submitted.append(True))
    for _ in range(5):
        finish(cache, capture_metadata)
    assert submitted == []
    finish(cache, real_metadata)
    assert submitted == [True]


def test_real_forward_mirror_errors_remain_visible():
    finish = definitions("attention/sfa_hisparse.py", {"finish_hisparse_kv_update"})["finish_hisparse_kv_update"]

    def duplicate():
        raise RuntimeError("HiSparse layer 0 mirrored twice.")

    with pytest.raises(RuntimeError, match="mirrored twice"):
        finish(NS(finish_kv_update=duplicate), NS(hisparse_for_capture=False))


def c8_module():
    path = ROOT / "vllm_ascend/ops/hisparse_c8.py"
    spec = importlib.util.spec_from_file_location("hisparse_c8_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c8_packing_preserves_opaque_rope_scale_and_nan_payloads():
    module = c8_module()
    rows = 3
    raw = [torch.arange(rows * width).to(torch.uint8).reshape(rows, width) for width in (512, 128, 16)]
    components = [values.view(torch.float8_e4m3fn) for values in raw]
    packed = module.pack_c8_components(*components)
    assert packed.dtype == torch.uint8 and packed.shape == (rows, 656)
    torch.testing.assert_close(packed, torch.cat(raw, dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("stride_rows", [128, 136])
def test_c8_storage_mapping_preserves_shared_slab_offsets(stride_rows):
    module = c8_module()
    pages, block, width, offset = 3, 128, 656, 1312
    backing = torch.full((offset + pages * stride_rows * width,), 17, dtype=torch.uint8)
    cache = backing.view(torch.float8_e4m3fn).as_strided(
        (pages, block, width), (stride_rows * width, width, 1), storage_offset=offset
    )
    view, actual_block, actual_stride, capacity = module.c8_storage_view(cache)
    assert (actual_block, actual_stride, capacity) == (block, stride_rows, pages * block)
    assert view.data_ptr() == cache.data_ptr()
    assert view.untyped_storage().data_ptr() == backing.data_ptr()
    indices = torch.tensor([[0, 127, 128, 255, 256, 383, -1, 384]], dtype=torch.int32)
    mapped = module.map_c8_physical_indices(indices, actual_block, actual_stride, capacity)
    expected = torch.tensor(
        [[0, 127, stride_rows, stride_rows + 127, 2 * stride_rows, 2 * stride_rows + 127, -1, -1]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(mapped, expected, rtol=0, atol=0)
    view[2 * stride_rows, 0, 1:2].view(torch.uint8).fill_(123)
    assert backing[offset + 2 * stride_rows * width + 1] == 123


def test_c8_compaction_preserves_duplicate_order_and_empty_rows():
    module = c8_module()
    indices = torch.tensor([[7, -1, 2, 7, -1, 0], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    expected = torch.tensor([[7, 2, 7, 0, -1, -1], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(module.compact_c8_indices(indices), expected, rtol=0, atol=0)


@pytest.mark.parametrize("padding", [0, 256])
def test_c8_indexer_views_and_writes_preserve_neighboring_slab_bytes(padding):
    functions = binding_functions()
    spec = IndexerSpec(block_size=128, head_size=128, dtype=torch.float8_e4m3fn, scale_dim=1, cache_sparse_li_c8=True)
    pages, offset = 3, 256
    key_bytes = 128 * 128
    stride = spec.page_size_bytes + padding
    raw = torch.full((offset + pages * stride + 256,), 165, dtype=torch.uint8)
    cache = raw.view(torch.float8_e4m3fn).as_strided(
        (pages, 1, 128, 128), (stride, key_bytes, 128, 1), storage_offset=offset
    )
    layer = NS()
    caches = {"indexer": cache}
    functions["_bind_indexer_views"](caches, {"indexer": spec}, {"indexer": layer})
    key, scale = caches["indexer"]
    assert key.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr() == raw.data_ptr()
    assert layer.hisparse_block_stride == 1
    assert layer.hisparse_indexer_payload_format == "ascend_indexer_e4m3fn_key_fp32_scale_v1"
    slots = torch.tensor([127, 0, 128, 383, -1, 256], dtype=torch.int32)
    values_raw = torch.arange(6 * 128).to(torch.uint8).reshape(6, 128)
    values = values_raw.view(torch.float8_e4m3fn)
    scales = torch.arange(6, dtype=torch.float32).reshape(6, 1) + 0.5

    def scatter(destination, indexes, source):
        indexes = indexes.flatten()
        valid = indexes >= 0
        destination[indexes[valid].long()] = source[valid]

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
    functions["write_hisparse_indexer_c8"](key, scale, values, scales, slots, scatter)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
