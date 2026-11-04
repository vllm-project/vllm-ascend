# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-isolated CPU boundary checks; these do not qualify NPU execution."""

import ast
import ctypes
import inspect
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


def test_merged_core_resolver_call_keeps_optional_outputs_in_their_current_positions():
    operation = definitions("ops/hisparse.py", {"HiSparseNPUOps"})["HiSparseNPUOps"].hisparse_resolve_residency
    # Argument positions from the reviewed merged Core Runtime.swap_in call.
    # The former pre-merge method rejected this call or shifted output buffers.
    arguments = [object() for _ in range(26)]
    bound = inspect.signature(operation).bind(object(), *arguments).arguments
    for name, position in {
        "miss_mask": 10,
        "stats": 11,
        "source_block_size": 16,
        "resolved_global_indices": 17,
        "valid_counts": 18,
        "resident_null_block": 24,
        "num_valid_rows": 25,
    }.items():
        assert bound[name] is arguments[position]


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
    allocated = []

    class Workspace:
        def __init__(self, **geometry):
            self.geometry = geometry
            allocated.append(self)

    class Ops:
        def __init__(self, mapping, workspace):
            self.mapping, self.workspace = mapping, workspace

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.hisparse", NS(ResolverWorkspace=Workspace, HiSparseNPUOps=Ops))
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
    assert follower.mapping is first.registered_host_pool._hisparse_registration
    assert second.cache_ops.workspace is not first.cache_ops.workspace
    assert len(allocated) == 2
    assert allocated[0].geometry == dict(
        max_rows=max_rows, hot_size=hot_rows, max_union_rows=union_rows, device=torch.device("cpu")
    )


@dataclass(frozen=True)
class IndexerSpec:
    block_size: int = 4
    num_kv_heads: int = 1
    head_size: int = 4
    block_stride_alignment: int = 3
    scale_dim: int = 0
    cache_sparse_li_c8: bool = False
    cache_sparse_li_c4: bool = False

    @property
    def page_size_bytes(self):
        return self.block_size * self.num_kv_heads * self.head_size * torch.float32.itemsize

    def get_block_stride_alignment(self):
        return self.block_stride_alignment


def binding_functions():
    return definitions(
        "distributed/kv_transfer/hisparse/binding.py",
        {"align_hisparse_indexer_specs", "get_hisparse_indexer_specs", "_bind_indexer_views"},
        AscendSFAIndexerCacheSpec=IndexerSpec,
        UniformTypeKVCacheSpecs=type("Uniform", (), {}),
        math=math,
        replace=replace,
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
    source, indexer = object(), IndexerSpec()
    values = {"source": source, "indexer": indexer}
    result = functions["align_hisparse_indexer_specs"](values)
    assert result["source"] is source
    assert result["indexer"].get_block_stride_alignment() == math.lcm(3, indexer.page_size_bytes)
    assert values["indexer"] is indexer and indexer.block_stride_alignment == 3


@pytest.mark.parametrize("field", ["scale_dim", "cache_sparse_li_c8", "cache_sparse_li_c4"])
def test_unsupported_indexer_formats_are_rejected_before_pool_binding(field):
    functions = binding_functions()
    spec = replace(IndexerSpec(), **{field: 1})
    config = NS(kv_cache_groups=[NS(layer_names=["indexer"], kv_cache_spec=spec)])
    with pytest.raises(ValueError, match="unquantized"):
        functions["get_hisparse_indexer_specs"](config)


@pytest.mark.parametrize("hash_lookahead", [0, 7])
def test_hisparse_roles_preserve_core_admission_and_explicit_hash_budget(hash_lookahead):
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
        num_hash_lookahead_tokens=hash_lookahead,
        enable_caching=False,
    )
    assert result is marker and len(calls) == 1
    assert calls[0]["kv_cache_config"] is config
    assert calls[0]["max_in_flight_tokens"] == 129
    assert calls[0]["num_hash_lookahead_tokens"] == hash_lookahead
    assert calls[0]["enable_caching"] is False
    assert "eagle_attn_layer_names" not in calls[0]
