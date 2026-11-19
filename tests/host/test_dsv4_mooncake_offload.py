# SPDX-License-Identifier: Apache-2.0
"""Host-only execution of production planning/range methods without vLLM/NPU.

AST loading isolates framework imports, not the method bodies under test.
Run: python -m pytest --noconftest tests/host/test_dsv4_mooncake_offload.py
"""

import ast
import ctypes
import hashlib
import json
from math import lcm
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
STORE = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/"


def definitions(path, namespace, names=None):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (names is None or node.name in names)
    ]
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec", flags=0x1000000), namespace)


@dataclass(frozen=True)
class Spec:
    block_size: int = 128
    head_size: int = 8
    dtype: str = "bf16"
    tokens_per_state: int = 4
    model_version: str = "deepseek_v4"
    sliding_window: int | None = None

    @property
    def page_size_bytes(self):
        return self.block_size // self.tokens_per_state * self.head_size * 2


@dataclass
class Uniform:
    kv_cache_specs: dict


@dataclass
class Tensor:
    size: int
    layers: list
    offset: int = 0
    layer_stride: int = 0
    block_stride: int = 0
    host_resident: bool = False


@dataclass
class Cache:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list


@pytest.fixture
def ns():
    n = dict(
        dataclass=dataclass,
        field=field,
        replace=replace,
        Any=Any,
        re=re,
        np=np,
        defaultdict=defaultdict,
        hashlib=hashlib,
        json=json,
        asdict=asdict,
        logger=MagicMock(),
        KVCacheConfig=Cache,
        KVCacheTensor=Tensor,
        UniformTypeKVCacheSpecs=Uniform,
        KVCacheSpec=Spec,
        _NUM_SHARED_BUFFERS="layerwise_num_shared_buffers",
        _PREFETCH_LAYERS="layerwise_prefetch_layers",
        _INDEPENDENT_LAYERS="layerwise_independent_layers",
        _DEFAULT_MAX_PREFETCH_LAYERS=8,
        _INDEXER_CACHE_SUFFIX=".indexer.k_cache",
        AttentionSpec=Spec,
        AscendMLAAttentionSpec=Spec,
        AscendSlidingWindowMLASpec=type("SlidingWindowSpec", (), {}),
        MambaSpec=type("Mamba", (), {}),
        is_prefix_cacheable=lambda spec: True,
    )
    definitions(
        STORE + "layerwise/layerwise_cache_layout.py",
        n,
        {
            "LayerwiseCacheLayout",
            "NamedKVCacheSpec",
            "LayerwiseLayerCacheSpecs",
            "LayerwiseReuseLayout",
            "LayerwiseCacheTopology",
            "get_physical_layer_index",
            "_parse_int_config",
            "build_layerwise_cache_layout",
            "build_layerwise_reuse_layout",
            "_fold_main_packs",
        },
    )
    n["get_layerwise_reuse_config"] = lambda config: config.kv_connector_extra_config if config else None
    planning = dict(n)
    planning.update(lcm=lcm, round_up=lambda size, alignment: (size + alignment - 1) // alignment * alignment)
    planning["cache_view_alignment"] = lambda spec: 1
    definitions(STORE + "layerwise/layerwise_cache_layout.py", planning, {"get_layerwise_kv_cache_specs"})
    definitions(
        STORE + "layerwise/cache_views.py", planning, {"LayerwiseCacheConfig", "LayerwiseReuseLayout", "get_cache_views"}
    )
    definitions(STORE + "layerwise/cache_planner.py", planning, {"LayerwiseCachePlanner"})
    n["LayerwiseCachePlanner"] = planning["LayerwiseCachePlanner"]
    definitions("vllm_ascend/models/deepseek_v4/cache_config.py", n)
    definitions(STORE + "layerwise/metadata.py", n, {"SharedBlockData", "LayerRangeReqMeta", "get_partial_block_index"})
    definitions(STORE + "layerwise/kv_transfer.py", n, {"LayerBatchBuilder"})
    definitions(
        STORE + "backend/mooncake_offload.py",
        n,
        {"snapshot_key", "_load_entries", "_save_entries", "_continuation_window_start"},
    )
    definitions("vllm_ascend/models/deepseek_v4/cache_transfer.py", n)
    n["block_hash_to_str"] = lambda value: value.hex()
    return n


def configuration(buffers=2):
    return NS(
        kv_transfer_config=NS(
            kv_connector_extra_config=dict(layerwise_num_shared_buffers=buffers, layerwise_independent_layers=[0, 1])
        ),
        model_config=NS(get_num_layers=lambda parallel: 43, enforce_eager=True),
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=NS(prefix_cache_retention_interval=None),
        scheduler_config=NS(async_scheduling=False),
        speculative_config=None,
    )


def groups():
    # Full-history group intentionally mixes shared main and resident indexer.
    full, auxiliary = {}, {}
    for layer in range(43):
        prefix = f"model.layers.{layer}.self_attn"
        auxiliary[prefix + ".swa_cache"] = Spec(16, tokens_per_state=1, sliding_window=128)
        if layer < 2:
            continue
        ratio = 4 if layer % 2 == 0 else 128
        full[prefix] = Spec(32 * ratio, tokens_per_state=ratio)
        auxiliary[prefix + ".compressor.state_cache"] = Spec(2, tokens_per_state=1, sliding_window=2)
        if ratio == 4:
            full[prefix + ".indexer.k_cache"] = Spec(128, head_size=4)
            auxiliary[prefix + ".indexer.compressor.state_cache"] = Spec(2, tokens_per_state=1, sliding_window=2)
    return [NS(layer_names=list(specs), kv_cache_spec=Uniform(specs)) for specs in (full, auxiliary)]


@pytest.mark.parametrize("buffers", [1, 2, 3, 4])
def test_component_slots_and_physical_capacity(ns, buffers):
    gs = groups()
    plan = ns["build_deepseek_v4_reuse_config"](configuration(buffers), gs, 7)
    layout = plan.layerwise_reuse_layout
    assert layout.independent_layers == [0, 1]
    assert layout.has_layer_reuse
    assert any(len(sources) > 1 for sources in layout.prefetch_layer_map.values())
    for layer, sources in layout.prefetch_layer_map.items():
        assert all(source < layer for source in sources)
    owners = {name: index for index, tensor in enumerate(plan.kv_cache_tensors) for name in tensor.layers}
    assert len(owners) == sum(len(group.layer_names) for group in gs)
    for suffix in (".self_attn", ".swa_cache", ".indexer.k_cache", ".compressor.state_cache"):
        assert any(name.endswith(suffix) for name in plan.dsv4_shared_cache_names)
    assert all(layout.cache_layer_indices[name] >= 2 for name in plan.dsv4_shared_cache_names)
    for group in gs:
        seen = set()
        for name in group.layer_names:
            key = (owners[name], layout.cache_layer_indices[name])
            assert key not in seen
            seen.add(key)
    probe = ns["build_deepseek_v4_reuse_config"](configuration(buffers), gs, 1)
    assert sum(t.size for t in plan.kv_cache_tensors) == 7 * sum(t.size for t in probe.kv_cache_tensors)
    resident_config = configuration()
    resident_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = "all"
    resident = ns["build_deepseek_v4_reuse_config"](resident_config, gs, 1)
    assert sum(t.size for t in probe.kv_cache_tensors) < sum(t.size for t in resident.kv_cache_tensors)


def test_dspark_draft_is_excluded_from_rotation(ns):
    config = configuration()
    config.speculative_config = NS(method="dspark")
    cache_groups = groups()
    draft = "model.layers.43.self_attn"
    cache_groups[0].layer_names.append(draft)
    cache_groups[0].kv_cache_spec.kv_cache_specs[draft] = Spec()
    plan = ns["build_deepseek_v4_reuse_config"](config, cache_groups, 7)
    assert plan.layerwise_draft_layer_names == (draft,)
    assert draft in plan.dsv4_resident_cache_names
    assert draft not in plan.dsv4_shared_cache_names
    assert 43 not in plan.layerwise_reuse_layout.prefetch_layer_map


def test_all_resident_control(ns):
    config = configuration()
    config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = "all"
    plan = ns["build_deepseek_v4_reuse_config"](config, groups(), 1)
    assert not plan.layerwise_reuse_layout.has_layer_reuse
    assert not plan.dsv4_shared_cache_names
    assert all(len(t.layers) == 1 for t in plan.kv_cache_tensors)


def test_layout_identity_changes_with_format_but_not_ring_size(ns):
    gs = groups()
    a = ns["build_deepseek_v4_reuse_config"](configuration(1), gs, 2)
    b = ns["build_deepseek_v4_reuse_config"](configuration(4), gs, 9)
    assert a.dsv4_component_identity == b.dsv4_component_identity
    name = gs[0].layer_names[0]
    gs[0].kv_cache_spec.kv_cache_specs[name] = replace(gs[0].kv_cache_spec.kv_cache_specs[name], dtype="fp8")
    assert ns["component_layout_identity"](gs) != a.dsv4_component_identity


@pytest.mark.parametrize(
    "field_name,value",
    [("pipeline_parallel_size", 2), ("prefill_context_parallel_size", 2), ("decode_context_parallel_size", 2)],
)
def test_reject_unsupported_parallel(ns, field_name, value):
    config = configuration()
    setattr(config.parallel_config, field_name, value)
    with pytest.raises(ValueError, match="PP=PCP=DCP"):
        ns["build_deepseek_v4_reuse_config"](config, groups(), 1)


@pytest.mark.parametrize("mode", ["speculative", "graph"])
def test_reject_unsupported_execution(ns, mode):
    config = configuration()
    if mode == "speculative":
        config.speculative_config = NS(method="unsupported")
    else:
        config.model_config.enforce_eager = False
    with pytest.raises(ValueError):
        ns["build_deepseek_v4_reuse_config"](config, groups(), 1)


def test_mixed_batch_restores_only_shared_bytes_for_continuation(ns):
    # Indexer and main occupy disjoint entries of one logical remote object.
    indexer = np.full((4, 4), 11, dtype=np.uint8)
    main = np.full((4, 8), 22, dtype=np.uint8)
    db = NS(
        group_block_len={0: [4, 8]},
        group_block_stride={0: [4, 8]},
        group_kv_caches_base_addr={0: [indexer.ctypes.data, main.ctypes.data]},
        group_layer_cache_entry_offsets={0: [0, 2]},
    )
    builder = ns["LayerBatchBuilder"](db, 12, 1)
    builder.shared_cache_entries = [False, True]
    shared = ns["SharedBlockData"](
        np.array([1, 2]),
        None,
        ["continued", "external"],
        [False, False],
        block_keys=["a", "b"],
        previous_step_rows=[True, False],
    )
    result = builder.build_addrs(shared, 0)
    assert result.all_offsets == [[4], [0, 4]]
    assert result.all_sizes == [[8], [4, 8]]
    for buffers, sizes in zip(result.all_buffers, result.all_sizes, strict=True):
        for address, size in zip(buffers, sizes, strict=True):
            ctypes.memset(address, 33, size)
    assert (indexer[1] == 11).all()  # continued resident data survives
    assert (indexer[2] == 33).all()  # external prefix restores indexer
    assert (main[1:3] == 33).all()


def test_live_table_replaces_holes_and_copies_without_new_blocks():
    tree = ast.parse((ROOT / STORE / "metadata.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RequestTracker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sync_live_block_ids")
    cls.body = [method]
    cls.decorator_list = []
    ns = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "tracker", "exec", flags=0x1000000),
        ns,
    )
    tracker = ns["RequestTracker"]()
    tracker.req_id = "request"
    tracker.allocated_block_ids_by_group = [[10, 20], [9]]
    live = [[0, 20, 35], [0, 18]]
    tracker.sync_live_block_ids(NS(kv_connector_block_state=NS(get_block_ids=lambda req: live)))
    live[0][1] = 99
    assert tracker.allocated_block_ids_by_group == [[0, 20, 35], [0, 18]]
    with pytest.raises(RuntimeError, match="requires"):
        tracker.sync_live_block_ids(NS())
    with pytest.raises(RuntimeError, match="no live"):
        tracker.sync_live_block_ids(NS(kv_connector_block_state=NS(get_block_ids=lambda req: None)))


def test_tail_snapshot_uses_raw_tokens_and_skips_resident_restore(ns):
    worker = NS(
        group_uses_align_state=[False],
        grouped_block_size=[128],
        use_eagle=False,
        kv_cache_config=NS(dsv4_shared_cache_names=("main",)),
        _group_reuses_buffer=lambda group: True,
        _layerwise_load_snapshot_start=lambda *args: 1,
        _make_layerwise_full_key=lambda group, key: f"{group}:{key}",
    )
    spec = NS(
        can_load=True,
        layerwise_previous_step=True,
        kvpool_cached_tokens=131,
        kvpool_store_skip_tokens=None,
        layerwise_snapshot_end_token=131,
    )
    request = NS(req_id="request", load_spec=spec, block_ids_by_group=[[4, 8]], layerwise_request_id="generation")
    entries = ns["_load_entries"](worker, request, 0, [b"hash"], None)
    assert entries == [("0:68617368", 0), ("0:offload_v1:generation:131:1", 1)]
    worker._group_reuses_buffer = lambda group: False
    assert ns["_load_entries"](worker, request, 0, [b"hash"], None) == []
    worker._group_reuses_buffer = lambda group: True
    spec.layerwise_previous_step = False
    worker._layerwise_load_snapshot_start = lambda *args: None
    with pytest.raises(ValueError, match="exact snapshot"):
        ns["_load_entries"](worker, request, 0, [b"hash"], None)


def test_glm_main_pack_planner_retains_group_predecessors(ns):
    names = [f"model.layers.{layer}.attn" for layer in range(6)]
    components = {name: Spec(tokens_per_state=1) for name in names}
    topology = ns["LayerwiseCacheTopology"](
        tuple(tuple(names[i : i + 2]) for i in (0, 2, 4)), tuple((name, i % 2) for i, name in enumerate(names))
    )
    layout = ns["build_layerwise_reuse_layout"](
        components, 6, dict(layerwise_num_shared_buffers=1, layerwise_independent_layers=[]), topology=topology
    )
    assert layout.prefetch_layer_map == {2: 0, 4: 2, 3: 1, 5: 3}


def test_mrv2_allocates_the_exact_plan_and_validates_before_allocation(ns):
    plan = ns["build_deepseek_v4_reuse_config"](configuration(), groups(), 3)
    specs = {name: g.kv_cache_spec.kv_cache_specs[name] for g in groups() for name in g.layer_names}
    allocations = []

    def allocate(size, alignment, device):
        backing = NS(data_ptr=lambda: len(allocations) * alignment)
        allocations.append((size, backing))
        return backing

    ns.update(
        get_current_vllm_config=lambda: configuration(),
        KVPPConfig=NS(from_vllm_config=lambda c: NS(size=1)),
        _is_dsv4_model=lambda c: True,
        _get_layer_kv_cache_specs=lambda c: specs,
        enable_sfa=lambda c: True,
        is_deepseek_v41_cache=lambda s: False,
        _allocate_int8_cache_tensor=allocate,
        get_kv_cache_tensor_layers=lambda t: t.layers,
    )
    definitions("vllm_ascend/worker/v2/attn_utils.py", ns, {"_allocate_kv_cache"})
    result = ns["_allocate_kv_cache"](plan, {}, "cpu")
    assert sum(size for size, _ in allocations) == sum(t.size for t in plan.kv_cache_tensors)
    assert len(allocations) == len(plan.kv_cache_tensors)
    for tensor, (_, backing) in zip(plan.kv_cache_tensors, allocations, strict=True):
        assert all(result[name] is backing for name in tensor.layers)
    allocations.clear()
    plan.kv_cache_tensors[-1].offset = 1
    with pytest.raises(ValueError, match="slot geometry"):
        ns["_allocate_kv_cache"](plan, {}, "cpu")
    assert not allocations




def test_missing_required_compressed_page_fails_closed(ns):
    worker = NS(
        group_uses_align_state=[False],
        grouped_block_size=[128],
        use_eagle=False,
        kv_cache_config=NS(dsv4_shared_cache_names=("main",)),
        _group_reuses_buffer=lambda group: True,
        _layerwise_load_snapshot_start=lambda *args: None,
    )
    request = NS(
        load_spec=NS(
            can_load=True, layerwise_previous_step=True, kvpool_cached_tokens=128, kvpool_store_skip_tokens=None
        ),
        block_ids_by_group=[[0]],
    )
    with pytest.raises(ValueError, match="required null"):
        ns["_load_entries"](worker, request, 0, [b"hash"], None)


def test_dsv4_window_continuation_restores_live_pages_only(ns):
    worker = NS(
        group_uses_align_state=[False],
        grouped_block_size=[128],
        use_eagle=False,
        kv_cache_config=NS(dsv4_shared_cache_names=("swa",), dsv4_window_group_ids=(0,)),
        _group_reuses_buffer=lambda group: True,
        _layerwise_load_snapshot_start=lambda *args: None,
        _make_layerwise_full_key=lambda group, key: key,
    )
    request = NS(
        load_spec=NS(
            can_load=True, layerwise_previous_step=True, kvpool_cached_tokens=256, kvpool_store_skip_tokens=None
        ),
        req_id="request",
        layerwise_request_id="incarnation",
        block_ids_by_group=[[0, 7]],
    )
    entries = ns["_load_entries"](worker, request, 0, [b"evicted", b"live"], [True, False])
    assert entries == [(b"live".hex(), 1)]
    worker._offload_active_pages = {("request", "incarnation", 0): (256, {1: "committed-private-page"})}
    assert ns["_load_entries"](worker, request, 0, [], None) == [("committed-private-page", 1)]
    worker._offload_active_pages[("request", "incarnation", 0)] = (128, {1: "stale"})
    with pytest.raises(ValueError, match="no committed version"):
        ns["_load_entries"](worker, request, 0, [], None)


def test_mtp_prefill_uses_same_window_for_save_and_continuation(ns):
    window_start = ns["_continuation_window_start"]
    worker = NS(kv_cache_config=NS(dsv4_nominal_windows={4: 8}))
    saved = NS(target_token_len=114688, num_prompt_tokens=118046)
    resumed = NS(target_token_len=118046, num_prompt_tokens=118046)
    assert window_start(worker, 4, 114688, 2, saved) == 57340
    assert window_start(worker, 4, 114688, 2, resumed) == 57340
    assert window_start(worker, 0, 114688, 128, resumed) is None
    resumed.target_token_len += 1
    assert window_start(worker, 4, 114688, 2, resumed) is None


def test_window_save_keeps_public_checkpoint_and_private_tail(ns):
    worker = NS(
        grouped_block_size=[128],
        group_uses_align_state=[False],
        kv_cache_config=NS(dsv4_continuation_windows={0: 129}),
        _is_layerwise_save_owner=lambda group: True,
        _layerwise_save_full_end=lambda *args: 4,
        _make_layerwise_full_key=lambda group, key: key,
    )
    request = NS(
        target_token_len=512, save_start_token=0, can_save=True,
        block_ids_by_group=[[1, 2, 3, 4]], store_masks=[[True, False, False, False]],
        layerwise_request_id="incarnation",
    )
    entries = ns["_save_entries"](worker, request, NS(), 0, [b"a", b"b", b"c", b"d"])
    assert entries == [(b"a".hex(), 0), ("offload_v1:incarnation:512:3", 3)]


def test_untrimmed_shared_window_saves_masked_live_page_for_continuation(ns):
    worker = NS(
        grouped_block_size=[128],
        group_uses_align_state=[False],
        use_eagle=True,
        kv_cache_config=NS(dsv4_window_group_ids=(0,)),
        _group_reuses_buffer=lambda group: True,
        _is_layerwise_save_owner=lambda group: True,
        _layerwise_save_full_end=lambda *args: 3,
        _layerwise_load_snapshot_start=lambda *args: None,
        _make_layerwise_full_key=lambda group, key: key,
    )
    request = NS(
        req_id="request", layerwise_request_id="incarnation",
        target_token_len=384, save_start_token=0, can_save=True,
        block_ids_by_group=[[1, 0, 3]], store_masks=[[False, False, True]],
        load_spec=NS(
            can_load=True, layerwise_previous_step=True,
            kvpool_cached_tokens=384, kvpool_store_skip_tokens=None,
        ),
    )
    hashes = [b"a", b"b", b"c"]
    saved = ns["_save_entries"](worker, request, NS(), 0, hashes)
    assert saved == [("offload_v1:incarnation:384:0", 0), (b"c".hex(), 2)]
    worker._offload_active_pages = {
        ("request", "incarnation", 0): (384, {index: key for key, index in saved})
    }
    assert ns["_load_entries"](worker, request, 0, hashes, request.store_masks[0]) == saved


def test_indexer_transport_drops_only_overlapping_view(ns):
    views = [object(), object(), object()]
    caches = {"layer.indexer.k_cache": views, "other": views}
    selected = ns["transfer_cache_views"](caches)
    assert selected["layer.indexer.k_cache"] == views[:2]
    assert selected["other"] is views
    assert len(caches["layer.indexer.k_cache"]) == 3


def isolated_method(path, cls_name, method_name, ns):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name]
    cls.bases = []
    cls.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), path, "exec", flags=0x1000000), ns)
    return ns[cls_name]


def test_offload_save_tasks_follow_opened_keys_instead_of_public_mask(ns):
    cls = isolated_method(
        STORE + "layerwise/pool_worker.py", "LayerwiseKVPoolWorker", "_process_save_for_layer_batch",
        {**ns, "get_group_block_size": lambda sizes, group: sizes[group],
         "LayerBlockRange": NS, "LayerTransferTask": NS},
    )
    worker = cls()
    worker._is_layerwise_save_owner = lambda group: True
    worker.grouped_block_size = [128]
    worker.num_layers = 1
    worker.layer_save_tasks = [[]]
    worker.use_block_key_layerwise = worker.layerwise_offload = True
    request = NS(
        can_save=True,
        save_key_block_offset=3,
        save_block_keys=["offload_v1:incarnation:768:3", None],
        save_last_block_key="offload_v1:incarnation:768:5",
        partial_block_index=5,
        store_masks=[[False] * 6],
    )
    worker._process_save_for_layer_batch([request], 0)
    [task] = worker.layer_save_tasks[0]
    [block_range] = task.block_ranges
    assert (block_range.start_block, block_range.end_block, block_range.partial_block_index) == (3, 5, 5)


@pytest.mark.parametrize(
    ("offload", "target", "is_last_chunk"),
    [(True, 1024, False), (True, 1030, True), (False, 1024, True)],
)
def test_layerwise_prefill_keeps_checkpoint_for_partial_tail(ns, offload, target, is_last_chunk):
    ns["ReqMeta"] = NS(from_request_tracker=lambda *args, **kwargs: NS(**kwargs))
    cls = isolated_method(STORE + "pool_scheduler.py", "BaseKVPoolScheduler", "_build_req_meta", ns)
    scheduler = NS(
        layerwise_offload=offload,
        _get_last_chunk_tokens_num=lambda tokens: len(tokens) // 128 * 128,
        cache_transfer_granularity=128,
        _discard_partial_chunks=True,
        original_block_size=128,
        kv_cache_group_families=(),
        hash_block_size=128,
    )
    tracker = NS(prefill_end_tokens=1030, token_len=target)
    meta = cls._build_req_meta(scheduler, tracker, [], None, [0] * 1030, False)
    assert meta.is_last_chunk is is_last_chunk


@pytest.mark.parametrize("fault", [None, "put", "commit"])
def test_publication_requires_all_layer_copies_and_failures_revoke(ns, fault):
    ns.update(
        require_aligned_batch_results=lambda operation, keys, results: results,
    )
    cls = isolated_method(
        STORE + "layerwise/kv_transfer.py", "KVCacheStoreLayerSendingThread", "_handle_range_request", ns
    )
    sender = cls()
    sender._active_put_keys = None
    sender.sync_save_events = [MagicMock(), MagicMock()]
    sender.final_layer_id = 1
    sender.raise_on_transfer_failure = True
    sender.max_transfer_blocks = 16
    sender.max_transfer_bytes = 1024
    sender._session_tracker = MagicMock()
    sender._remove_started_keys = MagicMock()
    sender._revoke_range_keys = MagicMock()
    sender._range_transfer_batches = lambda keys, buffers, sizes, offsets, *args: [(keys, buffers, sizes, offsets)]
    sender.m_store = NS(
        batch_copy_put=MagicMock(return_value=[-1 if fault == "put" else 0]),
        batch_commit=MagicMock(return_value=[-1 if fault == "commit" else 0]),
    )
    meta = NS(layer_id=0, keys=["object"], all_buffers=[[1234]], all_sizes=[[8]], all_offsets=[[0]])
    if fault == "put":
        with pytest.raises(RuntimeError, match="PUT failed"):
            sender._handle_range_request(meta, False)
        sender.m_store.batch_commit.assert_not_called()
        sender._revoke_range_keys.assert_called_once_with(["object"])
        return
    sender._handle_range_request(meta, False)
    sender.m_store.batch_commit.assert_not_called()
    sender._session_tracker.commit_put_keys.assert_not_called()
    meta.layer_id = 1
    if fault == "commit":
        with pytest.raises(RuntimeError, match="commit failed"):
            sender._handle_range_request(meta, True)
        sender._revoke_range_keys.assert_called_once_with(["object"])
        sender._session_tracker.commit_put_keys.assert_not_called()
    else:
        sender._handle_range_request(meta, True)
        sender._session_tracker.commit_put_keys.assert_called_once_with(["object"])


@pytest.mark.parametrize("save_failed", [False, True])
def test_actual_receive_handler_waits_for_source_before_overwrite(ns, save_failed):
    import threading

    events = []
    definitions(STORE + "layerwise/layerwise_cache_layout.py", ns, {"reuse_sources"})
    cls = isolated_method(
        STORE + "layerwise/kv_transfer.py", "KVCacheStoreLayerRecvingThread", "_handle_range_layer_task", ns
    )
    receiver = cls()
    receiver.layer_save_finished_events = [threading.Event(), threading.Event()]
    receiver.layer_load_finished_events = [threading.Event(), threading.Event()]
    receiver.sync_save_events = [MagicMock(), MagicMock()]
    receiver._active_load_indices = None
    receiver._group_load_indices = {}
    receiver._load_abort_event = threading.Event()
    receiver._mark_invalid_transfer_task_blocks = MagicMock()
    receiver.request_queue = MagicMock()
    receiver.get_event = threading.Event()
    receiver.external_slot_release_waiter = None
    started_wait = threading.Event()

    def wait_for_event(event, *args):
        started_wait.set()
        return event.wait(timeout=2)

    receiver._wait_for_event = wait_for_event
    receiver.sync_save_events[0].synchronize.side_effect = lambda: events.append("slot_release")

    def check_failure():
        if save_failed:
            raise RuntimeError("source PUT failed")

    receiver.save_failure_checker = check_failure
    shared = NS()
    meta = ns["LayerRangeReqMeta"]([], 1, [], [], [], [], [])
    receiver.group_builders = [NS(build_addrs=lambda *args: meta)]
    receiver._handle_range_request = lambda *args: events.append("actual_write")
    task = NS(shared_block_data=shared, group_id=0, layer_idx_in_group=0, final_group_layer=True)
    data = NS(layer_id=1, wait_for_save_layer=0, attention_start_gate=None, transfer_tasks=[task])
    failures = []

    def run():
        try:
            receiver._handle_range_layer_task(data)
        except RuntimeError as error:
            failures.append(str(error))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started_wait.wait(timeout=2)
        assert not receiver.layer_load_finished_events[1].is_set()
        assert "actual_write" not in events
        receiver.layer_save_finished_events[0].set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        if save_failed:
            assert failures == ["source PUT failed"]
            assert "actual_write" not in events
            assert receiver._load_abort_event.is_set()
        else:
            assert not failures
            assert events.index("slot_release") < events.index("actual_write")
    finally:
        receiver.layer_save_finished_events[0].set()
        thread.join(timeout=2)




@pytest.mark.parametrize("cacheable", [True, False])
def test_dsv4_push_final_chunk_uses_live_component_groups(ns, cacheable):
    path = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_worker.py"
    ns.update(
        MambaSpec=type("MambaSpec", (), {}),
        is_circular_kv_cache_spec=lambda _: False,
        is_prefix_cacheable=lambda _: cacheable,
        LayerTransferRequest=lambda *args: args,
    )
    worker = isolated_method(path, "MooncakePushConnectorWorker", "_build_layer_transfer_request", ns)()
    worker._is_dsv4 = True
    worker._layer_indices = {"component": 0}
    worker.layer_name_to_spec_index = {"component": 0}
    worker.kv_cache_specs = [NS(sliding_window=8)]
    worker.transfer_metadata = NS(group_indices=[1], block_size_scales=[[1]], layer_block_sizes=[4])
    peer = NS(layer_names=["component"], group_indices=[0], block_size_scales=[[1]], layer_block_sizes=[4])
    destination = NS(
        transfer_params=dict(
            num_computed_tokens=8, num_external_tokens=3, consumer_rank=(0, 0, 0), block_ids=((0, 7, 8),)
        ),
        sent_tokens={},
        metadata_future=NS(result=lambda: NS(metadata_by_pp_rank=[peer])),
    )
    state = NS(
        metadata=NS(num_prompt_tokens=11, local_block_ids=((99,), (0, 3, 4))),
        available_layers={"component": (8, "event")},
    )
    assert worker._build_layer_transfer_request("r", state, destination, "component") is None
    state.available_layers["component"] = 11, "event"
    transfer, event = worker._build_layer_transfer_request("r", state, destination, "component")
    expected = ("r", (4,), (8,), 8, 11) if cacheable else ("r", (3, 4), (7, 8), 3, 11)
    assert transfer == expected
    assert event == "event"


def test_shared_store_accepts_author_push_sibling(ns):
    ns["get_backend_class"] = lambda _: NS(layerwise_data_plane="block_key")
    definitions(STORE + "layerwise/layerwise_cache_layout.py", ns, {"get_layerwise_reuse_config"})
    store = dict(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config=dict(backend="mooncake", use_layerwise=True, layerwise_num_shared_buffers=2),
    )
    peer = dict(kv_connector="MooncakePushConnector")
    config = NS(
        kv_connector="MultiConnector", kv_role="kv_producer", kv_connector_extra_config={"connectors": [store, peer]}
    )
    assert ns["get_layerwise_reuse_config"](config) is store["kv_connector_extra_config"]
    peer["kv_connector"] = "MooncakeLayerwiseConnector"
    with pytest.raises(ValueError):
        ns["get_layerwise_reuse_config"](config)


def test_cold_slot_wait_has_evidence_without_get(ns):
    import threading

    definitions(STORE + "layerwise/layerwise_cache_layout.py", ns, {"reuse_sources"})
    cls = isolated_method(STORE + "layerwise/kv_transfer.py", "KVCacheStoreLayerRecvingThread", "_copy_gva_layers", ns)
    receiver = cls()
    receiver.layer_save_finished_events = [threading.Event()]
    receiver.layer_save_finished_events[0].set()
    receiver.layer_load_finished_events = [threading.Event(), threading.Event()]
    receiver.sync_save_events = [MagicMock()]
    receiver.save_failure_checker = MagicMock()
    receiver._wait_for_event = lambda event, *args: event.wait(timeout=1)
    receiver.load_disabled = threading.Event()
    receiver.external_slot_release_waiter = MagicMock()
    task = NS(wait_for_save_layer=0, transfer_tasks=[], layer_id=1, attention_start_gate=None)
    assert receiver._copy_gva_layers(task)
    receiver.sync_save_events[0].synchronize.assert_called_once_with()
    receiver.external_slot_release_waiter.assert_called_once_with(1)


@pytest.mark.parametrize("endpoint", [4095, 4096, 4097, 8193, 122798])
@pytest.mark.parametrize("size", [128, 4096])
@pytest.mark.parametrize("hash_lag", [0, 2])
def test_transfer_suffix_snapshots_match_next_step_keys(ns, endpoint, size, hash_lag):
    total = (endpoint + size - 1) // size
    hashes = [str(i).encode() for i in range(max(0, endpoint // size - hash_lag))]
    transfer = 4096
    cutoff = min(endpoint // transfer, len(hashes) * size // transfer) * transfer // size
    needs_snapshot = bool(endpoint % transfer or endpoint // transfer > len(hashes) * size // transfer)
    worker = NS(
        grouped_block_size=[size], group_uses_align_state=[False],
        cache_transfer_granularity=transfer, use_eagle=False,
        kv_cache_config=NS(dsv4_shared_cache_names=("main",)),
        _group_reuses_buffer=lambda group: True,
        _is_layerwise_save_owner=lambda group: True,
        _make_layerwise_full_key=lambda group, key: key,
    )
    worker._layerwise_save_full_end = lambda req, group, span, partial: (
        min(req.target_token_len // span, partial)
        if partial is not None else req.target_token_len // span
    )
    worker._layerwise_load_snapshot_start = lambda req, group, tokens: cutoff if needs_snapshot else None
    request = NS(
        target_token_len=endpoint, save_start_token=0, can_save=True,
        block_ids_by_group=[list(range(1, total + 1))], store_masks=None,
        req_id="request",
        layerwise_request_id="incarnation",
        load_spec=NS(
            can_load=True, layerwise_previous_step=True, kvpool_cached_tokens=endpoint,
            kvpool_store_skip_tokens=None, layerwise_snapshot_end_token=endpoint,
        ),
    )
    saved = ns["_save_entries"](worker, request, NS(), 0, hashes)
    saved_keys = {key for key, _ in saved}
    # Hashes arriving on the next scheduler step must not change which
    # namespace the preceding save used.
    for restore_hashes in (hashes, hashes + [b"new"] * 8):
        loaded = ns["_load_entries"](worker, request, 0, restore_hashes, None)
        assert {key for key, _ in loaded} <= saved_keys
    if endpoint == 122798 and size == 128 and not hash_lag:
        assert sum("offload_v1:" in key and slot is not None for key, slot in saved) == 31
        assert sum("offload_v1:" in key for key, _ in saved) == 32
