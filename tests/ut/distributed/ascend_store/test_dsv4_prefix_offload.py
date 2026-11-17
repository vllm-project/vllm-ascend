# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-request prefix contracts using real groups, keys, masks and ranges.

Only the Store transport is replaced. Payload tags identify every logical
member/page; these tests do not execute attention kernels or Memcache DMA.
"""

import threading
from math import ceil, lcm
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

from tests.ut.distributed.ascend_store.test_dsv4_layerwise_runtime import make_worker, reset_step
from tests.ut.models.test_dsv4_layerwise import make_layout, make_worker_config
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    ReqMeta,
    get_block_hashes,
    infer_group_block_sizes,
    infer_group_cache_families,
)


class PrefixPool:
    def __init__(self, model="flash", *, buffers=3, cache_format="a3", retention=None):
        register_all_kvcache_specs(None)
        register_ascend_kv_cache_specs()
        runtime, self.groups, _ = make_layout(model, buffers=buffers, independent=[0, 1], cache_format=cache_format)
        config = make_worker_config(runtime, self.groups, num_blocks=32)
        self.sizes = infer_group_block_sizes(32, self.groups)
        self.alignment, self.hash_size = lcm(*self.sizes), min(self.sizes)
        self.coordinator = AscendStoreCoordinator(
            self.groups,
            self.alignment,
            self.hash_size,
            self.sizes,
            infer_group_cache_families(self.groups, None),
            retention_interval=retention,
        )
        self.hashes = [i.to_bytes(8, "little") for i in range(16384 // self.hash_size)]
        self.keys = MemcacheBackend.bind_layerwise_keys(
            vllm_config=runtime,
            kv_cache_config=config,
            model_name=model,
            use_hybrid=True,
            grouped_block_size=self.sizes,
        )
        worker = self.worker = make_worker()
        worker.num_kv_cache_groups = len(self.groups)
        worker.grouped_block_size, worker.hash_block_size = self.sizes, self.hash_size
        worker.cache_coordinator = self.coordinator
        worker.cacheable_group_ids = set(range(len(self.groups)))
        worker.group_uses_align_state = [False] * len(self.groups)
        # Layer 1 is a representative transfer task for each group's members.
        # Per-model physical aliasing is tested separately by the allocator tests.
        worker.layerwise_offload = get_layerwise_reuse_layout(config) is not None
        worker._reused_layers_by_group = {g: {1} for g in range(len(self.groups))} if worker.layerwise_offload else {}
        worker.token_database = SimpleNamespace(
            store_mask=self.coordinator.store_mask, load_mask=self.coordinator.load_mask
        )
        worker._make_layerwise_full_key = self.full_key
        worker._make_layerwise_partial_key = lambda req, g, b, e: MemcacheBackend.make_partial_key(
            model, req.req_id, g, b, e, 0
        )
        self.payloads = {}
        worker._failed_requests = set()
        worker._layer_load_aborted = threading.Event()
        worker.kv_recv_thread = None

        scheduler = self.scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
        scheduler.cache_coordinator = self.coordinator
        scheduler.grouped_block_size = scheduler.original_block_size = self.sizes
        scheduler.cacheable_group_ids = set(range(len(self.groups)))
        scheduler.num_speculative_blocks_by_group = {}
        scheduler.layerwise_offload = worker.layerwise_offload
        scheduler.layerwise_keys = self.keys
        scheduler.layerwise_max_transfer_blocks = 0
        scheduler.dspark_prefix_keys = None
        scheduler.sharded_mamba_group_ids = set()
        scheduler.tp_size = scheduler.put_step = 1
        scheduler._request_trackers = {}
        scheduler.store_scheduler = SimpleNamespace(
            batch_is_readable=lambda keys: [key in worker.m_store.committed for key in keys]
        )
        scheduler.load_specs = {}
        scheduler._prefix_probe = None
        scheduler.kv_role, scheduler.consumer_is_to_load = "kv_producer", False
        scheduler.use_layerwise = scheduler.use_layerwise_transfer = True
        scheduler.use_block_key_layerwise = scheduler.use_eagle = scheduler.load_async = False
        scheduler.cache_transfer_granularity = scheduler.lcm_block_size = self.alignment

    def full_key(self, group, block_hash):
        return self.keys.make_full_key(group, block_hash, 0, 0, 0)

    def page_key(self, group, index):
        hashes = get_block_hashes(self.hashes, self.sizes[group], self.hash_size)
        return self.full_key(group, hashes[index].hex())

    def tables(self, start, end, *, request_offset=0):
        result = []
        for group, (size, spec) in enumerate(zip(self.sizes, self.coordinator.group_effective_specs)):
            window = getattr(spec, "sliding_window", None)
            skipped = 0 if window is None else max(0, start - window + 1) // size
            result.append(
                [0 if i < skipped else 1 + request_offset + group * 20000 + i for i in range(ceil(end / size))]
            )
        return result

    def meta(self, start, end, *, private=True, offset=0, req_id="a", load_spec=None):
        tables = self.tables(start, end, request_offset=offset)
        return ReqMeta(
            req_id,
            target_token_len=end,
            save_end_token=end // self.alignment * self.alignment,
            save_start_token=0,
            block_ids_by_group=tables,
            block_ids_by_group_np=[np.asarray(table, dtype=np.int64) for table in tables],
            block_hashes=self.hashes,
            num_prompt_tokens=8193,
            needs_private_snapshot=private,
            can_save=True,
            load_spec=load_spec,
        )

    def transfer_pairs(self, group, *, save):
        builder = LayerBatchBuilder.__new__(LayerBatchBuilder)
        builder.group_id = group
        builder._block_ids_buf = builder._block_gvas_buf = None
        tasks = self.worker.layer_save_tasks[1] if save else self.worker.layer_load_tasks[1]
        result = []
        for task in tasks:
            if task.group_id == group:
                data = builder.build_shared(task, is_save=save)
                # One HBM page can have both a private continuation object
                # and a public checkpoint. Preserve both copy destinations.
                result.extend(zip(data.block_ids_arr.tolist(), data.block_gvas_arr.tolist()))
        return result

    def transfers(self, group, *, save):
        return dict(self.transfer_pairs(group, save=save))

    def save(self, start, end, *, private=True, publish=True):
        worker = self.worker
        reset_step(worker)
        request = self.meta(start, end, private=private)
        request.store_masks = worker._compute_reachable_store_masks(request)
        for group in range(len(self.groups)):
            worker._process_save_for_layer_batch([request], 1, group)
        assert worker._alloc_gvas_for_save_for_request([request]) is not False
        transferred_gvas = set()
        for group, table in enumerate(request.block_ids_by_group):
            logical_indices = {block_id: index for index, block_id in enumerate(table) if block_id}
            for block_id, gva in self.transfer_pairs(group, save=True):
                transferred_gvas.add(gva)
                for name in self.groups[group].layer_names:
                    self.payloads[gva, name] = f"{name}:{logical_indices[block_id]}".encode()
        assert transferred_gvas == {worker.m_store.addresses[key] for key in request.save_keys}
        if publish:
            worker.m_store.committed.update(request.save_keys)
        return request

    def request(self, tokens):
        return SimpleNamespace(
            request_id="b", num_tokens=tokens, prompt_token_ids=range(tokens), block_hashes=self.hashes
        )

    def hit(self, tokens, *, local_tokens=0):
        return self.scheduler._get_layerwise_hit_tokens(self.request(tokens), tokens, local_tokens)

    def restore(self, tokens, *, replay=False, prompt_tokens=None):
        reset_step(self.worker)
        prompt = (tokens if replay else tokens + 1) if prompt_tokens is None else prompt_tokens
        count, _ = self.scheduler.get_num_new_matched_tokens(self.request(prompt), 0)
        assert count == tokens - int(replay)
        request = self.meta(count, prompt, req_id="b", offset=300000, load_spec=self.scheduler.load_specs["b"])
        assert self.worker._prepare_load_gvas_for_request([request]) is not False
        counts = []
        for group, (size, spec) in enumerate(zip(self.sizes, self.coordinator.group_effective_specs)):
            self.worker._process_load_for_layer_batch([request], 1, group)
            copied = self.transfers(group, save=False)
            window = getattr(spec, "sliding_window", None)
            first = 0 if window is None else max(0, tokens - window + 1) // size
            expected = {request.block_ids_by_group[group][i]: i for i in range(first, tokens // size)}
            assert set(copied) == set(expected)
            assert 0 not in copied
            for block_id, index in expected.items():
                for name in self.groups[group].layer_names:
                    assert self.payloads[copied[block_id], name] == f"{name}:{index}".encode()
            counts.append(len(copied))
        assert not self.worker._invalid_block_ids
        return counts


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("buffers", [2, 3, 4, 100])
@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
def test_cross_request_prefix_restores_every_member_to_new_block_ids(model, buffers, cache_format):
    pool = PrefixPool(model, buffers=buffers, cache_format=cache_format)
    pool.save(0, 4096)
    pool.save(4096, 8192)
    assert pool.hit(12288) == 8192
    counts = pool.restore(8192)
    if cache_format == "a3":
        assert counts == ([64, 2, 4, 4, 4, 16] if model == "flash" else [64, 2, 16, 4, 4, 4])


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("replay", [False, True])
def test_full_prefix_replay_has_the_compressor_and_swa_history(model, replay):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    expected = [32, 1, 4, 4, 4, 16] if model == "flash" else [32, 1, 16, 4, 4, 4]
    if replay:
        assert pool.scheduler.get_num_new_matched_tokens(pool.request(4096), 0) == (0, False)
        assert not pool.scheduler.load_specs
    else:
        assert pool.restore(4096) == expected


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("group", range(6))
def test_missing_boundary_group_falls_back_to_earlier_complete_prefix(model, group):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    pool.save(4096, 8192)
    pool.worker.m_store.committed.remove(pool.page_key(group, 8192 // pool.sizes[group] - 1))
    assert pool.hit(12288) == 4096


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_eviction_between_lookup_and_load_reports_failure(model):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    assert pool.hit(4097) == 4096
    state_group = next(
        g for g, group in enumerate(pool.groups) if any(name.endswith(".state_cache") for name in group.layer_names)
    )
    pool.worker.m_store.committed.remove(pool.page_key(state_group, 4096 // pool.sizes[state_group] - 1))
    meta = pool.meta(4096, 4097, req_id="b", offset=300000, load_spec=LoadSpec(0, 4096, True))
    pool.worker._prepare_load_gvas([meta])
    assert not meta.load_spec.can_load
    # Hybrid groups report the request, not a possibly ambiguous group-local ID.
    assert pool.worker._failed_requests == {"b"}
    with pytest.raises(RuntimeError, match="refusing incomplete KV/state"):
        pool.worker._check_hybrid_load_errors()


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_private_or_unpublished_data_does_not_admit_a_public_prefix(model):
    pool = PrefixPool(model)
    pending = pool.save(0, 4096, publish=False)
    assert pool.payloads  # Payload existence is insufficient without publication.
    assert pool.hit(8192, local_tokens=4096) == 0
    pool.worker.m_store.committed.update(pending.save_keys)
    assert pool.hit(8192) == 4096
    pool.worker.m_store.committed.clear()
    pool.save(4096, 4097)
    pool.worker.m_store.committed = {key for key in pool.worker.m_store.committed if "@partial@" in key}
    assert any("@partial@" in key for key in pool.worker.m_store.committed)
    assert pool.hit(8192) == 0


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_unaligned_prefill_chunks_preserve_a_crossed_prefix_boundary(model):
    pool = PrefixPool(model)
    for start, end in [(0, 1024), (1024, 3000), (3000, 4200), (4200, 8192)]:
        pool.save(start, end)
    assert pool.hit(4097) == 4096
    assert pool.hit(8193) == 8192
    pool.restore(4096)
    pool.restore(8192)


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("retention", [None, 0])
def test_terminal_save_retains_checkpoint_windows_without_dense_history(model, retention):
    pool = PrefixPool(model, retention=retention)
    # No private continuation: retain only the coordinator's reachable windows.
    # The final prompt token is at 8192; its replay boundary is 8192 tokens.
    pool.save(0, 8193, private=False)
    assert pool.hit(8193) == 8192
    assert pool.hit(4097) == (4096 if retention is None else 0)
    pool.restore(8192)


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_all_rank_publication_is_required_for_state_prefix(model):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    pool.scheduler.tp_size = 2
    assert pool.hit(8192) == 0
    rank_zero = list(pool.worker.m_store.committed)
    pool.worker.m_store.committed.update(key.rsplit("@", 1)[0] + "@1" for key in rank_zero)
    assert pool.hit(8192) == 4096
    state_group = next(
        g for g, group in enumerate(pool.groups) if any(name.endswith(".state_cache") for name in group.layer_names)
    )
    key = pool.page_key(state_group, 4096 // pool.sizes[state_group] - 1)
    pool.worker.m_store.committed.remove(key.rsplit("@", 1)[0] + "@1")
    assert pool.hit(8192) == 0


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_new_request_with_divergent_suffix_hits_only_shared_prefix(model):
    pool = PrefixPool(model)
    pool.save(0, 8192)
    request = pool.request(8193)
    # Chained hashes after the shared junction must differ for the new request.
    boundary = 4096 // pool.hash_size
    request.block_hashes = pool.hashes[:boundary] + [
        (i + 100000).to_bytes(8, "little") for i in range(boundary, len(pool.hashes))
    ]
    assert pool.scheduler._get_layerwise_hit_tokens(request, 8193, 0) == 4096


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_private_continuation_obeys_public_terminal_retention(model):
    pool = PrefixPool(model, retention=0)
    pool.save(0, 8192, private=True)
    # Private continuation does not publish extra intermediate checkpoints
    # when the user selected terminal-only public retention.
    assert pool.hit(4097) == 0
    assert pool.hit(8193) == 8192


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("missing_group", [None, *range(6)])
def test_local_prefix_limit_requires_every_boundary_group(model, missing_group):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    if missing_group is not None:
        pool.worker.m_store.committed.remove(pool.page_key(missing_group, 4096 // pool.sizes[missing_group] - 1))
    assert pool.scheduler.get_local_prefix_cache_limit(pool.request(4097)) == (4096 if missing_group is None else 0)


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_local_hit_still_restores_rotated_members_with_zero_external_increment(model):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    request = pool.request(4097)
    assert pool.scheduler.get_local_prefix_cache_limit(request) == 4096
    count, asynchronous = pool.scheduler.get_num_new_matched_tokens(request, 4096)
    assert (count, asynchronous) == (0, False)
    spec = pool.scheduler.load_specs["b"]
    assert spec.can_load and spec.vllm_cached_tokens == spec.kvpool_cached_tokens == 4096
    reset_step(pool.worker)
    # Adopt the existing IDs rather than requesting a new destination range.
    meta = pool.meta(4096, 4097, req_id="b", load_spec=spec)
    assert pool.worker._prepare_load_gvas_for_request([meta]) is not False
    for group in range(len(pool.groups)):
        pool.worker._process_load_for_layer_batch([meta], 1, group)
        copies = pool.transfers(group, save=False)
        assert copies and 0 not in copies


@pytest.mark.parametrize(
    "caller_limit,pool_limit,expected",
    [(None, 4096, 4096), (8192, 4096, 4096), (2048, 4096, 2048), (None, 0, 0)],
)
def test_scheduler_applies_store_limit_to_existing_hash_lookup(monkeypatch, caller_limit, pool_limit, expected):
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator as module

    request = object()
    original = Mock(return_value=("blocks", expected, 0, False))
    scheduler_class = type("Scheduler", (), {"_get_local_prefix_cache_hit": original})
    module._install_layerwise_prefix_limit(scheduler_class)
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(get_local_prefix_cache_limit=Mock(return_value=pool_limit)),
        kv_cache_manager=SimpleNamespace(prefix_cache_lookup_enabled=lambda request: True),
    )
    assert scheduler_class._get_local_prefix_cache_hit(scheduler, request, caller_limit) == (
        "blocks",
        expected,
        0,
        False,
    )
    original.assert_called_once_with(scheduler, request, expected)


@pytest.mark.parametrize("lookup_enabled,has_limit", [(False, True), (True, False)])
def test_non_rotating_or_disabled_prefix_lookup_preserves_native_limit(lookup_enabled, has_limit):
    from vllm_ascend.patch.platform import patch_kv_cache_coordinator as module

    original = Mock(return_value=("blocks", 128, 0, False))
    scheduler_class = type("Scheduler", (), {"_get_local_prefix_cache_hit": original})
    module._install_layerwise_prefix_limit(scheduler_class)
    get_limit = Mock(return_value=None)
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(get_local_prefix_cache_limit=get_limit) if has_limit else None,
        kv_cache_manager=SimpleNamespace(prefix_cache_lookup_enabled=lambda request: lookup_enabled),
    )
    request = object()
    assert scheduler_class._get_local_prefix_cache_hit(scheduler, request, 128) == ("blocks", 128, 0, False)
    original.assert_called_once_with(scheduler, request, 128)
    get_limit.assert_not_called()


@pytest.mark.parametrize("limits,expected", [([None, 4096, 2048], 2048), ([None, None], None), ([4096, 0], 0)])
def test_multi_connector_forwards_strictest_prefix_limit(limits, expected):
    from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector

    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [SimpleNamespace()]
    connector._connectors.extend(
        SimpleNamespace(get_local_prefix_cache_limit=Mock(return_value=limit)) for limit in limits
    )
    assert connector.get_local_prefix_cache_limit(object()) == expected


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_full_hit_does_not_advertise_an_unrestorable_external_tail(model):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    # 4096 cannot be adopted locally in full: generation recomputes a tail.
    # A state checkpoint at 4096 does not prove one at the previous boundary.
    request = pool.request(4096)
    assert pool.scheduler.get_local_prefix_cache_limit(request) == 0
    assert pool.scheduler._lookup_external_hit_tokens(request, 0) == 0
    assert pool.scheduler._prefix_probe is None


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("buffers", [None, 3])
def test_old_window_prefix_survives_request_advancement(model, buffers):
    """A's released device windows do not delete already published pool data."""
    pool = PrefixPool(model, buffers=buffers)
    pool.save(0, 4096, private=buffers is not None)
    before = set(pool.worker.m_store.committed)
    for start in range(4096, 8192, 512):
        pool.save(start, start + 512, private=buffers is not None)
    latest_tables = pool.tables(7680, 8192)
    for group, spec in enumerate(pool.coordinator.group_effective_specs):
        if getattr(spec, "sliding_window", None) is not None:
            # The original boundary is genuinely absent from A's HBM table.
            assert latest_tables[group][4096 // pool.sizes[group] - 1] == 0
    assert before <= pool.worker.m_store.committed
    assert pool.hit(4097) == 4096
    pool.restore(4096)
