# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(slots=True)
class _FakeKVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: object | None = None
    _block_hash_num_tokens: int | None = None
    prev_free_block: object | None = None
    next_free_block: object | None = None
    is_null: bool = False

    @property
    def block_hash(self):
        return self._block_hash

    @property
    def block_hash_num_tokens(self):
        return self._block_hash_num_tokens

    def set_block_hash(self, block_hash, num_tokens=None):
        assert self._block_hash is None
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self):
        self._block_hash = None
        self._block_hash_num_tokens = None


class _FakeFreeQueue:
    def __init__(self, blocks):
        self._blocks = list(blocks)

    @property
    def num_free_blocks(self):
        return len(self._blocks)

    def popleft(self):
        return self._blocks.pop(0)

    def remove(self, block):
        self._blocks.remove(block)

    def append(self, block):
        assert block not in self._blocks
        self._blocks.append(block)

    def prepend(self, block):
        assert block not in self._blocks
        self._blocks.insert(0, block)


class _FakeBlockPool:
    def __init__(
        self,
        num_gpu_blocks,
        enable_caching,
        hash_block_size,
        enable_kv_cache_events=False,
        metrics_collector=None,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.enable_kv_cache_events = enable_kv_cache_events
        self.blocks = [_FakeKVCacheBlock(index) for index in range(num_gpu_blocks)]
        self.free_block_queue = _FakeFreeQueue(self.blocks)
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self._cached = {}

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        result = []
        for group_id in kv_cache_group_ids:
            blocks = self._cached.get((block_hash, group_id))
            if not blocks:
                return None
            result.append(blocks[0])
        return result

    def cache_full_blocks(
        self,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask=None,
    ):
        new_blocks = blocks[num_cached_blocks:num_full_blocks]
        for offset, block in enumerate(new_blocks):
            if block.is_null or (block_mask is not None and not block_mask[offset]):
                continue
            block_index = num_cached_blocks + offset
            hash_index = ((block_index + 1) * block_size // self.hash_block_size) - 1
            raw_hash = request.block_hashes[hash_index]
            namespaced_hash = (raw_hash, kv_cache_group_id)
            block.set_block_hash(namespaced_hash, num_tokens=(block_index + 1) * block_size)
            self._cached.setdefault(namespaced_hash, []).append(block)

    def _maybe_evict_cached_block(self, block):
        removed = False
        for key, blocks in tuple(self._cached.items()):
            if block not in blocks:
                continue
            blocks.remove(block)
            removed = True
            if not blocks:
                del self._cached[key]
        block.reset_hash()
        return removed

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks):
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None:
                    self.free_block_queue.prepend(block)
                else:
                    self.free_block_queue.append(block)


@pytest.fixture
def runtime_modules(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm_v1 = types.ModuleType("vllm.v1")
    vllm_v1.__path__ = []
    vllm_core = types.ModuleType("vllm.v1.core")
    vllm_core.__path__ = []
    block_pool = types.ModuleType("vllm.v1.core.block_pool")
    block_pool.BlockPool = _FakeBlockPool
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.KVCacheBlock = _FakeKVCacheBlock
    for name, module in (
        ("vllm", vllm),
        ("vllm.v1", vllm_v1),
        ("vllm.v1.core", vllm_core),
        ("vllm.v1.core.block_pool", block_pool),
        ("vllm.v1.core.kv_cache_utils", kv_cache_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    core_path = Path(__file__).parents[3] / "vllm_ascend" / "core"
    package_name = "_jenga_prefix_runtime_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(core_path)]
    monkeypatch.setitem(sys.modules, package_name, package)

    loaded = {}
    for module_name in ("typed_kv_cache", "jenga_prefix_cache", "jenga_prefix_runtime"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, core_path / f"{module_name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified_name, module)
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


def _plan(runtime_modules, num_superpages=4):
    typed = runtime_modules["typed_kv_cache"]
    specs = (
        typed.TypedPageSpec(group_id=0, page_size_bytes=12, block_size_tokens=4),
        typed.TypedPageSpec(group_id=1, page_size_bytes=8, block_size_tokens=2),
    )
    return typed.TypedKVCachePlan.exact_lcm(specs, total_memory_bytes=24 * num_superpages)


@dataclass(slots=True)
class _Request:
    request_id: str
    block_hashes: list[str]


def _allocate_cache(
    runtime,
    request_id,
    counts,
    *,
    allocation_timestamp,
    release_timestamp=None,
):
    request = _Request(request_id, [f"{request_id}-p{tokens}" for tokens in range(2, 34, 2)])
    with runtime.request_context(request_id, timestamp=allocation_timestamp):
        runtime.prepare_allocation(request_id, counts)
        allocated = {group_id: runtime.for_group(group_id).get_new_blocks(count) for group_id, count in counts.items()}
        runtime.finish_allocation(request_id)
        for group_id, blocks in allocated.items():
            runtime.for_group(group_id).cache_full_blocks(
                request,
                blocks,
                0,
                len(blocks),
                runtime.plan.spec(group_id).block_size_tokens,
                group_id,
            )
    if release_timestamp is not None:
        with runtime.request_context(request_id, timestamp=release_timestamp):
            for group_id, blocks in allocated.items():
                runtime.for_group(group_id).free_blocks(reversed(blocks))
    return request, allocated


def test_atomic_stage_reserves_null_superpage_and_namespaces_hashes(runtime_modules):
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules), hash_block_size=2)

    # IDs [0, capacity) alias physical superpage zero. Only ID zero is the
    # canonical null block; every other alias has been removed from free lists.
    assert runtime.for_group(0).get_num_free_blocks() == 6
    assert runtime.for_group(1).get_num_free_blocks() == 9
    assert runtime._metadata_pools[0].blocks[1].ref_cnt == 1
    assert runtime._metadata_pools[1].blocks[1].ref_cnt == 1
    assert runtime._metadata_pools[1].blocks[2].ref_cnt == 1

    _, blocks = _allocate_cache(
        runtime,
        "shared",
        {0: 2, 1: 3},
        allocation_timestamp=10,
        release_timestamp=20,
    )

    assert [block.block_id for block in blocks[0]] == [2, 3]
    assert [block.block_id for block in blocks[1]] == [6, 7, 8]
    # Token prefix four maps to group-0 block zero and group-1 block one.
    hit = runtime.get_cached_block("shared-p4", [0, 1])
    assert hit == [blocks[0][0], blocks[1][1]]
    assert hit[0].block_hash == ("shared-p4", 0)
    assert hit[1].block_hash == ("shared-p4", 1)
    assert all(block.ref_cnt == 0 for group in blocks.values() for block in group)

    timestamps = {
        page.last_access
        for large_page in runtime.policy.snapshot().large_pages
        for page in large_page.pages
        if page.block_hash is not None
    }
    assert timestamps == {20}


def test_protected_hit_survives_atomic_large_page_rebind(runtime_modules):
    cache = runtime_modules["jenga_prefix_cache"]
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=3), hash_block_size=2)
    group1_request, group1 = _allocate_cache(
        runtime,
        "old-g1",
        {1: 3},
        allocation_timestamp=1,
        release_timestamp=10,
    )
    group0_request, group0 = _allocate_cache(
        runtime,
        "newer-g0",
        {0: 2},
        allocation_timestamp=2,
        release_timestamp=20,
    )

    protected_hit = runtime.get_cached_block(group1_request.block_hashes[0], [1])[0]
    with runtime.request_context("incoming", timestamp=30):
        allocation = runtime.prepare_allocation(
            "incoming",
            {0: 1},
            protected_blocks={1: [protected_hit]},
        )
        # The older group-1 large page would win LRU without protection. The
        # temporary admission protection forces the fully-evictable group-0
        # page to be reclaimed instead.
        assert allocation.decisions[0].tier is cache.AllocationTier.EVICTABLE_LARGE_PAGE
        assert {page.group_id for page in allocation.decisions[0].evicted_pages} == {0}
        runtime.for_group(1).touch([protected_hit])
        replacement = runtime.for_group(0).get_new_blocks(1)[0]
        runtime.finish_allocation("incoming")

    assert replacement.block_id == group0[0][0].block_id
    assert runtime.get_cached_block(group0_request.block_hashes[1], [0]) is None
    assert runtime.get_cached_block(group1_request.block_hashes[0], [1]) == [protected_hit]
    assert protected_hit.ref_cnt == 1


def test_small_page_eviction_removes_upstream_hash_before_reuse(runtime_modules):
    cache = runtime_modules["jenga_prefix_cache"]
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=2), hash_block_size=2)
    old_request, old_blocks = _allocate_cache(
        runtime,
        "old",
        {0: 2},
        allocation_timestamp=1,
        release_timestamp=2,
    )

    first_hit = runtime.get_cached_block(old_request.block_hashes[1], [0])[0]
    with runtime.request_context("holder", timestamp=3):
        runtime.for_group(0).touch([first_hit])

    with runtime.request_context("replacement", timestamp=4):
        allocation = runtime.prepare_allocation("replacement", {0: 1})
        decision = allocation.decisions[0]
        assert decision.tier is cache.AllocationTier.EVICTABLE_SMALL_PAGE
        assert decision.evicted_pages[0].block_id == old_blocks[0][1].block_id
        replacement = runtime.for_group(0).get_new_blocks(1)[0]
        runtime.finish_allocation("replacement")

    assert replacement is old_blocks[0][1]
    assert replacement.block_hash is None
    assert runtime.get_cached_block(old_request.block_hashes[3], [0]) is None
    assert runtime.get_cached_block(old_request.block_hashes[1], [0]) == [first_hit]


def test_deferred_free_keeps_identity_handoff_and_blocks_interleaving(runtime_modules):
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=3), hash_block_size=2)
    request, allocated = _allocate_cache(
        runtime,
        "deferred",
        {0: 1},
        allocation_timestamp=1,
    )
    block = allocated[0][0]

    runtime.release_for_deferred_free("deferred", [block], timestamp=5)
    assert block.ref_cnt == 1
    deferred_typed_id = runtime.plan.from_group_local_id(0, block.block_id)
    assert runtime.policy.get_request_handle("deferred", 0, block.block_id) is not None

    # The deferred page stays USED, so another request may allocate elsewhere
    # but can never reclaim the pending physical page.
    with runtime.request_context("other", timestamp=6):
        allocation = runtime.prepare_allocation("other", {1: 1})
        other_block = runtime.for_group(1).get_new_blocks(1)[0]
        runtime.finish_allocation("other")
        assert allocation.decisions[0].handle.superpage_id != deferred_typed_id.superpage_id
        runtime.for_group(1).free_blocks([other_block])

    runtime.free_blocks([block])
    assert block.ref_cnt == 0
    assert runtime.get_cached_block(request.block_hashes[1], [0]) == [block]
    assert runtime.reset_prefix_cache()
    assert runtime.get_cached_block(request.block_hashes[1], [0]) is None
    assert runtime._metadata_pools[0].blocks[1].ref_cnt == 1

    with pytest.raises(RuntimeError, match="request_context"):
        runtime.free_blocks([block])


def test_prepared_allocation_requires_complete_consumption_or_clean_cancel(runtime_modules):
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=3), hash_block_size=2)

    with runtime.request_context("cancel", timestamp=1):
        runtime.prepare_allocation("cancel", {0: 2})
        runtime.cancel_allocation("cancel")
    assert runtime.get_usage() == 0

    with runtime.request_context("partial", timestamp=2):
        runtime.prepare_allocation("partial", {0: 2})
        consumed = runtime.for_group(0).get_new_blocks(1)[0]
        with pytest.raises(RuntimeError, match="not consumed"):
            runtime.finish_allocation("partial")
        with pytest.raises(RuntimeError, match="cannot cancel"):
            runtime.cancel_allocation("partial")
        runtime.abort_allocation("partial")
    assert runtime._prepared is None
    assert consumed.ref_cnt == 0
    assert runtime.get_usage() == 0
    runtime.policy.check_invariants()


def test_abort_reverses_consumed_pages_and_cache_hit_touches(runtime_modules):
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=3), hash_block_size=2)
    cached_request, cached = _allocate_cache(
        runtime,
        "cached",
        {0: 1},
        allocation_timestamp=1,
        release_timestamp=2,
    )
    hit = runtime.get_cached_block(cached_request.block_hashes[1], [0])[0]
    assert hit.ref_cnt == 0

    with runtime.request_context("aborted", timestamp=3):
        runtime.prepare_allocation(
            "aborted",
            {1: 1},
            protected_blocks={0: [hit]},
        )
        runtime.for_group(0).touch([hit])
        consumed = runtime.for_group(1).get_new_blocks(1)[0]
        assert hit.ref_cnt == 1
        assert consumed.ref_cnt == 1
        runtime.abort_allocation("aborted")

    assert hit.ref_cnt == 0
    assert consumed.ref_cnt == 0
    assert runtime._prepared is None
    assert runtime.get_cached_block(cached_request.block_hashes[1], [0]) == [cached[0][0]]
    runtime.policy.check_invariants()


def test_incomplete_eviction_cleanup_poison_fails_closed(
    runtime_modules,
    monkeypatch,
):
    runtime_cls = runtime_modules["jenga_prefix_runtime"].JengaPrefixRuntimePool
    runtime = runtime_cls(_plan(runtime_modules, num_superpages=2), hash_block_size=2)
    _allocate_cache(
        runtime,
        "old-owner",
        {0: 2},
        allocation_timestamp=1,
        release_timestamp=2,
    )
    metadata_pool = runtime._metadata_pools[0]

    def injected_cleanup_failure(block):
        del block
        raise RuntimeError("injected hash cleanup failure")

    monkeypatch.setattr(
        metadata_pool,
        "_maybe_evict_cached_block",
        injected_cleanup_failure,
    )
    with (
        runtime.request_context("replacement", timestamp=3),
        pytest.raises(RuntimeError, match="failed to finish Jenga eviction cleanup"),
    ):
        runtime.prepare_allocation("replacement", {1: 1})

    assert runtime._prepared is None
    assert runtime._poisoned_error is not None
    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        runtime.can_allocate("later", {0: 1})


def test_unsupported_boundaries_fail_without_mutating_metadata(runtime_modules):
    runtime_module = runtime_modules["jenga_prefix_runtime"]
    runtime = runtime_module.JengaPrefixRuntimePool(_plan(runtime_modules), hash_block_size=2)

    assert runtime.cache_partial_block(object()) is None
    with pytest.raises(NotImplementedError, match="partial-prefix copy-on-write"):
        runtime.move_block_hashes(object(), object())
    with pytest.raises(NotImplementedError, match="connector-directed"):
        runtime.evict_blocks({1})
    with pytest.raises(NotImplementedError, match="events"):
        runtime_module.JengaPrefixRuntimePool(
            _plan(runtime_modules),
            hash_block_size=2,
            enable_kv_cache_events=True,
        )

    foreign = _FakeKVCacheBlock(2, ref_cnt=1)
    with runtime.request_context("foreign"), pytest.raises(ValueError, match="not created"):
        runtime.free_blocks([foreign])
